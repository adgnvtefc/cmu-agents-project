"""Online evaluation: UI-TARS (MLX, optional LoRA adapter) operates the mock workspace live.

One episode = reset app state for (persona, task) -> loop {screenshot -> model -> execute} until
`finished` or the step cap -> run the checker on final state.

  uv run python -m agent.run --persona alice --tasks calendar_5 send_email_5 \
      [--adapter adapters/C3/alice/calendar] [--context-file skills/alice/calendar.md] \
      --out results/online/C0
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from playwright.async_api import async_playwright

from agent.uitars import build_messages, parse_response
from apps.tasks import get_task
from recorder.record import START_URL, VIEWPORT

KEY_NAMES = {"ctrl": "ControlOrMeta", "cmd": "ControlOrMeta", "meta": "ControlOrMeta", "enter": "Enter",
             "esc": "Escape", "escape": "Escape", "tab": "Tab", "backspace": "Backspace", "delete": "Delete",
             "space": "Space", "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
             "shift": "Shift", "alt": "Alt", "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown"}


async def execute(page, action: dict) -> str:
    """Apply a parsed UI-TARS action to the page. Returns '' or an error note."""
    t = action["type"]
    try:
        before = await page.screenshot(type="jpeg", quality=30) if t != "wait" else None
    except Exception:
        before = None
    if t in ("click", "double_click", "right_click"):
        x, y = action["x"], action["y"]
        if not (0 <= x < VIEWPORT["width"] and 0 <= y < VIEWPORT["height"]):
            return f"click outside viewport ({x},{y})"
        if t == "click":
            await page.mouse.click(x, y)
        elif t == "double_click":
            await page.mouse.dblclick(x, y)
        else:
            await page.mouse.click(x, y, button="right")
    elif t == "type":
        content = action.get("content", "")
        submit = content.endswith("\n")
        await page.keyboard.type(content.rstrip("\n") if submit else content, delay=5)
        if submit:
            await page.keyboard.press("Enter")
    elif t == "hotkey":
        keys = [KEY_NAMES.get(k.lower(), k) for k in action.get("keys", [])]
        if keys:
            await page.keyboard.press("+".join(keys))
    elif t == "scroll":
        await page.mouse.move(action["x"], action["y"])
        dy = {"down": 400, "up": -400}.get(action.get("direction"), 0)
        dx = {"right": 400, "left": -400}.get(action.get("direction"), 0)
        await page.mouse.wheel(dx, dy)
    elif t == "wait":
        await asyncio.sleep(1.0)
    try:
        await page.wait_for_load_state("load", timeout=3000)
    except Exception:
        pass
    # Wait until the screen actually reflects the action (up to 1.5 s), so the next screenshot is
    # not a stale pre-action frame (a known cause of "repeat the last action" loops).
    if before is not None:
        for _ in range(10):
            await asyncio.sleep(0.15)
            try:
                if await page.screenshot(type="jpeg", quality=30) != before:
                    break
            except Exception:
                break
    await asyncio.sleep(0.15)
    return ""


def _same(a: dict | None, b: dict | None) -> bool:
    if not a or not b or a["type"] != b["type"]:
        return False
    if "x" in a:
        return abs(a["x"] - b["x"]) <= 8 and abs(a["y"] - b["y"]) <= 8
    return a == b


def _looping(log: list, n: int = 4) -> bool:
    """Stop when the same action (ignoring waits) repeats n times in a row: the episode is stuck."""
    acts = [s["action"] for s in log if not (s["action"] or {}).get("type") == "wait"]
    return len(acts) >= n and all(_same(acts[-1], a) for a in acts[-n:])


async def run_episode(model, processor, browser, base: str, persona: str, task_id: str, out: Path,
                      n_images: int = 3, max_steps: int = 25, context: str = "",
                      instruction: str | None = None) -> dict:
    """instruction: overrides the task's wording (the app state and checker still come from task_id)."""
    task = dict(get_task(task_id))
    if instruction:
        task["instruction"] = instruction
    async with httpx.AsyncClient(base_url=base) as api:
        (await api.post("/__reset", json={"persona": persona, "task_id": task_id})).raise_for_status()
        ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        await ctx.route("**/__*", lambda route: route.abort())
        page = await ctx.new_page()
        await page.goto(base + START_URL[task["skill"]])
        ep_dir = out / persona / task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        shots, history, log = [], [], []
        finished, t_start = False, time.perf_counter()
        for k in range(max_steps):
            shot = ep_dir / f"{k:03d}.png"
            await page.screenshot(path=str(shot))
            shots.append(str(shot))
            msgs, imgs = build_messages(task["instruction"], shots, history, n_images, context=context)
            prompt = apply_chat_template(processor, model.config, msgs, num_images=len(imgs))
            t0 = time.perf_counter()
            res = generate(model, processor, prompt, image=imgs, max_tokens=256, temperature=0.0, verbose=False)
            parsed = parse_response(res.text)
            act = parsed["action"]
            err = ""
            if act is None:
                err = "unparseable"
            elif act["type"] == "finished":
                finished = True
            else:
                err = await execute(page, act)
            history.append(res.text.strip())
            log.append({"step": k, "raw": res.text, "action": act, "error": err,
                        "latency": round(time.perf_counter() - t0, 2), "prompt_tokens": res.prompt_tokens})
            with open(ep_dir / "live.jsonl", "a") as fh:  # per-step progress for live viewers (Studio)
                fh.write(json.dumps(log[-1]) + "\n")
            if finished:
                break
            if _looping(log):
                break
        try:
            await page.screenshot(path=str(ep_dir / "final.png"))  # the end state (for viewers)
        except Exception:
            pass
        await ctx.close()
        result = (await api.get("/__check")).json()
    looped = _looping(log)
    ep = {"persona": persona, "looped": looped, "task_id": task_id, "skill": task["skill"], "instruction": task["instruction"],
          "steps": len(log), "finished": finished, "check": result, "seconds": round(time.perf_counter() - t_start, 1),
          "format_errors": sum(1 for s in log if s["error"] == "unparseable"), "log": log}
    (ep_dir / "episode.json").write_text(json.dumps(ep, indent=1))
    return ep


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/UI-TARS-1.5-7B-mlx-bf16")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--persona", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8917")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--context-file", default=None, help="text inserted into the prompt (ICL demos / SKILL.md)")
    ap.add_argument("--instruction", default=None, help="override the task's instruction wording")
    args = ap.parse_args()
    model, processor = load(args.model, adapter_path=args.adapter)
    context = Path(args.context_file).read_text() if args.context_file else ""
    out = Path(args.out)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for persona in args.persona:
            for tid in args.tasks:
                ep = await run_episode(model, processor, browser, args.base, persona, tid, out,
                                       args.n_images, args.max_steps, context, args.instruction)
                c = ep["check"]
                print(f"{persona:5s} {tid:17s} success={c['success']} joint={c['joint']} prefs={c['prefs']} "
                      f"steps={ep['steps']} finished={ep['finished']} fmt_err={ep['format_errors']} {ep['seconds']}s")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
