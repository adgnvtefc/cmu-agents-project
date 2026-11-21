"""Record human demonstrations in the mock workspace.

Usage (server must be running:  uv run uvicorn apps.server:app --port 8917):
  uv run python -m recorder.record --persona alice --split train      # all train tasks, skips done
  uv run python -m recorder.record --persona alice --task send_email_0  # one task (re-record)
  uv run python -m recorder.record --persona alice --split test

For each task: the instruction prints here, press Enter to open the browser, do the task in the
browser, then come back here and press Enter to finish. The checker runs and you choose whether
to keep the demo. Ctrl+C aborts the current demo (nothing saved).

Output: data/raw/<persona>/<task_id>/<timestamp>/
  frames/NNN.png    pre-action screenshot for step NNN (1288x812)
  steps.jsonl       one action per line
  meta.json         task, persona, check result, timings, accepted flag
  state_before.json, state_after.json, trace.zip (Playwright trace for debugging)
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

from apps.personas import PERSONAS, persona_card_markdown
from apps.tasks import get_task, task_ids
from recorder.actions import ActionBuilder, Frame

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_JS = (Path(__file__).parent / "capture.js").read_text()
VIEWPORT = {"width": 1288, "height": 812}  # multiple of 28 -> UI-TARS smart_resize is identity
START_URL = {"send_email": "/mail", "file_email": "/mail", "calendar": "/calendar", "organize_files": "/drive"}
FRAME_INTERVAL_S = 0.1


def accepted_demos(persona: str, task_id: str) -> list[Path]:
    d = ROOT / "data" / "raw" / persona / task_id
    out = []
    for meta in sorted(d.glob("*/meta.json")):
        if json.loads(meta.read_text()).get("accepted"):
            out.append(meta.parent)
    return out


async def ainput(prompt: str = "") -> str:
    print(prompt, end="", flush=True)
    return (await asyncio.to_thread(sys.stdin.readline)).strip()


async def record_one(browser, base: str, persona: str, task_id: str, driver=None,
                     out_root: Path | None = None) -> bool:
    """driver: optional async fn(page) that performs the demo (scripted mode, no prompts)."""
    interactive = driver is None
    task = get_task(task_id)
    async with httpx.AsyncClient(base_url=base) as api:
        (await api.post("/__reset", json={"persona": persona, "task_id": task_id})).raise_for_status()

        print("\n" + "=" * 72)
        print(f"  {persona.upper()} · {task_id} ({task['split']})")
        print(f"  TASK: {task['instruction']}")
        print("=" * 72)
        if interactive:
            await ainput("Press Enter to open the browser and start recording... ")

        context = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        await context.route("**/__*", lambda route: route.abort())  # control API is off-limits
        await context.tracing.start(screenshots=True, snapshots=True)

        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(source, payload):
            queue.put_nowait((time.monotonic(), payload))

        await context.expose_binding("__rec", on_event)
        await context.add_init_script(CAPTURE_JS)
        page = await context.new_page()
        await page.goto(base + START_URL[task["skill"]])
        await page.wait_for_load_state("load")

        frames: list[Frame] = []
        stop = asyncio.Event()

        async def capturer():
            while not stop.is_set():
                t0 = time.monotonic()
                try:
                    png = await page.screenshot(type="png", timeout=3000)
                    frames.append(Frame(t0, time.monotonic(), png))
                    if len(frames) > 60:
                        del frames[:-60]
                except Exception:
                    pass
                await asyncio.sleep(FRAME_INTERVAL_S)

        builder = ActionBuilder()
        t_start = time.monotonic()
        cap = asyncio.create_task(capturer())
        await asyncio.sleep(0.5)
        if interactive:
            print("● Recording. Do the task in the browser, then press Enter HERE to finish (Ctrl+C aborts).")
            done = asyncio.create_task(asyncio.to_thread(sys.stdin.readline))
        else:
            done = asyncio.create_task(driver(page))
        try:
            while not done.done():
                try:
                    t, ev = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                builder.feed(t, ev, frames)
            await asyncio.sleep(0.6)  # let the last action settle before the final frame
            while not queue.empty():
                t, ev = queue.get_nowait()
                builder.feed(t, ev, frames)
            t_end = time.monotonic()
            builder.finish(t_end, frames, page.url)
        finally:
            stop.set()
            await cap

        before = (await api.get("/__before")).json()
        after = (await api.get("/__state")).json()
        result = (await api.get("/__check")).json()

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = (out_root or ROOT / "data" / "raw") / persona / task_id / stamp
        (out / "frames").mkdir(parents=True)
        await context.tracing.stop(path=out / "trace.zip")
        await context.close()

        n_stale = 0
        with open(out / "steps.jsonl", "w") as fh:
            for i, a in enumerate(builder.actions):
                fname = f"frames/{i:03d}.png"
                if a["frame"] is not None:
                    (out / fname).write_bytes(a["frame"].png)
                n_stale += a["stale"]
                fh.write(json.dumps({
                    "step": i, "obs": fname if a["frame"] else None, "action": a["action"],
                    "target": a["target"], "url": a["url"], "t": round(a["t"] - t_start, 3),
                    "stale_obs": a["stale"],
                }) + "\n")
        (out / "state_before.json").write_text(json.dumps(before, indent=1))
        (out / "state_after.json").write_text(json.dumps(after, indent=1))

        # ---- report + accept
        print(f"\nRecorded {len(builder.actions)} steps in {t_end - t_start:.1f}s")
        for a in builder.actions:
            act = dict(a["action"])
            act.pop("value_before", None), act.pop("value_after", None)
            tgt = (a["target"] or {}).get("testid") or (a["target"] or {}).get("text", "")
            print(f"  {'!' if a['stale'] else ' '} {json.dumps(act)}  [{tgt}]")
        if n_stale:
            print(f"  ! {n_stale} step(s) happened very soon after the previous one; their screenshot may be stale.")
        ok = result["success"] and result["joint"] and not result["collateral"]
        print(f"\nCHECK  success={result['success']}  prefs={result['prefs']}  "
              f"collateral={result['collateral']}  {result['notes'] or ''}")
        default = "Y" if ok else "n"
        ans = ((await ainput(f"Keep this demo? [{'Y/n' if ok else 'y/N'}] ")) or default) if interactive else default
        accepted = ans.lower().startswith("y")
        meta = {
            "persona": persona, "task_id": task_id, "skill": task["skill"], "split": task["split"],
            "instruction": task["instruction"], "start_url": START_URL[task["skill"]],
            "viewport": VIEWPORT, "n_steps": len(builder.actions), "duration_s": round(t_end - t_start, 2),
            "n_stale": n_stale, "check": result, "accepted": accepted, "recorded_at": stamp,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(("✓ kept → " if accepted else "✗ discarded (files left for debugging) → ") + str(out))
        return accepted


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True, choices=list(PERSONAS))
    ap.add_argument("--task", help="single task id, e.g. send_email_0")
    ap.add_argument("--split", choices=["train", "test"], help="record every task in the split")
    ap.add_argument("--base", default="http://127.0.0.1:8917")
    ap.add_argument("--redo", action="store_true", help="record even if an accepted demo exists")
    args = ap.parse_args()

    todo = [args.task] if args.task else task_ids(args.split)
    if not args.redo and not args.task:
        todo = [t for t in todo if not accepted_demos(args.persona, t)]
    print(persona_card_markdown(args.persona))
    print(f"{len(todo)} task(s) to record: {', '.join(todo)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        for tid in todo:
            while True:
                try:
                    if await record_one(browser, args.base, args.persona, tid):
                        break
                except KeyboardInterrupt:
                    raise
                ans = await ainput("Re-record this task? [Y/n/q] ")
                if ans.lower() == "q":
                    return
                if ans.lower() == "n":
                    break
        await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")
