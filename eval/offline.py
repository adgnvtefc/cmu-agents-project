"""Offline (teacher-forced) next-action evaluation on recorded demos.

For every step k of every demo: give the model the gold history (steps < k) and the current
screenshot, compare its predicted action with the gold action.

  uv run python -m eval.offline --model models/UI-TARS-1.5-7B-mlx-bf16 [--adapter DIR] \
      --demos data/raw/alice --out results/offline/alice_C3.jsonl [--n-images 3]
"""

import argparse
import json
import time
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

from agent.uitars import build_messages, parse_response, to_uitars


def score(pred: dict | None, gold: dict, target: dict | None) -> dict:
    g = gold["type"]
    p = pred["type"] if pred else None
    same_type = p == g or (g == "double_click" and p == "click")
    out = {"type_ok": same_type, "ground_ok": None, "content_ok": None}
    if g in ("click", "double_click", "right_click") and target and pred and "x" in pred:
        x, y, w, h = target["bbox"]
        out["ground_ok"] = x <= pred["x"] <= x + w and y <= pred["y"] <= y + h
    if g == "type":
        out["content_ok"] = bool(pred) and pred.get("content", "").rstrip("\n") == gold["content"].rstrip("\n")
    if g == "hotkey":
        out["content_ok"] = bool(pred) and pred.get("keys") == [k if k != "meta" else "ctrl" for k in gold["keys"]]
    checks = [v for v in (out["ground_ok"], out["content_ok"]) if v is not None]
    out["step_ok"] = same_type and all(checks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--skill", default="")
    args = ap.parse_args()

    model, processor = load(args.model, adapter_path=args.adapter)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    demos = sorted(p.parent for root in args.demos for p in Path(root).rglob("meta.json"))
    n = ok = 0
    with open(out, "w") as fh:
        for d in demos:
            meta = json.loads((d / "meta.json").read_text())
            if args.skill and meta["skill"] != args.skill:
                continue
            steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
            shots = [str(d / s["obs"]) for s in steps]
            hist = [f"Thought: {s.get('thought') or ''}\nAction: {to_uitars(s['action'])}" for s in steps]
            for k, s in enumerate(steps):
                msgs, imgs = build_messages(meta["instruction"], shots[: k + 1], hist[:k], args.n_images)
                prompt = apply_chat_template(processor, model.config, msgs, num_images=len(imgs))
                t0 = time.perf_counter()
                res = generate(model, processor, prompt, image=imgs, max_tokens=200, temperature=0.0, verbose=False)
                parsed = parse_response(res.text)
                sc = score(parsed["action"], s["action"], s.get("target"))
                n += 1
                ok += sc["step_ok"]
                fh.write(json.dumps({
                    "demo": str(d), "persona": meta["persona"], "task_id": meta["task_id"], "skill": meta["skill"],
                    "step": k, "gold": s["action"], "pred": parsed["action"], "thought": parsed["thought"],
                    "raw": res.text,
                    **sc, "latency": round(time.perf_counter() - t0, 2),
                }) + "\n")
                mark = "✓" if sc["step_ok"] else "✗"
                print(f"{mark} {meta['persona']:5s} {meta['task_id']:17s} {k:2d} gold={to_uitars(s['action'])[:60]:60s} pred={parsed['action']}")
    print(f"\nstep accuracy: {ok}/{n} = {ok / max(n, 1):.2%}  → {out}")


if __name__ == "__main__":
    main()
