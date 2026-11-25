"""Benchmark UI-TARS-1.5-7B agent-step latency on Apple silicon (MLX), and sanity-check zero-shot
grounding on recorded demos (does the predicted click land in the gold element's bbox?).

  uv run python -m scripts.bench_mlx --model models/UI-TARS-1.5-7B-hf --demos <dir with demo folders>
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

from agent.uitars import build_messages, parse_response, to_uitars


def load_demo(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    return meta, steps


def step_inputs(d: Path, meta, steps, k: int, n_images: int):
    shots = [str(d / s["obs"]) for s in steps[: k + 1]]
    hist = [f"Thought: (step {i})\nAction: {to_uitars(s['action'])}" for i, s in enumerate(steps[:k])]
    return build_messages(meta["instruction"], shots, hist, n_images)


def run(model, processor, messages, images, max_tokens=160):
    prompt = apply_chat_template(processor, model.config, messages, num_images=len(images))
    t0 = time.perf_counter()
    res = generate(model, processor, prompt, image=images, max_tokens=max_tokens, temperature=0.0, verbose=False)
    wall = time.perf_counter() - t0
    return res, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--demos", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, processor = load(args.model)
    print(f"load: {time.perf_counter() - t0:.1f}s  peak mem {mx.get_peak_memory() / 1e9:.1f} GB")

    demos = [p for p in sorted(Path(args.demos).rglob("meta.json"))]
    demos = [p.parent for p in demos]

    # ---- latency vs history length (on the longest demo)
    d = max(demos, key=lambda p: len(load_demo(p)[1]))
    meta, steps = load_demo(d)
    k = min(len(steps) - 2, 8)
    print(f"\nlatency on {meta['task_id']} step {k}")
    run(model, processor, *step_inputs(d, meta, steps, 0, 1), max_tokens=8)  # warmup
    for n in (1, 3, 5):
        msgs, imgs = step_inputs(d, meta, steps, k, n)
        res, wall = run(model, processor, msgs, imgs)
        print(f"  history={n} imgs  prompt_tokens={res.prompt_tokens:5d}  prefill={res.prompt_tps:6.0f} tok/s  "
              f"gen={res.generation_tokens:3d} tok @ {res.generation_tps:5.1f} tok/s  wall={wall:5.2f}s  "
              f"peak={res.peak_memory:.1f} GB")
        print("    →", res.text.strip().replace("\n", " | ")[:220])

    # ---- zero-shot grounding sanity: every click step of every demo, history=3
    print("\nzero-shot next-action check (history=3):")
    hits = total = 0
    walls = []
    for d in demos:
        meta, steps = load_demo(d)
        for k, s in enumerate(steps):
            if s["action"]["type"] != "click" or not s.get("target"):
                continue
            msgs, imgs = step_inputs(d, meta, steps, k, 3)
            res, wall = run(model, processor, msgs, imgs)
            walls.append(wall)
            pred = parse_response(res.text)["action"]
            x, y, w, h = s["target"]["bbox"]
            ok = bool(pred and pred["type"] == "click" and x <= pred["x"] <= x + w and y <= pred["y"] <= y + h)
            hits += ok
            total += 1
            got = f"({pred['x']:.0f},{pred['y']:.0f})" if pred and "x" in pred else str(pred)
            print(f"  {'✓' if ok else '✗'} {meta['task_id']:17s} step {k:2d} gold={s['target'].get('testid') or s['target']['text']!r:24s} "
                  f"pred={got}  {wall:.1f}s")
    print(f"\nclick-in-bbox: {hits}/{total}   mean step wall: {sum(walls) / len(walls):.2f}s")


if __name__ == "__main__":
    main()
