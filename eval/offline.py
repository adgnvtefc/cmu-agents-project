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
from eval.scoring import score
from agent.thoughts import step_response
from agent.context import demos_context, load_demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--skill", default="")
    ap.add_argument("--split", default="", help="only evaluate demos of this split (train/test)")
    ap.add_argument("--context-from", default=None,
                    help="C1: demos root; put the persona's train demos of the same skill in the prompt")
    ap.add_argument("--skills-dir", default=None,
                    help="C2: dir with <persona>/<skill>.md skill files to put in the prompt")
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
            if args.split and meta["split"] != args.split:
                continue
            context = ""
            if args.context_from:
                context = demos_context(load_demos(args.context_from, meta["persona"], meta["skill"]))
            if args.skills_dir:
                md = Path(args.skills_dir) / meta["persona"] / f"{meta['skill']}.md"
                context = (context + "\n\n" if context else "") + "## Skill notes for this user\n" + md.read_text()
            steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
            shots = [str(d / s["obs"]) for s in steps]
            hist = [step_response(s, to_uitars) for s in steps]
            for k, s in enumerate(steps):
                msgs, imgs = build_messages(meta["instruction"], shots[: k + 1], hist[:k], args.n_images,
                                            context=context)
                prompt = apply_chat_template(processor, model.config, msgs, num_images=len(imgs))
                t0 = time.perf_counter()
                res = generate(model, processor, prompt, image=imgs, max_tokens=200, temperature=0.0, verbose=False)
                parsed = parse_response(res.text)
                sc = score(parsed["action"], s)
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
