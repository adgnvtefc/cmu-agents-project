"""Convert recorded demos into mlx-vlm SFT JSONL: one sample per step, UI-TARS multi-turn format,
loss on the final assistant turn only (--train-on-completions).

  uv run python -m train.build_sft --demos data/raw/alice --out data/sft/alice_send_email \
      [--skill send_email] [--n-images 3]
"""

import argparse
import json
from pathlib import Path

from agent.uitars import build_messages, to_uitars


def response(step: dict) -> str:
    thought = step.get("thought") or ""
    return f"Thought: {thought}\nAction: {to_uitars(step['action'])}"


def demo_samples(d: Path, n_images: int):
    meta = json.loads((d / "meta.json").read_text())
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    shots = [str((d / s["obs"]).resolve()) for s in steps]
    hist = [response(s) for s in steps]
    for k in range(len(steps)):
        msgs, imgs = build_messages(meta["instruction"], shots[: k + 1], hist[:k], n_images)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": hist[k]}]})
        yield {"images": imgs, "messages": msgs, "meta": {"task_id": meta["task_id"], "persona": meta["persona"], "step": k}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True, help="dirs searched recursively for accepted demos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skill", default="")
    ap.add_argument("--n-images", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in args.demos:
        for m in sorted(Path(root).rglob("meta.json")):
            meta = json.loads(m.read_text())
            if not meta.get("accepted") or (args.skill and meta["skill"] != args.skill):
                continue
            rows += list(demo_samples(m.parent, args.n_images))
    with open(out / "train.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} samples → {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
