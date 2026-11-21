"""Train the LoRA adapters for each condition in PLAN.md, locally with mlx-vlm.

  C3  per-(user, skill):        that user's 5 train demos of the skill
  C5  same skill, other users:  the other users' train demos of the skill (personalization control)
  C6  pooled:                   all users' train demos of all skills, no user signal
  N1  per-user, all skills:     that user's train demos of every skill

  uv run python -m train.train_adapters --condition C3 --demos-root data/oracle \
      --personas alice --skills calendar [--epochs 5 --lr 1e-4]

Adapters go to adapters/<demos-root name>/<condition>/<persona or ALL>/<skill or ALL>/.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from apps.personas import PERSONAS
from apps.tasks import SKILLS
from train.build_sft import demo_samples

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "UI-TARS-1.5-7B-mlx-bf16"


def reported_losses(log_path: Path) -> list[float]:
    """Distinct loss values from mlx-vlm's progress bar (reported every --steps-per-report)."""
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    vals, last = [], None
    for v in re.findall(r"loss=([0-9.]+|nan)", text):
        if v != last:
            vals.append(float(v))
            last = v
    return vals


def train_demos(root: Path, personas, skills) -> list[Path]:
    out = []
    for m in sorted(root.glob("*/*/*/meta.json")):
        meta = json.loads(m.read_text())
        if (meta.get("accepted") and meta["split"] == "train" and meta["persona"] in personas
                and meta["skill"] in skills):
            out.append(m.parent)
    return out


def jobs(condition: str, personas, skills):
    """(persona_key, skill_key, train_personas, train_skills) per adapter."""
    if condition == "C3":
        return [(p, s, [p], [s]) for p in personas for s in skills]
    if condition == "C5":
        return [(p, s, [o for o in PERSONAS if o != p], [s]) for p in personas for s in skills]
    if condition == "C6":
        return [("ALL", "ALL", list(PERSONAS), SKILLS)]
    if condition == "N1":
        return [(p, "ALL", [p], SKILLS) for p in personas]
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["C3", "C5", "C6", "N1"])
    ap.add_argument("--demos-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS))
    ap.add_argument("--skills", nargs="+", default=SKILLS)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="off by default: mlx-vlm 0.8 grad clipping produced NaN loss in our tests")
    args = ap.parse_args()

    root = Path(args.demos_root)
    for pkey, skey, tp, ts in jobs(args.condition, args.personas, args.skills):
        out = ROOT / "adapters" / root.name / args.condition / pkey / skey / f"seed{args.seed}"
        if (out / "adapters.safetensors").exists():
            print(f"skip {out} (exists)")
            continue
        demos = train_demos(root, tp, ts)
        rows = [r for d in demos for r in demo_samples(d, args.n_images)]
        if not rows:
            print(f"no demos for {pkey}/{skey}")
            continue
        data = out / "data"
        data.mkdir(parents=True, exist_ok=True)
        with open(data / "train.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        cmd = [sys.executable, "-m", "mlx_vlm.train", "--model", str(MODEL), "--data", str(data),
               "--train-type", "lora", "--lora-rank", str(args.rank), "--lora-alpha", str(args.alpha),
               "--lora-dropout", "0.05", "--learning-rate", str(args.lr), "--warmup-steps", str(args.warmup),
               "--batch-size", "1", "--iters", str(args.epochs * len(rows)), "--steps-per-report", "10",
               "--train-on-completions", "--max-seq-length", "8192", "--grad-checkpoint",
               "--seed", str(args.seed), "--output-path", str(out)]
        if args.grad_clip:
            cmd += ["--grad-clip", str(args.grad_clip)]
        t0 = time.time()
        print(f"train {args.condition} {pkey}/{skey}: {len(demos)} demos, {len(rows)} samples, {args.epochs} epochs", flush=True)
        diverged = False
        with open(out / "train.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(10)
                losses = reported_losses(out / "train.log")
                if losses and (any(math.isnan(x) for x in losses) or
                               (len(losses) >= 3 and losses[-1] > 3 * losses[0] and losses[-1] > 1.5)):
                    diverged = True
                    proc.kill()
                    break
            rc = proc.wait()
        losses = reported_losses(out / "train.log")
        info = {"condition": args.condition, "persona": pkey, "skill": skey, "train_personas": tp,
                "train_skills": ts, "demos": [str(d) for d in demos], "samples": len(rows),
                "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "alpha": args.alpha, "warmup": args.warmup,
                "n_images": args.n_images, "seed": args.seed, "returncode": rc,
                "minutes": round((time.time() - t0) / 60, 1), "losses": losses, "diverged": diverged}
        (out / "run.json").write_text(json.dumps(info, indent=1))
        print(f"  done rc={rc} diverged={diverged} losses={losses[:2]}..{losses[-2:]} in {info['minutes']} min → {out}", flush=True)
        if diverged:
            (out / "adapters.safetensors").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
