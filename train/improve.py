"""Five approaches to improve C3 (per-user, per-skill LoRA). Each uses only what a deployed system
has: the user's demos, other users' demos, the base model, the environment and persona-independent
task checks. Never the persona cards or the preference checker.

  A1 context distillation   teacher = C3 + user's demos in prompt, run on synthetic tasks; keep
                            task-successful episodes; continue C3 on teacher data + demos
  A2 contrastive DPO        DPO-Positive from C3; at the user's distinctive steps, the other users'
                            action at the same UI state is the rejected answer
  A3 two-stage warm start   start from C5 (same skill, other users), fine-tune on the user's demos
  A4 habit thoughts         distinctive steps get thoughts that state the habit; upsampled x3
  A5 decision-weighted loss action span weighted x4 in the loss (train/weighted_sft.py)

  uv run python -m train.improve --approach A4 --persona alice --skill calendar [--demos-root data/oracle]
Adapters: adapters/<root>/<A#>/<persona>/<skill>/seed0
"""

import argparse
import json
import sys
import time
from pathlib import Path

from agent.thoughts import _pretty, step_response, template_thought
from agent.uitars import build_messages, to_uitars
from apps.personas import PERSONAS
from eval.summarize import action_key
from train.train_adapters import MODEL, ROOT, reported_losses, run_guarded

N_IMAGES = 3


# ------------------------------------------------------------------ demo access
def demos(root: Path, persona: str, skill: str, split: str = "train") -> list[Path]:
    out = []
    for m in sorted(root.glob(f"{persona}/*/*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("accepted") and meta["skill"] == skill and meta["split"] == split:
            out.append(m.parent)
    return out


def load(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    shots = [str((d / s["obs"]).resolve()) for s in steps]
    return meta, steps, shots


def state_key(steps, k):
    """What has just happened: the previous action's type + element (content ignored)."""
    if k == 0:
        return ("START",)
    t = action_key(steps[k - 1])
    return (t[0], t[1])


def distinctive(root: Path, persona: str, skill: str) -> set:
    """Action keys this user uses that no other user uses for this skill (train demos only)."""
    mine, others = set(), set()
    for p in PERSONAS:
        for d in demos(root, p, skill):
            _, steps, _ = load(d)
            keys = {action_key(s) for s in steps}
            (mine if p == persona else others).update(keys)
    return mine - others


# ------------------------------------------------------------------ sample building
def rows_for(meta, steps, shots, responses, n_images=N_IMAGES, weight_steps=None, upsample=1):
    rows = []
    for k in range(len(steps)):
        msgs, imgs = build_messages(meta["instruction"], shots[: k + 1], responses[:k], n_images)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": responses[k]}]})
        reps = upsample if weight_steps and k in weight_steps else 1
        rows += [{"images": imgs, "messages": msgs}] * reps
    return rows


def habit_thought(step):
    a = step["action"]
    if a["type"] in ("click", "double_click"):
        return f"This user always chooses \"{_pretty(step.get('target'))}\" here, so I'll click it."
    if a["type"] == "type":
        c = a["content"].rstrip("\n")
        if len(c) <= 40 and "\n" not in c:
            return f"This user's usual choice here is \"{c}\", so I'll type it into the {_pretty(step.get('target'))} field."
        return "I'll write it the way this user always does, including their usual phrasing and sign-off."
    return template_thought(step)


# ------------------------------------------------------------------ training runners
def run_train(rows, out: Path, *, init=None, epochs=4, lr=5e-5, weighted=False, extra=None):
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    with open(data / "train.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    mod = "train.weighted_sft" if weighted else "mlx_vlm.train"
    cmd = [sys.executable, "-m", mod, "--model", str(MODEL), "--data", str(data),
           "--train-type", "lora", "--lora-rank", "16", "--lora-alpha", "16", "--lora-dropout", "0.05",
           "--learning-rate", str(lr), "--warmup-steps", "20", "--batch-size", "1",
           "--iters", str(max(1, int(epochs * len(rows)))), "--steps-per-report", "10",
           "--max-seq-length", "8192", "--grad-checkpoint", "--seed", "0", "--output-path", str(out)]
    if extra:
        cmd += extra
    if "--algorithm" not in (extra or []):
        cmd += ["--train-on-completions"]
    if init:
        cmd += ["--resume-adapter-file", str(Path(init) / "adapters.safetensors")]
    t0 = time.time()
    rc, diverged = run_guarded(cmd, out / "train.log")
    info = {"rows": len(rows), "epochs": epochs, "lr": lr, "init": str(init) if init else None,
            "weighted": weighted, "extra": extra, "returncode": rc, "diverged": diverged,
            "losses": reported_losses(out / "train.log"), "minutes": round((time.time() - t0) / 60, 1)}
    (out / "run.json").write_text(json.dumps(info, indent=1))
    if diverged:
        (out / "adapters.safetensors").unlink(missing_ok=True)
    print(f"  {out.relative_to(ROOT)}: rows={len(rows)} rc={rc} diverged={diverged} "
          f"loss {info['losses'][:1]}→{info['losses'][-1:]} in {info['minutes']} min", flush=True)
    return info


# ------------------------------------------------------------------ approaches
def a4(root, persona, skill, out):
    dk = distinctive(root, persona, skill)
    rows = []
    for d in demos(root, persona, skill):
        meta, steps, shots = load(d)
        hot = {k for k, s in enumerate(steps) if action_key(s) in dk}
        resp = [f"Thought: {habit_thought(s) if k in hot else template_thought(s)}\nAction: {to_uitars(s['action'])}"
                for k, s in enumerate(steps)]
        rows += rows_for(meta, steps, shots, resp, weight_steps=hot, upsample=3)
    print(f"A4 {persona}/{skill}: {len(dk)} distinctive action keys, {len(rows)} rows")
    return run_train(rows, out, epochs=3)  # ~same number of updates as C3 (more rows, fewer epochs)


def plain_rows(root, persona, skill):
    rows = []
    for d in demos(root, persona, skill):
        meta, steps, shots = load(d)
        rows += rows_for(meta, steps, shots, [step_response(s, to_uitars) for s in steps])
    return rows


def a5(root, persona, skill, out):
    return run_train(plain_rows(root, persona, skill), out, epochs=4, weighted=True)


def a3(root, persona, skill, out):
    init = ROOT / "adapters" / root.name / "C5" / persona / skill / "seed0"
    if not (init / "adapters.safetensors").exists():
        raise SystemExit(f"A3 needs the C5 adapter first: {init}")
    return run_train(plain_rows(root, persona, skill), out, init=init, epochs=2)


def a2(root, persona, skill, out):
    init = ROOT / "adapters" / root.name / "C3" / persona / skill / "seed0"
    # Pair at ANY state where another user did something this user never does (e.g. Alice clicks
    # "CC", Bob never does): rejected = the other user's action, chosen = this user's action.
    my_keys = {action_key(s) for d in demos(root, persona, skill) for s in load(d)[1]}
    pairs = []
    for d in demos(root, persona, skill):
        meta, steps, shots = load(d)
        resp = [step_response(s, to_uitars) for s in steps]
        for o in PERSONAS:
            if o == persona:
                continue
            od = [x for x in demos(root, o, skill) if json.loads((x / "meta.json").read_text())["task_id"] == meta["task_id"]]
            if not od:
                continue
            _, osteps, _ = load(od[0])
            for k, s in enumerate(steps):
                for j in range(len(osteps)):
                    ok = action_key(osteps[j])
                    if state_key(osteps, j) == state_key(steps, k) and ok != action_key(s) and ok not in my_keys:
                        msgs, imgs = build_messages(meta["instruction"], shots[: k + 1], resp[:k], N_IMAGES)
                        rej = step_response(osteps[j], to_uitars)
                        pairs.append({"images": imgs,
                                      "chosen": msgs + [{"role": "assistant", "content": [{"type": "text", "text": resp[k]}]}],
                                      "rejected": msgs + [{"role": "assistant", "content": [{"type": "text", "text": rej}]}]})
                        break
    print(f"A2 {persona}/{skill}: {len(pairs)} preference pairs")
    return run_train(pairs, out, init=init, epochs=3, lr=1e-5,
                     extra=["--algorithm", "dpo", "--loss-type", "dpop", "--beta", "0.1", "--delta", "50"])


def a1(root, persona, skill, out):
    init = ROOT / "adapters" / root.name / "C3" / persona / skill / "seed0"
    tdir = ROOT / "results" / "teacher" / root.name / persona / skill
    eps = sorted(tdir.glob(f"{persona}/*/episode.json"))
    if not eps:
        raise SystemExit(f"A1 needs teacher rollouts first in {tdir} (scripts/improve_pair.sh does this)")
    rows, kept = [], 0
    for f in eps:
        ep = json.loads(f.read_text())
        if not ep["check"]["success"]:          # persona-independent: right date/recipient/topic
            continue
        log = [s for s in ep["log"] if s["action"]]
        if not log or len(log) != len(ep["log"]):
            continue
        kept += 1
        shots = [str(f.parent / f"{s['step']:03d}.png") for s in ep["log"]]
        resp = [s["raw"].strip() for s in ep["log"]]
        meta = {"instruction": ep["instruction"]}
        rows += rows_for(meta, ep["log"], shots, resp)
    rows += plain_rows(root, persona, skill)
    print(f"A1 {persona}/{skill}: kept {kept}/{len(eps)} teacher episodes, {len(rows)} rows (incl. demos)")
    return run_train(rows, out, init=init, epochs=1, lr=3e-5)


APPROACHES = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True, choices=list(APPROACHES))
    ap.add_argument("--persona", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--demos-root", default=str(ROOT / "data" / "oracle"))
    args = ap.parse_args()
    root = Path(args.demos_root)
    out = ROOT / "adapters" / root.name / args.approach / args.persona / args.skill / "seed0"
    if (out / "adapters.safetensors").exists():
        print(f"skip {out} (exists)")
        return
    out.mkdir(parents=True, exist_ok=True)
    APPROACHES[args.approach](root, args.persona, args.skill, out)


if __name__ == "__main__":
    main()
