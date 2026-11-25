"""Summarize offline results per condition, split by preference-critical steps.

A gold step is PREFERENCE-CRITICAL if no other persona's demo of the same task instance contains
the same action (same type, same element, same typed text): e.g. Alice typing "14:00" into the
start-time field, or clicking the "30 min" chip. Steps every persona shares (clicking "New
event") are not critical.

  uv run python -m eval.summarize results/offline/oracle --demos-root data/oracle
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from eval.scoring import resolve


def action_key(step: dict):
    a = step["action"]
    tgt = None
    if "x" in a:
        tgt = resolve(a["x"], a["y"], step.get("elements")) or (step.get("target") or {}).get("id")
    elif step.get("target"):
        tgt = step["target"].get("id")
    return (a["type"], tgt, a.get("content"), tuple(a.get("keys") or ()))


def critical_map(demos_root: Path) -> dict:
    """(persona, task_id, step) -> is_critical"""
    by_task = defaultdict(dict)  # task_id -> persona -> [keys]
    for m in demos_root.glob("*/*/*/meta.json"):
        meta = json.loads(m.read_text())
        if not meta.get("accepted"):
            continue
        steps = [json.loads(line) for line in (m.parent / "steps.jsonl").read_text().splitlines()]
        by_task[meta["task_id"]][meta["persona"]] = [action_key(s) for s in steps]
    out = {}
    for tid, per in by_task.items():
        for p, keys in per.items():
            others = {k for o, ks in per.items() if o != p for k in ks}
            for i, k in enumerate(keys):
                out[(p, tid, i)] = k not in others
    return out


def ci(k: int, n: int) -> str:
    if n == 0:
        return "—"
    p = k / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    return f"{p:6.1%} ±{half:5.1%} ({k}/{n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("--demos-root", required=True)
    args = ap.parse_args()
    crit = critical_map(Path(args.demos_root))
    rows = []
    for f in sorted(Path(args.results_root).glob("*/*.jsonl")):
        cond = f.parent.name
        recs = [json.loads(line) for line in f.read_text().splitlines()]
        tot = sum(r["step_ok"] for r in recs)
        c = [r for r in recs if crit.get((r["persona"], r["task_id"], r["step"]))]
        nc = [r for r in recs if not crit.get((r["persona"], r["task_id"], r["step"]))]
        rows.append((cond, f.stem, ci(tot, len(recs)), ci(sum(r["step_ok"] for r in c), len(c)),
                     ci(sum(r["step_ok"] for r in nc), len(nc))))
    print(f"{'cond':5s} {'slice':16s} {'all steps':26s} {'preference-critical':26s} {'other steps':26s}")
    for r in rows:
        print(f"{r[0]:5s} {r[1]:16s} {r[2]:26s} {r[3]:26s} {r[4]:26s}")
    print("\n± = 95% normal-approx CI over steps (steps within an episode are correlated: treat as optimistic)")


if __name__ == "__main__":
    main()
