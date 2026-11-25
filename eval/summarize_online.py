"""Summarize online episodes: results/online/<set>/<condition>/<persona>/<task>/episode.json

  uv run python -m eval.summarize_online results/online/oracle
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    by = defaultdict(list)
    for f in sorted(root.glob("*/*/*/episode.json")):
        ep = json.loads(f.read_text())
        by[f.parts[-4]].append(ep)
    print(f"{'cond':5s} {'n':>3s} {'success':>8s} {'all prefs':>9s} {'adherence':>9s} {'collateral':>10s} {'steps':>6s} {'looped':>6s}")
    for cond, eps in sorted(by.items()):
        n = len(eps)
        succ = sum(e["check"]["success"] for e in eps)
        joint = sum(e["check"]["joint"] for e in eps)
        adh = sum(e["check"]["adherence"] for e in eps) / n
        coll = sum(bool(e["check"]["collateral"]) for e in eps)
        steps = sum(e["steps"] for e in eps) / n
        loop = sum(bool(e.get("looped")) for e in eps)
        print(f"{cond:5s} {n:3d} {succ:>4d}/{n:<3d} {joint:>5d}/{n:<3d} {adh:>9.0%} {coll:>6d}/{n:<3d} {steps:>6.1f} {loop:>6d}")
    print()
    for cond, eps in sorted(by.items()):
        for e in eps:
            bad = [k for k, v in e["check"]["prefs"].items() if not v]
            print(f"  {cond} {e['persona']} {e['task_id']:12s} success={e['check']['success']} "
                  f"missed_prefs={bad} steps={e['steps']}{' LOOP' if e.get('looped') else ''}")


if __name__ == "__main__":
    main()
