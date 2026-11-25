"""Recording self-consistency: for every click step, the element under the gold click point in the
frame's element map must be the element the click actually hit (target.id). A mismatch means the
screenshot/element map does not show the state the user acted on.

  uv run python -m scripts.check_demos data/oracle [data/raw]
"""

import json
import sys
from pathlib import Path

from eval.scoring import resolve


def main():
    total = bad = 0
    per_demo_bad = 0
    for root in sys.argv[1:]:
        for m in sorted(Path(root).rglob("meta.json")):
            d = m.parent
            meta = json.loads(m.read_text())
            demo_bad = []
            for line in (d / "steps.jsonl").read_text().splitlines():
                s = json.loads(line)
                a = s["action"]
                if a["type"] not in ("click", "double_click", "right_click") or not s.get("target"):
                    continue
                total += 1
                got = resolve(a["x"], a["y"], s.get("elements"))
                if got != s["target"].get("id"):
                    bad += 1
                    demo_bad.append(f"step {s['step']}: map→{got} hit→{s['target'].get('id')}")
            if demo_bad:
                per_demo_bad += 1
                print(f"{meta['persona']} {meta['task_id']}: " + "; ".join(demo_bad))
    print(f"\nclick steps: {total}, inconsistent: {bad}, demos affected: {per_demo_bad}")


if __name__ == "__main__":
    main()
