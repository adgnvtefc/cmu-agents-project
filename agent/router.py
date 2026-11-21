"""R: route an instruction to one of the user's skill adapters.

Nearest neighbour over the instructions of the user's recorded train demos (TF-IDF cosine), with
a confidence threshold: below it, the agent falls back to the base model (no adapter).

  uv run python -m agent.router --demos-root data/oracle     # accuracy on held-out instructions
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from apps.personas import CONTACTS, PERSONAS
from apps.tasks import get_task, task_ids


# Entities say WHO/WHEN, not WHICH SKILL: ignore them so "Tell Theo ... Friday" is not
# routed to calendar just because calendar demos also mention people and weekdays.
_IGNORE = {w for name, _, _ in CONTACTS for w in name.lower().split()} | {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "next", "today",
    "tomorrow", "morning", "afternoon", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december", "jan", "dec", "nov",
    "the", "a", "an", "to", "of", "on", "in", "my", "and", "for", "with", "that", "it", "is", "be",
    "can", "he", "her", "his", "from", "at", "if", "this", "s", "re", "pm", "am"}


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in _IGNORE]


class Router:
    def __init__(self, examples: list[tuple[str, str]], threshold: float = 0.2):
        """examples: (instruction, skill) pairs from the user's recorded demos."""
        self.threshold = threshold
        docs = [Counter(_tokens(t)) for t, _ in examples]
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.vecs = [(self._vec(d), skill) for d, (_, skill) in zip(docs, examples)]

    def _vec(self, counts: Counter) -> dict:
        v = {w: c * self.idf.get(w, 0.0) for w, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def route(self, instruction: str) -> tuple[str | None, float]:
        q = self._vec(Counter(_tokens(instruction)))
        best, score = None, 0.0
        for v, skill in self.vecs:
            s = sum(q.get(w, 0.0) * x for w, x in v.items())
            if s > score:
                best, score = skill, s
        return (best if score >= self.threshold else None), score


def router_for(demos_root: Path, persona: str) -> Router:
    ex = []
    for m in sorted(demos_root.glob(f"{persona}/*/*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("accepted") and meta["split"] == "train":
            ex.append((meta["instruction"], meta["skill"]))
    return Router(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos-root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.demos_root)
    tot = ok = 0
    for p in PERSONAS:
        r = router_for(root, p)
        for tid in task_ids("test"):
            t = get_task(tid)
            pred, score = r.route(t["instruction"])
            tot += 1
            ok += pred == t["skill"]
            if pred != t["skill"]:
                print(f"  ✗ {p} {tid}: routed to {pred} ({score:.2f}) — {t['instruction']}")
    print(f"router accuracy on held-out instructions: {ok}/{tot}")


if __name__ == "__main__":
    main()
