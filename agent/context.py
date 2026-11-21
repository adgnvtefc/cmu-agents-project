"""Prompt context for the non-parametric arms.

C1 (raw demos in context): the user's train demos for the skill, as text action logs (thought +
UI-TARS action per step). Screenshots are omitted: 5 demos x ~12 screenshots would exceed the
context budget (see PLAN.md), so all arms see the same text-level information.
C2 (skill file): a SKILL.md written from the same demos (generator pending: see STATUS.md).
"""

import json
from pathlib import Path

from agent.thoughts import template_thought
from agent.uitars import to_uitars


def load_demos(root: str | Path, persona: str, skill: str, split: str = "train") -> list[Path]:
    out = []
    for m in sorted(Path(root).glob(f"{persona}/*/*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("accepted") and meta["skill"] == skill and meta["split"] == split:
            out.append(m.parent)
    return out


def demos_context(demo_dirs: list[Path]) -> str:
    lines = ["## This user's past demonstrations of similar tasks",
             "Follow this user's habits (recipients, folders, names, times, settings) when the instruction "
             "leaves details unspecified."]
    for i, d in enumerate(demo_dirs, 1):
        meta = json.loads((d / "meta.json").read_text())
        steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
        lines.append(f"\n### Demonstration {i}: \"{meta['instruction']}\"")
        for k, s in enumerate(steps, 1):
            thought = s.get("thought") or template_thought(s)
            lines.append(f"{k}. {thought} → {to_uitars(s['action'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    # uv run python -m agent.context data/oracle alice calendar > ctx.md
    import sys
    root, persona, skill = sys.argv[1:4]
    print(demos_context(load_demos(root, persona, skill)))
