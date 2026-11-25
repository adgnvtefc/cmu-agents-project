"""C2: induce a SKILL.md from a user's demos of one skill.

The writer sees the same information as the other arms (instructions, step thoughts, actions with
the element each click hit and the text typed), never the persona card. Default writer is a local
model (Qwen2.5-VL-7B-Instruct, text-only); PLAN.md calls for a frontier model, so swap the writer
when an API key is available and regenerate.

  uv run python -m agent.skillfile --demos-root data/oracle --personas alice --skills calendar
Output: skills/<demos-root name>/<persona>/<skill>.md
"""

import argparse
import json
from pathlib import Path

from agent.context import load_demos
from agent.thoughts import template_thought
from apps.personas import PERSONAS
from apps.tasks import SKILLS

ROOT = Path(__file__).resolve().parents[1]
WRITER = "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"

PROMPT = """Below are {n} recorded demonstrations of ONE user doing the same kind of task in a web app.
Write a SKILL.md that lets an assistant do NEW instances of this task exactly the way THIS user does.

Use these sections:
# Skill: <short name>
## When to use
## Steps
(numbered, general: use slots like {{recipient}} or {{date}} for things that change between demos)
## This user's habits
(every value that stays THE SAME across the demos: addresses they always add, folders, names,
durations, times of day, sign-offs, prefixes, checkboxes. Quote exact strings.)
## Done when

Only use information from the demos. Under 250 words.

{demos}
"""


def demo_text(demo_dirs: list[Path]) -> str:
    out = []
    for i, d in enumerate(demo_dirs, 1):
        meta = json.loads((d / "meta.json").read_text())
        steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
        out.append(f"### Demonstration {i}: \"{meta['instruction']}\"")
        for k, s in enumerate(steps, 1):
            a = s["action"]
            detail = ""
            if a["type"] == "type":
                detail = f" [typed exactly: {a['content']!r}]"
            out.append(f"{k}. {s.get('thought') or template_thought(s)}{detail}")
    return "\n".join(out)


def write_skill(model, processor, demo_dirs: list[Path]) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    msgs = [{"role": "user", "content": PROMPT.format(n=len(demo_dirs), demos=demo_text(demo_dirs))}]
    prompt = apply_chat_template(processor, model.config, msgs, num_images=0)
    return generate(model, processor, prompt, max_tokens=600, temperature=0.0, verbose=False).text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS))
    ap.add_argument("--skills", nargs="+", default=SKILLS)
    ap.add_argument("--writer", default=WRITER)
    args = ap.parse_args()
    from mlx_vlm import load
    model, processor = load(args.writer)
    root = Path(args.demos_root)
    for p in args.personas:
        for s in args.skills:
            demos = load_demos(root, p, s)
            if not demos:
                continue
            md = write_skill(model, processor, demos)
            out = ROOT / "skills" / root.name / p / f"{s}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md + "\n")
            print(f"{p}/{s}: {len(demos)} demos → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
