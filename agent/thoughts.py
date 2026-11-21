"""Deterministic step thoughts for recorded demos.

UI-TARS is trained to write a short `Thought:` before each action. Training on empty thoughts
pushes the model off its native output style, so every recorded step gets a template thought
built from the gold action and the element it targets. Thoughts only describe the current
step (never later steps), and contain no persona rationale: the preference lives in the action.
"""

import re


def _pretty(target: dict | None) -> str:
    if not target:
        return "the element"
    label = (target.get("label") or "").strip()
    if not label and _is_field(target):  # fields: never use the typed value as a name
        label = (target.get("placeholder") or target.get("name") or "").strip()
    elif not label:
        label = (target.get("text") or "").strip()
    if not label and target.get("testid"):
        label = re.sub(r"[-_]", " ", target["testid"]).strip()
    label = re.sub(r"^[^\w(]+", "", re.sub(r"\s+", " ", label))  # drop leading icons/symbols
    return label[:50] or "the element"


def _is_field(target: dict | None) -> bool:
    return bool(target) and target.get("tag") in ("input", "textarea") and target.get("type") not in (
        "radio", "checkbox", "submit", "button", "hidden")


def template_thought(step: dict) -> str:
    a, tgt = step["action"], step.get("target")
    t = a["type"]
    name = _pretty(tgt)
    if t in ("click", "double_click"):
        if _is_field(tgt):
            return f"Next, I'll click the {name} field."
        if tgt and tgt.get("type") == "checkbox":
            return f"I'll toggle the \"{name}\" checkbox."
        return f"Next, I'll click \"{name}\"."
    if t == "right_click":
        return f"I'll right-click \"{name}\"."
    if t == "type":
        content = a["content"].rstrip("\n").replace("\n", " ")
        short = content if len(content) <= 48 else content[:45] + "..."
        where = f" into the {name} field" if tgt else ""
        return f"I'll type \"{short}\"{where}."
    if t == "hotkey":
        keys = [k.lower() for k in a["keys"]]
        if keys[-1] == "a" and len(keys) > 1:
            return "I'll select all the existing text so I can replace it."
        return f"I'll press {'+'.join(keys)}."
    if t == "scroll":
        return f"I'll scroll {a['direction']} to see more."
    if t == "finished":
        return "Everything the task asked for is done."
    return ""


def step_response(step: dict, to_uitars) -> str:
    thought = step.get("thought") or template_thought(step)
    return f"Thought: {thought}\nAction: {to_uitars(step['action'])}"
