"""UI-TARS-1.5 prompt construction and action (de)serialization.

Recorded actions (recorder/actions.py) <-> UI-TARS action strings. Coordinates are absolute pixels
on the screenshot; with our 1288x812 viewport (multiple of 28), Qwen2.5-VL's smart_resize is the
identity, so screenshot pixels == model pixels.
"""

import datetime
import json
import os
import re

# PROMPT_DATE=1 adds the environment's "today" to every prompt (training and evaluation alike), so
# relative days ("Tuesday afternoon") can be resolved. Off by default to keep earlier runs reproducible.
PROMPT_DATE = os.environ.get("PROMPT_DATE", "0") == "1"


def date_line() -> str:
    from apps.personas import TODAY
    d = datetime.date.fromisoformat(TODAY)
    return f"Today is {d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year} ({TODAY})."

COMPUTER_USE_PROMPT = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

KEYMAP = {"meta": "ctrl", "control": "ctrl", "escape": "esc", "arrowup": "up", "arrowdown": "down",
          "arrowleft": "left", "arrowright": "right"}


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def box(x, y) -> str:
    return f"<|box_start|>({int(round(x))},{int(round(y))})<|box_end|>"


def to_uitars(action: dict) -> str:
    t = action["type"]
    if t == "click":
        return f"click(start_box='{box(action['x'], action['y'])}')"
    if t == "double_click":
        return f"left_double(start_box='{box(action['x'], action['y'])}')"
    if t == "right_click":
        return f"right_single(start_box='{box(action['x'], action['y'])}')"
    if t == "type":
        return f"type(content='{_esc(action['content'])}')"
    if t == "hotkey":
        return f"hotkey(key='{' '.join(KEYMAP.get(k, k) for k in action['keys'])}')"
    if t == "scroll":
        return f"scroll(start_box='{box(action['x'], action['y'])}', direction='{action['direction']}')"
    if t == "finished":
        return f"finished(content='{_esc(action.get('content', ''))}')"
    raise ValueError(action)


_POINT = r"\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?"


def _point(s: str):
    s = s.replace("<|box_start|>", "").replace("<|box_end|>", "")
    m = re.search(r"<point>\s*(\d+)\s+(\d+)\s*</point>", s) or re.search(_POINT, s)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _arg(call: str, name: str) -> str | None:
    m = re.search(name + r"\s*=\s*'((?:[^'\\]|\\.)*)'", call, re.S)
    if not m:
        return None
    raw = m.group(1)
    try:
        return json.loads('"' + raw.replace('"', '\\"').replace("\\'", "'") + '"')
    except json.JSONDecodeError:
        return raw.replace("\\n", "\n").replace("\\'", "'")


def parse_response(text: str) -> dict:
    """Parse 'Thought: ...\nAction: ...' into {'thought', 'action': {...} | None, 'raw'}."""
    thought = ""
    m = re.search(r"Thought:(.*?)(?:\nAction:|$)", text, re.S)
    if m:
        thought = m.group(1).strip()
    am = re.search(r"Action:\s*(.*)", text, re.S)
    call = (am.group(1) if am else text).strip()
    out = {"thought": thought, "raw": text, "action": None}
    name = call.split("(", 1)[0].strip()
    if name in ("click", "left_single", "left_double", "right_single", "scroll"):
        pt = _point(_arg(call, "start_box") or _arg(call, "point") or call)
        if pt:
            typ = {"click": "click", "left_single": "click", "left_double": "double_click",
                   "right_single": "right_click", "scroll": "scroll"}[name]
            a = {"type": typ, "x": pt[0], "y": pt[1]}
            if typ == "scroll":
                a["direction"] = _arg(call, "direction") or "down"
            out["action"] = a
    elif name == "type":
        out["action"] = {"type": "type", "content": _arg(call, "content") or ""}
    elif name == "hotkey":
        out["action"] = {"type": "hotkey", "keys": (_arg(call, "key") or "").split()}
    elif name == "finished":
        out["action"] = {"type": "finished", "content": _arg(call, "content") or ""}
    elif name == "wait":
        out["action"] = {"type": "wait"}
    return out


def build_messages(instruction: str, screenshots: list, history: list[str], n_images: int = 5,
                   context: str = ""):
    """UI-TARS multi-turn format: user(prompt+img0), assistant(resp0), user(img1), ...

    screenshots: list of image paths for steps 0..k (the last one is the current screen).
    history: assistant responses ('Thought: ...\\nAction: ...') for steps 0..k-1.
    Only the last n_images screenshots are kept; older steps keep their text only.
    Returns (messages, images_in_order).
    """
    k = len(screenshots) - 1
    assert len(history) == k
    keep_from = max(0, k + 1 - n_images)
    system = COMPUTER_USE_PROMPT.format(instruction=instruction)
    if context:
        system = system.replace("## User Instruction", context.strip() + "\n\n## User Instruction")
    if PROMPT_DATE:
        system = system.replace("## User Instruction", date_line() + "\n\n## User Instruction")
    messages = [{"role": "user", "content": [{"type": "text", "text": system}]}]
    images = []
    for i in range(k + 1):
        if i >= keep_from:
            if i == 0:
                messages[0]["content"].append({"type": "image"})
            else:
                messages.append({"role": "user", "content": [{"type": "image"}]})
            images.append(screenshots[i])
        if i < k:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": history[i]}]})
    return messages, images
