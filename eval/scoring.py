"""Step-level scoring shared by offline evaluation and analysis.

A predicted click is correct if it would activate the SAME element as the gold click, judged
with the element map recorded alongside the gold frame (smallest interactive element under the
point; a <label> resolves to its control). Falls back to the gold target's bbox when a demo has
no element map.
"""


def resolve(x: float, y: float, elements: list | None) -> str | None:
    if not elements:
        return None
    hits = [e for e in elements
            if e["bbox"][0] <= x <= e["bbox"][0] + e["bbox"][2] and e["bbox"][1] <= y <= e["bbox"][1] + e["bbox"][3]]
    if not hits:
        return None
    return min(hits, key=lambda e: e["bbox"][2] * e["bbox"][3])["id"]


def _norm_keys(keys):
    m = {"meta": "ctrl", "control": "ctrl", "escape": "esc"}
    return [m.get(k.lower(), k.lower()) for k in keys]


def score(pred: dict | None, step: dict) -> dict:
    """pred: parsed action or None. step: recorded gold step (action, target, elements)."""
    gold = step["action"]
    g = gold["type"]
    p = pred["type"] if pred else None
    same_type = p == g or (g == "double_click" and p == "click")
    out = {"type_ok": same_type, "ground_ok": None, "content_ok": None}
    if g in ("click", "double_click", "right_click"):
        if pred and "x" in pred:
            els = step.get("elements")
            gold_id = resolve(gold["x"], gold["y"], els)
            if gold_id is not None:
                out["ground_ok"] = resolve(pred["x"], pred["y"], els) == gold_id
            elif step.get("target"):
                x, y, w, h = step["target"]["bbox"]
                out["ground_ok"] = x <= pred["x"] <= x + w and y <= pred["y"] <= y + h
            else:
                out["ground_ok"] = abs(pred["x"] - gold["x"]) < 20 and abs(pred["y"] - gold["y"]) < 20
        else:
            out["ground_ok"] = False
    if g == "type":
        out["content_ok"] = bool(pred) and (pred.get("content") or "").rstrip("\n").strip() == gold["content"].rstrip("\n").strip()
    if g == "hotkey":
        out["content_ok"] = bool(pred) and _norm_keys(pred.get("keys") or []) == _norm_keys(gold["keys"])
    checks = [v for v in (out["ground_ok"], out["content_ok"]) if v is not None]
    out["step_ok"] = bool(same_type and all(checks))
    return out
