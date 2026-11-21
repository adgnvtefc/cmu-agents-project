"""Turn raw DOM input events into agent-level actions (click / type / hotkey / scroll / ...).

Pure logic, no Playwright, so it can be unit-tested. Every emitted action carries the frame
(screenshot) chosen at the time of its FIRST underlying event, i.e. the last frame that finished
rendering before the user acted (pre-action keyframe, as in AgentNetTool).
"""

from dataclasses import dataclass, field

MODIFIER_KEYS = {"Shift", "Control", "Alt", "Meta", "CapsLock", "Fn"}
NAMED_KEYS = {"Enter", "Tab", "Escape", "Backspace", "Delete", "ArrowUp", "ArrowDown", "ArrowLeft",
              "ArrowRight", "PageUp", "PageDown", "Home", "End", " "}
DOUBLE_CLICK_S = 0.5
WHEEL_GAP_S = 0.6


@dataclass
class Frame:
    t_start: float
    t_end: float
    png: bytes


@dataclass
class Pending:
    kind: str  # "type" | "scroll"
    t: float
    frame: Frame | None
    stale: bool
    target: dict | None
    url: str
    data: dict = field(default_factory=dict)


def _elem_key(ev: dict) -> tuple:
    tgt = ev.get("target") or {}
    return (ev.get("url"), tgt.get("testid") or tgt.get("name") or tgt.get("placeholder"))


class ActionBuilder:
    def __init__(self, settle_s: float = 0.15):
        self.actions: list[dict] = []
        self.pending: Pending | None = None
        self.last_keydown: dict | None = None
        self.last_t: float | None = None
        self.settle_s = settle_s

    # -- frame choice -----------------------------------------------------------------------
    def choose_frame(self, frames: list[Frame], t: float) -> tuple[Frame | None, bool]:
        """Latest frame fully captured before t. Stale if not captured well after the previous event."""
        cands = [f for f in frames if f.t_end < t - 0.02]
        if not cands:
            return None, True
        f = cands[-1]
        stale = self.last_t is not None and f.t_start < self.last_t + self.settle_s
        return f, stale

    # -- emission ---------------------------------------------------------------------------
    def _emit(self, t, frame, stale, action, target, url):
        self.actions.append({"t": t, "frame": frame, "stale": stale, "action": action,
                             "target": target, "url": url})

    def flush(self):
        p, self.pending = self.pending, None
        if p is None:
            return
        if p.kind == "type":
            before, after, sel = p.data["before"], p.data["after"], p.data.get("sel")
            whole = sel is not None and before and sel[0] == 0 and sel[1] == len(before)
            if whole:
                mode, content = "replace", after
            elif after.startswith(before) and (sel is None or sel[0] == len(before)):
                mode, content = "append", after[len(before):]
            else:
                mode, content = "edit", after
            if p.data.get("submit"):
                content += "\n"
            act = {"type": "type", "content": content, "mode": mode,
                   "value_before": before, "value_after": after}
            self._emit(p.t, p.frame, p.stale, act, p.target, p.url)
        elif p.kind == "scroll":
            dy, dx = p.data["dy"], p.data["dx"]
            if abs(dy) >= abs(dx):
                direction = "down" if dy > 0 else "up"
            else:
                direction = "right" if dx > 0 else "left"
            act = {"type": "scroll", "x": p.data["x"], "y": p.data["y"], "direction": direction,
                   "amount": round(max(abs(dy), abs(dx)))}
            self._emit(p.t, p.frame, p.stale, act, p.target, p.url)

    # -- feeding ----------------------------------------------------------------------------
    def feed(self, t: float, ev: dict, frames: list[Frame]):
        kind = ev["kind"]
        frame, stale = self.choose_frame(frames, t)

        if kind == "wheel":
            if self.pending and self.pending.kind == "scroll" and t - self.pending.data["t_last"] < WHEEL_GAP_S:
                self.pending.data["dx"] += ev["dx"]
                self.pending.data["dy"] += ev["dy"]
                self.pending.data["t_last"] = t
            else:
                self.flush()
                self.pending = Pending("scroll", t, frame, stale, ev.get("target"), ev["url"],
                                       {"x": ev["x"], "y": ev["y"], "dx": ev["dx"], "dy": ev["dy"], "t_last": t})
            self.last_t = t
            return

        if kind == "input":
            key = _elem_key(ev)
            if self.pending and self.pending.kind == "type" and self.pending.data["key"] == key:
                self.pending.data["after"] = ev["value"]
            else:
                self.flush()
                kd = self.last_keydown
                if kd and _elem_key(kd["ev"]) == key and t - kd["t"] < 2.0:
                    before, sel, f0, s0, t0 = kd["ev"]["value"] or "", kd["ev"]["sel"], kd["frame"], kd["stale"], kd["t"]
                else:  # e.g. paste via context menu, autofill
                    before, sel, f0, s0, t0 = "", None, frame, stale, t
                self.pending = Pending("type", t0, f0, s0, ev.get("target"), ev["url"],
                                       {"key": key, "before": before, "after": ev["value"], "sel": sel})
            self.last_t = t
            return

        if kind == "keydown":
            k = ev["key"]
            if k in MODIFIER_KEYS:
                return
            mods = [m for m, on in (("ctrl", ev["ctrl"]), ("meta", ev["meta"]), ("alt", ev["alt"])) if on]
            if ev["editable"] and not mods and (len(k) == 1 or k in ("Backspace", "Delete")):
                # Printable / editing key inside a text field: becomes part of a `type` action via
                # the following input event. Remember pre-keystroke value + frame.
                if not (self.pending and self.pending.kind == "type" and self.pending.data["key"] == _elem_key(ev)):
                    self.last_keydown = {"t": t, "ev": ev, "frame": frame, "stale": stale}
                self.last_t = t
                return
            if ev["editable"] and mods and k.lower() == "v":
                self.last_keydown = {"t": t, "ev": ev, "frame": frame, "stale": stale}
                self.last_t = t
                return  # paste: captured by the input event
            if k == "Enter" and ev["editable"] and (ev.get("target") or {}).get("tag") == "input":
                # UI-TARS convention: type(content='...\n') submits.
                if self.pending and self.pending.kind == "type" and self.pending.data["key"] == _elem_key(ev):
                    self.pending.data["submit"] = True
                    self.flush()
                else:
                    self.flush()
                    self._emit(t, frame, stale, {"type": "hotkey", "keys": ["enter"]}, ev.get("target"), ev["url"])
                self.last_t = t
                return
            if k == "Enter" and ev["editable"]:
                self.last_t = t
                return  # newline in textarea: handled by input event
            self.flush()
            name = "space" if k == " " else k.lower()
            keys = mods + (["shift"] if ev["shift"] and (len(k) > 1 or mods) else []) + [name]
            self._emit(t, frame, stale, {"type": "hotkey", "keys": keys}, ev.get("target"), ev["url"])
            self.last_t = t
            return

        if kind == "click":
            self.flush()
            prev = self.actions[-1] if self.actions else None
            if (ev["detail"] == 2 and prev and prev["action"]["type"] == "click"
                    and t - prev["t"] < DOUBLE_CLICK_S):
                prev["action"]["type"] = "double_click"
            else:
                typ = "right_click" if ev["button"] == 2 else "click"
                self._emit(t, frame, stale, {"type": typ, "x": round(ev["x"]), "y": round(ev["y"])},
                           ev.get("target"), ev["url"])
            self.last_t = t
            return

    def finish(self, t: float, frames: list[Frame], url: str):
        self.flush()
        frame, _ = self.choose_frame(frames, t)
        self._emit(t, frame, False, {"type": "finished"}, None, url)
