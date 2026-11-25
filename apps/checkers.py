"""Programmatic checkers on final app state.

check(pid, task_id, before, after) -> {
    "success": bool,           # core task done (persona-independent)
    "prefs": {name: bool},     # each persona preference satisfied
    "adherence": float,        # fraction of prefs satisfied
    "joint": bool,             # success and all prefs
    "collateral": [str],       # unexpected changes elsewhere
    "notes": [str],
}
"""

from .personas import PERSONAS
from .tasks import get_task, expected_receipt


def _norm(s: str) -> str:
    return "\n".join(line.strip() for line in s.replace("\r\n", "\n").strip().split("\n"))


def _by_id(items):
    return {x["id"]: x for x in items}


def _strip(x: dict, ignore=("read",)) -> dict:
    return {k: v for k, v in x.items() if k not in ignore}


def _changed(before_items, after_items, allowed_ids=(), ignore=("read",)):
    """Ids of entities added/removed/modified outside allowed_ids."""
    b, a = _by_id(before_items), _by_id(after_items)
    out = []
    for i in set(b) | set(a):
        if i in allowed_ids:
            continue
        if i not in a or i not in b or _strip(a[i], ignore) != _strip(b[i], ignore):
            out.append(i)
    return sorted(out)


def check(pid: str, task_id: str, before: dict, after: dict) -> dict:
    task = get_task(task_id)
    prefs_all = PERSONAS[pid]["prefs"]
    skill, tgt = task["skill"], task["target"]
    keys = before.get("task_keys", {})
    prefs, notes, collateral = {}, [], []
    success = False

    if skill == "send_email":
        p = prefs_all["send_email"]
        before_ids = {m["id"] for m in before["mail"]["messages"]}
        new_sent = [m for m in after["mail"]["messages"]
                    if m["id"] not in before_ids and m["folder"] == "Sent"]
        if len(new_sent) != 1:
            notes.append(f"expected 1 sent email, found {len(new_sent)}")
        if new_sent:
            m = new_sent[-1]
            success = tgt["to"] in m["to"] and tgt["token"].lower() in m["body"].lower()
            if not success:
                notes.append(f"to={m['to']} body missing token? {tgt['token']!r}")
            prefs["cc"] = sorted(m["cc"]) == sorted(p["cc"])
            prefs["bcc"] = sorted(m["bcc"]) == sorted(p["bcc"])
            if p["subject_prefix"]:
                prefs["subject_prefix"] = m["subject"].startswith(p["subject_prefix"])
            if p.get("subject_lowercase"):
                prefs["subject_lowercase"] = bool(m["subject"]) and m["subject"] == m["subject"].lower()
            prefs["signoff"] = _norm(m["body"]).endswith(_norm(p["signoff"]))
        else:
            prefs = {"cc": False, "bcc": False, "signoff": False}
        collateral = _changed(before["mail"]["messages"], after["mail"]["messages"],
                              allowed_ids={m["id"] for m in new_sent})

    elif skill == "file_email":
        p = prefs_all["file_email"]
        msgs = _by_id(after["mail"]["messages"])
        t = msgs.get(keys["target"])
        if t is None:
            notes.append("target email deleted")
        else:
            success = t["folder"] not in ("Inbox", "Trash") or (t["starred"] and t["folder"] != "Trash")
            prefs["folder"] = t["folder"] == p["folder"]
            prefs["star"] = t["starred"] == p["star"]
        collateral = _changed(before["mail"]["messages"], after["mail"]["messages"],
                              allowed_ids={keys["target"]})

    elif skill == "calendar":
        p = prefs_all["calendar"]
        before_ids = {e["id"] for e in before["calendar"]["events"]}
        new = [e for e in after["calendar"]["events"] if e["id"] not in before_ids]
        if len(new) != 1:
            notes.append(f"expected 1 new event, found {len(new)}")
        if new:
            e = new[-1]
            success = (e["date"] == tgt["date"] and tgt["attendee"] in e["attendees"]
                       and tgt["keyword"].lower() in e["title"].lower())
            if not success:
                notes.append(f"date={e['date']} attendees={e['attendees']} title={e['title']!r}")
            prefs["start"] = e["start"] == p[tgt["part"]]
            prefs["duration"] = int(e["duration"]) == p["duration"]
            prefs["video"] = bool(e["video"]) == p["video"]
            prefs["location"] = e["location"].strip() == p["location"]
            prefs["reminder"] = int(e["reminder"]) == p["reminder"]
            if p["title_prefix"]:
                prefs["title_prefix"] = e["title"].startswith(p["title_prefix"])
        else:
            prefs = {k: False for k in ("start", "duration", "video", "location", "reminder")}
        collateral = _changed(before["calendar"]["events"], after["calendar"]["events"],
                              allowed_ids={e["id"] for e in new})

    elif skill == "organize_files":
        folder, name = expected_receipt(pid, tgt["vendor"], tgt["date"])
        f = _by_id(after["drive"]["files"]).get(keys["target"])
        if f is None:
            notes.append("receipt deleted")
            prefs = {"folder": False, "name": False}
        else:
            success = f["folder"] != "/Downloads"
            prefs["folder"] = f["folder"] == folder
            prefs["name"] = f["name"] == name
            if not prefs["name"]:
                notes.append(f"name {f['name']!r} != expected {name!r}")
        collateral = _changed(before["drive"]["files"], after["drive"]["files"],
                              allowed_ids={keys["target"]})
        extra_folders = set(after["drive"]["folders"]) - set(before["drive"]["folders"]) - {folder}
        collateral += [f"new folder {x}" for x in sorted(extra_folders)]

    adherence = sum(prefs.values()) / len(prefs) if prefs else 0.0
    return {
        "success": bool(success),
        "prefs": prefs,
        "adherence": adherence,
        "joint": bool(success) and all(prefs.values()),
        "collateral": collateral,
        "notes": notes,
    }
