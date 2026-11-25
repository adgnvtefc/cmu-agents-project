"""Mock Northwind Workspace: Mail, Calendar, Drive.

Run:  uv run uvicorn apps.server:app --port 8917
Hidden control endpoints (agents must never be allowed to open /__*):
  POST /__reset  {"persona": "alice", "task_id": "send_email_0"}  -> fresh state (+ task setup)
  GET  /__state                                                   -> full JSON state
  GET  /__check                                                   -> checker result vs reset state
  GET  /__tasks                                                   -> task list
"""

import copy
import datetime as dt
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .checkers import check
from .seed import base_state
from .tasks import apply_setup, get_task, task_ids

HERE = Path(__file__).parent
app = FastAPI(title="Northwind Workspace (mock)")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

STATE: dict = {}
BEFORE: dict = {}


def _reset(persona: str, task_id: str | None):
    global STATE, BEFORE
    s = base_state(persona)
    if task_id:
        s = apply_setup(s, get_task(task_id))
    STATE = s
    BEFORE = copy.deepcopy(s)


_reset("alice", None)


# ------------------------------------------------------------------ helpers
def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def fmt_date(s: str) -> str:
    return d(s).strftime("%a, %b %-d")


def fmt_mail_date(s: str) -> str:
    day, time = s.split(" ")
    if day == STATE["today"]:
        return time
    return d(day).strftime("%b %-d")


def add_minutes(hhmm: str, mins: int) -> str:
    h, m = map(int, hhmm.split(":"))
    t = h * 60 + m + int(mins)
    return f"{t // 60:02d}:{t % 60:02d}"


templates.env.filters["fmt_date"] = fmt_date
templates.env.filters["fmt_mail_date"] = fmt_mail_date
templates.env.globals["add_minutes"] = add_minutes


def render(request: Request, name: str, **ctx):
    ctx.setdefault("msg", request.query_params.get("msg", ""))
    return templates.TemplateResponse(request, name, {"S": STATE, **ctx})


def redirect(url: str, msg: str = ""):
    if msg:
        url += ("&" if "?" in url else "?") + "msg=" + quote(msg)
    return RedirectResponse(url, status_code=303)


def resolve_addresses(raw: str) -> list[str]:
    """Accept 'Raj Patel, mei.tanaka@northwind.test' -> list of emails."""
    out = []
    people = STATE["contacts"] + [STATE["me"]]
    for tok in [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]:
        if "@" in tok:
            out.append(tok.lower())
            continue
        hits = [c for c in people if tok.lower() in c["name"].lower()]
        out.append(hits[0]["email"] if len(hits) == 1 else tok)
    return out


def next_id(kind: str) -> str:
    STATE["counters"][kind] += 1
    return f"{kind}{STATE['counters'][kind]:03d}"


def msg_by_id(mid: str) -> dict:
    return next(m for m in STATE["mail"]["messages"] if m["id"] == mid)


# ------------------------------------------------------------------ hidden control API
@app.post("/__reset")
async def api_reset(payload: dict):
    _reset(payload.get("persona", "alice"), payload.get("task_id"))
    task = get_task(payload["task_id"]) if payload.get("task_id") else None
    return {"ok": True, "persona": STATE["persona"], "task": task}


@app.get("/__state")
async def api_state():
    return JSONResponse(STATE)


@app.get("/__before")
async def api_before():
    return JSONResponse(BEFORE)


@app.get("/__check")
async def api_check():
    if not STATE.get("task_id"):
        return {"error": "no task loaded"}
    return check(STATE["persona"], STATE["task_id"], BEFORE, STATE)


@app.get("/__tasks")
async def api_tasks():
    return [get_task(t) for t in task_ids()]


# ------------------------------------------------------------------ mail
@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/mail")


@app.get("/mail", response_class=HTMLResponse)
async def mail_list(request: Request, folder: str = "Inbox", q: str = "", only: str = ""):
    msgs = STATE["mail"]["messages"]
    if q:
        ql = q.lower()
        msgs = [m for m in msgs if m["folder"] != "Trash" and any(
            ql in (m[k] or "").lower() for k in ("subject", "body", "from", "from_name"))]
        title = f'Search results for "{q}"'
    else:
        msgs = [m for m in msgs if m["folder"] == folder]
        title = folder
    if only == "unread":
        msgs = [m for m in msgs if not m["read"]]
    elif only == "starred":
        msgs = [m for m in msgs if m["starred"]]
    msgs = sorted(msgs, key=lambda m: m["date"], reverse=True)
    counts = {f: sum(1 for m in STATE["mail"]["messages"] if m["folder"] == f and not m["read"])
              for f in STATE["mail"]["folders"]}
    return render(request, "mail_list.html", app_name="mail", folder=None if q else folder,
                  q=q, only=only, msgs=msgs, title=title, counts=counts)


@app.get("/mail/m/{mid}", response_class=HTMLResponse)
async def mail_view(request: Request, mid: str):
    m = msg_by_id(mid)
    m["read"] = True
    counts = {f: sum(1 for x in STATE["mail"]["messages"] if x["folder"] == f and not x["read"])
              for f in STATE["mail"]["folders"]}
    return render(request, "mail_view.html", app_name="mail", m=m, folder=m["folder"], counts=counts, q="")


@app.post("/mail/m/{mid}/star")
async def mail_star(mid: str):
    m = msg_by_id(mid)
    m["starred"] = not m["starred"]
    return redirect(f"/mail/m/{mid}", "Starred" if m["starred"] else "Unstarred")


@app.post("/mail/m/{mid}/move")
async def mail_move(mid: str, folder: str = Form(...)):
    m = msg_by_id(mid)
    src = m["folder"]
    m["folder"] = folder
    return redirect(f"/mail?folder={quote(src)}", f"Moved to {folder}")


@app.post("/mail/m/{mid}/unread")
async def mail_unread(mid: str):
    m = msg_by_id(mid)
    m["read"] = False
    return redirect(f"/mail?folder={quote(m['folder'])}", "Marked as unread")


@app.get("/mail/compose", response_class=HTMLResponse)
async def compose(request: Request, mode: str = "", ref: str = ""):
    draft = {"to": "", "cc": "", "bcc": "", "subject": "", "body": ""}
    if ref:
        m = msg_by_id(ref)
        quoted = "\n\n" + f"On {m['date']}, {m['from_name']} wrote:\n" + "\n".join(
            "> " + line for line in m["body"].split("\n"))
        if mode in ("reply", "replyall"):
            draft["to"] = m["from"]
            draft["subject"] = m["subject"] if m["subject"].startswith("Re:") else "Re: " + m["subject"]
            draft["body"] = quoted
            if mode == "replyall":
                others = [a for a in m["to"] + m["cc"] if a != STATE["me"]["email"]]
                draft["cc"] = ", ".join(others)
        elif mode == "forward":
            draft["subject"] = "Fwd: " + m["subject"]
            draft["body"] = quoted
    counts = {f: sum(1 for x in STATE["mail"]["messages"] if x["folder"] == f and not x["read"])
              for f in STATE["mail"]["folders"]}
    return render(request, "mail_compose.html", app_name="mail", draft=draft, folder=None,
                  counts=counts, q="")


@app.post("/mail/send")
async def mail_send(to: str = Form(""), cc: str = Form(""), bcc: str = Form(""),
                    subject: str = Form(""), body: str = Form("")):
    if not to.strip():
        return redirect("/mail/compose", "Add at least one recipient")
    STATE["mail"]["messages"].append({
        "id": next_id("m"), "folder": "Sent", "from": STATE["me"]["email"],
        "from_name": STATE["me"]["name"], "to": resolve_addresses(to), "cc": resolve_addresses(cc),
        "bcc": resolve_addresses(bcc), "subject": subject.strip(), "body": body.replace("\r\n", "\n"),
        "date": f"{STATE['today']} {STATE['now']}", "read": True, "starred": False,
    })
    return redirect("/mail?folder=Inbox", "Message sent")


# ------------------------------------------------------------------ calendar
HOURS = list(range(8, 19))


@app.get("/calendar", response_class=HTMLResponse)
async def calendar(request: Request, week: str = ""):
    today = d(STATE["today"])
    start = d(week) if week else today - dt.timedelta(days=today.weekday())
    days = [start + dt.timedelta(days=i) for i in range(5)]
    by_day = {x.isoformat(): [] for x in days}
    for e in STATE["calendar"]["events"]:
        if e["date"] in by_day:
            by_day[e["date"]].append(e)
    return render(request, "calendar.html", app_name="calendar", days=days, by_day=by_day,
                  hours=HOURS, prev=(start - dt.timedelta(days=7)).isoformat(),
                  next=(start + dt.timedelta(days=7)).isoformat(), today=today.isoformat())


DURATIONS = [15, 30, 45, 50, 60, 90]
REMINDERS = [0, 5, 10, 30, 60]


@app.get("/calendar/new", response_class=HTMLResponse)
async def cal_new(request: Request):
    ev = {"title": "", "date": "", "start": "", "duration": 60, "attendees": [], "location": "",
          "video": False, "reminder": 0, "description": ""}
    return render(request, "event_form.html", app_name="calendar", ev=ev, action="/calendar/new",
                  durations=DURATIONS, reminders=REMINDERS)


def _event_from_form(title, date, start, duration, guests, location, video, reminder, description):
    return {"title": title.strip(), "date": date.strip(), "start": start.strip(),
            "duration": int(duration), "attendees": resolve_addresses(guests),
            "location": location.strip(), "video": video == "on", "reminder": int(reminder),
            "description": description}


def _valid(ev) -> str:
    try:
        d(ev["date"])
    except ValueError:
        return "Date must look like 2026-11-18"
    try:
        h, m = map(int, ev["start"].split(":"))
        assert 0 <= h < 24 and 0 <= m < 60
    except Exception:
        return "Start time must look like 14:00"
    if not ev["title"]:
        return "Add a title"
    return ""


@app.post("/calendar/new")
async def cal_create(title: str = Form(""), date: str = Form(""), start: str = Form(""),
                     duration: int = Form(60), guests: str = Form(""), location: str = Form(""),
                     video: str = Form(""), reminder: int = Form(0), description: str = Form("")):
    ev = _event_from_form(title, date, start, duration, guests, location, video, reminder, description)
    if err := _valid(ev):
        return redirect("/calendar/new", err)
    ev["id"] = next_id("e")
    STATE["calendar"]["events"].append(ev)
    monday = d(ev["date"]) - dt.timedelta(days=d(ev["date"]).weekday())
    return redirect(f"/calendar?week={monday.isoformat()}", "Event created")


@app.get("/calendar/e/{eid}", response_class=HTMLResponse)
async def cal_view(request: Request, eid: str):
    ev = next(e for e in STATE["calendar"]["events"] if e["id"] == eid)
    return render(request, "event_view.html", app_name="calendar", ev=ev)


@app.get("/calendar/e/{eid}/edit", response_class=HTMLResponse)
async def cal_edit(request: Request, eid: str):
    ev = next(e for e in STATE["calendar"]["events"] if e["id"] == eid)
    return render(request, "event_form.html", app_name="calendar", ev=ev,
                  action=f"/calendar/e/{eid}/edit", durations=DURATIONS, reminders=REMINDERS)


@app.post("/calendar/e/{eid}/edit")
async def cal_update(eid: str, title: str = Form(""), date: str = Form(""), start: str = Form(""),
                     duration: int = Form(60), guests: str = Form(""), location: str = Form(""),
                     video: str = Form(""), reminder: int = Form(0), description: str = Form("")):
    ev = next(e for e in STATE["calendar"]["events"] if e["id"] == eid)
    new = _event_from_form(title, date, start, duration, guests, location, video, reminder, description)
    if err := _valid(new):
        return redirect(f"/calendar/e/{eid}/edit", err)
    ev.update(new)
    return redirect(f"/calendar/e/{eid}", "Event updated")


@app.post("/calendar/e/{eid}/delete")
async def cal_delete(eid: str):
    ev = next(e for e in STATE["calendar"]["events"] if e["id"] == eid)
    STATE["calendar"]["events"].remove(ev)
    monday = d(ev["date"]) - dt.timedelta(days=d(ev["date"]).weekday())
    return redirect(f"/calendar?week={monday.isoformat()}", "Event deleted")


# ------------------------------------------------------------------ drive
def _folder_tree():
    out = []
    for f in sorted(STATE["drive"]["folders"]):
        depth = 0 if f == "/" else f.count("/")
        out.append({"path": f, "name": "My Drive" if f == "/" else f.rsplit("/", 1)[1], "depth": depth})
    return out


@app.get("/drive", response_class=HTMLResponse)
async def drive(request: Request, folder: str = "/"):
    prefix = folder.rstrip("/") + "/"
    subfolders = [f for f in STATE["drive"]["folders"]
                  if f != folder and f.startswith(prefix) and "/" not in f[len(prefix):]]
    files = sorted([f for f in STATE["drive"]["files"] if f["folder"] == folder], key=lambda f: f["name"].lower())
    crumbs, acc = [("My Drive", "/")], ""
    for part in [p for p in folder.split("/") if p]:
        acc += "/" + part
        crumbs.append((part, acc))
    return render(request, "drive.html", app_name="drive", folder=folder, tree=_folder_tree(),
                  subfolders=sorted(subfolders), files=files, crumbs=crumbs,
                  new_folder=request.query_params.get("new") == "1")


@app.post("/drive/folder")
async def drive_new_folder(parent: str = Form("/"), name: str = Form("")):
    name = name.strip().strip("/")
    if not name:
        return redirect(f"/drive?folder={quote(parent)}", "Folder name can't be empty")
    path = (parent.rstrip("/") + "/" + name) if parent != "/" else "/" + name
    if path not in STATE["drive"]["folders"]:
        STATE["drive"]["folders"].append(path)
    return redirect(f"/drive?folder={quote(path)}", f"Created folder {name}")


@app.get("/drive/f/{fid}", response_class=HTMLResponse)
async def drive_file(request: Request, fid: str):
    f = next(x for x in STATE["drive"]["files"] if x["id"] == fid)
    return render(request, "drive_file.html", app_name="drive", f=f, tree=_folder_tree(),
                  folder=f["folder"], moving=request.query_params.get("move") == "1")


@app.post("/drive/f/{fid}/rename")
async def drive_rename(fid: str, name: str = Form("")):
    f = next(x for x in STATE["drive"]["files"] if x["id"] == fid)
    if name.strip():
        f["name"] = name.strip()
    return redirect(f"/drive/f/{fid}", "Renamed")


@app.post("/drive/f/{fid}/move")
async def drive_move(fid: str, folder: str = Form(...)):
    f = next(x for x in STATE["drive"]["files"] if x["id"] == fid)
    f["folder"] = folder
    return redirect(f"/drive?folder={quote(folder)}", f"Moved {f['name']}")


@app.post("/drive/f/{fid}/star")
async def drive_star(fid: str):
    f = next(x for x in STATE["drive"]["files"] if x["id"] == fid)
    f["starred"] = not f["starred"]
    return redirect(f"/drive/f/{fid}", "Starred" if f["starred"] else "Unstarred")


@app.post("/drive/f/{fid}/delete")
async def drive_delete(fid: str):
    f = next(x for x in STATE["drive"]["files"] if x["id"] == fid)
    STATE["drive"]["files"].remove(f)
    return redirect(f"/drive?folder={quote(f['folder'])}", f"Deleted {f['name']}")
