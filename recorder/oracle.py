"""Scripted 'oracle' demonstrations: perform each task the way the persona would, with real
mouse/keyboard input. Used to (1) test the recorder without a human and (2) validate checkers
(every oracle demo must pass; a no-op must fail).
"""

import asyncio

from apps.personas import PERSONAS
from apps.tasks import CALENDAR, SEND_EMAIL, expected_receipt, get_task

PAUSE = 0.45  # human-ish pause after each action (recorder needs frames between actions)


async def _pause():
    await asyncio.sleep(PAUSE)


async def click(page, selector: str):
    loc = page.locator(selector).first
    await loc.scroll_into_view_if_needed()
    box = await loc.bounding_box()
    await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    try:
        await page.wait_for_load_state("load", timeout=3000)
    except Exception:
        pass
    await _pause()


async def type_text(page, text: str):
    await page.keyboard.type(text, delay=25)
    await _pause()


async def fill_field(page, testid: str, text: str, replace: bool = False):
    await click(page, f"[data-testid='{testid}']")
    if replace:
        await page.keyboard.press("ControlOrMeta+A")
        await _pause()
    await type_text(page, text)


def tid(x: str) -> str:
    return f"[data-testid='{x}']"


def make_driver(persona: str, task_id: str, state_before: dict):
    task = get_task(task_id)
    prefs = PERSONAS[persona]["prefs"]
    skill = task["skill"]
    keys = state_before.get("task_keys", {})

    if skill == "send_email":
        p = prefs["send_email"]
        k = int(task_id.rsplit("_", 1)[1])
        handle, instruction, token = SEND_EMAIL[k]
        first = handle.split(".")[0].capitalize()
        subject = p["subject_prefix"] + "Quick update"
        if p.get("subject_lowercase"):
            subject = subject.lower()
        body = f"Hi {first},\n\n{instruction}\n\n{p['signoff']}"

        async def drive(page):
            await click(page, tid("compose"))
            await fill_field(page, "compose-to", task["target"]["to"])
            if p["cc"]:
                await fill_field(page, "compose-cc", ", ".join(p["cc"]))
            if p["bcc"]:
                await fill_field(page, "compose-bcc", ", ".join(p["bcc"]))
            await fill_field(page, "compose-subject", subject)
            await fill_field(page, "compose-body", body)
            await click(page, tid("send"))
        return drive

    if skill == "file_email":
        p = prefs["file_email"]
        target = next(m for m in state_before["mail"]["messages"] if m["id"] == keys["target"])

        async def drive(page):
            await fill_field(page, "mail-search", target["subject"].split("—")[0].strip())
            await click(page, tid("mail-search-btn"))
            await click(page, tid(f"msg-{target['id']}"))
            if p["star"]:
                await click(page, tid("star"))
            await click(page, tid(f"move-{p['folder']}"))
        return drive

    if skill == "calendar":
        p = prefs["calendar"]
        k = int(task_id.rsplit("_", 1)[1])
        handle, _, date, part, kw = CALENDAR[k]
        first = handle.split(".")[0].capitalize()
        title = f"{p['title_prefix']}{kw.capitalize()} chat with {first}"

        async def drive(page):
            await click(page, tid("new-event"))
            await fill_field(page, "ev-title", title)
            await fill_field(page, "ev-date", date)
            await fill_field(page, "ev-start", p[part])
            await click(page, f"label:has({tid('ev-dur-' + str(p['duration']))})")
            await fill_field(page, "ev-guests", task["target"]["attendee"])
            if p["location"]:
                await fill_field(page, "ev-location", p["location"])
            if p["video"]:
                await click(page, tid("ev-video"))
            await click(page, f"label:has({tid('ev-rem-' + str(p['reminder']))})")
            await click(page, tid("ev-save"))
        return drive

    if skill == "organize_files":
        folder, name = expected_receipt(persona, task["target"]["vendor"], task["target"]["date"])
        fid = keys["target"]
        need_folder = folder not in state_before["drive"]["folders"]

        async def drive(page):
            if need_folder:
                parent, new = folder.rsplit("/", 1)
                await click(page, tid(f"tree-{parent or '/'}"))
                await click(page, tid("new-folder"))
                await fill_field(page, "new-folder-name", new)
                await click(page, tid("new-folder-create"))
                await click(page, tid("tree-/Downloads"))
            else:
                await click(page, tid("tree-/Downloads"))
            await click(page, tid(f"file-{fid}"))
            await fill_field(page, "file-name", name, replace=True)
            await click(page, tid("file-rename"))
            await click(page, tid("file-move"))
            await click(page, tid(f"moveto-{folder}"))
        return drive

    raise KeyError(task_id)
