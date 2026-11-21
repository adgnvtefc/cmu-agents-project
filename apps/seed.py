"""Base world state for each persona.

The base world is the SAME for every persona except for identity (name, email, manager).
Distractor folders/rooms for every persona's preferences exist for everyone, so the environment
never reveals which persona you are. Sent mail starts empty for the same reason.
"""

import copy

from .personas import CONTACTS, PERSONAS, TODAY, NOW_TIME, email

MAIL_FOLDERS = ["Inbox", "Sent", "Archive", "Follow-up", "Finance", "Trash"]

# (from_handle_or_addr, from_name, subject, body, date "YYYY-MM-DD HH:MM", folder)
_BASE_MAIL = [
    ("nina.petrova", "Nina Petrova", "Open enrollment closes Friday",
     "Hi all,\n\nReminder that benefits open enrollment closes this Friday, Nov 20. "
     "Please review your selections in the HR portal.\n\nNina", "2026-11-16 08:12", "Inbox"),
    ("theo.brooks", "Theo Brooks", "Deploy freeze next week",
     "Heads up: we're freezing deploys Nov 25–27 for the holiday. Ship anything urgent before then.\n\nTheo",
     "2026-11-16 07:45", "Inbox"),
    ("newsletter@productweekly.test", "Product Weekly", "This week in product: pricing pages that convert",
     "Five pricing page patterns, and why annual toggles matter more than you think.", "2026-11-15 18:00", "Inbox"),
    ("mei.tanaka", "Mei Tanaka", "Design review deck",
     "I've put the draft deck in the shared drive under Projects/Atlas. Comments welcome before Thursday.\n\nMei",
     "2026-11-15 14:22", "Inbox"),
    ("omar.haddad", "Omar Haddad", "Office closed Thursday 11/26",
     "The office will be closed for Thanksgiving on Thursday 11/26 and Friday 11/27.\n\nOmar",
     "2026-11-14 16:40", "Inbox"),
    ("ivan.novak", "Ivan Novak", "Q4 pipeline snapshot",
     "Pipeline is at $2.1M weighted. Two deals slipped to Q1. Details at Monday's all-hands.\n\nIvan",
     "2026-11-14 11:05", "Inbox"),
    ("no-reply@calendar.northwind.test", "Calendar", "Updated: Weekly staff meeting",
     "The weekly staff meeting has moved to Monday 16:30.", "2026-11-13 17:30", "Inbox"),
    ("grace.kim", "Grace Kim", "NDA template updated",
     "Legal has a new mutual NDA template. Please use v4 going forward; older versions are deprecated.\n\nGrace",
     "2026-11-13 10:15", "Inbox"),
    ("security@northwind.test", "IT Security", "Phishing test results",
     "Thanks to everyone who reported last week's simulated phishing email. Report rate: 81%.",
     "2026-11-12 15:00", "Inbox"),
    ("leo.martins", "Leo Martins", "Expense policy reminder",
     "Receipts over $25 must be attached to expense reports. Thanks!\n\nLeo", "2026-11-12 09:48", "Inbox"),
    ("hana.sato", "Hana Sato", "Hello from the new hire!",
     "Hi! I just joined the team this week. Looking forward to working with you.\n\nHana",
     "2026-11-11 13:20", "Inbox"),
    ("events@devmeetups.test", "Dev Meetups", "Tonight: Rust in production",
     "Join us at 6pm for talks on Rust in production systems.", "2026-11-11 08:00", "Inbox"),
    ("priya.nair", "Priya Nair", "Hackathon teams",
     "Hackathon team sign-ups are open until Wednesday. Ping me if you want to pitch an idea.\n\nPriya",
     "2026-11-10 12:34", "Inbox"),
    ("sam.lee", "Sam Lee", "Vendor review process",
     "Starting next quarter, all vendor contracts over $10k go through the new review process.\n\nSam",
     "2026-11-10 09:02", "Inbox"),
    ("dana.ortiz", "Dana Ortiz", "Roadmap offsite agenda",
     "Draft agenda for the roadmap offsite is attached. Please add topics by end of week.\n\nDana",
     "2026-11-09 17:11", "Inbox"),
    ("newsletter@cloudnews.test", "Cloud News", "The state of serverless in 2026",
     "Our annual survey of 4,000 engineers is out.", "2026-11-09 07:00", "Inbox"),
    ("raj.patel", "Raj Patel", "Vendor shortlist",
     "We're down to three vendors for office supplies: Acme Supplies, Staples, and Office Depot.\n\nRaj",
     "2026-11-06 15:45", "Inbox"),
    ("no-reply@hr.northwind.test", "HR Portal", "Your PTO request was approved",
     "Your PTO request for Dec 24–Jan 1 has been approved.", "2026-11-05 10:00", "Inbox"),
    # Pre-filed mail, identical for everyone (does not reveal personal preference)
    ("billing@cloudhost.test", "CloudHost Billing", "Your September invoice",
     "Invoice CH-0921 for $212.00 is available.", "2026-10-02 09:00", "Finance"),
    ("theo.brooks", "Theo Brooks", "Retro notes",
     "Notes from the sprint retro are in the wiki.", "2026-10-20 16:00", "Archive"),
    ("mei.tanaka", "Mei Tanaka", "Can you look at the mockups?",
     "When you get a chance, could you look at the onboarding mockups?", "2026-10-27 11:00", "Follow-up"),
]

# (title, date, start, minutes) -- identical busy blocks for all personas; they avoid every
# persona's preferred morning/afternoon slots so no preference is ever blocked.
_BASE_EVENTS = [
    ("Team standup", d, "08:30", 15) for d in
    ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20",
     "2026-11-23", "2026-11-24", "2026-11-25"]
] + [
    ("Lunch & learn", "2026-11-17", "12:00", 60),
    ("1:1 with manager", "2026-11-18", "16:30", 30),
    ("Weekly staff meeting", "2026-11-16", "16:30", 60),
    ("Weekly staff meeting", "2026-11-23", "16:30", 60),
    ("Focus time", "2026-11-19", "12:00", 60),
    ("All-hands", "2026-11-23", "12:00", 60),
    ("Interview panel", "2026-11-20", "16:00", 60),
]

DRIVE_FOLDERS = [
    "/", "/Downloads", "/Finance", "/Finance/2026", "/Finance/2026/Receipts", "/Receipts",
    "/Expenses", "/Expenses/October", "/Expenses/November", "/Projects", "/Projects/Atlas",
    "/Team", "/Personal",
]

# (name, folder, content, modified) -- folder contents follow the convention of whoever would
# use that folder, and EVERY persona sees all of them, so none reveals who you are.
_BASE_FILES = [
    ("scan_0399.pdf", "/Downloads", "Receipt — Panera Bread — 2026-11-02 — $14.20", "2026-11-02"),
    ("Atlas_roadmap_v3.pdf", "/Downloads", "Atlas roadmap, Q1–Q2 2027 (draft v3)", "2026-11-10"),
    ("2026-10-14 Chipotle.pdf", "/Finance/2026/Receipts", "Receipt — Chipotle — 2026-10-14 — $12.85", "2026-10-14"),
    ("2026-11-04 Caltrain.pdf", "/Finance/2026/Receipts", "Receipt — Caltrain — 2026-11-04 — $9.50", "2026-11-04"),
    ("sweetgreen-2026-10-21.pdf", "/Receipts", "Receipt — Sweetgreen — 2026-10-21 — $16.10", "2026-10-21"),
    ("bart-2026-11-05.pdf", "/Receipts", "Receipt — BART — 2026-11-05 — $6.80", "2026-11-05"),
    ("Starbucks - Oct 8.pdf", "/Expenses/October", "Receipt — Starbucks — 2026-10-08 — $7.45", "2026-10-08"),
    ("Marriott - Nov 6.pdf", "/Expenses/November", "Receipt — Marriott — 2026-11-06 — $310.00", "2026-11-06"),
    ("Q4 budget.xlsx", "/Finance", "Q4 budget workbook", "2026-11-01"),
    ("Atlas design review.pdf", "/Projects/Atlas", "Design review deck (draft)", "2026-11-15"),
    ("Atlas PRD.docx", "/Projects/Atlas", "Product requirements for Atlas", "2026-10-30"),
    ("Team wiki links.txt", "/Team", "wiki.northwind.test/team", "2026-09-12"),
    ("Onboarding checklist.docx", "/Team", "New hire onboarding checklist", "2026-11-11"),
    ("Vacation ideas.txt", "/Personal", "Lisbon? Kyoto?", "2026-08-01"),
]


def _addr(handle: str) -> str:
    return handle if "@" in handle else email(handle)


def base_state(pid: str) -> dict:
    p = PERSONAS[pid]
    me = {"name": p["name"], "email": p["email"], "role": p["role"], "manager": p["manager"]}
    contacts = [
        {"name": n, "email": email(h), "role": r} for n, h, r in CONTACTS if email(h) != p["email"]
    ]
    messages = []
    for i, (frm, frm_name, subj, body, date, folder) in enumerate(_BASE_MAIL):
        messages.append({
            "id": f"m{i + 1:03d}", "folder": folder, "from": _addr(frm), "from_name": frm_name,
            "to": [p["email"]], "cc": [], "bcc": [], "subject": subj, "body": body, "date": date,
            "read": folder != "Inbox" or i > 8, "starred": False,
        })
    events = []
    for i, (title, d, start, mins) in enumerate(_BASE_EVENTS):
        events.append({
            "id": f"e{i + 1:03d}", "title": title, "date": d, "start": start, "duration": mins,
            "attendees": [], "location": "", "video": False, "reminder": 0, "description": "",
        })
    files = []
    for i, (name, folder, content, modified) in enumerate(_BASE_FILES):
        files.append({
            "id": f"f{i + 1:03d}", "name": name, "folder": folder, "content": content,
            "modified": modified, "starred": False, "shared_with": [],
        })
    return copy.deepcopy({
        "persona": pid,
        "task_id": None,
        "today": TODAY,
        "now": NOW_TIME,
        "me": me,
        "contacts": contacts,
        "mail": {"folders": list(MAIL_FOLDERS), "messages": messages},
        "calendar": {"events": events},
        "drive": {"folders": list(DRIVE_FOLDERS), "files": files},
        "counters": {"m": len(messages), "e": len(events), "f": len(files)},
    })
