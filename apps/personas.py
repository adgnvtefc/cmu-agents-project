"""Persona definitions: who each user is and how they like each skill done.

Preferences deliberately CONFLICT across personas for the same skill, and every persona's
environment contains the other personas' preferred targets (folders, rooms, ...) as distractors,
so the environment alone never reveals the right answer.
"""

DOMAIN = "northwind.test"
TODAY = "2026-11-16"  # pinned "now": Monday
NOW_TIME = "09:00"


def email(handle: str) -> str:
    return f"{handle}@{DOMAIN}"


# Everyone in the fictional company (shared contact list for all personas).
CONTACTS = [
    ("Alice Chen", "alice.chen", "Product Manager"),
    ("Bob Okafor", "bob.okafor", "Software Engineer"),
    ("Carol Ruiz", "carol.ruiz", "Operations Lead"),
    ("Dana Ortiz", "dana.ortiz", "Director of Product"),
    ("Priya Nair", "priya.nair", "Engineering Manager"),
    ("Sam Lee", "sam.lee", "VP Operations"),
    ("Raj Patel", "raj.patel", "Procurement"),
    ("Mei Tanaka", "mei.tanaka", "Design Lead"),
    ("Leo Martins", "leo.martins", "Finance Analyst"),
    ("Nina Petrova", "nina.petrova", "People Ops"),
    ("Omar Haddad", "omar.haddad", "Office Manager"),
    ("Grace Kim", "grace.kim", "Legal Counsel"),
    ("Theo Brooks", "theo.brooks", "DevOps Engineer"),
    ("Hana Sato", "hana.sato", "New Hire"),
    ("Ivan Novak", "ivan.novak", "Sales Lead"),
]

PERSONAS = {
    "alice": {
        "name": "Alice Chen",
        "email": email("alice.chen"),
        "role": "Product Manager",
        "manager": email("dana.ortiz"),
        "prefs": {
            "send_email": {
                "cc": [email("dana.ortiz")],  # always CC her manager
                "bcc": [],
                "subject_prefix": "[Update] ",
                "signoff": "Best,\nAlice",
            },
            "file_email": {"folder": "Archive", "star": False},
            "calendar": {
                "afternoon": "14:00",
                "morning": "10:00",
                "duration": 30,
                "video": True,
                "location": "",
                "reminder": 10,
                "title_prefix": "",
            },
            "organize_files": {
                "folder": "/Finance/2026/Receipts",
                "name": "{date} {Vendor}.pdf",  # 2026-11-12 Blue Bottle.pdf
            },
        },
    },
    "bob": {
        "name": "Bob Okafor",
        "email": email("bob.okafor"),
        "role": "Software Engineer",
        "manager": email("priya.nair"),
        "prefs": {
            "send_email": {
                "cc": [],  # never CCs anyone
                "bcc": [],
                "subject_prefix": "",
                "subject_lowercase": True,
                "signoff": "- Bob",
            },
            "file_email": {"folder": "Follow-up", "star": False},
            "calendar": {
                "afternoon": "15:00",
                "morning": "11:00",
                "duration": 50,
                "video": False,
                "location": "Room 4B",
                "reminder": 0,
                "title_prefix": "",
            },
            "organize_files": {
                "folder": "/Receipts",
                "name": "{vendor-slug}-{date}.pdf",  # blue-bottle-2026-11-12.pdf
            },
        },
    },
    "carol": {
        "name": "Carol Ruiz",
        "email": email("carol.ruiz"),
        "role": "Operations Lead",
        "manager": email("sam.lee"),
        "prefs": {
            "send_email": {
                "cc": [],
                "bcc": [email("carol.ruiz")],  # BCCs herself for her records
                "subject_prefix": "Northwind | ",
                "signoff": "Thanks!\nCarol",
            },
            "file_email": {"folder": "Finance", "star": True},
            "calendar": {
                "afternoon": "13:30",
                "morning": "09:30",
                "duration": 45,
                "video": True,
                "location": "",
                "reminder": 30,
                "title_prefix": "Sync: ",
            },
            "organize_files": {
                "folder": "/Expenses/{Month}",  # month of the receipt, create if missing
                "name": "{Vendor} - {Mon} {d}.pdf",  # Blue Bottle - Nov 12.pdf
            },
        },
    },
}


def _example_receipt(pid: str) -> tuple[str, str]:
    from .tasks import expected_receipt
    return expected_receipt(pid, "Blue Bottle", "2026-11-12")


def persona_card_markdown(pid: str) -> str:
    """Human-readable preference card that the demonstrator reads before recording."""
    p = PERSONAS[pid]
    pr = p["prefs"]
    se, fe, cal, of = pr["send_email"], pr["file_email"], pr["calendar"], pr["organize_files"]
    lines = [
        f"# {p['name']} ({p['role']}) — {p['email']}",
        f"Manager: {p['manager']}",
        "",
        "## Sending email",
        f"- CC: {', '.join(se['cc']) or 'never CC anyone'}",
        f"- BCC: {', '.join(se['bcc']) or 'none'}",
        f"- Subject: prefix `{se['subject_prefix']}`" if se["subject_prefix"] else "- Subject: no prefix",
    ]
    if se.get("subject_lowercase"):
        lines.append("- Subject: all lowercase")
    lines += [
        "- End the body with exactly these line(s):",
        *[f"      {line}" for line in se["signoff"].split("\n")],
        "",
        "## Finding & filing an email",
        f"- Move it to folder **{fe['folder']}**" + (" and **star** it" if fe["star"] else ""),
        "",
        "## Creating calendar events",
        f"- 'Morning' means {cal['morning']}, 'afternoon' means {cal['afternoon']}",
        f"- Duration {cal['duration']} min",
        f"- Video call: {'yes' if cal['video'] else 'no'}",
        f"- Location: {cal['location'] or '(leave empty)'}",
        f"- Reminder: {str(cal['reminder']) + ' min' if cal['reminder'] else 'none'}",
        f"- Title prefix: `{cal['title_prefix']}`" if cal["title_prefix"] else "- Title: no prefix",
        "",
        "## Organizing downloaded files (receipts)",
        f"- Move to `{of['folder']}`",
        f"- Rename to `{of['name']}`",
        "- Example: a Blue Bottle receipt dated 2026-11-12 → `{}` in `{}`".format(
            *reversed(_example_receipt(pid))),
        "",
        "Use the receipt's own date (shown in the file preview), not today's date.",
        "Create the folder first if it doesn't exist.",
    ]
    return "\n".join(lines) + "\n"
