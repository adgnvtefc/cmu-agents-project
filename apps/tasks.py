"""Task instances. Every instance has the SAME instruction and setup for all personas; only the
persona's preferences determine what the correct outcome is.

Per skill: instances 0-4 are TRAIN demos, 5-7 are TEST. Task id = f"{skill}_{k}".
"""

import copy
import re

from .personas import email

SKILLS = ["send_email", "file_email", "calendar", "organize_files"]
TRAIN_IDX = range(0, 5)
TEST_IDX = range(5, 8)

# ---------------------------------------------------------------- send_email
# (recipient handle, instruction, token that must appear in the body)
SEND_EMAIL = [
    ("raj.patel", "Let Raj Patel know the vendor call went well: we're going with Acme Supplies, delivery on Dec 2.", "acme"),
    ("mei.tanaka", "Email Mei Tanaka to confirm the design review has moved to Thursday.", "thursday"),
    ("leo.martins", "Send Leo Martins the Q4 budget numbers: total spend $48,200, 6% under plan.", "48,200"),
    ("nina.petrova", "Tell Nina Petrova the onboarding docs are ready for her review.", "onboarding"),
    ("grace.kim", "Give Grace Kim a heads-up that the vendor contract expires on Jan 15.", "jan 15"),
    ("theo.brooks", "Let Theo Brooks know the staging server will be down Friday from 6 to 8 pm.", "staging"),
    ("hana.sato", "Email Hana Sato the link to the team wiki: wiki.northwind.test/team", "wiki.northwind.test/team"),
    ("ivan.novak", "Ask Ivan Novak if he can present at Monday's all-hands.", "all-hands"),
]

# ---------------------------------------------------------------- file_email
# instruction, target email, distractor email. Email = (from_addr, from_name, subject, body, date)
FILE_EMAIL = [
    ("Find the November invoice from Acme Supplies and file it.",
     ("billing@acme-supplies.test", "Acme Supplies", "Invoice #A-1182 — November 2026",
      "Invoice #A-1182 for $1,240.00 is due Dec 15.", "2026-11-13 09:30"),
     ("billing@acme-supplies.test", "Acme Supplies", "Invoice #A-1104 — October 2026",
      "Invoice #A-1104 for $980.00 was paid Oct 30. Thank you!", "2026-11-02 09:30")),
    ("Find the purchase order confirmation from Globex and file it.",
     ("orders@globex.test", "Globex", "PO-5531 confirmed",
      "Your purchase order PO-5531 (40 monitor arms) is confirmed.", "2026-11-12 14:10"),
     ("news@globex.test", "Globex", "Globex newsletter: November",
      "New ergonomic accessories are here.", "2026-11-14 08:00")),
    ("Find my Lyft receipt for the ride on Nov 10 and file it.",
     ("receipts@lyft.test", "Lyft", "Your ride on Nov 10",
      "Thanks for riding! Total: $23.40. Pickup 8:12 AM.", "2026-11-10 08:40"),
     ("receipts@lyft.test", "Lyft", "Your ride on Nov 4",
      "Thanks for riding! Total: $18.90. Pickup 6:47 PM.", "2026-11-04 19:15")),
    ("Find the signed contract from Initech and file it.",
     ("legal@initech.test", "Initech Legal", "Signed: Master Services Agreement",
      "Attached is the fully executed MSA. Effective Dec 1, 2026.", "2026-11-11 17:05"),
     ("legal@initech.test", "Initech Legal", "Initech: contract draft v2",
      "Here's draft v2 with our redlines.", "2026-11-03 10:20")),
    ("Find the AWS bill for October and file it.",
     ("billing@aws.test", "AWS Billing", "AWS Billing Statement — October 2026",
      "Your October 2026 statement: $3,412.77.", "2026-11-03 06:00"),
     ("billing@aws.test", "AWS Billing", "AWS Billing Statement — September 2026",
      "Your September 2026 statement: $3,105.20.", "2026-10-03 06:00")),
    ("Find the DevSummit registration confirmation and file it.",
     ("tickets@devsummit.test", "DevSummit", "You're registered: DevSummit 2026",
      "Your ticket for DevSummit 2026 (Dec 8–9, Oakland) is confirmed.", "2026-11-12 12:00"),
     ("cfp@devsummit.test", "DevSummit", "DevSummit: call for speakers closes soon",
      "Submit your talk by Nov 20.", "2026-11-15 12:00")),
    ("Find the shipping notice from Umbrella Corp and file it.",
     ("orders@umbrella.test", "Umbrella Corp", "Your order has shipped (#UC-7781)",
      "Order #UC-7781 shipped via UPS. Arrives Nov 19.", "2026-11-14 15:30"),
     ("orders@umbrella.test", "Umbrella Corp", "Order received (#UC-7781)",
      "We've received your order #UC-7781.", "2026-11-10 11:00")),
    ("Find Finance's approval of my October expense report and file it.",
     (email("leo.martins"), "Leo Martins", "Approved: October expense report",
      "Your October expense report ($412.60) is approved and will be reimbursed Nov 25.", "2026-11-13 13:00"),
     (email("leo.martins"), "Leo Martins", "Reminder: submit November expenses",
      "Please submit November expenses by Dec 3.", "2026-11-15 09:00")),
]

# ---------------------------------------------------------------- calendar
# (attendee handle, instruction, date, part of day, title keyword)
CALENDAR = [
    ("raj.patel", "Set up a meeting with Raj Patel on Wednesday afternoon to go over the vendor contract.", "2026-11-18", "afternoon", "vendor"),
    ("mei.tanaka", "Set up a meeting with Mei Tanaka on Thursday morning about the design review.", "2026-11-19", "morning", "design"),
    ("leo.martins", "Set up a meeting with Leo Martins on Friday afternoon to discuss the Q4 budget.", "2026-11-20", "afternoon", "budget"),
    ("grace.kim", "Set up a meeting with Grace Kim on Tuesday morning about the NDA questions.", "2026-11-17", "morning", "nda"),
    ("omar.haddad", "Set up a meeting with Omar Haddad next Monday afternoon about offsite logistics.", "2026-11-23", "afternoon", "offsite"),
    ("theo.brooks", "Set up a meeting with Theo Brooks on Thursday afternoon to plan the staging migration.", "2026-11-19", "afternoon", "staging"),
    ("nina.petrova", "Set up a meeting with Nina Petrova on Wednesday morning about the onboarding plan.", "2026-11-18", "morning", "onboarding"),
    ("ivan.novak", "Set up a meeting with Ivan Novak next Tuesday afternoon to review the sales pipeline.", "2026-11-24", "afternoon", "pipeline"),
]

# ---------------------------------------------------------------- organize_files
# (vendor, receipt date, amount, downloaded filename)
RECEIPTS = [
    ("Blue Bottle", "2026-11-12", "$18.40", "scan_0412.pdf"),
    ("Lyft", "2026-11-03", "$27.15", "receipt(1).pdf"),
    ("Staples", "2026-10-28", "$64.99", "IMG_2231.pdf"),
    ("Delta Air Lines", "2026-11-09", "$412.30", "eticket_receipt.pdf"),
    ("Hilton", "2026-09-22", "$689.00", "folio_88213.pdf"),
    ("Uber Eats", "2026-11-14", "$36.75", "order_confirmation.pdf"),
    ("Amtrak", "2026-09-30", "$96.00", "amtrak_trip.pdf"),
    ("Office Depot", "2026-10-05", "$142.18", "scan_0418.pdf"),
]


def task_ids(split: str | None = None) -> list[str]:
    idx = {"train": TRAIN_IDX, "test": TEST_IDX, None: range(8)}[split]
    return [f"{s}_{k}" for s in SKILLS for k in idx]


def parse_task_id(task_id: str) -> tuple[str, int]:
    skill, k = task_id.rsplit("_", 1)
    assert skill in SKILLS, task_id
    return skill, int(k)


def get_task(task_id: str) -> dict:
    """Persona-independent task spec: instruction + setup ops + core targets."""
    skill, k = parse_task_id(task_id)
    split = "train" if k in TRAIN_IDX else "test"
    if skill == "send_email":
        to, instr, token = SEND_EMAIL[k]
        return {"id": task_id, "skill": skill, "split": split, "instruction": instr,
                "setup": [], "target": {"to": email(to), "token": token}}
    if skill == "file_email":
        instr, target, distractor = FILE_EMAIL[k]
        ops = [{"op": "add_email", "key": "target", "email": target},
               {"op": "add_email", "key": "distractor", "email": distractor}]
        return {"id": task_id, "skill": skill, "split": split, "instruction": instr,
                "setup": ops, "target": {}}
    if skill == "calendar":
        who, instr, date, part, kw = CALENDAR[k]
        return {"id": task_id, "skill": skill, "split": split, "instruction": instr, "setup": [],
                "target": {"attendee": email(who), "date": date, "part": part, "keyword": kw}}
    if skill == "organize_files":
        vendor, date, amount, fname = RECEIPTS[k]
        instr = f"File the {vendor} receipt that's in my Downloads."
        ops = [{"op": "add_file", "key": "target",
                "file": (fname, "/Downloads", f"Receipt — {vendor} — {date} — {amount}", "2026-11-16")}]
        return {"id": task_id, "skill": skill, "split": split, "instruction": instr,
                "setup": ops, "target": {"vendor": vendor, "date": date}}
    raise KeyError(task_id)


def apply_setup(state: dict, task: dict) -> dict:
    """Apply a task's setup ops to a fresh base state. Records created ids in state['task_keys']."""
    state = copy.deepcopy(state)
    keys = {}
    for op in task["setup"]:
        if op["op"] == "add_email":
            frm, frm_name, subj, body, date = op["email"]
            state["counters"]["m"] += 1
            mid = f"m{state['counters']['m']:03d}"
            state["mail"]["messages"].append({
                "id": mid, "folder": "Inbox", "from": frm, "from_name": frm_name,
                "to": [state["me"]["email"]], "cc": [], "bcc": [], "subject": subj, "body": body,
                "date": date, "read": False, "starred": False,
            })
            keys[op["key"]] = mid
        elif op["op"] == "add_file":
            name, folder, content, modified = op["file"]
            state["counters"]["f"] += 1
            fid = f"f{state['counters']['f']:03d}"
            state["drive"]["files"].append({
                "id": fid, "name": name, "folder": folder, "content": content,
                "modified": modified, "starred": False, "shared_with": [],
            })
            keys[op["key"]] = fid
        else:
            raise ValueError(op)
    state["task_id"] = task["id"]
    state["task_keys"] = keys
    return state


# ---------------------------------------------------------------- expected (persona-specific)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def expected_receipt(pid: str, vendor: str, date: str) -> tuple[str, str]:
    """(folder, filename) a persona would file this receipt under."""
    from .personas import PERSONAS
    pref = PERSONAS[pid]["prefs"]["organize_files"]
    y, m, d = date.split("-")
    month = MONTHS[int(m) - 1]
    folder = pref["folder"].replace("{Month}", month)
    name = (pref["name"].replace("{date}", date).replace("{Vendor}", vendor)
            .replace("{vendor-slug}", slug(vendor)).replace("{Mon}", month[:3])
            .replace("{d}", str(int(d))))
    return folder, name
