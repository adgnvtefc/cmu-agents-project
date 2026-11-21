"""Task instances. Every instance has the SAME instruction and setup for all personas; only the
persona's preferences determine what the correct outcome is.

Per skill: instances 0-4 are TRAIN demos, 5-7 are TEST, 8-13 are DEV (method development only;
never used for the final numbers). Task id = f"{skill}_{k}".
"""

import copy
import re

from .personas import email

SKILLS = ["send_email", "file_email", "calendar", "organize_files"]
TRAIN_IDX = range(0, 5)
TEST_IDX = range(5, 8)
DEV_IDX = range(8, 14)

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
    # dev
    ("dana.ortiz", "Tell Dana Ortiz the roadmap draft is ready for her comments.", "roadmap"),
    ("priya.nair", "Let Priya Nair know our team signed up for the hackathon.", "hackathon"),
    ("sam.lee", "Email Sam Lee that the vendor review checklist is done.", "checklist"),
    ("omar.haddad", "Ask Omar Haddad to book the large conference room for Thursday.", "conference room"),
    ("leo.martins", "Let Leo Martins know the October receipts are uploaded.", "receipts"),
    ("mei.tanaka", "Send Mei Tanaka the feedback summary: users want darker contrast.", "contrast"),
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
    # dev
    ("Find the Zoom invoice for November and file it.",
     ("billing@zoom.test", "Zoom", "Zoom invoice — November 2026", "Your invoice INV-3321 for $149.90.", "2026-11-12 07:00"),
     ("billing@zoom.test", "Zoom", "Zoom invoice — October 2026", "Your invoice INV-3170 for $149.90.", "2026-10-12 07:00")),
    ("Find the hotel booking confirmation for DevSummit and file it.",
     ("reservations@hilton.test", "Hilton", "Booking confirmed: Oakland, Dec 7–9", "Confirmation #HX-55120 for DevSummit week.", "2026-11-11 19:20"),
     ("offers@hilton.test", "Hilton", "Members save 20% this winter", "Exclusive offers inside.", "2026-11-14 10:00")),
    ("Find the W-9 form Acme Supplies sent and file it.",
     ("ops@acme-supplies.test", "Acme Supplies", "W-9 for vendor onboarding", "Attached is our W-9.", "2026-11-09 13:40"),
     ("ops@acme-supplies.test", "Acme Supplies", "Holiday shipping schedule", "Last ship date Dec 18.", "2026-11-13 09:10")),
    ("Find the parking permit renewal notice and file it.",
     ("parking@northwind.test", "Facilities", "Parking permit renewal — 2027", "Renew by Dec 15.", "2026-11-10 08:00"),
     ("parking@northwind.test", "Facilities", "Garage closed Saturday", "Maintenance work all day.", "2026-11-14 08:00")),
    ("Find Finance's approval of my DevSummit reimbursement and file it.",
     (email("leo.martins"), "Leo Martins", "Approved: DevSummit registration reimbursement", "$399.00 approved.", "2026-11-13 15:00"),
     (email("leo.martins"), "Leo Martins", "Question about your September report", "Can you add the missing receipt?", "2026-11-08 15:00")),
    ("Find the Dropbox storage receipt and file it.",
     ("no-reply@dropbox.test", "Dropbox", "Your Dropbox receipt", "Plan: Business, $45.00.", "2026-11-07 06:30"),
     ("no-reply@dropbox.test", "Dropbox", "Someone shared a folder with you", "Q4 assets.", "2026-11-15 11:00")),
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
    # dev
    ("dana.ortiz", "Set up a meeting with Dana Ortiz on Tuesday afternoon about the roadmap offsite.", "2026-11-17", "afternoon", "roadmap"),
    ("priya.nair", "Set up a meeting with Priya Nair on Friday morning about hackathon judging.", "2026-11-20", "morning", "hackathon"),
    ("sam.lee", "Set up a meeting with Sam Lee next Wednesday morning to review the vendor process.", "2026-11-25", "morning", "vendor"),
    ("hana.sato", "Set up a meeting with Hana Sato on Thursday morning for her onboarding check-in.", "2026-11-19", "morning", "onboarding"),
    ("raj.patel", "Set up a meeting with Raj Patel next Monday morning about the office supplies order.", "2026-11-23", "morning", "supplies"),
    ("grace.kim", "Set up a meeting with Grace Kim on Wednesday afternoon to finalize the NDA.", "2026-11-18", "afternoon", "nda"),
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
    # dev
    ("Sweetgreen", "2026-11-11", "$15.20", "IMG_2290.pdf"),
    ("Chevron", "2026-10-19", "$58.43", "receipt(2).pdf"),
    ("United Airlines", "2026-09-14", "$388.60", "itinerary_receipt.pdf"),
    ("Best Buy", "2026-11-06", "$129.99", "scan_0421.pdf"),
    ("Marriott", "2026-10-25", "$274.10", "folio_90311.pdf"),
    ("Lyft", "2026-11-13", "$19.75", "receipt(3).pdf"),
]


def task_ids(split: str | None = None) -> list[str]:
    idx = {"train": TRAIN_IDX, "test": TEST_IDX, "dev": DEV_IDX, None: range(14)}[split]
    return [f"{s}_{k}" for s in SKILLS for k in idx]


def parse_task_id(task_id: str) -> tuple[str, int]:
    skill, k = task_id.rsplit("_", 1)
    assert skill in SKILLS, task_id
    return skill, int(k)


def get_task(task_id: str) -> dict:
    """Persona-independent task spec: instruction + setup ops + core targets."""
    skill, k = parse_task_id(task_id)
    split = "train" if k in TRAIN_IDX else ("test" if k in TEST_IDX else "dev")
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


# ---------------------------------------------------------------- synthetic pool (training only)
# System-generated instances for self-training / distillation. Never used for evaluation, and
# disjoint from train/dev/test instructions. Calendar instructions carry the explicit date (the
# system that generates a task knows its date), so distillation data is not polluted by date errors.
SYN_START = 100
_SYN_PEOPLE = ["dana.ortiz", "priya.nair", "sam.lee", "raj.patel", "mei.tanaka", "leo.martins",
               "nina.petrova", "omar.haddad", "grace.kim", "theo.brooks", "hana.sato", "ivan.novak"]
_SYN_DAYS = [("Tuesday, Nov 17", "2026-11-17"), ("Wednesday, Nov 18", "2026-11-18"),
             ("Thursday, Nov 19", "2026-11-19"), ("Friday, Nov 20", "2026-11-20"),
             ("Monday, Nov 23", "2026-11-23"), ("Tuesday, Nov 24", "2026-11-24"),
             ("Wednesday, Nov 25", "2026-11-25")]
_SYN_TOPICS = [("the launch checklist", "launch"), ("Q1 hiring", "hiring"), ("the security audit", "audit"),
               ("team offsite ideas", "offsite"), ("the pricing page", "pricing"), ("customer feedback", "feedback"),
               ("the budget forecast", "forecast"), ("the API migration", "migration"),
               ("holiday coverage", "coverage"), ("the partner contract", "contract")]
_SYN_MESSAGES = [("the launch checklist is ready", "launch checklist"), ("the security audit passed", "security audit"),
                 ("the pricing page is live", "pricing page"), ("the Q1 hiring plan is approved", "hiring plan"),
                 ("the customer survey closes Friday", "customer survey"), ("the API migration finished", "api migration"),
                 ("the holiday coverage sheet is updated", "coverage sheet"), ("the partner contract was signed", "partner contract")]


def _name(handle: str) -> str:
    return " ".join(w.capitalize() for w in handle.split("."))


def synthetic_ids(skill: str, n: int = 24) -> list[str]:
    return [f"{skill}_{SYN_START + i}" for i in range(n)]


def _synthetic(skill: str, i: int) -> dict:
    person = _SYN_PEOPLE[(i * 5) % len(_SYN_PEOPLE)]
    if skill == "calendar":
        day, date = _SYN_DAYS[(i * 3) % len(_SYN_DAYS)]
        part = "morning" if i % 2 == 0 else "afternoon"
        topic, kw = _SYN_TOPICS[(i * 7) % len(_SYN_TOPICS)]
        instr = f"Set up a meeting with {_name(person)} on {day} in the {part} about {topic}."
        return {"skill": skill, "instruction": instr, "setup": [],
                "target": {"attendee": email(person), "date": date, "part": part, "keyword": kw}}
    if skill == "send_email":
        msg, tok = _SYN_MESSAGES[(i * 3) % len(_SYN_MESSAGES)]
        verb = ["Let {n} know that {m}.", "Tell {n} that {m}.", "Email {n} to say {m}."][i % 3]
        return {"skill": skill, "instruction": verb.format(n=_name(person), m=msg), "setup": [],
                "target": {"to": email(person), "token": tok}}
    raise KeyError(skill)


_get_task_base = get_task


def get_task(task_id: str) -> dict:  # noqa: F811  (extends the definition above)
    skill, k = task_id.rsplit("_", 1)
    if int(k) >= SYN_START:
        t = _synthetic(skill, int(k) - SYN_START)
        return {"id": task_id, "split": "synthetic", **t}
    return _get_task_base(task_id)
