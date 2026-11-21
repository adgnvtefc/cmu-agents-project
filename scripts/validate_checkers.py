"""Checker validation: for every (persona, task), the oracle demo must pass (success, all prefs,
no collateral) and a no-op must fail. Also checks that every OTHER persona's preferences fail on
this persona's oracle outcome, i.e. tasks really are personal.

  uv run python -m scripts.validate_checkers [--base http://127.0.0.1:8917] [--only send_email]
"""

import argparse
import asyncio

import httpx
from playwright.async_api import async_playwright

from apps.checkers import check
from apps.personas import PERSONAS
from apps.tasks import get_task, task_ids
from recorder import oracle
from recorder.record import START_URL, VIEWPORT


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8917")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    oracle.PAUSE = 0.02
    failures = 0
    async with async_playwright() as p, httpx.AsyncClient(base_url=args.base) as api:
        browser = await p.chromium.launch()
        for tid in task_ids():
            if args.only and not tid.startswith(args.only):
                continue
            for persona in PERSONAS:
                (await api.post("/__reset", json={"persona": persona, "task_id": tid})).raise_for_status()
                before = (await api.get("/__before")).json()
                noop = (await api.get("/__check")).json()
                ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
                page = await ctx.new_page()
                await page.goto(args.base + START_URL[get_task(tid)["skill"]])
                err = ""
                try:
                    await oracle.make_driver(persona, tid, before)(page)
                except Exception as e:  # noqa: BLE001
                    err = repr(e)[:200]
                await ctx.close()
                after = (await api.get("/__state")).json()
                r = check(persona, tid, before, after)
                cross = [o for o in PERSONAS if o != persona and check(o, tid, before, after)["joint"]]
                ok = (r["joint"] and not r["collateral"] and not noop["success"] and not err and not cross)
                failures += not ok
                flag = "ok  " if ok else "FAIL"
                print(f"{flag} {persona:5s} {tid:17s} success={r['success']} prefs={r['prefs']} "
                      f"collateral={r['collateral']} noop_success={noop['success']} "
                      f"also_passes_for={cross} {err} {r['notes']}")
        await browser.close()
    print(f"\n{failures} failure(s)")


if __name__ == "__main__":
    asyncio.run(main())
