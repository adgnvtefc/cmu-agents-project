"""Record scripted oracle demos for every (persona, task) through the real recorder.

Pipeline-development data only (clean, consistent demos); paper results use human demos.
  uv run python -m scripts.make_oracle_demos [--out data/oracle] [--personas alice bob carol]
"""

import argparse
import asyncio
import contextlib
import io
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

from apps.personas import PERSONAS
from apps.tasks import task_ids
from recorder import oracle
from recorder.record import ROOT, record_one


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "oracle"))
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS))
    ap.add_argument("--base", default="http://127.0.0.1:8917")
    args = ap.parse_args()
    out = Path(args.out)
    oracle.PAUSE = 0.35
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for persona in args.personas:
            for tid in task_ids():
                if list((out / persona / tid).glob("*/meta.json")):
                    continue
                before = httpx.post(f"{args.base}/__reset", json={"persona": persona, "task_id": tid})
                before.raise_for_status()
                state = httpx.get(f"{args.base}/__before").json()
                drv = oracle.make_driver(persona, tid, state)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ok = await record_one(browser, args.base, persona, tid, driver=drv, out_root=out, source="oracle")
                stale = buf.getvalue().count("\n  ! ")
                print(f"{'ok ' if ok else 'BAD'} {persona:5s} {tid:17s} stale_steps={stale}", flush=True)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
