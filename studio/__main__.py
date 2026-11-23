"""Studio: a small overlay on the workspace for creating a skill from your own demos.

  uv run python -m studio --persona alice

Opens the workspace (Mail · Calendar · Drive) in a browser window with an overlay:
  Create skill → name it → demo bar (x/5) + Record
  Record       → Stop · Submit · Redo
  5/5 demos    → the skill's adapter trains automatically (C3, this user only)

Demos:    data/studio/<persona>/<skill>_<n>/<stamp>/  (same format as the recorder)
Adapters: adapters/studio/C3/<persona>/<skill>/seed0
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.async_api import async_playwright

from recorder.actions import ActionBuilder, Frame
from recorder.record import CAPTURE_JS, VIEWPORT

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data" / "studio"
ADAPTERS = ROOT / "adapters" / "studio" / "C3"
GOAL = 5
EPOCHS = int(os.environ.get("STUDIO_EPOCHS", "4"))
OVERLAY_JS = (Path(__file__).parent / "overlay.js").read_text()
HIDE_OVERLAY = "#__studio { display: none !important; }"  # never in the training screenshots


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "skill"


class Studio:
    def __init__(self, persona: str, base: str):
        self.persona, self.base = persona, base
        self.skill = None            # display name
        self.phase = "none"          # none | idle | recording | stopped | training | ready
        self.instruction = ""
        self.error = None
        self.page = None
        self.frames: list[Frame] = []
        self.builder = None
        self.capture_task = None
        self.t_start = 0.0
        self.train_proc = None
        self.lock = asyncio.Lock()  # overlay clicks are handled one at a time (Stop then Submit)

    # ------------------------------------------------------------------ state
    def demos(self) -> int:
        if not self.skill:
            return 0
        n = 0
        for m in (DEMOS / self.persona).glob(f"{slug(self.skill)}_*/*/meta.json"):
            if json.loads(m.read_text()).get("accepted"):
                n += 1
        return n

    def train_progress(self) -> dict:
        log = ADAPTERS / self.persona / slug(self.skill) / "seed0" / "train.log"
        if not log.exists():
            return {}
        text = log.read_text(errors="ignore").replace("\r", "\n")
        m = re.findall(r"(\d+)/(\d+) \[", text)
        return {"step": int(m[-1][0]), "total": int(m[-1][1])} if m else {}

    def state(self) -> dict:
        if self.phase == "training" and self.train_proc and self.train_proc.poll() is not None:
            ok = (ADAPTERS / self.persona / slug(self.skill) / "seed0" / "adapters.safetensors").exists()
            self.phase = "ready" if ok else "idle"
            self.error = None if ok else "Training failed (see the train.log next to the adapter)."
        return {"phase": self.phase, "skill": self.skill, "demos": self.demos(), "goal": GOAL,
                "instruction": self.instruction, "steps": len(self.builder.actions) if self.builder else 0,
                "train": self.train_progress() if self.phase == "training" else {}, "error": self.error}

    # ------------------------------------------------------------------ recording
    async def _capture(self):
        while True:
            t0 = time.time()
            try:
                url0 = self.page.url
                elements = await self.page.evaluate("window.__elementMap ? window.__elementMap() : null")
                png = await self.page.screenshot(type="png", timeout=3000, style=HIDE_OVERLAY)
                if self.page.url == url0:
                    self.frames.append(Frame(t0, time.time(), png, elements))
                    del self.frames[:-60]
            except Exception:
                pass
            await asyncio.sleep(0.1)

    async def _start(self):
        # Every demo starts from the same clean workspace, on the app you are in.
        httpx.post(f"{self.base}/__reset", json={"persona": self.persona}).raise_for_status()
        app = "/" + (urlparse(self.page.url).path.strip("/").split("/")[0] or "mail")
        await self.page.goto(self.base + app)
        await self.page.wait_for_load_state("load")
        self.frames = []
        self.builder = ActionBuilder()
        self.t_start = time.time()
        self.capture_task = asyncio.create_task(self._capture())
        await asyncio.sleep(0.4)
        self.phase = "recording"

    async def _stop(self):
        if self.phase != "recording":
            return
        await asyncio.sleep(0.6)  # let the last action settle
        self.builder.finish(time.time(), self.frames, self.page.url)
        self.capture_task.cancel()
        self.phase = "stopped"

    def on_event(self, source, payload):
        if self.phase == "recording":  # fed live: the frame buffer only keeps the last few seconds
            self.builder.feed(payload.get("ts", time.time() * 1000) / 1000.0, payload, self.frames)

    def _save(self):
        n = self.demos()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_id = f"{slug(self.skill)}_{n}"
        out = DEMOS / self.persona / task_id / stamp
        (out / "frames").mkdir(parents=True)
        with open(out / "steps.jsonl", "w") as fh:
            for i, a in enumerate(self.builder.actions):
                fname = f"frames/{i:03d}.png"
                if a["frame"] is not None:
                    (out / fname).write_bytes(a["frame"].png)
                fh.write(json.dumps({"step": i, "obs": fname if a["frame"] else None, "action": a["action"],
                                     "target": a["target"], "url": a["url"], "t": round(a["t"] - self.t_start, 3),
                                     "stale_obs": a["stale"],
                                     "elements": a["frame"].elements if a["frame"] else None}) + "\n")
        meta = {"persona": self.persona, "task_id": task_id, "skill": slug(self.skill), "skill_name": self.skill,
                "split": "train", "instruction": self.instruction, "viewport": VIEWPORT,
                "n_steps": len(self.builder.actions), "accepted": True, "recorded_at": stamp, "source": "studio"}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))

    def _train(self):
        out = ADAPTERS / self.persona / slug(self.skill) / "seed0"
        if (out / "adapters.safetensors").exists():  # keep the old one
            out.rename(out.with_name(f"seed0.prev-{datetime.now():%Y%m%d-%H%M%S}"))
        cmd = [sys.executable, "-m", "train.train_adapters", "--condition", "C3", "--demos-root", str(DEMOS),
               "--personas", self.persona, "--skills", slug(self.skill), "--epochs", str(EPOCHS)]
        log = DEMOS / self.persona / f"train_{slug(self.skill)}.log"
        self.train_proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "PROMPT_DATE": "1"},
                                           stdout=open(log, "w"), stderr=subprocess.STDOUT)
        self.phase = "training"

    # ------------------------------------------------------------------ overlay calls
    async def handle(self, source, msg: dict):
        async with self.lock:
            return await self._handle(msg)

    async def _handle(self, msg: dict):
        op = msg.get("op")
        if op == "state":
            return self.state()
        self.error = None
        try:
            if op == "create":
                name = (msg.get("name") or "").strip()
                if not name:
                    self.error = "Give the skill a name."
                else:
                    self.skill, self.phase = name, "idle"
                    if self.demos() >= GOAL:
                        self.phase = "ready" if (ADAPTERS / self.persona / slug(name) / "seed0" / "adapters.safetensors").exists() else "idle"
            elif op == "record" and self.phase == "idle":
                self.instruction = (msg.get("instruction") or "").strip()
                await self._start()
            elif op == "stop":
                await self._stop()
            elif op == "redo" and self.phase in ("recording", "stopped"):
                await self._stop()
                await self._start()
            elif op == "submit" and self.phase in ("recording", "stopped"):
                await self._stop()
                if len(self.builder.actions) <= 1:
                    self.error = "Nothing was recorded. Redo it."
                else:
                    self._save()
                    self.phase, self.instruction, self.builder = "idle", "", None
                    if self.demos() >= GOAL:
                        self._train()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        return self.state()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="alice")
    ap.add_argument("--base", default="http://127.0.0.1:8917")
    args = ap.parse_args()
    server = None
    try:
        httpx.get(args.base + "/mail", timeout=2).raise_for_status()
    except Exception:
        port = urlparse(args.base).port
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "apps.server:app", "--port", str(port),
                                   "--log-level", "warning"], cwd=ROOT)
        for _ in range(40):
            await asyncio.sleep(0.25)
            try:
                httpx.get(args.base + "/mail", timeout=1).raise_for_status()
                break
            except Exception:
                pass
    httpx.post(f"{args.base}/__reset", json={"persona": args.persona}).raise_for_status()
    studio = Studio(args.persona, args.base)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=os.environ.get("STUDIO_HEADLESS") == "1")
        ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        await ctx.route("**/__*", lambda route: route.abort())  # the page never sees the control API
        await ctx.expose_binding("__rec", studio.on_event)
        await ctx.expose_binding("__studio", studio.handle)
        await ctx.add_init_script(CAPTURE_JS)
        await ctx.add_init_script(OVERLAY_JS)
        studio.page = await ctx.new_page()
        await studio.page.goto(args.base + "/mail")
        print(f"Studio open for {args.persona}. Close the browser window to quit.")
        closed = asyncio.Event()
        studio.page.on("close", lambda _: closed.set())
        await closed.wait()
        await browser.close()
    if server:
        server.terminate()


if __name__ == "__main__":
    asyncio.run(main())
