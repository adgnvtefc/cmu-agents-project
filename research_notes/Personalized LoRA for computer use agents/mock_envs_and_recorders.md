# Mock-app evaluation environments with deterministic state checkers, demo recorders, and evaluation-metric norms

Scope: evaluation environments built from mock or cloned web apps with programmatic state checks; open-source tools for recording human computer-use demonstrations; offline and online metric conventions; and statistical norms for small evaluations. Written for a 2-person, about 2-week CMU course project: FastAPI + SQLite mock mail/calendar/drive apps, Playwright 1280x800 recording, UI-TARS-1.5-7B or Qwen2.5-VL-7B agent. Research date: 2026-11-20. Items marked UNVERIFIED come from memory or secondary snippets and were not confirmed against a primary source.

---

## Q1. The most relevant environments and benchmarks, and whether any can be reused or adapted in a few days

### Takeaway
No existing benchmark gives per-user seeded mail + calendar + drive with a DB-state checker that you can drop in. The closest is **OpenApps (Meta FAIR, ICLR 2026)**. It is lightweight Python (FastHTML), its app state is set by YAML, its rewards check the full app state, it is already wired into BrowserGym, and it has already been run with UI-TARS-1.5-7B. It has Calendar, Files, Messenger, and Contacts, but no email app, and it is CC-BY-NC. **REAL (AGI Inc.)** has the Gmail and Google Calendar clones (GoMail, GoCalendar), but they are hosted on Vercel, keep state in browser localStorage, and their site source does not appear to be released. Heavy self-hosted suites (WebArena, VisualWebArena, WorkArena, OSWorld, TheAgentCompany) cannot be adapted in a few days. They are mainly worth copying for their evaluator design and reporting conventions.

### Cited Findings

**1. OpenApps (best candidate to fork, or at least copy the design from)**
- Name and org: "OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability". Authors are Ullrich, Su, Subramonian, Shi, Bar, Evtimov, Tsilivis, Balestriero, Kempe, and Ibrahim, at FAIR at Meta, with some also at NYU or Brown. Submitted to arXiv on 25 Nov 2025 as arXiv:2511.20766; accepted to ICLR 2026 (oral listing). — [arXiv HTML](https://arxiv.org/html/2511.20766v1); [ICLR 2026](https://iclr.cc/virtual/2026/oral/10008477)
- Summary: It is a lightweight open-source ecosystem of apps whose appearance and content are configurable, built to measure how agent reliability changes across app variations. The paper describes the six original apps (OpenToDo, OpenCalendar, OpenMessenger, OpenMaps, OpenCodeEditor, OpenShop) as "a fully functional Python application with editable state and appearance", built on FastHTML. — [arXiv HTML](https://arxiv.org/html/2511.20766v1)
- The current docs list more apps: Calendar, Code Editor, Files, Maps, Messenger, OpenBanking, Todo, UI Library, plus Online Shop, Contacts, Passwords, and a Start Page. **There is no email app.** Messenger handles messages by phone number or email handle, but it is not an email client. — [OpenApps docs](https://facebookresearch.github.io/OpenApps/)
- State and seeding: Appearance and data are set in YAML files that "fully represent the environment" and serve as the initial state. — [arXiv HTML](https://arxiv.org/html/2511.20766v1). Initial state can be overridden from the command line, e.g. `uv run launch.py 'apps.todo.init_todos=[["Call Mom", false]]'`. Seeded files are "copied in on every launch and reset". Dates can be pinned (`as_of_date`, `apps.files.today`). Overrides use Hydra syntax. — [OpenApps docs](https://facebookresearch.github.io/OpenApps/); [GitHub README](https://github.com/facebookresearch/OpenApps)
- Success check: The reward is a deterministic indicator that the full app state matches the target state, so a task counts as complete only if all state conditions hold. The paper says this avoids rewarding unintended side actions. Each run starts from a local copy of the full state, which is reset at initialization. — [arXiv HTML](https://arxiv.org/html/2511.20766v1). Each app exposes its state at `/<app>_all` endpoints (e.g. `/files_all`), and the agent's `goto` refuses those URLs. Some fields are compared exactly and others loosely (titles, websites). Transient UI such as focus or open popovers is excluded from scoring. — [OpenApps docs](https://facebookresearch.github.io/OpenApps/)
- Tasks: The paper uses 15 simple tasks, each with several goal phrasings, and at least two tasks per app. — [arXiv HTML](https://arxiv.org/html/2511.20766v1). The docs add goal-wording variations keyed `<task>__<style>_<n>`, with styles casual, formal, and unrelated_context and 9 variations per task, each preserving the original reward. Task YAMLs live in `config/tasks/`, but the docs do not show the schema. — [OpenApps docs](https://facebookresearch.github.io/OpenApps/)
- Agent results: More than 10,000 independent evaluations across seven agents, including UI-TARS-1.5-7B, Qwen2.5-VL, Kimi-VL-A3B, GPT-4o, and Claude 4 Sonnet, using BrowserGym for observations (screenshot, optional AXTree) and actions. Visual-only agents such as UI-TARS use a reduced action subset. — [arXiv HTML](https://arxiv.org/html/2511.20766v1). Success rates for many agents fluctuate by more than 50% across app variations; Kimi-VL-3B ranges from 63% to 4%. — [arXiv abs / search summary](https://arxiv.org/abs/2511.20766)
- Setup cost: Very low. It runs on a single CPU with under 10 MB of memory per instance and needs no emulators or containers. — [arXiv HTML](https://arxiv.org/html/2511.20766v1). Install is `uv sync`, then `uv run launch.py`, plus `playwright install chromium` for agents. — [GitHub README](https://github.com/facebookresearch/OpenApps)
- License: CC-BY-NC, copyright Meta Platforms. — [GitHub README](https://github.com/facebookresearch/OpenApps); [OpenApps docs](https://facebookresearch.github.io/OpenApps/)
- BrowserGym now lists OpenApps as an included environment. — [BrowserGym README](https://github.com/ServiceNow/BrowserGym)
- Personalization: The paper does not describe per-user configuration; researchers set content through YAML. — [arXiv HTML](https://arxiv.org/html/2511.20766v1)

**2. REAL (AGI Inc.): deterministic clones of real sites, including Gmail and Google Calendar**
- Name and org: "REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites". Divyansh Garg et al., 18 authors, with the GitHub org agi-inc. arXiv:2504.11543, April 2025. — [arXiv abs](https://arxiv.org/abs/2504.11543)
- Summary: 11 deterministic replicas of widely used sites, 112 tasks, and a harness for black-box agents. Frontier models reach at most a 41% success rate. — [arXiv abs](https://arxiv.org/abs/2504.11543)
- The clones are Omnizon (Amazon), DashDish (DoorDash), FlyUnified (United), Staynb (Airbnb), GoCalendar (Google Calendar), GoMail (Gmail), OpenDining (OpenTable), NetworkIn (LinkedIn), Udriver (Uber), TopWork (UpWork), and Zilloft (Zillow). They are built with React and Next.js. — [agisdk README](https://github.com/agi-inc/agisdk)
- Hosting and state: "all websites are publicly deployed via Vercel" (e.g. evals-staynb.vercel.app). State persists in browser localStorage. `/finish` shows the diff between the initial and current state, `/clear` resets localStorage, and `/config` sets run_id and task_id. — [arXiv HTML v2](https://arxiv.org/html/2504.11543v2)
- Success check: Action tasks use a "deterministic programmatic comparison of pre-task and post-task browser local storage states", and the reward is 1 only if all key-value assertions pass with an exact match. Information-retrieval tasks use a rubric-guided LLM judge. — [arXiv HTML v2](https://arxiv.org/html/2504.11543v2)
- Harness: "BrowserGym's interface forms the foundation for our REAL implementation". It supports Playwright, CDP, and URL-based integration. The paper says researchers can define new tasks with custom goals, and the setup allows trajectory collection for post-training. — [arXiv HTML v2](https://arxiv.org/html/2504.11543v2)
- SDK: `pip install agisdk`, Apache-2.0. Observations include a screenshot, an AXTree, the DOM, and a raw Playwright `browser` object. Actions are strings such as `click('id')`. — [agisdk README](https://github.com/agi-inc/agisdk)
- Open source: The SDK is Apache-2.0. Neither the paper nor the README says the site source code is released, and the README gives no self-hosting instructions. — [agisdk README](https://github.com/agi-inc/agisdk); [arXiv HTML v2](https://arxiv.org/html/2504.11543v2)

**3. BrowserGym / AgentLab (ServiceNow): framework, not a benchmark**
- Name: "The BrowserGym Ecosystem for Web Agent Research", arXiv:2412.05467 (Dec 2024), ServiceNow and collaborators. — [arXiv HTML](https://arxiv.org/html/2412.05467)
- Summary: A gym environment that unifies web benchmarks. Its task API is `AbstractBrowserTask`: `setup(page)` returns the goal, and `validate(page, chat_messages)` returns a scalar reward, a done flag, and an optional message after each action. — [arXiv HTML](https://arxiv.org/html/2412.05467)
- Observations: screenshot, DOM, AXTree, and `extra_element_properties` (per-element `bid`, `bbox` as left/top/width/height, a 0-1 `visibility`, and a `set_of_marks` flag). Actions: `click("bid")`, `press(...)`, coordinate `mouse_click(x, y)`, `send_msg_to_user`, or raw Playwright Python. — [arXiv HTML](https://arxiv.org/html/2412.05467)
- Included benchmarks: MiniWoB, WebArena, WebArenaVerified, VisualWebArena, WorkArena, AssistantBench, WebLINX, OpenApps, TimeWarp, and Knows. Custom benchmarks subclass `AbstractBrowserTask`. — [BrowserGym README](https://github.com/ServiceNow/BrowserGym)
- AgentLab: parallel `Study` runs with ray or joblib (up to about 20 tasks on a laptop), retries, the AgentXRay trace inspector (Gradio), and a ReproducibilityAgent. — [arXiv HTML](https://arxiv.org/html/2412.05467)
- License: The README links a LICENSE file but these pages do not name it (UNVERIFIED: believed to be Apache-2.0). — [BrowserGym README](https://github.com/ServiceNow/BrowserGym)

**4. WebArena / VisualWebArena (CMU): self-hosted Docker apps with functional evaluators**
- WebArena is arXiv:2307.13854 (2023), with 812 tasks and 30 max steps in BrowserGym. VisualWebArena is arXiv:2401.13649 (2024), with 910 tasks. Both use self-hosted Docker backends and are reset with `prepare_backend()` in BrowserGym. — [BrowserGym paper Table 1](https://arxiv.org/html/2412.05467)
- Setup cost: The recommended AWS AMI is a t3a.xlarge with a 1000 GB EBS root volume. GitLab takes at least 5 minutes to boot. A self-built map backend needs a ~180 GB download and 60-90 minutes. The documented reset happens "after evaluating the 812 examples": remove the containers and `docker run` them again from fresh images. — [WebArena environment_docker README](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)
- Evaluators: Functional-correctness checks on final state or answers (UNVERIFIED detail: string match, URL match, and program-based DB/page checks, from memory of arXiv:2307.13854).

**5. WorkArena / WorkArena++ (ServiceNow)**
- WorkArena-L1: "19,912 unique instances drawn from 33 tasks". WorkArena++ (L2): 682 compositional tasks. You need access to ServiceNow instances via a gated Hugging Face form, then `pip install browsergym-workarena`. — [WorkArena README](https://github.com/ServiceNow/WorkArena)
- Validation goes through `env.task.validate(page, chat_messages)`. BrowserGym lists L1 at 33 templates and 30 steps, and L2/L3 at 341 each and 50 steps, all on a ServiceNow backend. — [WorkArena README](https://github.com/ServiceNow/WorkArena); [BrowserGym paper](https://arxiv.org/html/2412.05467)
- arXiv IDs: WorkArena is 2403.07718 and WorkArena++ is 2407.05291 (UNVERIFIED: IDs from memory, not fetched).

**6. OSWorld / OSWorld-Verified (HKU XLANG and others)**
- arXiv:2404.07972 (Apr 2024): 369 tasks on Ubuntu, Windows, and macOS. Each has an initial-state setup config and a custom execution-based evaluation script. Humans complete 72.36% of tasks; the best model at release reached 12.24%. — [arXiv abs](https://arxiv.org/abs/2404.07972)
- Setup cost: VM providers are VMware, VirtualBox, Docker (KVM recommended), AWS, and others. "macOS hosts generally do not support KVM", so VMware Fusion is the route on Apple silicon. Apache-2.0. OSWorld-Verified was released 2025-07-28, and AWS parallelization can get a full evaluation under 1 hour. — [OSWorld GitHub](https://github.com/xlang-ai/OSWorld)

**7. TheAgentCompany (CMU, Neubig group)**
- arXiv:2412.14161: 175 tasks in a simulated company with GitLab, ownCloud (documents and office), Plane (project management), and RocketChat. Checkpoint-based evaluators are mostly deterministic Python, with an LLM judge for unstructured deliverables. Best result: Gemini 2.5 Pro at 30.3% success and 39.3% partial score. — [arXiv HTML](https://arxiv.org/html/2412.14161)
- Setup cost: Docker plus "30+ GB of free disk space". The reference machine is an EC2 t3.2xlarge, and host networking is required. MIT license. The repo claims "Quick system reset in minutes". — [TheAgentCompany GitHub](https://github.com/TheAgentCompany/TheAgentCompany)

**8. AppWorld (Stony Brook): API-only, but the closest to the planned FastAPI + DB design**
- 9 apps, 457 APIs, more than 100 DB tables, and about 100 simulated people. The APIs are FastAPI HTTP endpoints, and agents act by running Python. Evaluation uses "database-state-based unit tests", including collateral-damage checks. Loading a task (re)sets the DB and clock for that `task_id` in under 0.5 s after the first load. Apache-2.0, with an encrypted-redistribution requirement for the app and test code. There is no GUI for the apps. — [AppWorld GitHub](https://github.com/StonyBrookNLP/appworld)
- arXiv:2407.18901 (UNVERIFIED: ID from memory).

**9. Smaller or lighter suites (sanity checks and design references)**
- **MiniWoB++** (Farama): MIT, more than 100 web environments, `pip install miniwob`, Gymnasium API with Selenium, now in "maintenance mode". — [miniwob-plusplus GitHub](https://github.com/Farama-Foundation/miniwob-plusplus). Tasks include email and flight booking, and rewards are terminal, either binary or partial. — [arXiv 2305.11854 (WebGUM)](https://arxiv.org/html/2305.11854v4)
- **WebShop** (Princeton, arXiv:2207.01206): 1.18M real products and 12,087 crowd-sourced instructions. The reward r ∈ [0,1] measures similarity between the purchased product and the gold product. — [arXiv PDF](https://arxiv.org/pdf/2207.01206v1)
- **WebGames** (Convergence Labs, arXiv:2502.18356, Feb 2025): 50+ hermetic challenges with verifiable ground truth. Best AI 41.2% vs humans 95.7%. Code at convergence-ai/webgames. An alphaXiv page lists 150 challenges, possibly from a later version, so this is unresolved. — [arXiv abs](https://arxiv.org/abs/2502.18356); [alphaXiv](https://www.alphaxiv.org/benchmarks/convergence-labs-ltd/webgames)
- **Mind2Web 2** (OSU, arXiv:2506.21506, NeurIPS 2025 D&B): 130 live-web agentic-search tasks judged by Agent-as-a-Judge rubrics. It is **not** a mock-app environment and is not relevant for reuse. — [arXiv abs](https://arxiv.org/abs/2506.21506)
- **τ-bench** (arXiv:2406.12045, June 2024): A tool-agent benchmark (not GUI) that compares the end-of-episode DB state with an annotated goal state and introduces pass^k for reliability. — [arXiv abs](https://arxiv.org/abs/2406.12045)

**10. 2026 synthetic-environment work (not reusable now)**
- **AutoWebWorld** (arXiv:2602.14296, Feb 2026): Models websites as finite state machines with verifiable goal states. 29 environments, 11,663 verified trajectories at about $0.04 per trajectory. The abs page does not link any code. — [arXiv abs](https://arxiv.org/abs/2602.14296)
- **VeriEnv** (Chae, Park, Ritter, arXiv:2603.10505, Mar 2026): Uses LLMs to clone real websites into executable environments with "deterministic, programmatically verifiable" rewards. "Code and resources will be released ... upon acceptance." — [arXiv abs](https://arxiv.org/abs/2603.10505)
- **ForgingGround-Bench** (gated HF dataset): A spec and verifier framework with a worked Gmail example. It restores known state, lets the agent act, reads the final state, and asserts. Date, license, and authors are UNVERIFIED. — [HF dataset](https://huggingface.co/datasets/EchoRaven/forgingground-bench)

### Inferences
- **Reuse recommendation, ranked:**
  1. **OpenApps: fork it or mirror its pattern (high value, about 1-2 days).** Its architecture already matches the plan: Python server-rendered apps, a YAML or CLI-seeded initial state (which maps naturally onto per-user seeds), reset on launch, a `/<app>_all` state endpoint for the checker, a reward based on full state match, a BrowserGym wrapper, and prior UI-TARS-1.5-7B results. Its Calendar and Files apps cover calendar and drive. You would still add a Mail app, and could adapt Messenger. Caveats: it is CC-BY-NC (fine for a course project, but cite it and do not use it commercially), it uses FastHTML and YAML state rather than FastAPI + SQLite, and the task YAML schema is undocumented, so expect to read the source. If forking feels risky, keep your own FastAPI + SQLite apps but copy three ideas: full-state-match rewards with transient UI excluded, a hidden state endpoint the agent cannot navigate to, and goal-wording variations.
  2. **BrowserGym as the harness (medium value, about 0.5-1 day).** Wrapping each mock app as an `AbstractBrowserTask` (`setup` calls your `/reset?user=...` and returns the goal; `validate` queries SQLite) gives you parallel runs, AgentXRay trace viewing, and per-element bboxes for free. `mouse_click(x, y)` supports UI-TARS-style coordinate actions. Skip it if the UI-TARS action parser is easier to write directly against Playwright.
  3. **REAL GoMail and GoCalendar (low to medium value): optional out-of-distribution check only.** They cannot be self-hosted or seeded per user, as far as I can verify. Running a few GoMail and GoCalendar tasks through agisdk could show that personalization does not hurt general Gmail-like competence, but this needs internet access and their localStorage state model.
  4. **WebArena, VisualWebArena, WorkArena, OSWorld, TheAgentCompany: do not adopt.** Setup cost ranges from hours to days, needing a 1 TB AMI, a ServiceNow instance, VMs and KVM, or 30+ GB of Docker. None supports per-user seeded personal data. Borrow their conventions instead: max-step budgets, checkpoint partial credit, and averaging over 3 runs.
  5. **AppWorld: borrow the evaluator design, not the environment.** Its DB-state unit tests with collateral-damage checks are the right template for a SQLite checker: assert the intended rows changed and that nothing else did.
- For per-user personalized tasks, every benchmark above uses a single fixed world. The project's per-user seeding, where each user has their own contacts, folders, and calendar habits, appears to be novel relative to these suites. That is a framing point for the paper.

### Gaps
- I could not confirm whether REAL's clone source (React/Next.js) is public or self-hostable. The paper links github.com/agi-inc/REAL, but I did not inspect that repo. It is also unknown whether `/config` can inject custom seed data.
- The OpenApps task YAML schema and reward-function code were not visible in the docs or README. You need to read the repo to see how a target state is expressed.
- No 2026 (post-March) mock mail/calendar/drive suite with released code turned up. The search may have missed late-2026 releases.
- The BrowserGym software license was not confirmed. WorkArena instance logistics (free developer instance, hibernation) were not confirmed.
- A personalized web-agent benchmark (e.g. PersonalWAB) was not researched. Its ID and details are unknown here.

---

## Q2. Is there an open-source mock Gmail / calendar / drive clone with a state-inspection API that we could fork?

### Takeaway
Partially. **OpenApps** is the only verified open-source, self-hostable suite with a calendar and a files/drive-like app plus a state endpoint, but it is CC-BY-NC and has no email client. **REAL's GoMail and GoCalendar** are faithful Gmail and Google Calendar clones, but they are hosted-only on Vercel with localStorage state, and their source was not confirmed as released. Building a small FastAPI + SQLite mail app is still needed. Calendar and drive could be forked from OpenApps or written in the same style.

### Cited Findings
- OpenApps apps: Calendar, Files, Messenger, Contacts, Todo, Maps, Code Editor, OpenBanking, Shop, and Passwords. No email app. State is exposed at `/<app>_all`. CC-BY-NC. — [OpenApps docs](https://facebookresearch.github.io/OpenApps/)
- OpenApps runs anywhere Python runs, in under 10 MB per instance, with no containers. — [arXiv HTML](https://arxiv.org/html/2511.20766v1)
- REAL's GoMail and GoCalendar are React/Next.js clones served from Vercel. `/finish` shows the state diff and `/clear` resets it. The paper states no license for the site code. — [arXiv HTML v2](https://arxiv.org/html/2504.11543v2); [agisdk README](https://github.com/agi-inc/agisdk)
- TheAgentCompany includes ownCloud, a real document-storage and office app, with pre-baked data in Docker. It needs 30+ GB of disk. — [TheAgentCompany GitHub](https://github.com/TheAgentCompany/TheAgentCompany); [arXiv HTML](https://arxiv.org/html/2412.14161)
- AppWorld's apps are FastAPI APIs backed by a DB with state-based tests, but they have no GUI. — [AppWorld GitHub](https://github.com/StonyBrookNLP/appworld)
- ForgingGround-Bench has a Gmail reference verifier spec, but it is a framework and spec, not a runnable clone (UNVERIFIED details). — [HF dataset](https://huggingface.co/datasets/EchoRaven/forgingground-bench)
- MiniWoB++ includes email-inbox-style micro-tasks. These are toy single-screen widgets, not a full app (UNVERIFIED: the exact task names were not fetched). — [arXiv 2305.11854](https://arxiv.org/html/2305.11854v4)

### Inferences
- Practical plan: write the mail app yourself in FastAPI + SQLite. A minimal Gmail-like inbox with list, read, compose, label, archive, and search is roughly a day of work. Either fork OpenApps Calendar and Files and replace the YAML state with a SQLite loader, or write those apps too, copying OpenApps' state-endpoint and reward pattern. Using your own SQLite DB keeps per-user seeding and checking simple, e.g. `POST /reset?user_id=u3&seed=…` and a `GET /__state` that the harness can reach but the agent cannot.
- Mind visual realism. REAL's clones imitate real product UIs closely. Homemade server-rendered HTML will look simpler, which may make the base UI-TARS model look better or worse than it would on real Gmail. OpenApps' finding of more than 50% success fluctuation across appearance variations means visual design choices are a confound worth naming in the paper.

### Gaps
- No verified open-source, self-hostable Gmail clone with a state API was found.
- I did not check whether OpenApps Files supports folders, sharing, or document editing, which is what "drive" tasks need.

---

## Q3. Demo recorders: which capture screenshots + actions + element bounding boxes in a form convertible to VLM SFT data, and under what licenses?

### Takeaway
For a browser-only project on your own apps, the best fit is a **thin custom Playwright recorder** (Apache-2.0 Playwright, UNVERIFIED license). It hooks DOM events, records click x/y plus the target's `getBoundingClientRect()`, and screenshots the frame before each action. Turn on **Playwright tracing** alongside it as a debug and audit artifact. **AgentNetTool (OpenCUA, MIT)** is the most complete open recorder for SFT, with video, pynput events, a11y tree, an HTML browser plugin, and a keyframe-matching pipeline, but it is desktop-level and heavier than needed. **DuckTrack** and **OpenAdapt** (both MIT) capture raw input and video without element bboxes in a ready-made SFT form. **rrweb** (MIT) records the DOM, not pixels.

### Cited Findings
- **AgentNetTool (OpenCUA)** — "OpenCUA: Open Foundations for Computer-Use Agents", arXiv:2508.09123 (Aug 2025). Affiliations: XLANG Lab at HKU, Moonshot AI, Stanford, Waterloo, and CMU. — [arXiv HTML](https://arxiv.org/html/2508.09123)
  - Captures synchronized screen video, mouse and keyboard events, and accessibility trees, on Windows, macOS, and Ubuntu. — [OpenCUA GitHub](https://github.com/xlang-ai/OpenCUA)
  - Atomic events are captured with pynput and reduced to semantic actions (click, type, drag, scroll, and so on). The a11y tree is pruned, and a browser plugin captures website HTML. — [arXiv HTML](https://arxiv.org/html/2508.09123)
  - State-action matching pairs each action with the keyframe just before it. For clicks, it backtracks to before the mouse pre-movement "to avoid leaking future information". Mouse moves are kept only as preconditions, scrolls are merged, and key presses become strings or hotkeys. A reflector LLM flags incorrect or redundant steps to exclude from training. — [arXiv HTML](https://arxiv.org/html/2508.09123)
  - Dataset: 22,625 tasks with an average of 18.6 steps. Processed data are ⟨s_i, a_i⟩ pairs, with a reduced PyAutoGUI action matched to the last visually distinct frame. The README says the model, dataset, tools, and code are all under the MIT License. — [arXiv HTML](https://arxiv.org/html/2508.09123); [OpenCUA GitHub](https://github.com/xlang-ai/OpenCUA)
- **DuckTrack (TheDuckAI)** — MIT. Records all keyboard and mouse input plus the screen via OBS. Output is a folder with `events.jsonl` (timestamped events with x/y), `metadata.json`, a README, and an MP4. Supports replay. Known issues include a macOS segfault after many playbacks and some Linux OBS problems. It is early-stage (40 commits) and records no element or a11y data. — [DuckTrack GitHub](https://github.com/TheDuckAI/DuckTrack)
- **OpenAdapt (OpenAdaptAI)** — MIT. Now a restructured 1.x monorepo (`openadapt-capture`, `openadapt-flow`, `openadapt-privacy`, with `openadapt-ml` and `openadapt-evals` in separate repos); the old v0.46 is frozen under `legacy/`. It reads the DOM (web), UIA (Windows), AX (macOS), and AT-SPI (Linux), and redacts PII. The README does not document the storage format or a training-data export. — [OpenAdapt GitHub](https://github.com/OpenAdaptAI/OpenAdapt)
- **Playwright tracing / trace viewer** — `context.tracing.start({screenshots: true, snapshots: true})` and `tracing.stop({path: 'trace.zip'})` produce a zip with per-action locators, Before/Action/After DOM snapshots (the Action snapshot "highlights both, the DOM Node as well as the exact click position"), a screencast film strip, network requests, and console logs. View it with `npx playwright show-trace`. — [Playwright docs](https://playwright.dev/docs/trace-viewer)
- **Mind2Web's annotation tool** was Playwright-based. Annotators clicked a highlighted element (the click itself was blocked), then chose Click, Type, or Select Option. Released data per task include MHTML, DOM snapshots with layout and style, a screenshot, a HAR network file, and a Playwright trace. Code is MIT and data CC BY 4.0. — [Mind2Web arXiv HTML](https://arxiv.org/html/2306.06070)
- **WebLINX's tool** was a custom app, browser extension, and processing pipeline. It recorded chat turns and browser actions tied to the DOM, screenshots, and video frames, with candidate-element bboxes in the representation. 2,337 demos, 155 sites, about 43 turns per demo. The tool does not appear to be released standalone (UNVERIFIED). — [WebLINX arXiv HTML](https://arxiv.org/html/2402.05930)
- **rrweb** — MIT. Records an initial DOM snapshot plus mutations and interactions, and replays them. Its snapshots are HTML, not pixel screenshots. — [rrweb GitHub](https://github.com/rrweb-io/rrweb)
- OSWorld's manual-examination tool can record executions with screenshots and video, and agent runs save screenshots, actions, and video. — [OSWorld GitHub](https://github.com/xlang-ai/OSWorld)

### Inferences
- **Recommended recorder: custom Playwright, about half a day to build.**
  - Launch headed Chromium with `viewport=1280x800` and `device_scale_factor=1` so screenshot pixels match CSS coordinates.
  - Add a capture-phase `click`/`input`/`keydown`/`wheel` listener via `add_init_script`, and report events through `expose_binding`. Each event carries clientX/Y, the target's `getBoundingClientRect()`, tag, role, aria-label, text, and a stable `data-testid` that your own apps can emit.
  - Pre-action screenshots are the trap. By the time a DOM event fires, the page may already be changing. Copy AgentNetTool's approach: keep the last settled frame by screenshotting after each `networkidle` or DOM-quiet period, and pair each action with that frame.
  - Merge consecutive keystrokes into a single `type(content=…)`, as OpenCUA does.
  - Output JSONL with `{user_id, task_id, step, screenshot_path, action_type, x, y, bbox, text, element_meta}`. Convert to the UI-TARS or Qwen2.5-VL action strings at training time.
  - Record a Playwright trace in parallel for debugging and audit. Store `__state` DB dumps before and after each demo so recorded demos can double as checker test cases.
- Because you control the HTML, DOM-derived bboxes are exact. This is better than AgentNetTool's a11y-tree boxes or DuckTrack's lack of boxes, and it makes bbox-based offline metrics straightforward.
- Use AgentNetTool only if you later need desktop-level demos. DuckTrack and OpenAdapt add OS permissions and conversion work without giving you element boxes.

### Gaps
- I did not verify the Playwright license (believed Apache-2.0), the internal schema of `trace.zip`, or whether element bboxes can be extracted from trace snapshots without re-rendering.
- AgentNetTool's on-disk file format (video, event log, a11y JSON names) and its separate license were not found. The README's blanket MIT statement covers "tools".
- The Chrome DevTools Recorder and PC-Tracker (from the PC Agent work) were not researched.

---

## Q4. Offline metric conventions: step accuracy, element accuracy, operation F1, click-in-bbox vs distance thresholds, and how UI-TARS and OpenCUA report them

### Takeaway
The community norm for coordinate-output VLM agents is per-step **action-type accuracy + grounding accuracy (predicted point inside the gold element's bbox) + step success rate (both correct, with text arguments matched)**. Results are reported Mind2Web-style (Ele.Acc / Op.F1 / Step SR) for web, or AndroidControl/OS-Atlas-style (Type / Grounding / SR) for mobile and desktop. Click-in-bbox is the modern default. The AITW 14%-of-screen distance rule is legacy, from mobile datasets without reliable boxes. Text is matched exactly, by token F1 (Mind2Web), or by edit distance (OpenCUA).

### Cited Findings
- **Mind2Web** (arXiv:2306.06070) metrics:
  - **Element Accuracy** compares the selected element with all acceptable elements, including equivalents such as a clickable ancestor.
  - **Operation F1** is a token-level F1 over the predicted operation. It equals accuracy for Click and also checks the value for Type and Select.
  - **Step SR** counts a step as successful only if both the element and the operation are correct.
  - **Task SR** requires all steps to succeed.
  - Step metrics are macro-averaged across tasks. — [Mind2Web arXiv HTML](https://arxiv.org/html/2306.06070)
- **AndroidControl** (Google DeepMind, arXiv:2406.03679): A step is correct if the action and its arguments match, allowing equivalent actions (back = tapping Back; open_app = tapping the app-name element). For click, long-press, and type, the step is correct if the coordinates fall within the gold element's bbox. The paper reports high-level (goal only) and low-level (goal plus step instruction) settings. 15,283 demos across 833 apps. — [AndroidControl arXiv HTML](https://arxiv.org/html/2406.03679)
- **AITW legacy rule:** A tap counts as correct if it is within 14% of screen distance from the gold point, or in the same detected bbox enlarged by 1.4x. The two checks are combined with OR. — [action_matching.py copy (HF Space)](https://huggingface.co/spaces/strikerhell/SEER/blob/main/util/action_matching.py); [alphaXiv AITW](https://alphaxiv.org/benchmarks/shanghai-jiao-tong-university/androidinthewild)
- **WebLINX:** intent match (IM), then IoU of the predicted and gold element bboxes times IM for click, textinput, and submit, plus chrF (text) and URLF (load), micro-averaged per turn. — [WebLINX arXiv HTML](https://arxiv.org/html/2402.05930)
- **UI-TARS** (arXiv:2501.12326):
  - Grounding correctness is "whether the model's predicted location accurately lies within the bounding box of the target element". — [UI-TARS arXiv HTML](https://arxiv.org/html/2501.12326)
  - Multimodal Mind2Web uses the original Ele.Acc / Op.F1 / Step SR. AndroidControl and GUI Odyssey "follow the OS-Atlas settings" and report Type / Grounding / SR.
  - UI-TARS-7B scores: Mind2Web Cross-Task 73.1 / 92.2 / 67.1; AndroidControl-Low 98.0 / 89.3 / 90.8; AndroidControl-High 83.7 / 80.5 / 72.5; GUI Odyssey 94.6 / 90.1 / 87.0.
  - Offline numbers are single values with no seeds or error bars. — [UI-TARS arXiv HTML (Tables 7-8)](https://arxiv.org/html/2501.12326)
- **OpenCUA AgentNetBench:** 100 held-out tasks with multiple plausible actions annotated per step. Matching rules:
  - Coordinate actions are correct if the point falls within the target bbox.
  - `write` is scored by edit distance.
  - `press` and `hotkey` need an exact match.
  - `scroll` needs the point in the bbox plus the exact direction.
  - `terminate` must come at the right step, "neither prematurely nor delayed".
  - Reports Coord / Content / Func / Avg SR, e.g. OpenCUA-7B 75.2 Avg and OpenCUA-32B 79.1. — [OpenCUA arXiv HTML](https://arxiv.org/html/2508.09123); [OpenCUA GitHub](https://github.com/xlang-ai/OpenCUA)
- **MobileRL:** When ground truth is a point and the prediction is a box (or vice versa), it counts the step correct if the point falls inside the box, following Lian et al. (2025). — [MobileRL arXiv PDF](https://arxiv.org/pdf/2509.18119)
- **AndroidControl-Curated** argues for bbox-based matching (E_bbox) over strict point matching for grounding accuracy. — [AndroidControl_Curated GitHub](https://github.com/batechworks/AndroidControl_Curated)

### Inferences
- **Recommended offline metric set:**
  1. **Type Acc**: the action type matches.
  2. **Grounding Acc**: the predicted (x, y) is inside the gold element's DOM bbox. Optionally also report a distance-based variant, e.g. within 14% of the 1280x800 diagonal, for comparability with AITW-style numbers. Report it as secondary.
  3. **Step SR**: type, target, and arguments all correct. For text, report exact match and a normalized edit-distance or token-F1 variant, as in OpenCUA and Mind2Web.
  4. Optionally **Episode SR** under teacher forcing: all steps correct.
- Macro-average per task and then per user, Mind2Web-style, so long demos don't dominate. Also report per user, since personalization effects are per user.
- Because demos come from many users, also accept equivalent actions where possible. For example, a click on a parent element whose bbox contains the gold element, and Enter vs clicking Send. AndroidControl and Mind2Web both relax for equivalences.
- Keep the UI-TARS coordinate convention consistent. Rescale model-space coordinates back to the 1280x800 page before the bbox test (UNVERIFIED: the exact UI-TARS-1.5 and Qwen2.5-VL coordinate spaces were not checked here).

### Gaps
- I did not retrieve OS-Atlas's exact AndroidControl and GUI-Odyssey scoring code, which UI-TARS follows. I also did not confirm whether it uses bbox-only or adds a distance fallback.
- I did not confirm how AgentNetBench derives its bboxes (a11y tree vs manual).
- I did not check UI-TARS-1.5's own offline reporting, as opposed to UI-TARS v1.

---

## Q5. Online metric conventions: success rate, partial credit, steps

### Takeaway
The headline metric is binary **task success rate** from a deterministic final-state check. REAL, OpenApps, OSWorld, and τ-bench all use this. It is reported at fixed **step budgets** (OSWorld: 15/50/100; BrowserGym: 10/30/50 by benchmark) and usually **averaged over 3 runs**. Common secondary metrics are **partial credit** from checkpoints (TheAgentCompany: 0.5 x checkpoint fraction + 0.5 x full success), **steps and cost per task**, and **reliability across repeats** (τ-bench pass^k, OpenCUA pass@n). Strong checkers also penalize side effects (AppWorld collateral damage; OpenApps full-state match).

### Cited Findings
- REAL: reward 1 only if all state assertions pass with an exact match; an LLM judge is used only for retrieval tasks. — [REAL arXiv HTML v2](https://arxiv.org/html/2504.11543v2)
- OpenApps: the reward requires the full app state to match the target, which avoids rewarding side actions. Transient UI state is excluded. — [arXiv HTML](https://arxiv.org/html/2511.20766v1); [OpenApps docs](https://facebookresearch.github.io/OpenApps/)
- AppWorld: database-state unit tests plus collateral-damage checks, reported as task-level and scenario-level goal completion (TGC / SGC). — [AppWorld GitHub](https://github.com/StonyBrookNLP/appworld)
- TheAgentCompany: $S_{partial} = 0.5 \cdot \frac{\text{Result}}{\text{Total}} + 0.5 \cdot S_{full}$, from checkpoint evaluators. It also reports steps (LLM calls) and cost per task. — [TheAgentCompany arXiv HTML](https://arxiv.org/html/2412.14161)
- WebShop: a continuous reward r ∈ [0,1] based on attribute similarity to the gold product. — [WebShop arXiv PDF](https://arxiv.org/pdf/2207.01206v1)
- BrowserGym step budgets: MiniWoB 10, WebArena and VisualWebArena 30, WorkArena L1 30, L2/L3 50. — [BrowserGym paper](https://arxiv.org/html/2412.05467)
- OSWorld conventions:
  - UI-TARS reports 15- and 50-step budgets, "averaged over 3 runs", screenshot-only. UI-TARS-72B-DPO scores 22.7 at 15 steps and 24.6 at 50. — [UI-TARS arXiv HTML](https://arxiv.org/html/2501.12326)
  - OpenCUA reports 15/50/100 steps at temperature 0 and 1920x1080, as "the average of 3 runs". OpenCUA-7B scores 24.3 / 27.9 / 26.6. — [OpenCUA GitHub](https://github.com/xlang-ai/OpenCUA); [arXiv HTML](https://arxiv.org/html/2508.09123)
- Reliability across repeats:
  - τ-bench introduces pass^k; gpt-4o's pass^8 is below 25% in retail. — [τ-bench arXiv abs](https://arxiv.org/abs/2406.12045)
  - OpenCUA reports Pass@n over 16 runs at temperature 0.1. — [OpenCUA arXiv HTML](https://arxiv.org/html/2508.09123)
- Failure modes as metrics: OpenApps reports hallucinated-action and looping rates, and UI-TARS is "5x more likely to hallucinate actions depending on the app variation". — [OpenApps search summary / arXiv](https://arxiv.org/abs/2511.20766)

### Inferences
- **Recommended online metrics:**
  1. Task SR, from a full-DB-state match against the target, with the "nothing else changed" check. Report at 15 and 30 steps (or one budget about 2x the median human demo length).
  2. Partial credit, TheAgentCompany-style, if tasks have natural sub-goals (e.g. the event was created and also has the correct attendee).
  3. Mean steps on successful episodes.
  4. Optionally pass^k (k=3) to show the personalized LoRA is more consistent, not just luckier.
  5. Failure taxonomy: wrong element, wrong text, premature stop, looping, and side effects.
- Run the base model, generic SFT (if any), and the personalized LoRA on the same seeded user states and the same task instances to enable paired analysis (see Q6).

### Gaps
- OpenApps' exact loop and hallucination metric definitions were not extracted.
- I did not confirm the default step budget OpenApps used for UI-TARS-1.5-7B.

---

## Q6. Statistical norms for small agent evaluations (CIs, paired tests, multiple seeds)

### Takeaway
Agent papers usually report the mean success over **3 runs** (OSWorld, UI-TARS, OpenCUA), sometimes with a ± spread, and BrowserGym reports a **standard error σ/√N**. Offline step metrics usually come without error bars. The most-cited statistical guidance (Miller 2024, Anthropic) recommends:
- CLT standard errors with 95% CIs (±1.96·SE)
- **clustered SEs** when items share a cluster (here: per user)
- **paired-difference SEs** when comparing two models on the same items
- power analysis

Bootstrap is "unnecessary unless a complicated sampling scheme or estimator is being used". With tens of tasks, CIs will be wide (about ±15 pp at n=40), so paired designs are essential.

### Cited Findings
- **"Adding Error Bars to Evals"** (Evan Miller, Anthropic, arXiv:2411.00640, Nov 2024):
  - Report the mean with SE and n; for binary scores, SE = √(p(1−p)/n), and the 95% CI is ±1.96·SE.
  - Use clustered SEs for grouped questions; these can be up to 3x the naive SE.
  - Resample K times per question to cut variance, and do not lower temperature to reduce variance.
  - Paired SE = √(SE_A² + SE_B² − 2·SE_A·SE_B·Corr).
  - Power formula: an example with δ = 0.03, 80% power, and α = 0.05 needs n ≈ 969.
  - "We regard bootstrapping as unnecessary unless a complicated sampling scheme or estimator is being used." — [arXiv HTML](https://arxiv.org/html/2411.00640v1); [arXiv abs](https://arxiv.org/abs/2411.00640)
- **BrowserGym/AgentLab:** reports task success rate with standard error (σ/√N). Uses multiple seeds per task for MiniWoB and WorkArena L1 and one seed for AssistantBench. Recommends temperature 0, while noting it is not always sufficient for determinism. — [BrowserGym arXiv HTML](https://arxiv.org/html/2412.05467)
- **UI-TARS:** OSWorld scores are "averaged over 3 runs" to reduce network and environment noise. Offline results are single values with no error bars. — [UI-TARS arXiv HTML](https://arxiv.org/html/2501.12326)
- **OpenCUA:** OSWorld-Verified is "the average of 3 runs", with asymmetric ± values shown (e.g. 45.0 +1.1/−1.2), which the paper does not define clearly. Pass@n uses 16 runs. — [OpenCUA arXiv HTML](https://arxiv.org/html/2508.09123)
- **OpenApps:** more than 10,000 evaluations. Success varies by more than 50% across app variations, so a single app version can badly misestimate reliability. — [arXiv HTML](https://arxiv.org/html/2511.20766v1)
- **Agentic Benchmark Checklist** (Zhu et al., arXiv:2507.02825, Jul 2025; co-authors include Percy Liang, Daniel Kang, and Ion Stoica): task-setup and reward-design flaws can under- or overestimate agent performance by up to 100% in relative terms. Examples: τ-bench counted empty responses as success, and SWE-bench Verified had insufficient tests. — [arXiv abs](https://arxiv.org/abs/2507.02825)

### Inferences
- **Sample-size reality check (computed from Miller's formula):** at p ≈ 0.5, SE = √(0.25/n). That gives a 95% CI of about ±18 pp at n=30, ±15.5 pp at n=40, ±9.8 pp at n=100, and ±6.9 pp at n=200. A two-week project cannot power small effects. Detecting a 10-15 pp gain needs on the order of 100+ task instances per arm, or a paired design that exploits correlation.
- **Recommended reporting protocol:**
  1. Evaluate base and personalized-LoRA models on identical (user, task, seed-state) instances.
  2. Report the paired difference with a paired SE, or McNemar's exact test on discordant pairs, for binary success.
  3. Cluster by user, since tasks within a user share a seed DB and a LoRA. Use clustered SEs or a user-level cluster bootstrap. With few users (e.g. 3-6), say plainly that the user-level n is tiny and show per-user bars.
  4. Run 3 episodes per instance (temperature > 0, or temperature 0 with re-seeded environments) and report mean ± SE across runs, following OSWorld, UI-TARS, and OpenCUA.
  5. For offline step accuracy, n is the number of steps, which is much larger, but steps are clustered within episodes and users. Use clustered SEs or an episode-level bootstrap, not step-level i.i.d. SEs.
  6. Validate the checker before trusting numbers, following ABC. Replay every recorded human demo through the checker (it should pass) and run a no-op or random agent (it should fail). This guards against the "empty response counts as success" class of bugs.
- The "about 1,000 questions" guidance in Miller is for detecting 3 pp differences. Workshop papers with 50-200 tasks typically just report SE or CIs and avoid strong significance claims. That is acceptable if stated honestly.

### Gaps
- I found no survey quantifying how often GUI-agent papers report bootstrap CIs vs SEs. The norms above come from individual papers and Miller's guidance.
- I did not find a GUI-agent paper that explicitly uses McNemar or paired bootstrap for model comparison. The recommendation rests on general statistics plus Miller's paired-SE argument.
