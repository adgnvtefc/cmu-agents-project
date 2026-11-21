# Weights, Context, or Skill File?

Personalizing computer-use agents from a handful of user demonstrations. See [PLAN.md](PLAN.md) for
the research design and [reports/](reports/) for the literature review.

## Setup
```bash
uv sync
uv run playwright install chromium
```

## Run the mock workspace (Mail · Calendar · Drive)
```bash
uv run uvicorn apps.server:app --port 8917
```
Open http://127.0.0.1:8917. Hidden control endpoints (`/__reset`, `/__state`, `/__check`) are for
the recorder and evaluator only; the recording browser blocks them.

## Record demonstrations
1. Read your persona card in [personas/](personas/) (alice, bob, carol). Follow it **exactly**:
   the checker scores both whether the task got done and whether your persona's preferences were
   followed.
2. Start the server (above), then in another terminal:
```bash
uv run python -m recorder.record --persona alice --split train
```
3. For each task, the instruction prints in the terminal. Press Enter, do the task in the browser
   window, then switch back to the terminal and press Enter. The checker runs, and you keep the
   demo or redo it.

Tips for clean demos:
- Pause about half a second between actions. Actions done too quickly are flagged `!` because
  their screenshot may be stale.
- Click into a field before typing, and avoid unnecessary detours.
- To replace text in a field, press ⌘A and then type.
- Re-run the same command to resume: tasks that already have an accepted demo are skipped.
  Use `--task <id> --redo` to re-record a single task.

Output goes to `data/raw/<persona>/<task_id>/<timestamp>/`. Each step in `steps.jsonl` has its
pre-action screenshot in `frames/`. Open `trace.zip` with
`uv run playwright show-trace <path>/trace.zip` to debug a recording.

## Layout
```
apps/        mock workspace: personas, seeds, tasks, checkers, FastAPI server, templates
recorder/    capture.js (in-page listeners), actions.py (events → actions), record.py (CLI), oracle.py
scripts/     validate_checkers.py: scripted oracle demos for all 96 (persona, task) pairs
personas/    preference cards for demonstrators
```

## Sanity check
```bash
uv run python -m scripts.validate_checkers   # expects "0 failure(s)"
```
