# Weights, Context, or Skill File?

Personalizing computer-use agents from a handful of user demonstrations. See [PLAN.md](PLAN.md) for
the research design and [reports/](reports/) for the literature review.

## Setup
```bash
uv sync
uv run playwright install chromium
```

## Studio: create a skill from your own demos
```bash
uv run python -m studio --persona alice
```
A browser window opens on the workspace with a small overlay:
- **Create skill** → name it → a demo bar (x/5) and **Record**.
- Type what you will do, then **Record** and do it in the window. **Stop** ends the take,
  **Submit** saves it, **Redo** throws it away and starts again from a clean workspace.
- At 5/5 demos the skill's adapter trains automatically (C3, this user only).

Demos are saved to `data/studio/<user>/<skill>_<n>/` and adapters to
`adapters/studio/C3/<user>/<skill>/seed0`. The overlay never appears in the recorded screenshots.

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

## Model pipeline (all local, Apple silicon via MLX)
```bash
# one-time: weights → MLX bf16 (16 GB)
uv run hf download ByteDance-Seed/UI-TARS-1.5-7B --local-dir models/UI-TARS-1.5-7B-hf
uv run mlx_vlm.convert --hf-path models/UI-TARS-1.5-7B-hf --mlx-path models/UI-TARS-1.5-7B-mlx-bf16 --dtype bfloat16

# adapters (C3 per user+skill, C5 same skill other users, C6 pooled, N1 per user)
uv run python -m train.train_adapters --condition C3 --demos-root data/raw --epochs 4 --lr 1e-4

# C2 skill files from the same demos
uv run python -m agent.skillfile --demos-root data/raw

# offline: teacher-forced next-action accuracy on test demos
uv run python -m eval.offline --demos data/raw/alice --split test --skill calendar \
    [--adapter adapters/raw/C3/alice/calendar/seed0 | --context-from data/raw | --skills-dir skills/raw] \
    --out results/offline/raw/C3/alice_calendar.jsonl

# online: the agent operates the apps live; the checker scores the final state
uv run python -m agent.run --persona alice --tasks calendar_5 calendar_6 calendar_7 \
    [--adapter ...] [--context-file ...] --out results/online/raw/C3
```
`data/oracle/` holds scripted demos (`python -m scripts.make_oracle_demos`) for pipeline
development only. Paper results use human demos in `data/raw/`. Check any recording set with
`python -m scripts.check_demos data/raw`.

## Layout
```
apps/        mock workspace: personas, seeds, tasks, checkers, FastAPI server, templates
recorder/    capture.js (in-page listeners), actions.py (events → actions), record.py (CLI), oracle.py
agent/       uitars.py (prompt + action format), thoughts.py, context.py (C1), skillfile.py (C2), run.py (online)
train/       build_sft.py (demos → SFT rows), train_adapters.py (C3/C5/C6/N1 via mlx-vlm LoRA)
eval/        scoring.py (element-level step scoring), offline.py (teacher-forced eval)
scripts/     validate_checkers.py, make_oracle_demos.py, check_demos.py, bench_mlx.py
personas/    preference cards for demonstrators
```

## Sanity check
```bash
uv run python -m scripts.validate_checkers   # expects "0 failure(s)"
```
