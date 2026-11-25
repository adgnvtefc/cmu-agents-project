# Status — Sat Nov 21 (autonomous block, 14:53–16:50)

Architecture (unchanged from PLAN.md): **mock apps → recorder → demos → {C1 demos-in-prompt,
C2 skill file, C3 per-(user, skill) LoRA, C5/C6 controls} → offline + online evaluation.**
Everything runs locally on the Mac Studio (M5 Ultra, 96 GB) via MLX. No cloud.

## 1. What is true now

| Layer | State | Evidence |
|---|---|---|
| Mock apps (Mail, Calendar, Drive) | ✅ done | 32 task instances × 3 personas; checkers pass all 96 scripted demos, fail no-op, never pass another persona's prefs |
| Recorder | ✅ done, 3 bugs fixed today | element map per frame; event time taken in-page; 601/608 recorded clicks self-consistent (the 7 others are in 3 demos recorded under GPU load, flagged stale; see §4) |
| Demo data | ⚠️ oracle only | `data/oracle/`: 96 scripted demos (dev data). **Human demos: 0 recorded yet** |
| Model runtime | ✅ local | UI-TARS-1.5-7B bf16 in MLX: 2–3 s per agent step, 18 GB |
| Training (C3/C5/C6/N1) | ✅ pipeline works, recipe being tuned | `train/train_adapters.py`, divergence guard; ~25 min per adapter (70 samples × 4 epochs); see §3 |
| Full-study batch | ✅ written, not run | `scripts/overnight.sh` (skill files → C3/C5/C6 → offline eval of all arms) |
| C1 demos in prompt | ✅ | `agent/context.py` |
| C2 skill file | ⚠️ provisional writer | `agent/skillfile.py` uses local Qwen2.5-VL-7B; PLAN says frontier model (needs API key) |
| Offline eval | ✅ | element-level click scoring + preference-critical split (`eval/summarize.py`) |
| Online eval (live agent) | ✅ | `agent/run.py`; zero-shot gate passed; loop detector |
| Router (R) | ✅ simple version | `agent/router.py`: 33/36 held-out instructions to the right skill; the 3 misses fall back to the base model (0 wrong-adapter routes) |

## 2. Results so far (oracle demos, Alice × calendar, 3 test demos = 42 steps)

**Offline** (teacher-forced next action; click correct if it would hit the same element):

| Arm | What the model gets | All steps (n=42) | Pref-critical (n=9) | Other steps (n=33) |
|---|---|---|---|---|
| C0 | nothing (zero-shot) | 31% | 3/9 | 30% |
| C1 | Alice's 5 demos as text in the prompt | **81%** | **9/9** | **76%** |
| C2 | skill file written from the 5 demos | 69% | 8/9 | 64% |
| C3 | per-(Alice, calendar) LoRA, no prompt context | 62% | 6/9 | 61% |
| C4 | LoRA + skill file | 71% | 6/9 | 73% |
| C5 | calendar LoRA trained on **Bob's + Carol's** demos (control) | 45% | **0/9** | 58% |

95% CIs are about ±14 points on all-steps and ±30 on pref-critical, so only C0 vs the rest is
clearly separated. Read these as a pipeline dry run, not a result.

What the errors show:
- **C5 is the key control, and it behaves as the design needs:** an adapter trained on other
  users' calendar demos gets 0/9 of Alice's preference steps. It applies *their* habits instead:
  it types Bob's 15:00 / 11:00 start times where Alice uses 14:00 / 10:00. Alice's own adapter
  (C3) gets 6/9. So the per-user adapter carries *that user's* choices, not just "how to use the
  calendar app".
- **C3 learned Alice's values but not all of her steps.** It types her 14:00 / 10:00 start
  times and picks her 10-minute reminder, but in all 3 test demos it skips clicking "30 min"
  (it goes straight to Guests, leaving the 60-minute default). It also sometimes repeats the
  previous step's action, the same "where am I" failure seen in the one-demo overfit test.
- **C2's one critical miss is the writer's fault:** the local model wrote "start time is always
  14:00", so the agent booked a morning meeting at 14:00. A stronger writer should fix this.
- **Some offline 'errors' are harmless:** C3 types "Theo Brooks" where the demo typed the email
  address; the app accepts names. The online checker (below) is the fairer judge.
- **Oracle demos flatter C1:** scripted demos repeat an identical procedure, so copying them from
  the prompt is easy. Human demos will vary more.

**Online** (agent operates the app live; checker scores the final state; same 3 test tasks):

| Arm | Task success | All prefs + success | Pref adherence | Avg steps | Stuck in a loop | Prompt tokens / step | s / step |
|---|---|---|---|---|---|---|---|
| C0 zero-shot | 0/3 | 0/3 | 0% | 19.3 | 2/3 | 4,999 | 3.1 |
| C1 demos in prompt | 0/3 | 0/3 | **87%** | 14.3 | 0/3 | 6,814 | 3.3 |
| C2 skill file | 0/3 | 0/3 | 0% | 18.7 | 3/3 | 5,295 | 3.2 |
| C3 per-user LoRA | **2/3** | 0/3 | 40% | **11.0** | 0/3 | **4,114** | **2.4** |
| C4 LoRA + skill file | 0/3 | 0/3 | **93%** | 15.3 | 0/3 | 4,642 | 2.6 |

Why each arm fails (read from the episode logs):
- **C0 / C2** never save the event: they click ~26 px below the "Save" button and repeat until the
  loop detector stops them. The skill file does not fix this; it is a grounding problem.
- **C1 / C4** do Alice's whole procedure (14:00/10:00, 30 min, video, 10-min reminder) but type the
  **wrong date** for relative days ("Thursday" → 2026-11-21 instead of 11-19). Date arithmetic,
  not personalization, is what fails them.
- **C3** is the only arm that reliably completes the task (it learned exactly where "Save" is
  and the date pattern) but skips Alice's 30-min / video / reminder steps.

**The picture this suggests (n = 3, oracle demos, so a hypothesis, not a finding):** the adapter
carries *how to operate the UI* (grounding, step order, finishing), while prompt context carries
*this user's choices*. That matches the split the literature predicts and is exactly what the
full C1–C5 comparison on human demos is designed to test.

Preference-critical step = a step where Alice's correct action differs from every other persona's
on the same task (e.g. typing "14:00", clicking "30 min", "10 min before"). 9 of 42 steps.

Zero-shot live gate (base model, Alice, one test task per skill): 2/4 tasks completed
(send email, file email); 1/4 also matched all preferences (file email, by coincidence:
"file it" → Archive is Alice's habit). Failures: calendar — repeatedly clicks 26 px below
"Save"; receipts — never opens files to find the right one.

## 3. Training recipe findings

1. **Average loss hides the decision.** In UI-TARS output, almost every target token is
   boilerplate; the action is decided by 1–2 tokens (e.g. first coordinate digit, first word typed).
   An adapter at loss 0.16 still picked the wrong action on 10/15 training steps; the decision
   tokens were at 2–2.5 nats. → Judge adapters by held-out step accuracy, not loss.
2. **Stability.** lr 2e-4 → NaN; lr 1e-4 + mlx-vlm grad-clip → NaN by iter 25; lr 1e-4 (α=32)
   diverged after ~40 iters on 70 samples. Current recipe: **lr 5e-5, rank 16, α 16, warmup 20,
   4 epochs, no grad-clip**, with an automatic divergence guard.
3. Step thoughts are now templated from the recorded element (e.g. *Next, I'll click the Start
   time field.*) instead of empty, so the model keeps its native Thought→Action style.

## 4. Bugs found and fixed today
- Element map could come from a later page (captured after the screenshot) → captured first, URL checked.
- Click events were timestamped on arrival in Python; under GPU load they arrived late, so
  post-click frames were used as "pre-action" frames → events now carry page-side timestamps.
- `mlx_vlm.train --epochs` crashes without a config file → we pass `--iters` instead.

## 5. Not done / decisions for you
**Not done**
- **No human demos yet.** All numbers above come from scripted oracle demos of 1 persona × 1
  skill. The paper needs the 96 human recordings (recorder is ready; see README).
- C5 (same skill, other users), C6 (pooled) and N1 are implemented in `train/train_adapters.py`
  but not trained yet. Each run takes ~25–50 min on this Mac, so they belong in an overnight batch.
- Router is lexical (TF-IDF over the user's demo instructions); paraphrases with no shared
  words fall back to the base model. An embedding router is an easy upgrade if needed.
- 3 oracle demos still have flagged stale frames (recorded under GPU load); re-record when idle.
- The proposal's "record → train → use" end-user flow has no single UI yet (recorder CLI +
  training CLI + agent CLI exist separately).

**Decisions for you**
1. **When do the 96 human demos get recorded, and who plays which persona?** Everything
   downstream waits on this. Two people can cover 3 personas (one person plays two).
2. **C2 skill writer:** add an Anthropic API key so a frontier model writes the skill files (as
   PLAN.md says), or accept the local 7B writer and report it as a limitation.
3. **LoRA recipe before the full batch:** current recipe (lr 5e-5, 4 epochs) is stable but trails
   in-context demos. Proposed: one overnight sweep on 2 dev pairs (epochs 4 vs 8, lr 5e-5 vs
   1e-4 with warmup), picked by held-out step accuracy, then freeze it for all adapters.
4. **Calendar dates:** relative days ("next Tuesday") make every arm fail on date arithmetic,
   which hides the personalization signal. Recommended: keep the wording but add the date
   ("Thursday, Nov 19") to calendar instructions, so success measures the skill, not arithmetic.
   This changes 8 task instances; human recording should start after this is decided.

## 6. C3-improvement study (Sat Nov 21, 17:48–23:40) — dev split, oracle demos, seed 0
Five ways to improve the per-(user, skill) LoRA, each using only what a deployed system has (the
user's demos, other users' demos, the base model, the app, persona-independent task checks).
Code: `train/improve.py`, `train/weighted_sft.py`, `scripts/improve_pair.sh`.
Research behind the five: `reports/Improving per user LoRA adapters.md`.

| Arm | Change vs C3 | Alice×calendar offline all / critical (18) | online success · pref adherence | Bob×send_email offline all / critical (12) | online success · pref adherence |
|---|---|---|---|---|---|
| C0 | base model | 42% / 5 | 0/6 · 0% | 44% / 0 | 3/6 · 25% |
| C1 | demos in prompt | **85% / 18** | 1/6 · **77%** | 74% / 3 | 2/6 · 29% |
| C3 | per-user LoRA | 71% / 12 | 2/6 · 40% | 80% / 5 | **5/6 · 75%** |
| A1 | + self-training on C3+demos rollouts | 51% / 6 | 0/6 · 0% | **87%** / 6 | 0/6 · 75% |
| A2 | + DPO vs other users' actions | 76% / **17** | 0/6 · 0% | 72% / 3 | 2/6 · 33% |
| A3 | init from C5, then user demos | 52% / 6 | 4/6* · 40% | — (C5 hit NaN) | — |
| A4 | habit thoughts, ×3 upsampling | 37% / 8 | 0/6 · 0% | — (hit NaN) | — |
| A5 | hard-token weighted loss | 50% / 0 | 0/6 · 0% | 80% / 5 | 5/6 · 75% |

\* A3 never emits `finished`; it starts a second event in 2 of 6 runs, so 4/6 overstates it.

**Result: no approach beats plain C3 on both pairs.**
- A2 is the only arm that adds the missing *click* preferences (it clicks "30 min"; C3 never
  does), but online its Thought and Action stop matching and it loses track of the form → 0/6.
  On Bob (preferences inside typed text) it is worse than C3.
- A1 is best offline on Bob but online sends emails with no body (clicks Message, then Send).
- A4 and A5 degrade the model on Alice (A4 types without clicking fields; A5 types garbage).
- Biggest confound: relative weekdays ("Tuesday") → every arm types wrong dates; this, not
  preferences, decides most calendar failures (see decision 4).
- Instability: 2 of 8 trainings hit NaN in bf16 (Bob C5, Bob A4).
- Bob's sign-off ("- Bob") is missed by every arm: the one preference inside long free text.

Caveats: oracle (scripted) demos, 6 dev tasks per pair, one seed, step CIs are optimistic.

## 7. Rerun with today's date in the prompt + A2b (Sun Nov 22–Mon Nov 23) — dev split, oracle demos, seed 0
`PROMPT_DATE=1` adds "Today is Monday, November 16, 2026" to every prompt (training and eval); C3
retrained with it. A2b = A2 + SFT anchor (ORPO from C3 over every demo step). Script:
`scripts/dated_pair.sh`; results under `results/*/oracle_dated/`.

| Arm | Alice×calendar offline all / critical | online success · adherence | Bob×send_email offline all / critical | online success · adherence |
|---|---|---|---|---|
| C0 | 40% / 4 of 18 | 0/6 · 0% | 44% / 0 of 12 | 0/6 · 0% |
| C1 | **88%** / 18 | 3/6 · **87%** | 70% / 2 | 3/6 · 29% |
| C3 | 81% / **18** | **4/6** · 67% | **78%** / 3 | **5/6 · 88%** |
| A2b | 67% / 17 | 0/6 · 0% | 70% / 2 | 3/6 · 71% |

- The date line lifted C3 on Alice from 2/6 → 4/6 and 40% → 67% adherence; offline critical 12 → 18/18.
- A2b still fails online on Alice (date typed as 11/17/2026, morning time, then loops) and is below C3 on Bob.
- Bob's emails contain no dates, so his C3 change (75% → 88%) is retraining variance, not the date.
- Scope: 2 of 12 pairs, scripted demos, 6 tasks each, one seed. A pilot finding, not a settled recipe.
