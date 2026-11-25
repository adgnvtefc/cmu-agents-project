# Weights, Context, or Skill File? Personalizing Computer-Use Agents from a Few Demos

Team: Rakshanda Hassan Abhinandan, Hanmo Zhang. Literature review:
[reports/Personalized LoRA for computer use agents.md](reports/Personalized%20LoRA%20for%20computer%20use%20agents.md)
(notes in `research_notes/`).

## Research question
A user records 3–5 demos of a GUI skill. **Where should what they demonstrated live: in the
model's weights (a per-(user, skill) LoRA), in its context (raw demos), or in an induced skill
file (SKILL.md)?** We test this on underspecified tasks whose correct completion differs between
users, and we check correctness programmatically.

**Why this framing.** Every component has precedent: AdaptAgent (LoRA from 2 web demos), FingerTip
(pooled personalized LoRA), OPPU (per-user LoRA, text only), Skill-to-LoRA (per-skill LoRA library,
text only), and Work2Skill (GUI recordings → SKILL.md). But **no work compares weights vs context
vs skill file on the same GUI model with the same few demos from one user**. The (user, skill)-keyed
library is what makes the integration worth more than its parts. It enables experiments none of
those papers could run: adapter swap, a granularity ladder, a step-position analysis, and cost
measured in one stack.

**Don't claim:** first personalized GUI agent / benchmark, first few-demo LoRA for GUI, first skill
LoRA library, first per-user LoRA. Say "to our knowledge", and re-search arXiv before the poster.

**Trust → controllability.** The proposal's trust motivation isn't backed by evidence for CUAs. We
reframe it as a scoped, revocable, per-skill delegation boundary, and measure over-application of
habits and wrong irreversible actions.

## Setup
- **Users:** 3 personas (played by teammates) with **conflicting preferences on the same skills**,
  e.g. receipts → `/Finance/2026` vs `/Receipts`; CC manager vs never CC; 30- vs 50-minute
  meetings. Every persona's environment is seeded with the other personas' choices as
  distractors, so the environment alone can't reveal the answer.
- **Skills (4):** send email, search/filter email, create calendar event, organize files.
- **Data:** per (user, skill) pair, 5 train demos + 3 recorded test demos. Test demos are
  *variants* of the skill (new entities and values). 12 pairs, 96 recordings.
- **Preference-critical step:** a step where the gold actions differ across personas for the same
  instruction. This gives an objective split for the offline metrics.

## Conditions (all on UI-TARS-1.5-7B; same 36 test tasks; paired)
| # | Condition | Priority |
|---|---|---|
| C0 | Zero-shot | essential |
| C1 | Raw demos in context (best demo with screenshots + others as text logs) | essential |
| C2 | Demo → SKILL.md (frontier LLM writes one hierarchical skill file per pair) | essential |
| C3 | **Per-(user, skill) LoRA** | essential |
| C4 | LoRA + SKILL.md | essential |
| C5 | Same-skill, *other-users* LoRA (key control: skill vs personalization) | essential |
| C6 | Pooled LoRA (all users, all skills, no key) | essential |
| C7 | Adapter swap: A's adapter in B's environment | essential (cheap) |
| R | Instruction → skill router, evaluated vs oracle | essential-light |
| N1–N7 | per-user all-skills LoRA, pooled + SKILL.md, LoRA + raw demos, k∈{1,3,5}, Qwen2.5-VL-7B backbone, layout perturbation, forgetting probe | cut in reverse order of value |

## Hypotheses (pre-registered)
- **H1:** C3 alone ties or loses to C1/C2 on overall online success (the likely outcome; the paper must not depend on refuting it).
- **H2:** C3 beats C1 on preference-critical steps *after step 1*.
- **H3:** C4 ≥ the best single arm.
- **H4:** C3 ≈ C5 on success but C3 ≫ C5 on preference adherence.
- **H5:** C6 is near chance on conflicting preferences.
- **H6:** under swap (C7) the agent follows the adapter owner's preferences.
- **H7:** the LoRA arms over-apply habits when the instruction explicitly overrides them.

## Build decisions (from the literature)
- **Model:** UI-TARS-1.5-7B (Apache-2.0, Qwen2.5-VL arch). Native `Thought: … / Action: …`. Use
  whichever click syntax the checkpoint emits zero-shot.
- **Coordinates:** absolute pixels in the smart-resized image. **Use a viewport that's a multiple of
  28 (1288×812)** so resizing is the identity. Pin `max_pixels` identically in the trainer, vLLM, and
  the evaluator. Gold-click replay unit test before any training.
- **History:** last ≤5 screenshots + all prior Thought/Action text; loss on the current turn only.
- **Thoughts:** short post-hoc thoughts written by a strong VLM from current and past screens only.
- **LoRA:** r=16, α=32, dropout 0.05, all LLM linear layers, vision tower + merger frozen,
  LR 1e-4 (sweep 5e-5 to 2e-4), ~50–200 steps chosen by leave-one-demo-out on a dev pair, 2–3 seeds
  for C3/C5. Trainer: LLaMA-Factory or ms-swift. Pin versions.
- **Serving:** vLLM multi-LoRA (`--enable-lora`, runtime load), with no tower/connector LoRA.
  Measure swap latency and tokens per step.
- **Environment:** build FastAPI + SQLite mail/calendar/drive, copying the OpenApps design (hidden
  state endpoint, reset per episode, full-state checks, pinned dates).
- **Checkers:** success assertions + per-preference assertions + collateral-damage check.
  Validate: all human test demos pass, a no-op agent fails.
- **Recorder:** custom Playwright with capture-phase listeners + `getBoundingClientRect()`, the last
  settled *pre-action* frame, keystrokes merged into `type`, JSONL + trace.zip + DB dumps.
- **Metrics:**
  - Offline: action-type accuracy, click-in-bbox, step success, split by preference-critical
    vs other and by step index.
  - Online: task success, preference adherence, joint success, collateral damage, steps,
    over-application rate, wrong irreversible actions.
- **Statistics:** paired McNemar / permutation tests; SEs clustered by test task (online) and by
  episode (offline); 3 instruction paraphrases per task. No user-level claims with 3 personas.

## Schedule
| Date | Milestone |
|---|---|
| Fri Nov 20 | Lit review done, plan locked ✓ |
| Sat Nov 21 | Mock apps, personas, 32 task instances, checkers (validated on all 96 pairs), recorder ✓. Next: UI-TARS harness on Modal + **C0 gate** (zero-shot on a few tasks) + gold-click replay |
| Sun Nov 22 | Record all 96 demos (~3–4 h, split across the team); thought annotation; SKILL.md generation |
| Mon Nov 23 | Pick LoRA recipe on a dev pair; train all adapters on Modal; offline eval |
| Tue Nov 24 | Online eval for all conditions + router |
| Wed Nov 25 | Analysis + figures (buffer for reruns) |
| Thu–Sun Nov 26–29 | Poster + report draft |
| Mon Nov 30 | Poster printed |
| Tue Dec 1 / Thu Dec 3 | Poster session |
| Fri Dec 4 | Report (late days until Mon Dec 7) |

## Repo layout (planned)
```
apps/        mock mail/calendar/drive (FastAPI + SQLite), persona seeds, task + checker defs
recorder/    Playwright demo recorder
data/        raw trajectories + converted SFT data + SKILL.md files
agent/       UI-TARS harness: prompts, action parser, Playwright executor, router
train/       LoRA configs + Modal training script
serve/       vLLM multi-LoRA on Modal
eval/        offline step metrics, online runner, stats + figures
paper/       ACL LaTeX report + poster
```

## Before citing
Several items in the review are marked UNVERIFIED: venues, abstract-only details, whether
LearnAct's ACL 2026 version has a demo-LoRA vs ICL comparison, AdaptAgent's LoRA settings, and the
UI-TARS click syntax and default `max_pixels`. Check them before they go into the report.
