# Open GUI / Computer-Use Agent Models, SFT Data & Action Formats, Data-Efficient Training, and Forgetting (as of Nov 2026)

Scope note: written for a 2-week CMU workshop project that LoRA-fine-tunes an open GUI-agent VLM on 3–5 human demos per user/skill (≈15–75 step-level multimodal samples per adapter) recorded in self-built mock web apps in Chromium at 1280×800. Candidates: Qwen2.5-VL-7B-Instruct (proposal) vs UI-TARS-1.5-7B (current plan). Items marked UNVERIFIED come from memory or secondary summaries that I could not confirm against a primary source during this session. Benchmark numbers are as self-reported by each project unless noted; harnesses differ, so cross-paper comparisons are approximate.

---

## Q1. UI-TARS, UI-TARS-1.5 (7B), UI-TARS-2: action space, prompt template, coordinates, history, license, fine-tuning

### Takeaway
UI-TARS-1.5-7B is open (Apache-2.0), uses the Qwen2.5-VL architecture (so any Qwen2.5-VL LoRA tooling works), emits `Thought: … / Action: …` with **absolute pixel coordinates in the smart-resized image space**, and its reference harness keeps ~5 recent screenshots plus text history of all earlier steps. There is no official fine-tuning recipe (the repo is archived and contains inference/parsing code only); UI-TARS-2 (Sep 2025) appears not to have released weights.

### Cited Findings

**UI-TARS (v1) — ByteDance Seed & Tsinghua (UNVERIFIED affiliation detail), Jan 2025, arXiv 2501.12326 — https://arxiv.org/abs/2501.12326**
- Base: Qwen2-VL 7B/72B continually trained on ~50B tokens → UI-TARS-7B / 72B. — [UI-TARS paper](https://arxiv.org/html/2501.12326)
- Unified action space: shared Click(x,y), Drag, Scroll, Type, Wait, Finished, CallUser; desktop Hotkey, LeftDouble, RightSingle; mobile LongPress, PressBack, PressHome, PressEnter. — [UI-TARS paper](https://arxiv.org/html/2501.12326)
- v1 coordinates were normalized to screen dimensions (resolution-independent; consumers divide by 1000). — [UI-TARS paper](https://arxiv.org/html/2501.12326); [OSWorld uitars15_v1.py](https://raw.githubusercontent.com/xlang-ai/OSWorld/main/mm_agents/uitars15_v1.py) (non-qwen25vl model types divide by 1000)
- History: model sees only the last N screenshots; the full history of thoughts+actions is kept as text "short-term memory"; ~32k token budget. — [UI-TARS paper](https://arxiv.org/html/2501.12326)
- Thought generation is **post hoc**: *ActRe* prompts a VLM to write a thought conditioned on the ground-truth action (authors note this can produce thoughts that only superficially match), and *thought bootstrapping* samples thought–action pairs from an early checkpoint and keeps thoughts whose action matches GT. Action-only ("vanilla") traces are also mixed into training. Reflection tuning uses annotated error→correction pairs trained with DPO. — [UI-TARS paper](https://arxiv.org/html/2501.12326)
- UI-TARS-72B numbers visible: ScreenSpot-Pro 38.1, OSWorld 24.6 (50 steps) / 22.7 (15 steps), AndroidWorld 46.6. — [UI-TARS paper](https://arxiv.org/html/2501.12326)

**UI-TARS-1.5-7B — ByteDance Seed, 2025 (release ~Apr 2025, UNVERIFIED date) — https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B**
- License Apache-2.0; architecture tag `qwen2_5_vl` (base model not named on the card; card lists "8B params"). — [HF model card](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- Numbers: UI-TARS-1.5-7B OSWorld (100 steps) 27.5, ScreenSpot-Pro 49.6. Full (unreleased-size) UI-TARS-1.5: OSWorld 42.5, WindowsAgentArena 42.1, WebVoyager 84.8, Online-Mind2Web 75.8, AndroidWorld 64.2, ScreenSpot-V2 94.2, ScreenSpot-Pro 61.6. — [HF model card](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- Microsoft's independent re-run: UI-TARS-1.5-7B WebVoyager 66.4, Online-Mind2Web 31.3, ≈41 steps/task on average (vs ≈16 for Fara-7B). — [Fara-7B search summary of MSR blog/paper](https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/), [Fara paper arXiv 2511.19663](https://arxiv.org/abs/2511.19663)
- Card caveat: 7B "focuses primarily on enhancing general computer use capabilities" and is not optimized for games. — [HF model card](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- **Coordinates:** README states "Qwen 2.5-VL-based models utilize absolute coordinates to ground objects"; the reference parser is called with `factor=1000, model_type="qwen25vl"` plus original image W/H. — [UI-TARS GitHub README](https://github.com/bytedance/UI-TARS)
- **Conversion recipe:** `smart_resize` with factor 28, `min_pixels = 100*28*28`, `max_pixels = 16384*28*28`, max aspect ratio 200; map back with `x_orig = x_model / new_width * width` (same for y). — [README_coordinates.md](https://raw.githubusercontent.com/bytedance/UI-TARS/main/README_coordinates.md)
- **Reference agent harness (OSWorld `uitars15_v1.py`)**: default `history_n = 5`; system message "You are a helpful assistant."; user prompt from `UITARS_USR_PROMPT_THOUGHT` or `..._NOTHOUGHT`; for qwen25vl output is divided by the smart-resized H/W of the last image, other model types divide by 1000; message structure = instruction, then per past step an assistant turn (older steps text-only, the most recent ~history_n steps preceded by their screenshot), then current screenshot; `frequency_penalty=1`; on parse failure retries with temperature 1. — [OSWorld uitars15_v1.py](https://raw.githubusercontent.com/xlang-ai/OSWorld/main/mm_agents/uitars15_v1.py)
- **Prompt template (current repo main, `codes/ui_tars/prompt.py`)**: templates are now named `COMPUTER_USE_DOUBAO`, `MOBILE_USE_DOUBAO`, `GROUNDING_DOUBAO`. Computer-use template: "You are a GUI agent. You are given a task and your action history, with screenshots…", Output Format ```Thought: ... / Action: ...```; actions `click(point='<point>x1 y1</point>')`, `left_double(...)`, `right_single(...)`, `drag(start_point=..., end_point=...)`, `hotkey(key='ctrl c')` (≤3 keys, lowercase, space-separated), `type(content='xxx')` (end with `\n` to submit), `scroll(point=..., direction='down or up or right or left')`, `wait()` (5 s), `finished(content='xxx')`; Note: "Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part." GROUNDING template outputs only `Action:`. — [prompt.py](https://raw.githubusercontent.com/bytedance/UI-TARS/main/codes/ui_tars/prompt.py)
- The README's own example still shows the older `click(start_box='(100,200)')` syntax, i.e., the repo has two syntaxes (start_box vs `<point>`). — [UI-TARS GitHub README](https://github.com/bytedance/UI-TARS)
- Repo is **archived/read-only**; no training or fine-tuning instructions; post-processing via `pip install ui-tars`. — [UI-TARS GitHub README](https://github.com/bytedance/UI-TARS)
- Community: the HF card lists 12 fine-tunes and 2 adapters derived from UI-TARS-1.5-7B (no recipe given); a HF Space loading the model hit `ValueError: Unrecognized configuration class` (transformers-version mismatch). — [HF model card](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B); [search summary incl. HF Space](https://huggingface.co/spaces/premuim89/GUITARS)
- A third-party repo (2U1/Qwen-VL-Series-Finetune) supports LoRA/full fine-tuning of Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3.5 using only HF + Liger-Kernel; usable for UI-TARS-1.5 because it shares the qwen2_5_vl architecture. — [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)

**UI-TARS-2 — ByteDance Seed, Sep 2025, arXiv 2509.02544 — https://arxiv.org/abs/2509.02544**
- Technical report on multi-turn RL: stabilized multi-turn RL framework, data flywheel, hybrid GUI+file-system+terminal environment; reports large gains over UI-TARS-1.5 (88.2 on a GUI benchmark per abstract snippet). — [arXiv 2509.02544 / search summary](https://arxiv.org/abs/2509.02544)
- UNVERIFIED (from memory of the abstract): Online-Mind2Web 88.2, OSWorld 47.5, WindowsAgentArena 50.6, AndroidWorld 73.3.
- No evidence of an open-weights release; a HF paper-page comment asks "Is the model going to be open source?" with no answer found. — [HF papers page](https://huggingface.co/papers/2509.02544)

### Inferences
- **Recommendation (UI-TARS-1.5-7B):** it is a defensible primary base: Apache-2.0, Qwen2.5-VL arch (LLaMA-Factory / ms-swift / 2U1 / qwen-vl-finetune all apply), +20.6 ScreenSpot-Pro points over Qwen2.5-VL-7B (49.6 vs 29.0), and a native agent output format — so 15–75 samples only need to teach *user/skill-specific behavior*, not the format.
- **Fix one prompt syntax and never mix.** Because the repo shows both `start_box='(x,y)'` and `<point>x y</point>` syntaxes, decode a few zero-shot rollouts of UI-TARS-1.5-7B on your mock app with the OSWorld `uitars15_v1.py` prompt and adopt whatever syntax the model actually emits as the SFT target. Training on a syntax the model was not trained on wastes the tiny data budget on format learning.
- **Coordinates for 1280×800:** under UI-TARS's own `max_pixels=16384*28*28`, smart_resize gives 1288×812 (29×46 = 1,334 visual tokens/screenshot); the model's absolute coords live in that space (≈ +0.6%/+1.5% vs viewport). If a training framework uses a smaller `max_pixels` (e.g., qwen-vl-finetune example `576*28*28` → 840×504, or HF-card `1280*28*28` → 1260×784), **targets must be recomputed in that resized space and inference must use the same max_pixels** — this mismatch is the single most likely silent failure. (Arithmetic derived from the smart_resize spec in [README_coordinates.md](https://raw.githubusercontent.com/bytedance/UI-TARS/main/README_coordinates.md).)
- **History:** mirror the harness — ≤5 screenshots (most recent steps) + text of all prior Thought/Action turns. With 15–75 samples, each training sample = one step with its history prefix; loss only on the final assistant turn.
- Expect UI-TARS-1.5-7B to take many steps (≈41 avg on web tasks); mock-app tasks of 5–15 steps are short, but step-cap and loop detection should be in the eval harness.
- Do not plan around UI-TARS-2; treat it as closed.

### Gaps
- Exact text of `UITARS_USR_PROMPT_THOUGHT` used for UI-TARS-1.5 in OSWorld was not extracted (only its name); UNVERIFIED from memory that it uses `click(start_box='<|box_start|>(x1,y1)<|box_end|>')` syntax with `call_user()` and `finished()`.
- No official statement of how many screenshots UI-TARS-1.5 was *trained* with; `history_n=5` is the evaluation-harness default.
- No public report found of LoRA fine-tuning UI-TARS-1.5-7B with measured forgetting; no GitHub issue thread with a recipe was found.
- UI-TARS-1.5-7B exact release date and preprocessor `max_pixels` default not confirmed.

---

## Q2. Qwen2.5-VL-7B and newer Qwen VLMs (Qwen3-VL, Qwen3.5): grounding, coordinate convention, zero-shot agent ability

### Takeaway
Qwen2.5-VL-7B-Instruct grounds well on ordinary screenshots (ScreenSpot 84.7) but is weak on hi-res (ScreenSpot-Pro 29.0) and as an autonomous agent (AndroidWorld 25.5); it uses absolute pixel coordinates of the resized image. Qwen3-VL (late 2025) switched to **normalized 0–1000** coordinates, and Qwen3.5 (Feb 2026, natively multimodal, Apache-2.0) is far stronger as a zero-shot computer-use agent (OSWorld-Verified 41.8, ScreenSpot-Pro 65.2 for 9B) but uses a new hybrid Gated-DeltaNet architecture.

### Cited Findings

**Qwen2.5-VL-7B-Instruct — Alibaba Qwen, Jan/Feb 2025, arXiv 2502.13923 — https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct**
- Agent/GUI table (7B): ScreenSpot 84.7, ScreenSpot-Pro 29.0, AITZ_EM 81.9, AndroidControl-High EM 60.1, AndroidControl-Low EM 93.7, AndroidWorld SR 25.5, MobileMiniWob++ SR 91.4; OSWorld not reported for 7B. License Apache-2.0. — [HF model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Native-resolution input; default 4–16,384 visual tokens per image; card suggests 256–1,280 tokens (`min_pixels=256*28*28`, `max_pixels=1280*28*28`); `resized_height/width` rounded to multiples of 28. — [HF model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Qwen2.5-VL-based GUI models use **absolute** coordinates in the smart-resized image (UI-TARS README; OpenCUA card: "absolute, relative to the smart-resized image … factor 28, min_pixels 3136, max_pixels 12845056"). — [UI-TARS README](https://github.com/bytedance/UI-TARS); [OpenCUA-7B card](https://huggingface.co/xlangai/OpenCUA-7B)
- Official finetune example grounding format: `{"bbox_2d": [135, 114, 1016, 672]}` with prompt "Locate … output the bbox coordinates in JSON format". — [qwen-vl-finetune](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)
- With small RL, Qwen2.5-VL-7B base ScreenSpot-Pro is reported as 25.8 in UIShift's re-eval. — [UIShift arXiv 2505.12493](https://arxiv.org/html/2505.12493v2)

**Qwen3-VL — Alibaba Qwen, tech report arXiv 2511.21631 (v2 27 Nov 2025) — https://arxiv.org/abs/2511.21631**
- Dense 2B/4B/8B/32B + MoE 30B-A3B and 235B-A22B; Instruct and Thinking variants for each size. — [Qwen3-VL report](https://arxiv.org/html/2511.21631)
- Grounding uses "a normalized coordinate system scaled to the range [0,1000]", a change from Qwen2.5-VL. — [Qwen3-VL report](https://arxiv.org/html/2511.21631)
- 32B: OSWorld 41, AndroidWorld 63.7; report gives no 8B/4B GUI numbers in text (says small models "highly competitive"). — [Qwen3-VL report](https://arxiv.org/html/2511.21631)
- Third-party evals of Qwen3-VL-8B: ScreenSpot-v2 91.7 and ~64.4 on an OSWorld-G-style table (MAI-UI report); GroundGUI-8B gains +14.0 ScreenSpot-Pro / +10.6 OSWorld-G over Qwen3-VL-8B. — [MAI-UI arXiv 2512.22047](https://arxiv.org/pdf/2512.22047); [GroundGUI arXiv 2610.19100](https://arxiv.org/pdf/2610.19100)
- Practical note: in "relative mode" one third-party harness maps Qwen3-VL coordinates via `x_screen = floor(x · w / 999)` (a 0–999 grid); Qwen-CUA uses a normalized 0..999 grid. Check the exact denominator (999 vs 1000) in the target model's repo. — [CutVerse arXiv 2605.19484](https://arxiv.org/pdf/2605.19484); [Qwen-CUA card](https://huggingface.co/xlangai/Qwen-CUA)
- Qwen3-VL-8B-Instruct is Apache-2.0. — [HF card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

**Qwen3.5 (e.g., Qwen3.5-9B) — Alibaba Qwen, Feb 2026 — https://huggingface.co/Qwen/Qwen3.5-9B**
- Apache-2.0; natively multimodal (early fusion), claims to outperform Qwen3-VL; hybrid architecture (3× Gated DeltaNet + 1× Gated Attention blocks, ×8; 32 layers; 262k context; MTP). Thinking mode on by default (`enable_thinking: False` to disable); thinking content should be excluded from multi-turn history. — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- GUI numbers (9B): ScreenSpot-Pro 65.2, OSWorld-Verified 41.8, AndroidWorld 57.8. — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- A user reported a "large gap" reproducing the OSWorld-Verified score (unanswered). — [HF discussion #54](https://huggingface.co/Qwen/Qwen3.5-9B/discussions/54)
- On web agent benchmarks, Qwen3.5-9B: Online-Mind2Web 27.3, WebVoyager 46.9 (as reported in UI-Venus-2's table). — [UI-Venus-2-9B card](https://huggingface.co/inclusionAI/UI-Venus-2-9B)

### Inferences
- **Recommendation (Qwen2.5-VL-7B-Instruct):** keep it as the *general-VLM baseline* (same architecture as UI-TARS-1.5-7B → an apples-to-apples "does GUI pretraining matter for tiny-data personalization?" comparison), not as the main system. With 15–75 samples it must learn the action grammar, history handling and grounding style at once; expect format errors and weak multi-step behavior.
- For 1280×800 mock web apps (large, clean elements), Qwen2.5-VL-7B's grounding (ScreenSpot 84.7) is probably adequate; its weakness is policy/planning, which tiny SFT is least able to fix.
- Qwen3-VL/Qwen3.5 are stronger but (a) use normalized 0–1000/0–999 coordinates — a different target space from Qwen2.5-VL derivatives, (b) Qwen3.5's linear-attention hybrid may have less mature LoRA/vLLM/flash-linear-attention support. Within 2 weeks, only switch if the team already has a working Qwen3.5 training stack. If used, train in non-thinking mode to keep sequences short.

### Gaps
- Official Qwen3-VL-8B/4B ScreenSpot-Pro, OSWorld, AndroidWorld numbers not extracted (HF card shows them only as images).
- Qwen3.5's official computer-use prompt / coordinate convention not found on the card.
- Qwen2.5-VL-7B zero-shot OSWorld / web-agent success rate not found from a primary source.

---

## Q3. Other open GUI agents & datasets (OpenCUA/AgentNet, Aguvis, OS-Atlas, ShowUI, OS-Genesis, AgentTrek, GUI-Actor, Jedi/OSWorld-G, Fara, GUI-Owl-1.5, MAI-UI, UI-Venus-2, newer) — and the strongest open 3B–9B web model as of late 2026

### Takeaway
Through 2025 the open 7B agents clustered at ~27% OSWorld (UI-TARS-1.5-7B 27.5, OpenCUA-7B 26.6) and Fara-7B led 7B web agents (WebVoyager 73.5, Online-Mind2Web 34.1). In 2026 Qwen3.5-based ~9B agents jumped far ahead: Fara1.5-9B (MIT; Online-Mind2Web 63.4) and UI-Venus-2-9B (license pending; Online-Mind2Web 74.0, OSWorld-Verified 70.8) are the strongest open ≤9B web agents found, with GUI-Owl-1.5-8B next. Among **Qwen2.5-VL-architecture** 7B models (easiest to LoRA), Fara-7B is strongest on web and UI-TARS-1.5-7B strongest on grounding/general desktop.

### Cited Findings

**OpenCUA / AgentNet / AgentNetTool — XLANG Lab (HKU, UNVERIFIED) et al., Aug 2025, arXiv 2508.09123, NeurIPS 2025 Spotlight — https://huggingface.co/xlangai/OpenCUA-7B**
- AgentNet: 22,625 human-annotated desktop tasks (~12K Windows, 5K macOS, 5K Ubuntu), avg 18.6 steps, 720p–4K, 140+ apps and 190 websites; 41,428 trajectories used to train OpenCUA-7B/32B. — [OpenCUA paper](https://arxiv.org/html/2508.09123)
- AgentNetTool: background cross-OS recorder (screen video, mouse/keyboard, a11y tree) → state–action trajectories annotators review; tolerates imperfect demos to teach error recovery. — [OpenCUA paper](https://arxiv.org/html/2508.09123)
- Action space: pyautogui (`pyautogui.click(x=960, y=324)`), click variants, moveTo, dragTo, scroll, write, press, hotkey, wait, terminate; absolute coordinates on the smart-resized image. — [OpenCUA paper](https://arxiv.org/html/2508.09123); [OpenCUA-7B card](https://huggingface.co/xlangai/OpenCUA-7B)
- Reasoning levels L3 (observation) / L2 (thought: reflection+planning) / L1 (action); OSWorld agent uses L2 by default; history = 3 screenshots (3→5 gave marginal gains for ~3K extra tokens). — [OpenCUA paper](https://arxiv.org/html/2508.09123)
- OpenCUA-7B: OSWorld-Verified 24.3 / 27.9 / 26.6 (15/50/100 steps), OSWorld-G 55.3, ScreenSpot-V2 92.3, ScreenSpot-Pro 50.0, UI-Vision 29.7, AgentNetBench avg 75.2; OpenCUA-32B 34.8, OpenCUA-72B 45.0 OSWorld-Verified. License MIT. — [OpenCUA-7B card](https://huggingface.co/xlangai/OpenCUA-7B); [OpenCUA paper](https://arxiv.org/html/2508.09123)
- **Fine-tuning hazard:** OpenCUA replaced Qwen's M-RoPE with 1D RoPE and uses the Kimi-VL tokenizer/chat template; card says to align these if you train. — [OpenCUA-7B card](https://huggingface.co/xlangai/OpenCUA-7B)
- Training: 7B joint training 200B tokens, peak LR 2.5e-5, batch 512, 128 A100s; ~35% general text data in the mix, and removing general data slightly *reduced* agent performance. — [OpenCUA paper](https://arxiv.org/html/2508.09123)

**Aguvis — Xu et al. (HKU / Salesforce, UNVERIFIED affiliations), Dec 2024, arXiv 2412.04454, ICML 2025 — https://arxiv.org/abs/2412.04454**
- Pure-vision agent on Qwen2-VL (7B/72B); two stages: Stage 1 grounding (~1.036M), Stage 2 planning & reasoning with inner monologue (~35K trajectories); pyautogui actions plus pluggable mobile/terminate/answer actions. — [Aguvis paper](https://arxiv.org/html/2412.04454v2)
- Aguvis-7B: ScreenSpot 84.4; Multimodal-Mind2Web step SR 60.4 / 54.6 / 56.6 (cross-task/website/domain); AndroidControl High 61.5 / Low 80.5; OSWorld (GPT-4o planner + Aguvis-7B) 14.79; Aguvis-72B alone 10.26. — [Aguvis paper](https://arxiv.org/html/2412.04454v2)

**OS-Atlas — Wu et al. (Shanghai AI Lab et al., UNVERIFIED), Oct 2024, arXiv 2410.23218 — https://arxiv.org/abs/2410.23218**
- "Foundational GUI action model"; cross-platform grounding synthesis toolkit; >13M GUI elements, the largest open cross-platform grounding corpus at the time; evaluated on 6 benchmarks. — [arXiv abstract](https://arxiv.org/abs/2410.23218)
- OS-Atlas-7B AndroidControl SFT used ~76K samples (as cited by UI-R1). — [UI-R1 paper](https://arxiv.org/html/2503.21620)
- UNVERIFIED: OS-Atlas-Base-7B is built on Qwen2-VL-7B (4B on InternVL2).

**ShowUI — Lin et al. (Show Lab NUS / Microsoft), Nov 2024, arXiv 2411.17465 — https://arxiv.org/abs/2411.17465**
- 2B VLA model (UNVERIFIED: Qwen2-VL-2B base) with UI-guided visual token selection (−33% redundant tokens, 1.4× speed) and interleaved vision-language-action streaming; ~256K training examples; 75.1% zero-shot screenshot grounding; navigation on Mind2Web, AITW, MiniWob. — [arXiv abstract](https://arxiv.org/abs/2411.17465)

**ShowUI-Aloha: Human-Taught GUI Agent — Zhang et al. (Show Lab), Jan 2026, arXiv 2601.07181 — https://arxiv.org/abs/2601.07181**
- Pipeline that turns unannotated human desktop screen recordings into structured tasks: Recorder (video + clicks/keys/scrolls) → Learner (captions) → Planner → Executor. — [arXiv abstract](https://arxiv.org/abs/2601.07181)

**GUI-Actor — Wu et al. (Microsoft, UNVERIFIED), Jun 2025, arXiv 2506.03143 — https://arxiv.org/abs/2506.03143**
- Coordinate-free grounding: an `<ACTOR>` token attends over visual patches via an attention action head, plus a grounding verifier. GUI-Actor-7B ScreenSpot-Pro 40.7 (Qwen2-VL) / 44.6 (Qwen2.5-VL), above UI-TARS-72B (38.1). Training only the ~100M-param action head with a frozen backbone reaches performance comparable to prior SOTA, preserving general VLM ability. — [arXiv abstract](https://arxiv.org/abs/2506.03143)

**Jedi / OSWorld-G — Xie et al. (XLANG Lab), May 2025 (v3 Oct 2025), arXiv 2505.13227 — https://arxiv.org/abs/2505.13227**
- OSWorld-G: 564 grounding samples (text matching, element recognition, layout understanding, precise manipulation); Jedi: ~4M grounding examples; Jedi grounding lifted general foundation models' OSWorld agentic performance from 5% to 27%. — [arXiv abstract](https://arxiv.org/abs/2505.13227)
- UNVERIFIED: Jedi-3B/7B are Qwen2.5-VL-based; Jedi-7B ≈ 39.5 ScreenSpot-Pro, ≈ 91.7 ScreenSpot-v2, ≈ 54.1 OSWorld-G.

**OS-Genesis (arXiv 2412.19723) and AgentTrek (arXiv 2412.09605)**
- Not fetched this session. UNVERIFIED: OS-Genesis (Shanghai AI Lab et al.) synthesizes trajectories by "reverse task synthesis" (explore UI first, then derive tasks) and improved Qwen2-VL-7B on AndroidWorld/WebArena; AgentTrek (HKU/Salesforce) replays web tutorials with a VLM agent to synthesize ~10K web trajectories cheaply and trains Qwen2-VL-7B for Mind2Web/WebArena.

**Fara-7B — Microsoft Research, Nov 2025, arXiv 2511.19663 — https://huggingface.co/microsoft/Fara-7B**
- Base Qwen2.5-VL-7B; MIT license; SFT only on synthetic multi-agent (Magentic-One-derived) web trajectories (~145K trajectories per MSR summary) plus grounding, UI VQA/OCR and safety/refusal data; trained on 64 H100 for ~2.5 days. — [HF card](https://huggingface.co/microsoft/Fara-7B); [MSR blog](https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/)
- Action space (`computer_use` tool): key, type, mouse_move, left_click, scroll, visit_url, web_search, history_back, pause_and_memorize_fact, wait, terminate; `coordinate` as (x, y); screen 1428×896; output = chain-of-thought then a JSON tool call in XML tags; ChatML with a required system prompt. — [HF card](https://huggingface.co/microsoft/Fara-7B)
- WebVoyager 73.5, Online-Mind2Web 34.1, DeepShop 26.2, WebTailBench 38.4 (3-run averages); ≈16 steps/task vs ≈41 for UI-TARS-1.5-7B. Trained to stop at "critical points" (purchases, emails, sign-ins). — [HF card](https://huggingface.co/microsoft/Fara-7B); [MSR blog](https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/)

**Fara1.5 (4B / 9B / 27B) — Microsoft Research, May 21 2026 (weights MIT on HF per Jul 22 2026 update) — https://www.microsoft.com/en-us/research/articles/fara1-5-computer-use-agent**
- Base Qwen3.5; SFT only (cross-entropy on thought+action tokens); ~2M samples (60% web trajectories, 12.8% synthetic envs, 12.5% form-filling/user interaction, 8.8% grounding, 4.9% VQA…); trajectories from FaraGen1.5 (GPT-5.4 solver + user simulator + verifiers). Each step sees full text history + 3 most recent screenshots; loss only on actions in the most recent three turns. — [MSR Fara1.5 article](https://www.microsoft.com/en-us/research/articles/fara1-5-computer-use-agent)
- WebVoyager / Online-Mind2Web: Fara1.5-4B 80.8 / 57.3; Fara1.5-9B 86.6 / 63.4 (WebTailBench 64.5 process / 32.3 outcome); Fara1.5-27B 88.6 / 72.3. Comparators in same table: MolmoWeb-8B 78.2 / 35.3, Holo2-8B 80.2 / –, GUI-Owl-1.5-8B 78.1 / 48.6, Operator 87.0 / 58.3. — [MSR Fara1.5 article](https://www.microsoft.com/en-us/research/articles/fara1-5-computer-use-agent)

**GUI-Owl-1.5 / Mobile-Agent-v3.5 — Alibaba (Tongyi), Feb 2026, arXiv 2602.16855 — https://arxiv.org/abs/2602.16855**
- Instruct and thinking variants at 2B/4B/8B/32B/235B, open-sourced; Qwen3-VL-based (per third-party summary); hybrid data flywheel, unified thought synthesis, MRPO RL. Headline (largest model): OSWorld 56.5, AndroidWorld 71.6, WebArena 48.4, ScreenSpot-Pro 80.3. — [arXiv abstract](https://arxiv.org/abs/2602.16855); [search summary on Qwen3-VL base](https://arxiv.org/pdf/2602.16855)

**MAI-UI — Alibaba Tongyi-MAI, Dec 2025, arXiv 2512.22047 — https://arxiv.org/abs/2512.22047**
- Sizes 2B, 8B, 32B, 235B-A22B; abstract reports ScreenSpot-Pro 73.5, AndroidWorld 76.7, MobileWorld 41.7, OSWorld-G 70.9 (size not specified). — [arXiv abstract](https://arxiv.org/abs/2512.22047)

**UI-Venus-2 (9B / 27B) — inclusionAI (Ant Group), Aug 2026, arXiv 2609.00028 — https://huggingface.co/inclusionAI/UI-Venus-2-9B**
- Initialized from Qwen3.5-9B; mid-training on trajectories → offline step-level RL → multi-teacher on-policy distillation. **Weight license "pending final confirmation"** (Apache-2.0 deliberately not carried over). — [HF card](https://huggingface.co/inclusionAI/UI-Venus-2-9B)
- UI-Venus-2-9B vs Qwen3.5-9B: OSWorld-Verified 70.8 vs 41.8; AndroidWorld 80.2 vs 57.8; ScreenSpot-Pro 73.0; OSWorld-G-R 78.5; Online-Mind2Web 74.0 vs 27.3; WebVoyager 90.8 vs 46.9; OSHarm ASR 11.3 vs 25.3. Recommended: temp 1.0 with reasoning for agentic tasks, temp 0 without reasoning for grounding. — [HF card](https://huggingface.co/inclusionAI/UI-Venus-2-9B)

**Frontier-scale open models (for context, not trainable on a course budget)**
- Qwen-CUA (Qwen Team + XLANG Lab; Qwen3.5-family MoE 397B total / 17B active; Apache-2.0; arXiv 2608.02352): OSWorld-Verified 86.2, WebArena 64.16; 0..999 normalized coordinate grid; XML `computer_use` tool calls; scaffold keeps 20 most recent screenshots; ~40K verifiable tasks, iterative SFT + RLVR. — [HF card](https://huggingface.co/xlangai/Qwen-CUA)
- Qwen-UI-Agent (Alibaba, Jul 2026, arXiv 2607.28227; 27B mentioned; weights not confirmed): OSWorld-Verified 79.5, WebArena 73.6, ScreenSpot-Pro 81.5; batches multiple actions per turn. — [alphaXiv](https://www.alphaxiv.org/abs/2607.28227)

### Inferences
- **Strongest open ≤9B web agent (late 2026), by self-reported numbers:** UI-Venus-2-9B (Online-M2W 74.0 / WebVoyager 90.8; license unresolved) > Fara1.5-9B (63.4 / 86.6; MIT) > Fara1.5-4B (57.3 / 80.8; MIT) > GUI-Owl-1.5-8B (48.6 / 78.1) > Fara-7B (34.1 / 73.5) ≈ UI-TARS-1.5-7B (31.3 / 66.4 in MSR's run) > Qwen3.5-9B base (27.3 / 46.9). Different harnesses/judges; treat ±5 pts as noise.
- **Recommendation for this project:** stay on the Qwen2.5-VL architecture for tooling safety (UI-TARS-1.5-7B primary; Qwen2.5-VL-7B-Instruct as the general-VLM ablation). If one extra web-specialist comparison is affordable, **Fara-7B** is the best fit: same Qwen2.5-VL-7B backbone, MIT, web-native action set (visit_url, history_back, terminate), concise trajectories, and its own "stop at critical points" behavior. In the write-up, explicitly position results against Fara1.5 / UI-Venus-2 as the 2026 frontier and note that personalization (user-specific choices/preferences) is orthogonal to raw capability.
- **Avoid OpenCUA-7B as a LoRA base** in a 2-week window: 1D-RoPE + Kimi tokenizer/template means stock Qwen2.5-VL trainers can silently mis-train it.
- GUI-Actor's result (frozen backbone + ~100M action head ≈ SOTA) is evidence that grounding can be improved without touching most weights — supports freezing the vision tower and using small LoRA.
- Design lessons to copy: OpenCUA's 3-screenshot history and mixed CoT levels; Fara1.5's "loss only on the last few turns"; AgentNetTool-style recording (screen + input events + a11y/DOM) — the project's mock apps can additionally log **DOM bounding boxes** of every target element for free.

### Gaps
- OS-Genesis and AgentTrek details not verified this session.
- Base model / licenses of GUI-Owl-1.5 small sizes and MAI-UI open-weight sizes not confirmed from primary pages; MAI-UI per-size numbers not found.
- Holo2-8B (H Company) and MolmoWeb-8B (AI2) appear in Fara1.5's table but were not researched.
- No neutral leaderboard comparing all ≤9B models under one web harness was found.

---

## Q4. Data-efficient training: UI-R1, GUI-R1, InfiGUI-R1, GUI-G1, UIShift, PC-Agent-E, and few-demo adaptation — smallest data regime that helps; tiny-data SFT vs RFT

### Takeaway
The smallest published regime that clearly helps is **UI-R1: 136 samples with rule-based GRPO** on Qwen2.5-VL-3B (ScreenSpot 65.0→83.3, AndroidControl avg 75.8→88.5). Matched comparisons (UIShift on 1K pairs; CGL continual-learning) show GRPO beating SFT on held-out success and forgetting far less; but almost all of this evidence is for single-step grounding/action prediction, not multi-step personalized trajectories. In-context use of even one demo is a very strong competitor (LearnAct: 19.3→51.7 offline).

### Cited Findings

**UI-R1 — Lu, Chai et al. (vivo / CUHK MMLab, UNVERIFIED affiliations), Mar 2025 (v5 May 2025), arXiv 2503.21620 — https://arxiv.org/abs/2503.21620**
- 136 curated mobile tasks (101 click, 19 open_app, 9 back, 5 scroll, 2 input_text) chosen for quality, difficulty (only samples the base model got wrong) and diversity; GRPO, LR 9.98e-7→0, 8 epochs, 8 rollouts, batch 1×grad-acc 2, 8× RTX 4090 for ~8 h. Reward = action-type + binary click-in-bbox + format. Output `<think>…</think><answer>[action, coordinate:[x,y]]</answer>`. — [UI-R1 paper](https://arxiv.org/html/2503.21620)
- Base Qwen2.5-VL-3B → UI-R1-3B → UI-R1-E-3B: ScreenSpot 65.0 → 83.3 → 89.2; ScreenSpot-V2 70.4 → 85.4 → 89.5; ScreenSpot-Pro 11.8 → 17.8 → 33.5; AndroidControl type/grounding/avg 79.3/72.3/75.8 → 94.3/82.6/88.5. — [UI-R1 paper](https://arxiv.org/html/2503.21620)
- SFT comparison is not same-data: SFT on 500 ScreenSpot-mobile samples gave ScreenSpot 75.7 / ScreenSpot-Pro 10.8 (below base on Pro); UI-R1-3B is competitive with OS-Atlas-7B trained on 76K. Abstract's "+22.1%" ScreenSpot gain does not match Table 1 (+18.3). — [UI-R1 paper](https://arxiv.org/html/2503.21620); [arXiv abstract](https://arxiv.org/abs/2503.21620)

**GUI-R1 — Luo et al., Apr 2025 (v4 Oct 2025), arXiv 2504.10458 — https://arxiv.org/abs/2504.10458**
- GRPO with unified action-space rule rewards on ~3K samples across Windows/Linux/macOS/Android/Web; claims superior results to OS-Atlas using 0.02% of its data (3K vs 13M) across eight benchmarks; no matched SFT baseline. — [arXiv abstract](https://arxiv.org/abs/2504.10458); [search summary](https://arxiv.org/html/2504.10458v4)

**InfiGUI-R1 — Liu et al. (Zhejiang Univ. / InfiX, UNVERIFIED), Apr 2025, arXiv 2504.14239 — https://arxiv.org/abs/2504.14239**
- Actor2Reasoner: (1) Reasoning Injection via spatial-reasoning distillation from teacher models, (2) Deliberation Enhancement via RL with sub-goal guidance rewards and constructed error-recovery scenarios. — [arXiv abstract](https://arxiv.org/abs/2504.14239)
- UNVERIFIED: InfiGUI-R1-3B on Qwen2.5-VL-3B, ScreenSpot ≈87.5, ScreenSpot-Pro ≈35.7.

**GUI-G1 — Zhou et al. (Renmin Univ., UNVERIFIED), May 2025, arXiv 2505.15810 — https://arxiv.org/abs/2505.15810**
- R1-Zero-style grounding analysis: longer CoT before grounding *hurts*; hit/box-area rewards invite box-size reward hacking (fixed with a box-size constraint). GUI-G1-3B (Qwen2.5-VL-3B, 17K public samples): ScreenSpot 90.3, ScreenSpot-Pro 37.1, beating UI-TARS-7B. — [arXiv abstract](https://arxiv.org/abs/2505.15810)

**UIShift — Gao, Zhang, Xu (BUPT), May 2025, arXiv 2505.12493 — https://arxiv.org/abs/2505.12493**
- Self-supervised inverse-dynamics task (predict the action between two UI states) on 1K/2K transition pairs; **matched SFT vs GRPO** on Qwen2.5-VL (full-parameter). 7B, 1K: AndroidControl-Low SR SFT 38.9 vs GRPO 79.8; AC-High SR 20.4 vs 54.2. 3B, 1K: AC-Low SR 28.0 vs 85.4; AC-High 11.3 vs 50.0. SFT's grounding collapsed (7B AC-Low GR 23.3 vs 88.4). GRPO hyperparams: LR 1e-6→0, 8 generations, 8 epochs, KL β 0.04, clip 0.2, temp 0.9. — [UIShift paper](https://arxiv.org/html/2505.12493v2)
- Caveat: SFT here was on a proxy task (inverse dynamics) and evaluated on a different task, which disadvantages SFT more than in-task SFT would.

**CGL: Continual GUI Learning via RFT — Yao et al. (Huawei Noah's Ark / HIT et al.), Mar 2026, arXiv 2603.02951 — https://arxiv.org/abs/2603.02951**
- AndroidControl-CL (89 apps, 6,043 trajectories, 7 app super-classes, sequential tasks). Qwen2.5-VL-3B: SFT avg step-acc 76.90 / traj-acc 23.53 / forgetting measure −5.73; GRPO 81.53 / 36.78 / −0.62; CGL (GRPO + entropy-scheduled SFT on failed groups + gradient surgery) 82.33 / 38.03 / −0.02. SFT adapts fast but overwrites; GRPO retains but adapts slowly. — [CGL paper](https://arxiv.org/html/2603.02951)

**PC-Agent-E — He, Jin, Liu (GAIR-NLP; SJTU UNVERIFIED), May 2025 (v2 Mar 2026), ICLR 2026, arXiv 2505.13909 — https://arxiv.org/abs/2505.13909**
- Starts from only **312 human computer-use trajectories**, augments them by having Claude 3.7 Sonnet synthesize diverse alternative action decisions ("trajectory boost"); reports 141% relative improvement and +10% relative over Claude 3.7 Sonnet on WindowsAgentArena-V2; beats training on human trajectories alone and direct Claude distillation. — [arXiv abstract](https://arxiv.org/abs/2505.13909)
- UNVERIFIED (from memory): the method also reconstructs missing human "thoughts" post hoc ("thought completion") before boosting; base Qwen2.5-VL-72B.

**An Efficient Training Pipeline for Reasoning GUI Agents — Pantazopoulos & Özyiğit, Nov 2025, arXiv 2511.08172**
- Filters 4.8M synthetic examples to 12K (≈0.25%) via difficulty/misalignment/diversity filtering; trains a 3B VLM with SFT, CoT-augmented SFT, and GRPO (with PEFT); "match or surpass larger baselines" on ScreenSpot, Multimodal-Mind2Web, AndroidControl. — [arXiv abstract](https://arxiv.org/abs/2511.08172)

**Few-demo adaptation / personalization-adjacent work**
- LearnAct (Liu et al., Apr 2025, arXiv 2504.13805): LearnGUI benchmark (2,252 offline + 101 online tasks with human demos); multi-agent DemoParser/KnowSeeker/ActExecutor; **one demonstration** raised Gemini-1.5-Pro offline accuracy 19.3%→51.7% and UI-TARS-7B-SFT online success 18.1%→32.8%. — [arXiv abstract](https://arxiv.org/abs/2504.13805)
- AdaptAgent (ACL 2025): adapts multimodal web agents to new sites from ≤2 human demos via in-context and meta-learning adaptation (details not verified; PDF unreadable this session). — [ACL Anthology PDF](https://aclanthology.org/2025.acl-long.1008.pdf); [search summary](https://pith.science/citations/8dba372a-b244-4199-a234-d20a9d4e8875)
- CoAdapt-GUI (Yu et al., Aug 2026, arXiv 2608.11588): test-time adaptation that jointly updates a structured workflow context and a **LoRA policy adapter** on a frozen VLM (group-relative optimization on own rollouts); AndroidWorld-Generalization 45.0% vs 37.5% policy-only TTA; AndroidWorld Plus 38.6%→52.9%. — [arXiv PDF](https://arxiv.org/pdf/2608.11588)
- "Not All Experience Belongs in the Weights" (Wu, Ding, Huang, Oct 2026, arXiv 2610.01787): splits experience into locators, procedures, state facts, lessons; **locators and lessons work better in weights, procedures and state facts better in context**; a routing rule beat every whole-trajectory baseline (+3.5 pts vs best single destination). — [arXiv abstract](https://arxiv.org/abs/2610.01787)
- ALLOY (arXiv 2510.10049) names continued fine-tuning on user demonstrations as a promising but unimplemented direction. — [search summary](https://arxiv.org/pdf/2510.10049)

### Inferences
- **Smallest helpful regime:** ~136 samples (UI-R1, RL, single-step). No paper found that shows multi-step GUI-agent SFT gains from ≤75 samples; the project would be in genuinely new territory — frame it as such.
- **SFT on tiny data:** evidence (UI-R1's 500-sample SFT lowering ScreenSpot-Pro below base; UIShift; CGL; "SFT Memorizes, RL Generalizes") predicts that 15–75-sample SFT will fit the demos' exact click positions/paths and can degrade off-distribution grounding. LoRA + few epochs + early stopping mitigates but does not remove this.
- **RFT is feasible on this data:** because the mock apps are self-built, log each target element's DOM bbox at record time → a UI-R1-style reward (action-type match + click-inside-bbox + typed-text match + format) is free. A GRPO stage (8 rollouts, LR ~1e-6, LoRA) on the same 15–75 steps is a cheap, well-motivated ablation ("SFT vs RFT vs SFT→RFT on identical tiny data") that the literature has not run for personalized multi-step web tasks.
- **Must-have baselines:** (1) zero-shot base; (2) **in-context demo** (retrieve the user's 1–2 demos as text or screenshots in the prompt — LearnAct shows big gains); (3) LoRA-SFT; (4) optional RFT. The component-routing paper predicts LoRA will win on *locators* (where user-specific buttons are) and *lessons*, while in-context will win on *procedures/state facts* — design eval tasks that separate these.
- Augmentation à la PC-Agent-E: use a strong model to synthesize alternative valid actions/thoughts per recorded step to multiply 15–75 samples (watch for drift away from the user's actual preference — that's the thing being personalized).

### Gaps
- No same-data SFT-vs-RFT comparison exists for ≤100 multi-step web trajectories; no study of per-user LoRA for GUI agents with forgetting measurements was found.
- InfiGUI-R1 data size/base/numbers and GUI-R1 per-benchmark numbers not verified.
- AdaptAgent's fine-tuning results (whether 1–2-demo fine-tuning helped or overfit) could not be read.

---

## Q5. Catastrophic forgetting when fine-tuning GUI agents on narrow domains; LoRA vs full FT; LoRA rank/alpha/LR/epochs/target modules for Qwen2.5-VL-based agents

### Takeaway
GUI-specific forgetting evidence is recent and mostly about SFT vs RL (SFT forgets ~10× more than GRPO in CGL), while LoRA-vs-full evidence comes from general LLM work ("LoRA Learns Less and Forgets Less"). For Qwen2.5-VL, the official finetune code freezes the vision tower by default and its LoRA example is r=8, α=16; no GUI-specific LoRA hyperparameter study was found, so settings must be chosen by small sweeps with a forgetting probe.

### Cited Findings
- **CGL** (GUI, 2026): sequential app-group SFT on Qwen2.5-VL-3B gives forgetting measure −5.73 vs −0.62 for GRPO and −0.02 for CGL; SFT+replay recovers part of the gap (paper has an inconsistency: 27.19 vs 30.19 traj-acc for SFT+Replay). LoRA vs full not studied. — [CGL paper](https://arxiv.org/html/2603.02951)
- **Continual GUI Agents / GUI-AiF** (arXiv 2601.20732, Jan 2026): frames GUI continual learning as shifts in domain (mobile/desktop/web) and screen resolution; proposes RFT with anchoring rewards for grounding. — [search summary of arXiv 2601.20732](https://arxiv.org/abs/2601.20732)
- **LoRA Learns Less and Forgets Less** (Biderman et al., Databricks/Columbia [affiliations UNVERIFIED], TMLR 2024, arXiv 2405.09673): LoRA substantially underperforms full FT on code/math targets but better preserves out-of-domain ability than full FT, weight decay or dropout; full FT learns perturbations of 10–100× higher rank than typical LoRA. — [arXiv abstract](https://arxiv.org/abs/2405.09673)
- UNVERIFIED (recall of the same paper's best-practice section): target all linear modules (attention + MLP) rather than attention only; LoRA is more LR-sensitive than full FT; rank 16 is usually sufficient for instruction-style tuning.
- **RL's Razor** (Shenfeld, Pari, Agrawal, MIT [UNVERIFIED], Sep 2025, arXiv 2509.04259): forgetting tracks the KL divergence between fine-tuned and base model *measured on the new task*; on-policy RL stays closer in KL than SFT even at equal new-task performance. — [arXiv abstract](https://arxiv.org/abs/2509.04259)
- **SFT Memorizes, RL Generalizes** (Chu et al., Jan 2025, arXiv 2501.17161): SFT memorizes and fails OOD; outcome-reward RL generalizes to unseen textual and visual variants and improves visual recognition; SFT is still useful to stabilize output format before RL. — [arXiv abstract](https://arxiv.org/abs/2501.17161)
- **General-data mixing helps agents:** OpenCUA used ~35% general text; removing it slightly lowered agent performance. — [OpenCUA paper](https://arxiv.org/html/2508.09123)
- **Official Qwen2.5-VL finetune defaults (qwen-vl-finetune):** `tune_mm_llm True`, `tune_mm_vision False`, `tune_mm_mlp False`; full-FT LLM LR 2e-7 (range ~1e-6–2e-7), `mm_projector_lr 1e-5`, `vision_tower_lr 1e-6`; LoRA example `lora_r 8`, `lora_alpha 16`, `lora_dropout 0.0`; 3 epochs, cosine, warmup 0.03, weight decay 0.01, per-device batch 4 × grad-acc 4, `max_pixels 576*28*28`, `min_pixels 16*28*28`, `model_max_length 4096`, ZeRO-3, bf16; resolution called "critical". — [qwen-vl-finetune](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)
- **Narrow-domain risk (general sources):** fine-tuning breaks hardest when domain is narrow, data small, and eval mirrors training; mitigations = rehearsal data, KL anchors, early stopping; LaLoRA proposes curvature-aware regularization for LoRA. — [Mitigating Forgetting in LoRA, arXiv 2512.17720](https://arxiv.org/pdf/2512.17720); [blog summary](https://alpeshnakrani.com/blog/catastrophic-forgetting/)
- **Frozen-backbone evidence:** GUI-Actor trains only a ~100M action head with frozen VLM and preserves general ability. — [GUI-Actor abstract](https://arxiv.org/abs/2506.03143)

### Inferences
- **Recommended starting config (UI-TARS-1.5-7B or Qwen2.5-VL-7B), per adapter, 15–75 step samples:** LoRA on all LLM linear layers (q,k,v,o,gate,up,down); **freeze ViT and the merger/projector**; r=16, α=32 (α/r=2 as in the official r=8/α=16 example), dropout 0.05; LR 1e-4 (sweep {5e-5, 1e-4, 2e-4}; UNVERIFIED general LoRA practice, no GUI source), cosine, warmup ~5–10%, weight decay 0.01; effective batch 4–8; 3–10 epochs chosen by leave-one-demo-out step accuracy (with 15–75 samples, one epoch is only a few optimizer steps — think in total steps, ~50–200); bf16; `max_pixels` identical between training and inference (e.g., 16384*28*28 → 1288×812 for 1280×800, or set explicit `resized_height/width`).
- **Forgetting probe (cheap and convincing for a workshop paper):** before/after each adapter, evaluate (a) a fixed set of ~50–100 ScreenSpot-v2 web / Mind2Web-style single-step grounding items, and (b) tasks in the *same* mock app that the user never demonstrated. Report Δ vs base; also log per-token KL to base on the new task (RL's Razor predicts forgetting).
- **Mitigations to try if forgetting shows up:** mix a small replay set (e.g., 1:1 with base-model self-generated correct steps or public web grounding samples), lower rank (8), fewer epochs, or an RFT stage instead of more SFT. Per-user adapters are hot-swappable, so the base is never damaged — but the *adapter-on* model is what the user runs, so forgetting inside an adapter still matters.
- Train in the base model's native format; any format change consumes capacity that tiny data cannot spare and increases drift from base.

### Gaps
- No GUI-agent paper found that compares LoRA ranks/target modules or LoRA vs full FT for forgetting.
- No published LoRA hyperparameters specifically for UI-TARS-1.5-7B; community fine-tune configs (e.g., Fintor-GUI-S2 `args.json`) were not inspected.
- Whether freezing vs training the merger helps grounding transfer at tiny scale is untested.

---

## Q6. Reasoning / thought annotations: do (post-hoc) thoughts help over action-only SFT in low-data settings?

### Takeaway
For multi-step agent SFT at scale, thoughts help (Aguvis inner monologue; OpenCUA reflective L2 CoT; UI-TARS/Fara train on thoughts), and post-hoc thought synthesis conditioned on the ground-truth action is the standard practice. For grounding and small-data RL, thinking is unnecessary or harmful (UI-R1-E, GUI-G1, UIShift). No study isolates thought vs action-only for ≤100-sample multi-step SFT.

### Cited Findings
- **Aguvis ablation (7B):** removing inner monologue: ScreenSpot 84.4→79.3; Mind2Web 60.4/54.6/56.6→55.4/53.7/54.9; AndroidControl High 61.5→60.3, **Low 80.5→69.1**; authors conclude monologue helps precise low-level execution most. — [Aguvis paper](https://arxiv.org/html/2412.04454v2)
- **OpenCUA ablation (Qwen2-VL-7B):** reflective long CoT raised OSWorld 11.5→15.3; training on a mix of CoT levels 18.5 vs L2-only 13.1; at inference L2 18.5 > L3 17.6 > L1 16.9. — [OpenCUA paper](https://arxiv.org/html/2508.09123)
- **UI-TARS:** thoughts are synthesized post hoc (ActRe, conditioned on GT action) and refined by bootstrapping (keep sampled thoughts whose predicted action matches GT); ActRe thoughts can be superficial; action-only traces are also mixed in. — [UI-TARS paper](https://arxiv.org/html/2501.12326)
- **Fara1.5:** SFT loss on thought + action tokens; Fara-7B outputs CoT then tool call. — [Fara1.5 article](https://www.microsoft.com/en-us/research/articles/fara1-5-computer-use-agent); [Fara-7B card](https://huggingface.co/microsoft/Fara-7B)
- **UI-R1:** prompting with reasoning beat direct action by ~6% (App. B.1), yet the no-think fast-grounding variant UI-R1-E-3B (89.5 / 33.5 SS-V2 / SS-Pro) beat the long-think UI-R1-3B (85.4 / 17.8); authors: reasoning not essential for grounding. — [UI-R1 paper](https://arxiv.org/html/2503.21620)
- **GUI-G1:** longer thinking before grounding hurts; proposes a "fast thinking" template. — [GUI-G1 abstract](https://arxiv.org/abs/2505.15810)
- **UIShift:** reasoning vs no-reasoning nearly equal (7B ScreenSpot 86.6 vs 86.9; AC-Low SR 83.1 vs 79.8; 3B AC-Low 82.1 vs 85.4); no-reasoning cut 7B training time 19 h→13 h. — [UIShift paper](https://arxiv.org/html/2505.12493v2)
- **UI-TARS-1.5 prompt instruction:** "Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part." — [prompt.py](https://raw.githubusercontent.com/bytedance/UI-TARS/main/codes/ui_tars/prompt.py)

### Inferences
- **Recommendation:** with UI-TARS-1.5-7B, keep the `Thought:` field (it is the native format; dropping it is itself a distribution shift), and fill it with **short post-hoc thoughts** (1–3 sentences: what the user wants here + which element + why — e.g., the user's preference) generated by a strong VLM given the screenshot, task, history and the GT action (ActRe), then filtered by checking that the base model, conditioned on that thought, predicts the GT action (lightweight bootstrapping). Personalization signals (e.g., "this user always picks express shipping") naturally live in thoughts, which is a reason to keep them.
- Run a cheap ablation: thought-SFT vs action-only SFT (using the GROUNDING-style `Action:`-only output) on the same demos. Literature predicts thoughts matter more for multi-step/low-level execution and little for pure grounding.
- If doing RFT, follow UI-R1-E/GUI-G1: short or no thinking, to avoid length blow-up and degraded grounding.
- Risk: post-hoc thoughts can "rationalize" and leak future information (e.g., mention elements not yet visible) — instruct the annotator model to use only the current screenshot + history.

### Gaps
- No controlled study of thought vs action-only SFT at ≤100 multi-step samples; no study of how thought quality (strong vs weak annotator) matters at tiny scale.
- UI-TARS's own thought-vs-no-thought ablation numbers (Sec. 5.5) were not reachable in the fetched text.

---

## Q7. Synthesis for this project: base model, action/output format, hyperparameters, failure modes

### Takeaway
Use **UI-TARS-1.5-7B** as the primary base (Apache-2.0, Qwen2.5-VL tooling, strong grounding, native Thought/Action), keep **Qwen2.5-VL-7B-Instruct** as the general-VLM ablation, train LoRA on LLM linear layers only in the model's exact native format and coordinate space, and include an in-context-demo baseline and a forgetting probe. A same-data SFT vs RFT comparison is a high-value, low-cost addition because DOM bboxes from the mock apps give a free verifiable reward.

### Cited Findings
- UI-TARS-1.5-7B: Apache-2.0, qwen2_5_vl, OSWorld 27.5, ScreenSpot-Pro 49.6 vs Qwen2.5-VL-7B 29.0. — [UI-TARS-1.5-7B card](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B); [Qwen2.5-VL-7B card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Absolute coords on smart-resized image; factor 28, max_pixels 16384·28·28; history_n 5 in reference harness. — [README_coordinates.md](https://raw.githubusercontent.com/bytedance/UI-TARS/main/README_coordinates.md); [OSWorld uitars15_v1.py](https://raw.githubusercontent.com/xlang-ai/OSWorld/main/mm_agents/uitars15_v1.py)
- Fara-7B (same backbone) is the stronger 7B web agent: WebVoyager 73.5 / Online-M2W 34.1 vs UI-TARS-1.5-7B 66.4 / 31.3; ≈16 vs ≈41 steps. — [Fara-7B card](https://huggingface.co/microsoft/Fara-7B); [MSR blog](https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/)
- One in-context demo: Gemini-1.5-Pro 19.3→51.7; UI-TARS-7B-SFT 18.1→32.8. — [LearnAct](https://arxiv.org/abs/2504.13805)
- GRPO on 136 samples helps; SFT forgets more than GRPO. — [UI-R1](https://arxiv.org/html/2503.21620); [CGL](https://arxiv.org/html/2603.02951)

### Inferences
**Base model**
- Primary: UI-TARS-1.5-7B. Ablation: Qwen2.5-VL-7B-Instruct (same arch → isolates the effect of GUI pretraining). Optional: Fara-7B (web specialist, same arch, MIT). Mention Fara1.5-4B/9B and UI-Venus-2-9B (Qwen3.5) as the late-2026 frontier but do not switch architectures with 2 weeks left.

**Action / output format**
- UI-TARS: `Thought: <1–3 sentences>\nAction: click(start_box='(x,y)')` (or the `<point>` variant — pick whichever the checkpoint emits zero-shot), `type(content='...\n')`, `scroll(...)`, `hotkey(...)`, `wait()`, `finished(content='...')`; coordinates = absolute in smart-resized space (1288×812 for 1280×800 at default max_pixels). For Qwen2.5-VL-7B-Instruct, reuse the *same* textual action grammar so both models are trained on identical targets.
- Store raw demos in viewport pixels + DOM bbox + element id; render targets per model at data-build time (Qwen2.5-VL family → absolute resized px; Qwen3-VL/3.5 → 0–1000 normalized).
- History: ≤5 recent screenshots (or 3 for memory, as OpenCUA/Fara1.5) + full text of previous Thought/Action; loss on the current step's assistant turn only.

**Hyperparameters (starting point; sweep LR first)**
- LoRA r=16, α=32, dropout 0.05, all LLM linear layers; ViT + merger frozen; LR 1e-4 (sweep 5e-5–2e-4); cosine, warmup 5–10%; effective batch 4–8; ~50–200 optimizer steps total (≈3–10 epochs) with leave-one-demo-out early stopping; bf16; identical max_pixels at train/inference; greedy decoding (temp 0) at eval.
- RFT ablation: GRPO, 8 rollouts, LR ~1e-6 (full-FT values from UI-R1/UIShift; with LoRA likely needs ~1e-5–1e-4, UNVERIFIED), KL β ≈ 0.04, reward = action type + click-in-DOM-bbox + text match + format.

**Anticipated failure modes**
1. Coordinate-space mismatch (resize at train ≠ inference; 999 vs 1000 normalization; raw viewport px used as targets) → systematic click offsets.
2. Prompt/format drift (start_box vs `<point>`; system prompt differs from the one UI-TARS was trained with) → parse failures; keep a robust parser and count format errors as a metric.
3. Memorization of absolute positions: adapter clicks where the button *was* in the demo; test with layout perturbations (moved/reordered elements, different window size) in the mock apps.
4. Over-personalization / negative transfer: adapter applies the user's preference to tasks where it doesn't apply; include counterfactual tasks.
5. Off-distribution forgetting: drops on held-out grounding/other apps (measure; mitigate with replay or RFT).
6. Long-horizon looping/termination errors (UI-TARS-1.5-7B averages many steps): cap steps, check `finished()` precision.
7. Post-hoc thought leakage/rationalization — thoughts that reference not-yet-visible info.
8. Tooling: transformers-version incompatibilities loading UI-TARS-1.5 (seen in the wild); OpenCUA's non-standard RoPE/template if attempted.
9. Evaluation noise: with 3–5 demos per user, use leave-one-demo-out and report per-step accuracy + task success with confidence intervals across users.

### Gaps
- No published per-user LoRA personalization result for GUI agents to benchmark against; the closest are CoAdapt-GUI (LoRA test-time adaptation per app) and LearnAct (in-context demos).
- LoRA-specific RFT learning rates for Qwen2.5-VL GUI agents were not found.
