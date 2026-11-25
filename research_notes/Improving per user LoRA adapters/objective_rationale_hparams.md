# Training objective, rationale format, history format, LoRA hyperparameters, and bf16 stability for few-shot per-user LoRA on UI-TARS-1.5-7B (mlx-vlm)

Scope note: findings are from web sources fetched on 2026-11-21, plus a direct read of the installed library source in this repo's `.venv` (mlx-vlm 0.8.0, mlx 0.32.4). Items marked **UNVERIFIED** come from prior knowledge or from search snippets and were not confirmed against the primary text in this session. Several 2026 arXiv papers turned up. They are cited from their abstract pages only, so exact formulas and ablations are not confirmed.

## Q1. Loss weighting: should we reweight decision tokens (coordinates, action type, first typed word), use focal loss, or train on actions only?

### Takeaway
A mean completion loss of 0.02 tells us little, because about 90% of completion tokens are boilerplate. Several 2025-2026 GUI and tool-use papers upweight exactly the tokens that decide the action: coordinate digits, digit significance, and the first token of a tool name. None of them reports a large, clean SFT-only ablation that we could copy directly. The cheapest well-supported move is a fixed weight mask on decision spans (weight about 3-5x), with digit-significance weights on the coordinates. Avoid DFT-style p-weighting: it shrinks the gradient on exactly the low-probability decision tokens we want to fix.

### Cited Findings
- **TL-Training** (Junjie Ye et al., arXiv 2412.15495, Findings of EMNLP 2025). The paper argues that standard SFT "treats all tokens as equally important". It splits tool-call tokens into key and non-key sets and raises the loss weight on key tokens: the first token of the tool name, plus tokens that share prefixes with other tools. It also blocks gradients from erroneous trajectories and adds a PPO stage. It reports strong tool-use results for CodeLLaMA-2-7B trained on only 1,217 examples. — [arXiv 2412.15495](https://arxiv.org/abs/2412.15495v1); [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.15)
- **Perplexity-Weighted Cross-Entropy** (Ziwei Wang, Junjie Zheng, ... Jiajun Bu, "Towards Scalable Lightweight GUI Agents via Multi-role Orchestration", arXiv 2604.13488, Findings of ACL 2026). A search snippet says the loss gives larger weights to high-perplexity tokens, with coordinate tokens as the main example. The abstract confirms only that "Perplexity-Weighted Cross-Entropy optimization" is used in stage 1. The formula and ablation are **UNVERIFIED**. — [arXiv abs](https://arxiv.org/abs/2604.13488); [HTML](https://arxiv.org/html/2604.13488v1)
- **GUI-SD** (Yan Zhang, Daiqing Wu, Huawen Shen, Can Ma, Yu Zhou, "Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding", arXiv 2605.00642, May 2026). The abstract says it "adaptively weights tokens based on digit significance and teacher confidence". A search snippet says higher-order coordinate digits get more weight and entropy is used to filter unreliable supervision. The formula and the uniform-vs-weighted ablation are **UNVERIFIED**. The method is self-distillation, not plain SFT. — [arXiv 2605.00642](https://arxiv.org/abs/2605.00642)
- **Quality-aware self-distillation for GUI grounding** (arXiv 2606.18101). A search snippet describes a weighted reverse-KL objective that keeps ordinary tokens at weight 1 and scales coordinate tokens by a fixed factor. **UNVERIFIED** beyond the snippet. — [arXiv PDF](https://arxiv.org/pdf/2606.18101)
- **Focused Omni-Modal GUI Agents** (arXiv 2610.26203). Per a search snippet, the loss is computed only over the assistant's JSON action tokens. This is the "action-only loss" design. **UNVERIFIED** detail. — [arXiv PDF](https://arxiv.org/pdf/2610.26203)
- **GUI-Actor** (Qianhui Wu et al., arXiv 2506.03143, NeurIPS 2025) avoids coordinate-token generation altogether. It uses an attention-based action head trained with NTP loss plus an action-attention loss. It is too invasive for our setup, but it shows that coordinate tokens are a known weak point of text-coordinate SFT. — [NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/16130af940e9dabb43c726119bd3b42e-Paper-Conference.pdf)
- **DFT, Dynamic Fine-Tuning** (Yongliang Wu, Yizhou Zhou et al., "On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification", arXiv 2508.05629, ICLR 2026 per v3). DFT multiplies each token's loss by its own (detached) predicted probability, a one-line change. It reports gains over SFT on math, code, and multimodal tasks. Because it multiplies by p, low-probability tokens are down-weighted. — [arXiv 2508.05629](https://arxiv.org/html/2508.05629v2); [HF papers](https://huggingface.co/papers/2508.05629)
- **Rho-1: "Not All Tokens Are What You Need"** (Zhenghao Lin et al., arXiv 2404.07965, NeurIPS 2024). Selective language modeling trains only on tokens with high excess loss relative to a reference model. **UNVERIFIED** (prior knowledge, not re-fetched). — [arXiv 2404.07965](https://arxiv.org/abs/2404.07965)
- **Number Token Loss** (Jonas Zausinger et al., "Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models", arXiv 2411.02083). It adds a regression-like loss over digit-token distributions so that near-miss digits are penalized less than far-off ones. This directly suits coordinate digits. **UNVERIFIED** (prior knowledge, not re-fetched). — [arXiv 2411.02083](https://arxiv.org/abs/2411.02083)
- **Critical tokens** (Zicheng Lin et al., "Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability", arXiv 2411.19943). A few "critical tokens" decide whether a trajectory is correct, and the paper upweights or penalizes them at token level (cDPO). **UNVERIFIED** (prior knowledge). — [arXiv 2411.19943](https://arxiv.org/abs/2411.19943)
- **Thought vs. no-thought evidence relevant to action-only loss:**
  - **Aguvis** (Yiheng Xu et al., arXiv 2412.04454) Table 6, Aguvis-7B. Removing inner monologue drops AndroidControl low-level from 80.5 to 69.1, high-level from 61.5 to 60.3, ScreenSpot from 84.4 to 79.3, and MM-Mind2Web cross-task from 60.4 to 55.4. — [arXiv HTML](https://arxiv.org/html/2412.04454)
  - **UI-R1** (Zhengxi Lu et al., arXiv 2503.21620). "Agents producing reasoning processes outperform those directly outputting action predictions by approximately 6%" on action prediction, but "reasoning is not essential for simpler tasks, such as GUI grounding". UI-R1-E-3B (DAST then NOTHINK, 4 epochs each, about 2K samples) scores 89.2 on ScreenSpot vs 83.3, and 33.5 on ScreenSpot-Pro vs 17.8, compared with the thinking UI-R1-3B trained on 136 samples. — [arXiv HTML v5](https://arxiv.org/html/2503.21620v5)
  - **GUI-G1** (Yuqi Zhou, Sunhao Dai, Shuai Wang, Kaiwen Zhou, Qinglin Jia, Jun Xu, arXiv 2505.15810). "Longer chains unexpectedly lead to worse grounding performance." The paper proposes a Fast Thinking Template and a box-size constraint. GUI-G1-3B (17K samples) reaches 90.3% on ScreenSpot and 37.1% on ScreenSpot-Pro. — [arXiv 2505.15810](https://arxiv.org/abs/2505.15810)
  - **UI-TARS** (Yujia Qin et al., arXiv 2501.12326). In-domain at Best-of-1, System-2 (with thought) is "slightly worse" than System-1. It shows "a clear advantage" at Best-of-16 and Best-of-64, and "significantly outperforms" System-1 out of domain (AndroidWorld). The authors' explanation: with one sample "the model may fixate on a flawed reasoning path." — [arXiv HTML](https://arxiv.org/html/2501.12326)
- Local check of our tokenization assumption: Qwen2/2.5 tokenizers split numbers into single-digit tokens, so "(523,311)" becomes per-digit tokens and the first digit is the hundreds place. **UNVERIFIED** in this session (prior knowledge). Check it with `tokenizer.tokenize("(523,311)")`.

### Inferences
- Our symptom (mean loss 0.02, decision tokens above 2 nats) is what you expect when about 35 of 40 completion tokens are deterministic template text. The mean is dominated by tokens with no information. The useful metrics are NLL on decision tokens only and teacher-forced step accuracy (whether the argmax coordinate lands inside the target element's box).
- Do not use DFT for this problem. Its gradient on a token is about p·∇log p. For a decision token at 2 nats (p ≈ 0.13), that is roughly 7.5x weaker than plain CE relative to boilerplate tokens at p ≈ 1, the opposite of what we need. Focal-style weighting, (1-p)^γ with γ = 1-2, or a fixed span mask both go the right way.
- Pure action-only loss (masking the thought) is reasonable only while our thoughts are pure restatements, since those carry no extra signal. Once thoughts carry preference and progress content (Q2), keep them in the loss. Aguvis, UI-R1, and OpenCUA all show reasoning helps action prediction, though not grounding. Our grounding is already handled by the base model.
- Our "first coordinate digit" and "first typed word" tokens are the analogue of TL-Training's "first token of tool name". Its 1,217-example result is the closest low-data evidence that key-token weighting helps.

### Gaps
- I found no GUI-agent paper with a clean SFT-only ablation of "upweight coordinate/action tokens vs uniform CE" with numbers. The 2604.13488 and 2605.00642 formulas and ablations could not be read from the abstract pages.
- I found no direct evidence for focal loss in LLM or agent SFT. Its use here is an extrapolation.

## Q2. Rationale-augmented SFT: do preference-stating and progress-stating thoughts beat template thoughts, and do they fix skipped user-specific steps?

### Takeaway
The strongest evidence says the content of a thought matters more than its presence. UI-TARS explicitly warns that thoughts written after the fact from the ground-truth action (ActRe) can match the action "only superficially", which describes our template restatements. Aguvis and OpenCUA show gains from predictive or reflective thoughts. GUI-G1 and UI-R1 warn that long reasoning hurts grounding. So the recommendation is short thoughts (1-2 sentences) that state observable state, the user's preference, and why the action follows, not long chain-of-thought.

### Cited Findings
- **UI-TARS ActRe and thought bootstrapping** (arXiv 2501.12326). ActRe generates a thought from the prior context and the ground-truth action, which "can produce false positives" where the thought matches the action superficially. Thought bootstrapping instead samples thought-action pairs from an early checkpoint and keeps a thought whose action matches the ground truth, generating thoughts "without prior knowledge of the ground-truth action". Target thought patterns are task decomposition, long-term consistency, milestone recognition, trial and error, and reflection. The training data also included "vanilla action traces without thoughts." — [arXiv HTML](https://arxiv.org/html/2501.12326)
- **UI-TARS reflection tuning.** Error-correction and post-reflection pairs are trained with DPO. OSWorld (15 steps) goes from 17.7 to 18.7 (7B) and 18.8 to 22.7 (72B) from SFT to DPO. — [arXiv HTML](https://arxiv.org/html/2501.12326)
- **Aguvis inner monologue.** GPT-4o writes "Thought" plus "Low-level Instruction" from the goal, previous actions, and a screenshot with the target highlighted. The prompt tells it to "avoid using the labeled action or the highlighted bounding box as reasoning support" so that the reasoning is predictive. Ablation numbers are in Q1 (AndroidControl low-level 80.5 vs 69.1 without it). — [arXiv HTML](https://arxiv.org/html/2412.04454)
- **OpenCUA** (Xinyuan Wang et al., arXiv 2508.09123). Reflective long CoT raises OSWorld from 11.5 to 15.3 over short CoT, which the authors attribute to "improved self-correction capability". By inference format, L2 (reflective thought) scores 18.5, L1 (action only) 16.9, and L3 (observation + thought) 17.6. Training on a mixture of L1/L2/L3 scores 18.5 vs 13.1 for L2-only training. A Claude-based reflector checks each step "for correctness and redundancy by comparing screenshots before and after the action" and drops incorrect or redundant steps from training. — [arXiv HTML v3](https://arxiv.org/html/2508.09123v3)
- **GUI-Rise** (Tao Liu, Chongyu Wang, Rongjie Li, Yingchen Yu, Xuming He, Bai Song, arXiv 2510.27210, NeurIPS 2025 proceedings). The output is structured as `<Progress Estimation>`, `<Decision Reasoning>`, `<Action>`, and `<Memory Summary>`, with pseudo-labels from GPT-4o-mini conditioned on the ground-truth action. AITW overall success rate: baseline 67.2, structured CoT via SFT only 42.6, two-stage training + structured CoT 69.8, plus history summary 70.7, full model with a summary reward 71.1. Structured CoT via SFT alone hurt badly; the gains needed the two-stage and RL pieces. — [arXiv HTML](https://arxiv.org/html/2510.27210); [NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/f144ab9985c739a5091ec188a2688644-Paper-Conference.pdf)
- **Mobile-Agent-v2** (Junyang Wang, Haiyang Xu, ... Jitao Sang, arXiv 2406.01014). A planning agent writes "task progress", defined as "the sub-tasks that have already been completed", in order to "summarize the history operations and track task progress". In Table 4 the full system scores SR 88.6 (basic) and 61.4 (advanced). The weakest ablated configuration scores 59.1 and 29.5, and the text says removing the planning agent hurts most. The mapping of rows to removed components is **UNVERIFIED** because the checkmarks did not survive extraction. This is a prompting (multi-agent) system, not SFT. — [arXiv HTML](https://arxiv.org/html/2406.01014)
- **Android in the Zoo / Chain-of-Action-Thought (CoAT)** (Jiwen Zhang et al., arXiv 2403.02713). The thought contains a screen description, action thinking, the next action description, and the action result. **UNVERIFIED** (prior knowledge). — [arXiv 2403.02713](https://arxiv.org/abs/2403.02713)
- **Personalization:** I found no paper that tests preference-stating thoughts (for example, "Alice always books 30 minutes") in GUI-agent SFT.

### Inferences
- Problem 2 (the adapter keeps a default instead of clicking it) is a preference decision. The template thought "Next, I'll click the Start time field" gives no reason to deviate from the default, so the model's prior that a pre-filled field can be skipped wins. A thought of the form "Duration shows 60 min; Alice always books 30 min, so I'll change it" puts the reason in tokens right before the action. The decision span is then the field name plus "change", which Q1's weighting can target.
- To avoid the ActRe "superficial match" trap, the rationale should cite visible evidence: the default value on screen, which fields are already filled, and the persona rule. A rationale that merely restates the action does not do that. We can produce these deterministically from the persona file and the per-step UI-state diff in our recorder, with no LLM needed.
- Add contrastive "no-change" steps, where the default already matches the preference and the thought says so and moves on. Otherwise "always change the field" becomes the shortcut.
- Keep thoughts short, about 15-30 tokens. GUI-G1 and UI-R1-E show that long reasoning hurts grounding. GUI-Rise shows structured CoT via SFT alone can collapse performance (42.6 vs 67.2 baseline). OpenCUA shows mixing formats in training helps.

### Gaps
- No found study measures whether preference rationales improve per-user personalization. This is new for our project and should be tested directly (see Q6).
- The GUI-Rise cold-start set size is not reported. The HAR paper (AAAI 2026) could not be fetched.

## Q3. History and state tracking: how many screenshots, text-only history, previous thoughts, and do progress summaries fix "where am I" repetition?

### Takeaway
UI-TARS was trained and evaluated with the last N = 5 screenshots plus the full text history of previous thoughts and actions. OpenCUA found that 3 screenshots with action-only (L1) text history is the best trade-off, and that including previous thoughts (L2 history) hurt. Explicit progress state (completed subtasks, current subgoal, whether the last action took effect) helps in prompting systems (Mobile-Agent-v2; Task-State Representation, up to +12 points). Our "repeat the previous step" error is a known failure mode. One named cause, "stale-screen repetition" (the screenshot was taken before the UI updated), is worth checking in our recorder or harness.

### Cited Findings
- **UI-TARS.** The model sees only the last N observations to fit a 32k budget, while "the full history of previous actions and thoughts is retained as short-term memory". "We set the N in Eq. 3 to 5" for the evaluations. — [arXiv HTML](https://arxiv.org/html/2501.12326)
- **OpenCUA history ablations** (Fig. 10; values read approximately from a garbled figure). OSWorld success rate with L1 history: about 6.5% with 1 image, 9.6% with 3, and 9.9% with 5. Going from 3 to 5 gives "marginal gains while incurring ~3K more context tokens". At 3 images, L1 action history scores about 9.6 vs about 7.5 for L2 thought history, and "L2 history offers no benefit and may introduce hallucinations that distract attention". The default is L1 history plus 3 images. — [arXiv HTML v3](https://arxiv.org/html/2508.09123v3)
- **Aguvis.** History is text only: "a stack of all previous low-level instructions", plus the single current screenshot. — [arXiv HTML](https://arxiv.org/html/2412.04454)
- **Task-State Representation for Long-Horizon Mobile GUI Agents** (Yujie Zheng, Zikang Liu, Xin Zhao, Ji-Rong Wen, arXiv 2607.00502, July 2026). It names two failure modes: "progress hallucination", where the actor "fabricates past states", and "stale-screen repetition", where it "misinterprets a delayed update in the user interface as a failed action and enters a localized loop". Its JSON state has `task_summary`, `completed_progress`, `current_subgoal`, `remaining_requirements`, `action_effective`, `last_step_result`, and `next_action_focus`. On MobileWorld, Qwen3.5-plus goes from 43 to 55 and Kimi-k2.5 from 49 to 58. But Qwen3.5-plus on AndroidWorld goes from 61.21 down to 57.76. Ablation on MobileWorld: full 55, without summary 50, without tracker 49, without transition focus 48. Repeated actions are not quantified. This is externally maintained context, not training. — [arXiv HTML](https://arxiv.org/html/2607.00502)
- **History-Aware Reasoning for GUI Agents (HAR-GUI-3B, AAAI 2026).** Per a search snippet, agents that treat screens in isolation suffer "short-term memory loss", which leads them to repeat actions or miss task completion. HAR uses an SFT warm-up with teacher-written goal-oriented summaries of previous actions (Act2Sum), then reflective RL with a memory-augmented reward. Numbers are **UNVERIFIED**; the paper page was blocked by a bot filter. — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/40966/44927)
- **GUI-Rise.** A compact history summary carried between steps gives a +0.9 SR gain on AITW, on top of two-stage training plus structured CoT. — [arXiv HTML](https://arxiv.org/html/2510.27210)
- **Mobile-Agent-v2.** The text progress record shortens context and helps the decision agent pick the next step. Numbers are in Q2. — [arXiv HTML](https://arxiv.org/html/2406.01014)

### Inferences
- UI-TARS-1.5's native format has multiple screenshots and prior thought/action turns in context. If our harness feeds 1 screenshot plus text history, the LoRA has to bridge a format shift with about 70 samples. Matching the native format, or adding 1-2 prior screenshots, is a cheap experiment. Two to three images at about 1-1.5K visual tokens each fit our 8192 max sequence length. **UNVERIFIED** token count; measure it.
- Following OpenCUA, a good history is the prior actions as L1 text plus one compact progress line ("Done: title, date. Next: duration."), not all prior thoughts. Putting progress in the current thought (Q2) and keeping the history action-only fits both the OpenCUA and Mobile-Agent-v2 evidence.
- Before changing the model, check for stale screenshots. If the screenshot after step t-1 is captured before the UI re-renders, the model sees an "unchanged" screen and rationally repeats the action. Add a settle delay, or a check that the screenshot changed, in the recorder and harness.

### Gaps
- No found SFT ablation isolates "progress-aware thoughts" vs "template thoughts" for repetition rate. The evidence is from prompting systems (Mobile-Agent-v2, Task-State) or mixed SFT+RL (GUI-Rise, HAR).
- I could not confirm the exact UI-TARS-1.5 inference history format (images vs text per previous turn) from the paper text. It should be checked against the official UI-TARS / UI-TARS-desktop prompt code.

## Q4. LoRA variants and hyperparameters for tiny data (DoRA, LoRA+, PiSSA, rsLoRA, VeRA; rank, target modules, epochs, early stopping)

### Takeaway
Multiple 2025-2026 re-evaluations find that variant gains mostly disappear once learning rate is tuned (all methods within 1-2%; DoRA shows no supported gain). Keep vanilla LoRA. Make sure MLP layers are adapted, since attention-only underperforms. Treat learning rate as the main knob, using the effective LR lr × α/r. Rank 16 has far more capacity than about 70 samples need. Use held-out step accuracy, not training loss, for early stopping.

### Cited Findings
- **"LoRA Without Regret"** (John Schulman and Thinking Machines Lab, blog, 2025-09-29):
  - The optimal LoRA LR is "consistently 10x" the full-fine-tune LR (fitted multiplier 9.8), and about 15x for short runs under about 100 steps.
  - Attention-only LoRA underperforms MLP-only LoRA even at matched parameter count (rank-256 attention-only, 0.25B params, loses to rank-128 MLP-only, 0.24B). Adding attention on top of MLP gives no further gain. Recommendation: apply LoRA to all layers, especially MLP.
  - Optimal LR changes by less than 2x between rank 4 and rank 512, under α/r scaling with α = 32.
  - LoRA is less tolerant of large batches.
  - High-rank LoRA matches full fine-tuning on small-to-medium datasets.
  - [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/)
- **"Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning"** (Yu-Ang Lee, Ching-Yun Ko, Pin-Yu Chen, Mi-Yen Yeh, arXiv 2602.04998, Feb/May 2026). Nine LoRA variants were re-tuned over LR, batch size, rank, and duration. Once LR is tuned, "all methods achieve similar peak performance" (within 1-2%), and vanilla LoRA "remains a competitive baseline". Different methods prefer different LR ranges, which the authors link to the top Hessian eigenvalue. — [arXiv 2602.04998](https://arxiv.org/abs/2602.04998)
- **Paired multi-seed PEFT audit** (AACL 2026 short paper, 2026.aacl-short.32). Per a search snippet: DoRA shows no supported gain in any of 7 audited cells, PiSSA's GSM8K advantage reverses under paper-faithful hyperparameters, and rsLoRA/DoRA/PiSSA behavior tracks the effective LR lr × α/√r. **UNVERIFIED** beyond the snippet. — [ACL Anthology PDF](https://aclanthology.org/2026.aacl-short.32.pdf)
- **LoRA-Pro** (arXiv 2407.18242), GLUE subset with T5-base. Averages: PiSSA 84.71, rsLoRA 79.64, LoRA+ 84.95, DoRA 82.57. Overfitting is "particularly pronounced on small datasets, such as MRPC". These come from a search snippet. — [arXiv PDF](https://arxiv.org/pdf/2407.18242)
- **FinLoRA** (arXiv 2505.19819). Vanilla LoRA beats the variants on general financial tasks and DoRA does worst, possibly because its magnitude vector shares the low-rank LR. These come from a search snippet. — [arXiv PDF](https://arxiv.org/pdf/2505.19819)
- **Variant definitions.** All of these are **UNVERIFIED** (prior knowledge, not re-fetched).
  - DoRA (Shih-Yang Liu et al., arXiv 2402.09353, ICML 2024) decomposes weights into magnitude and direction. — [arXiv](https://arxiv.org/abs/2402.09353)
  - LoRA+ (Soufiane Hayou, Nikhil Ghosh, Bin Yu, arXiv 2402.12354, ICML 2024) sets lr_B = λ·lr_A with λ ≈ 16 suggested. — [arXiv](https://arxiv.org/abs/2402.12354)
  - PiSSA (Fanxu Meng, Zhaohui Wang, Muhan Zhang, arXiv 2404.02948, NeurIPS 2024) initializes A and B from the top singular vectors and freezes the residual. — [arXiv](https://arxiv.org/abs/2404.02948)
  - rsLoRA (Damjan Kalajdzievski, arXiv 2312.03732) scales by α/√r. — [arXiv](https://arxiv.org/abs/2312.03732)
  - VeRA (Dawid Kopiczko, Tijmen Blankevoort, Yuki Asano, arXiv 2310.11454, ICLR 2024) uses shared frozen random matrices with trainable scaling vectors. — [arXiv](https://arxiv.org/abs/2310.11454)
  - "LoRA Learns Less and Forgets Less" (Dan Biderman et al., arXiv 2405.09673, TMLR 2024). — [arXiv](https://arxiv.org/abs/2405.09673)
- **Our installed mlx-vlm 0.8.0** (local source read):
  - The non-legacy path computes `scale = alpha / rank`, so rank 16 with α 16 gives scale 1.0 ([peft/utils.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/peft/utils.py), `_lora_scale`).
  - `LoRALinear` initializes A ~ U(±1/√d_in) and B = 0, which matches the standard PEFT init ([peft/lora_layers.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/peft/lora_layers.py)).
  - The legacy `LoRaLayer` instead initializes A ~ U(±1/√rank), about 15x larger for d_in = 3584 and r = 16. It is used only with `legacy=True` ([peft/lora.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/peft/lora.py)).
  - By default, targets are all `nn.Linear` layers whose suffix appears in `find_all_linear_names(target)` within the adapter target, the language model, so attention and MLP are likely both covered.
  - DoRA is available (`peft/dora_layers.py`, `fine_tune_type="dora"`).
- **Reference recipes:**
  - Aguvis: lr 1e-5 (7B), batch 128, 1 epoch per stage, 3% warmup, cosine, max sequence 8192, ViT frozen, bf16. — [arXiv HTML](https://arxiv.org/html/2412.04454)
  - OpenCUA: lr 2e-5 to 2.5e-5, weight decay 0.1, batch 384-512; multi-image trajectory samples trained "for 3 epochs". — [arXiv HTML v3](https://arxiv.org/html/2508.09123v3)
  - UI-R1-E: 4 epochs per stage on about 2K samples (RL). — [arXiv HTML v5](https://arxiv.org/html/2503.21620v5)

### Inferences
- With full-fine-tune LRs for 7B VLM agent SFT of about 1e-5 to 2.5e-5 (Aguvis, OpenCUA) and LoRA ≈ 10-15x, the "regret-free" LoRA LR would be about 1e-4 to 3e-4 at large batch. Our divergence at 1e-4 with batch size 1 and very few supervised tokens per step points to an optimization-noise or precision issue (Q5), not too high an LR in principle. With batch 1 the safe LR is lower. Gradient accumulation of 4-8 would let us move toward 1e-4.
- Capacity is not the limit. About 70 samples × about 40 supervised tokens ≈ 3K tokens, far below rank-16 capacity per the TM blog. Lowering rank to 4-8 is a cheap regularizer but should change little. Raising rank is pointless.
- Restricting LoRA to attention only, or to the last N layers, goes against the TM evidence (MLP matters). Keep all linear layers. Do not adapt the vision tower. Confirm by printing the trainable parameter names.
- Variants: skip DoRA, PiSSA, and VeRA. Expected gain is at most 1-2% after LR tuning. PiSSA needs an SVD of the 7B weights and changes the base residual, which complicates serving per-user adapters on one shared base. LoRA+ (λ = 16) is the only cheap one to try if per-parameter-group LR is easy in mlx (separate LR for `lora_b`), but it is low priority.
- Epochs: with 70 samples, overfitting shows up as training loss near 0 while held-out step accuracy plateaus or drops. Evaluate after each epoch on held-out demos and keep the best checkpoint.

### Gaps
- No found study compares LoRA variants specifically in the 50-100 sample regime or for VLM GUI agents.
- I could not confirm which exact module names mlx-vlm targets for Qwen2.5-VL by default (for example, whether the vision merger is included). Print the trainable parameter names.

## Q5. bf16 training stability: why does lr 1e-4 diverge and lr 2e-4 give NaN, and why did mlx-vlm grad clipping give NaN?

### Takeaway
In our installed mlx-vlm 0.8.0, LoRA weights are created as fp32, and MLX Adam state takes the parameter's dtype, so the classic "bf16 weights swallow small updates" failure probably does not apply to the adapters. Confirm at runtime. The more likely culprits are high-variance batch-1 gradients on sequences with only about 40 supervised tokens, Adam's second moment lagging behind sudden large gradients (the documented loss-spike mechanism), and no protection against non-finite steps. MLX's `clip_grad_norm` is mathematically plain. The source shows no obvious NaN bug, but one non-finite element anywhere turns the whole clipped update into NaN.

### Cited Findings
- **Local source, mlx 0.32.4** ([optimizers.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx/optimizers/optimizers.py)):
  - Adam/AdamW `init_single` uses `mx.zeros_like(parameter)` for m and v, so the state dtype equals the parameter dtype. The LR is cast with `self.learning_rate.astype(gradient.dtype)`. The update is `m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g²`, with denominator `sqrt(v)*c2 + eps`.
  - `clip_grad_norm` upcasts only **float16** squares to fp32 ("Sum float16 squares in float32 so they do not overflow"), not bfloat16. It then computes `normalizer = min(max_norm / (total_norm + 1e-6), 1.0)` and multiplies every gradient by it.
  - If any gradient element is inf or NaN, `total_norm` is non-finite. The normalizer becomes 0 or NaN, and every gradient becomes NaN or 0·inf = NaN. Clipping therefore spreads a single bad element to every parameter.
- **Local source, mlx-vlm 0.8.0:**
  - `LoRALinear` creates `lora_a` with `mx.random.uniform` (fp32 default) and `lora_b` with `mx.zeros` (fp32). The forward pass casts the LoRA output to the activation dtype: `y + (self.scale * z).astype(x.dtype)`.
  - Grepping `trainer/common/model.py`, `engine.py`, and `runner.py` found no later cast to bf16.
  - Clipping is applied after accumulation via `optim.clip_grad_norm` ([common/utils.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/common/utils.py), `apply_accumulated_gradients`). The CLI config default is `grad_clip=None` ([common/config.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/common/config.py)), though `core.py` has a dataclass default of 1.0. The schedule is linear warmup then decay (`core.py`, lines 73-80).
  - [peft/lora_layers.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/peft/lora_layers.py)
- **Our training call** ([train/train_adapters.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/train/train_adapters.py)): batch size 1, `--train-on-completions`, `--max-seq-length 8192`, `--grad-checkpoint`, `--lora-dropout 0.05`, and grad clip off by default, with the comment "mlx-vlm 0.8 grad clipping produced NaN loss in our tests". `--iters` = epochs × number of rows.
- **Wortsman, Dettmers, Zettlemoyer, Morcos, Farhadi, Schmidt, "Stable and low-precision training for large-scale vision-language models"** (arXiv 2304.13013, NeurIPS 2023). Loss spikes "consistently occur 1-8 iterations after the squared gradients become under-estimated by their AdamW second moment estimator". StableAdamW (AdamW with AdaFactor-style update clipping, which lowers the LR when g²/v is large) "removes instabilities" and "outperforms gradient clipping". Lowering β2 as a mitigation is **UNVERIFIED** (prior knowledge of the paper body). — [arXiv 2304.13013](https://arxiv.org/abs/2304.13013); [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2023/file/20bd42d82998bc61732c00452228e814-Paper-Conference.pdf)
- **Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities"** (arXiv 2309.14322). It covers Adam epsilon effects, logit growth, and qk-layernorm. **UNVERIFIED** (prior knowledge). — [arXiv 2309.14322](https://arxiv.org/abs/2309.14322)
- **Zamirai et al., "Revisiting BFloat16 Training"** (arXiv 2010.06192). Pure-bf16 training loses small weight updates to round-to-nearest unless stochastic rounding or Kahan summation is used. **UNVERIFIED** (prior knowledge). It applies only if adapter weights are bf16. — [arXiv 2010.06192](https://arxiv.org/abs/2010.06192)
- **Reports of the same symptom elsewhere:**
  - bf16 LoRA/DoRA NaN reports in PEFT (Llama-3.1-8B): the gradient norm turns NaN in the first steps, which in PyTorch stacks was sometimes fixed by switching the attention implementation to eager. — [HF forum](https://discuss.huggingface.co/t/bf16-training-instability-with-llama-3-1-8b-lora-dora-peft/170326)
  - An Optimisers.jl issue notes that momentum state needs more precision than bf16. — [GitHub FluxML/Optimisers.jl #215](https://github.com/FluxML/Optimisers.jl/issues/215)
  - mlx #3362 is a float32 kernel crash on M5 Max MoE LoRA (workaround `MLX_ENABLE_TF32=0`), which shows precision paths matter on Apple GPUs. — [mlx issue mirror](https://upd.dev/ml-explore/mlx/issues/3362)
- I found **no** mlx or mlx-vlm GitHub issue reporting NaN from `--grad-clip` or bf16 LoRA NaN on Qwen2.5-VL. — [search result summary](https://www.mintlify.com/yocxy2/mlx-vlm/fine-tuning/lora)

### Inferences
- The likely divergence mechanism at lr 1e-4 (inference): each step has about 30-50 supervised tokens, most near zero loss. A step whose decision token is wrong (2+ nats) yields a gradient much larger than the running v, which was dominated by near-zero-loss steps. With β2 = 0.999 (mlx default, **UNVERIFIED** whether mlx-vlm overrides it), v adapts over about 1,000 steps but the run is only about 280 steps. g/√v then spikes: the Wortsman mechanism. Lower β2 (0.95-0.98), a longer warmup, gradient accumulation, and update clipping all target this mechanism.
- On the NaN with clipping, the source is plain. Two candidate explanations remain:
  - A step already produced a non-finite gradient element. Clipping then turns the whole update to NaN at once, while without clipping the damage is local but still fatal later.
  - Some trainable leaves are bf16 and their norm loses precision. Clipping is not the root cause.

  A custom step is the diagnostic: compute the norm in fp32, check `mx.isfinite(total_norm)`, and skip the update and log the sample id if it is not finite.
- Adam epsilon 1e-8 is fine if the state is fp32. If any state is bf16, 1e-8 is representable (bf16 has the fp32 exponent range), but v loses precision with only about 3 significant digits.
- The attention-implementation NaN class from PyTorch (SDPA/FA) may have an MLX analogue in `mx.fast.scaled_dot_product_attention` with long multimodal sequences in bf16. This is unverified; testing a single forward and backward pass in fp32 on the failing sample would isolate it.

### Gaps
- I could not confirm mlx-vlm 0.8.0's default AdamW betas and epsilon from the files grepped. Check where the optimizer is constructed (`optim.AdamW(...)` call) in the installed package.
- I found no public mlx-vlm issue matching our grad-clip NaN. If it reproduces, it may be worth filing.

## Q6. What concrete settings should we use, and how do we test them cheaply?

### Takeaway
Change the supervision content first (preference and progress thoughts, decision-span loss weights, stale-screen check), then stabilize the optimizer so higher LRs become usable. Keep vanilla LoRA. Judge everything by held-out decision-token NLL and teacher-forced step accuracy rather than mean loss. Each training run is about 280 steps × 5 s ≈ 23 min. A six-run ablation fits in about 2.5 hours overnight.

### Cited Findings
- Sources behind each setting are in Q1-Q5: TL-Training key tokens ([arXiv 2412.15495](https://arxiv.org/abs/2412.15495v1)); GUI-SD digit significance ([arXiv 2605.00642](https://arxiv.org/abs/2605.00642)); Aguvis predictive inner monologue ([arXiv 2412.04454](https://arxiv.org/html/2412.04454)); UI-TARS ActRe caveat and N = 5 history ([arXiv 2501.12326](https://arxiv.org/html/2501.12326)); OpenCUA 3 images with L1 history and mixed CoT training ([arXiv 2508.09123](https://arxiv.org/html/2508.09123v3)); Task-State "stale-screen repetition" ([arXiv 2607.00502](https://arxiv.org/html/2607.00502)); TM LoRA LR and MLP findings ([blog](https://thinkingmachines.ai/blog/lora/)); vanilla LoRA suffices ([arXiv 2602.04998](https://arxiv.org/abs/2602.04998)); Adam second-moment spikes ([arXiv 2304.13013](https://arxiv.org/abs/2304.13013)); local mlx and mlx-vlm source (Q4, Q5).

### Inferences (recommended settings for our setup)

**A. Loss**
- Pass a per-token weight mask through the batch and write a custom CE in mlx-vlm's loss path: `loss = Σ w_t·CE_t / Σ w_t`. Weights:
  - boilerplate tokens: 1
  - target-element name tokens in the thought: 3
  - action-type token: 3
  - coordinate digits: hundreds 4, tens 2, units 1 (digit significance)
  - first 1-2 tokens of typed text: 4
  - the preference phrase in the thought: 3
- Alternative arm: focal weighting with w_t = (1 - p_t)^2, detached, on all completion tokens.
- Do not use DFT (×p).

**B. Thought format** (about 15-30 tokens, generated deterministically from the persona file and the UI-state diff)
- Example: `Thought: Title and date are set. Duration shows 60 min; Alice always books 30 min, so I'll change it.`
- Include "no-change" counterexamples where the default already matches the preference.
- Keep about 20-30% of samples with the plain template, OpenCUA-style mixed formats, so the format does not become brittle.

**C. History**
- Current screenshot plus the previous 1-2 screenshots (test 1 vs 3 total).
- Text history: previous actions only (L1), with no previous thoughts. Progress lives in the current thought.
- Add a screenshot-settle check (wait until the frame changes or a 300-500 ms delay) in the recorder and agent harness to rule out stale-screen repetition.

**D. LoRA**
- Vanilla LoRA on all linear layers of the LLM (confirm MLP is included), with no vision-tower adapters.
- Rank 16, α 16 (scale 1.0). Optionally α 32 (scale 2) with LR halved, which is the same effective LR.
- Dropout 0.05.
- LR sweep {3e-5, 5e-5, 8e-5} at batch 1. After stabilization (E), try 1e-4 with gradient accumulation 4.
- Up to 6 epochs, with held-out evaluation after every epoch; keep the best epoch.
- Skip DoRA, PiSSA, and VeRA. LoRA+ (λ = 16) is optional and low priority.

**E. Stability**
- Assert that LoRA params and optimizer state are fp32 at startup: print `{k: v.dtype}` for trainable params.
- AdamW β2 = 0.98 (try 0.95), eps 1e-8, weight decay 0-0.01.
- Warmup about 10% of updates (about 28) with cosine decay.
- Replace mlx-vlm clipping with a custom step:
  1. Cast gradients to fp32.
  2. Compute the global norm.
  3. If the norm is not finite, skip the update and log the sample id.
  4. Otherwise clip to 1.0.
- Log the gradient norm every step.
- Gradient accumulation of 4 reduces batch-1 variance. It cuts updates to about 70 per run, so raise LR about 2x.

**Cheap test protocol**
1. **Metric harness first (about 30 min of work, minutes to run).** For held-out demos (leave-one-demo-out per persona/skill), compute with teacher forcing:
   - mean NLL on decision tokens only
   - greedy step accuracy, i.e. whether the parsed click lands inside the target element's box and the typed text matches
   - "default-kept" error rate on preference steps
   - repeat-previous-action rate

   One forward pass per sample takes about 1-2 s. This replaces rollouts for most ablations.
2. **Stability smoke test (about 5 min per arm).** Train 50 steps on 5 samples at lr 1e-4 with the custom fp32 clip and skip, at β2 0.999 vs 0.98. Watch the per-step gradient norm and loss for spikes and NaN. Also run one fp32 forward and backward pass on any sample that produced a non-finite gradient, to separate kernel precision issues from optimizer issues.
3. **Overnight ablation (6 runs × about 23 min ≈ 2.5 h; one persona/skill, fixed seed):**
   - (a) current baseline
   - (b) baseline plus weighted CE
   - (c) preference and progress thoughts
   - (d) (c) plus weighted CE
   - (e) (d) plus 3-image history
   - (f) (d) with action-only loss (thought masked)

   Then run 2 seeds of the best 2 runs.
4. **Final check.** Mock-workspace rollouts (3-5 tasks per persona) only for the top 2 configs, reporting success, default-kept errors, and repeated-action loops.

### Gaps
- None of the recommended weights (3-5x, digit 4/2/1, focal γ = 2) comes from a published ablation in our regime. They are starting points to sweep.
- How much the improved thoughts help depends on whether the base UI-TARS-1.5 prior (multi-image history, its own thought style) cooperates with a new thought style learned from about 70 samples. Only the ablation above can tell.
