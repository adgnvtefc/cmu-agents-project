# Contrastive and preference-optimization objectives for few-shot per-user GUI adapters (teaching WHICH valid action this user prefers, with other users' actions as negatives)

Scope note: research done 2026-11-21. "Verified" means I read the primary source (paper page, official repo, or the installed mlx-vlm 0.8.0 source) in this session. Items marked **UNVERIFIED** come from prior knowledge or a search snippet only, and their exact numbers should be checked before they go in the paper. Local project facts come from `/Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md` (Nov 21 block).

Project facts these notes rely on (from STATUS.md): C3 (Alice x calendar LoRA, about 70 SFT steps from 5 demos) scores 6/9 on preference-critical steps but skips clicking "30 min" in all 3 test demos. C5 (the LoRA trained on Bob + Carol) scores 0/9 and applies their values (15:00/11:00). The decision is carried by 1-2 tokens: an adapter at loss 0.16 still chose the wrong action on 10/15 training steps, with the decision tokens at 2-2.5 nats. The stable SFT recipe is lr 5e-5, rank 16, alpha 16, warmup 20, 4 epochs, batch 1, about 25 min per adapter (280 iterations, so about 5.4 s per iteration with 3 screenshots of history). Preference-critical steps are 9 of 42 test steps (about 21%).

---

## Q1. DPO, ORPO, KTO, SimPO, unlikelihood (plus DPOP, IPO, NPO): what each needs, how stable it is, and whether it works with a few dozen pairs

### Takeaway
For a handful of same-state pairs that differ by a few tokens, use **reference-based DPO with a chosen-likelihood anchor**: DPO-Positive (`dpop`, built into mlx-vlm) or DPO + NLL. Plain DPO is the textbook case for likelihood displacement, because our chosen and rejected outputs are near-identical. ORPO and SimPO length-normalize the whole response, which dilutes a 1-2-token decision inside a long shared Thought/Action string. KTO and unlikelihood are not built into mlx-vlm. Unlikelihood restricted to the first divergent tokens is a cheap custom alternative.

### Cited Findings
**What mlx-vlm 0.8.0 (installed in `.venv`) actually implements (read from source):**
- Train modes are `sft, uft, orpo, dpo, grpo` — [mlx_vlm/train.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/train.py) (`choices=("sft", "uft", "orpo", "dpo", "grpo")`).
- **DPO** loss types are `sigmoid` (standard DPO), `hinge`, `ipo` and `dpop` (DPO-Positive). Defaults are `beta=0.1` and `delta=50.0`. The dpop form is `-logsigmoid(beta * (logits - delta * max(0, ref_chosen - policy_chosen)))`. Metrics logged: `reward_accuracy`, `reward_margin`, `chosen_reward`, `rejected_reward`, `chosen_logps`, `rejected_logps` — [losses/preference.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/losses/preference.py), [vlm/dpo/config.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/vlm/dpo/config.py)
- DPO scores each response by the **summed** log-prob over response tokens only (completion mask). Only `ipo` length-normalizes. **No SFT/NLL term** is added to DPO. — [vlm/preference/loss.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/vlm/preference/loss.py) (`sequence_logps(..., length_normalize=loss_type == "ipo")`).
- DPO loads a **second full model as the frozen reference** (`--reference-model`, which defaults to `--model`, i.e. the base). The policy can start from an existing adapter via `--resume-adapter-file` / `--adapter-path` ("Load weights; optimizer state is not restored"). There is no fuse-adapter helper and no option to precompute reference log-probs. — [mlx_vlm/train.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/train.py) (lines ~159-168, 403-405)
- **ORPO** is `loss = -chosen_logps + beta * (-logsigmoid(log_odds(chosen) - log_odds(rejected)))`, with **length-normalized** (mean per-token) log-probs for both terms, default `beta=0.1` (the paper's lambda), and no reference model — [vlm/preference/loss.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/vlm/preference/loss.py), [vlm/orpo/config.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/vlm/orpo/config.py)
- The preference dataset accepts a shared `prompt` (message list) plus string `chosen`/`rejected`, or full conversations, with `images` on the row (attached to both candidates). It forces `train_on_completions=True`. — [vlm/preference/dataset.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/vlm/preference/dataset.py)
- KTO, SimPO, unlikelihood, NPO and RPO (DPO+NLL) are **not** in mlx-vlm 0.8.0. Searching the trainer source turned up no such loss. — same directory as above.

**The objectives themselves:**
- **DPO** (Rafailov, Sharma, Mitchell, Manning, Ermon, Finn; NeurIPS 2023; arXiv 2305.18290) needs paired (prompt, chosen, rejected) data and a frozen reference policy, usually the SFT model. Its loss is `-log sigma(beta * [(log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))])`, and beta = 0.1 is the common default. — [arXiv 2305.18290](https://arxiv.org/abs/2305.18290) (formula verified via mlx-vlm's implementation; paper hyperparameters **UNVERIFIED** this session)
- **Likelihood displacement:** during DPO the loss keeps falling while the preferred response's log-prob *decreases*. The effect is driven by preferred/dispreferred pairs with similar hidden embeddings (the CHES score), can shift mass to responses with the opposite meaning, and is mitigated by filtering out the most similar pairs. Razin et al., "Unintentional Unalignment: Likelihood Displacement in DPO", ICLR 2025 — [arXiv 2410.08847](https://arxiv.org/abs/2410.08847)
- **DPO-Positive (DPOP)** (Pal et al., "Smaug", arXiv 2402.13228) adds a penalty when the policy's chosen log-prob falls below the reference's. The paper motivates it by DPO lowering the preferred likelihood when pairs have a **small edit distance**. — [arXiv 2402.13228](https://arxiv.org/abs/2402.13228) (motivation **UNVERIFIED** this session beyond the mlx-vlm implementation; mlx-vlm's default `delta=50` matches the form)
- **ORPO** (Hong, Lee, Thorne; EMNLP 2024; arXiv 2403.07691) is single-stage and reference-free: NLL on chosen plus lambda times an odds-ratio term, with lambda = 0.1 in the paper. — [arXiv 2403.07691](https://arxiv.org/abs/2403.07691) (lambda verified as the mlx default; paper results **UNVERIFIED**)
- **KTO** (Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela; ICML 2024; arXiv 2402.01306) needs only "a binary signal of whether an output is desirable", so unpaired data works. It "matches or exceeds the performance of preference-based methods at scales from 1B to 30B." — [arXiv 2402.01306](https://arxiv.org/abs/2402.01306). The abstract does not cover data imbalance or skipping SFT (any such claims are **UNVERIFIED**).
- **SimPO** (Meng, Xia, Chen; NeurIPS 2024; arXiv 2405.14734) is reference-free, uses a length-normalized reward and a target margin gamma, and recommends beta 2.0-10 with gamma/beta 0.1-0.8. Learning rate is "the most critical hyperparameter": 1e-5 can cause incoherent or repetitive outputs, the recommended grid is 3e-7 to 1e-6, and the v0.2 model struggles with structured outputs such as JSON. The SFT term defaults to 0, because it preserves GSM8K but hurts chat. Gemma-2-9B: SimPO 72.4 vs DPO 67.8 AlpacaEval-2 LC ("perform comparably"). — [princeton-nlp/SimPO README](https://github.com/princeton-nlp/SimPO)
- **Unlikelihood training** (Welleck, Kulikov, Roller, Dinan, Cho, Weston; ICLR 2020; arXiv 1908.04319) adds `-sum_c log(1 - p(c | context))` over a set of negative candidate tokens to the MLE loss, at token or sequence level. — [arXiv 1908.04319](https://arxiv.org/abs/1908.04319) (**UNVERIFIED** this session, from prior knowledge)
- **NPO** (Zhang, Lin, Bai, Mei; arXiv 2404.05868) is a bounded, DPO-style loss on negatives only. It was proposed because plain gradient ascent on negatives collapses quickly. — [arXiv 2404.05868](https://arxiv.org/abs/2404.05868) (**UNVERIFIED** this session)
- Few-dozen-pair regime: the most direct tiny-data GUI evidence found is TIPO (Q3), with 151 tasks / 302 trajectories, where vanilla DPO did **not** improve persona compliance over SFT (31.94 vs 36.86). — [arXiv 2604.11259](https://arxiv.org/html/2604.11259)

### Inferences
- **The structure of our pairs is the worst case for plain DPO.** "click(30 min)" vs "click(50 min)", or "type 14:00" vs "type 15:00", share all prompt context, the history and most response tokens. Razin et al. tie that similarity to displacement of the chosen likelihood. Expect `chosen_logps` to fall under `sigmoid` DPO, so the guard is to use `--loss-type dpop` or add an NLL anchor (Agentic-DPO's ablation in Q2 shows the anchor is worth 14-18 points).
- **Summed vs mean log-probs matters for us.** With a summed score (mlx DPO), tokens before the first divergence have identical log-probs under the same model and cancel exactly in `log pi(y_w) - log pi(y_l)`. DPO's gradient therefore lands only on the divergent suffix, which is where the decision lives. ORPO's and SimPO's per-token *mean* does not cancel the shared prefix: a 40-token response that differs in 3 tokens produces a small odds-ratio difference, which weakens the preference signal. If you use ORPO, raise lambda (`--beta` 0.5-1.0) or shorten completions.
- **KTO fits a different data shape.** It suits many negatives with no clean pairing, for example sampled model errors from online runs. We have clean same-state pairs, so KTO buys little and would need custom code.
- **SimPO is a poor fit:** it needs no reference (a minor win at our scale), but it is learning-rate-fragile, length-normalized, and reported to hurt structured output. UI-TARS's `Action: click(start_box='(x,y)')` grammar is structured output.
- **Unlikelihood is the cheapest custom add-on to SFT:** add `-log(1 - p(t_neg))` at the first divergent token (for example the first digit of Bob's coordinate or the "1" of "15:00") to the existing SFT loss. It is easy to write in an mlx loss function and needs no reference model. The risk is pushing down tokens shared with valid actions. Restrict it to the first divergent token at the aligned state, which is Welleck's "candidate set excludes the ground-truth token" rule.

### Gaps
- I found no paper that benchmarks DPO, ORPO, KTO and SimPO head-to-head at about 10-100 pairs on a VLM agent. The tiny-data comparison has to be run by us.
- The exact Smaug/DPOP numbers and recommended lambda range were not re-verified. mlx-vlm defaults to delta = 50.
- ORPO paper results (for example AlpacaEval numbers for Mistral-ORPO) were not re-fetched.

---

## Q2. Step-level and action-level DPO for agents: Agent Q, Step-DPO, IPR, ETO, StepAgent, and GUI-agent DPO 2024-2026. How are pairs built?

### Takeaway
The field has converged on **same-state, action-level pairs**: the expert or verified action at a state vs an alternative at the *same* state. The alternative is a failed-rollout step, an MCTS sibling, a student-sampled action, or a task-irrelevant element. Training uses DPO **plus an SFT anchor** on an SFT-warmed reference. Step-level terms beat trajectory-level terms, and the best recent results restrict pairs to the small set of **critical / forking steps** (about 16% of steps in CSO).

### Cited Findings
- **IPR** (Xiong et al., "Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement", EMNLP 2024, arXiv 2406.11176). The agent is SFT'd on expert trajectories, then explores new actions *along the expert trajectory*. Each new action is scored against the expert step with Monte Carlo step rewards, and mismatches become contrastive action pairs. The loss is outcome-DPO + **step-DPO** + **SFT** (the SFT term "maintains the absolute likelihood of successful actions"). Removing step-DPO hurts more than removing outcome-DPO. — [ACL Anthology 2024.emnlp-main.93](https://aclanthology.org/2024.emnlp-main.93); [arXiv 2406.11176](https://arxiv.org/abs/2406.11176)
- A later critique says IPR's MC rollouts are prohibitively expensive on complex tasks — [arXiv 2602.03412](https://arxiv.org/abs/2602.03412)
- **ETO** (Song et al., "Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents", ACL 2024, arXiv 2403.02502). It runs SFT, then explores, then builds trajectory-level pairs of *expert success vs agent failure* on the same task, trains DPO, and iterates (WebShop, ScienceWorld, ALFWorld). — [arXiv 2403.02502](https://arxiv.org/abs/2403.02502) (**UNVERIFIED** details this session, from prior knowledge)
- **Agent Q** (Putta et al., arXiv 2408.07199). MCTS with AI self-critique on web tasks. Step-level DPO pairs come from sibling branches at the same node, ranked by MCTS Q-values blended with critic scores. Reported gains on OpenTable for Llama-3-70B are large (about 18.6% to 81.7%, and higher with online search). — [arXiv 2408.07199](https://arxiv.org/abs/2408.07199) (numbers **UNVERIFIED** this session)
- **Step-DPO** (Lai et al., arXiv 2406.18629, math). It localizes the *first erroneous step* and pairs a correct vs incorrect version of that step given the same correct prefix, using about 10K pairs. — [arXiv 2406.18629](https://arxiv.org/abs/2406.18629) (**UNVERIFIED** details this session)
- **StepAgent** (Deng et al., "From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning", arXiv 2411.03817). It compares the agent's step against the expert's step and turns the difference into a step-wise reward (implicit-reward / inverse-RL variants). — [arXiv 2411.03817](https://arxiv.org/abs/2411.03817) (**UNVERIFIED** this session)
- **Agentic-DPO** (Yixiong Chen, Alan Yuille; arXiv 2607.10601, July 2026):
  - **Pairs:** for each expert step, K=4 candidate one-step actions are sampled from the *current student at the same state*. Unparseable or expert-equivalent candidates are dropped, and the negative is the remaining candidate with the highest student log-prob. Negatives are refreshed for R=5 rounds. There is no environment execution or reward model.
  - **Loss:** action-level DPO over action tokens against a frozen SFT-warmed reference, plus lambda * L_SFT with lambda = 0.5. Beta is 0.008, scaled by 1/max(|a+|,|a-|)^0.5.
  - **Results:** Mind2Web step success 45.6 to 64.4 (Qwen3.5-9B), tau-bench retail 21.7 to 41.4 (comparable to GRPO's 40.0), StableToolBench 57.1 to 90.9 (2B).
  - **Ablations:** removing the SFT anchor costs 14-18 points and raises seed variance. Removing the SFT warm-up drops accuracy to 0.0%. Trained on 25% of expert trajectories, it matches PPA+SFT trained on 100%.
  - Source: [arXiv 2607.10601](https://arxiv.org/html/2607.10601v1)
- **CSO, Verified Critical Step Optimization** (Li et al., Tencent; arXiv 2602.03412; ACL 2026 Findings). It starts from *failed* policy trajectories, and a PRM flags candidate critical steps. Expert models propose alternatives there, and the policy continues from each alternative to completion; only verified-successful alternatives become DPO data. It reports +37% relative over SFT on GAIA-Text-103 and +26% on XBench-DeepSearch, supervising **only about 16% of trajectory steps**. — [arXiv 2602.03412](https://arxiv.org/abs/2602.03412)
- **UI-TARS** (ByteDance; arXiv 2501.12326), our base model's own recipe. During online bootstrapping, erroneous steps and their corrections form DPO pairs (prefer the corrected action over the erroneous one, relative to the SFT reference, with beta controlling divergence). — [arXiv 2501.12326](https://arxiv.org/abs/2501.12326) (via search snippet of the paper. Exact pair-filtering details **UNVERIFIED**)
- **EvoCUA** (Meituan; arXiv 2601.15876). Step-level DPO at "critical forking points": reference-guided diagnosis finds the step where a failed rollout diverges from a successful reference, and builds correction pairs (chosen = optimal action at that step, rejected = the error) plus reflection pairs. — [arXiv 2601.15876](https://arxiv.org/pdf/2601.15876); details via a secondary review only ([Moonlight review](https://www.themoonlight.io/review/evocua-evolving-computer-use-agents-via-learning-from-scalable-synthetic-experience)), so **UNVERIFIED**
- **Executable Agentic Memory for GUI Agent** (arXiv 2605.12294) builds step-level pairs from AMEX expert trajectories: the correct action vs several task-irrelevant alternatives on the same page — [arXiv 2605.12294](https://arxiv.org/pdf/2605.12294) (search snippet only, **UNVERIFIED**)
- A survey of RL for GUI agents notes that GUI DPO pairs are usually built from successful vs failed trajectories in static datasets — [arXiv 2604.27955](https://arxiv.org/html/2604.27955v1) (snippet, **UNVERIFIED**)
- Thinking-level DPO for mobile agents (CoaT-tree, rule-based rewards, backward credit assignment) — [arXiv 2505.12299](https://arxiv.org/html/2505.12299v1) (snippet, **UNVERIFIED**)
- DPO with LLM-judge-built pairs on UI-TARS-2B (Intel) — [arXiv 2506.03095](https://arxiv.org/html/2506.03095v1) (snippet, **UNVERIFIED**)

### Inferences
- All of these papers build negatives from the policy's own errors or from search, and they need a *success signal* to label which action is better. **Our setting is easier.** Every persona's action is "valid", and the label comes from *who the user is*, so the persona checker or demo ownership is the oracle and no rollouts or PRM are needed. The closest analogue to our negative source is Agentic-DPO's student-sampled negative at the same state. Combine both: **other-persona negatives** (the personalization signal) plus **on-policy negatives from the C3 adapter** (which cover the skip failure that other personas' actions may not represent).
- The consistent recipe details we should copy: (1) SFT warm-up first, with the reference = the SFT model; (2) an SFT/NLL anchor during preference training; (3) the loss only on action tokens of the divergent step; (4) pairs only at critical steps.
- Agentic-DPO's very small beta (0.008/sqrt(len)) works only *because* of the 0.5 SFT anchor. Without an anchor, use the mlx `dpop` loss and a conventional beta (0.1).

### Gaps
- I found no GUI paper that does step-DPO with only about 5 demos (tens of pairs). The smallest pools found are 151 tasks (TIPO) and 3.8K pairs (Agentic-DPO).
- Agentic-DPO's learning rate and epochs were not stated in the content I read.
- The full EvoCUA and UI-TARS DPO procedures (beta, data volumes, filtering) were not read directly.

---

## Q3. Personalization with contrastive signals: personalized DPO, P-DPO, and preference optimization with user-specific negatives

### Takeaway
One 2026 paper, **TIPO** (arXiv 2604.11259), matches our design almost exactly: mobile GUI agent, annotators role-play personas on the *same* task instance, and the other persona's trajectory is the negative. It shows that **vanilla DPO on such pairs does not beat SFT on persona compliance (31.94 vs 36.86)**. A **step-gated, step-weighted** DPO that focuses on persona-critical steps raises compliance to 46.22 without losing task success. The LLM personalization literature (P-DPO, FSPO, LoRe) is mostly about *sharing* a model across users with user embeddings. It supports, but does not test, "per-user LoRA + other users as negatives".

### Cited Findings
- **TIPO: Mobile GUI Agent Privacy Personalization with Trajectory Induced Preference Optimization** (Zhixin Lin, Jungang Li, Dongliang Xu, Shidong Pan, Yibo Shi, Yuchi Liu, Yuecong Min, Yue Yao; arXiv 2604.11259, 13 Apr 2026):
  - **Setup:** two personas (Privacy-first vs Utility-first) role-played by 8 annotators on the same task instances, giving y+ (persona-consistent) and y- (the other persona's trajectory). Data: 151 tasks, 302 trajectories, about 8.2K steps, 8 categories. Backbone: Qwen2.5-VL-3B, with an SFT stage on the same data before preference training.
  - **Alignment:** trajectories are aligned to T = max(|y+|, |y-|), and the shorter one is padded with `no_action`. Divergence points are found by an LLM and checked manually.
  - **Loss:** `L = E[softplus(-m_t * alpha_t * z_t)]`, where z_t is a step-wise DPO log-ratio scaled by beta. Gate m_t = 0 when the chosen step is padding, and weight alpha_t = clip(Delta s_t / Delta_max, 0, 1)^gamma from a rule-based action score difference.
  - **Results (Overall SR / Compliance / Persona Distinction):** SFT 65.17 / 36.86 / 59.26, DPO 65.34 / 31.94 / 59.26, TIPO 65.60 / 46.22 / 66.67.
  - **Stated DPO failure modes on trajectory pairs:** padding adds "semantic placeholder noise"; uniform treatment dilutes persona-critical steps with neutral ones; gradients can be dominated by padding; sparse critical steps make DPO insensitive and unstable.
  - Not stated: beta, gamma, Delta_max, LoRA vs full fine-tuning, learning rate, epochs.
  - Source: [arXiv 2604.11259](https://arxiv.org/html/2604.11259)
- **P-RLHF / P-DPO** (Li, Zhou, Lipton, Leqi; "Personalized Language Modeling from Personalized Human Feedback", arXiv 2402.05133). A light user model produces user embeddings, combining an individual and a generic component, and the policy is trained with a personalized DPO objective on user-labelled pairs. — [arXiv 2402.05133](https://arxiv.org/abs/2402.05133) (**UNVERIFIED** details this session)
- **FSPO, Few-Shot Preference Optimization** (arXiv 2502.19312, Feb 2025) frames personalization as meta-learning, so the model adapts to a new user from a few of that user's labelled preferences (synthetic preference data, transfer to real users) — [ResearchGate entry for FSPO](https://www.researchgate.net/publication/389392363_FSPO_Few-Shot_Preference_Optimization_of_Synthetic_Preference_Data_in_LLMs_Elicits_Effective_Personalization_to_Real_Users); [arXiv 2502.19312](https://arxiv.org/abs/2502.19312) (snippet-level only)
- **LoRe** (arXiv 2504.14439) uses a shared low-rank reward basis with a small per-user weight vector, and only that vector is learned for a new user from few-shot examples — [arXiv 2504.14439](https://arxiv.org/pdf/2504.14439) (snippet)
- **Personalized Preference Fine-tuning of Diffusion Models** (Dang et al., CVPR 2025, arXiv 2501.06655) is DPO conditioned on per-user embeddings. With **4 preference examples** from a new user it reaches a 76% average win rate over Stable Cascade. — [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Dang_Personalized_Preference_Fine-tuning_of_Diffusion_Models_CVPR_2025_paper.pdf); [arXiv 2501.06655](https://arxiv.org/html/2501.06655) (snippet)
- **Personalized Federated DPO via LoRA Adapters** (arXiv 2610.12885): the base is FedAvg-synced, and the LoRA adapters stay local to capture client-specific preferences. This is the closest published "per-user LoRA trained with DPO". — [arXiv 2610.12885](https://arxiv.org/abs/2610.12885) (snippet only, **UNVERIFIED** details)
- **C-DPO** for personalized image editing: user embeddings plus graph-based collaboration, LoRA-tuning Qwen2.5 — [arXiv 2511.05616](https://arxiv.org/html/2511.05616) (snippet)
- Project evidence that other users' habits are a real, learnable *wrong* signal: C5 (Bob+Carol adapter) scores 0/9 on Alice's critical steps and types Bob's 15:00/11:00 — [STATUS.md](file:///Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md)

### Inferences
- TIPO's vanilla-DPO result (compliance fell vs SFT) is the most important caution for us. "Just run DPO on persona pairs" can *hurt*. The fix it reports is the same as Q2's (gate out non-critical and padding steps, weight critical ones), which supports building pairs **only** at preference-critical states.
- Our C5 result is effectively a measured "negative direction". A contrastive objective that pushes down C5-style actions at Alice's states targets exactly what an adapter can get wrong.
- **Leakage metric.** Report a "wrong-persona leakage" rate: the fraction of critical steps where the adapter outputs another persona's value. Analogous to TIPO's "Persona Distinction", it is directly what the negatives should reduce.
- Embedding/meta-learning approaches (P-DPO, FSPO, LoRe) are a different architecture (one shared model plus user vectors). Cite them as related work and as a possible future "N-user shared adapter + user embedding" arm, not as the recipe.

### Gaps
- The P-DPO and FSPO method and hyperparameter details were not read in full this session.
- No paper found tests "per-user LoRA, trained from about 5 demos, with other users' actions as negatives" in GUI agents. TIPO has 2 personas and a shared model rather than per-user adapters.

---

## Q4. Joint objectives and stability: does SFT + DPO (or ORPO) beat SFT alone on tiny data? Risks: over-optimization, lost formatting, chosen-likelihood decrease

### Takeaway
On agent data, **preference + SFT-anchor beats SFT** in every agent paper found (Agentic-DPO, IPR, CSO, TIPO). The anchor is what makes it stable: removing it costs 14-18 points in Agentic-DPO, and removing the SFT warm-up gives 0%. The main tiny-data risks are (1) chosen-likelihood decrease on near-identical pairs, (2) overfitting a 30-pair set in a few epochs, which shows as reward accuracy at 1.0 and rising margins while held-out accuracy falls, and (3) broken action grammar from a learning rate that is too high. Guard all three with dpop/NLL, early stopping on held-out step accuracy, and a parse-failure check.

### Cited Findings
- Agentic-DPO: L_ADPO + 0.5 * L_SFT. Dropping the SFT anchor costs 14-18 points of canonical accuracy and raises seed variance. Dropping the SFT warm-up gives 0.0% accuracy. The authors motivate the anchor as keeping the expert action's probability from being pushed down. — [arXiv 2607.10601](https://arxiv.org/html/2607.10601v1)
- IPR includes an SFT term to "maintain the absolute likelihood of successful actions" alongside outcome- and step-DPO — [ACL Anthology 2024.emnlp-main.93](https://aclanthology.org/2024.emnlp-main.93)
- In TIPO, SFT is a prior stage. Vanilla DPO after SFT lowered compliance (36.86 to 31.94), while gated/weighted step-DPO raised it (46.22) — [arXiv 2604.11259](https://arxiv.org/html/2604.11259)
- Likelihood displacement is driven by similar preferred/dispreferred responses (the CHES score), and can be catastrophic — [arXiv 2410.08847](https://arxiv.org/abs/2410.08847)
- SimPO's lr sensitivity: 1e-5 can produce incoherent or repetitive outputs, and the larger 1e-6 lr is blamed partly for degraded structured (JSON) output. An SFT term preserved GSM8K but hurt chat. — [SimPO README](https://github.com/princeton-nlp/SimPO)
- ORPO puts NLL and preference in one loss with no reference (it is the "joint" objective by construction). mlx-vlm implements it with length-normalized log-probs — [vlm/preference/loss.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/vlm/preference/loss.py); [arXiv 2403.07691](https://arxiv.org/abs/2403.07691)
- Iterative RPO (Pang et al., NeurIPS 2024, arXiv 2404.19733) found that adding an NLL term on the chosen output to DPO was important for reasoning gains, and the TRL library exposes it as `rpo_alpha` — [arXiv 2404.19733](https://arxiv.org/abs/2404.19733) (**UNVERIFIED** details and numbers this session)
- Over-optimization in direct alignment algorithms (Rafailov et al., "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms", arXiv 2406.02900): quality can degrade as the KL from the reference grows, even without an explicit reward model — [arXiv 2406.02900](https://arxiv.org/abs/2406.02900) (**UNVERIFIED** this session)
- Project facts: our SFT already diverged at lr 1e-4 (alpha 32) after about 40 iterations on 70 samples, and NaN'd at 2e-4. Average loss hides the decision tokens. — [STATUS.md](file:///Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md)

### Inferences
- **SFT-only vs SFT-then-dpop vs SFT + unlikelihood is the experiment.** The literature predicts SFT-then-dpop wins on critical steps. TIPO warns that unweighted DPO can lose, so we must gate to critical steps and anchor.
- **Formatting loss is checkable.** Count parse failures of `Thought: ... Action: click(start_box='(x,y)')` on held-out steps and treat any rise above the SFT adapter as a stop signal. Formatting should be protected because (a) the shared prefix cancels in summed DPO and (b) dpop penalizes chosen log-prob dropping. The risk comes mainly from lr.
- **Over-optimization at 30 pairs** looks like `reward_accuracy` hitting 1.0 within an epoch while `reward_margin` keeps climbing. Stop at 2-4 epochs, or when held-out critical-step accuracy stops improving, whichever comes first.
- **Reference choice.** mlx-vlm's reference defaults to the *base* model, not our SFT adapter. With policy = C3 (resumed adapter) and reference = base, the implicit reward subtracts the *base model's* bias (for example its preference for keeping the 60-min default). That is acceptable and needs no code. The cleaner "reference = SFT" setup needs a fused C3 copy, which takes about 1 h of work (merge `W + (alpha/r) * B @ A` and save) and about 16-18 GB more disk.

### Gaps
- No tiny-data (under 100 pairs) ablation of "SFT + DPO vs SFT" for GUI/VLM agents was found. The evidence is from thousands of pairs.
- The exact RPO and over-optimization numbers were not re-verified.

---

## Q5. Pairing: how to pair at the step level when trajectories differ in length, and whether to use only preference-critical steps

### Takeaway
Pair by **aligned state, not by step index**. For each target-user step, find the other persona's demo of the *same task instance* at the *same UI state* and use their action there as the rejected, re-expressed in the target user's screenshot. When the target user has an extra step (Alice clicks "30 min", Bob keeps the 60-min default and moves on), **the other persona's next action becomes a "skip negative"**. That directly targets C3's observed failure, and it removes the padding problem TIPO had to gate out. Use only preference-critical states. Literature support: CSO supervises about 16% of steps, TIPO gates and weights, and IPR/Step-DPO pair at the divergence point.

### Cited Findings
- TIPO pads to max length with `no_action` and gates padded chosen steps out (m_t = 0). Its listed failure modes are padding noise, neutral steps diluting critical ones, and gradients dominated by padding. — [arXiv 2604.11259](https://arxiv.org/html/2604.11259)
- CSO trains only at verified critical steps, about 16% of steps, and reports +37% / +26% relative over SFT — [arXiv 2602.03412](https://arxiv.org/abs/2602.03412)
- EvoCUA pairs at the "critical forking point" where a failed rollout diverges from a successful reference — [arXiv 2601.15876](https://arxiv.org/pdf/2601.15876) (secondhand, **UNVERIFIED**)
- Agentic-DPO pairs strictly at the same state (expert action vs student negative), dropping candidates equivalent to the expert action after normalization — [arXiv 2607.10601](https://arxiv.org/html/2607.10601v1)
- Likelihood displacement is worst for the most similar pairs, so filtering the highest-CHES pairs helps — [arXiv 2410.08847](https://arxiv.org/abs/2410.08847)
- Token-level / critical-token variants exist: "Critical Tokens Matter" (cDPO; Lin et al., arXiv 2411.19943) estimates which tokens decide outcomes by contrasting positive vs negative models and weights DPO toward them; TDPO (Zeng et al., arXiv 2404.11999) does token-level DPO — [arXiv 2411.19943](https://arxiv.org/abs/2411.19943), [arXiv 2404.11999](https://arxiv.org/abs/2404.11999) (both **UNVERIFIED** this session)
- Project: the "preference-critical step" definition (target action differs from every other persona's on the same task) already exists in `eval/summarize.py`. About 21% of steps are critical. Demos have per-frame element maps, and `train/build_sft.py` builds `messages` + `images` per step via `build_messages(instruction, shots[:k+1], hist[:k], n_images)`. — [STATUS.md](file:///Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md), [train/build_sft.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/train/build_sft.py)

### Inferences
**Pair-construction algorithm (fits our data):**
1. For target persona P, skill S and train task instance i, take P's demo d_P and the other personas' demos d_Q of the *same* instance i. Our design guarantees the same instances.
2. **State key.** Key each step by a coarse UI-state signature: (page/URL, open dialog or dropdown, focused field, which form fields are already filled), not by pixel equality, since values differ (14:00 vs 15:00). Align d_P to each d_Q by dynamic programming (Needleman-Wunsch) over these keys. Element labels serve as the tiebreaker.
3. For each P step k at state s with action a_P:
   - **Value negative:** Q has a step at the same key with a_Q != a_P (for example "50 min" vs "30 min", "15:00" vs "14:00", a different folder or CC). Rejected = a_Q **transplanted into P's screenshot**: map Q's target element *label* to its bbox in P's element map, re-emit UI-TARS coordinates, and re-template the Thought ("Next, I'll click the 50 min option."). For `type`, swap only the content string.
   - **Skip negative:** P has a step at key s and Q has no step there (Q kept the default). Rejected = Q's next action from the same key, re-expressed in P's screenshot (for example "click Guests"). This turns length differences into the most useful negative and needs **no padding**.
   - **No pair** if a_Q == a_P, or if the target element is absent from P's map. Also no pair where Q's step comes from an earlier divergence that left P's screen in a different state (the key mismatch handles that).
4. **On-policy negatives** (Agentic-DPO style): run the current C3 adapter on each train state (greedy plus K=4 samples). At critical states and wherever its action is wrong (element-level check, as in the offline eval), add (a_P, a_model) pairs. This catches errors that no persona makes, such as repeating the previous step.
5. **Keep only critical states.** That means value negatives, skip negatives, and on-policy errors at those states. Do not pair non-critical steps: they are already covered by the SFT anchor, and pairing them is TIPO's dilution failure.
6. Expected size per (persona, skill): about 70 train steps x 21% ≈ 15 critical states, with up to 2 other personas each, giving about 20-30 value/skip pairs plus 0-15 on-policy pairs, so **about 25-45 pairs**. Deduplicate identical (state, rejected) pairs.

- Because the Thought is templated from the target element, the first divergent token is the element-name or value token in the Thought, followed by the coordinates. Summed DPO concentrates the update there.
- **Same-state pairs and the "other-persona state" problem.** Never use Q's *own* screenshot as the prompt for P's chosen action. P's history (`hist[:k]`) and screenshots must form the prompt, otherwise the pair teaches "Bob's screen implies Alice's action".

### Gaps
- No paper evaluates alignment-by-state-key vs pad-and-gate (TIPO) directly. Our choice rests on TIPO's stated padding failure modes.
- Whether CHES-style filtering is needed at about 30 pairs is unknown, and probably not, since all pairs are "minimal pairs" by design. Log `chosen_logps` instead.

---

## Q6. Concrete recipe for our setup (pair construction, beta, lr, steps) and an estimate of hours and compute

### Takeaway
Run **C3-SFT (unchanged) and then DPO-Positive on about 25-45 critical-state pairs** with mlx-vlm's built-in `--train-mode dpo --loss-type dpop`. Start the policy from the C3 adapter. The reference is the base model by default, or a fused C3 copy if time allows. Settings: beta 0.1, delta 50, lr 1e-5, 3 epochs, batch 1, same LoRA r16/alpha16, chosen from {5e-6, 1e-5, 2e-5} x {beta 0.1, 0.3} on 2 dev pairs by held-out critical-step accuracy. The cost is about 30-45 min of training per adapter, so about 6-9 h for 12 adapters overnight. Building the pair pipeline takes about 1 working day. Compare against two cheap baselines: SFT with critical-step upweighting, and SFT + first-divergent-token unlikelihood.

### Cited Findings
- mlx-vlm DPO defaults: beta 0.1, delta 50, loss types sigmoid/hinge/ipo/dpop. A reference model is loaded in full, and the policy can resume from an adapter. — [losses/preference.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/losses/preference.py), [train.py](file:///Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/train.py)
- Agentic-DPO's anchor (lambda 0.5) and SFT warm-up are essential, and same-state student negatives come from K=4 samples — [arXiv 2607.10601](https://arxiv.org/html/2607.10601v1)
- Critical-step-only supervision (about 16% of steps) is effective — [arXiv 2602.03412](https://arxiv.org/abs/2602.03412); step gating and weighting beat vanilla DPO for persona compliance — [arXiv 2604.11259](https://arxiv.org/html/2604.11259)
- Our stable SFT settings and timings: lr 5e-5, r16/alpha16, warmup 20, about 25 min per 280 iterations (about 5.4 s/iteration). The model is 18 GB bf16 on a 96 GB M5 Ultra, at 2-3 s per agent step. — [STATUS.md](file:///Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md)

### Inferences (the recipe)
**0. Baselines to run first (cheap, about 2 h total, and they may close most of the gap):**
- **B1, SFT with critical-step upweighting:** duplicate critical-state SFT samples 3x in `train.jsonl` (no code change). This tests whether the "skips 30 min" failure is just under-weighting.
- **B2, SFT + unlikelihood** (custom mlx loss, about 2-3 h of code): add `0.5 * -log(1 - p(t_neg))` at the first divergent token of each value/skip negative.

**1. Build pairs** (`train/build_pairs.py`, about 4-6 h of engineering; see the Q5 algorithm). Write mlx-vlm preference JSONL rows with these fields:
- `images`: the same list `build_messages` returns for P's step k
- `prompt`: P's messages up to the user turn
- `chosen`: P's templated response string
- `rejected`: the transplanted other-persona, skip, or on-policy response string

Keep per row the metadata `kind ∈ {value, skip, onpolicy}`, `task_id`, `step`, `source_persona`. Hold out the pairs from the 3 test demos for evaluation only. On-policy negative mining costs about 70 states x 5 generations x 2.5 s ≈ 15 min per adapter.

**2. Train** (per persona x skill):
```
uv run python -m mlx_vlm.train --train-mode dpo --loss-type dpop --beta 0.1 --delta 50 \
  --model models/UI-TARS-1.5-7B-mlx-bf16 --resume-adapter-file adapters/<C3 adapter> \
  [--reference-model models/UI-TARS-1.5-7B-C3fused-<persona>-<skill>] \
  --train-type lora --lora-rank 16 --lora-alpha 16 --lora-dropout 0.05 \
  --learning-rate 1e-5 --warmup-steps 10 --batch-size 1 --iters $((3 * N_pairs)) \
  --steps-per-report 5   # data/dataset flags as for SFT
```
(Flag names for beta, loss type and delta are generated from the config dataclass fields, so check them with `--help`. The dataset/column flags should match how `train_adapters.py` passes SFT data.)
- **beta = 0.1** is the default. A 30-nat summed margin over the divergent tokens saturates the loss, so try 0.3 if the margin explodes or held-out accuracy drops (it keeps the policy closer to the reference). **delta = 50** strongly penalizes any chosen-logp drop.
- **lr = 1e-5**, which is 5x below the SFT lr. Preference gradients on minimal pairs are sharp, and our SFT already diverged at 1e-4. Sweep {5e-6, 1e-5, 2e-5}.
- **Steps:** 3 epochs over about 25-45 pairs, so about 75-135 iterations. Early-stop on held-out critical-step accuracy, evaluated every 25 iterations with the existing offline eval.
- **Monitor (logged by mlx):**
  - `chosen_logps` must not fall below its start value by more than about 1 nat.
  - `reward_accuracy` at 1.0 within epoch 1 means stop early.
  - Parse-failure rate and all-steps accuracy must stay at least at C3's level.
- **Option, if dpop alone leaves all-steps accuracy below C3:** alternate 1 epoch of SFT (the 70 C3 samples, lr 5e-5 → 2e-5) with 1 epoch of dpop. This approximates Agentic-DPO's DPO + 0.5 * SFT, since mlx-vlm cannot mix both losses in one run. A true joint loss (DPO + 0.5 NLL over all 70 steps, with reference log-probs precomputed once from C3 so only one model sits in memory) is a custom mlx loop of about 4-6 h of code.
- **ORPO alternative:** single-stage from base on the pair set plus all non-critical steps. ORPO needs a rejected for every row, so use on-policy base-model errors on non-critical rows; `--beta` (lambda) 0.5-1.0 offsets the mean-log-prob dilution. Treat it as secondary, since its length-normalization fights a 1-2-token decision.

**3. Evaluate:** run offline on the held-out critical steps, pooled across all 12 (persona x skill) adapters (about 100+ critical steps, so CIs are about ±10 points instead of ±30 at n=9). Report all-steps accuracy, parse failures, wrong-persona leakage rate (outputs another persona's value) and the skip rate. Then run online on the same 3 test tasks per pair. Compare C3 vs B1 vs B2 vs C3+dpop (and the optional joint run).

**Compute and time estimate (M5 Ultra, 96 GB, bf16 7B, 3 history screenshots):**
- **Per DPO iteration:** policy forward+backward on chosen and rejected (about 2 x 5.4 s) plus reference forward on both (about 2 x 1.8 s) ≈ **14-15 s**. At 75-135 iterations that is **about 20-35 min per adapter**, plus about 5 min loading 2 models and about 15 min of on-policy mining, so **about 40-55 min per adapter in all**.
- **ORPO:** about 11 s per iteration, no reference, so about 15-25 min per adapter.
- **Memory:** policy 18 GB + reference 18 GB + activations for two 3-image sequences ≈ **45-60 GB peak**. That fits, but do not run it alongside agent evaluation or recording, since the GPU load caused stale frames before. A fused C3 reference needs about 16-18 GB of extra disk per adapter. Fuse, train, then delete it.
- **Sweep on 2 dev pairs:** 3 lr x 2 beta x 2 adapters x about 30 min ≈ **6 h**, one overnight run.
- **Full study:** 12 adapters x about 45 min ≈ **9 h** for the dpop arm. B1 adds about 12 x 25 min ≈ 5 h of SFT and B2 a similar amount, so all three take about 2 nights, or 1 night if B1 is run only on dev pairs.
- **Engineering:**
  - Pair builder plus transplanting: 4-6 h
  - On-policy miner (reusing the offline eval): 1-2 h
  - Unlikelihood loss: 2-3 h
  - Optional fuse script: about 1 h
  - Optional joint custom loss: 4-6 h
  - **Total: about 1.5-2 working days for the core path, plus 2 overnight runs.**
- All of this is local, with no cloud spend.

### Gaps
- mlx-vlm 0.8.0 CLI flag spellings for `beta`/`loss_type`/`delta` (auto-generated from dataclass fields) and its dataset-column flags were not executed or tested here. Check with `uv run python -m mlx_vlm.train --train-mode dpo --help`.
- It is unknown whether mlx-vlm batches chosen and rejected together. The timing estimate assumes sequential forwards and could be about 30% faster.
- There is no published evidence for the exact lr/beta at about 30 pairs on a 7B VLM with LoRA. The values above are extrapolated from our SFT stability limits and common DPO defaults, so the dev-pair sweep is what decides them.
- With oracle (scripted) demos, persona pairs are perfectly aligned. Human demos will vary in step order, so expect fewer clean value pairs and more reliance on state-key alignment and on-policy negatives.
