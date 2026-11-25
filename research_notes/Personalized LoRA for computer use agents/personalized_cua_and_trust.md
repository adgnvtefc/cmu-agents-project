# Personalized Computer-Use / GUI / Web Agents, Personalized-Agent Benchmarks, and Trust / Human-in-the-Loop for CUAs

Scope note: research done 2026-11-20. Covers work through Nov 2026, prioritizing 2024–2026. Most details come from arXiv abstract or HTML pages fetched directly. Items seen only in search snippets or reference lists are marked UNVERIFIED. Several 2026 papers report results on models that were current in 2026 (e.g., Claude Sonnet 4.6 / Opus 5, Gemini 3.1 Pro, Qwen3.x); those names are reported exactly as the papers give them.

**Headline for the project:** I found **no paper that trains a per-user (or per-skill) LoRA adapter for a GUI / computer-use agent from that user's own few demonstrations**. That held across 6 targeted searches and ~20 primary-source fetches. The closest prior work is listed below, and each is a partial threat:
- (a) **FingerTip 20K** LoRA-fine-tunes Qwen2.5-VL-7B on user-behavior data, but as one *shared* model pooled across 95 users.
- (b) **PersonalWAB/PUMA** LoRA-tunes LLaMA2-7B with SFT+DPO for personalized web *function calls*, also shared across users.
- (c) **AdaptAgent** adapts open-weight web agents from ≤2 demos via meta-learning, but per website/domain, not per user.
- (d) **OPPU** does one LoRA per user, but for *text* LLM personalization, not GUI agents.
- (e) **ALLOY** explicitly names "continuously fine-tuning the base model on user-provided demonstrations" as future work.

Every 2026 personalized-GUI benchmark (KnowU-Bench, iPrefBench, PrefDiag-Bench, PersonalAlign, PSPA-Bench, Persona2Web) personalizes through context, memory, retrieval or clarification. None does it through per-user weights.

---

## Q1. Which benchmarks evaluate personalized web / GUI agents? (user/preference definition, data, models, results)

### Takeaway
There is a rapidly growing set of benchmarks: about 10 personalized-GUI benchmarks have appeared since Oct 2024, most of them in 2026. They almost all define the "user" as a synthetic or LLM-generated profile plus a behavioral log or history. They test underspecified instructions on mobile (Android/iOS) or shopping/web environments. Frontier models reach only about 26–60% success on personalized splits, and open 7–8B GUI models are far lower. Only FingerTip and MobileIAR use *real* users' own demonstrations. None evaluates desktop mail/calendar/drive workflows of the kind the project targets.

### Cited Findings

#### 1. PersonalWAB — "Large Language Models Empowered Personalized Web Agents"
- **Authors / venue / ID:** Hongru Cai, Yongqi Li, Wenjie Wang, Fengbin Zhu, Xiaoyu Shen, Wenjie Li, Tat-Seng Chua. WWW 2025 (ACM Web Conference, Sydney). arXiv 2410.17236. [arXiv HTML](https://arxiv.org/html/2410.17236v2)
- **Summary:** The first "personalized web agent" benchmark. It covers 3 tasks (personalized search, recommendation, review generation) for 1,000 users sampled from the Amazon Review dataset across 5 product categories. Each user has an LLM-generated profile (basic info, shopping preferences, behavioral tendencies) plus a time-ordered purchase/review history, split chronologically 80% history / 10% train / 10% test. Instructions are LLM-synthesized from ground-truth items and reviews. — [arXiv](https://arxiv.org/html/2410.17236v2)
- **Environment:** It is *not a GUI*. The web is abstracted into 5 functions (search_product_by_query, get_recommendations_by_history, add_product_review, respond, stop). The authors argue GUIs are not essential for agents. It also covers shopping only. — [arXiv](https://arxiv.org/html/2410.17236v2)
- **Metrics:** Function accuracy and result accuracy. Result accuracy is rank-based for search and recommendation and uses embedding similarity for reviews. Multi-turn mode adds average steps and uses an LLM user simulator. — [arXiv](https://arxiv.org/html/2410.17236v2)
- **Method, PUMA (Personalized User Memory-enhanced Alignment):**
  - Memory bank retrieved by cosine similarity and filtered by function type.
  - SFT on heuristic pseudo-labels, then DPO over beam-searched candidates ranked by result accuracy.
  - Backbone is **LLaMA2-7B with LoRA**, one model shared across users. Memory is capped at 256/512/768 tokens.
  - — [arXiv](https://arxiv.org/html/2410.17236v2)
- **Key results (single-turn, overall):**
  - No-memory gpt-4o: function acc 0.684, result acc 0.355.
  - PUMA (gpt-4o): 0.979 / 0.373.
  - PUMA (LLaMA-7B): 0.994 / 0.406.
  - Recommendation result accuracy is very low in absolute terms (0.000 → 0.054).
  - PUMA takes about 2.8 s vs 6.5–6.9 s for GPT-based agents.
  - The paper is internally inconsistent on whether the GPT backbone is gpt-4o or gpt-4o-mini.
  - PUMA was not evaluated in full multi-turn mode.
  - — [arXiv](https://arxiv.org/html/2410.17236v2)
- **Relation to project:** Low threat. It is a precedent for "a small LoRA-tuned 7B model beats a frontier model on personalization when given user memory," but the model is shared across users, the environment is function calls rather than GUI, and the domain is shopping only.

#### 2. FingerTip 20K — "A Benchmark for Proactive and Personalized Mobile LLM Agents"
- **Authors / venue / ID:** Qinglong Yang, Haoming Li, Haotian Zhao, Xiaokai Yan, Jingtao Ding, Fengli Xu, Yong Li (Tsinghua). arXiv 2507.21071 (cs.HC). It is listed as an ICLR 2026 poster on [iclr.cc](https://iclr.cc/virtual/2026/poster/10007512) and [mlanthology](https://mlanthology.org/iclr/2026/yang2026iclr-fingertip/), although the arXiv v2 page itself lists no venue. — [arXiv HTML v2](https://arxiv.org/html/2507.21071v2)
- **Data (real users):**
  - 95 crowdsourced Android users (54 M / 41 F, ages 18–60, mean 25.9), each collecting for one month with at most 12 episodes per day.
  - 21,437 episodes across 506 apps.
  - All users are in mainland China and mostly use Chinese apps, which the authors acknowledge as a limitation.
  - — [arXiv](https://arxiv.org/html/2507.21071v2)
- **"User" definition:**
  - Profile: sex, age, occupation, address, marital status, phone brand.
  - Context: timestamp plus one of 12 location categories.
  - Up to 20 recent intents (used for the proactive track).
  - The full action sequence from a *similar past task by the same user*, used in-context for personalized execution.
  - — [arXiv](https://arxiv.org/html/2507.21071v2)
- **Tracks:** (1) Proactive task suggestion infers the user's intent from profile, time, scenario, history and the first screenshots. (2) Personalized task execution carries out a known instruction while reflecting the user's own action habits. — [arXiv](https://arxiv.org/html/2507.21071v2)
- **Preference-adherence metric (separate from success):**
  - SR2 is task success, checked manually.
  - **Sim2 = S_I / S_II.** S_I is the Levenshtein similarity between the agent's action sequence and *this user's* sequence. S_II is the similarity to the most similar sequence from a user of a different age-based type.
  - Sim2 > 1 means the agent is closer to this user's own style than to other users' styles.
  - Step ratio measures efficiency relative to the user's golden steps.
  - — [arXiv](https://arxiv.org/html/2507.21071v2)
- **Models tested:** GPT-4.1, Qwen-VL-Max, DeepSeek-VL2, Qwen2.5-VL-7B/72B, QVQ-Max, Aguvis-7B, CogAgent-9B, **UI-TARS-1.5-7B**, AutoDroid, and AppAgent (GPT-4.1 backbone). — [arXiv](https://arxiv.org/html/2507.21071v2)
- **Key results:**
  - Proactive: best model QVQ-Max reached SR1 12.8% vs humans 30.3%.
  - Personalized: **UI-TARS-1.5-7B had the best SR2 at 38.5%**. AppAgent had the best Sim2 at 1.12. Most models had Sim2 ≈ 1, meaning they did not reflect the individual user.
  - — [arXiv](https://arxiv.org/html/2507.21071v2)
- **Fine-tuning experiment:**
  - Base model Qwen2.5-VL-7B, **LoRA rank 4 or 64**.
  - Trained on 1k proportionally sampled episodes or the full 16k-episode training set, with one model per track **pooled across all users (not per-user)**.
  - Results with full data, rank 64: personalized SR2 went from 1.5% to 15.5% and Sim2 from 0.95 to 1.42. Proactive SR1 went from 3.1% to 26.0%.
  - The fine-tuned model's SR2 stayed below UI-TARS-1.5-7B, but its Sim2 was higher.
  - — [arXiv](https://arxiv.org/html/2507.21071v2)
- **Relation to project:** The **closest methodological threat**. It uses LoRA on Qwen2.5-VL-7B to make a GUI agent act like individual users. It differs in four ways: one shared adapter conditioned on in-context history, not per-user adapters; thousands of episodes rather than 3–5 demos; mobile rather than web mail/calendar/drive; and trajectory similarity as a proxy for preference rather than programmatic preference checks. The project should cite it and position per-user adapters as the parametric, few-shot counterpart.

#### 3. KnowU-Bench — "Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation"
- **Authors / venue / ID:** Tongbo Chen, Zhengxi Lu, Zhan Xu, Guocheng Shao, Shaohan Zhao, Fei Tang, Yong Du, Kaitao Song, Yizhou Liu, Yuchen Yan, Wenqi Zhang, Xu Tan, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen (Zhejiang U., Apple, Tencent). arXiv 2604.08455, Apr 2026, no venue listed. — [arXiv HTML](https://arxiv.org/html/2604.08455)
- **User definition:**
  - A hidden structured YAML profile (identity, locations, digital context, habits, preferences, decision criteria, social graph) for 4 roles: Researcher, Developer, Student, Grandma.
  - The agent sees only timestamped behavior logs. Clean logs contain only preference-relevant events. Noisy logs add about 25% distractor events.
  - Profiles and logs are LLM-synthesized and then manually reviewed.
  - — [arXiv](https://arxiv.org/html/2604.08455)
- **Tasks and environment:**
  - 42 general, 86 personalized (ambiguous, e.g., "order lunch") and 64 proactive tasks.
  - Containerized Pixel 8 Android emulator, 23 apps, including 4 service apps rebuilt as local H5 apps with mock backends.
  - A gpt-4o user simulator answers `ask_user` queries.
  - — [arXiv](https://arxiv.org/html/2604.08455)
- **Scoring:**
  - Hybrid score S = λ·S_rule + (1−λ)·S_llm. A rule checker verifies state (recipient, order, settings, time windows). An LLM judge scores preference alignment and communication style. λ depends on how preference-dependent the task is.
  - Binary success is defined as a score > 0.99.
  - Interaction efficiency is mean S_i / max(c_i, 1), where c_i is the number of questions asked.
  - Human validation used 4 experts on 26 trajectories.
  - — [arXiv](https://arxiv.org/html/2604.08455)
- **Models tested:** MAI-UI-8B, UI-Venus-1.5-8B, GUI-Owl-1.5-8B, Qwen3-VL-8B/32B, Qwen3.5 (9B / 122B-A10B / 397B-A17B), Gemini 3.1 Pro Preview, Claude Sonnet 4.6, Seed 2.0 Pro. — [arXiv](https://arxiv.org/html/2604.08455)
- **Key results:**
  - Overall SR: Claude Sonnet 4.6 60.4%, Seed 2.0 Pro 51.6%, Gemini 3.1 Pro 44.3%.
  - Personalized-hard split: Claude 44.2% SR and 78.9% average score. **All open-source models are below 12% SR.**
  - Claude asks about 0.4 questions per task. Seed asks about twice as many but scores lower.
  - 93.8% of Claude's personalized failures are clarification or partial-preference errors.
  - Grandma is the hardest role.
  - RAG over logs helps some small models (Qwen3-VL-8B: 13.6% → 20.4% on clean logs) and hurts others (MAI-UI-8B drops to 9.3% under noisy RAG).
  - — [arXiv](https://arxiv.org/html/2604.08455)
- **Relation to project:** A strong evaluation-design template: hidden profile, user simulator for `ask_user`, and a rule checker plus judge. It also shows that open 8B GUI models are weak at personalization, which motivates fine-tuning. It does not do per-user training.

#### 4. iPrefBench — "A User-Centric Benchmark for Personalized Mobile Use Agents"
- **Authors / venue / ID:** Yuting Ning, Ruijia Cheng, Sushobhan Ghosh, Peitong Duan, HungShi Lin, Abhishek Sundararajan, Shreyash Pandey, Huan Sun, Jeffrey Nichols, Eldon Schoop (OSU and Apple). arXiv 2610.34735, 26 Oct 2026, preprint. — [arXiv HTML](https://arxiv.org/html/2610.34735v1)
- **User definition:**
  - A single synthetic persona ("Jordan Avery") seeded into 26 mock iOS apps on iOSWorld.
  - 3,051 history events, converted to natural language by an LLM.
  - A hidden profile of 13 dimensions and 92 entries, each linked to supporting history events.
  - — [arXiv](https://arxiv.org/html/2610.34735v1)
- **Tasks:**
  - 130 underspecified tasks. Each omitted detail is either **inferable** from history or **ask-only** (known only to the user).
  - 58.5% of tasks have at least one ask-only requirement. 26.9% require cross-app preference transfer.
  - Perturbations: instruction (11 tasks), preference drift (10), environment unavailability (13).
  - Tasks were drafted by coding agents, then reviewed by humans and checked by execution.
  - — [arXiv](https://arxiv.org/html/2610.34735v1)
- **Metrics (explicitly separates task success from personalization):**
  - Success rate means all rubric criteria are met.
  - **Action-completion score** is reported separately from a **personalization score**, which is split into inferable and ask-only criteria. An LLM judge (Claude Sonnet 5) scores each criterion.
  - Interaction metrics: ask-only **recall** (fraction of ask-only requirements elicited) and **precision** (fraction of questions that reveal a new requirement).
  - — [arXiv](https://arxiv.org/html/2610.34735v1)
- **Models tested:** Claude Opus 5, Claude Sonnet 5, Gemini 3.1 Pro, Gemini 3.7 Flash, Kimi K3, Qwen3.8-27B, Qwen3.8-Flash-Next. Specialized GUI agents were excluded because of context and invalid-action problems. — [arXiv](https://arxiv.org/html/2610.34735v1)
- **Key results:**
  - Opus 5 is best: 47.7% success, **91.6% action completion but only 69.4% personalization**, 44.2% ask-only recall at 65.8% precision.
  - Success ranges from 26.2% to 47.7% across models.
  - A pre-execution clarification phase raised Sonnet 5 from 39.2% to 53.8%.
  - More history raised inferable personalization but not ask-only recall.
  - No fine-tuning or LoRA is used.
  - — [arXiv](https://arxiv.org/html/2610.34735v1)
- **Relation to project:** The best metric design to borrow: action completion vs personalization, inferable vs ask-only. The gap between 91.6% action completion and 69.4% personalization directly supports the project's thesis that agents can complete a task while still getting it wrong for the user.

#### 5. PrefDiag-Bench — "Diagnosing Personalization Failures in Mobile GUI Agents"
- **Authors / venue / ID:** Zhixin Lin, Dongliang Xu, Jungang Li, Shidong Pan, Jun Feng, Yanbin Sun, Jing Li, Liang Liu, Yue Yao. arXiv 2610.30573, 22 Oct 2026, preprint. — [arXiv HTML](https://arxiv.org/html/2610.30573)
- **Design:**
  - 53 basic tasks adapted from KnowU-Bench, each with 5 preference-conditioned variants, for 318 tasks (265 personalized).
  - Preference types: **Outcome, Constraint, Ranking, Process, Interaction**.
  - Each task has two context forms: an explicit profile, or synthetic behavioral memory logs.
  - 1,664 diagnostic probes covering 260 preference atoms. Five stages: P1 log-to-memory induction, P2 evidence selection, P3 relevance selection, P4 trigger recognition, P5 operation mapping.
  - — [arXiv](https://arxiv.org/html/2610.30573)
- **Metrics:** Binary success from **deterministic programmatic checkers**. Success requires the goal *and* all preference requirements, including any prescribed procedure or confirmation. No LLM judge. — [arXiv](https://arxiv.org/html/2610.30573)
- **Systems tested (11):** Seed-2.0-Pro, Claude Sonnet 4.6, Qwen3.7-Plus, MAI-UI-8B, Qwen3-VL-8B (Instruct and Thinking), Step-GUI-4B, GUI-Owl-1.5-8B, UI-Venus-1.5-8B, and M3A/T3A (GPT-5.4). — [arXiv](https://arxiv.org/html/2610.30573)
- **Key results:**
  - Seed-2.0-Pro: basic task success 81.13% vs personalized average 62.64% (profile) and 65.66% (logs).
  - Ranking and Process preferences are hardest. Process reaches only 47.17% for Seed under profiles.
  - Weakest stages are evidence selection (40.06%) and trigger recognition (39.69%), vs 88.80% for relevance selection.
  - Labeling the evidence raised Seed from 76.47% to 83.53%.
  - A memory-module ablation compares Mem0, LangMem and Hindsight. There is no per-user fine-tuning.
  - — [arXiv](https://arxiv.org/html/2610.30573)
- **Relation to project:** The preference taxonomy maps directly onto the project's mock-app checks, e.g., Process = "always CC my manager", Constraint = "never send after 6pm", Outcome = "file receipts into /Finance/2026". Its programmatic-checker design matches the project's plan.

#### 6. PersonalAlign / AndroidIntent — "Hierarchical Implicit Intent Alignment for Personalized GUI Agent with Long-Term User-Centric Records"
- **Authors / venue / ID:** Yibo Lyu, Gongwei Chen, Rui Shao, Weili Guan, Liqiang Nie. arXiv 2601.09636, Jan 2026. **ACL 2026 Main.** — [arXiv](https://arxiv.org/abs/2601.09636)
- **Summary:**
  - The AndroidIntent benchmark targets vague-instruction resolution and proactive suggestion, with 775 annotated user-specific preferences and 215 routines drawn from 20k long-term records.
  - The method, HIM-Agent, keeps a continuously updated hierarchical memory of preferences and routines.
  - Evaluated with GPT-5, Qwen3-VL and UI-TARS. HIM-Agent improves execution by 15.7% and proactive performance by 7.3%.
  - — [arXiv](https://arxiv.org/abs/2601.09636)
- **Relation:** A memory-based competitor. The baselines and exact metrics were not given in the abstract (UNVERIFIED).

#### 7. Other personalized-GUI benchmarks (abstract-level only)
- **PSPA-Bench** (Hongyi Nie, Xunyuan Liu, Yudong Bai, Yaqing Wang, Yang Liu, Quanming Yao, Zhen Wang; arXiv 2603.29318, Mar 2026):
  - Over 12,855 personalized instructions across 10 scenarios and 22 mobile apps, with "structure-aware process evaluation."
  - 11 GUI agents tested; even the best has limited success. Reasoning models beat general LLMs, and reflection plus long-term memory are key.
  - Its search snippet also says fine-tuning approaches "require large amounts of data and significant computational resources."
  - The models tested and the data-collection method were not on the abstract page (UNVERIFIED).
  - — [arXiv](https://arxiv.org/abs/2603.29318); [PDF](https://arxiv.org/pdf/2603.29318)
- **Persona2Web** (Serin Kim, Sangam Lee, Dongha Lee; arXiv 2602.17003; ICML 2026 per its arXiv page):
  - Claims to be the "first benchmark for evaluating personalized web agents on the real open web."
  - Uses a "clarify-to-personalize" principle: ambiguous queries must be resolved from long-span implicit user history.
  - Uses a "reasoning-aware evaluation framework." Metrics and numbers were not in the abstract (UNVERIFIED).
  - — [arXiv](https://arxiv.org/abs/2602.17003)
- **Me-Agent / "User FingerTip"** (Shuoxin Wang et al.; arXiv 2601.20162, Jan 2026):
  - Two-level habit learning: a prompt level with a Personal Reward Model, and a memory level with a Hierarchical Preference Memory separating long-term user memory from app memory.
  - Introduces the User FingerTip benchmark of ambiguous daily instructions and claims state-of-the-art personalization.
  - Whether any weights are tuned is not stated (UNVERIFIED).
  - — [arXiv](https://arxiv.org/abs/2601.20162)
- **MobileIAR** (from "Quick on the Uptake", Zheng Wu et al., arXiv 2508.08645):
  - Collected from 40 English- and Chinese-speaking users.
  - Measures **Intention Alignment Rate (IAR)**, i.e., whether actions match human-intent-aligned actions rather than only ground-truth actions.
  - — [arXiv](https://arxiv.org/abs/2508.08645)
- **PAIR** ("Can Agents Work for Everyone? Cross-User Reliability for Mobile GUI Agents in Personalized User Interfaces", Yeji Park, Jaeyun Shim, Taesik Gong; arXiv 2610.07972, Oct 2026):
  - Builds user-conditioned app states so the same task runs across different users' content.
  - Subgoal achievement drops 6.98–15.4 points in user-conditioned UIs, and **8.77–22.0 points for "personal targets drawn from each user's own content."**
  - The dominant failure is picking the wrong item, especially before the target appears on screen.
  - — [arXiv](https://arxiv.org/abs/2610.07972)
- **Other names seen only in reference lists (UNVERIFIED, not opened):** ProactiveMobile (2602.21858), PIRA-Bench (2603.08013), Pare (2604.00842), MemGUI-Bench (2602.06075), VitaBench 2.0 (2605.27141). — [KnowU-Bench refs](https://arxiv.org/html/2604.08455); [iPrefBench refs](https://arxiv.org/html/2610.34735v1)

### Inferences
- **Field saturation risk.** "Personalized GUI agent benchmark" is no longer novel by itself. At least 8 benchmarks appeared in 2026. The project's benchmark contribution should be framed around (i) desktop/web productivity apps (mail, calendar, drive), which none of the 2026 benchmarks cover since they are almost all mobile, (ii) **personal environment** (the user's own folders and contacts), which PAIR shows is a distinct failure mode, and (iii) recurring workflows learned from **3–5 demonstrations**, not long behavior logs.
- **The user model is almost always synthetic.** FingerTip and MobileIAR are the exceptions with real users. A small mock-app benchmark with scripted personas is therefore methodologically in line with the 2026 state of the art, as long as its limitations are acknowledged.
- **Base-model choice.** On FingerTip, UI-TARS-1.5-7B (SR2 38.5%) far outperformed base Qwen2.5-VL-7B (1.5%) on personalized execution. Starting from UI-TARS-1.5-7B likely gives a much better execution floor, which makes preference effects measurable.

### Gaps
- I could not open the full PSPA-Bench, Persona2Web, Me-Agent or PersonalAlign papers. Their metrics, models and numbers beyond the abstracts remain UNVERIFIED.
- I did not find any personalized benchmark for desktop (Windows/macOS/Linux) computer-use agents, i.e., an OSWorld-style personalized benchmark. "Mypcbench" appeared only as a reference-list title (UNVERIFIED, not located).
- No benchmark found uses the *few-demonstration-per-user* protocol (3–5 demos, then underspecified test tasks).

---

## Q2. What methods personalize GUI agents (memory, user profiles, retrieval, fine-tuning)? Has anyone fine-tuned a per-user adapter?

### Takeaway
Most personalized GUI-agent methods work in context or in memory:
- profile prompting
- full-log vs RAG retrieval of history
- hierarchical preference memory (HIM-Agent, Me-Agent)
- SOP/habit repositories built from demos (IFRAgent)
- latent-space experience activation (ExpActivator)
- clarification via `ask_user`

Weight-based methods exist but train **one shared model**: PUMA (LoRA SFT+DPO), FingerTip (LoRA on pooled users), RePAIR (RL on cross-user differences) and TIPO (privacy-persona preference optimization). **No per-user adapter for a GUI agent was found.** Per-user LoRA (OPPU) is established only for text LLMs.

### Cited Findings
- **Memory / RAG (KnowU-Bench).** Full-history context vs embedding RAG gives model-dependent results. RAG lifts Qwen3-VL-8B from 13.6% to 20.4% on clean logs. UI-Venus prefers full logs. MAI-UI-8B drops to 9.3% under noisy RAG. — [KnowU-Bench](https://arxiv.org/html/2604.08455)
- **Memory modules (PrefDiag-Bench).** An ablation compares Mem0, LangMem and Hindsight with Seed-2.0-Pro. The main bottleneck is evidence selection and trigger recognition (about 40% accuracy), not relevance (about 89%). — [PrefDiag-Bench](https://arxiv.org/html/2610.30573)
- **Hierarchical memory (HIM-Agent, ACL 2026).** A continuously updated hierarchy of preferences and routines from long-term records gives +15.7% execution and +7.3% proactive performance. — [PersonalAlign](https://arxiv.org/abs/2601.09636)
- **Two-level habit learning (Me-Agent).** A prompt-level preference learner with a Personal Reward Model, plus a Hierarchical Preference Memory. — [Me-Agent](https://arxiv.org/abs/2601.20162)
- **Demonstrations → SOP library + user-habit repository (IFRAgent, "Quick on the Uptake").**
  - Explicit intention flows from demos form a query-level SOP vector library. Implicit intention flows form a user-level habit repository.
  - A RAG SOP extractor and a query rewriter trained by knowledge distillation turn an ambiguous query into a personalized query plus SOP.
  - Results: IAR +9.88 points (+53.33% relative) and step completion +8.26 points (+32.40% relative).
  - The query rewriter is trained, but the training is not per user.
  - Authors: Zheng Wu, Heyuan Huang, Yanjia Yang, Yuanyi Song, Xingyu Lou, Weiwen Liu, Weinan Zhang, Jun Wang, Zhuosheng Zhang. arXiv 2508.08645, v3 Aug 2026.
  - — [arXiv](https://arxiv.org/abs/2508.08645)
  - **Relation:** This is the **closest conceptual competitor**: it learns personal preferences from a user's own demonstrations for underspecified mobile instructions. It is non-parametric per user, and the project is parametric per user. It should be a baseline or at least a comparison point.
- **Experience activation (ExpActivator, arXiv 2609.33304, Sep 2026; Fuyao Zhang, Xuan Wang, Zherui Li, Jiaming Zhang, Longtao Huang, Wei Yang Bryan Lim).**
  - Key finding: appending retrieved personal history to context **improves mainly the first step, adding little over the remaining ~90% of steps**.
  - The training-free fix matches each new screen to historical states in the frozen backbone's latent space and supplies the past action at that state.
  - Results: +28% average within-trajectory step success, using about 1/5 the history tokens, and about 2.3× the MCC of the strongest proactive baseline.
  - — [arXiv](https://arxiv.org/abs/2609.33304)
- **Few-shot demonstration learning (non-personal).**
  - **LearnAct / LearnGUI** (ACL 2026 Findings) uses multi-agent demonstration parsing, retrieval and execution, evaluated on 2,252 offline and 101 online tasks. Online, it raises UI-TARS-7B-SFT from 18.1% to 32.8% success. — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491.pdf); [alphaXiv](https://www.alphaxiv.org/audio/2504.13805)
  - **AdaptAgent** (Gaurav Verma, Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Tucker Balch, Manuela Veloso, JPMorgan AI Research; arXiv 2411.13451; NeurIPS 2024 AFM Workshop, abridged): adapts to new websites from **≤2 human demos**. Proprietary models use in-context demos. **Open-weight models use meta-learning with "meta-adaptation demonstrations."** Gains are +3.36 to +7.21 points (21–66% relative) on Mind2Web and VisualWebArena. Multimodal demos beat text-only. — [arXiv](https://arxiv.org/abs/2411.13451)
  - Search summaries describe AdaptAgent as motivating "personalized web agents that can learn from individual user demonstrations" — [alphaXiv](https://www.alphaxiv.org/abs/2411.13451). This wording came from a search snippet and is UNVERIFIED against the full paper.
  - **ShowUI-Aloha** (Yichun Zhang et al., Mike Zheng Shou; arXiv 2601.07181, Jan 2026): humans teach by recording desktop screen video plus clicks and keystrokes. A Learner captions the demo, a Planner uses it, and an Executor acts at the OS level. Demos appear to be used in context, not for weight updates. The abstract gives no numbers. — [arXiv](https://arxiv.org/abs/2601.07181)
  - **Instruction Agent** (arXiv 2509.07098) extracts action plans from a single expert demonstration. It argues a short demonstration recording is the most reliable way to convey what the user needs (search snippet). — [arXiv PDF](https://arxiv.org/pdf/2509.07098)
- **Weight-based but shared across users.**
  - PUMA uses LLaMA2-7B with LoRA, SFT plus DPO. — [PersonalWAB](https://arxiv.org/html/2410.17236v2)
  - FingerTip uses Qwen2.5-VL-7B with LoRA rank 4/64 on pooled 16k episodes. — [FingerTip](https://arxiv.org/html/2507.21071v2)
  - **RePAIR** uses RL with personalization-aware interaction rewards on an SFT parent. It gives +5.87 user-conditioned subgoal achievement, +7.50 all-success and +9.42 task SR on *unseen users*. — [PAIR](https://arxiv.org/abs/2610.07972)
  - **TIPO** (Zhixin Lin et al., arXiv 2604.11259) is privacy-preference optimization for privacy-first vs utility-first personas, with preference-intensity weighting and padding gating for variable-length trajectories. It reports 65.60% SR, 46.22 Compliance and 66.67% PD. Whether it trains per persona or shared is not stated (UNVERIFIED). — [arXiv](https://arxiv.org/abs/2604.11259)
- **Per-user PEFT for text LLMs (the methodological precedent).**
  - **OPPU**, "Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning" (Tan et al., EMNLP 2024, arXiv 2402.04401), gives each user a PEFT module (typically under 1% of parameters) trained on that user's history.
  - It significantly beats retrieval-based methods under **user behavior shift**, and it works with PEFT types beyond LoRA.
  - — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.372); [arXiv](https://arxiv.org/abs/2402.04401v3)
  - Follow-ups critique the per-user-LoRA storage cost (PLUME, arXiv 2609.04715) and move to shared anchors or latent vectors (PRISP, arXiv 2601.06471; LatentPersonal, arXiv 2609.33182). — [PLUME](https://arxiv.org/pdf/2609.04715); [PRISP](https://arxiv.org/pdf/2601.06471); [LatentPersonal](https://arxiv.org/html/2609.33182v1)
- **On-device infrastructure.** MobileFineTuner (arXiv 2512.08211) supports on-phone LoRA fine-tuning; its case study is a health agent, not a GUI agent. LoRA-generating hypernetworks (arXiv 2609.24979) personalize on-device LLMs using forward passes only. — [MobileFineTuner](https://arxiv.org/abs/2512.08211); [Hypernetwork LoRA](https://arxiv.org/abs/2609.24979v1)

### Inferences
- The project's per-user / per-skill LoRA idea is the GUI-agent analogue of OPPU and appears **unexplored for CUAs**.
- ExpActivator's finding that appended history mostly helps step 1 gives a principled argument for parametric adaptation: weights influence *every* step's policy, while context helps mainly at the start. This is a hypothesis the project could test directly by measuring per-step accuracy, not only end success.
- Reviewers will expect non-parametric baselines on the same 3–5 demos:
  - (a) demos in context (AdaptAgent / LearnAct style)
  - (b) retrieved demo or SOP memory (IFRAgent style)
  - (c) a profile or summary prompt
  - (d) a shared LoRA trained on all users' demos (FingerTip / PUMA style)
  - The case for per-user adapters is strongest if they beat (d) on preference adherence without losing general task success.
- OPPU follow-ups argue for shared-subspace alternatives to per-user LoRA because of storage costs. The project's "per-skill" adapter variant, or a shared base plus a tiny per-user delta, is a natural ablation.

### Gaps
- AdaptAgent's meta-learning details are not on the abstract page. It is UNVERIFIED whether its open-weight adaptation is gradient fine-tuning at test time (FOMAML-style) or only meta-trained in-context learning. This matters because gradient adaptation from ≤2 demos would be the closest technical precedent.
- TIPO's per-persona vs shared training is unknown.
- No source found reporting catastrophic-forgetting or general-capability retention numbers for few-shot LoRA on GUI VLMs.

---

## Q3. How do they handle underspecified instructions, and how do they measure preference adherence separately from task success?

### Takeaway
There are three main strategies for underspecification:
1. **Infer from history.** Logs, profiles, retrieved past trajectories.
2. **Ask.** An `ask_user` action answered by an LLM user simulator that holds a hidden profile.
3. **Act proactively or abstain**, for routines.

The 2026 benchmarks separate "did the job" from "did it the user's way" in several ways:
- iPrefBench: action completion vs personalization score, split into inferable vs ask-only.
- FingerTip: SR vs Sim2 trajectory-similarity ratio.
- KnowU-Bench: rule-based state checks plus an LLM rubric.
- PrefDiag-Bench: preference requirements built into programmatic checkers.

### Cited Findings
- **iPrefBench** labels every omitted detail as *inferable* or *ask-only*. It reports action completion (up to 91.6%) separately from personalization (up to 69.4%), and measures ask-only recall (≤ about 44%) and question precision (about 60–77%). Showing more history improves inferable personalization but not ask-only recall. A pre-execution clarification phase raised Sonnet 5 from 39.2% to 53.8%. — [iPrefBench](https://arxiv.org/html/2610.34735v1)
- **KnowU-Bench** uses ambiguous instructions such as "order lunch," a hidden profile, `ask_user`, and the hybrid score S = λ·S_rule + (1−λ)·S_llm. It does not have a dedicated preference-only metric; average score captures partial alignment and binary success captures full completion. Asking more questions does not by itself help (Seed asks about 2× as many as Claude and scores lower). — [KnowU-Bench](https://arxiv.org/html/2604.08455)
- **PrefDiag-Bench** builds 5 preference-conditioned variants of each basic task, so personalized vs basic success can be compared within the same task. Success requires all preference requirements, checked programmatically. — [PrefDiag-Bench](https://arxiv.org/html/2610.30573)
- **FingerTip** compares Sim2 (similarity to the user's own trajectory relative to other users) with SR2. Most models have Sim2 ≈ 1, i.e., they ignore individual style. LoRA training raised Sim2 from 0.95 to 1.42. — [FingerTip](https://arxiv.org/html/2507.21071v2)
- **MobileIAR (IFRAgent)** introduces the Intention Alignment Rate: agreement with human-intent-aligned actions, distinct from ground-truth step completion. — [arXiv](https://arxiv.org/abs/2508.08645)
- **Persona2Web** uses a "clarify-to-personalize" design: ambiguity is resolved from user history rather than explicit instructions, at varying ambiguity levels. — [arXiv](https://arxiv.org/abs/2602.17003)
- **Magentic-UI** co-planning: the orchestrator may ask one directed clarifying question before planning. Users valued co-planning especially for "subjective preferences that are hard to predict." — [Magentic-UI](https://arxiv.org/html/2507.22358)

### Inferences
- For the project's mock apps, a clean design combining iPrefBench and PrefDiag would be:
  - **Task-success checks:** e.g., an email was sent with the right attachment.
  - **Separate preference checks:** recipient list, folder path, subject format, CC rules, timing. Tag each with PrefDiag's type (Outcome / Constraint / Ranking / Process) and with iPrefBench's *inferable-from-demos* vs *not inferable* label.
  - Report task SR, preference-adherence rate, and the joint "SR ∧ all preferences" rate.
- Because the project learns from demos rather than asking, it should include a few **ask-only / not-demonstrated** preferences. These test whether the adapter wrongly over-applies demonstrated habits, which is a trust-relevant failure.
- Preference drift (iPrefBench has 10 such tasks) and environment unavailability (13) are good stress tests for a frozen per-user adapter. OPPU's claim that parametric personalization handles behavior shift better than retrieval gives a hypothesis to test.

### Gaps
- No benchmark found reports how adherence scales with the *number of demonstrations* (e.g., 1 vs 3 vs 5).
- I did not verify Persona2Web's or PSPA-Bench's exact separation of personalization vs success metrics.

---

## Q4. Does any paper already do per-user LoRA fine-tuning of a computer-use agent from that user's own demonstrations?

### Takeaway
**No.** As of 2026-11-20, I found no paper that trains a separate LoRA (or other PEFT) adapter per user, or per user skill, for a GUI / web / computer-use agent from that user's own few demonstrations. This held across six searches with differing phrasing (per-user LoRA GUI agent; few-shot demonstrations adapter; on-device personalized fine-tuning mobile GUI agent; etc.) and the related-work lists of KnowU-Bench, iPrefBench and PrefDiag-Bench. The novelty claim appears to hold, with five adjacent threats that must be cited and distinguished.

### Cited Findings
- **Closest threat 1: FingerTip 20K (ICLR 2026).** LoRA on Qwen2.5-VL-7B on real users' mobile trajectories to improve personalized execution (Sim2 0.95 → 1.42). It is **shared across users, not per-user**, and uses about 16k episodes. — [FingerTip](https://arxiv.org/html/2507.21071v2)
- **Closest threat 2: PersonalWAB / PUMA (WWW 2025).** LoRA SFT plus DPO on LLaMA2-7B for personalized web function calls. It is shared across 1,000 users, uses no GUI, and covers shopping only. — [PersonalWAB](https://arxiv.org/html/2410.17236v2)
- **Closest threat 3: AdaptAgent (2024).** Adapts open-weight multimodal web agents from ≤2 demos via meta-learning. Adaptation is to websites/domains, not users; the per-user framing appears only as motivation. — [AdaptAgent](https://arxiv.org/abs/2411.13451)
- **Closest threat 4: OPPU (EMNLP 2024).** One PEFT module per user, the exact architectural pattern, but for text personalization tasks, not agents. — [OPPU](https://aclanthology.org/2024.emnlp-main.372)
- **Closest threat 5: IFRAgent / "Quick on the Uptake" (2025–26).** Learns personal implicit intents from a user's own demonstrations for mobile agents, but via a habit repository plus RAG and a (shared) distilled query rewriter, not per-user weights. — [arXiv](https://arxiv.org/abs/2508.08645)
- **Explicitly flagged as open.** The ALLOY authors (CMU/Notre Dame HCI, Toby Jia-Jun Li's group) name "continuously fine-tuning the base model on user-provided demonstrations" as a promising direction they did not pursue (search snippet). — [ALLOY PDF](https://arxiv.org/pdf/2510.10049)
- PSPA-Bench notes that fine-tuning approaches "require large amounts of data and significant computational resources" (search snippet). This is a likely reviewer objection, which the project's few-demo LoRA directly addresses. — [PSPA-Bench PDF](https://arxiv.org/pdf/2603.29318)

### Inferences
- The defensible novelty statement is: "*first per-user/per-skill PEFT adapters for a GUI VLM agent trained from a handful (3–5) of the user's own demonstrations, evaluated on preference adherence in underspecified productivity tasks.*" Avoid claiming "first personalized GUI agent," "first personalized benchmark," or "first LoRA for personalized GUI agents."
- Given that the field moved fast in 2026, a quick arXiv check right before submission is advisable. Two weeks is enough time for a competing preprint to appear.

### Gaps
- Industry systems that might do per-user fine-tuning without publishing (e.g., on-device assistants) cannot be checked.
- arXiv listings from the last 2–3 weeks may be incompletely indexed by search. UNVERIFIED that nothing appeared in Nov 2026.

---

## Q5. Trust / human-in-the-loop / oversight for CUAs, and does demonstration-based teaching build trust?

### Takeaway
HITL CUA work focuses on:
- **co-planning and co-tasking** (Magentic-UI, CowPilot, Cocoa)
- **action guards / approvals** (Magentic-UI)
- **comparative oversight strategies** (Chen et al., 2026, N=48)

The findings:
- Trust is **context-dependent**.
- Plan-level review reduces how often problematic actions occur, but users intervene on only about 21% of problematic actions that reach execution.
- Users want approvals only for high-risk actions.
- Users set **local delegation boundaries** rather than trusting the agent wholesale.

Direct evidence that *teaching by demonstration builds trust in CUAs* is thin:
- ALLOY (N=12) shows demonstrations capture procedural preferences better than prompts.
- ReUseIt ties trust to accurate error identification.
- The strongest "teaching → trust" evidence (N=160) comes from robotics, not CUAs.

### Cited Findings
- **Magentic-UI: "Towards Human-in-the-loop Agentic Systems"** (Mozannar, Bansal, Tan, Fourney, Dibia, ... Kamar, Amershi; Microsoft Research; arXiv 2507.22358, Jul 2025; preprint). — [arXiv](https://arxiv.org/abs/2507.22358)
  - **Mechanisms:**
    - Co-planning: an editable plan that the user must accept before execution.
    - Co-tasking: the user can take over the browser.
    - Multi-tasking.
    - **Action guards:** each action is tagged always / never / maybe irreversible. "Maybe" actions go to an LLM judge. Off-allow-list sites need approval.
    - Answer verification.
    - **Saved plans:** an opt-in LLM turns a finished execution trace into a reusable (task, plan) pair, retrieved for new tasks. This is the non-parametric analogue of "skills."
    - — [arXiv HTML](https://arxiv.org/html/2507.22358)
  - **Simulated users (GAIA validation, about 162 tasks, GPT-4o):** autonomous 30.3% → **51.9% with a simulated user holding side information** (+71% relative), with help requested in only about 10% of tasks. A smarter-model simulated user gave 42.6%. — [arXiv HTML](https://arxiv.org/html/2507.22358)
  - **User study (N=12):**
    - SUS 74.58. Only 41.7% would use it frequently.
    - Users wanted approvals for payments, email and subscriptions, but found approvals for low-risk actions (add to cart) excessive.
    - Co-tasking was used to keep control over sensitive data.
    - Latency was the top complaint.
    - Users learned the agent's capabilities by trial and error.
    - — [arXiv HTML](https://arxiv.org/html/2507.22358)
  - **Safety:** 0 of 24 adversarial scenarios succeeded with default mitigations. With mitigations off, prompt injections achieved SSH-key exfiltration, among other things. — [arXiv HTML](https://arxiv.org/html/2507.22358)
  - **Relation:** Its "saved plans" are a direct non-parametric alternative to per-skill adapters and a natural comparison point. It does **not** learn from user demonstrations, only from execution traces.
- **CowPilot** (Faria Huq, Zora Zhiruo Wang, Frank F. Xu, Tianyue Ou, Shuyan Zhou, Jeffrey P. Bigham, Graham Neubig; CMU; NAACL 2025 System Demos; arXiv 2501.16609).
  - The agent proposes steps and the human can pause, reject, override or hand control back.
  - Case studies on 5 websites: collaborative mode reached **95% success with humans doing only 15.2% of steps**.
  - The authors suggest it as a data-collection tool.
  - — [arXiv](https://arxiv.org/abs/2501.16609)
  - **Relation:** A CMU-local tool that could also serve as the project's demonstration recorder, or as a source of interleaved human/agent demos.
- **"Comparing Human Oversight Strategies for Computer-Use Agents"** (Chaoran Chen, Zhiping Zhang, Zeya Chen, Eryue Xu, Yinuo Yang, Ibrahim Khalilov, Simret A. Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li; arXiv 2604.04918). Search results list it as CHI 2026, but the fetched arXiv page does not state a venue (venue UNVERIFIED). — [arXiv v2](https://arxiv.org/html/2604.04918v2); [search listing](https://arxiv.org/html/2604.04918v1)
  - **Design:** N=48 within-subjects, 192 live sessions on 6 replica services, each with one embedded problematic action (privacy leak, prompt injection, dark pattern). The agent was Gemini 3.1 Flash-Lite.
  - **Four strategies compared:** Action Confirmation (Claude-Computer-Use-like), Risk-Gated (Operator-like), Supervisory Co-Execution (Magentic-UI-like), and Structurally Enriched.
  - **Results:**
    - Trust is context-dependent: Action Confirmation is more trusted on low-stakes tasks, and the reverse holds on high-stakes tasks (interaction p=.006).
    - Plan-based strategies reduced how often problematic actions occurred (56–60% vs 83–88%; OR=0.24).
    - **Users blocked only 29 of 138 (21%) problematic actions that reached execution.**
    - Even the best strategy left 45.8% of sessions with an uncorrected problem.
    - — [arXiv](https://arxiv.org/html/2604.04918v2)
  - **Qualitative findings:**
    - "Approve similar" shortcuts granted broader authority than users expected.
    - Approving a plan reduced scrutiny during execution.
    - Users checked whether the procedure was followed rather than whether the step should happen at all.
    - Users set **local delegation boundaries**.
    - Design implications: treat delegation as an explicit, renewable boundary, and treat plans as trajectory constraints rather than authorization.
    - — [arXiv](https://arxiv.org/html/2604.04918v2)
- **ALLOY: "Generating Reusable Agent Workflows from User Demonstration"** (Jiawen Li, Zheng Ning, Yuan Tian, Toby Jia-Jun Li; arXiv 2510.10049, Oct 2025; cs.HC).
  - Programming-by-demonstration for LLM web agents: users express *procedural preferences* through demonstrations, which become visual, editable, generalizable workflows.
  - User study, N=12: outperformed prompt-based agents and manual workflows "in capturing user intent and procedural preferences." Demonstration complements rather than replaces prompting.
  - Trust-specific findings are not in the abstract (UNVERIFIED).
  - — [arXiv](https://arxiv.org/abs/2510.10049)
  - **Relation:** The strongest HCI evidence for the project's premise that demonstrations convey preferences better than prompts. It uses workflows, not weight updates.
- **ReUseIt** (arXiv 2510.14308) synthesizes reusable web-agent workflows. Its user study found that **accurate agent error identification is critical for user understanding and trust**, and it recommends showing which UI elements the agent considered (search snippet). — [arXiv PDF](https://arxiv.org/pdf/2510.14308)
- **Teaching → trust (robotics, not CUAs).** In an N=160 study, participants reported significantly more trust, believability and emotional response when they *taught* a robot than when they only watched it learn (search snippet). — [ResearchGate](https://www.researchgate.net/publication/369195572_People_Dynamically_Update_Trust_When_Interactively_Teaching_Robots)
- **Other CHI 2026 items (abstract or snippet only, UNVERIFIED details):**
  - "Characterizing Unintended Consequences of GUI Agents For Web Browsing" reports erosion of trust and hindrance to adoption as participant themes. — [ACM DL](https://dl.acm.org/doi/10.1145/3772318.3790696)
  - "Overseeing Agents Without Constant Oversight" (arXiv 2602.16844). — [arXiv PDF](https://arxiv.org/pdf/2602.16844)
  - ADeptS-Bench (arXiv 2608.26204) measures CUA trustworthiness across devices. — [arXiv](https://arxiv.org/abs/2608.26204)

### Inferences
- The project's trust motivation is supported mainly *indirectly*. Users want control over subjective preferences (Magentic-UI). Demonstrations capture procedural preferences better than prompts (ALLOY, N=12). Teaching increases trust in robots (N=160). No study measures whether *demonstration-trained CUAs* earn more calibrated trust. The proposal should say this explicitly as a gap it motivates, not claim it as established.
- The oversight study's warning that approving a plan reduces scrutiny, and that "approve similar" over-broadens authority, applies directly to per-skill adapters. A "send the weekly update" adapter is effectively a standing delegation. That suggests:
  - (a) scoping each adapter to a named skill, an explicit and renewable delegation boundary
  - (b) keeping action guards (Magentic-UI style) on irreversible steps such as send or delete, even when the adapter is confident
- A cheap trust-related evaluation within 2 weeks: report the rate of *wrong-but-confident irreversible actions*, e.g., sending to the wrong contact or filing to the wrong folder, separately from preference misses. Also report how often the agent applies a demonstrated habit where it should not. Both are measurable programmatically in the mock apps.

### Gaps
- I found **no CUA-specific user study measuring trust as a function of teaching by demonstration** vs instruction. This is a genuine gap, and evidence about it remains indirect.
- UNVERIFIED leads not checked in this pass:
  - Cocoa, co-planning/co-execution (K. Feng et al., arXiv 2412.10999)
  - GUM, "Creating General User Models from Computer Use" (Shaikh et al., arXiv 2505.10831)
  - DITTO, aligning LLMs with fewer than 10 user demonstrations (Shaikh et al., arXiv 2406.00888; text-only, but a close methodological cousin for "few demos → per-user alignment")
  - SUGILITE / PUMICE (Toby Li et al., CMU), classic smartphone programming-by-demonstration. PUMICE appeared in a search snippet with a 10-user usability (not trust) study.
- The venue of "Comparing Human Oversight Strategies" (CHI 2026) appears only in a search-result listing and is not confirmed on the arXiv page.
