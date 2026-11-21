# Few-Shot / Demonstration-Based Adaptation of GUI, Web, Mobile, and Computer-Use Agents

Scope: prior work on adapting GUI agents from a handful of human demonstrations, covering parametric routes (fine-tuning, meta-learning, LoRA) and non-parametric routes (in-context demos, demo retrieval). The goal is to judge how much this work threatens the novelty of the planned "record 3–5 demos → train a per-skill/per-user LoRA on Qwen2.5-VL-7B / UI-TARS-1.5-7B → adapter library → swap at inference" system. Research date: 2026-11-20.

Bottom line, stated up front: (1) all four papers the team cited exist. Three match the team's descriptions; the UI-Mate description is partly misleading (see Q1). (2) **AdaptAgent (ACL 2025) already LoRA-adapts an open VLM web agent (CogAgent-18B) to each new website from 2 human demonstrations, after a meta-learning (FOMAML) stage.** It is the strongest threat to any claim that "LoRA from a few demos" is new. (3) I found **no paper that runs a same-model, head-to-head comparison of LoRA on ≤10 user demos versus in-context demos for a GUI agent**, and **no paper that does the full record → per-skill/per-user LoRA → adapter library → swap/route loop for a computer-use agent**. Those two points are the project's most defensible contribution.

---

## Q1. Verifying the team's four cited papers (AdaptAgent, LearnAct, UI-Mate, WebSTAR)

### Takeaway
All four papers exist and are verifiable. AdaptAgent and LearnAct match the team's summaries. WebSTAR matches too, but it is a large-scale data-synthesis/SFT paper, not a few-shot method. **UI-Mate is a 2026 Tencent foundation agent that uses its single demonstration *in context*.** Its SFT is large-scale training on demonstration-augmented data, not fine-tuning on the user's own demo, so "demonstration-guided GUI agents with SFT, 1 demo" should be reworded to avoid implying per-demo fine-tuning.

### Cited Findings

#### 1. AdaptAgent: VERIFIED (matches team summary; note the venue history)
- **Title:** "AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations". arXiv:2411.13451 (Nov 2024) — [arXiv](https://arxiv.org/abs/2411.13451v1)
- **Authors:** Gaurav Verma (Georgia Tech); Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Tucker Balch, Manuela Veloso (J.P. Morgan AI Research) — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- **Venue:** a NeurIPS 2024 workshop paper ([NeurIPS virtual page](https://neurips.cc/virtual/2024/104890)), later published at ACL 2025 (long), anthology ID 2025.acl-long.1008 ([ACL Anthology](https://aclanthology.org/2025.acl-long.1008)). The year is 2024 (arXiv/workshop) or 2025 (ACL), so cite the ACL 2025 version.
- **What it does:** adapts multimodal web agents to unseen websites/domains from 1–2 human demonstrations. Proprietary models (GPT-4o via SeeAct) use in-context multimodal demos. Open-weights models (CogAgent-18B) are first meta-trained with FOMAML and then fine-tuned on the demos — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- **Parametric details (most relevant to the project):** FOMAML meta-training on 292 Mind2Web train tasks (4 per website × 73 websites; each meta-batch is one website, 2 inner-loop and 2 outer-loop tasks). At test time the meta-trained model is **fine-tuned on 2 demonstration tasks per new website/domain** and evaluated on the website's remaining tasks. **LoRA is used** for fine-tuning, meta-learning, and adaptation (LoRA α=20, lr=1e-5; target modules not stated) — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- **Headline numbers:** +4.18 to +7.21 absolute task success on Mind2Web cross-website/cross-domain and +3.36 to +5.11 on VisualWebArena, over unadapted counterparts. Relative improvement is 21.03%–65.75% per the abstract; the body attributes 21.03–45.40% to Mind2Web and 28.32–65.75% to VWA — [arXiv](https://arxiv.org/abs/2411.13451v1), [search summary of abstract/body](https://arxiv.org/html/2411.13451v1)
- **Mind2Web, Step SR / Overall SR (Cross-Task | Cross-Website | Cross-Domain)** — [arXiv HTML](https://arxiv.org/html/2411.13451v1):
  - CogAgent pretrained: 25.11/2.80 | 21.29/2.11 | 23.32/2.59
  - CogAgent-FT (full Mind2Web train set): 54.43/13.36 | 47.14/12.42 | 55.71/15.20
  - CogAgent-FT (DE, data-equivalent): 50.25/10.43 | 44.27/10.10 | 52.20/13.28
  - **CogAgent-FOMAML + LoRA adapted on 2 demos: 53.32/11.89 | 55.38/16.96 | 57.29/19.66**
  - SeeAct (GPT-4o) zero-shot: 56.31/14.37 | 49.90/15.83 | 53.72/19.49
  - SeeAct + 1 in-context multimodal demo (1-ICMD): 60.37/19.69 | 53.91/22.46 | 57.40/23.97
  - SeeAct\* + 1-ICMD: 61.88/22.46 | 53.98/23.10 | 58.33/24.06
- **VisualWebArena overall SR (human trajectories | live env):** CogAgent pretrained 1.31 | 0.46; CogAgent-FT 8.78 | 6.43; **CogAgent-FOMAML adapted 11.01 | 8.47**; SeeAct zero-shot 18.75 | 17.56; SeeAct + 1-ICMD 22.42 | 21.36; SeeAct\* + 1-ICMD 23.86 | 23.15 — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- **Demo modality:** multimodal (screenshot + action) demos beat text-only demos by 0.95–3.78 absolute overall SR. Example: SeeAct cross-task overall SR is 15.91 with text-only vs 19.69 with multimodal — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- **Demo count:** 1/3/5/10 in-context demos were tested on a 30-task Mind2Web subset. Gains rise slightly, then saturate; exact values appear only in Figure 2 — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- **Threat to novelty: HIGH for the "LoRA adapter from ≤5 human demos on an open VLM" component.** AdaptAgent already trains a per-website LoRA from 2 human demos. It does **not** compare fine-tuning and ICL on the same model with the same demos (the FT baselines use the Mind2Web train set, not per-site demos). It also has no adapter library, routing, or per-user framing — [arXiv HTML](https://arxiv.org/html/2411.13451v1)

#### 2. LearnAct: VERIFIED (matches team summary of "1–3 demos"; author list and year differ by version)
- **Title:** "LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark". arXiv:2504.13805 (Apr 2025) — [arXiv](https://arxiv.org/abs/2504.13805v1)
- **Venue:** Findings of ACL 2026 (San Diego, July 2026), pp. 29820–29843 — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491/)
- **Authors:** the arXiv v1 lists 9 (Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, Wenchao Meng) — [arXiv HTML v1](https://arxiv.org/html/2504.13805v1). The ACL 2026 version lists 14 (adds Yaozhen Liang, WenHao Wang, Siheng Chen, Zhengxi Lu, Yong Liu) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491/). Affiliations reported as Zhejiang University with vivo AI Lab — [search summary](https://arxiv.org/html/2504.13805v1)
- **What it does:** introduces the LearnGUI benchmark (2,252 offline tasks built on AMEX + 101 online tasks built on AndroidWorld, across 20 apps). Each task comes with k support demonstrations, scored by task/UI/action similarity to the query. The LearnAct framework has three agents: DemoParser (extracts knowledge from demos), KnowSeeker (retrieves relevant knowledge), and ActExecutor (demo-enhanced execution). It works purely from visual input — [arXiv HTML v1](https://arxiv.org/html/2504.13805v1), [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491/)
- **Offline (LearnGUI-Offline, action-match accuracy %, 0/1/2/3-shot)** — [arXiv HTML v1](https://arxiv.org/html/2504.13805v1):
  - Gemini-1.5-Pro: 19.3 / 51.7 / 55.6 / 57.7 (+32.4 with 1 demo)
  - UI-TARS-7B-SFT: 77.5 / 82.8 / 81.9 / 82.1 (+5.3 with 1 demo; no gain beyond 1)
  - Qwen2-VL-7B: 71.8 / 77.3 / 78.5 / 79.4 (+5.5 → +7.6)
- **Online (LearnGUI-Online, task SR %, 1-shot retrieval):** Qwen2-VL-7B 9.9 → 21.1 (+11.2); UI-TARS-7B-SFT 18.1 → 32.8 (+14.7). References: GPT-4o 34.5, Claude Computer-Use 27.9, Aguvis-72B 26.1, Gemini-Pro-1.5 22.8 — [arXiv HTML v1](https://arxiv.org/html/2504.13805v1)
- **ACL 2026 version:** six backbones, e.g., Gemini-2.5-Pro 38.5% → 58.9% — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491/)
- **Parametric component (often missed):** the open models (Qwen2-VL-7B, UI-TARS-7B-SFT) were **LoRA fine-tuned on the LearnGUI-Offline training set**. Settings: rank 64, α 128, dropout 0.1, all modules except the frozen vision encoder, lr 1e-5 cosine, effective batch 8, 1 epoch, 8× L40S. Data was 2,001 tasks per k-shot configuration (k=1,2,3), 26,184 step actions total. The paper does **not** directly compare fine-tuning against in-context demos — [arXiv HTML v1](https://arxiv.org/html/2504.13805v1). One search-engine summary claimed LearnAct includes "a demo-specific fine-tuning comparison", which conflicts with the fetched paper text — [search summary pointing to ACL PDF](https://aclanthology.org/2026.findings-acl.1491.pdf)
- **Threat to novelty: MEDIUM.** LearnAct is the canonical few-shot mobile-demo benchmark and framework, and it shows large ICL gains. Its LoRA is large-scale training that teaches the model to *use* demos, not per-user/per-skill adapters trained on 3–5 demos. It has no adapter library.

#### 3. UI-Mate: VERIFIED as existing, but the team's description needs correcting
- **Title:** "UI-Mate: Advancing Open-Weight Foundation GUI Agents with In-Context Demonstrations". arXiv:2608.15930, submitted **16 Aug 2026** (very recent) — [arXiv](https://arxiv.org/abs/2608.15930)
- **Authors:** Zihan Ding, Longxu Dou, Qi Gao, Xiangwu Guo, Shengchao Hu, Zilong Huang, Zihang Jiang, Lei Ke, Mengcheng Lan, Weixian Lei, Hanxuan Li, Honglin Li, Xiyun Li, Zaitang Li, Leowei Liang, Xin Luo, Haozhe Ma, Jiayi Mao, Zhoujie Pan, Can Qin, Tianyuan Qu, Weiqi Wang, Wenkai Wang, Yonglin Wang, Yuxin Wang, Chenxu Wu, Yingchen Yu, Chenyu Zhang, Yuhao Zheng (Tencent HY team per model hub) — [arXiv](https://arxiv.org/abs/2608.15930), [Hugging Face](https://huggingface.co/tencent/UI-Mate-democua-27B)
- **What it does:** a foundation GUI agent with an environment-grounded training pipeline (automated task generation, rollouts, filtering, SFT, online RL) plus in-context demonstration learning. Demonstrations are converted into subtask-level workflows, and the agent re-plans from the live UI (it follows, skips, or adapts demo steps rather than replaying them) — [arXiv](https://arxiv.org/abs/2608.15930), [search summary](https://huggingface.co/tencent/UI-Mate-democua-27B)
- **How the demo variant is trained:** UI-Mate-democua-27B (base Qwen3.6-27B) starts from an RL checkpoint and gets SFT on a mix of general computer-use data and **demonstration-augmented** data. A demo is a recorded successful run (every key/pointer action, with before/after screenshots), normalized and VLM-annotated into named subtasks with goals and completion criteria. At inference the model sees workflow progress, the current subtask, and that subtask's action list. **One demonstration per task** at inference. Apache-2.0, code at github.com/Tencent/UI-Mate — [Hugging Face model card](https://huggingface.co/tencent/UI-Mate-democua-27B)
- **Numbers (instruction-only → with 1 demo):** OSWorkerBench self-demo subset (33 tasks) strict success 17.17 → 35.35 (+18.18 pp), progress 67.85 → 81.14; OSWorld-subset progress 40.27 → 65.75 (+25.48); GameDev score 76.76 → 81.15 — [Hugging Face model card](https://huggingface.co/tencent/UI-Mate-democua-27B), [arXiv](https://arxiv.org/abs/2608.15930)
- **General results (no demo):** UI-Mate-27B reaches 77.0% on OSWorld-Verified and 66.2% on WindowsAgentArena. OSWorkerBench (100 long-horizon office tasks, 41 apps) overall 41.0% strict / 76.9% progress, +17.7 strict over the Qwen3.6-27B base — [arXiv](https://arxiv.org/abs/2608.15930)
- **OSWorkerBench demo splits:** 33 self-demo tasks (successful strong-agent rollouts of the same task) and 45 variant-demo tasks (human recordings of related but non-identical tasks) — [arXiv](https://arxiv.org/abs/2608.15930)
- **Discrepancy vs team's summary:** the team wrote "demonstration-guided GUI agents with SFT, 1 demo". The 1 demo is correct. The SFT, however, is offline training at scale so the model can *consume* demos in context; UI-Mate does not fine-tune on the user's demo. The team's description reads as though UI-Mate does per-demo SFT, which it does not — [Hugging Face model card](https://huggingface.co/tencent/UI-Mate-democua-27B)
- **Threat to novelty: MEDIUM-HIGH as a baseline, LOW as a competitor on integration.** It is the strongest current "1 human demo → big gain" result and shows a recorder-style demo format. Reviewers will expect it, or LearnAct-style ICL, as the non-parametric baseline. It does not train adapters.

#### 4. WebSTAR: VERIFIED (matches "large-scale GUI trajectory SFT"; not a few-shot paper)
- **Title:** "WebSTAR: Scalable Data Synthesis for Computer Use Agents with Step-Level Filtering". arXiv:2512.10962 (v1 22 Nov 2025; v3 4 Feb 2026) — [arXiv](https://arxiv.org/abs/2512.10962v3). Listed in ACL 2026 main (long), ID 2026.acl-long.21 — [ACL Anthology](https://aclanthology.org/2026.acl-long.21/)
- **Authors:** Yifei He, Pranit Chawla, Yaser Souri, Subhojit Som, Xia Song (UIUC and Microsoft per search summary) — [arXiv](https://arxiv.org/abs/2512.10962v3)
- **What it does:** synthesizes CUA trajectories from OpenAI's computer-use-preview on OpenWebVoyager queries, adds reasoning, grades **each step**, and keeps only correct steps. Trajectory-level success filtering lets flawed intermediate steps through. Also releases WebSCORE (step-grading data) and StepRM (a 7B process reward model distilled from o4-mini) — [arXiv](https://arxiv.org/abs/2512.10962v3), [project site](https://yifei-he.github.io/webstar-website)
- **Numbers:** 13.3K trajectories / 267K steps (v1 reported ~100K steps, so the counts changed across versions). WebSTAR-7B (Qwen2.5-VL-7B, SFT only) gets **47.0%** average SR on WebVoyager vs **30.0%** for UI-TARS-1.5-7B — [project site](https://yifei-he.github.io/webstar-website), [search summary](https://arxiv.org/html/2512.10962v1)
- **Threat to novelty: LOW.** WebSTAR is a recipe and data reference: it shows SFT of Qwen2.5-VL-7B on CUA trajectories and offers a step-level filtering idea that would apply to cleaning user demos. It is not about few-shot adaptation.

### Inferences
- The team's citation list is accurate overall. The most important correction is UI-Mate, which should be cited as **in-context** demo use, with SFT only for learning to read demos. The second is AdaptAgent, which should be cited as a **direct parametric precedent** (LoRA from 2 human demos per website), not just as an ICL paper.
- AdaptAgent's own tables show its open-weights LoRA path (CogAgent, overall SR 16.96 cross-website) still trails GPT-4o + 1 in-context demo (22.46–23.10). Even with meta-learning, few-demo LoRA on an 18B VLM did not beat a frontier model using ICL. That is a realistic expectation-setter for the project.
- LearnAct shows a ceiling effect for GUI-specialized 7B models offline: UI-TARS-7B-SFT gains only +5.3 from 1 demo, and 2–3 demos add nothing. The online SR gain (+14.7) is much larger. The project should evaluate in an executable environment (its mock apps), not only with offline step accuracy.

### Gaps
- I could not confirm the exact NeurIPS 2024 workshop name for AdaptAgent, or the LoRA target modules and rank it used.
- AdaptAgent's Figure 2 values (1/3/5/10-demo curve) were not available as text.
- AdaptAgent does not report whether plain LoRA on 2 demos *without* FOMAML meta-training helps or hurts; no such row appeared in the fetched tables.
- WebSTAR-32B and Online-Mind2Web numbers were not available as text.
- UI-Mate's variant-demo (human, related-task) results were not found in the fetched pages; only self-demo results were.

---

## Q2. Other work on few-shot GUI-agent adaptation from human demonstrations (non-parametric and parametric)

### Takeaway
The non-parametric side is crowded: trajectory exemplars (Synapse), demo retrieval and parsing (LearnAct), instruction extraction from one demo (Instruction Agent), recorder → learner → planner pipelines (ShowUI-Aloha), and programming-by-demonstration workflows (ALLOY). The parametric side is thin. AdaptAgent (meta-learning + LoRA on 2 demos) is the only verified GUI paper doing weight updates from a few human demos. SOLO (Sep 2026) updates a small adapter at test time on GUI agents but uses self-generated episodes, not human demos.

### Cited Findings

#### Synapse (non-parametric; trajectory-as-exemplar + exemplar memory)
- "Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control", Longtao Zheng, Rundong Wang, Xinrun Wang, Bo An. ICLR 2024 poster; arXiv:2306.07863 (June 2023); code at github.com/ltzheng/synapse — [ICLR](https://iclr.cc/virtual/2024/poster/18701), [arXiv](https://arxiv.org/pdf/2306.07863), [GitHub](https://github.com/ltzheng/synapse)
- Prompts with *whole trajectories* as exemplars, abstracts state to fit context, and retrieves exemplars from a memory by task similarity — [ICLR](https://iclr.cc/virtual/2024/poster/18701)
- 99.2% average SR on MiniWoB++ (64 tasks, using demos from only 48 tasks; a 10% relative improvement). On Mind2Web, a 56% relative improvement in average step SR over the previous best prompting scheme — [ICLR](https://iclr.cc/virtual/2024/poster/18701)
- **Relation:** Synapse is the canonical demonstration-retrieval baseline, and a "retrieve the user's recorded demo as an exemplar" baseline is essentially Synapse. Threat: LOW-MEDIUM (it is a baseline, not a competitor).

#### Instruction Agent (non-parametric; "show me once")
- "Instruction Agent: Enhancing Agent with Expert Demonstration", Yinheng Li, Hailey Hultquist, Justin Wagle, Kazuhito Koishida. arXiv:2509.07098 (8 Sep 2025). A search summary attributes it to Microsoft; affiliation not shown on the abstract page (UNVERIFIED) — [arXiv](https://arxiv.org/abs/2509.07098)
- Given **a single demonstration**, it extracts step-by-step instructions and follows them, with verifier and backtracker modules for interruptions such as pop-ups. Described as test-time only and training-free, built on existing LLM APIs and open models — [arXiv](https://arxiv.org/abs/2509.07098), [arXiv HTML (search summary)](https://arxiv.org/html/2509.07098)
- Reports a **60% success rate** on a set of OSWorld tasks that all top-ranked agents failed; no head-to-head baseline numbers on the abstract page — [arXiv](https://arxiv.org/abs/2509.07098)
- **Relation:** the closest "record once, replay intelligently" system for desktop CUAs, and it explicitly motivates "personalized trajectories". Threat: MEDIUM to the *user story* (record → agent does the task); LOW to the parametric/adapter contribution.

#### ShowUI-Aloha (non-parametric; recorder → learner → planner → executor)
- "ShowUI-Aloha: Human-Taught GUI Agent", Yichun Zhang, Xiangwu Guo, Yauhong Goh, Jessica Hu, Zhiheng Chen, Xin Wang, Difei Gao, Mike Zheng Shou. arXiv:2601.07181 (12 Jan 2026) — [arXiv](https://arxiv.org/abs/2601.07181)
- Turns in-the-wild desktop screen recordings into structured tasks. The pipeline is a Recorder (screen video + clicks/keys/scrolls), a Learner (VLM captions of interactions), a Planner (uses parsed demos, tracks state), and an Executor (OS-level actions with safety checks) — [arXiv](https://arxiv.org/abs/2601.07181)
- Number of demos, whether any fine-tuning happens, and benchmark numbers were **not available** from the abstract page (UNVERIFIED) — [arXiv](https://arxiv.org/abs/2601.07181)
- **Relation:** its "Recorder" front end is essentially the project's record button. Threat: MEDIUM to the system-integration story (record → learn → act), LOW to the LoRA/library contribution as far as the abstract shows.

#### SOLO, "One Rollout Is All You Get" (parametric; test-time adapter updates on GUI agents, no human demos)
- "One Rollout Is All You Get: Fully Test-Time Adaptation for GUI Agents", Ziqiang Wang, Li Gu, Zhixiang Chi, Linlian Jiang, Zihuan Jiang, Linqiang Guo, Siobhan Reid, Zhi Liu, Yang Wang. arXiv:2609.34321 (28 Sep 2026) — [arXiv](https://arxiv.org/abs/2609.34321)
- Setting: one attempt per task, no ground truth, no extra rollouts. A judge picks likely successes; a proposer-verifier relabels which subtask a failed prefix actually completed. Admitted episodes go into a sliding window, and **a small adapter is updated** by top-K self-distillation on the agent's own predictions — [arXiv](https://arxiv.org/abs/2609.34321)
- Evaluated on **recurring task streams** from WebArena, VisualWebArena, and MobileWorld, with UI-TARS-7B and Qwen3-VL-8B. Improves SR by about 3–6 points over the frozen agent and beats two in-setting memory methods on the web streams — [arXiv](https://arxiv.org/abs/2609.34321)
- **Relation:** the closest recent "update adapter weights per deployment on a 7–8B GUI VLM" work, and it uses the UI-TARS backbone the team is considering. It differs in supervision (self-generated, no human demos) and has no per-skill/per-user library. Threat: MEDIUM. It shows reviewers that small-adapter updates on GUI agents give single-digit gains, and SOLO-style "memory vs weights" comparisons already exist.

#### ALLOY (non-parametric; programming-by-demonstration workflows; HCI)
- "ALLOY: Generating Reusable Agent Workflows from User Demonstration", Jiawen Li, Zheng Ning, Yuan Tian, Toby Jia-jun Li. arXiv:2510.10049 (11 Oct 2025, cs.HC) — [arXiv](https://arxiv.org/abs/2510.10049)
- Users demonstrate web tasks instead of writing prompts. The system produces editable, visualized workflows that generalize across task variations. In a 12-participant study, demonstrations beat prompt-based agents and manual workflows at capturing user intent — [arXiv](https://arxiv.org/abs/2510.10049)
- A search summary says ALLOY names "continuously fine-tuning [the base model] on user-provided demonstrations" as a promising future direction (UNVERIFIED wording) — [search summary of PDF](https://arxiv.org/pdf/2510.10049)
- **Relation:** a "reusable per-user skill from demos" library, but non-parametric. Useful as HCI framing and as a related-work contrast.

#### Other adjacent items (lower confidence / peripheral)
- **Wilbur** (arXiv:2404.05902) trains a model to rank and select demonstrations because raw demos are limited by context window size (search summary only) — [arXiv](https://arxiv.org/html/2404.05902)
- **Personal GUI-agent benchmarks** found by title only (methods UNVERIFIED): PersonalAlign (arXiv:2601.09636), KnowU-Bench (arXiv:2604.08455; reportedly 42 general, 86 personalized, and 64 proactive tasks), PSPA-Bench (arXiv:2603.29318), "Mobile GUI Agent Privacy Personalization with Trajectory Induced Preference Optimization" (arXiv:2604.11259), and "Relevance Does Not Imply Applicability: Experience Activation for Personal GUI Agents" (arXiv:2609.33304) — [PersonalAlign](https://arxiv.org/pdf/2601.09636), [KnowU-Bench](https://arxiv.org/pdf/2604.08455), [PSPA-Bench](https://awesomepapers.io/ai-agents/papers/2603.29318), [TIPO](https://arxiv.org/pdf/2604.11259), [Experience Activation](https://arxiv.org/pdf/2609.33304)
- **MobileForge** (arXiv:2606.19930): "Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization". Title only; method UNVERIFIED — [arXiv](https://arxiv.org/pdf/2606.19930)

### Inferences
- **MAML-style adaptation for web agents** is covered by AdaptAgent (FOMAML + LoRA). I found no other verified GUI paper that meta-learns for few-demo adaptation.
- **Non-parametric "show me once" is the dominant 2025–2026 paradigm** (Instruction Agent, ShowUI-Aloha, LearnAct, UI-Mate). Reviewers will see the project's ICL/retrieval baselines as mainstream and its LoRA route as the contrarian bet, so the project must show when LoRA wins: e.g., context-length/latency savings, robustness to UI drift, or tasks where 3–5 demos give consistent procedural knowledge.
- SOLO plus the project together suggest a natural framing: "human-supervised (demos) vs self-supervised (SOLO) adapter updates" for GUI agents.

### Gaps
- ShowUI-Aloha's quantitative results and whether it fine-tunes were not verifiable from the abstract.
- I did not find a "demonstration retrieval for web agents" paper beyond Synapse, LearnAct's KnowSeeker, and Wilbur. Agent Workflow Memory-style methods were not checked within this tool budget.
- The exact form of SOLO's adapter (LoRA or not) and its per-benchmark numbers were not in the abstract.

---

## Q3. Has anyone trained a LoRA on ≤10 user demonstrations for a GUI agent and compared it against in-context demos?

### Takeaway
**Partially: LoRA on ≤10 demos, yes (AdaptAgent: 2 demos per website, CogAgent-18B, after FOMAML). A same-model, same-demos LoRA vs ICL comparison: not found.** AdaptAgent compares *different* models: ICL on GPT-4o vs LoRA on CogAgent, where GPT-4o + 1 demo wins. LearnAct's LoRA uses thousands of tasks, and neither LearnAct nor UI-Mate fine-tunes on the test-time demos. The controlled comparison on one backbone (Qwen2.5-VL-7B / UI-TARS-1.5-7B) is an open gap the project can claim.

### Cited Findings
- AdaptAgent fine-tunes (LoRA) the FOMAML-meta-trained CogAgent on **2 demos per unseen website/domain**: Mind2Web cross-website overall SR 16.96 and cross-domain 19.66, vs CogAgent-FT on the full train set at 12.42 / 15.20 — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- In the same paper, GPT-4o (SeeAct) + 1 in-context multimodal demo reaches 22.46 / 23.97 overall SR, higher than the CogAgent LoRA path. VWA live: 21.36 for GPT-4o + 1-ICMD vs 8.47 for CogAgent-FOMAML — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- AdaptAgent itself does **not** test fine-tuning vs ICL on the same model with the same demos — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- LearnAct LoRA-tunes Qwen2-VL-7B and UI-TARS-7B-SFT on 2,001 tasks per k-shot configuration (26,184 step actions; r=64, α=128, 1 epoch). This is large-scale training, and the paper does not compare it head-to-head with pure ICL — [arXiv HTML v1](https://arxiv.org/html/2504.13805v1)
- UI-Mate-democua is SFT'd on demonstration-augmented data and uses 1 in-context demo at inference (17.17 → 35.35 strict success); there is no per-demo fine-tuning arm — [Hugging Face model card](https://huggingface.co/tencent/UI-Mate-democua-27B)
- SOLO updates a small adapter on UI-TARS-7B / Qwen3-VL-8B from self-generated episodes and gets about +3–6 SR. It beats two *memory* (non-parametric) methods on web streams. This is the closest "weights vs memory" comparison on GUI agents, but without human demos — [arXiv](https://arxiv.org/abs/2609.34321)
- Non-GUI evidence that per-skill LoRA can replace in-context skill text: Skill-to-LoRA (Qwen3.6-27B, 21 skills from SWE-Skills-Bench) gets +2.9 pp pass rate over no-skill and +5.2 pp over full-skill-text prompting, with 6.6% lower per-step token cost. It matches or beats full skill text on 18/21 skills, and a wrong or shared adapter hurts — [arXiv](https://arxiv.org/abs/2606.16769)
- General NLP evidence: Mosbach et al. 2023, "Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation" (arXiv:2305.16938), is the standard reference for a fair FT-vs-ICL protocol. Details not fetched here — [arXiv](https://arxiv.org/pdf/2305.16938)
- UNVERIFIED: a search summary claimed TimeWarp (arXiv:2603.04949, Ishmam & Marino) reports that LoRA significantly underperforms full fine-tuning for web agents across ranks. The abstract I fetched does not mention LoRA vs FFT. It reports trajectory-training gains (Qwen-3 4B 20.4% → 37.7%; Llama-3.1 8B 0% → 27.0%) — [arXiv](https://arxiv.org/abs/2603.04949)

### Inferences
- Expected effect sizes from the closest analogues: AdaptAgent's few-demo LoRA gains are about **+4 to +7 absolute SR**, and SOLO's adapter updates about **+3 to +6**. In-context single demos on modern models can give **+15 to +18 pp** (UI-Mate self-demo, LearnAct UI-TARS online). The project should therefore expect LoRA-from-3–5-demos to be **competitive with, not dominant over, ICL** on a 7B GUI model. The likely wins are context cost, latency, and robustness when the demo cannot fit or the task is long.
- A key confound to control: AdaptAgent needed meta-training (FOMAML on 292 tasks) to make 2-demo LoRA work well. A plain 3–5-demo LoRA on Qwen2.5-VL-7B / UI-TARS-1.5-7B may overfit, especially to coordinates. Ablations should include LoRA rank/steps, demo count (1/3/5/10), LoRA + ICL combined, and evaluation on held-out *variants* of the recorded task (mirroring UI-Mate's "variant-demo" split), not the identical task.
- The project's cleanest contribution is a **controlled same-backbone comparison of (a) zero-shot, (b) ICL with the same 3–5 demos, (c) LoRA on those demos, (d) LoRA + ICL**, plus adapter-library routing. None of the verified papers has this design.

### Gaps
- No verified paper reports same-model LoRA-on-few-demos vs ICL numbers for any GUI benchmark.
- Whether LearnAct's LoRA-tuned models took demos in their prompt during training is strongly implied by the "per k-shot configuration" data description, but I did not confirm it explicitly.
- The TimeWarp LoRA vs FFT claim could not be verified.

---

## Q4. Does any paper already do the exact end-to-end integration (record demos → per-skill/per-user LoRA → adapter library → swap/route at inference) for a computer-use agent?

### Takeaway
**No such paper found (as of 2026-11-20), so there is no prominent red flag.** Every component exists separately: recording (ShowUI-Aloha, UI-Mate, Instruction Agent), LoRA from few human demos for a GUI agent (AdaptAgent), small-adapter updates on GUI agents at deployment (SOLO), and per-skill LoRA libraries with dynamic loading (Skill-to-LoRA, non-GUI). The closest single threat to the parametric piece is AdaptAgent, which effectively trains one LoRA per website from 2 demos but does not store, route, or swap them as a library.

### Cited Findings
- **AdaptAgent**: per-website LoRA adaptation from 2 human multimodal demos on CogAgent-18B. No library, routing, or per-user framing — [arXiv HTML](https://arxiv.org/html/2411.13451v1)
- **Skill-to-LoRA (S2L)**: "Skill-to-LoRA: From Using Skills to Learning Behaviors for Token-Efficient LLM Agents", Tianyi Zhang, Zhonghao Qi, arXiv:2606.16769 (15 Jun 2026). It trains one LoRA per skill on demos *synthesized* from a SKILL.md, then dynamically loads the matching adapter at inference over a frozen Qwen3.6-27B. The selection mechanism is not described in the abstract. It is evaluated on SWE-Skills-Bench (software engineering, not GUI) — [arXiv](https://arxiv.org/abs/2606.16769). A search summary reports adapters of about 6–24 MB each (UNVERIFIED) — [arXiv HTML](https://arxiv.org/html/2606.16769v1)
- **SOLO**: a small adapter on a GUI agent updated during deployment from self-judged episodes. No human demos, no library — [arXiv](https://arxiv.org/abs/2609.34321)
- **ShowUI-Aloha**: record → learn → plan → execute for desktop. Fine-tuning not mentioned in the abstract — [arXiv](https://arxiv.org/abs/2601.07181)
- **Instruction Agent**: one recorded demo → instructions, training-free — [arXiv](https://arxiv.org/abs/2509.07098)
- Per-user LoRA in non-GUI personalization: "Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation" (arXiv:2610.02353) argues that per-user adapter storage grows with user count and proposes shared capacity instead. It cites OPPU (private adapter per user) and Per-Pcs (assembling/gating adapter pieces). Search summary only, details UNVERIFIED — [arXiv HTML](https://arxiv.org/html/2610.02353)
- Unverified adapter-routing items (title or summary only): MaP / "Towards Omni-dimensional GUI Agent Navigation with Masked Trajectory Prediction" (arXiv:2609.25769), reportedly with token-wise routing among multiple LoRA experts by trajectory-token role. This is not per-skill or per-user — [arXiv](https://arxiv.org/pdf/2609.25769). "Adaptive Minds" (arXiv:2510.15416), reportedly 98.3% routing accuracy over a 30-adapter pool, not GUI — [arXiv HTML](https://arxiv.org/html/2510.15416). "WeAct: Dynamic Weight Switching as Agent Actions", known only from a blog summary — [blog](https://kenashe.ai/blog/2026-10-20-weact-turns-lora-adapters-into-something-an-agent-can-pick-mid-task)

### Inferences
- The integration claim is defensible but should be framed precisely: "first system and evaluation of per-skill/per-user LoRA adapters trained from 3–5 *human GUI recordings*, stored in a library and routed at inference, compared against same-backbone ICL/retrieval baselines." It should explicitly credit AdaptAgent (few-demo LoRA for web agents), Skill-to-LoRA (skill-adapter library, non-GUI), and SOLO (deployment-time adapters for GUI agents).
- The routing/library component is where novelty is highest but evidence is thinnest. Skill-to-LoRA's finding that wrong or shared adapters hurt suggests the project should report **routing accuracy** and **performance under mis-routing**, plus a "single merged/shared adapter" baseline.
- The non-GUI per-user-LoRA debate (2610.02353) is a likely reviewer objection, since storage scales with users. A short discussion of adapter size and count would pre-empt it.

### Gaps
- I may have missed industrial or system papers, or GitHub-only projects, that do record → LoRA for desktop agents. A search for "internalize demonstrations into LoRA GUI agent" returned only non-GUI skill-to-LoRA work and blog posts — [search results incl. S2L](https://arxiv.org/html/2606.16769v1)
- MaP, Adaptive Minds, WeAct, and CUA-Universe (arXiv:2609.05374, reportedly a shared LoRA for hybrid GUI+CLI) were not fetched; their details are UNVERIFIED — [CUA-Universe](https://arxiv.org/pdf/2609.05374)

---

## Q5. Benchmarks, metrics, and typical effect sizes for few-shot demo adaptation

### Takeaway
Web papers use Mind2Web (Element Acc, Step SR, overall/task SR across cross-task/website/domain splits) and VisualWebArena (SR). Mobile uses LearnGUI (offline action-match accuracy; online task SR on AndroidWorld-derived tasks). Desktop uses OSWorld(-Verified), WindowsAgentArena, and OSWorkerBench (strict success + progress, with self-demo and variant-demo splits). Typical gains: parametric few-demo adaptation **+3 to +7 absolute SR**; one in-context demo **+5 to +7 offline step accuracy** for GUI-specialized 7B models but **+11 to +18 SR online/long-horizon**; and very large gains (+20–32) for general-purpose VLMs that start weak at GUI.

### Cited Findings
- **Mind2Web** (cross-task / cross-website / cross-domain; Element Acc, Step SR, Overall SR) and **VisualWebArena** (human-trajectory subset and live env; SR): AdaptAgent +4.18–7.21 abs SR on Mind2Web, +3.36–5.11 on VWA — [arXiv](https://arxiv.org/abs/2411.13451v1)
- **MiniWoB++** and **Mind2Web** (Synapse): 99.2% MiniWoB++ SR, +56% relative step SR on Mind2Web — [ICLR](https://iclr.cc/virtual/2024/poster/18701)
- **LearnGUI-Offline** (2,252 tasks on AMEX; action-match accuracy, 0–3 shots) and **LearnGUI-Online** (101 tasks, 20 apps, built on AndroidWorld; task SR): offline +5.3 (UI-TARS-7B-SFT), +5.5–7.6 (Qwen2-VL-7B), +32.4–38.4 (Gemini-1.5-Pro); online +11.2 (Qwen2-VL-7B), +14.7 (UI-TARS-7B-SFT) — [arXiv HTML v1](https://arxiv.org/html/2504.13805v1); Gemini-2.5-Pro 38.5 → 58.9 — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491/)
- **OSWorkerBench** (100 long-horizon office tasks, 41 apps; strict success + progress; 33 self-demo / 45 variant-demo tasks), **OSWorld-Verified**, **WindowsAgentArena**: UI-Mate 1-demo +18.18 strict success and +13.29 progress on the self-demo subset; OSWorld-subset progress +25.48 — [Hugging Face](https://huggingface.co/tencent/UI-Mate-democua-27B), [arXiv](https://arxiv.org/abs/2608.15930)
- **OSWorld** (Instruction Agent): 60% on a set of tasks that all top agents failed — [arXiv](https://arxiv.org/abs/2509.07098)
- **Recurring task streams** from WebArena / VisualWebArena / MobileWorld (SOLO): +3–6 SR — [arXiv](https://arxiv.org/abs/2609.34321)
- **WebVoyager** (WebSTAR, SFT recipe): Qwen2.5-VL-7B SFT 47.0% vs UI-TARS-1.5-7B 30.0% — [project site](https://yifei-he.github.io/webstar-website)
- **SWE-Skills-Bench** (Skill-to-LoRA, non-GUI): +2.9 / +5.2 pp — [arXiv](https://arxiv.org/abs/2606.16769)

### Inferences
- The project's self-built mock mail/calendar/docs apps resemble the setups above: LearnGUI-Online's small curated online sets (101 tasks) and OSWorkerBench's "self-demo vs variant-demo" split. Borrowing that split (train adapter on demos of task T; test on T and on variants T′) would make results legible to reviewers and directly comparable in spirit to UI-Mate.
- Report both step-level metrics (action-match / Step SR, as in Mind2Web/LearnGUI-Offline) and episode SR in the live mock apps. LearnAct shows the two can diverge sharply (+5 offline vs +15 online for UI-TARS-7B).
- With about 2 weeks left, adding one public benchmark is feasible: a small LearnGUI-Online slice, or a few Mind2Web websites mirroring AdaptAgent's 2-demo protocol. This would anchor the mock-app numbers against the published few-shot baselines.
- I found no few-shot demo-adaptation paper among those verified that uses AndroidControl as its few-shot benchmark. It is used for general mobile-agent training rather than demo-conditioned evaluation, a claim based on absence in the papers checked.

### Gaps
- No standard benchmark exists for *per-user* or *per-skill adapter* evaluation of GUI agents. Personal-GUI benchmarks (KnowU-Bench, PSPA-Bench, PersonalAlign) were found by title only and their protocols were not verified — [KnowU-Bench](https://arxiv.org/pdf/2604.08455), [PSPA-Bench](https://awesomepapers.io/ai-agents/papers/2603.29318), [PersonalAlign](https://arxiv.org/pdf/2601.09636)
- Variance and seed counts in these few-shot papers were not checked; small online sets (33, 45, 101 tasks) imply wide confidence intervals.
