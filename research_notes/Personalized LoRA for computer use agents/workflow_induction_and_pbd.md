# Workflow / Skill Induction from Demonstrations and Programming-by-Demonstration (PBD): Non-Parametric Alternatives to a Per-User LoRA

Scope: non-parametric ways to turn 3–5 user GUI demonstrations into something reusable (workflow memory, skill library, SOP, script, skill file). For each, where it stands relative to the project (per-skill / per-user LoRA on a GUI VLM, swapped in at inference). Coverage runs through 20 Nov 2026. "UNVERIFIED" marks anything I could not confirm against a primary page in this session. Venue claims not printed on the arXiv page are marked UNVERIFIED.

---

## Q1. Agent Workflow Memory (AWM): offline vs online induction, WebArena / Mind2Web results

### Takeaway
AWM is the canonical "induce reusable workflows from examples and put them in memory/prompt" baseline. It needs no training. It reached 35.5% on WebArena with GPT-4, above human-written workflows (SteP, 33.0%). On Mind2Web, offline workflows induced from annotated training examples beat both a supervised-pipeline baseline (MindAct) and raw retrieved trajectories (Synapse). Offline and online induction score about the same in-distribution. Online induction holds up better as the train/test gap widens (cross-domain).

### Cited Findings
- **Paper card.** *Agent Workflow Memory*. Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, Graham Neubig (CMU/MIT). arXiv:2409.07429, Sep 2024. The arXiv page lists no venue; ICML 2025 acceptance is UNVERIFIED. — [arXiv](https://arxiv.org/abs/2409.07429)
- **Summary.** An LM induces "workflows" from past trajectories: commonly reused sub-routines, with example-specific values abstracted into slots such as `{product-name}`. These are added to agent memory and guide later tasks. *Offline*: induce from annotated training examples before test time. *Online*: induce on the fly from test queries the agent itself judged successful. — [arXiv HTML](https://arxiv.org/html/2409.07429)
- **Relative gains (abstract).** +24.6% relative success on Mind2Web and +51.1% relative on WebArena. AWM also reduces steps. Online AWM beats baselines by 8.9 to 14.0 absolute points as the train/test distribution gap widens (cross-task, cross-website, cross-domain). The benchmarks cover 1000+ tasks across 200+ domains. — [arXiv](https://arxiv.org/abs/2409.07429)
- **WebArena (GPT-4), total SR and avg steps.** AWM (online only) 35.5 / 5.9 steps. SteP (human-written workflows) 33.0. BrowserGym 23.5. BrowserGym (ax-tree) 15.0 / 7.9 steps. AutoEval 20.2 / 46.7 steps. WebArena baseline 14.9. That is +12.0 absolute (+51.1% relative) over BrowserGym. — [arXiv HTML](https://arxiv.org/html/2409.07429)
- **Mind2Web offline, cross-task (Elem Acc / Action F1 / Step SR / Task SR).**
  - GPT-3.5: MindAct 20.3/56.6/17.4/0.8; Synapse 34.0/–/30.6/2.4; AWM 39.0/52.8/34.6/2.8.
  - GPT-4: MindAct 41.6/60.6/36.2/2.0; AWM 50.6/57.3/45.1/4.8.
  - Source: [arXiv HTML](https://arxiv.org/html/2409.07429)
- **Offline vs online (GPT-4, Mind2Web).**
  - Cross-task: offline 50.6 EA / 45.1 step SR / 4.8 SR vs online 50.0 / 43.6 / 4.0.
  - Cross-website: offline 41.4 / 33.7 / 2.3 vs online 42.1 / 33.9 / 1.6.
  - Cross-domain: offline 36.4 / 32.6 / 0.7 vs online 40.9 / 35.5 / 1.7.
  - MindAct cross-domain is 21.6 / 18.6 / 1.0.
  - Source: [arXiv HTML](https://arxiv.org/html/2409.07429)
- **Workflows vs raw trajectories in context.** Against Synapse, which retrieves whole training trajectories as exemplars, AWM gains +5.0 element accuracy and +4.0 step SR. The authors attribute this to less "bias on element selection" from abstracting away example-specific context, and to the fact that reusable sub-routines recur more often than full trajectories. — [arXiv HTML](https://arxiv.org/html/2409.07429)
- **Representation.** Each workflow has an NL description plus steps. Each step has an NL state description, reasoning, and an executable action (e.g., `CLICK({submit-id})`). A text-only variant (AWM-text) scores almost the same: 51.2/57.4/45.4/3.6 vs 50.6/57.3/45.1/4.8. Offline, all training examples for a website are concatenated into one induction prompt. The paper induces ~7.4 workflows per example on WebArena and 7.3 on Mind2Web. The WebArena learning curve rises within roughly the first ~40 queries. — [arXiv HTML](https://arxiv.org/html/2409.07429)

### Inferences
- AWM-offline matches the project's setting almost exactly: a small set of annotated demos for one site or app goes into a single LLM call, producing text workflows that go into the prompt. It is the most direct non-parametric control. It is cheap: one induction call per skill or user, with no GPU training.
- Because AWM-text works as well as the code format, a demo-induced workflow can be plain markdown with abstracted parameters. That is essentially a SKILL.md (see Q6).
- The Synapse comparison is the best evidence that *abstracted workflows > raw demos in context* for web agents. Reviewers will expect both a "raw demos in context" arm and an "induced workflow" arm.
- Caveat: all AWM results use frontier text LMs (GPT-3.5/4) over HTML/ax-tree. Whether a 7B GUI VLM benefits similarly from text workflows has not been shown by AWM. That is an opening for the project.

### Gaps
- AWM does not report how performance scales with the *number* of offline examples (e.g., 3 vs 5 vs all). I found no per-count curve.
- AWM never compares against fine-tuning the same base model on the same examples. MindAct is a different (supervised-pipeline) system; the paper does not explain its asterisk, and I could not confirm MindAct's training details from the AWM page.

---

## Q2. SkillWeaver, ASI (Agent Skill Induction / "Inducing Programmatic Skills for Agentic Tasks"), and 2025–2026 skill-library work

### Takeaway
Since AWM, the field has moved from *text* workflows to *executable, verified* skills (Python/Playwright functions, APIs). ASI beats AWM-style text skills by 11.3% and the static agent by 23.5%. SkillWeaver gets +31.8% / +39.8% relative from self-discovered APIs, and those APIs transfer to weaker agents (+54.3%). By 2026, "skills" means Agent-Skills-format markdown bundles. Work2Skill (Oct 2026) induces SKILL.md files from *human desktop GUI recordings* and is the closest existing work to the project's non-parametric arm.

### Cited Findings
- **ASI paper card.** *Inducing Programmatic Skills for Agentic Tasks*. Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, Daniel Fried. arXiv:2504.06821, Apr 2025, rev. Aug 2025. The arXiv page lists no venue; COLM 2025 is UNVERIFIED. — [arXiv](https://arxiv.org/abs/2504.06821)
  - Summary: the agent induces *programmatic* skills (executable functions composed of primitive actions) online and verifies them by execution before adding them to its action space.
  - Numbers: on WebArena, +23.5% success over the static baseline and +11.3% over the text-skill counterpart (AWM-style). Steps drop by 10.7–15.3%. — [arXiv](https://arxiv.org/abs/2504.06821)
- **SkillWeaver paper card.** *SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills*. Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su. arXiv:2504.07079, Apr 2025. Venue UNVERIFIED. — [arXiv](https://arxiv.org/abs/2504.07079)
  - Summary: the agent explores a website, proposes skills, practices them, and distills them into reusable APIs.
  - Numbers: +31.8% relative success on WebArena and +39.8% on real-world websites. APIs synthesized by strong agents improve weaker agents by up to 54.3% on WebArena. — [arXiv](https://arxiv.org/abs/2504.07079)
- **Work2Skill paper card (most relevant 2026 item).** *Work2Skill: Capturing Human GUI Experience for Reusable Agent Skill Induction*. Yuqi Huang, Hui Li, Yifei Cao, Zikun Cai, Tianci Xie, Yangfan Gao, Pinxue Guo, Chaoyun Zhang, Bingxuan Li, Rui Zheng, Hang Yan, Ning Liao. arXiv:2610.12984, 9 Oct 2026. Preprint, no venue. — [arXiv](https://arxiv.org/abs/2610.12984)
  - Pipeline: WorkRecorder aligns screen recordings, mouse/keyboard events, and accessibility trees from human desktop demos. The WorkDemo-350 dataset has 350 demos. An *agentic* induction method inspects the evidence on demand and writes skills that capture workflow rules, step dependencies, and completion checks. — [arXiv](https://arxiv.org/abs/2610.12984)
  - Format: each skill is a **SKILL.md with YAML frontmatter (name, description)**, "the Agent Skills file convention". It is appended to the executor's system prompt. The induction agent follows Anthropic's skill-creator as its authoring policy. — [arXiv HTML](https://arxiv.org/html/2610.12984)
  - Baselines: No Skill; ShowUI-Aloha (text-adapted); Demo2Tutorial (D2T, text-adapted). There is **no raw-demo-in-context, AWM, ASI, or fine-tuning baseline**. — [arXiv HTML](https://arxiv.org/html/2610.12984)
  - Results (Score / SR %, MyPCBench N=64 | OSWorld-Verified subset N=286). — [arXiv HTML](https://arxiv.org/html/2610.12984)
    - GPT-5.5: No Skill 74.73/23.44 | 74.47/70.98 → Work2Skill 85.27/48.44 | 81.97/78.67.
    - Opus-4.8: 81.53/42.19 | 79.23/77.97 → 86.20/50.00 | 82.34/79.37.
    - Qwen3.5-35B-A3B: 38.38/9.38 | 32.52/32.17 → 44.56/20.31 | 36.36/36.01.
    - Qwen3.5-9B: 24.27/6.25 | 27.48/26.92 → 34.11/12.50 | 33.41/32.52.
    - Score gains over No Skill: +4.67 to +10.54 (MyPCBench) and +3.11 to +7.50 (OSWorld-V subset).
  - The demo-based baselines sometimes *hurt*. Opus-4.8 + Aloha SR on MyPCBench is 34.38 vs 42.19 for No Skill. GPT-5.5 + Aloha on OSWorld-V scores 72.57 vs 74.47. Qwen3.5-9B + Aloha SR on MyPCBench is 3.13 vs 6.25. — [arXiv HTML](https://arxiv.org/html/2610.12984)
- **LatentSkill (textual skills vs per-skill LoRA, non-GUI).** *LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents*. Aofan Yu et al. arXiv:2606.06087 (v3, Aug 2026). — [arXiv HTML](https://arxiv.org/html/2606.06087)
  - Method: a hypernetwork compiles a textual skill doc into a LoRA on frozen Qwen3-8B.
  - Baselines: in-context skill text; Full SFT and Shared LoRA (trained on the same teacher trajectories); **Per-skill LoRA**.
  - ALFWorld SR (seen/unseen): In-Context Skill 52.9/56.0; Full SFT 53.6/51.5; Shared LoRA 45.0/47.8; Per-skill LoRA 45.7/50.0; LatentSkill 74.3/69.4.
  - Prefill per step (seen split): 1.21k tokens in-context vs 0.60k per-skill LoRA vs 0.44k LatentSkill.
  - Search-QA EM: in-context 32.6, Full SFT 34.2, Per-skill LoRA 32.8, LatentSkill 35.6.
  - WebShop (OOD): vanilla 10.3, in-context skill 6.7, LatentSkill (SFT) 23.9.
- **Other 2026 leads (search-snippet only, UNVERIFIED, not opened).** *Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval* (arXiv:2606.04391); *Workflow-to-Skill* (arXiv:2606.06893); *Back to Basics: A Skill-Centric Approach to Generalist Web Agents* (reportedly fine-tunes a compact model on skill-specific trajectories); *SoK: Agentic Skills* (arXiv:2602.20867); *Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills* (arXiv:2512.16301). — [search results: 2606.04391](https://arxiv.org/html/2606.04391v1), [2606.06893](https://arxiv.org/pdf/2606.06893), [2512.16301](https://arxiv.org/html/2512.16301v3), [2602.20867](https://arxiv.org/pdf/2602.20867)

### Inferences
- **Novelty threat: Work2Skill.** It already does "human GUI recordings → reusable SKILL.md skills → better desktop agent" across four executors, including small open models (Qwen3.5-9B: +9.8 score and 2× SR on MyPCBench). A reviewer can point to it as the non-parametric answer to the project. The project should cite it, and ideally reimplement a lightweight version as its main baseline.
- **Supports the project: LatentSkill's numbers.** Naive per-skill LoRA trained on trajectories (45.7/50.0) *underperformed* simply putting the skill text in context (52.9/56.0) on ALFWorld. Its main benefit was ~50% lower prefill. If the project's per-skill LoRA only matches a skill file, the honest selling point is cost, latency, and privacy (no plaintext skill), not accuracy. LatentSkill also shows in-context skill text can *hurt* an 8B model out of domain (WebShop 10.3 → 6.7). That is an argument that small GUI VLMs may use long text skills poorly, which favors weights.
- ASI and SkillWeaver need an environment with programmatic/API access (Playwright on web) and exploration budget. They are heavier to implement on a pixel-level GUI VLM than AWM- or Work2Skill-style text skills.

### Gaps
- I could not open Work2Skill's induction prompts, cost (tokens/time per skill), or the remaining 65k characters of the HTML, which may contain ablations such as skills from 1 vs many demos.
- "Demo2Tutorial (Bai et al., 2026)" is cited by Work2Skill, but I could not locate the paper itself.
- I did not verify the numbers in the 2026 leads listed above.

---

## Q3. ECLAIR, WorkflowLLM, demonstrations → SOP, and LearnAct's knowledge extraction from demos

### Takeaway
The enterprise line (ECLAIR, Stanford) treats demonstrations and SOPs as the knowledge carrier and reports that traditional RPA is costly and brittle. The mobile-GUI line (LearnAct, AppAgent, UI-Mate, ShowUI-Aloha) shows that *one* demonstration, parsed into knowledge or in-context guidance, gives large gains: LearnAct took Gemini-1.5-Pro from 19.3% to 51.7%; AppAgent's demo-derived docs reached 84.4% vs 48.9% with no doc; UI-Mate's one self-demo took strict success from 17.2% to 35.4%. WorkflowLLM is the *parametric* counterpart (fine-tuning on 106k workflows), not a few-demo method.

### Cited Findings
- **ECLAIR paper card.** *Automating the Enterprise with Foundation Models*. Michael Wornow, Avanika Narayan, Krista Opsahl-Ong, Quinn McIntyre, Nigam H. Shah, Christopher Ré. arXiv:2405.03710, May 2024. — [arXiv](https://arxiv.org/abs/2405.03710)
  - RPA case studies (a hospital and a large B2B enterprise): 12–18 months set-up, 60% initial execution accuracy, and multiple FTEs for maintenance.
  - ECLAIR reaches 93% accuracy on a workflow-understanding task and 40% end-to-end completion when built from an NL workflow description. — [arXiv](https://arxiv.org/abs/2405.03710)
- **WorkflowLLM paper card.** *WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models*. Shengda Fan et al. (Tsinghua). arXiv:2411.05451, Nov 2024. ICLR 2025 is UNVERIFIED. — [arXiv](https://arxiv.org/abs/2411.05451)
  - Fine-tunes Llama-3.1-8B into WorkflowLlama on WorkflowBench: 106,763 samples, 1,503 APIs, 83 apps, 28 categories, built from Apple Shortcuts and RoutineHub.
  - The abstract gives no numeric comparison with GPT-4o. — [arXiv](https://arxiv.org/abs/2411.05451)
- **LearnAct paper card.** *LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark*. Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, Wenchao Meng. arXiv:2504.13805, Apr 2025. Venue UNVERIFIED. — [arXiv](https://arxiv.org/abs/2504.13805)
  - LearnGUI benchmark: 2,252 offline + 101 online tasks, each with human demos.
  - With one demo, Gemini-1.5-Pro offline accuracy goes from 19.3% to 51.7%. Online, UI-TARS-7B-SFT task success goes from 18.1% to 32.8%. — [arXiv](https://arxiv.org/abs/2504.13805)
  - Knowledge extraction from demos is done by a multi-agent pipeline (DemoParser / KnowSeeker / ActExecutor). These component names are from my memory of the paper; UNVERIFIED in this session.
- **AppAgent: docs generated by watching human demos.** *AppAgent: Multimodal Agents as Smartphone Users*. Chi Zhang et al. arXiv:2312.13771. CHI 2025 is UNVERIFIED. — [arXiv HTML](https://arxiv.org/html/2312.13771)
  - 45 tasks, 9 apps. Success rate (reward, steps):
    - No doc, raw action space: 2.2%.
    - No doc, their action space: 48.9% (3.5, 6.9).
    - Docs from autonomous exploration: 73.3% (5.1, 4.4).
    - **Docs from watching demos: 84.4% (4.7, 5.1).**
    - Manually crafted docs: 95.6% (5.5, 5.5).
  - In demo mode, "a human user operates the apps while the agent observes, recording only the elements and actions employed by the human". — [arXiv HTML](https://arxiv.org/html/2312.13771)
- **UI-Mate paper card.** *UI-Mate: Advancing Open-Weight Foundation GUI Agents with In-Context Demonstrations*. Zihan Ding et al. (29 authors, Tencent). arXiv:2608.15930, Aug 2026. — [arXiv](https://arxiv.org/abs/2608.15930)
  - Converts multimodal demos into *subtask-level workflows*, follows relevant steps, and replans from the live UI.
  - On a 33-task self-demo subset of OSWorkerBench, one demo raises strict success from 17.2% to 35.4% and progress from 67.9% to 81.1%.
  - UI-Mate-27B: 77.0% OSWorld-Verified and 66.2% WindowsAgentArena (self-reported open-weight SOTA). — [arXiv](https://arxiv.org/abs/2608.15930)
  - The GitHub/search summary gave different subset numbers (OSWorld-Subset 40.27% → 65.75%). I cite the abstract. — [GitHub](https://github.com/Tencent/UI-Mate)
- **ShowUI-Aloha.** *ShowUI-Aloha: Human-Taught GUI Agent*. Yichun Zhang, …, Mike Zheng Shou (NUS Show Lab). arXiv:2601.07181, Jan 2026. Pipeline: Recorder → Learner (captions raw interactions) → Planner → Executor. — [arXiv](https://arxiv.org/abs/2601.07181)
  - UNVERIFIED (search summary only): 361 OSWorld-style tasks with one human demo each; removing the TeachTrace component drops success from 63.3% to 36.7%. — [search result / alphaxiv](https://www.alphaxiv.org/overview/2601.07181)
- **Demo representation matters.** *How Should Agents Read Demonstrations? Hierarchical Structure Beats Flat Action Logs*. Honjar Xing, Jefferson Lin, Henry Lieberman (MIT CSAIL). arXiv:2606.20978, DL4C Workshop @ ICML 2026. — [arXiv HTML](https://arxiv.org/html/2606.20978v1)
  - Setup: 85 Google Slides tasks, Claude Sonnet 4.
  - Results on vague NL tasks: zero-shot 76.7%; flat demo log 86.0% (p=0.289, n.s.); hierarchical "Agentic Goal Stack" (subgoals tagged process/input/output) 90.7% (McNemar p=0.031).
  - Hierarchy alone gives the full gain. Pre/postconditions and parameter annotations "add no measurable benefit". On precise structured tasks, all conditions score 85.7–88.1% with no difference. — [arXiv HTML](https://arxiv.org/html/2606.20978v1)

### Inferences
- The "demo → SOP / docs / workflow" literature consistently finds that *one* demonstration converted to text guidance gives large gains for both frontier models and 7B GUI models (LearnAct UI-TARS-7B: +14.7 pts). So the project's base VLM can probably *use* in-context demo knowledge. A demo-in-context baseline will not be trivially weak.
- The AppAgent ladder (exploration 73.3 < demo-docs 84.4 < manual docs 95.6) is a useful framing. Demo-derived text knowledge closes most of the gap to hand-written docs.
- Lieberman-group results suggest a cheap trick for the baseline: format the demo-derived skill hierarchically (subgoals with grouped actions) rather than as a flat log.
- WorkflowLLM is not a competitor at 3–5 demos. It shows parametric workflow learning needs about 10^5 samples, which contrasts with the project's few-shot LoRA claim.

### Gaps
- I did not obtain ECLAIR's specific numbers for "demonstration → SOP" generation quality, or its WONDERBREAD benchmark numbers.
- I found no source that ran LearnAct's knowledge extraction against a LoRA fine-tuned on the same demos.

---

## Q4. Classic PBD for GUIs (SUGILITE, Rousillon, Kite/PUMICE, CoScripter, Ringer, macro recorders): generalization from one demo, robustness to UI change, HCI lessons

### Takeaway
Classic PBD generalized from a *single* demo through explicit, inspectable structure: UI-hierarchy parameterization (SUGILITE), relation/list detection (Rousillon), personal-database variables (CoScripter). Users could edit the result. Robustness to UI change came from multi-feature element matching, error handling, and context checks. Reliability was nonetheless the top complaint in deployment (CoScripter). HCI's main lessons are that the learned program must be transparent and editable, and that the system must fail gracefully. A LoRA adapter is neither.

### Cited Findings
- **SUGILITE.** *SUGILITE: Creating Multimodal Smartphone Automation by Demonstration*. Toby Jia-Jun Li, Amos Azaria, Brad A. Myers. CHI 2017, DOI 10.1145/3025453.3025483. — [CMU project page](https://www.cs.cmu.edu/~NatProg/sugilite.html)
  - Uses the Android accessibility API, so it works in any app or across apps. It combines the spoken command, the recorded steps, and the UI hierarchy to learn "tasks with different variations and parameters from a single demonstration".
  - Error handling and context checking let it fork scripts in new situations and stay robust "if the apps change their user interface".
  - A lab study suggests users with little or no programming knowledge can automate tasks. No numbers on the page. — [CMU project page](https://www.cs.cmu.edu/~NatProg/sugilite.html)
  - Follow-ons: APPINITE (verbal data descriptions); PUMICE (NL plus demonstration to teach concepts and conditionals, e.g., "hot"); EPIDOSITE (IoT). — [CMU project page](https://www.cs.cmu.edu/~NatProg/sugilite.html)
- **Rousillon.** *Rousillon: Scraping Distributed Hierarchical Web Data*. Sarah Chasins, Maria Mueller, Rastislav Bodik. UIST 2018, pp. 963–975, DOI 10.1145/3242587.3242661. — [ACM DL](https://dl.acm.org/doi/10.1145/3242587.3242661)
  - The user demonstrates collecting the *first row* of a "universal table" view. New relation-selection and generalization algorithms then extend that to all rows.
  - In a within-subjects study with 15 computer scientists, users wrote hierarchical scrapers 8× faster than with traditional programming. — [ACM DL](https://dl.acm.org/doi/10.1145/3242587.3242661) (via search summary)
  - Output is editable only through the Helena block language. — [Katongo et al., PX/21](https://kapaya.github.io/px21/)
- **CoScripter.** *CoScripter: Automating & Sharing How-To Knowledge in the Enterprise*. Gilly Leshed, Eben M. Haber, Tara Matthews, Tessa Lau (IBM). CHI 2008, pp. 1719–1728. — [PDF](https://plunk.org/eben/PublishedPapers/chi-leshed.pdf)
  - Scripts are human-readable "sloppy" NL steps. Generalization uses *personal database* variables.
  - Usage logs: 601 users. 54 (9%) were active: on average they created 2.1 scripts, ran 5.4 distinct scripts and 28.6 runs total, about one run every 4.5 days. 43 (7%) were occasional; 504 (84%) only experimented. 23 scripts were run 10+ times by single users. 24% of 307 user-created scripts were run by 2+ users. 465 users (78%) ran scripts they had not created. — [PDF](https://plunk.org/eben/PublishedPapers/chi-leshed.pdf)
  - Robustness: 4 of 8 surveyed lapsed users reported reliability and robustness problems, all from CoScripter *misinterpreting* instructions. Example: a "click the Health link" step clicked the wrong link on a page that had no such link. The authors call reliability "critical", and harder to keep while preserving the NL representation "in the ever-changing Web".
  - Generalization was hard for end users: one user spent 2–3 hours generalizing a script for 7,000 employees. "Personal database variables were a good start." — [PDF](https://plunk.org/eben/PublishedPapers/chi-leshed.pdf)
- **Lau, "Why PBD Systems Fail".** *Why Programming-By-Demonstration Systems Fail: Lessons Learned for Usable AI*. Tessa Lau. AI Magazine 30(4), 2009, DOI 10.1609/aimag.v30i4.2262; also IBM RJ10426. — [AAAI OJS](https://ojs.aaai.org//index.php/aimagazine/article/view/2262)
  - Usability, not learning power, kept PBD from widespread adoption.
  - Guidelines include detecting failure and failing gracefully. SMARTedit, for example, had no graceful way to handle noise such as a user's demonstration mistake. Systems should also cope with interruptions and let users modify learned behavior. — [AAAI OJS](https://ojs.aaai.org//index.php/aimagazine/article/view/2262) (via search summary; the full guideline list was not retrieved)
- **ALLOY (PBD reborn for LLM agents).** *ALLOY: Generating Reusable Agent Workflows from User Demonstration*. Jiawen Li, Zheng Ning, Yuan Tian, Toby Jia-Jun Li. arXiv:2510.10049, Oct 2025. — [arXiv](https://arxiv.org/abs/2510.10049)
  - Turns a demo into a visualized workflow that is transparent, editable, and generalizable across task variations.
  - In a 12-participant study, it outperformed prompt-based agents and manual workflows "in capturing user intent and procedural preferences". No numbers on the abstract page.
- **RPA / macro recorders.** ECLAIR's RPA case studies report 12–18 months set-up, 60% initial accuracy, and multi-FTE maintenance. — [arXiv](https://arxiv.org/abs/2405.03710)
- **Ringer.** Ringer (Barman, Chasins, Bodik, Gulwani, OOPSLA 2016, DOI 10.1145/2983990.2984020) is the robust web record-and-replay layer under Rousillon. I could not retrieve its replay-success or robustness-over-time numbers. — [DOI](https://www.doi.org/10.1145/2983990.2984020)

### Inferences
- **How classic PBD generalized from one demo:** (1) explicit parameter or variable slots inferred from UI structure (SUGILITE, CoScripter); (2) structural generalization over lists and relations (Rousillon); (3) asking the user in NL or dialog to resolve ambiguity (PUMICE, APPINITE). AWM's `{product-name}` slots and Work2Skill's "workflow rules and completion checks" are LLM-era versions of (1). A LoRA does none of these explicitly.
- **Robustness to UI change:** classic systems relied on multi-feature element identification plus context checks. CoScripter's sloppy NL steps traded precision for robustness and failed through misinterpretation. A VLM agent conditioned on a skill file inherits CoScripter's misinterpretation failure mode. A LoRA overfit to 3–5 demos risks the opposite: brittle pixel and position memorization. Neither has been measured under UI drift. A small UI-perturbation test (theme or layout change) would be a novel, cheap experiment for the project.
- **HCI implications for reviewers:** the PBD literature (Lau; CoScripter; ALLOY) values transparency, editability, and shareability. An opaque per-user LoRA is weak on all three, and a skill file is strong. The project should state this limitation openly, or argue for a hybrid: show users the induced skill file while the LoRA does the low-level grounding.

### Gaps
- Kite (Li & Riva, MobiSys 2018, "Kite: Building Conversational Bots from Mobile Apps") and PUMICE's quantitative results were not retrieved (UNVERIFIED).
- I found no primary source with numbers on Apple Automator or UiPath selector brittleness; only ECLAIR's RPA case-study figures.
- I did not obtain SUGILITE's study sample size or task-completion numbers, or Ringer's replay numbers.

---

## Q5. Has any paper directly compared (a) fine-tuning on demos, (b) in-context demos, and (c) induced workflows/skills from the same demos for GUI agents?

### Takeaway
**No.** I found no paper that runs all three arms on the *same* demos, with the *same* base model, for GUI agents. That three-arm comparison is a real gap and a defensible contribution. The closest evidence is partial:
- AdaptAgent compares test-time fine-tuning on 2 demos with 1-demo ICL, but on different base models.
- ICAL and Learn-by-interact compare fine-tuning with retrieval/ICL on the same synthesized data, also on different models.
- AWM compares induced workflows with raw-trajectory ICL (Synapse).
- LatentSkill (non-GUI) compares in-context skill text, per-skill LoRA, and SFT on the same trajectories. In-context text beat naive per-skill LoRA.
- CoAdapt-GUI shows workflow context plus LoRA policy adaptation together beat LoRA alone on mobile GUIs.

### Cited Findings
- **AdaptAgent (closest GUI-agent FT vs ICL with ≤2 demos).** *AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations*. Gaurav Verma, Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Tucker Balch, Manuela Veloso (J.P. Morgan AI Research). arXiv:2411.13451, NeurIPS 2024 AFM Workshop. — [arXiv](https://arxiv.org/abs/2411.13451)
  - Mind2Web overall SR (cross-task / cross-website / cross-domain):
    - SeeAct (GPT-4o), 0 demos: 14.37 / 15.83 / 19.49.
    - SeeAct + 1 in-context multimodal demo: 19.69 / 22.46 / 23.97.
    - CogAgent-FOMAML, meta-trained on 292 tasks and **fine-tuned at test time on 2 demos per new website/domain**: 11.89 / 16.96 / 19.66.
    - CogAgent-FT on the full Mind2Web train set: 13.36 / 12.42 / 15.20.
    - CogAgent-FT on the same 292 examples: 10.43 / 10.10 / 13.28.
    - Pretrained CogAgent zero-shot: 2.80 / 2.11 / 2.59.
  - VisualWebArena (human traj / live): GPT-4o + 1 demo 22.42 / 21.36; CogAgent-FOMAML 11.01 / 8.47; CogAgent-FT 8.78 / 6.43.
  - The paper does *not* compare ICL and FT on the same model. Its explicit claim is meta-learning > regular FT at equal data. — [arXiv HTML](https://arxiv.org/html/2411.13451)
- **ICAL.** *VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs of Thought*. Gabriel Sarch, Lawrence Jang, Michael J. Tarr, William W. Cohen, Kenneth Marino, Katerina Fragkiadaki. arXiv:2406.14596 (v6, Sep 2025). NeurIPS 2024 is UNVERIFIED. — [arXiv](https://arxiv.org/abs/2406.14596)
  - On VisualWebArena, retrieval-augmented GPT-4V with ICAL abstractions gives 1.6× task success. *Fine-tuned* Qwen2-VL on the same ICAL examples gives 2.8× over its base.
  - On TEACh, fine-tuning plus retrieval on ICAL examples beats raw human demos by 17.5% goal-condition success. The approach "scales 2x better than raw demonstrations". — [arXiv](https://arxiv.org/abs/2406.14596)
- **Learn-by-interact.** Hongjin Su, Ruoxi Sun, Jinsung Yoon, Pengcheng Yin, Tao Yu, Sercan Ö. Arık. arXiv:2501.10893, Jan 2025. ICLR 2025 is UNVERIFIED. — [arXiv](https://arxiv.org/abs/2501.10893)
  - Same synthesized trajectory data used two ways: training-free ICL improves by up to 12.2% (Claude-3.5); training improves by 19.5% (Codestral-22B). Different models, "up to" numbers. — [arXiv](https://arxiv.org/abs/2501.10893)
- **AWM vs raw exemplars.** Induced workflows beat Synapse's retrieved full trajectories by +5.0 EA and +4.0 step SR (GPT-3.5, Mind2Web cross-task). — [arXiv HTML](https://arxiv.org/html/2409.07429)
- **LatentSkill (non-GUI, same trajectories).**
  - ALFWorld seen/unseen: in-context skill text 52.9/56.0 vs Per-skill LoRA 45.7/50.0 vs Full SFT 53.6/51.5 vs hypernetwork LoRA 74.3/69.4.
  - Per-skill LoRA halves prefill tokens: 0.60k vs 1.21k per step. — [arXiv HTML](https://arxiv.org/html/2606.06087)
- **CoAdapt-GUI (context + LoRA).** *CoAdapt-GUI: Joint Workflow Context and Policy Adaptation for Unseen GUI Applications*. Linqiang Guo, …, Tse-Hsun (Peter) Chen. arXiv:2608.11588, Aug 2026. — [arXiv](https://arxiv.org/abs/2608.11588)
  - Test-time adaptation on the target app from the agent's *own rollouts*, with no target demos. It jointly updates structured workflow notes and a LoRA on a frozen VLM (group-relative optimization).
  - AndroidWorld-Generalization: joint 45.0% vs Policy-Only (LoRA) TTA 37.5%. AndroidWorld Plus: 38.6% → 52.9%.
  - No separate context-only number in the abstract. — [arXiv](https://arxiv.org/abs/2608.11588)
- **Non-GUI hint (search snippet only, not opened, UNVERIFIED).** *In-Context Prompting Obsoletes Agent Orchestration for Procedural Tasks* (arXiv:2604.27891) reportedly finds in-context prompting beats orchestration in most comparisons. It suggests fine-tuning the procedure into a smaller model mainly for high-volume cost reasons. — [arXiv HTML](https://arxiv.org/html/2604.27891v1)

### Inferences
- **The evidence tilts as follows.**
  1. For frontier API models, in-context demos or induced workflows are cheap and effective: +5 to +7 SR from 1 demo in AdaptAgent, +12 pts in AWM.
  2. For small open GUI models, fine-tuning gives larger *relative* gains from a low base (ICAL 2.8× vs 1.6×; CogAgent 2.1 → 17.0 cross-website with 2-demo test-time FT). Even so, the absolute numbers stayed *below* GPT-4o + 1 in-context demo.
  3. Naive per-skill LoRA on trajectories can lose to the same knowledge as text in context (LatentSkill).
  4. Context and weights look complementary (CoAdapt-GUI joint > policy-only; ICAL FT + retrieval best).
- **Recommended design for the project's main table.** Same base GUI VLM, same 3–5 demos, five arms:
  1. zero-shot;
  2. raw demos in context (LearnAct / UI-Mate style, 1–2 demos);
  3. demo-induced skill file (AWM-offline / Work2Skill style, hierarchical);
  4. per-skill LoRA (the project);
  5. LoRA + skill file.
  
  This three-way-plus comparison does not appear to exist for GUI agents, and it directly answers the reviewer question.
- **Arguments for the LoRA under a matched comparison:**
  - per-step token and latency savings (no long multimodal demo context; LatentSkill-type prefill reductions);
  - small VLMs' weak use of long textual guidance (LatentSkill WebShop drop; Work2Skill baselines that sometimes hurt);
  - privacy (no plaintext skill);
  - the possibility of learning low-level grounding (exact click targets) that text cannot convey.

### Gaps
- No same-model, same-demo (a)/(b)/(c) comparison for GUI agents found as of Nov 2026. This should be confirmed by whoever covers the parametric/LoRA literature.
- AdaptAgent and ICAL comparisons are confounded by base model (GPT-4o/GPT-4V vs CogAgent/Qwen2-VL).
- CoAdapt-GUI's context-only ablation and interaction budget were not available from the abstract.

---

## Q6. "Agent Skills" (Anthropic SKILL.md; OpenAI equivalents): documentation, and could a "skill file generated from demos" be a strong baseline?

### Takeaway
Yes, the format is well documented and now cross-vendor. Anthropic launched Agent Skills on 16 Oct 2025 and published them as an open standard on 18 Dec 2025 (agentskills.io); OpenAI Codex documents the same SKILL.md format. Evidence on whether skills help is mixed in a way that matters here:
- Curated skills help: SkillsBench +16.2 pp (v1) / +16.6 pp (v4).
- Ungrounded *self-generated* skills gave no average benefit (SkillsBench v1).
- Skills grounded in *human GUI demos* do help: Work2Skill +3 to +10.5 score points, including on 9B models.

A demo-grounded SKILL.md is therefore the most credible, cheapest non-parametric baseline.

### Cited Findings
- **Anthropic Agent Skills.**
  - Published 16 Oct 2025. A skill is a folder whose `SKILL.md` starts with YAML frontmatter; `name` and `description` are required, followed by instruction body text.
  - Progressive disclosure in three levels: (1) all skills' name and description loaded at startup; (2) the full SKILL.md read when relevant; (3) linked files read only as needed.
  - Skills can bundle scripts that run deterministically without being loaded into context.
  - A 18 Dec 2025 update says Anthropic "published Agent Skills as an open standard for cross-platform portability".
  - Supported in Claude.ai, Claude Code, Agent SDK, and the Developer Platform. — [Anthropic Engineering](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- **OpenAI Codex skills.** Codex docs say skills build on the open Agent Skills standard. A skill is a folder with SKILL.md, required name and description, and optional scripts and references. Codex loads only name, description, and path at startup and reads the full file when it selects the skill. Skills can be disabled via `[[skills.config]]` in `~/.codex/config.toml`. — [OpenAI Codex docs](https://developers.openai.com/codex/skills)
- **OpenAI adoption timeline.** Codex CLI gained experimental skills support in Dec 2025, followed by docs. Skills are also usable via the OpenAI API with the shell tool. — [Simon Willison, 12 Dec 2025](https://simonwillison.net/2025/Dec/12/openai-skills/); [Simon Willison, 19 Dec 2025](https://simonwillison.net/2025/dec/19/agent-skills)
- **Wider adoption claims (secondary, UNVERIFIED).** Third-party blogs say SKILL.md is supported in GitHub Copilot, Cursor, Vercel, and "30+ tools". — [DEV Community](https://dev.to/nathanielc85523/skillmd-goes-multi-ecosystem-how-the-agent-skills-standard-jumped-from-anthropic-to-openai-and-3oeg)
- **SkillsBench.** *SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks*. Xiangyi Li, Yimin Liu, Wenbo Chen, and 76 others. arXiv:2602.12670; listed as a NeurIPS 2026 poster. — [arXiv](https://arxiv.org/abs/2602.12670), [NeurIPS 2026 virtual page](https://neurips.cc/virtual/2026/poster/139353)
  - v1 (Feb 2026): 86 tasks across 11 domains, 7 agent-model configs, 7,308 trajectories. Three conditions: no skills, curated skills, self-generated skills.
  - v1 results: curated skills +16.2 pp average (+4.5 pp in software engineering up to +51.9 pp in healthcare); 16 of 84 tasks show negative deltas. **Self-generated skills "provide no benefit on average"**, i.e., models cannot reliably author the procedural knowledge they benefit from consuming. — [arXiv v1](https://arxiv.org/abs/2602.12670v1)
  - v4 (Jun 2026): 87 tasks, 8 domains, 18 configs. Curated skills raise pass rate from 33.9% to 50.5% (+16.6 pp). Configuration gains range +4.1 to +25.7 pp. "Focused Skills with at most three modules outperform larger or exhaustive bundles." — [arXiv v4](https://arxiv.org/abs/2602.12670)
  - A follow-up (SkillGenBench) notes that SkillsBench's self-generated condition is *task-local*: the skill is written after the task is revealed, without external grounding. — [SkillGenBench, arXiv:2605.18693](https://arxiv.org/abs/2605.18693) (via search summary)
  - A search summary gave a "−1.3 pp" average for self-generated skills; this is UNVERIFIED.
- **SkillGenBench.** Yifan Zhou et al. arXiv:2605.18693, May 2026. Benchmarks skill *generation* pipelines in task-conditioned vs task-agnostic regimes, from repositories or long documents. It reports "substantial performance variation" and difficulty distilling reusable skills. It does not cover demonstrations. — [arXiv](https://arxiv.org/abs/2605.18693)
- **Work2Skill, the demo-grounded counter-evidence.**
  - Demo-grounded SKILL.md files (authored by an agent following Anthropic's skill-creator) improve every executor tested. Qwen3.5-9B: MyPCBench score 24.27 → 34.11 and SR 6.25 → 12.50. GPT-5.5: MyPCBench SR 23.44 → 48.44. — [arXiv HTML](https://arxiv.org/html/2610.12984)

### Inferences
- **Strongest and cheapest non-parametric baseline:** a "**Demo → SKILL.md**" induction step that combines AWM-offline and Work2Skill.
  1. Serialize each of the 3–5 demos as a step list: action, target element description (a11y label or OCR), screenshot caption.
  2. Make one call to a strong LLM, instructed to write one SKILL.md per skill containing:
     - frontmatter `name` and `description` (when to use it);
     - hierarchical subgoals (per Xing/Lin/Lieberman), each with grouped steps;
     - parameter slots for values that varied across demos, AWM-style `{slot}`;
     - pitfalls and completion checks (Work2Skill);
     - at most ~3 modules, short (SkillsBench).
  3. Prepend the skill to the *same base GUI VLM* used for the LoRA at inference.
  - Estimated effort: about a day of engineering, no GPUs, about cents of API cost per skill (my estimate, not sourced).
  - It is grounded in user demos, so it avoids SkillsBench's "self-generated skills don't help" failure, and it matches the format reviewers will recognize.
- **Second cheap baseline:** raw demos in context, i.e., 1–2 demos as interleaved screenshots and actions (LearnAct / UI-Mate style). It is needed to show the comparison is not just "abstraction vs nothing". It may be context-limited for a 7B VLM with 5 multimodal demos.
- **Heavier, optional:** ASI-style programmatic skills (verified Python/Playwright functions). They are strongest on web but need API- or DOM-level actions and verification, which a pixel-level GUI VLM project probably lacks.
- **Positioning for reviewers:**
  - If LoRA ≈ skill file in accuracy, argue for LoRA's lower per-step prompt cost and better low-level grounding, and show the LoRA + skill-file combination is best (as CoAdapt-GUI and ICAL found for context + weights).
  - If LoRA < skill file, the paper becomes an honest negative/analysis result. That is still workshop-worthy given that no matched three-arm GUI study exists.
  - Main novelty threats to cite: Work2Skill (demos → skills for desktop GUI agents), CoAdapt-GUI (workflow context + LoRA for GUI apps, though self-rollout RL rather than user demos), LatentSkill (per-skill LoRA vs in-context skills, non-GUI), and AdaptAgent (few-demo fine-tuning of web agents).

### Gaps
- I found no official Anthropic or OpenAI guidance or evaluation on generating skills *from GUI demonstrations*. Work2Skill's use of skill-creator is the only instance found.
- I did not verify whether SkillsBench v4 still reports the self-generated-skills result; its abstract no longer mentions it.
- No published cost or latency comparison of SKILL.md prompting vs LoRA swapping for GUI VLMs was found.
