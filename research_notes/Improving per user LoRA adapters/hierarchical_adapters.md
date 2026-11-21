# Hierarchical / two-stage adapters for personalization: a shared skill adapter plus a small per-user component

Scope: schemes that learn "how to operate the app" from all users' demos and "this user's choices" from the user's own few demos. The target setup is UI-TARS-1.5-7B in mlx-vlm, with 3 users × 4 skills × 5 train demos and conflicting preferences.

Conventions:
- **UNVERIFIED** means not confirmed by a page fetched or searched in this session. These items come from prior knowledge, or from a search snippet with no primary text, and need a check before you cite them.
- Facts about the project come from local files, cited by path.

Project facts the recipe depends on (all checked locally this session):
- **Model shape.** UI-TARS-1.5-7B has a 28-layer LLM (hidden 3584, MLP 18944, 28 heads, 4 KV heads). The vision tower has 32 layers, with full attention at blocks 7/15/23/31. Source: [models/UI-TARS-1.5-7B-mlx-bf16/config.json](/Users/hanmozhang/schoolwork/cmu-agents-project/models/UI-TARS-1.5-7B-mlx-bf16/config.json).
- **Current recipe.** lr 5e-5, rank 16, α 16, warmup 20, 4 epochs, batch 1, dropout 0.05, no grad-clip. Divergence history: lr 2e-4 → NaN; lr 1e-4 with grad-clip → NaN by iter 25; lr 1e-4 with α=32 diverged after about 40 iters. An adapter takes about 25 min (70 samples × 4 epochs ≈ 280 iters, so about 5.4 s/iter). Sources: [STATUS.md](/Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md), [train/train_adapters.py](/Users/hanmozhang/schoolwork/cmu-agents-project/train/train_adapters.py).
- **LoRA scale in mlx-vlm 0.8.0.** The scale is `alpha / rank`, so α16/r16 gives scale 1.0 and there is no hidden ×20 multiplier. With no `--target-modules`, LoRA goes on all Linear layers of `language_model`, excluding the LM head. The vision tower stays frozen unless `--train-vision` is set. `--target-modules` accepts suffixes or fully qualified names, so per-layer targeting is possible. `--resume-adapter-file` exists, which enables continued fine-tuning from an existing adapter. `LoRALinear.fuse()` computes `weight + delta` in the weight's dtype (bf16 here). Sources: [mlx_vlm/trainer/peft/utils.py](/Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/peft/utils.py), [lora_layers.py](/Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/trainer/peft/lora_layers.py), [mlx_vlm/train.py](/Users/hanmozhang/schoolwork/cmu-agents-project/.venv/lib/python3.12/site-packages/mlx_vlm/train.py).
- **Adapter size.** Computed from the config: rank-16 all-linear LoRA across 28 layers is 90,112·r params per layer, or **40.4M params** per per-user adapter (MLP 67,584·r per layer, attention 22,528·r). That is about 10× OPPU's per-user rank-8 q/v LoRA (4.19M on a 7B Llama). See the OPPU and LINEUP rows below.
- **Dry-run results** (oracle demos, Alice × calendar, 42 test steps, 9 of them preference-critical):
  - C3 (Alice's own LoRA): 6/9 preference-critical steps, 61% on other steps.
  - C5 (Bob and Carol's demos, same skill): 0/9 preference-critical, 58% on other steps. It applies Bob's habits.
  - C3 skips Alice's "30 min" click.
  - Preference-critical steps are defined as steps where Alice's correct action differs from every other persona's **on the same task instance**. Pooled data therefore contains identical inputs with conflicting labels.
  - Planned conditions already include C5 (other users, same skill) and C6 (pooled, all users, all skills, no user signal).
  - Source: [STATUS.md](/Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md).

---

## Q1. OPPU and related shared-plus-per-user schemes (OPPU, Per-Pcs, HYDRA, PLoRA, Profile-to-PEFT, LINEUP, federated dual adapters)

### Takeaway
The canonical recipe is OPPU: task-adapt the base on **other** users' data, merge it in, then train a small per-user LoRA on top at a **low learning rate** (rank 8, q/v only, lr 1e-5, 2–3 epochs). It beats retrieval/profile-only personalization on all 7 LaMP tasks.

Newer work keeps the "shared component + tiny per-user component" structure and shrinks the per-user part: gated pieces (Per-Pcs), per-user heads (HYDRA), user embeddings (PLoRA), hypernetwork-generated LoRA (P2P), and 8 scalars over a shared factor bank (LINEUP). All of the shrinking approaches need hundreds to thousands of training users. With 3 users, only the OPPU-style two-stage recipe and its federated cousins (global + local adapter) transfer directly.

### Cited Findings

**OPPU** — Tan, Zeng, Tian, Liu, Yin, Jiang, "Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning", arXiv 2402.04401 (v3 Feb 2025; EMNLP 2024 venue **UNVERIFIED**, since the arXiv page lists none). Source: [arXiv HTML](https://arxiv.org/html/2402.04401).
- **Stage 1.** Llama-2-7B is fine-tuned on task data from users outside the test set ("all other users"), giving base models for plain, retrieval-augmented, and profile-augmented inputs. The stage-1 LoRA is then **merged into the base**.
- **Stage 2.** For each test user, a fresh personal PEFT is trained on that user's history with the base frozen. This is "stack a new LoRA on a merged task LoRA", not continued training of the task LoRA.
- **Hyperparameters.** The PEFT is LoRA on W_q and W_v, rank 8, lr 1e-5, 2–3 epochs, batch 3–16.
- **Test set.** The 100 users with the longest histories in time-based LaMP.
- **Gains over the best baselines.** Product rating: −17.38% MAE and −8.89% RMSE (relative). Movie tagging: +11.87% accuracy and +7.56% F1. Scholarly title generation: +3.42% ROUGE-1 and +3.87% ROUGE-L.
- **Averages across tasks.** Retrieval+OPPU is +1.93% over non-retrieval and +2.48% over retrieval without OPPU. Profile+OPPU is +4.56% over non-profile and +7.18% over profile without OPPU.
- **Non-parametric inputs combine with the adapter.** The personal PEFT is trained on prompts that already contain retrieved items or a profile, and the two are complementary.
- **Behavior shift.** When history similar to the test query is removed, retrieval-only methods collapse toward the non-personalized baseline while OPPU stays robust.
- **Activity level.** OPPU beats baselines at all levels, and the gain grows with history length.
- **PEFT type.** LoRA beats (IA)³, which beats prompt tuning.
- **Missing ablation.** I found no explicit test of a task-adapted base vs a raw base.
- **Inconsistency.** The paper reports LoRA's share of parameters as both "about 0.5%" and "0.01%".

**Per-Pcs** — Tan, Liu, Jiang, "Personalized Pieces: Efficient Personalized Large Language Models through Collaborative Efforts", arXiv 2406.10471, EMNLP 2024 Main (per the arXiv comments). Source: [arXiv](https://arxiv.org/abs/2406.10471).
- "Sharer" users split their PEFTs into pieces and train a gate per piece.
- Target users assemble a personal PEFT from the shared pool using their own history.
- Performance is "comparable to OPPU with significantly lower resource use across six tasks". I found no numbers on the abstract page.

**HYDRA** — Zhuang, Sun, Yu, Qiang, Wang, Zhang, Dai, "HYDRA: Model Factorization Framework for Black-Box LLM Personalization", arXiv 2406.02888, NeurIPS 2024. Source: [arXiv](https://arxiv.org/abs/2406.02888).
- The reranker and the adapter are each factorized into a **shared base** ("knowledge shared across users") and **multiple user-specific heads** ("each user's preferences").
- It reports +9.01% average relative improvement over prompt-based methods on 5 LaMP tasks.
- It is black-box, so it adapts outputs rather than the LLM's weights.

**PLoRA** — Zhang et al., "Personalized LoRA for Human-Centered Text Understanding", AAAI 2024, arXiv 2403.06208. Sources: [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29931), [arXiv](https://arxiv.org/abs/2403.06208).
- Personalized knowledge injection (PKI) puts user attributes and preferences into a unified embedding, combined with one shared LoRA, plug-and-play.
- **Personalized dropout** and a **mutual-information-maximizing** objective handle few-shot and zero-shot cold start.
- It reports gains in full, few-shot, and zero-shot settings on 4 datasets, with fewer trainable parameters.

**Profile-to-PEFT (P2P)** — Tan et al. (11 authors), "Instant Personalized Large Language Model Adaptation via Hypernetwork", arXiv 2510.16282, ACL 2026 long, pp. 23557–23580. Sources: [arXiv HTML](https://arxiv.org/html/2510.16282), [ACL Anthology](https://aclanthology.org/2026.acl-long.1081/).
- **Architecture.** An MLP hypernetwork maps a profile embedding (frozen Qwen3-Emb-4B over a global preference summary plus top-k retrieved history) to rank-8 q/v LoRA weights for every layer, using learned module and depth embeddings.
- **Training.** Base is Qwen2.5-7B-Instruct. Training users come from LaMP, LongLaMP, Personal Reddit, and Empathetic Conversations, with thousands of users per LaMP subtask (for example 19,244 for product rating), but only 409 (PR) and 53 (EC). Settings: lr 2e-5, 20k steps, batch 32.
- **Random-split results.** Classification accuracy: P2P 0.580 vs OPPU 0.568 vs RAG 0.507. Generation ROUGE-L: P2P 0.244 vs OPPU 0.221.
- **OOD-user split.** Classification: P2P 0.581 vs OPPU 0.528.
- **Cost.** P2P takes about 0.57 s per user vs about 20 s per user for OPPU. Hypernetwork training takes 27,167 s once and is amortized after about 1,450 users.
- **No stage-1 base.** P2P is trained from scratch, with no task-adapted base LoRA stage.

**LINEUP** — Sui, Malla, Choi, Choi (Samsung Semiconductor; Rice), "Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation", arXiv 2610.02353, submitted 1 Oct 2026, no venue listed. Sources: [arXiv](https://arxiv.org/abs/2610.02353), [chatpaper listing](https://chatpaper.com/chatpaper/paper/355478).
- **Analyses.** "Independent user adapters share substantial reusable structure". The value of shared directions depends on user relevance and on variation across queries.
- **Method.** A bank of reusable low-rank personalization factors, composed through user-conditioned recall and query-dependent calibration. **Each target user optimizes only eight scalars** while all shared components stay fixed. The private-LoRA baseline uses 4.19M per-user params.
- **Theory.** A finite-step, finite-history risk bound.
- **Results.** It leads on 12/12 metrics across 6 tasks (3 seeds), with −11.4% LaMP-3 RMSE vs the strongest baseline, and keeps its advantage with limited history.
- **Unknowns.** The base model and the training-user counts are not in the abstract.

**PRISP** — Park, Kim, Moon, "PRISP: Privacy-Safe Few-Shot Personalization via Lightweight Adaptation", arXiv 2601.06471 (v2 Jul 2026), preprint. Source: [arXiv](https://arxiv.org/abs/2601.06471).
- A Text-to-LoRA hypernetwork generates a **task-aware** LoRA from the task description.
- Per-user personalization then tunes only "a small subset" of those LoRA params plus minimal modules, using few-shot user data on a few-shot LaMP variant.
- The abstract gives no numbers.

**Federated analogues** (a client corresponds to a user):
- **FedDPA** — Yang, Long, Shen, Jiang, Blumenstein, "Dual-Personalizing Adapter for Federated Foundation Models", arXiv 2403.19211. A global adapter and a local adapter are combined by **instance-wise dynamic weighting** at inference. Source: [arXiv](https://arxiv.org/abs/2403.19211).
- **FedALT** — Bian, Wang, Zhang, Xu, AAAI 2026, arXiv 2503.11880. Each client keeps its **own individual LoRA** plus a frozen **Rest-of-World LoRA** aggregated from other clients, mixed by an input-adaptive MoE-style mixer. It argues that FedAvg-style "start local training from the aggregate" causes **cross-client interference**. No numbers on the abstract page. Source: [arXiv](https://arxiv.org/abs/2503.11880).

**Hypernetwork init + short fine-tune, image domain** — HyperDreamBooth (Ruiz et al., CVPR 2024, arXiv 2307.06949). A hypernetwork predicts an initial LoRA from one face image, then a short "rank-relaxed" fine-tune refines it. Personalization takes about 20 s, 25× faster than DreamBooth, and the model is 10,000× smaller. Source: [arXiv](https://arxiv.org/html/2307.06949v2).

**GUI-agent personalization work found** (none uses per-user weights):
- IFRAgent (arXiv 2508.08645) builds a shared procedure library plus a per-user habit store via retrieval/memory.
- PersonalAlign / HIM-Agent (arXiv 2601.09636) uses AndroidIntent: 775 preferences and 215 routines from 20k records, handled with memory.
- PSPA-Bench (arXiv 2603.29318).
- Sources: [IFRAgent](https://arxiv.org/html/2508.08645v2), [PersonalAlign](https://arxiv.org/html/2601.09636v1), [PSPA-Bench](https://arxiv.org/pdf/2603.29318).

### Inferences
- **Our C5 adapter is OPPU's stage 1** for the target user: same skill, other users only, as in OPPU's "users outside the test set". C6 is a closed-world variant, since it includes the user's own train demos. A faithful OPPU arm is therefore "C5 (or C6) → merge → small per-user LoRA, lr ~1e-5 to 5e-5". It reuses adapters already scheduled in `scripts/overnight.sh`.
- **Why stage 2 should target the missed preference steps.** On top of a base that already grounds and finishes the task, loss on routine steps is already low. Gradient therefore concentrates on high-loss tokens, which are the preference-divergent decision tokens. STATUS §3 notes that "the action is decided by 1–2 tokens". This implicit reweighting is the mechanism by which stage 2 should fix "misses some preference steps", but it is an inference, not a measured result.
- **Our per-user adapter is oversized.** At 40.4M params trained on about 70 samples, it is about 10× OPPU's per-user LoRA. The LINEUP and P2P results suggest the per-user component can be much smaller once a shared component exists.
- **Arithmetic check on LINEUP's baseline (inference).** 4.19M matches exactly a rank-8 q/v LoRA on a 32-layer, 4096-hidden model: 32 × 2 × 8 × 8192 = 4,194,304. So LINEUP's private-LoRA baseline is probably OPPU-style on a Llama-7B/8B-class model.
- **Infeasible approaches for us.** Hypernetworks (P2P, LINEUP's bank, PRISP's Text-to-LoRA) need hundreds to thousands of training users. P2P amortizes only after about 1,450 users. With 3 users, cite them as related work or future work.
- **The closest feasible analogue to HYDRA and PLoRA** is a shared adapter conditioned on a user ID or embedding token (see Q5).

### Gaps
- OPPU's per-user history sizes are large (100-user test set, long histories). I found no OPPU-style result at about 5 examples per user. The closest are PRISP (few-shot LaMP, no numbers seen) and P2P's activity-level analysis.
- I found no ablation of an OPPU stage-1 base vs a raw base in the OPPU text.
- Per-Pcs storage and accuracy numbers, FedDPA and FedALT numbers, and LINEUP's base model and its prior on which layers to adapt were not visible in the abstracts.
- I found no paper that trains per-user weight adapters for GUI or computer-use agents. The 2026 GUI personalization papers use memory or retrieval.

---

## Q2. Initialization: continued fine-tuning from a pooled/skill LoRA vs stacking a new LoRA on a merged one; meta-learned initializations

### Takeaway
Both warm-start styles have support:
- "Fine-tune the global model locally" (FTFA) is a tough-to-beat personalization baseline in federated learning.
- OPPU uses "merge, then stack a fresh LoRA".

The main documented risk is **overfitting or interference as local epochs grow**. Keep the per-user stage short (1–3 epochs), at low LR, selected by held-out accuracy on preference-critical steps.

Meta-learned initializations (FOMAML/Reptile) help web agents, but AdaptAgent needed 292 meta-training tasks across 73 sites. With 12 (user, skill) tasks, meta-learning is not worth its cost here.

### Cited Findings

**Continued fine-tuning (warm start):**
- **Cheng, Chadha, Duchi**, "Fine-tuning in Federated Learning: a simple but tough-to-beat baseline" (later "Federated Asymptotics"), arXiv 2108.07313. FedAvg followed by client-side fine-tuning (FTFA, i.e. a warm start from the global model) is competitive with meta-learning and proximal-regularized personalization, and cheaper. Source: [arXiv](https://arxiv.org/abs/2108.07313v2).
- **Chen, Ghoukasian, Jin**, "A Closer Look at Personalized Fine-Tuning in Heterogeneous Federated Learning", arXiv 2511.12695. Local fine-tuning from the global model **overfits to skewed client data and degrades as local epochs increase**. They propose LP-FT (linear-probe, then full fine-tune) as a stronger baseline. Source: [arXiv](https://arxiv.org/html/2511.12695v1).
- **FedALT** (arXiv 2503.11880) argues **against** initializing local training from the aggregate, because of cross-client interference. It prefers a separate individual LoRA plus a frozen shared ("rest-of-world") LoRA with a learned mixer. Source: [arXiv](https://arxiv.org/abs/2503.11880).

**Stacking on a merged base:**
- **OPPU** merges the stage-1 LoRA into the base and trains a fresh per-user rank-8 LoRA at lr 1e-5 for 2–3 epochs. Source: [arXiv HTML](https://arxiv.org/html/2402.04401).

**Hypernetwork-predicted init + short fine-tune:**
- **HyperDreamBooth** shows the "good init → few fine-tune steps" pattern works and is fast (about 20 s per subject, 25× faster than DreamBooth). Source: [arXiv](https://arxiv.org/html/2307.06949v2).

**Meta-learning for web agents — AdaptAgent** (Verma, Kaur, Srishankar, Zeng, Balch, Veloso), arXiv 2411.13451, abridged at the NeurIPS 2024 AFM workshop. Source: [arXiv HTML](https://arxiv.org/html/2411.13451).
- **Setup.** CogAgent-18B with LoRA (lr 1e-5), meta-trained with **first-order MAML** on the Mind2Web train set: 4 tasks per site × 73 sites = 292 tasks. Each inner loop uses 2 tasks; the outer set has 1 task from the same site and 1 from another site in the same domain. Test-time adaptation uses **2 demos** of the new site or domain.
- **Cross-website step success.** Pre-trained 21.29, fine-tuned (FT) 47.14, data-equivalent FT 44.27, FOMAML + 2-demo adaptation **55.38**.
- **Cross-domain.** 52.20 (data-equivalent FT) vs **57.29** (FOMAML).
- **Cross-task.** FOMAML 53.32 trails FT at 54.43.
- **Abstract.** +3.36 to +7.21 points task success, or +21.03% to +65.75% relative.
- **Confound.** Only the meta-learned row had test-time adaptation; the FT rows were not adapted.

**Meta-learning LoRA for LLMs:**
- **ABMLL** (Amortized Bayesian Meta-Learning for LoRA, arXiv 2508.14285) works at Llama3-8B / Qwen2-7B scale with a small memory overhead and reports stronger unseen-task performance than plain LoRA. It also notes Reptile's limited expressivity and the per-task parameter growth of hierarchical Bayesian LoRA. Source: [arXiv](https://arxiv.org/html/2508.14285v3).
- **MeTA-LoRA** (arXiv 2510.11598): data-efficient multi-task LoRA that starts from shared LoRA params. Source: [arXiv](https://arxiv.org/html/2510.11598v1).
- **ReptiLoRA** (Issues in Information Systems 2025): pairs Reptile with LoRA. Source: [IIS](https://iacis.org/iis/2025/3_iis_2025_275-284.pdf).
- Benchmark numbers for these were not verified this session.

**LR evidence:**
- **"LoRA Without Regret"**, Schulman / Thinking Machines, 29 Sep 2025. The optimal LoRA LR is about **10× the full-fine-tuning LR** (fitted multiplier 9.8 across 14 models), and about **15×** for short runs under ~100 steps. The 1/r scaling makes the optimal LR nearly rank-independent. LoRA tolerates large batches less well than full fine-tuning. Source: [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/).
- **OPPU's stage-2 LR** of 1e-5 and **AdaptAgent's inner-loop LR** of 1e-5 are both low. Sources: [OPPU](https://arxiv.org/html/2402.04401), [AdaptAgent](https://arxiv.org/html/2411.13451).

### Inferences

**Continued FT vs stacking for our setup:**
- **Continued FT** (`--resume-adapter-file` from C5/C6) is a **zero-code** change. It updates the same rank-16, 40M-param subspace. The only thing holding it near the pooled solution is early stopping and low LR. With 70 samples and lr ≥ 1e-4 it will likely reproduce the instability already seen.
- **Stacking** (fuse C5/C6, then train a fresh rank 4–8 LoRA) starts exactly at pooled behavior, because B=0. It gives a separable Δ_user, so you get an inference-time **strength knob** (residual scale) and a clean object for the arithmetic in Q3. It also allows a much smaller per-user component.
- **Recommendation.** Run continued FT first because it is free to implement, then stacking if time allows.

**Fusing caveat (inference from code):**
- mlx-vlm's `fuse()` adds the LoRA delta in **bf16**, the weight dtype. bf16 keeps about 8 significant bits, so small LoRA deltas can be partly rounded away when fused into bf16 weights.
- Either fuse in fp32/fp16, or keep the stage-1 adapter **unfused** in a small custom "two LoRAs on one Linear" wrapper.
- Sanity check: fused-pooled and unfused-pooled should give the same step accuracy on the dev split before training stage 2.

**Meta-learning:**
- AdaptAgent's gain came from 292 tasks across 73 sites. With 12 (user, skill) tasks and 5 s per step, FOMAML/Reptile would cost hours per meta-epoch and the "meta" signal would be weak.
- Pooled stage-1 training is already roughly the first-order, single-inner-step limit of Reptile, so skip explicit meta-learning.

**Stage-2 hyperparameters:**
- Given the 10× rule (full-FT LR for 7B is about 1e-5, so LoRA LR is about 1e-4), stage 2, which only needs a small correction, should sit at **2e-5 to 5e-5**.
- The divergence at ≥1e-4 with batch 1 is consistent with an over-parameterized 40M-param adapter on 70 samples. That is an inference, not a diagnosis.

### Gaps
- I found no paper that compares head-to-head, at about 5 examples per user, (a) continued fine-tuning of a pooled LoRA vs (b) merge plus a fresh stacked LoRA. The FL papers compare warm-start FT against more elaborate pFL methods, not against stacking.
- I found no LP-FT analogue for LoRA. A possible stand-in is "train only B first" or "train top layers first", but this is untested.
- ABMLL, MeTA-LoRA, and ReptiLoRA numbers were not checked against the primary text.

---

## Q3. Merging and arithmetic: LoRA merging, task arithmetic, subtracting a generic direction to isolate preference

### Takeaway
Weight-space arithmetic on PEFT modules works: addition composes them, and negation removes behaviors. Naive subtraction also removes shared, generic knowledge. Ethos shows that projecting the task vector to keep only the "specific" components is safer.

For us, the useful versions are:
- **Δ_pref = ΔW_C3(u) − ΔW_C5(u)**, computed on the dense products BA and not on the A/B factors, which come from different random inits.
- A **residual-scale knob** (amplify the per-user residual) to target the "misses preference steps" failure.

Both are training-free, so they are cheap analyses rather than primary methods.

### Cited Findings
- **Composing PEFT modules.** Zhang, Chen, Liu, He, "Composing Parameter-Efficient Modules with Arithmetic Operations", NeurIPS 2023, arXiv 2306.14870. Defines **addition and negation** operators for PEFT modules (including LoRA) in weight space, with no extra training. Applications: distribution generalization, multi-tasking, unlearning, domain transfer, and detoxifying Alpaca-LoRA. Numbers were not on the abstract page. Source: [arXiv](https://arxiv.org/abs/2306.14870).
- **Ethos.** Gao, Niu, Tang, Avestimehr, Annavaram, "Ethos: Rectifying Language Models in Orthogonal Parameter Space", Findings of NAACL 2024, arXiv 2403.08994.
  - Plain task-vector negation also removes **general knowledge**, because task vectors mix general and undesired knowledge.
  - Ethos takes an SVD of the pre-trained weights, projects the task vector onto those principal components, and negates only the "undesired" components.
  - It reports better removal with less damage to general performance for debiasing, detoxification, and memorization.
  - Sources: [ACL Anthology](https://aclanthology.org/2024.findings-naacl.132), [arXiv](https://arxiv.org/abs/2403.08994v1).
- **Shared A vs shared B — conflicting evidence:**
  - **FedSA-LoRA** (Guo et al., ICLR 2025, arXiv 2410.01463): A matrices learn general knowledge and B matrices learn client-specific knowledge, so only A is aggregated. Source: [arXiv](https://arxiv.org/pdf/2410.01463v2).
  - **Contradicted by** Ban & Ji, "Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs" (ACL 2026 Findings, arXiv 2509.25414). A's similarity across tasks or clients is "largely attributable to the identical initialization rather than shared knowledge", and **B encodes knowledge**. They propose ALoRA (multiple A, shared B) and Fed-ALoRA. Source: [arXiv](https://arxiv.org/abs/2509.25414).
- **HydraLoRA** (Tian, Shi, Guo, Li, Xu, arXiv 2404.19245). Asymmetric LoRA framework. Source: [arXiv](https://arxiv.org/abs/2404.19245). The details are **UNVERIFIED**, because the abstract page did not state them. From prior knowledge: one shared A, multiple B heads, an MoE router, and a NeurIPS 2024 acceptance.
- **Personalized merging (P-ShareLoRA)** — "A Shared Low-Rank Adaptation Approach to Personalized RLHF", arXiv 2503.19201. Applies LoRA to an aggregated parameter matrix that captures both individual preferences and shared structure. Source: [arXiv](https://arxiv.org/pdf/2503.19201).
- **Further merging methods.** All **UNVERIFIED** this session, from prior knowledge:
  - Task arithmetic: Ilharco et al., ICLR 2023, [arXiv 2212.04089](https://arxiv.org/abs/2212.04089).
  - TIES-merging (trim, elect sign, merge): Yadav et al., NeurIPS 2023, [arXiv 2306.01708](https://arxiv.org/abs/2306.01708).
  - DARE (drop-and-rescale): Yu et al., ICML 2024, [arXiv 2311.03099](https://arxiv.org/abs/2311.03099).
  - LoraHub, gradient-free few-shot composition of LoRA modules: Huang et al., [arXiv 2307.13269](https://arxiv.org/abs/2307.13269).
  - Personalized Soups, merging per-preference-dimension policies at inference: Jang et al., [arXiv 2310.11564](https://arxiv.org/abs/2310.11564).
  - ExPO, extrapolating θ_aligned + α(θ_aligned − θ_SFT) to strengthen alignment: Zheng et al., [arXiv 2404.16792](https://arxiv.org/abs/2404.16792).

### Inferences

**How to compute the preference direction:**
- Independently trained LoRAs (C3, C5) have different random A inits (uniform(±1/√d_in) in mlx-vlm), so factor-level arithmetic is meaningless. Ban & Ji's point about init-driven A similarity reinforces this.
- Compute per-module dense ΔW = scale·(B^T A^T in mlx layout), take differences, and re-factor with truncated SVD to rank r. That is 196 modules (7 per layer × 28 layers). The largest is 18944×3584, which a randomized SVD handles in seconds per module on a 96 GB Mac. This is an engineering estimate.

**Candidate operators** (each costs only offline evaluation):
- (a) **Amplify the residual.** In the stacked setup, W = W0 + ΔW_pool + s·ΔW_user with s ∈ {1, 1.5, 2}. This is the ExPO-style fix for "missed preference steps".
- (b) **Swap habits.** W = W0 + ΔW_C5 + λ·(ΔW_C3 − ΔW_C5). λ=1 recovers C3. λ ∈ (0, 1] trades grounding learned from other users against the target user's preferences.
- (c) **Ethos-style.** Project Δ_pref onto the low-variance subspace of ΔW_pool before adding it, to avoid overwriting grounding.

**Expected value:**
- Modest as a method; high as **analysis**, for example measuring the per-layer norm of Δ_pref to show *where* preferences live (feeds Q4).
- Interpolation between C3 and C5/C6 is a cheap way to test whether one weight-space direction carries the preference.

**A/B sharing for the per-user component:**
- Training only B on top of a frozen pooled A (FedSA-LoRA style) halves the per-user params and may stabilize training.
- Ban & Ji suggest the opposite factorization. The evidence is mixed, so treat this as a low-priority ablation.

### Gaps
- I found no result applying task-arithmetic subtraction to isolate **user preference** (as opposed to toxicity or tasks) in agents or VLMs.
- I found no evidence on whether GUI preference deltas are low-rank or near-orthogonal to grounding deltas. This would need measurement on our adapters.
- The arithmetic paper's detoxification numbers and the HydraLoRA design details were not verified.

---

## Q4. Splitting by layer or module: share grounding layers, adapt preference layers

### Takeaway
VLM interpretability work locates image→text integration in the **lower-to-middle** LLM layers. Deep layers attend little to image tokens: FastV prunes 50% of image tokens after layer 2 in LLaVA-1.5 with no average loss.

So a per-user component restricted to the **upper half of the LLM**, and to **MLP** modules (which beat attention-only LoRA in "LoRA Without Regret"), is a reasonable bet for "choices, not grounding". The evidence is indirect: no paper tests where user preferences live in GUI agents. Treat it as an ablation, not a default.

### Cited Findings
- **Cross-modal information flow.** Zhang et al., "Cross-modal Information Flow in Multimodal Large Language Models", CVPR 2025, arXiv 2411.18620. Uses attention knockout on LLaVA-family models for VQA:
  - **Lower layers** move general whole-image features into the question tokens.
  - **Middle layers** move object-specific visual information to the relevant question tokens.
  - **Higher layers** propagate the multimodal representation to the last position for prediction.
  - Sources: [arXiv](https://arxiv.org/abs/2411.18620), [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf).
- **Complementary evidence**, partly in tension: arXiv 2601.08151 (title not captured) uses layer-wise masking of visual features. It finds fusion mainly at a few **shallow** layers, plus a "review" where a **deep** layer re-attends to the image before output. Source: [arXiv](https://arxiv.org/pdf/2601.08151).
- **FastV.** Chen et al., "An Image is Worth 1/2 Tokens After Layer 2", ECCV 2024, arXiv 2403.06764. In deep layers of LLaVA-1.5, image tokens get little attention (one version reports 0.21% of the score given to system prompts). Pruning 50% of image tokens after layer 2 in LLaVA-1.5-13B causes no average loss on the tested tasks and cuts FLOPs by 45%. Source: [arXiv](https://arxiv.org/html/2403.06764v3).
  - **Critique.** FasterVLM (arXiv 2412.01818) argues that text→visual attention is an unreliable importance signal because of "attention shift". Source: [arXiv](https://arxiv.org/html/2412.01818v1).
- **Image-LoRA.** "Towards Minimal Fine-Tuning of VLMs", arXiv 2512.19219, Dec 2025. Adapters only on the attention **value path of visual tokens**, on selected heads, match or come close to standard LoRA on **screen-centric grounding** benchmarks, with fewer params, while preserving text-only behavior. Source: [arXiv](https://arxiv.org/html/2512.19219).
- **Module choice.** "LoRA Without Regret": **attention-only LoRA significantly underperforms MLP-only**. Attention-only at r=256 (0.25B params) loses to MLP-only at r=128 (0.24B), and adding attention to MLP adds nothing. This held for Llama-3.1-8B and Qwen3-30B-A3B, in small-data SFT and in RL. Text-only benchmarks. Source: [Thinking Machines](https://thinkingmachines.ai/blog/lora/).
- **OPPU and P2P** use q/v-only LoRA for per-user modules (text personalization, where no grounding is needed). Sources: [OPPU](https://arxiv.org/html/2402.04401), [P2P](https://arxiv.org/html/2510.16282).
- **ZonUI-3B** (arXiv 2506.23491) trains GUI grounding with LoRA on Qwen2.5-VL-3B in two stages (cross-platform, then high-resolution) without restricting depth. Source: [arXiv](https://arxiv.org/html/2506.23491v3).

### Inferences

**Proposed split for our 28-layer model** (to test, not assume):
- **Shared/pooled stage:** all 28 layers, all linear modules. Leave the vision tower frozen, as it is now.
- **Per-user residual:** layers 14–27, MLP only (gate/up/down), rank 4–8. That is 7.6M params at r=8 or 3.8M at r=4, vs 40.4M now. (Computed: 67,584·r per layer × 14 layers.)
- In mlx-vlm this needs only `--target-modules` with qualified names such as `model.layers.20.mlp.down_proj`, which `_resolve_lora_targets` accepts.

**Why it could work:**
- Preference steps in our data are **value and branch choices**: type "14:00" vs "15:00", click "30 min" or skip it. They are not localization problems.
- The C5 failure (right UI handling, wrong values) shows such choices are separable from grounding.
- Upper-layer MLPs are the usual place for "which token or action to emit". That is the standard key-value-memory view of FFNs, which was **UNVERIFIED** this session (Geva et al., [arXiv 2012.14913](https://arxiv.org/abs/2012.14913)).

**Risks:**
- The first coordinate digit of a click is a "decision token" (STATUS §3), so preference *clicks* (choosing the "30 min" button) need the model to map a choice to a screen location. Upper-only adapters may pick the right intent but click wrongly.
- The deep "review" layer in 2601.08151 also suggests grounding is not purely shallow.
- **Measure, don't assume.** Run the per-layer Δ_pref norms from Q3 and a {all-layers, top-half MLP, top-half attention+MLP} ablation on 2 dev pairs.

### Gaps
- I found no study of which layers of Qwen2.5-VL / UI-TARS carry GUI grounding vs action choice.
- I found no study of where **user preferences** (as opposed to facts, safety, or personas) live in agent models.
- "LoRA Without Regret" is text-only. Image-LoRA addresses grounding, not preferences.
- The persona-vector and steering literature, which locates behaviors in middle layers, was not searched this session.

---

## Q5. Pooled pretraining risk: do conflicting labels average out preferences, and should pooled training use a user/persona token?

### Takeaway
Yes, untagged pooling is risky in exactly our design. Every task instance exists for all 3 personas with **conflicting labels on identical inputs** (STATUS's definition of preference-critical steps). Cross-entropy then fits a mixture, and greedy decoding returns the plurality habit. Our own C5 already shows "other users' habits" at 0/9.

The literature consistently handles disagreement by **conditioning a shared model on user identity**:
- HYDRA: per-user heads.
- PLoRA: user embedding with personalized dropout for cold start.
- P-RLHF: per-user embeddings plus a generic embedding for unseen users.
- Multi-annotator learning: per-annotator heads instead of majority vote.

A user tag (and optionally a one-line preference profile) in the pooled stage is the cheapest fix and a direct test of the averaging hypothesis.

### Cited Findings
- **P-RLHF.** Li et al., "Personalized Language Modeling from Personalized Human Feedback", arXiv 2402.05133. Standard RLHF assumes one shared preference distribution, which fails when preferences are diverse or conflicting. P-RLHF trains a lightweight **user model** jointly with the LLM, giving a user-specific embedding per known user plus a **generic embedding for unseen users**, via personalized DPO. Tested on summarization with GPT-J 6B. Source: [arXiv](https://arxiv.org/html/2402.05133v3).
- **Shared-reward averaging — UNVERIFIED** (from a search snippet; identifier and date not confirmed). "Using Context Is Not Enough: Test-Time Training for Personalized Reward Modeling", arXiv 2609.35109, argues that a single shared reward model fit to conflicting labels **averages them toward the majority**, and proposes storing user history in parameters (P-TTT). Source: [arXiv](https://arxiv.org/pdf/2609.35109).
- **Per-person baseline.** "When Personalization Meets Reality: A Multi-Faceted Analysis of Personalized Preference Learning", arXiv 2502.19158. Fine-tuning a separate reward model per person is a strong baseline that collaborative methods must beat. Source: [arXiv](https://arxiv.org/html/2502.19158).
- **HYDRA**: shared base for common knowledge plus per-user heads for preferences, +9.01% average relative gain on LaMP. Source: [arXiv](https://arxiv.org/abs/2406.02888).
- **PLoRA**: user-preference embedding injected into a shared LoRA model, with **personalized dropout** and mutual-information maximization so the model works for few-shot and zero-shot users. Source: [arXiv](https://arxiv.org/abs/2403.06208).
- **LINEUP**: independent user adapters share substantial reusable structure, which supports learning the shared part from all users. Source: [arXiv](https://arxiv.org/abs/2610.02353).
- **Project evidence.** The C5 adapter (Bob and Carol, calendar) gets **0/9** of Alice's preference-critical steps and types Bob's 15:00/11:00 where Alice uses 14:00/10:00. Preference-critical steps are those where the target persona's action differs from every other persona's **on the same task**. Source: [STATUS.md](/Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md).
- **Conditional training and multi-annotator learning** — **UNVERIFIED** this session, from prior knowledge:
  - Korbak et al., "Pretraining Language Models with Human Preferences", ICML 2023, [arXiv 2302.08582](https://arxiv.org/abs/2302.08582). **Conditional training** (prefixing control tokens to the data) was the best of several feedback-aware objectives.
  - Davani, Díaz, Prabhakaran, "Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations", TACL 2022, [arXiv 2110.05719](https://arxiv.org/abs/2110.05719). A multi-task model with per-annotator heads matches or beats majority-vote training and keeps the minority signal.
  - Lee et al., "Aligning to Thousands of Preferences via System Message Generalization" (Janus), [arXiv 2405.17977](https://arxiv.org/abs/2405.17977). Putting preferences in the system message lets one model serve many preference profiles.

### Inferences

**What untagged pooling (C6) will do:**
- At a preference-critical step, the training targets for the same screenshot and instruction are, for example, "14:00" ×1, "15:00" ×1, and a third persona's value ×1.
- The model learns a roughly uniform mixture, and greedy decoding gives one persona's habit (or a token-level hybrid such as "1" + "1:00").
- For a **skipped-step** preference ("click 30 min" vs leave the default), the pooled model learns the 2-of-3 majority behavior. That is very likely why C3-style adapters, and any pooled base, skip Alice's "30 min".
- So C6 should match or beat C3 on non-preference steps (12× more data across skills), but be at or below chance on preference-critical steps.

**Tagged pooling (C6-tag):**
- Prefix the system prompt with `User: alice` (or a one-line, persona-written preference profile). This makes the supervision consistent, so the shared weights learn grounding from all 60 demos and the tag absorbs the preference.
- Add PLoRA- and P-RLHF-style **tag dropout** (about 20% of samples with `User: unknown`). The model then also learns an unconditional mode, which serves as the stage-1 base for a new user. Train stage 2 on the new user's 5 demos, either with a new tag (embedding-only tuning is not available in mlx-vlm's LoRA CLI, so use a small LoRA) or with the generic tag plus a per-user residual LoRA.

**The C6 vs C6-tag comparison is the cleanest experiment for the "averaging" question.** Hold out pref-critical and other-step accuracy per persona, with the same data and compute. It isolates conditioning as the only variable.

**Alternative to tagging: preference-masked pooled loss.**
- Zero the loss on preference-critical steps of the **train** demos in the pooled stage. These steps are computable from the train demos by comparing personas on the same task instance; the checker-based definition already exists for evaluation.
- The pooled adapter then learns only consensus operation ("how to operate the app"), and the per-user stage learns the rest.
- This is a cleaner "skill only" base than untagged C6, but it needs a train-time pref-critical labeler. It also leaks the experimental design (it assumes we know which steps are preferences). Report it as an oracle-ish upper bound.

**Caveat on closed vs open world:**
- With only 3 users, a tag model evaluated on seen users is closed-world: multi-task learning, not few-shot personalization of a new user.
- To claim personalization from 5 demos, also report leave-one-user-out: stage 1 on the 2 other users (that is C5, or C6-tag without the target user), then stage 2 on the target user's 5 demos.

### Gaps
- I found no GUI-agent or VLM paper that measures preference averaging under pooled SFT with conflicting demos. The evidence is from text personalization, reward modeling, and annotation, plus our own C5.
- Whether a short text profile ("prefers 14:00 starts, 30-min meetings, video on") beats an opaque ID tag in the pooled stage is untested here. Janus-style results suggest profiles generalize better but are **UNVERIFIED**.
- The P-TTT (2609.35109) claims are unconfirmed.

---

## Q6. Ranked recipe for our setup, with hours and compute

### Takeaway
1. First train **one tagged pooled adapter** (C6-tag). It is a single 2.5 h run that tests the averaging hypothesis and is likely the strongest closed-world arm.
2. Then run the **OPPU-faithful two-stage arm**: frozen C5 or C6 base, plus a short, low-LR per-user stage. Do continued fine-tuning first (zero code), then a stacked small residual.
3. Add the training-free **residual-scale and arithmetic** analyses, and a **top-half MLP-only** per-user ablation.

Skip meta-learning and hypernetworks: with 3 users and 12 tasks they are not feasible.

### Cited Findings
- Cost basis: about 5.4 s per training iter (280 iters ≈ 25 min). Online agent steps take 2.4–3.3 s, with 11–19 steps per episode. The test set is 3 test demos (42 steps) per user-skill. Source: [STATUS.md](/Users/hanmozhang/schoolwork/cmu-agents-project/STATUS.md).
- C5, C6, and N1 are already implemented. C3, C5, and C6 are scheduled in the overnight batch with `--epochs 4`. Sources: [train/train_adapters.py](/Users/hanmozhang/schoolwork/cmu-agents-project/train/train_adapters.py), [scripts/overnight.sh](/Users/hanmozhang/schoolwork/cmu-agents-project/scripts/overnight.sh).
- Stage-2 LR precedent is 1e-5 (OPPU, AdaptAgent). The LoRA-to-full-FT LR ratio is about 10× (about 15× for runs under 100 steps). Sources: [OPPU](https://arxiv.org/html/2402.04401), [AdaptAgent](https://arxiv.org/html/2411.13451), [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).
- Overfitting grows with local epochs when fine-tuning from a global model. Source: [arXiv 2511.12695](https://arxiv.org/html/2511.12695v1).

### Inferences

**Ranked recipe.** All steps assume the current mlx-vlm 0.8.0 CLI unless marked "code". Times are wall-clock on the M5 Ultra at about 5.4 s/iter. The first number is a 2-dev-pair pilot; the full figure is all 12 user-skills. Evaluation estimates: offline (teacher-forced, 42 steps) about 2–3 min per user-skill, or about 30 min per arm; online about 3 episodes × 15 steps × 3 s ≈ 2–3 min per user-skill plus environment overhead, or about 30–45 min per arm.

**1. C6-tag: tagged pooled adapter.** *Highest expected value per hour.*
- **Recipe:**
  - Data: all 60 train demos (about 840 samples). Prepend `User: <persona>` to the system prompt (optionally plus a one-line profile). Use 20% tag dropout to `User: unknown`.
  - LoRA: rank 16, α 16, all linear modules. lr 5e-5, warmup 20, **2 epochs** (more data, fewer passes).
  - Evaluate seen users **with their tag and no stage 2**.
- **Cost:** about 1,680 iters ≈ **2.5 h** train. Code is about 0.5 h (one `build_sft.py` flag). Eval about 1 h.
- **Decision rule:**
  - If C6-tag ≥ C3 on pref-critical steps and > C3 on other steps, then pooling with conditioning solves both failures in closed-world.
  - Compare to untagged C6 (already scheduled) for the "averaging" result.

**2. Two-stage, continued FT** (`--resume-adapter-file`). *Zero code, open-world-valid with a C5 base.*
- **Recipe:**
  - For each (u, s): start from C5(u, s), which is OPPU stage 1 with other users only, or from C6 or C6-tag with u's tag. Continue on u's 5 demos.
  - lr **2e-5** (try 5e-5), warmup 10, **2 epochs (about 140 iters)**, with a per-epoch checkpoint.
  - Select the checkpoint by held-out **pref-critical** step accuracy, not loss (STATUS §3: average loss hides the decision).
- **Cost:** about 13 min per user-skill. Pilot 0.5 h; full **2.6 h per base** (about 5.2 h for both the C5 and C6 bases). Eval about 1 h per base.

**3. Two-stage, stacked residual (OPPU-exact).** *Code; strength knob.*
- **Recipe:**
  - Load base, apply C5 or C6, fuse **in fp32/fp16** or keep it unfused (custom wrapper). Verify fused accuracy equals unfused.
  - Attach a fresh LoRA, **rank 8, α 8** (scale 1), all-linear or MLP-only. lr 2e-5 to 5e-5, 2–3 epochs.
  - At inference, sweep the residual scale s ∈ {1, 1.5, 2} (edit `scale` in the adapter config). This is the training-free fix for "misses preference steps".
- **Cost:** code about 2–4 h (custom training entry point that fuses in memory). Train the same as #2 (about 2.6–3.9 h per base). Scale sweep: 3 × offline eval ≈ 1.5 h.
- **Disk note:** avoid writing 12 fused 15 GB copies; fuse in memory.

**4. Layer/module-restricted per-user residual.** *Ablation for the "where preferences live" story; also a stability lever.*
- **Recipe:** in #2 or #3, use `--target-modules` set to qualified names for layers 14–27 `mlp.{gate,up,down}_proj` at rank 4–8 (3.8–7.6M params). Compare with all-layers and with top-half attention+MLP.
- **Cost:** on 2 dev pairs × 3 configs × 13 min ≈ **1.3 h** plus about 20 min eval. Scale to 12 pairs (about 2.6 h per config) only if it wins.

**5. Arithmetic and preference-direction analysis.** *No training.*
- **Recipe:** dense ΔW per module for C3, C5, and C6. Δ_pref = ΔW_C3 − ΔW_C5. Report per-layer and per-module ‖Δ_pref‖ / ‖ΔW_C3‖. Evaluate interpolations W0 + ΔW_C5 + λΔ_pref for λ ∈ {0.5, 1, 1.5}, optionally Ethos-style projected.
- **Cost:** code about 2 h. SVD re-factoring roughly 10–20 min per adapter (estimate). Eval 3 λ × 30 min ≈ 1.5 h for one pair set.

**6. Preference-masked pooled stage.** *Optional upper bound.*
- **Recipe:** like C6, but zero the loss on train-demo pref-critical steps (computed by comparing personas on the same task instance). Then do stage 2 as in #2.
- **Cost:** code about 1 h, train about 2.5 h, stage 2 about 2.6 h.

**7. Not recommended** for this project:
- FOMAML/Reptile meta-init: 12 tasks, and each inner loop costs minutes; AdaptAgent needed 292 tasks.
- Hypernetworks (P2P, LINEUP, Text-to-LoRA/PRISP): they need hundreds to thousands of users, and P2P amortizes after about 1,450 users. Cite them as future work.

**Budget summary:**
- Pilot on 2 dev pairs (#1 full, #2 C5 base, #4 three configs, #3 scale sweep, if #3's code is written) is about **8–10 h** wall-clock, about one overnight plus one afternoon.
- Full 12-pair study of the winners (#1 + best of #2/#3 on both bases + #5 analysis) is about **12–16 h** of compute.
- Peak memory stays within the 96 GB Mac: bf16 model about 18 GB, plus LoRA optimizer state for at most 40M params. Everything runs locally; no cloud.

**Defaults for any per-user stage on a pooled base:**
- rank ≤ 8, α = rank, lr 2e-5 to 5e-5, warmup 10–20, 2 epochs, batch 1.
- Select by held-out pref-critical accuracy. Never go above lr 1e-4 with batch 1 and 40M params.

**Protocol must-haves:**
- Report closed-world (C6 or C6-tag base) and leave-one-user-out (C5 base) separately.
- Always report pref-critical and other-step accuracy separately; the confidence intervals are wide (±30 points on 9 pref-critical steps), so use all 12 pairs before claiming a result.

### Gaps
- Iteration time for the 840-sample pooled run assumes the same 5.4 s/iter as per-user runs. Longer multi-image contexts could be slower; this is unmeasured.
- Stage-2 benefit over C3 on non-preference steps is unproven on our data: in the dry run, C5's other-step accuracy (58%) was not above C3's (61%). Pooling's grounding gain must be shown with human demos.
- Fused-in-bf16 precision loss is an inference from mlx-vlm code. Its practical size is unmeasured.
