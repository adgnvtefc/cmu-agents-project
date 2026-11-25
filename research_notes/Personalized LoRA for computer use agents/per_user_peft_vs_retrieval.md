# Per-user PEFT (one LoRA per user) vs retrieval / in-context personalization

Scope: text-LLM personalization literature (LaMP family), its 2025-2026 follow-ups, the closest GUI-agent / multimodal evidence, and the general few-shot fine-tuning vs ICL literature. Written for a project that trains per-user or per-skill LoRA on Qwen2.5-VL-7B / UI-TARS-1.5-7B from 3-5 GUI demonstrations and compares it against putting the same demos in context. Research date: 2026-11-20. Items not opened during this session are marked UNVERIFIED.

---

## Q1. One PEFT Per User (OPPU, Tan et al. 2024): method, results vs RAG/PAG on LaMP, complementarity, few-sample behavior

### Takeaway
On LaMP, OPPU's per-user LoRA beats prompt-only personalization mainly when it is **stacked on retrieval or profiles** (PEFT+RAG/PAG is the best configuration). It is also more robust than retrieval when retrieved history is irrelevant to the query ("behavior shift"). However, OPPU was evaluated on the **100 most active users** and on top of a **task-adapted base LoRA trained on other users**, so it says little about a 3-5 example regime. On generation tasks its margin over a tuned RAG baseline is tiny (≤0.4 ROUGE-1 points on LaMP-4/7).

### Cited Findings
**Paper card: "Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning" (OPPU).** Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, Meng Jiang. arXiv 2402.04401 (v3 revised Feb 2025); venue EMNLP 2024 (UNVERIFIED this session; an NSF PAR copy exists). Links: [arXiv](https://arxiv.org/abs/2402.04401), [HTML v3](https://arxiv.org/html/2402.04401v3), code github.com/TamSiuhin/OPPU ([search result](https://arxiv.org/html/2402.04401v3)).

- **Method:** Each user gets a private PEFT module (LoRA by default) on a frozen base LLM, trained on that user's behavior history, so "personalized PEFT parameters encapsulate behavior patterns and preferences." — [OPPU arXiv](https://arxiv.org/abs/2402.04401)
- **Two-stage design:** first a LoRA is trained on held-out users to adapt the base model to the task, in three variants (task-only Θ(B), retrieval-augmented Θ(R), profile-augmented Θ(P)), and **merged** into the base. The per-user LoRA is then trained on top. The base model is Llama-2-7B, retrieval is BM25 top-k, and profiles are generated with Vicuna-7B. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- When history items don't match the task's input-output format, OPPU trains with right-shifted history (next-token prediction on raw history). — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **Main results (Table 1, Llama-2-7B):**
  - LaMP-1 acc: non-personalized 0.659, RAG(k=4) 0.691, PAG 0.756, **PAG+OPPU 0.797**.
  - LaMP-2N acc: 0.787 / RAG 0.832 / **RAG+OPPU 0.838**.
  - LaMP-3 MAE: 0.223 / RAG 0.214 / PAG 0.321 / **PAG+OPPU 0.143**.
  - LaMP-4 R-1: 0.186 / RAG 0.198 / **RAG+OPPU 0.199**.
  - LaMP-5 R-1: 0.476 / RAG 0.505 / **RAG+OPPU 0.526**.
  - LaMP-7 R-1: 0.527 / RAG 0.577 / **RAG+OPPU 0.581**.
  - Source: [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **Complementarity:** adding retrieval to OPPU gives +1.93% average relative gain over OPPU without retrieval and **+2.48% over RAG without OPPU**. Adding profiles gives +4.56% over OPPU without profiles and **+7.18% over PAG without OPPU**. The authors conclude that parametric plus non-parametric is best. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **Behavior shift:** when only the 100 *least relevant* history items are available, retrieval-only "performs close to the non-personalized baseline" while OPPU stays more robust. The exact Table 3 cells were garbled in extraction. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **Retrieval depth:** as retrieved k grows, the gap between retrieval-only and retrieval+OPPU narrows (Fig. 4, no numbers in text). — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **User activity levels:** OPPU is "consistently better" across activity buckets (20 users per bucket). Figure 3 only; no numbers in text. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **Test set:** the main test set uses the **100 most active users**, so the paper reports no numbers for very sparse users. Training time grows linearly with the number of history items. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **PEFT type:** LoRA > (IA)³ > prompt tuning on LaMP-2M and LaMP-5 (Fig. 6). Parameter shares reported: LoRA 0.01%, (IA)³ 0.06%, prompt tuning 0.001%. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **Hyperparameters** (Table 5 / App. A):
  - LoRA rank 8 on W_q and W_v; alpha not reported.
  - LR 1e-5 for all tasks.
  - 3 epochs (classification/rating tasks), 2 epochs (generation tasks).
  - Batch size 3-16.
  - An "R2" regularization of 1e-2 (classification) or 1e-1 (generation), as extracted.
  - The paper is internally inconsistent on adapter size: "about 0.5%" vs "0.01%" of parameters.
  - Source: [OPPU HTML](https://arxiv.org/html/2402.04401v3)

### Inferences
- OPPU's per-user gains sit on top of a **task-adapted, merged base LoRA trained on many other users**. Its RAG baseline also uses a base fine-tuned for RAG prompts. For the project, the analogous control is a shared "GUI-skill" LoRA trained on pooled demos from many users. Without it, a per-user LoRA comparison mixes up "domain adaptation" with "personalization."
- The behavior-shift result supports a concrete hypothesis. Per-user LoRA should help most, relative to in-context demos, when the test task is **dissimilar** to the 3-5 demos (different app/screen/goal). ICL should win when a near-duplicate demo exists.
- The gain from OPPU over RAG on generation tasks is ≤0.4 R-1 points (LaMP-4/7). Large wins appear only on classification/rating (LaMP-1, LaMP-3). GUI action prediction is closer to structured classification (action type + target), but this is speculative.

### Gaps
- No numbers in the paper for users with ≤10 history items. The activity-level and k-sweep figures are not quantified in text.
- LoRA alpha and dropout are not reported. The parameter-share inconsistency is unresolved.

---

## Q2. Per-Pcs, HYDRA, PLoRA, LaMP/LongLaMP, PersonalLLM, and 2025-2026 follow-ups: which won, with how many examples per user?

### Takeaway
The cleanest head-to-head (Salemi & Zamani, ICTIR 2025) finds **RAG >> per-user PEFT on average** (+14.92% vs +1.07% over non-personalized), with **RAG+PEFT best** (+15.98%). PEFT's gain grows with the user's profile size, while RAG's gain shrinks. Every 2025-2026 follow-up moves *away* from independently trained per-user LoRAs toward **shared/collaborative parameters plus tiny per-user state**: Per-Pcs piece sharing, PROPER group LoRAs, Profile-to-PEFT hypernetworks, and LINEUP's 8-scalar user codes. Those beat OPPU, especially for sparse or cold-start users.

### Cited Findings
**Paper card: "Comparing Retrieval-Augmentation and Parameter-Efficient Fine-Tuning for Privacy-Preserving Personalization of Large Language Models."** Alireza Salemi, Hamed Zamani. arXiv 2409.09510 (Sep 2024, rev. Jun 2025), ICTIR '25. [arXiv HTML v2](https://arxiv.org/html/2409.09510v2)
- **Setup:** FlanT5-XXL (11B). Per-user LoRA trained on each user's profile, r ∈ {8,16,32,64}, α=32, dropout 0.1, on Q/K/V, 50 epochs, LR 5e-4 (linear, 5% warmup), weight decay 1e-4, effective batch 16. RAG uses k=4 retrieved items with BM25, Contriever, Recency, or RSPG retrievers. — [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)
- **Average results over 7 LaMP tasks:** RAG +14.92%, PEFT +1.07%, **RAG+PEFT +15.98%** over the non-personalized LLM. RAG+PEFT beats RAG alone on only 4/7 tasks. — [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)
- **Per-task examples:**
  - LaMP-1 acc: none 0.502 / best PEFT 0.506 / RSPG-RAG 0.672 / RAG+PEFT 0.668-0.671.
  - LaMP-3 MAE: 0.308 / 0.301 / 0.264 / **0.259**.
  - LaMP-6 R-1: 0.335 / 0.343 / 0.433 / **0.437**.
  - PEFT's largest gains are on LaMP-6, a private email dataset likely absent from pretraining.
  - Source: [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)
- **Data dependence:** PEFT gain correlates positively with the number of profile items on 5/7 datasets. RAG's gain *falls* as profiles grow. No numeric threshold is given; the figure is a regression on a ±1 gain indicator. — [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)
- **Cost:**
  - Storage: a r=8 adapter for FlanT5-XXL is ~55 MB, so ~55 TB per 1M users.
  - Compute: >10,000 A100-hours to train 37,560 adapters (~18 TB total).
  - Source: [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)

**Paper card: "Personalized Pieces: Efficient Personalized Large Language Models through Collaborative Efforts" (Per-Pcs).** Zhaoxuan Tan, Zheyuan Liu, Meng Jiang. EMNLP 2024 main. [ACL Anthology 2024.emnlp-main.371](https://aclanthology.org/2024.emnlp-main.371/), [arXiv 2406.10471](https://arxiv.org/abs/2406.10471)
- **Method:** selects "sharer" users, decomposes their PEFT modules into pieces, and trains a gate per piece. Target users assemble a personal PEFT from the shared pool using their own history (training-free assembly). On six LaMP tasks it outperforms non-personalized and PEFT-retrieval baselines and performs comparably to OPPU at significantly lower resource cost. — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.371/); assembly described as training-free in [MTA arXiv 2511.20072](https://arxiv.org/pdf/2511.20072)
- Exact per-task numbers were not retrieved (see Gaps).

**Paper card: "HYDRA: Model Factorization Framework for Black-Box LLM Personalization."** Yuchen Zhuang, Haotian Sun, Yue Yu, Rushi Qiang, Qifan Wang, Chao Zhang, Bo Dai. NeurIPS 2024. [arXiv 2406.02888](https://arxiv.org/abs/2406.02888), [NeurIPS poster](https://neurips.cc/virtual/2024/poster/96150)
- **Method:** for black-box LLMs, trains (i) a personalized reranker over retrieved history and (ii) a personalized adapter that steers outputs. The model is factorized into a **shared base** that captures cross-user knowledge plus **lightweight per-user heads**. — [HYDRA arXiv](https://arxiv.org/abs/2406.02888)
- **Result:** +9.01% average relative improvement over state-of-the-art RAG and PAG baselines across five LaMP tasks. Code at github.com/night-chen/HYDRA. — [HYDRA arXiv](https://arxiv.org/abs/2406.02888)
- The authors argue that prompt-only personalization "struggle[s] to generalize" because it cannot capture shared knowledge among users. — [HYDRA arXiv](https://arxiv.org/abs/2406.02888)

**Paper card: "Instant Personalized Large Language Model Adaptation via Hypernetwork" (Profile-to-PEFT, P2P).** Zhaoxuan Tan, Zixuan Zhang, Haoyang Wen, Zheng Li, Rongzhi Zhang, Pei Chen, Fengran Mo, Zheyuan Liu, Qingkai Zeng, Qingyu Yin, Meng Jiang. arXiv 2510.16282 (Oct 2025). ACL 2026 long per an [ACL Anthology listing](https://aclanthology.org/2026.acl-long.1081/) seen in search results (listing not opened). [arXiv HTML](https://arxiv.org/html/2510.16282)
- **Method:** the user profile (an LLM-generated preference summary plus top-k retrieved history) is embedded with Qwen3-Emb-4B. An MLP hypernetwork maps it, plus module and depth embeddings, to LoRA weights in one forward pass, with no per-user training. — [P2P](https://arxiv.org/html/2510.16282)
- **Setup:** Qwen2.5-7B-Instruct, generated LoRA rank 8 on q_proj/v_proj, LR 2e-5, 20k steps, batch 32. — [P2P](https://arxiv.org/html/2510.16282)
- **Baselines:** non-personalized, RAG, PAG, Full-History-in-context, **MT-LoRA** (one shared LoRA), and OPPU. — [P2P](https://arxiv.org/html/2510.16282)
- **Results:**
  - Random-user split, average classification acc: **P2P 0.580** vs OPPU 0.568 vs RAG 0.507. The text cites 0.577, which doesn't match the table.
  - Random-user split, generation ROUGE-L: P2P 0.244 vs OPPU 0.221.
  - **OOD-user split**, average classification acc: P2P 0.581 vs RAG 0.534 vs **OPPU 0.528**. OPPU drops below RAG on unseen-distribution users.
  - Source: [P2P](https://arxiv.org/html/2510.16282)
- **Speed:** ~0.57 s per user to generate parameters vs ~18.8-22.7 s for OPPU variants (~33×). The one-time hypernetwork training cost is 27,167 s, amortized after ~1,450 users. — [P2P](https://arxiv.org/html/2510.16282)

**Paper card: "Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation" (LINEUP).** Songyuan Sui, Srikanth Malla, Chiho Choi, Joon Hee Choi (Samsung Semiconductor US / Rice). arXiv 2610.02353, 1 Oct 2026; no venue listed. [arXiv HTML](https://arxiv.org/html/2610.02353)
- **Claim:** "rich personalization can be supported primarily by reusable, conditionally composed shared capacity, while independent user adaptation remains confined to a tiny correction state." — [LINEUP](https://arxiv.org/html/2610.02353)
- **Method:** a bank of 32 shared low-rank factors learned across source users. User history recalls candidate factors, and the query calibrates their composition, with four rank-4 factors active per request. Each target user optimizes only an **8-dimensional user code** (8 scalars), vs **4.19M per-user parameters** for the evaluated private-LoRA configuration. The backbone is Llama-2-7B, built on OPPU's task-adapted base. — [LINEUP](https://arxiv.org/html/2610.02353)
- **Results (Table 1):**
  - LaMP-2M acc: LINEUP .645 vs OPPU .619 vs PerFit .632.
  - LaMP-2N acc: .822 vs .801 vs .818.
  - LaMP-3 MAE: .152 vs .181 vs .172.
  - LaMP-7 ROUGE-L: .551 vs OPPU .522.
  - Baselines also included RAG, Per-Pcs, P2P, and MTA.
  - Source: [LINEUP](https://arxiv.org/html/2610.02353)
- **One history record (LaMP-2N):** LINEUP 0.806 acc / 0.560 F1 vs **OPPU 0.793 / 0.536**. With full history, LINEUP F1 reaches 0.621. — [LINEUP](https://arxiv.org/html/2610.02353)

**Other 2025-2026 follow-ups (lighter verification):**
- **PROPER** (arXiv 2503.01303, 2025) inserts a group-level stage between population and user. It has per-group LoRAs plus per-user LoRAs mixed by a LoRA-aware router, aimed at sparse-data users. — [search result / arXiv PDF](https://arxiv.org/pdf/2503.01303) (details beyond this UNVERIFIED)
- **Meta-LoRA** (arXiv 2608.12389) reports cold-start gains for unseen users on HiCUPID with PROPER as a baseline. — [search snippet](https://arxiv.org/html/2608.12389) (UNVERIFIED, not opened)
- **Cluster-level LoRA** (arXiv 2601.06352) combines cluster-level LoRAs with user-level personalization, motivated by per-user storage and cold-start costs. — [search snippet](https://arxiv.org/pdf/2601.06352v1) (UNVERIFIED)
- **On-device LoRA-generating hypernetwork** (arXiv 2609.24979) generates a personal LoRA once on-device from a frozen LLM used as an encoder. — [search snippet](https://arxiv.org/pdf/2609.24979) (UNVERIFIED)
- **PLoRA** ("Personalized LoRA for Human-Centered Text Understanding", AAAI 2024) — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29931/31627). Details not retrieved; see Gaps.

### Inferences
- **"Which won" depends on data per user and on the split.**
  - Averaged over *all* users (many sparse), RAG wins by a wide margin over PEFT alone (Salemi & Zamani).
  - On the *most active* users, with a task-adapted base, PEFT+RAG/PAG wins (OPPU).
  - On *new/OOD* users, shared or generated parameters (P2P, LINEUP) beat independently trained per-user LoRA, and OPPU can fall below RAG (P2P OOD split: 0.528 vs 0.534).
- **Reconciling OPPU vs Salemi & Zamani:** the two differ in user selection (top-100 active vs all), base model (Llama-2-7B with a task-adapted base vs FlanT5-XXL), and training recipe (2-3 epochs at 1e-5 vs 50 epochs at 5e-4). The PEFT-gain-vs-profile-size correlation is the likely common thread.
- **For a 3-5 demo regime, the literature predicts:**
  - (a) In-context demos ≥ per-user LoRA alone.
  - (b) LoRA + in-context demos ≥ either alone, but with a small margin. Text results show ~0.4-2.5% relative over RAG.
  - (c) A pooled multi-user "skill" LoRA (P2P's MT-LoRA analogue) captures much of the parametric gain.
- **Recommended baseline set for the project:**
  - Zero-shot base model.
  - All 3-5 demos in context (the "Full History" baseline, feasible because history is tiny).
  - Top-1 retrieved demo in context.
  - Pooled MT-LoRA / per-skill LoRA across users.
  - Per-user LoRA.
  - Per-user LoRA + demos in context.
- Shared-basis methods (Per-Pcs, LINEUP, P2P) are the right related work to cite as "what to do next." They are too heavy for 2 weeks.

### Gaps
- **Per-Pcs:** per-task numbers and its exact storage/compute savings not retrieved.
- **HYDRA:** whether the black box was GPT-3.5 and how the adapter works (reward-style candidate selection?) not confirmed; UNVERIFIED.
- **PLoRA** (Zhang et al., AAAI 2024; arXiv 2403.06208): only the AAAI link was seen. My recollection, UNVERIFIED: a plug-and-play personal-knowledge module for zero/few-shot users on sentiment datasets.
- **LaMP:** benchmark paper (Salemi, Mysore, Bendersky, Zamani; arXiv 2304.11406; ACL 2024) not opened; UNVERIFIED. Per-user profile sizes in LaMP (believed to be tens to hundreds of items) are not verified here.
- **LongLaMP** (Kumar et al., arXiv 2407.11016) and **PersonalLLM** (Zollo et al., arXiv 2409.20296, ICLR 2025): not opened; UNVERIFIED. No per-user-PEFT vs ICL numbers obtained from either.
- **PerFit and MTA:** baselines in LINEUP, not investigated.

---

## Q3. Very low data per user (≤10 examples): overfitting, rank, epochs, learning rate

### Takeaway
No paper found reports a clean per-user-LoRA sweep at ≤10 examples. The evidence is indirect:
- Per-user PEFT gains shrink toward zero as user data shrinks (Salemi & Zamani).
- With one history record, OPPU is barely above a non-personalized baseline (~0.793 vs ~0.787 on LaMP-2N).
- Methods that shrink per-user capacity win in sparse regimes: 8 scalars (LINEUP), generated LoRA (P2P), group LoRAs (PROPER).

Published recipes cluster around **LoRA r=8 on q/v**. Learning rates span 1e-5 to 5e-4, and epochs range from 2-3 at low LR to 50 at high LR.

### Cited Findings
- **Hyperparameters used by the core papers:**

| Paper | Base model | Rank / α / dropout | Target modules | LR | Epochs / steps | Batch | Per-user data |
|---|---|---|---|---|---|---|---|
| OPPU — [src](https://arxiv.org/html/2402.04401v3) | Llama-2-7B | r=8 / α n.r. / n.r. | W_q, W_v | 1e-5 | 2-3 epochs | 3-16 | full history of the 100 most active users |
| Salemi & Zamani — [src](https://arxiv.org/html/2409.09510v2) | FlanT5-XXL 11B | r∈{8,16,32,64} / 32 / 0.1 | Q, K, V | 5e-4, 5% warmup, wd 1e-4 | 50 epochs | eff. 16 | each user's LaMP profile |
| P2P (hypernetwork) — [src](https://arxiv.org/html/2510.16282) | Qwen2.5-7B-Instruct | generated r=8 | q_proj, v_proj | 2e-5 (hypernet) | 20k steps | 32 | profile summary + top-k items |
| LearnAct (GUI VLM, *not per-user*) — [src](https://arxiv.org/html/2504.13805v1) | Qwen2-VL-7B, UI-TARS-7B-SFT | r=64 / 128 / 0.1 | all LM modules, vision encoder frozen | 1e-5, cosine | 1 epoch | 1 × 8 grad-accum | LearnGUI-Offline training set |
| LINEUP — [src](https://arxiv.org/html/2610.02353) | Llama-2-7B | 32 shared rank-4 factors (4 active) + 8-dim user code | — | n.r. | — | — | down to 1 history record |

- **Rank:** Salemi & Zamani swept r=8-64. Best single-PEFT results were often at r=64 (LaMP-1 0.506, LaMP-3 0.301, LaMP-6 0.343), but all PEFT-only gains were tiny. — [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)
- **Data dependence:** PEFT gain rises with profile size on 5/7 LaMP tasks. On LaMP-5, drops were associated with users who were not primary authors of their history (17/18 drop cases). Noisy or "not-really-the-user's" history hurts per-user PEFT. — [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)
- **One-record regime (LaMP-2N):** OPPU 0.793 acc / 0.536 F1; LINEUP (8 scalars per user) 0.806 / 0.560. — [LINEUP](https://arxiv.org/html/2610.02353). The non-personalized Llama-2-7B baseline is 0.787 in OPPU's paper. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)
- **Capacity:** LINEUP's per-user state is 8 scalars vs 4.19M parameters for a private LoRA, and it outperforms OPPU on LaMP-2M/2N/3/7. — [LINEUP](https://arxiv.org/html/2610.02353)
- **Few-shot PEFT recipe (T-Few):** uses (IA)³, which adds only learned activation-scaling vectors, as a fixed recipe that "applies to new tasks without task-specific tuning." — [T-Few arXiv](https://arxiv.org/abs/2205.05638). My recollection of the recipe, UNVERIFIED this session: ~1,000 steps, batch 8, LR 3e-3, plus unlikelihood and length-normalized losses.
- **Instability:** few-shot fine-tuning and ICL both show high variance in- and out-of-domain. Checkpoint selection strongly affects conclusions; selecting on OOD data inflates OOD results. — [Mosbach et al. 2023](https://arxiv.org/abs/2305.16938)

### Inferences
- **Effective sample count:** a GUI demonstration is a multi-step trajectory, so 3-5 demos yield maybe 20-100 (screenshot, history, action) step samples. That is closer to T-Few's few-shot regime than to "3 examples." It is still far below LaMP's per-user histories, and steps within a demo are highly correlated.
- **Overfitting controls** (recommendations derived from the above, not tested in the cited work):
  - Keep per-user capacity small: r=4-8 on q/v (OPPU/P2P convention), or (IA)³ as a cheaper ablation.
  - Use a pooled skill LoRA as the starting point.
  - Run leave-one-demo-out validation for early stopping.
  - Report mean ± std over ≥3 seeds; Mosbach shows single-run conclusions are unreliable.
  - Never select checkpoints on the test tasks.
- **Suggested sweep within budget:** LR {1e-5, 1e-4, 5e-4} × epochs/steps {few, ~10, ~30 epochs-equivalent} × rank {4, 8, 16}. The published range of 1e-5 (OPPU, LearnAct) to 5e-4 (Salemi & Zamani) is wide. LearnAct's r=64/α=128 on all modules is a *dataset-scale* setting and likely too large for per-user data.
- **Data quality:** demos must actually reflect the user's behavior. Salemi's LaMP-5 finding implies that mixed or noisy demos make per-user PEFT worse than no personalization.

### Gaps
- No paper found that sweeps per-user LoRA rank/LR/epochs specifically at 1-10 examples and reports overfitting curves.
- OPPU's LoRA alpha and dropout are not reported. LINEUP's Appendix E.5 (private-LoRA hyperparameters) was not read.
- T-Few's exact hyperparameters were not verified in this session.

---

## Q4. Per-user PEFT for agents or multimodal models

### Takeaway
I found **no published work that trains a per-user LoRA for a GUI/computer-use agent from a handful of that user's demonstrations and compares it to ICL**. This appears to be an open gap, which supports the project's novelty. The closest evidence is LearnAct (2025): **in-context demonstrations** substantially improve GUI VLMs. UI-TARS-7B-SFT online success went from 18.1% to 32.8% with one retrieved demo. Its fine-tuning is general (dataset-level, LoRA r=64), not per-user, and the already-GUI-specialized UI-TARS saturates at 1 in-context demo.

### Cited Findings
**Paper card: "LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark."** Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, Wenchao Meng. arXiv 2504.13805 (Apr 2025). [arXiv](https://arxiv.org/abs/2504.13805), [HTML](https://arxiv.org/html/2504.13805v1), [project](https://lgy0404.github.io/LearnAct/)
- **Benchmark and method:** LearnGUI has 2,252 offline and 101 online tasks, each with human demonstrations. LearnAct is a three-agent pipeline: DemoParser extracts knowledge, KnowSeeker retrieves the relevant demo knowledge, and ActExecutor acts. — [LearnAct abs](https://arxiv.org/abs/2504.13805)
- **Fine-tuning:** Qwen2-VL-7B and UI-TARS-7B-SFT were fine-tuned on the LearnGUI-Offline training set with LoRA r=64, α=128, dropout 0.1, all LM modules, vision encoder frozen, LR 1e-5 cosine, 1 epoch, batch 1 × 8 grad-accum. It is unclear whether the 0-shot baselines also use the fine-tuned weights. — [LearnAct HTML](https://arxiv.org/html/2504.13805v1)
- **Offline action-match accuracy, 0 → 1 → 2 → 3 in-context demos:**
  - Gemini-1.5-Pro: 19.3 → 51.7 → 55.6 → 57.7.
  - UI-TARS-7B-SFT: 77.5 → **82.8** → 81.9 → 82.1 (peaks at 1-shot).
  - Qwen2-VL-7B: 71.8 → 77.3 → 78.5 → 79.4.
  - Source: [LearnAct HTML](https://arxiv.org/html/2504.13805v1)
- **Online task success with 1 retrieved demo:** Qwen2-VL-7B 9.9 → 21.1; UI-TARS-7B-SFT 18.1 → 32.8 (GPT-4o 34.5 for reference). — [LearnAct HTML](https://arxiv.org/html/2504.13805v1)
- **Demo-task similarity:** the authors claim UI similarity correlates with gains. Their own Table 5 doesn't clearly support this for Gemini, whose highest score is in the low-UI-similarity / high-action-similarity quadrant (60.0 at 3-shot). — [LearnAct HTML](https://arxiv.org/html/2504.13805v1)

**Adjacent signals (search snippets only, UNVERIFIED):**
- **ALLOY** (arXiv 2510.10049) adapts agents to user preferences via few-shot use of user demonstrations. It names continued fine-tuning on user demos as future work. — [ALLOY](https://arxiv.org/pdf/2510.10049)
- **SEE** (arXiv 2607.18046), a GUI trajectory-synthesis paper, reportedly finds fine-tuning plus "+context" gives further gains, smaller than context gives without fine-tuning. — [SEE](https://arxiv.org/pdf/2607.18046)
- **ShowUI-Aloha: Human-Taught GUI Agent** (arXiv 2601.07181) is title-only. — [ShowUI-Aloha](https://arxiv.org/pdf/2601.07181)
- **Personalization survey** (arXiv 2504.10147, "From RAG to Agent") notes LoRA is the most widely adopted PEFT for personalization. Its examples are mostly text generation. — [Survey](https://arxiv.org/pdf/2504.10147)

### Inferences
- **Key ICL baseline:** LearnAct's numbers set the ICL baseline the project must beat or complement. A single relevant demo in context roughly doubles online success for 7B GUI models.
- **UI-TARS as base model:** UI-TARS peaks at 1 demo and declines slightly with more, so a GUI-specialized model may not use extra in-context screenshots well. That leaves room for parametric (LoRA) personalization on UI-TARS-1.5-7B, and is a testable hypothesis: "LoRA helps more for UI-TARS than for Qwen2.5-VL, which uses ICL better."
- **Context cost:** multi-screenshot demos are expensive in context (many image tokens per step). Context length and latency are a legitimate axis where per-user LoRA can win even at equal accuracy, mirroring the cost arguments in the text literature.

### Gaps
- No verified paper on per-user or per-skill PEFT for GUI/web agents. **PersonalWAB** (Cai et al., arXiv 2410.17236, WWW 2025, personalized web-agent benchmark with fine-tuned function-parameter generation) was not opened; UNVERIFIED.
- No verified data on **multimodal concept personalization** that learns per-user tokens or adapters from ~3-10 images: MyVLM (Alaluf et al. 2024), Yo'LLaVA (Nguyen et al., NeurIPS 2024). These are relevant analogues for "few-shot per-user parametric adaptation of a VLM" but were not opened; UNVERIFIED.

---

## Q5. ICL vs fine-tuning in few-shot regimes, and the general lesson for 3-5 examples

### Takeaway
There is no universal winner. Each finding holds under specific conditions:
- **T-Few (2022):** few-shot PEFT can be *better and cheaper* than ICL with a good PEFT recipe and a task-pretrained base.
- **Mosbach et al. (2023):** under matched models and examples, few-shot FT and ICL generalize *similarly* and both are high-variance.
- **Lampinen et al. (2025):** in data-matched settings, **ICL generalizes more flexibly** than fine-tuning (reversals, simple deductions). Adding in-context-generated inferences to the fine-tuning data closes much of the gap.

For 3-5 demos, expect ICL to be the stronger single method when demos are relevant and fit in context. Expect LoRA to add value through robustness to irrelevant or shifted test tasks, lower inference cost, and complementarity with ICL. Treat any single-seed comparison as unreliable.

### Cited Findings
**Paper card: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (T-Few).** Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, Colin Raffel. arXiv 2205.05638 (2022). The arXiv page lists no venue; NeurIPS 2022 per my recollection, UNVERIFIED. [arXiv](https://arxiv.org/abs/2205.05638)
- Few-shot PEFT gives "better accuracy as well as dramatically lower computational costs" than few-shot ICL. Introduces (IA)³ (learned activation-scaling vectors) and the T-Few recipe on T0, which reaches super-human performance on RAFT and beats the prior state of the art by 6% absolute. — [T-Few](https://arxiv.org/abs/2205.05638)
- The FLOPs advantage over GPT-3 ICL (~1000× fewer inference FLOPs, per recollection) is UNVERIFIED this session.

**Paper card: "Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation."** Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, Yanai Elazar. Findings of ACL 2023, arXiv 2305.16938. Author list from recollection, UNVERIFIED. [arXiv](https://arxiv.org/abs/2305.16938), [ACL PDF](https://aclanthology.org/2023.findings-acl.779.pdf)
- When model size, number of examples, and parameter count are controlled (OPT 125M-30B, on RTE/MNLI), few-shot FT and ICL generalize similarly out-of-domain. Both show large variation depending on model size and number of examples. — [Mosbach et al.](https://arxiv.org/abs/2305.16938)
- On MNLI, only the 30B model showed good OOD generalization, in 4 of 10 runs. With checkpoints selected on OOD data, "almost all models perform better out-of-domain than in-domain," which is a selection artifact. — [Mosbach et al. via search summary](https://arxiv.org/abs/2305.16938)

**Paper card: "On the generalization of language models from in-context learning and finetuning: a controlled study."** Andrew K. Lampinen, Arslan Chaudhry, Stephanie C.Y. Chan, Cody Wild, Diane Wan, Alex Ku, Jörg Bornschein, Razvan Pascanu, Murray Shanahan, James L. McClelland. arXiv 2505.00661 (May 2025, v3 Nov 2025); NeurIPS 2025 FoRLM workshop. [arXiv](https://arxiv.org/abs/2505.00661)
- In data-matched settings ICL generalizes several kinds of inference more flexibly than fine-tuning. Fine-tuning fails at simple reversals and simple logical deductions, though it can generalize reversals embedded in a larger knowledge structure. — [Lampinen et al.](https://arxiv.org/abs/2505.00661)
- **Fix:** augment fine-tuning data with in-context inferences (reasoning traces the model produces with the data in context). This improves generalization across splits and other benchmarks. — [Lampinen et al.](https://arxiv.org/abs/2505.00661)

**Personalization-specific evidence (from Q1/Q2):**
- RAG +14.92% vs PEFT +1.07% vs both +15.98% on LaMP. — [Salemi & Zamani](https://arxiv.org/html/2409.09510v2)
- Retrieval collapses toward non-personalized when retrieved items are irrelevant, while per-user PEFT is more robust. — [OPPU HTML](https://arxiv.org/html/2402.04401v3)

### Inferences
**General lesson for 3-5 demos:**
1. ICL with all demos is the strongest simple baseline when the test task resembles a demo (LearnAct, Salemi & Zamani, Lampinen).
2. Per-user LoRA alone will likely match or slightly trail ICL. Its plausible advantages are robustness when test tasks differ from the demos (OPPU behavior shift), zero extra context tokens at inference (T-Few cost argument), and stacking with ICL (OPPU, Salemi & Zamani).
3. Gains will be noisy, so multiple seeds and a held-out-demo protocol are required (Mosbach).

**Concrete hypotheses to pre-register:**
- **H1:** ICL(3-5 demos) > per-user LoRA on test tasks similar to the demos.
- **H2:** The per-user LoRA vs ICL gap narrows or reverses on dissimilar tasks (different app or screen).
- **H3:** LoRA + ICL ≥ max(LoRA, ICL), with a small margin.
- **H4:** A pooled per-skill LoRA (trained across users) captures most of the per-user LoRA's gain. This is the "personalization vs domain adaptation" control.
- **H5:** Augmenting LoRA training data, Lampinen-style, narrows the gap to ICL. Examples: model-generated rationales or thoughts for each demo step, paraphrased task instructions, or the same demo replayed with in-context variations.

**Cheap ablations within 2 weeks:**
- Rank {4, 8, 16} on q/v vs all-linear.
- (IA)³-style minimal adapter vs LoRA, following T-Few/OPPU.
- Number of demos k ∈ {1, 3, 5} for both ICL and LoRA.
- Demo-to-test similarity buckets.

### Gaps
- Many-shot ICL vs FT studies were not retrieved this session, e.g., Bertsch et al. 2024, "In-Context Learning with Long-Context Models," and Agarwal et al. 2024, "Many-Shot In-Context Learning." Their reported crossover points are UNVERIFIED.
- No study found that compares few-shot LoRA vs ICL specifically for **VLMs with multi-image (screenshot) contexts**, apart from LearnAct's ICL-only sweep. This regime is therefore uncharacterized in the literature.
