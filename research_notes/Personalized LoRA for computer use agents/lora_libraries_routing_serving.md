# LoRA Adapter Libraries: Training, Routing/Composition, Multi-Adapter Serving, and Instant (Hypernetwork) Adapter Generation, with GUI/Agent Applications

Scope note: covers work through Nov 20, 2026. "UNVERIFIED" means the claim comes from the researcher's prior knowledge or a secondary source and was not confirmed against a primary source in this session. All arXiv metadata below was checked on the arXiv abstract page unless marked otherwise.

---

## (a) LoRA libraries, routing, and composition: how is routing done, and does anyone route by skill for agents?

### Takeaway
Routing in the literature comes in three forms. (1) **Zero-shot routing from adapter weights alone.** Arrow (Ostapenko et al. 2024) builds a prototype from each LoRA's top singular vector and routes per token and per layer, with no training. (2) **Few-shot learned composition weights.** LoraHub uses gradient-free search; LoRA Soups/CAT learns per-layer weights; MoLE learns per-layer gates. (3) **Instruction/input-embedding retrieval** (LoraRetriever-style, plus 2025–26 "LoRA-as-tools" agent work). Per-token mixing (Arrow/MoLE) does not fit vLLM, which applies one adapter per request. So for this project the practical design is **request-level routing** (by instruction embedding or user ID), with composition done **offline by concatenating LoRAs into one higher-rank adapter**. Skill-level routing for *LLM* agents appeared in 2026 (Skill-to-LoRA, LatentSkill, Adaptive Minds). None of it is for GUI agents (see (d)).

### Cited Findings

**1. Towards Modular LLMs by Building and Reusing a Library of LoRAs** (the paper the proposal cites; metadata verified)
- Authors: Oleksiy Ostapenko, Zhan Su, Edoardo Maria Ponti, Laurent Charlin, Nicolas Le Roux, Matheus Pereira, Lucas Caccia, Alessandro Sordoni. arXiv:2405.11157, submitted 18 May 2024. — [arXiv](https://arxiv.org/abs/2405.11157)
- Venue: the arXiv page lists no venue or journal-ref. "ICML" appears in the paper's keyword line, and the researcher recalls ICML 2024 (PMLR 235). **Venue UNVERIFIED against the PMLR listing**, so cite it as "ICML 2024" only after checking proceedings.mlr.press. — [arXiv HTML](https://arxiv.org/html/2405.11157)
- Summary: builds a library of LoRAs from multi-task data (256 Flan-v2 tasks, up to 10k examples each; LoRA rank 4 on attention; Phi-2 2.8B and Mistral-7B). It introduces **MBC (model-based clustering)**: train private per-task LoRAs for 40% of the budget, cluster tasks by cosine similarity of their flattened, SVD-reduced LoRA parameters with k-means (K=10), then train one adapter per cluster for the remaining 60%. It also introduces **Arrow**: for each expert and layer, take the SVD of A_i·B_iᵀ and use the first right singular vector as a prototype. Each token's hidden state is dotted with the prototypes, the absolute value is the logit, top-k (k=4) are kept, and a softmax mixes them. Routing is per token and per layer and needs no training data. — [arXiv HTML](https://arxiv.org/html/2405.11157)
- Key numbers, zero-shot average over 10 held-out tasks (PIQA, BoolQ, ARC, HumanEval, MBPP, BBH, …), Phi-2: Base 63.8; Shared multitask LoRA 65.5; FullFT 65.6; Private+uniform 64.8; Private+Arrow 66.6; MBC+uniform 67.1; **MBC+Arrow 67.4** (+3.6 over base, +1.8 over FullFT). Mistral-7B: Base 59.5; Shared 63.1; Private+uniform 61.9; Private+Arrow 64.4; MBC+uniform 64.6; MBC+Arrow 63.9. Arrow nearly matches oracle routing on the training tasks for private libraries. On 12 SuperNI held-out tasks (Rouge-L, Phi-2) at 10% data: No library 53.9; Shared 56.4; MBC+uniform 67.0; MBC+Poly 68.2; LoraHub only 44.7. — [arXiv HTML](https://arxiv.org/html/2405.11157)
- Relevance: (i) Arrow is the canonical "route from adapter weights, no router training" baseline, and it can route a new user's skill adapter immediately after it is added to the library. (ii) Per-token routing is **not supported by vLLM's one-adapter-per-request multi-LoRA** (see (b)), so Arrow would need HF/PEFT inference or a custom kernel. (iii) Note for the paper: on Mistral-7B, uniform averaging of cluster adapters (64.6) slightly beat Arrow (63.9), so simple baselines are competitive.
- UNVERIFIED: the researcher recalls that Hugging Face PEFT added an Arrow-routing implementation (an "ArrowConfig"/LoRA-routing feature) in 2025. If true, it makes Arrow a few-line baseline. Check the PEFT docs.

**2. LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition**
- Authors: Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin. arXiv:2307.13269 (Jul 2023; v3 Aug 2024). COLM 2024 per arXiv/SAIL listings. Code: github.com/sail-sg/lorahub. — [arXiv](https://arxiv.org/abs/2307.13269); [SAIL](https://sail.sea.com/research/publications/36)
- Summary: given a few examples of an unseen task, it finds scalar weights to linearly combine existing LoRAs, gradient-free (black-box optimization), adding no parameters. On Big-Bench Hard it does not beat in-context learning but uses far fewer inference tokens. — [arXiv](https://arxiv.org/abs/2307.13269)
- Key numbers: in Ostapenko et al.'s supervised SNI comparison, LoraHub scored 44.7–45.3 Rouge-L vs 67–79 for MBC-based methods (Phi-2). — [Ostapenko et al.](https://arxiv.org/html/2405.11157)
- Relevance: the "compose a new skill from 3–5 demos without gradients" baseline. It is directly comparable to "train a new LoRA from 3–5 demos" and to in-context demos. Its weakness against ICL on BBH suggests the project should report ICL-with-demos as a baseline.

**3. Mixture of LoRA Experts (MoLE)**
- Authors: Xun Wu, Shaohan Huang, Furu Wei (MSRA / Tsinghua). ICLR 2024; arXiv:2404.13628. — [arXiv](https://arxiv.org/abs/2404.13628); [ICLR poster](https://iclr.cc/virtual/2024/poster/17571)
- Summary: treats each layer of each trained LoRA as an expert and learns a per-layer gating function (softmax with learnable temperature) over the LoRA outputs. All LoRA and base weights stay frozen and only the gates are trained. At inference, unwanted LoRAs can be masked and the weights renormalized. Evaluated in both NLP and **vision-and-language** domains. — [arXiv HTML](https://arxiv.org/html/2404.13628v1)
- Relevance: learned-gate composition of skill LoRAs, e.g. "send email" + "attach file". It needs a small training set for the gate. Masking lets the user exclude adapters, which fits per-user privacy scoping.

**4. LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks**
- Authors: Akshara Prabhakar et al. (Princeton, Microsoft Research). arXiv:2410.13025; COLING 2025 (Industry track, 2025.coling-industry.55). Code: github.com/aksh555/LoRA-Soups. — [arXiv](https://arxiv.org/abs/2410.13025v2); [ACL Anthology](https://aclanthology.org/2025.coling-industry.55)
- Summary: asks whether separately trained skill LoRAs can be merged to solve a task that needs **both** skills (e.g., math + code). The proposed **CAT (learnable concatenation)** learns per-layer weights for each skill LoRA from very few examples. It beats TIES/DARE merging and data mixing on binary skill composition. — [arXiv HTML](https://arxiv.org/html/2410.13025v2)
- Key numbers: average +43% over existing merging methods and +12% over data mixing. These figures come from a **secondary review and are UNVERIFIED against the paper**. Limited to **two-skill** composition. — [Moonlight review](https://www.themoonlight.io/review/lora-soups-merging-loras-for-practical-skill-composition-tasks)
- Relevance: the closest prior work to "compose skill adapters for multi-step tasks" (e.g., "find file then email it"). CAT's output is a weighted sum of LoRAs, which can be re-expressed as **one** LoRA of rank r1+r2 and served by vLLM (see Inferences).

**5. Multimodal routing / mixture-of-LoRA variants (brief)**
- MixLoRA (ACL 2024): conditional mixture of LoRA for **multimodal instruction tuning**. It builds per-input LoRA A/B factors with two independent routers. — [ACL Anthology](https://aclanthology.org/2024.acl-long.38.pdf)
- Task-aware MoILE (arXiv 2506.04595): hierarchical task routing over incremental LoRA experts for embodied continual learning, selecting experts **without task IDs** using cross-modal task clustering. — [arXiv](https://arxiv.org/html/2506.04595)
- HotMoE / "Hybrid Routing for a Mixture of LoRA Experts" (AAAI 2026): layer-wise hierarchical routing for multi-task instruction tuning. No GUI evaluation was seen. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40383)
- LD-MoLE (arXiv 2509.25684): learnable dynamic routing for mixture of LoRA experts. — [arXiv](https://arxiv.org/pdf/2509.25684)

**6. Routing by skill for agents (2025–2026, LLM-only)**
- Adaptive Minds: Empowering Agents with LoRA-as-Tools (arXiv 2510.15416): treats domain-specific LoRAs as tools that a router selects among. Not a GUI agent. — [arXiv](https://arxiv.org/html/2510.15416)
- Skill-to-LoRA and LatentSkill (2026) are per-skill LoRA libraries for LLM agents. They are covered in (d) because they are the closest analogues to this project.
- WeAct (reported Oct 2026 by a blog): independent capability-specific LoRAs on a frozen backbone, chosen by the agent mid-task, for deep-research benchmarks. **UNVERIFIED** (blog only; primary paper not found). — [Ken Ashe blog](https://kenashe.ai/blog/2026-10-20-weact-turns-lora-adapters-into-something-an-agent-can-pick-mid-task)

**7. Related merging / routing works not fetched this session (UNVERIFIED metadata)**
- Task arithmetic: Ilharco et al., "Editing Models with Task Arithmetic," ICLR 2023, arXiv:2212.04089. Task vectors (fine-tuned minus base weights) can be added or negated to compose or remove behaviors. UNVERIFIED in this session.
- TIES-Merging: Yadav et al., NeurIPS 2023, arXiv:2306.01708. UNVERIFIED. DARE is referenced as a baseline in LoRA Soups. — [LoRA Soups](https://arxiv.org/html/2410.13025v2)
- PHATGOOSE: Muqeeth et al., "Learning to Route Among Specialized Experts for Zero-Shot Generalization," ICML 2024. Per-module post-hoc gates trained per expert. UNVERIFIED.
- LoraRetriever: Zhao et al., 2024 (ACL Findings). Retrieves LoRAs by embedding the input against per-LoRA sample embeddings, then mixes or fuses them. UNVERIFIED.

### Inferences
- **The routing design that fits vLLM is request-level.** At each task start (or each step), embed the user instruction and do nearest-neighbor matching against per-adapter "skill descriptions / demo instructions" (LoraRetriever-style), with optional filtering by user ID. Then send the request with `model=<adapter_name>`. Arrow and MoLE need per-token mixing, so they belong in an offline HF/PEFT ablation, not the vLLM serving path.
- **Composition can be served as a single adapter.** Σ_i w_i·B_i·A_i = [w_1B_1 … w_nB_n]·[A_1; …; A_n]. Concatenating k rank-8 skill LoRAs gives one exact rank-8k LoRA, which vLLM serves like any other adapter if `--max-lora-rank` ≥ 8k. This makes CAT/LoraHub-style compositions (weights per adapter, or per layer) servable without kernel changes. Weights that vary per token are not.
- Skill-specificity matters. Skill-to-LoRA found that a wrong or shared adapter **hurts** performance (see (d)). So routing errors have real cost, and the project should report router accuracy and the "wrong-adapter" penalty, plus a fallback to the base model when retrieval confidence is low.
- The Mistral-7B result where uniform averaging beat Arrow suggests a "uniform merge of the top-k retrieved adapters" baseline is cheap and worth including.

### Gaps
- Exact venue page for Ostapenko et al. (ICML 2024 believed, not confirmed on PMLR) and the exact CAT percentage gains (secondary source only).
- The current status of an Arrow implementation in HF PEFT was not verified.
- No paper was found that applies Arrow/LoraHub/MoLE routing to a **VLM GUI agent**.

---

## (b) Multi-adapter serving: S-LoRA, Punica, dLoRA, LoRAX, vLLM (including Qwen2.5-VL and vision-tower LoRA), and hot-swap latency/memory

### Takeaway
Multi-LoRA serving is a solved systems problem for the LLM part. S-LoRA, Punica, and dLoRA showed thousands of adapters per GPU at roughly 2 ms/token overhead, and vLLM inherits these ideas. **vLLM serves Qwen2.5-VL with LoRA on the language model.** LoRA on the **vision tower and connector** became available as an *experimental* opt-in (`--enable-tower-connector-lora`, initial PR #26674, "Qwen VL series" listed). An open bugfix PR reports intermittent CUDA illegal-memory-access errors on Qwen2/2.5/3-VL with that flag on v0.34.0. **Recommendation: train LLM-only LoRAs (vision tower frozen)** so serving stays on the stable path. A cold load of a rank-8 adapter is tens of milliseconds at the systems level. Through vLLM's HTTP load endpoint it is plausibly sub-second, but no vLLM-specific benchmark was found.

### Cited Findings

**1. S-LoRA: Serving Thousands of Concurrent LoRA Adapters**
- Authors: Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica. arXiv:2311.03285 (Nov 2023; v3 Jun 2024). The arXiv page lists no venue. The researcher recalls **MLSys 2024 (UNVERIFIED in this session)**. — [arXiv](https://arxiv.org/abs/2311.03285)
- Summary: keeps all adapters in host memory and moves only active ones to the GPU. **Unified Paging** pools adapter weights of varying ranks together with KV cache to reduce fragmentation. Custom CUDA kernels batch heterogeneous LoRA computation, and a new tensor-parallel strategy is used. — [arXiv](https://arxiv.org/abs/2311.03285)
- Key numbers: up to **4× throughput** vs HF PEFT and vLLM with naive LoRA support. Serves "thousands" of adapters on one or more GPUs with small overhead. — [arXiv](https://arxiv.org/abs/2311.03285)
- Relevance: justifies a "library of thousands of per-user/per-skill adapters on one GPU" at the systems level. Most adapters sit in CPU RAM, which maps onto vLLM's `max_cpu_loras`.

**2. Punica: Multi-Tenant LoRA Serving**
- Authors: Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo, Luis Ceze, Arvind Krishnamurthy (UW, Duke). arXiv:2310.18547 (Oct 2023). The arXiv page lists no venue. The researcher recalls **MLSys 2024 (UNVERIFIED)**. — [arXiv](https://arxiv.org/abs/2310.18547)
- Summary: a new CUDA kernel batches GPU operations for **different** LoRA models in one batch while keeping a single copy of the base model. This is the SGMV kernel. The name is from the paper body and the researcher's knowledge; the abstract does not name it. — [arXiv](https://arxiv.org/abs/2310.18547)
- Key numbers: **12× higher throughput** than state-of-the-art serving systems on a fixed GPU cluster. Only **+2 ms latency per token**. — [arXiv](https://arxiv.org/abs/2310.18547)
- Relevance: the per-token overhead number to cite when arguing per-user adapters are "free" at inference. UNVERIFIED: vLLM's multi-LoRA layer uses Punica-derived kernels (the code module is named `punica_wrapper`).

**3. dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving**
- Authors: Bingyang Wu et al. (Peking University / Shanghai AI Lab; Xin Jin among the authors). OSDI 2024, pp. 911–927. — [USENIX](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
- Summary: within a replica, it dynamically **merges** an adapter into the base weights when one adapter dominates and **unmerges** for mixed batches, using credit-based batching. Across replicas, it migrates requests and adapters together to balance load. Built on vLLM; evaluated with LLaMA-2 7B/13B/70B, 128 LoRAs, and ShareGPT on 4×8 A800-80GB. — [USENIX](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang); [slides](https://usenix.org/system/files/osdi24_slides-wu-bingyang.pdf)
- Key numbers: up to **57.9×** throughput vs vLLM and **26.0×** vs HF PEFT. Up to **1.8× lower average latency** than S-LoRA. — [USENIX](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
- Relevance: per-user traffic is skewed (one user and one skill at a time), and merged mode removes LoRA overhead entirely. The project probably does not need this, but it is the citation for "merge when single-tenant."

**4. CaraServe / cold-start mitigations (hot-swap latency)**
- CaraServe (arXiv:2401.11240): "Depending on the adapter size, a single cold-start can take tens of milliseconds." A cold start hurts time-to-first-token of the new request and, under continuous batching, the decoding of in-flight requests. CaraServe prefills on the CPU while the adapter loads to the GPU. It reports up to 1.4× lower average latency and up to 99% SLO attainment. — [arXiv](https://www.arxiv.org/abs/2401.11240)
- Predictive-LoRA (arXiv 2512.20210) cuts cold-start latency by up to 68% via prefetching. TOPPINGS (ATC 2025) also uses CPU-assisted prefill during loading. — [Predictive-LoRA](https://arxiv.org/html/2512.20210v1); [TOPPINGS](https://cse.hkust.edu.hk/~weiwa/papers/toppings-atc25.pdf)
- Diffusers data point (different stack, SDXL, ~24 MB LoRA): adapter load 1.4–3.5 s, unload 0.13–0.52 s, swap ≈ 3 s. A shared warm base cut first-response time from ~35 s to ~13 s. Hundreds of LoRAs were served on fewer than 5 A10Gs. This is Python-level `load_lora_weights` and is **not representative of vLLM**, which preallocates LoRA slots. — [HF blog, Dec 2023](https://huggingface.co/blog/lora-adapters-dynamic-loading)

**5. VaLoRA — "Empower Vision Applications with LoRA LMM"** (the multimodal multi-LoRA serving system)
- Authors: Liang Mi, Weijun Wang, … Yunxin Liu (13 authors). arXiv:2411.00915. **EuroSys 2025**. — [arXiv](https://arxiv.org/abs/2411.00915)
- Summary: an end-to-end system for serving LoRA-adapted **large multimodal models**. It has accuracy-aware adapter generation, an "adaptive-tiling" operator that batches heterogeneous LoRAs, and request/adapter orchestration. Evaluated on 5 vision tasks across 3 LMMs. — [arXiv](https://arxiv.org/abs/2411.00915)
- Key numbers: **24–62% accuracy** gain over the original LMMs and **20–89% latency reduction** vs state-of-the-art LoRA serving systems. — [arXiv](https://arxiv.org/abs/2411.00915)
- Relevance: the closest systems precedent for "many task LoRAs on a VLM." Cite it in related work for the serving side.

**6. vLLM multi-LoRA (the project's serving stack)**
- Serve with `--enable-lora --lora-modules name=path`. Other settings are `--max-loras` (adapters co-resident in a GPU batch), `--max-lora-rank` (set it to the largest rank used; larger wastes memory), and `--max-cpu-loras` (host-side cache). Adapter requests batch alongside base-model requests and alongside each other when `max_loras` is large enough. — [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/)
- **Runtime hot-loading:** set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, then POST to `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`. `load_inplace: true` replaces a same-named adapter, which suits "retrain a user's adapter and swap it." The docs warn this is a security risk outside trusted environments. **LoRAResolver plugins** load adapters on demand when a request names an unknown model: `lora_filesystem_resolver` for a local directory and `lora_hf_hub_resolver`, configured via `VLLM_PLUGINS`, `VLLM_LORA_RESOLVER_CACHE_DIR`, and `VLLM_LORA_RESOLVER_HF_REPO_LIST`. — [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/)
- **Multimodal:** "vLLM experimentally supports LoRA for the Tower and Connector components of multi-modal models." Each model needs token-helper functions implemented. `--default-mm-loras` maps a modality to an adapter applied automatically (Granite Speech, Phi-4-multimodal); only one LoRA applies per prompt. — [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/)
- **Qwen2.5-VL status:** the latest supported-models page lists Qwen2.5-VL with LoRA support. — [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/). A 2025 issue documents the earlier behavior: LoRA applied only to the language model, and vision-layer adapter weights were ignored with a warning. — [vLLM issue #23796](https://github.com/vllm-project/vllm/issues/23796)
- **Tower/connector LoRA:** foundational PR #26674 ("Initialize LoRA support for tower and connector in multi-modal models") lists "Qwen VL series and idefics3." Tracking issue #31479 (opened Dec 29, 2025; closed Sep 2, 2026) says "most models' towers already support LoRA." The flag is `--enable-tower-connector-lora`. Qwen2.5-VL is not named explicitly, only "Qwen VL series." — [vLLM issue #31479](https://github.com/vllm-project/vllm/issues/31479)
- **Known bug:** PR #68754 fixes an intermittent CUDA illegal memory access with `--enable-tower-connector-lora` that affects Qwen2/2.5/3-VL and reproduces on **v0.34.0 with Qwen2.5-VL-3B**. — [vLLM PR #68754](https://github.com/vllm-project/vllm/pull/68754)
- Unsloth's docs still say vLLM does not support vision/encoder-layer LoRA and recommend `finetune_vision_layers=False` when serving with vLLM. **This conflicts with the vLLM docs above** and is likely stale, but the advice is still the safe default. — [Unsloth VLM RL docs](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl)

**7. LoRAX (Predibase)** — **UNVERIFIED in this session** (no fetch)
- An open-source (Apache-2.0) multi-LoRA inference server forked from HF TGI. It loads adapters dynamically from HF Hub, local disk, or S3 per request, does heterogeneous continuous batching with Punica/SGMV kernels, and has an adapter-exchange scheduler with tiered GPU/CPU/disk caching. The marketing line is "thousands of fine-tuned models on one GPU." Multimodal/VLM LoRA support in LoRAX is unknown. Use vLLM for Qwen2.5-VL.

### Inferences
- **Adapter size (computed, not sourced).** Qwen2.5-VL-7B's LLM has hidden size 3584, 28 layers, MLP 18944, and KV dim 512 (GQA, 4 KV heads). A rank-8 LoRA on all linear LLM layers (q,k,v,o,gate,up,down) is about 0.72M params/layer × 28 ≈ **20M params ≈ 40 MB in bf16**. Attention-only is about 5M params ≈ 10 MB. A 1,000-adapter library is about 10–40 GB on disk or in CPU RAM, which is trivial on a Modal volume. Config values are from the researcher's knowledge of the Qwen2.5-7B config (UNVERIFIED this session).
- **Hot-swap cost in vLLM:** loading 10–40 MB from local NVMe or a Modal volume into a preallocated slot should take tens to hundreds of ms, consistent with CaraServe's "tens of ms" systems-level figure. Steady-state overhead is about +2 ms/token (Punica). For GUI agents that emit roughly 20–100 tokens per step, both are negligible next to VLM prefill of a 1–3k-token screenshot. No vLLM-specific swap benchmark was found, so **measure it** (a good small systems table for the paper).
- **Modal deployment pattern:** a single vLLM container with `--enable-lora --max-loras 4–8 --max-lora-rank 16 --max-cpu-loras 64` and `lora_filesystem_resolver` pointed at a Modal Volume of adapters. A new user adapter becomes servable as soon as its files land in the volume, with no server restart.
- Since UI-TARS-1.5-7B is tagged `qwen2_5_vl` on HF (see Practical), the same vLLM multi-LoRA path should apply to it (UNVERIFIED that vLLM loads UI-TARS LoRAs without issues).

### Gaps
- No primary benchmark of vLLM's `/v1/load_lora_adapter` latency or of multi-LoRA overhead for Qwen2.5-VL specifically.
- The exact vLLM release that first shipped stable tower/connector LoRA for Qwen2.5-VL, and whether PR #68754 has merged, were not determined.
- MLSys 2024 venue for S-LoRA and Punica not confirmed (arXiv pages list none). LoRAX details not fetched.

---

## (c) Hypernetwork / instant adapter generation: can one generate a LoRA from a few GUI demos in seconds, and how does quality compare?

### Takeaway
Instant LoRA generation is real for **text LLMs**. Text-to-LoRA (Sakana, ICML 2025) works from a task description. Drag-and-Drop LLMs (NeurIPS 2025) works from a batch of unlabeled prompts and reports beating trained LoRAs on unseen benchmarks. Doc-to-LoRA (Sakana, Feb 2026) works from a document or context. HyperLoRA (EMNLP Findings 2024) is the closest "**demonstrations → LoRA**" method. LatentSkill (2026) maps textual agent skills to LoRAs. **No hypernetwork was found that generates a LoRA for a GUI/computer-use VLM from screenshot demonstrations.** All of these need an expensive meta-training stage over a large pool of tasks or adapters, which a 2-week project cannot produce for a 7B VLM. Treat hypernetworks as related work and future work, not a baseline to build.

### Cited Findings

**1. Text-to-LoRA: Instant Transformer Adaption (T2L)**
- Authors: Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange (Sakana AI). arXiv:2506.06105 (Jun 6, 2025; v2 Jun 9). **ICML 2025 (PMLR 267)**. Code: github.com/SakanaAI/text-to-lora. — [arXiv](https://arxiv.org/abs/2506.06105); [PMLR](https://proceedings.mlr.press/v267/charakorn25a.html)
- Summary: a hypernetwork that outputs a LoRA in **one forward pass** from a natural-language task description. Trained to reconstruct 9 pre-trained benchmark LoRAs (GSM8K, ARC, …), its generated adapters **match task-specific adapters** on those test sets. It can compress hundreds of LoRAs and generalizes zero-shot to unseen tasks. — [arXiv](https://arxiv.org/abs/2506.06105); [ICML poster](https://icml.cc/virtual/2025/poster/43471)
- Key numbers (secondary, **UNVERIFIED against the paper**): on Mistral-7B-Instruct, average 73.9% with T2L vs 60.0% with no adapter and 66.3% with a multi-task LoRA. — [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/text-to-lora-generates-task-specific-lora-adapters-directly-from-natural-language-descriptions)
- Relevance: "describe the skill → get an adapter" is the conceptual competitor to "record 3–5 demos → train an adapter." But T2L conditions on **text descriptions**, not demonstrations, and on text-only LLMs.

**2. Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights (DnD)**
- Authors: Zhiyuan Liang + 13 co-authors. arXiv:2506.16406 (Jun 19, 2025). **NeurIPS 2025** main conference. Project page: jerryliang24.github.io/DnD. — [arXiv](https://arxiv.org/abs/2506.16406); [NeurIPS](https://neurips.cc/virtual/2025/poster/116841)
- Summary: a text encoder compresses a batch of **unlabeled task prompts** into condition embeddings, and a "hyper-convolutional decoder" turns them into full LoRA matrices. There is no per-task training. — [arXiv](https://arxiv.org/abs/2506.16406)
- Key numbers: up to **12,000× lower overhead** than full fine-tuning. Average gains **up to 30% over the strongest trained LoRAs** on unseen benchmarks. — [arXiv](https://arxiv.org/abs/2506.16406). UNVERIFIED: the benchmark families include common-sense reasoning, math, coding, and a multimodal benchmark, base models are Qwen2.5 0.5B–7B, and the claim is adapters generated "in seconds."
- Relevance: the strongest published claim that generated adapters can **beat** trained LoRAs. But "trained LoRAs" here means LoRAs trained on *other* (source) tasks and applied to the unseen task, not LoRAs trained on the target task's data. That distinction matters when comparing it with per-user trained adapters.

**3. Doc-to-LoRA: Learning to Instantly Internalize Contexts (D2L)**
- Authors: Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, Robert Tjarko Lange (Sakana AI). arXiv:2602.15902 (Feb 13, 2026). Preprint. — [arXiv](https://arxiv.org/abs/2602.15902); [Sakana blog](https://sakana.ai/doc-to-lora/)
- Summary: a hypernetwork meta-learned to approximate **context distillation** in one forward pass. It turns an unseen prompt or document into a LoRA so later queries do not need the context, which cuts latency and KV memory. — [arXiv](https://arxiv.org/abs/2602.15902)
- Key numbers: near-perfect zero-shot needle-in-a-haystack accuracy at more than 4× the native context length. It beats standard context distillation on real QA with lower peak memory and update latency. Exact latencies are not on the abstract page. The authors pitch "personalized chat" as a use case. — [arXiv](https://arxiv.org/abs/2602.15902)
- Relevance: conceptually the closest to "demos → adapter": treat the 3–5 demo trajectories as a "document" to internalize. Text-only, though.

**4. HyperLoRA: Efficient Cross-task Generalization via Constraint-Guided … (EMNLP Findings 2024)**
- Takes the **task instruction plus k-shot demonstrations** as input and generates LoRA parameters, using the few-shot examples as task information. — [ACL Anthology PDF](https://aclanthology.org/2024.findings-emnlp.956.pdf). Full title and author list **UNVERIFIED** (not fetched).
- Relevance: the only found method that explicitly conditions adapter generation on **demonstrations**. Text-only.

**5. LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents**
- Authors: Aofan Yu, Chenyu Zhou, Tianyi Xu, Zihan Guo, Rong Shan, Zhihui Fu, Jun Wang, Weiwen Liu, Yong Yu, Weinan Zhang, Jianghao Lin. arXiv:2606.06087 (Jun 4, 2026; v3 Aug 26, 2026). Preprint. — [arXiv](https://arxiv.org/abs/2606.06087)
- Summary: a pretrained hypernetwork converts textual agent **skills** into plug-and-play LoRAs, cached per skill ("compiled once"). It supports scaling via the LoRA coefficient and **composition via parameter-space arithmetic**. — [arXiv](https://arxiv.org/abs/2606.06087)
- Key numbers: ALFWorld success **+21.4 (seen) / +13.4 (unseen)** points over in-context skills with **63.9% fewer prefill tokens**. Search-QA EM +3.0 with 71.8% fewer tokens per step. It is not compared with gradient-trained per-skill LoRAs. — [arXiv](https://arxiv.org/abs/2606.06087)
- Relevance: the closest **agent** analogue of hypernetwork-generated skill adapters, with composition. Not GUI or multimodal.

**6. Other hypernetwork-LoRA work (brief)**
- HyperTuning and HINT generate PEFT parameters from demonstrations or instructions. "Generative Adapter" (Chen et al., 2025) trains a hypernetwork with next-token loss. These are mentioned in 2026 related-work sections seen via search; primary papers not fetched (**UNVERIFIED** details). — [arXiv 2610.01934 (citing context)](https://arxiv.org/pdf/2610.01934)
- HyLoVQA (arXiv 2605.22035): hypernetwork-generated LoRA for continual **VQA**, a rare **multimodal** hypernetwork-LoRA. Details not fetched. — [arXiv](https://arxiv.org/pdf/2605.22035)

### Inferences
- **Could any of these produce a GUI-skill LoRA from 3–5 demos in seconds today? No, not off the shelf.** (1) All are trained for text LLMs (Mistral/Llama/Gemma/Qwen2.5 text). None targets Qwen2.5-VL or UI-TARS. (2) Meta-training needs a large corpus of (task description or demos, LoRA) pairs. For GUI that would mean hundreds of per-skill GUI LoRAs, which is the very library the project is building. (3) The output would need to cover action/coordinate grounding, which text hypernetworks have never been shown to do.
- **Quality vs trained LoRAs:** T2L *matches* task-specific trained LoRAs on the tasks it was trained to reconstruct and is weaker zero-shot on unseen tasks (secondary numbers). DnD claims to beat "trained LoRAs" only in a cross-task transfer sense. No paper shows a hypernetwork-generated adapter beating a LoRA trained on the **same target task's** data.
- **Framing for the paper:** "gradient-based per-skill LoRA (minutes) vs in-context demos (zero training) vs hypernetwork (seconds, needs meta-training)." The per-skill library the project builds is exactly the training data a future GUI Text-to-LoRA/Demo-to-LoRA would need. That is a good future-work hook.

### Gaps
- Exact T2L numbers and DnD per-benchmark numbers were not read from the papers (abstracts and secondary sources only).
- No source found on generation latency in seconds for T2L/D2L on a specific GPU.
- No multimodal or GUI demonstration-to-LoRA generator found as of Nov 2026.

---

## (d) Applied to agents / GUI / multimodal models: has anyone built a per-skill LoRA library for GUI/computer-use agents or composed skill adapters?

### Takeaway
**FLAG — closest prior work, all 2026:**
- **Skill-to-LoRA** (arXiv 2606.16769, Jun 2026) builds a **per-skill LoRA library for LLM agents**: one adapter per skill, loaded at runtime. Wrong or shared adapters hurt.
- **LatentSkill** (arXiv 2606.06087) generates per-skill LoRAs with a hypernetwork and composes them.
- **CoAdapt-GUI** (arXiv 2608.11588, Aug 2026) adapts a **LoRA on a frozen VLM per unseen mobile app** via test-time RL. It uses no demos and is a single adapter, not a library.

**No paper was found (through Nov 2026) that builds a per-skill or per-user LoRA library for a GUI/computer-use VLM from user demonstrations, routes among the adapters, or composes skill adapters for multi-step GUI tasks.** The project's novelty claim holds, but it must be positioned against Skill-to-LoRA (text agents), CoAdapt-GUI (per-app LoRA, RL), and LearnAct (in-context demos for GUI). Note that Skill-to-LoRA's own stated limitation ("one skill per task") is exactly the composition gap.

### Cited Findings

**1. Skill-to-LoRA: From Using Skills to Learning Behaviors for Token-Efficient LLM Agents** — closest conceptual match
- Authors: Tianyi Zhang, Zhonghao Qi. arXiv:2606.16769 (Jun 15, 2026). Preprint; code "upon acceptance." — [arXiv](https://arxiv.org/abs/2606.16769)
- Summary: offline, uses each SKILL.md to generate skill-guided demonstrations and trains **one LoRA per skill**, forming an adapter library. Online, it drops the skill text and loads the matching adapter. It models "the behavioral change induced by the skill text" instead of compressing the document. It assumes **one skill per task** and activates a single adapter, so multi-skill composition is explicitly left open. Base model: Qwen3.6-27B. Benchmark: a 21-skill subset of SWE-Skills-Bench. — [arXiv](https://arxiv.org/abs/2606.16769); [arXiv HTML](https://arxiv.org/html/2606.16769v1)
- Key numbers: pass rate **+2.9 pp vs no-skill** and **+5.2 pp vs full skill text**. **−6.6%** per-step tokens vs full skill text. Matches or beats skill-text prompting on 18/21 skills and no-skill on 15/21. **A wrong adapter or a shared adapter reduces performance.** Adapters are about 24 MB each (per search summary; size and training time not confirmed in the abstract). How the adapter is selected at inference is not described in the abstract. — [arXiv](https://arxiv.org/abs/2606.16769)
- Relevance: **must-cite.** Same architecture as this project (per-skill LoRA library plus runtime loading) but for text-only SWE agents. The gains over prompting are modest, so the project should expect modest gains too and report per-skill wins and losses.

**2. CoAdapt-GUI: Joint Workflow Context and Policy Adaptation for Unseen GUI Applications** — closest GUI match
- Authors: Linqiang Guo, Li Gu, Zihuan Jiang, Zhixiang Chi, Siobhan Reid, Ziqiang Wang, Yuanhao Yu, Wei Liu, Yang Wang, Tse-Hsun (Peter) Chen. arXiv:2608.11588 (Aug 12, 2026). Preprint. — [arXiv](https://arxiv.org/abs/2608.11588)
- Summary: test-time adaptation for mobile GUI agents on **apps unseen in training**, with **no target-app demonstrations** and a limited interaction budget. It jointly updates (i) a structured **workflow context** (transferable procedures, failure modes, verification rules) and (ii) a **LoRA adapter on a frozen VLM**, using task-context-matched group-relative (GRPO-style) optimization on its own rollouts and rewards. — [arXiv](https://arxiv.org/abs/2608.11588)
- Key numbers: AndroidWorld-Generalization **45.0% vs 37.5%** (policy-only TTA). AndroidWorld Plus **38.6% → 52.9%**. The authors say **the workflow context provides most of the gain** and joint LoRA adaptation adds more on held-out tasks. Base VLM, LoRA rank, and compute are not stated in the abstract. — [arXiv](https://arxiv.org/abs/2608.11588)
- Relevance: shows per-app LoRA adaptation of a GUI VLM is viable. It also warns that **textual/workflow memory may beat weight updates**, so the project should include a "demos-in-context / workflow-memory" baseline.

**3. LatentSkill** (see (c)): hypernetwork-generated per-skill LoRAs for LLM agents with a cache and parameter-arithmetic composition. ALFWorld +21.4/+13.4 over in-context skills. Not GUI. — [arXiv](https://arxiv.org/abs/2606.06087)

**4. Towards Omni-dimensional GUI Agent Navigation with Masked Trajectory Prediction (MaP)**
- Authors: Yan Zhang, Pei Fu, Daiqing Wu, Huawen Shen, Ruoceng Zhang, Shaojie Zhang, Jiahui Yang, Yu Zhou, Can Ma, Zhenbo Luo, Jian Luan. arXiv:2609.25769 (Sep 22, 2026). **EMNLP 2026**. — [arXiv](https://arxiv.org/abs/2609.25769)
- Summary: unifies GUI navigation sub-objectives as masked trajectory prediction. A **role-aware adapter module routes each token** to a specialized representation space to handle heterogeneous data. It outperforms direct mixture training on 5 GUI navigation benchmarks. Per a search summary, it uses multiple specialized adapters with a token-wise router. The abstract does not say "LoRA." — [arXiv](https://arxiv.org/abs/2609.25769)
- Relevance: an instance of **mixture-of-adapters inside a GUI agent**, but routing is by token role (e.g., state vs action), not by skill or user.

**5. UniCompress: From Multi-LoRA Experts to One Compact Model for Edge Intelligence**
- Authors: Mingkang Liu, Peipei Jiao, Weiming Meng, Tianyuan Liu, Lichen Pang, Junyan Li. arXiv:2611.06075 (Nov 4, 2026). Preprint. — [arXiv](https://arxiv.org/abs/2611.06075)
- Summary: consolidates **four task-specific LoRA experts on Qwen3-VL-4B**, including one for **mobile-GUI action prediction**, into a single BF16 teacher, then into an INT2 student (GPTQ init plus teacher-guided recovery, quantizing both vision encoder and LLM). — [arXiv](https://arxiv.org/abs/2611.06075)
- Key numbers: 8.9 GB → 2.1 GB (−76%). +34.3 pp macro score over GPTQ-INT2. Within 0.9 pp of the BF16 teacher. No per-expert or routing comparison on the abstract page. — [arXiv](https://arxiv.org/abs/2611.06075)
- Relevance: evidence that a mobile-GUI LoRA expert can sit alongside other task LoRAs on one VLM. Its direction (merge everything into one model) is the opposite of the project's (keep a routed library).

**6. LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark** — key *in-context demos* baseline
- Authors: Guangyi Liu, Pengxiang Zhao, Liang Liu, Zhiming Chen, Yuxiang Chai, Shuai Ren, Hao Wang, Shibo He, Wenchao Meng. arXiv:2504.13805 (Apr 2025). Also appears as **ACL 2026 Findings** (2026.findings-acl.1491). — [arXiv HTML](https://arxiv.org/html/2504.13805); [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491.pdf)
- Summary: the LearnGUI benchmark (2,252 offline and 101 online tasks, with k = 1–3 support demonstrations) and a multi-agent framework that uses **retrieved human demonstrations in context**. Open models (Qwen2-VL-7B, UI-TARS-7B-SFT) were LoRA-fine-tuned to use demonstrations: rank 64, alpha 128, all modules, **vision encoder frozen**, lr 1e-5, 1 epoch on 2,001 tasks per k-shot, 8× L40S. — [arXiv HTML](https://arxiv.org/html/2504.13805)
- Key numbers (offline action-match average): UI-TARS-7B-SFT 77.5 (0-shot) → **82.8** (1-shot) / 81.9 / 82.1. Qwen2-VL-7B 71.8 → 77.3 / 78.5 / **79.4**. Gemini-1.5-Pro 19.3 → 51.7 / 55.6 / 57.7. — [arXiv HTML](https://arxiv.org/html/2504.13805)
- A search snippet mentioned a "Demo-specific LoRA" row in a results table, but the fetched portion of the paper did not confirm a per-demo LoRA comparison. **UNVERIFIED.** Check the ACL 2026 version's tables, since that would be the most direct prior comparison of per-task LoRA vs in-context demos on GUI. — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1491.pdf)
- Relevance: the natural **baseline and benchmark** for "learn a skill from 1–5 demos." Its LoRA recipe (vision frozen) also matches the vLLM-friendly design.

**7. Other demonstration-learning GUI agents (non-LoRA; for positioning)**
- ShowUI-Aloha: Human-Taught GUI Agent (Zhang, Guo, Goh, Hu, Chen, Wang, Gao, Shou; arXiv:2601.07181, Jan 2026). A recorder → learner (captions) → planner → executor pipeline over human screen recordings. The planner reads parsed demonstrations, so it appears to be in-context or memory-based rather than weight-based (inference; no numbers on the abstract page). — [arXiv](https://arxiv.org/abs/2601.07181)
- AdaptAgent (arXiv 2411.13451; ACL 2025): adapts multimodal web agents with few-shot human demonstrations, with meta-learning before few-shot adaptation. — [arXiv](https://arxiv.org/html/2411.13451); [ACL](https://aclanthology.org/2025.acl-long.1008.pdf)
- ALLOY (arXiv 2510.10049): generates reusable agent workflows from user demonstration. It names continual fine-tuning on user demos as a promising direction. — [arXiv](https://arxiv.org/pdf/2510.10049)
- CUA-Skill: a large engineered skill library for Windows computer-use agents with dynamic skill retrieval. Skills are code or prompts, not LoRAs (per search summary). — [HF papers search](https://huggingface.co/papers?q=skill-use+agents)

### Inferences
- **Novelty statement the project can defend:** "To our knowledge, the first per-skill/per-user LoRA library for a GUI VLM agent trained from a handful of user demonstrations, with instruction-based routing and multi-LoRA serving." Cite Skill-to-LoRA as the text-agent analogue, CoAdapt-GUI as per-app RL adaptation, and LearnAct as in-context demo learning. Hedge with "to our knowledge"; the search was broad but not exhaustive.
- **Composition is open for GUI.** Skill-to-LoRA explicitly assumes one skill per task, and LoRA Soups only does two-skill text composition. A small experiment composing two GUI skill adapters (e.g., "open file manager → find file" + "compose email → attach") with uniform or CAT weights, served as a concatenated rank-16 LoRA, would be a genuine contribution even if it is negative.
- **Expect baseline pressure.** CoAdapt-GUI found the workflow context gave most of the gain, and LearnAct shows 1-shot in-context demos add +5 to +32 points. The project's main comparison should be per-skill LoRA vs the same 3–5 demos in context (and vs a single shared LoRA trained on all skills, Ostapenko's "Shared" baseline).

### Gaps
- Whether LearnAct's ACL 2026 version contains a "demo-specific LoRA" result (snippet suggests yes; unconfirmed).
- CoAdapt-GUI's base VLM, LoRA rank, and adaptation compute; MaP's adapter type; UniCompress per-expert scores. Abstract pages only.
- The WeAct primary paper was not found.
- Searches were limited to arXiv/web. OpenReview submissions for ICLR 2027 (under review in Nov 2026) were not searched and may contain closer work.

---

## Practical: LoRA SFT time for Qwen2.5-VL-7B on ~50–200 multimodal samples, supported frameworks, and feasibility of "live" (minutes-scale) personalization

### Takeaway
No published wall-clock benchmark matches this setting (Qwen2.5-VL-7B, LoRA, single A100/H100, 50–200 GUI samples). A back-of-envelope estimate gives **about 2–10 min of pure training for 50 samples and about 10–45 min for 200 samples × 3 epochs on one A100**, and roughly 2–3× faster on an H100, plus about 1–3 min of model load or container cold start. "Live" minutes-scale personalization is therefore **feasible at the 3–5-demo scale (tens of samples), especially on an H100 with reduced screenshot resolution**. LLaMA-Factory, ms-swift, and Unsloth all support Qwen2.5-VL LoRA with a frozen-vision-tower option. UI-TARS-1.5-7B uses the Qwen2.5-VL architecture, so the same tooling should apply.

### Cited Findings
- **GUI-specific recipe precedent:** a 2026 paper fine-tunes Qwen2.5-VL-7B with **LoRA r=8, α=16** on AITW, Mind2Web, AndroidControl, and GUI-Odyssey on **4×A100-80GB** with DeepSpeed ZeRO-2. No wall-clock time reported. — [arXiv 2605.29447](https://arxiv.org/pdf/2605.29447)
- **Generic timing prior (informal):** about 2–4 h for 1,000 images × 3 epochs of Qwen2.5-VL LoRA on one A100 (blog estimate, no methodology). — [Medium guide](https://medium.com/@sohan.lal_54278/step-by-step-guide-fine-tuning-qwen-2-5-vl-with-your-own-data-78f2ca5fe661). Another guide gives 4–12 h on a 24 GB GPU with a different pipeline. — [Labellerr](https://www.labellerr.com/blog/fine-tune-qwen-2-5-vl/)
- **LearnAct recipe:** LoRA r=64 on all modules, vision frozen, 1 epoch on about 2k tasks, 8× L40S (no time reported). — [arXiv HTML](https://arxiv.org/html/2504.13805)
- **LLaMA-Factory:** supports LoRA on Qwen2-VL/Qwen2.5-VL with `freeze_vision_tower`, `freeze_multi_modal_projector`, and `freeze_language_model` flags. The test suite asserts that visual LoRA params are trainable only when the tower is unfrozen (seen in an unofficial mirror of the repo's test file). — [HF mirror of test_visual.py](https://huggingface.co/Xin-Rui/LLAMA-Fac-NEW-A800/blob/main/tests/model/model_utils/test_visual.py). Papers report using LLaMA-Factory for Qwen2.5-VL. — [arXiv 2506.07227](https://arxiv.org/pdf/2506.07227)
- **ms-swift:** Qwen2.5-VL LoRA examples use `--freeze_llm false --freeze_vit true --freeze_aligner true --lora_rank 8 --lora_alpha 32 --target_modules all-linear`, with CPT/SFT/DPO/KTO/RM support. These are Megatron-SWIFT docs; flags for the standard `swift sft` path were not confirmed. — [ms-swift docs](https://swift.readthedocs.io/en/v4.4/Megatron-SWIFT/Multimodal-Model.html)
- **Unsloth:** supports Qwen2.5-VL vision fine-tuning with a `finetune_vision_layers` toggle and `use_gradient_checkpointing="unsloth"`. Vendor claims for VLM RL: 1.5–2× faster and 90% less VRAM than FA2 setups, and Qwen2.5-VL-7B GRPO fits on a free Colab T4. — [Unsloth docs](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl); [Unsloth blog](https://unsloth.ai/blog/vision-rl)
- **NVIDIA NeMo Megatron Bridge** supports fine-tuning Qwen2.5-VL 3B/7B/32B/72B on single- and multi-image data. — [NeMo docs](https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/models/qwen/qwen2.5-vl.html)
- **TRL** (`SFTTrainer` with VLM collators) and **MLX-VLM** (LoRA fine-tuning of Qwen2-VL/2.5-VL on Apple Silicon): support believed present from the researcher's knowledge, **UNVERIFIED in this session**.
- **UI-TARS-1.5-7B:** the HF card is tagged `qwen2_5_vl`, Apache-2.0, and documents vLLM and SGLang serving (OpenAI-compatible, image inputs). 7B scores: **OSWorld 27.5, ScreenSpot-Pro 49.6**. The full UI-TARS-1.5 scores OSWorld 42.5 and AndroidWorld 64.2. The card does not name the base model explicitly. — [HF model card](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B)
- **Visual-token cost matters:** several 2026 papers focus on pruning visual tokens for high-resolution GUI agents, which shows screenshot tokens dominate compute. — [arXiv 2602.23235](https://arxiv.org/pdf/2602.23235); [arXiv 2603.26041](https://arxiv.org/pdf/2603.26041)

### Inferences
- **Back-of-envelope training time (researcher's computation, not sourced).** Assumptions: one GUI step sample ≈ one 1080p screenshot ≈ 2.6k visual tokens at Qwen2.5-VL native resolution, so about 3k tokens per sample. LoRA with frozen base and gradient checkpointing ≈ 6·N FLOPs/token with N ≈ 7.6B. 200 samples × 3 epochs ≈ 1.8M tokens ≈ 8×10¹⁶ FLOPs (the frozen ViT forward adds about 10–30%). At an effective 100 TFLOP/s on an A100 (about 30% MFU, batch 1), that is **about 15 min**; on an H100 at about 300 TFLOP/s, **about 5 min**. 50 samples is about 4 min on an A100 and about 1.5 min on an H100. Capping `max_pixels` near 1280×28×28 (about 1.3k visual tokens) roughly halves this. The informal "2–4 h per 1,000 images × 3 epochs" blog figure scales to about 25–50 min for 200 samples, an upper bound for an unoptimized pipeline.
- **Sample count:** 3–5 demonstrations × about 5–15 steps each ≈ 15–75 step-level samples per skill. The project is at the **low end (≈ minutes)**, so "record demos → adapter served within about 5–10 min" is a realistic live-demo claim on a Modal H100, including cold start. Pre-warm the container and keep base weights on a Modal Volume.
- **Recommended config for vLLM compatibility:** LoRA r=8 on the LLM (all-linear or attention+MLP), vision tower and merger frozen (LLaMA-Factory `freeze_vision_tower=true`, ms-swift `--freeze_vit true --freeze_aligner true`, Unsloth `finetune_vision_layers=False`). Export in PEFT format and serve via `--enable-lora`. This avoids the experimental tower/connector path and its open bug on Qwen2/2.5/3-VL.
- With so few samples, overfitting is the main risk. Use a few epochs, a held-out demo for validation, and compare against the same demos given in context.

### Gaps
- No measured wall-clock numbers for Qwen2.5-VL-7B LoRA on fewer than 200 GUI samples on an A100 or H100. The project should measure and report this, since it is useful for the "live" claim.
- Official LLaMA-Factory, TRL, and MLX-VLM docs were not fetched (only mirrors or secondary evidence). Unsloth's SFT (not RL) speedup numbers for Qwen2.5-VL were not found.
- Whether UI-TARS-1.5-7B LoRA checkpoints load cleanly in vLLM multi-LoRA (expected given the `qwen2_5_vl` architecture tag, but not verified).
