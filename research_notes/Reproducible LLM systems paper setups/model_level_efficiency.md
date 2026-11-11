# Model-level and algorithm-level efficiency for LLM inference (2024-2026): problems and evaluation setups

Notes on method. Venue and year were checked against the OpenReview search API (venue string of the accepted record) unless marked otherwise; paper facts were read from the arXiv HTML full text of each paper. "Unverified" means I could not confirm the item from a primary source in this session. The "Reproducible on 4x48 GB" line in each entry is my own assessment from the hardware and model sizes the paper states, not a claim by the authors. OpenReview forum pages (reviews) were blocked by a bot challenge, so reviewer-raised weaknesses are not recorded anywhere below.

## Key Question 1: Which top-venue papers from 2024-2026 address these topics, and at which venue?

### Takeaway
Forty-five papers in scope were confirmed at NeurIPS, ICML, ICLR or MLSys for 2024-2026, including nine orals (Judge Decoding, Faster Cascades, FlexPrefill, KVzip, ThinKV, HSD, Prophet, Block Diffusion, Scaling Test-Time Compute Optimally) and ten spotlights. Three venues (MoE-Lightning at ASPLOS 2025, LayerSkip at ACL 2024, Spec-Bench at ACL Findings 2024) could not be confirmed from a primary source and are marked unverified.

### Cited Findings

Speculative decoding, multi-token prediction, layer skipping
- EAGLE-3 is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=4exx1hUffq)
- Judge Decoding is listed as ICLR 2025 Oral — [OpenReview](https://openreview.net/forum?id=mtSSFiqW6y)
- SWIFT (on-the-fly self-speculative decoding) is listed as ICLR 2025 Poster — [OpenReview](https://openreview.net/forum?id=EKJhH5D5wA)
- HSD (Lossless Hierarchical Speculative Decoding) is listed as ICLR 2026 Oral — [OpenReview](https://openreview.net/forum?id=LaVrNaBNwM)
- SuffixDecoding is listed as NeurIPS 2025 spotlight — [OpenReview](https://openreview.net/forum?id=uwL0vbeEVn)
- Kangaroo is listed as NeurIPS 2024 poster — [OpenReview](https://openreview.net/forum?id=lT3oc04mDp)
- Sequoia is listed as NeurIPS 2024 spotlight — [OpenReview](https://openreview.net/forum?id=rk2L9YGDi2)
- PEARL is listed as ICLR 2025 Poster — [OpenReview](https://openreview.net/forum?id=QOXrVMiHGK)
- Faster Cascades via Speculative Decoding is listed as ICLR 2025 Oral — [OpenReview](https://openreview.net/forum?id=vo9t20wsmd)

Sparse / approximate attention and KV pruning for long context
- MInference 1.0 is listed as NeurIPS 2024 spotlight — [OpenReview](https://openreview.net/forum?id=fPBACAbqSN); the arXiv comment field also says "Accepted at NeurIPS 2024 (Spotlight)" — [arXiv](https://arxiv.org/abs/2407.02490)
- FlexPrefill is listed as ICLR 2025 Oral — [OpenReview](https://openreview.net/forum?id=OfjIlbelrT)
- LServe is listed as MLSys 2025 — [OpenReview](https://openreview.net/forum?id=KQ4pJFFqT1)
- SampleAttention is listed as MLSys 2025 (with shepherding) — [OpenReview](https://openreview.net/forum?id=RuZ80yl71h)
- ShadowKV is listed as ICML 2025 spotlight poster — [OpenReview](https://openreview.net/forum?id=oa7MYAO6h6)
- MagicPIG is listed as ICLR 2025 Spotlight — [OpenReview](https://openreview.net/forum?id=ALzTQUgW8a)
- DuoAttention is listed as ICLR 2025 Poster — [OpenReview](https://openreview.net/forum?id=cFu7ze7xUm)
- SnapKV is listed as NeurIPS 2024 poster — [OpenReview](https://openreview.net/forum?id=poE54GOq2l)
- KVzip is listed as NeurIPS 2025 oral — [OpenReview](https://openreview.net/forum?id=JFygzwx8SJ)
- SmallKV is listed as NeurIPS 2025 spotlight — [OpenReview](https://openreview.net/forum?id=0BVrpXMr5Y)
- ChunkKV is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=20JDhbJqn3)
- SCBench (benchmark) is listed as ICLR 2025 Poster — [OpenReview](https://openreview.net/forum?id=gkUyYcY1W9)
- "Rethinking Key-Value Cache Compression Techniques for LLM Serving" is listed as MLSys 2025 — [OpenReview](https://openreview.net/forum?id=ou0AsNbXVG)

Quantization (weights, activations, KV)
- KIVI is listed as ICML 2024 Poster — [OpenReview](https://openreview.net/forum?id=L057s2Rq8O)
- QTIP is listed as NeurIPS 2024 spotlight — [OpenReview](https://openreview.net/forum?id=7sdkLVuYCU)
- SpinQuant is listed as ICLR 2025 Poster — [OpenReview](https://openreview.net/forum?id=ogO6DGE6FZ)
- FlatQuant is listed as ICML 2025 poster — [OpenReview](https://openreview.net/forum?id=uTz2Utym5n)
- QServe is listed as MLSys 2025 — [OpenReview](https://openreview.net/forum?id=1FfmStySS1)

Efficient decoding for reasoning models and test-time compute
- Certaindex / Dynasor ("Efficiently Scaling LLM Reasoning Programs with Certaindex") is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=nn51ewu5k2)
- SpecReason is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=wCbOKbZ7kf)
- Thinkless is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=ariVQf0KZx)
- R-KV is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=2jwAjomEDB)
- Controlling Thinking Speed in Reasoning Models is listed as NeurIPS 2025 spotlight — [OpenReview](https://openreview.net/forum?id=ESELaMThLN)
- DEER (Dynamic Early Exit in Reasoning Models) is listed as ICLR 2026 Poster — [OpenReview](https://openreview.net/forum?id=NpU7ZXafRi)
- ThinKV is listed as ICLR 2026 Oral — [OpenReview](https://openreview.net/forum?id=M3CeHnZKNC)
- "Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Parameters" is listed as ICLR 2025 Oral — [OpenReview](https://openreview.net/forum?id=4FWAwZtd2n)

Parallel / non-autoregressive decoding
- Hogwild! Inference is listed as NeurIPS 2025 spotlight — [OpenReview](https://openreview.net/forum?id=roKj4IwaVT)
- PASTA ("Learning to Keep a Promise") is listed as ICML 2025 poster — [OpenReview](https://openreview.net/forum?id=ZfX43ZZRZR)
- Multiverse is listed as NeurIPS 2025 spotlight — [OpenReview](https://openreview.net/forum?id=r9YDEErKXU)

Diffusion language models (inference speed)
- Block Diffusion is listed as ICLR 2025 Oral — [OpenReview](https://openreview.net/forum?id=tyEyYT267x)
- dKV-Cache is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=Gppo2JImHs)
- Fast-dLLM is listed as ICLR 2026 Poster — [OpenReview](https://openreview.net/forum?id=3Z3Is6hnOT)
- Prophet ("Diffusion Language Model Knows the Answer Before It Decodes") is listed as ICLR 2026 Oral — [OpenReview](https://openreview.net/forum?id=g88nt4ieTG)

Efficient architectures (hybrids, MoE)
- The Mamba in the Llama is listed as NeurIPS 2024 poster — [OpenReview](https://openreview.net/forum?id=uAzhODjALU)
- Jet-Nemotron is listed as NeurIPS 2025 poster — [OpenReview](https://openreview.net/forum?id=WZQXaTNYEB)
- Read-ME (router-decoupled MoE) is listed as NeurIPS 2024 poster — [OpenReview](https://openreview.net/forum?id=i8JaxY7tDI)

### Inferences
- The count of 2026-venue follow-ups found per line (listed under Key Question 2) is largest for KV compression, reasoning-model early exit / budget control, and speculative decoding; each has ten or more ICLR 2026 / ICML 2026 records in the OpenReview search results.
- No OSDI / SOSP / EuroSys / ATC paper appears in the verified list; the systems-venue papers found in scope are the MLSys 2025 ones (LServe, QServe, SampleAttention, the KV-compression rethink) plus the unverified ASPLOS 2025 MoE-Lightning.

### Gaps
- MoE-Lightning at ASPLOS 2025: unverified. The arXiv full text does not state the venue and I did not reach the ASPLOS program page.
- LayerSkip at ACL 2024 and Spec-Bench at ACL Findings 2024: unverified (not on OpenReview search; ACL Anthology not checked).
- TokenSkip (EMNLP 2025?), LLaDA (ICML 2025?), "Reasoning Models Can Be Effective Without Thinking", and The Sparse Frontier: no accepted-venue record found; left out of the per-paper entries.
- OpenReview topic searches for MoE inference, parallel decoding, layer skipping / early exit, and multi-token prediction were not completed (HTTP 429 rate limit), so coverage of those four sub-areas is thinner than the others and there may be 2025-2026 main-track papers I did not see.
- Oral/spotlight status comes from the OpenReview venue string only; it was not cross-checked against the conference virtual sites except HSD.

## Key Question 2: For each paper, what is the problem, the evaluation setup, the reproducibility on 4x48 GB, the stated limitations, and the follow-ups?

### Takeaway
Most algorithm-level papers in this pool were evaluated on a single GPU with 7B-32B open models and public benchmarks, so their setups fit a 4x48 GB node; the exceptions are the ones whose headline uses 70B-405B targets on multi-H100 nodes (Judge Decoding, parts of EAGLE-3 and HSD), 1M-token contexts on an 80 GB card (MInference, ShadowKV), or large training runs (Jet-Nemotron, Mamba in the Llama, Multiverse). Two papers report results directly on 48 GB cards: SpecReason (2x A6000) and SWIFT (4x A6000), and LServe / QServe / Hogwild! report on a single L40S.

### Cited Findings

#### A. Speculative decoding, multi-token prediction, layer skipping

**A1. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test — NeurIPS 2025 (poster)**
- Links: paper [arXiv 2503.01840](https://arxiv.org/abs/2503.01840), venue record [OpenReview](https://openreview.net/forum?id=4exx1hUffq), code [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE).
- Problem: the EAGLE draft head predicts next features, which constrains it and stops it from benefiting from more training data; EAGLE-3 predicts tokens directly and fuses multi-layer features, trained with a "training-time test" procedure. Users: anyone serving an open LLM who wants lossless single-request latency reduction — [arXiv](https://arxiv.org/abs/2503.01840)
- Models: Vicuna 13B, LLaMA-Instruct 3.1 8B, LLaMA-Instruct 3.3 70B, DeepSeek-R1-Distill-LLaMA 8B; "Due to the GPU constraint, we are unable to test EAGLE-3 on the 405B and 671B models." — [arXiv](https://arxiv.org/abs/2503.01840)
- Tasks: "Following EAGLE and Spec-Bench, we evaluate on five common tasks, using the same weights for all tasks": MT-bench, HumanEval, GSM8K, Alpaca, CNN/Daily Mail — [arXiv](https://arxiv.org/abs/2503.01840)
- Metrics: speedup ratio vs vanilla autoregressive decoding, average acceptance length tau, acceptance rate n-alpha; generation quality is not evaluated because strict speculative sampling is used ("ensuring no loss in performance") — [arXiv](https://arxiv.org/abs/2503.01840)
- Baselines: SpS, PLD, Medusa, Lookahead, Hydra, EAGLE, EAGLE-2 (HASS also compared) — [arXiv](https://arxiv.org/abs/2503.01840)
- Headline: abstract states "speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2"; mean at T=0: Vicuna-13B 5.51x (tau 6.62) vs EAGLE-2 4.22x, L3.1-8B 4.44x vs 3.23x, L3.3-70B 4.12x vs 2.85x, DSL-8B 4.16x vs 3.26x — [arXiv](https://arxiv.org/abs/2503.01840)
- Batched results: in SGLang v0.4.4 on a single H100 with LLaMA-Instruct 3.1 8B on MT-Bench (run by the SGLang team, chain length 3, no tree) EAGLE-3 gives 1.38x throughput at batch 64 while EAGLE falls below 1.0x from batch 24; in vLLM, EAGLE peaks at batch 24 and EAGLE-3 at 56 (the body text says RTX 3090, the Table 5 caption says A100 — the paper is internally inconsistent here) — [arXiv](https://arxiv.org/abs/2503.01840)
- Framework: main results use the authors' HuggingFace-transformers-based code; batched studies use SGLang and vLLM — [arXiv](https://arxiv.org/abs/2503.01840)
- Reproducible on 4x48 GB (assessment): yes for the 8B and 13B targets with released draft heads; scaled down for the 70B target (needs FP16 across 4 cards or quantization; training a 70B draft head is the costly part).
- Stated limitations: no limitations section was found in the arXiv text; the only stated constraint is the GPU limit that prevented 405B / 671B tests — [arXiv](https://arxiv.org/abs/2503.01840)
- Follow-ups: HSD integrates into EAGLE-3 for "over a 12% performance gain" — [arXiv 2601.05724](https://arxiv.org/abs/2601.05724); SuffixDecoding compares against and hybridises with EAGLE-3 — [arXiv 2411.04975](https://arxiv.org/abs/2411.04975); "Efficient EAGLE-3 Training via Structure-Aware Execution" (ICLR 2027 submission, under review) — [OpenReview](https://openreview.net/forum?id=FdEDds2GWP); SpecForge training framework (ICML 2026) — [OpenReview](https://openreview.net/forum?id=CQOEbxy0tE); DFlash block-diffusion drafter (ICML 2026) — [OpenReview](https://openreview.net/forum?id=Oz335dV48X)

**A2. Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment — ICLR 2025 (Oral)**
- Links: paper [arXiv 2501.19309](https://arxiv.org/abs/2501.19309), venue record [OpenReview](https://openreview.net/forum?id=mtSSFiqW6y), code: no repository link found in the paper text (unverified whether code exists elsewhere).
- Problem: standard speculative-decoding verification rejects draft tokens that are correct but not aligned with the target's distribution, so acceptance stays low even for very strong drafters; the paper trains a small linear "judge" head on target embeddings to accept contextually correct tokens — [arXiv](https://arxiv.org/abs/2501.19309)
- Models and hardware: Llama-3.1-8B draft with Llama-3.1-70B and 405B targets; "single node of H100-SXM5 GPUs. For Llama-405B we use 8 GPUs and 8-bit quantization ... For Llama-70B ... 8-bit quantization but only 2 GPUs"; frameworks HuggingFace and gpt-fast — [arXiv](https://arxiv.org/abs/2501.19309)
- Benchmarks: GSM8K, MT-Bench, HumanEval among others — [arXiv](https://arxiv.org/abs/2501.19309)
- Headline: 9x speedup over standard decoding for 8B/405B at 129 tokens/s on 8 H100s, and 141 tokens/s for 8B/70B on 2 H100s, "while maintaining Llama-405B quality" — [arXiv](https://arxiv.org/abs/2501.19309)
- Reproducible on 4x48 GB (assessment): scaled down — the 8B/70B pair with 8-bit 70B fits in 4x48 GB (the paper used 2x 80 GB); the 405B headline does not fit. No public code found, so the judge head and its annotated data would have to be rebuilt.
- Stated limitations (Appendix A, quoted): "An obvious limitation is the loss of the mathematical guarantee to match target quality ... there is no certainty for novel tasks."; "The draft model needs to be of high-quality ... Self-speculation and small drafters in the spirit of Medusa or Eagle are thus not ideal since their generations quickly deteriorate when drafting too far into the future."; "the target model needs to be of sufficient size to be able to provide accurate judgements. Speedups for smaller models such as Llama-8B are hence tougher to achieve."; "new tasks do require careful annotation of data to maintain quality."; "If the draft model has safety issues, the target model could potentially accept safety-critical tokens through the judge" — [arXiv](https://arxiv.org/abs/2501.19309)
- Follow-ups: SelfJudge (ICML 2026) — [OpenReview](https://openreview.net/forum?id=9DAX4oLVhy); Training-Free Loosely Speculative Decoding (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=JjoTg34YiU)

**A3. SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration — ICLR 2025 (Poster)**
- Links: paper [arXiv 2410.06916](https://arxiv.org/abs/2410.06916), venue record [OpenReview](https://openreview.net/forum?id=EKJhH5D5wA), code [hemingkx/SWIFT](https://github.com/hemingkx/SWIFT).
- Problem: most speculative-decoding methods need an auxiliary draft model or extra training; SWIFT is plug-and-play and selects, per input stream, which intermediate layers of the target to skip when drafting — [arXiv](https://arxiv.org/abs/2410.06916)
- Models: LLaMA-2 13B / 70B (and chat variants), CodeLLaMA 13B / 34B; also Yi-34B and DeepSeek-Coder-33B — [arXiv](https://arxiv.org/abs/2410.06916)
- Datasets: CNN/DM, GSM8K, TinyStories, HumanEval — [arXiv](https://arxiv.org/abs/2410.06916)
- Hardware and framework: "Pytorch 2.1.0 on 4×NVIDIA RTX A6000 GPU (40GB) [sic] with CUDA 12.1, and an Intel(R) Xeon(R) Platinum 8370C CPU with 32 cores. Inference for our method and all baselines was performed using the Huggingface transformers package ... speculative sampling ... batch size is set to 1." — [arXiv](https://arxiv.org/abs/2410.06916)
- Baselines: "two existing plug-and-play methods: Parallel Decoding (Santilli et al., 2023) and Lookahead Decoding (Fu et al., 2024)"; other SD methods are excluded "as they necessitate additional modules or extensive training" — [arXiv](https://arxiv.org/abs/2410.06916)
- Metrics: speedup, mean generated length M per target forward pass, token acceptance rate alpha — [arXiv](https://arxiv.org/abs/2410.06916)
- Headline: 1.3x-1.6x speedup, lossless — [arXiv](https://arxiv.org/abs/2410.06916)
- Reproducible on 4x48 GB (assessment): yes — the paper's own testbed is 4x RTX A6000.
- Stated limitations / open problems: larger models show more layer sparsity, and accelerating larger LLMs "(e.g., 175B) ... we leave for future investigation"; SWIFT is described as orthogonal to Jacobi-based methods and "integrating Swift with them could further boost inference efficiency". No dedicated limitations section was located — [arXiv](https://arxiv.org/abs/2410.06916)
- Follow-ups: CAS-Spec (NeurIPS 2025 poster) — [OpenReview](https://openreview.net/forum?id=m0bR0sxhfL); Vegas, self-speculative decoding with verification-guided sparse attention (ICML 2026) — [OpenReview](https://openreview.net/forum?id=vht4Pn0swG)

**A4. HSD: Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding — ICLR 2026 (Oral)**
- Links: paper [arXiv 2601.05724](https://arxiv.org/abs/2601.05724), venue record [OpenReview](https://openreview.net/forum?id=LaVrNaBNwM), code [ZhouYuxuanYX/Hierarchical-Speculative-Decoding](https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding).
- Problem: token-wise and block-wise lossless verification accept fewer draft tokens than the ideal joint acceptance rule; HSD is a provably lossless verification rule that accepts more — [arXiv](https://arxiv.org/abs/2601.05724)
- Setup: "GPTQ-quantized 8-bit instruction-tuned Qwen2.5 series ... 0.5B as the draft model and 72B as the target models, with a temperature of 1"; GSM8K, HumanEval, CNN/DailyMail; "All experiments were conducted on a single NVIDIA H20 GPU with 96 GB of memory, unless otherwise specified" — [arXiv](https://arxiv.org/abs/2601.05724)
- Baselines and metrics: token-wise and block-wise verification; Block Efficiency (tokens per serial target call) and Decoding Speed (tokens/s) — [arXiv](https://arxiv.org/abs/2601.05724)
- Headline: "integrating HSD into EAGLE-3 yields over a 12% performance gain"; up to 12.3% on individual datasets — [arXiv](https://arxiv.org/abs/2601.05724)
- Reproducible on 4x48 GB (assessment): scaled down — an 8-bit 72B target (roughly 72 GB of weights) needs two or more 48 GB cards instead of one 96 GB card, which changes the tokens/s numbers; Block Efficiency is hardware-independent by the paper's definition.
- Stated limitations: none located in the arXiv text (the word appears only when describing prior work) — [arXiv](https://arxiv.org/abs/2601.05724)
- Follow-ups: none found (the paper is from 2026).

**A5. SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications — NeurIPS 2025 (spotlight)**
- Links: paper [arXiv 2411.04975](https://arxiv.org/abs/2411.04975), venue record [OpenReview](https://openreview.net/forum?id=uwL0vbeEVn), code in [snowflakedb/ArcticInference](https://github.com/snowflakedb/ArcticInference).
- Problem: agentic workloads issue repetitive, predictable requests that model-based drafters do not exploit; SuffixDecoding drafts from suffix trees over prompts and previous outputs with no draft model (about 20 microseconds per draft token, no GPU overhead) — [arXiv](https://arxiv.org/abs/2411.04975)
- Benchmarks: AgenticSQL, SWE-Bench (OpenHands), and Spec-Bench ("to stress-test SuffixDecoding's limitations and evaluate the hybrid fallback mechanism") — [arXiv](https://arxiv.org/abs/2411.04975)
- Hardware: "a single p5.48xlarge AWS instance equipped with 8× NVIDIA H100 80G GPUs and 2TB of main memory"; end-to-end evaluation implemented in vLLM running OpenHands live — [arXiv](https://arxiv.org/abs/2411.04975)
- Baselines: EAGLE-2/3, PLD, Token Recycling — [arXiv](https://arxiv.org/abs/2411.04975)
- Headline: AgenticSQL mean speedup 5.3x over vanilla, "a 2.8× improvement over EAGLE-2/3, and 1.9× higher than Token Recycling"; SWE-Bench mean 2.5x, 1.7x over PLD; "In SWE-Bench, EAGLE-2/3 fail due to their maximum sequence length limitations"; mean accepted tokens per step 6.3 on AgenticSQL vs 3.6 for EAGLE-3 — [arXiv](https://arxiv.org/abs/2411.04975)
- Reproducible on 4x48 GB (assessment): yes for the single-GPU, batch-1 comparisons with Llama-3.1-8B-Instruct; scaled down for the end-to-end run (the paper serves a 32B OpenHands model with 4-way tensor parallelism on 4 H100; the same model fits 4x48 GB but with less KV headroom).
- Stated limitations: on Spec-Bench (non-agentic) SuffixDecoding alone is outperformed by model-based methods, which is why the paper adds a hybrid fallback to EAGLE-3 — [arXiv](https://arxiv.org/abs/2411.04975)
- Follow-ups: PRISM (MLSys 2026) — [OpenReview](https://openreview.net/forum?id=cvU2HuuxEf); "Beat the long tail: Distribution-Aware Speculative Decoding for RL Training" (MLSys 2026) — [OpenReview](https://openreview.net/forum?id=kMeqqPBjSl)

**A6. Shorter records in this line**
- Kangaroo (NeurIPS 2024 poster): self-speculative decoding with a shallow sub-network plus adapter and double early exit; Vicuna 7B/13B on Spec-Bench, NVIDIA V100 GPUs, baselines Lookahead / Medusa / REST, up to 1.7x speedup with 88.7% fewer extra parameters than Medusa-1 — [arXiv 2404.18911](https://arxiv.org/abs/2404.18911), [OpenReview](https://openreview.net/forum?id=lT3oc04mDp), code [Equationliu/Kangaroo](https://github.com/Equationliu/Kangaroo)
- LayerSkip (venue ACL 2024 unverified): layer dropout plus early-exit loss at training time, then self-speculative decoding; up to 2.16x on CNN/DM summarization, 1.82x on coding, 2.0x on TOPv2; the authors state it requires finetuning or pretraining with their recipe and tuning of p_max, e_scale and R — [arXiv 2404.16710](https://arxiv.org/abs/2404.16710), code [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
- Spec-Bench (venue ACL Findings 2024 unverified): see Key Question 3 — [arXiv 2401.07851](https://arxiv.org/abs/2401.07851)
- Other 2026 records found in this line (not read in full): Speculative Speculative Decoding (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=aL1Wnml9Ef); "Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling" (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=DjOmnwX4wJ); MineDraft, batch parallel SD (ICML 2026) — [OpenReview](https://openreview.net/forum?id=mapFG1BrNv); SpecBranch (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=BrnlCSqO6n); Learning To Draft (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=IK9cbzzXLt); RAPID (ICML 2025 spotlight) — [OpenReview](https://openreview.net/forum?id=73mDARqOtQ)

#### B. Sparse / approximate attention and KV pruning for long context

**B1. MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention — NeurIPS 2024 (spotlight)**
- Links: paper [arXiv 2407.02490](https://arxiv.org/abs/2407.02490), venue record [OpenReview](https://openreview.net/forum?id=fPBACAbqSN), code: github.com/microsoft/MInference (from memory, not re-checked in this session — unverified).
- Problem: prefill over a 1M-token prompt takes about 30 minutes for an 8B model on a single A100; the paper identifies three attention patterns (A-shape, Vertical-Slash, Block-Sparse) and builds per-head dynamic sparse masks with custom kernels. Users: long-document / repository-level prompting on one GPU — [arXiv](https://arxiv.org/abs/2407.02490)
- Models: LLaMA-3-8B-Instruct-262k / 1M, GLM-4-9B-1M, Yi-9B-200K, Phi-3-Mini-128K, Qwen2-7B-128K — [arXiv](https://arxiv.org/abs/2407.02490)
- Benchmarks: InfiniteBench (10 tasks), RULER, PG-19 language modeling, Needle In A Haystack — [arXiv](https://arxiv.org/abs/2407.02490)
- Hardware and framework: "The latency experiments are conducted on a single Nvidia A100 GPU in the bfloat16 format"; custom PyTorch implementation on FlashAttention, Triton and PIT — [arXiv](https://arxiv.org/abs/2407.02490)
- Headline: "up to 10x speedup for 1M contexts on a single A100" (30 min to 3 min) while matching or surpassing baselines on accuracy — [arXiv](https://arxiv.org/abs/2407.02490)
- Reproducible on 4x48 GB (assessment): scaled down — accuracy benchmarks at 128K-200K fit a 48 GB card for 8B-9B models; the 1M-token latency point was measured on an 80 GB A100 and may not fit in 48 GB.
- Stated limitations (Appendix A): at short contexts (10k) the index-building overhead share rises from 5% to 30% so latency approaches FlashAttention; "when using a higher sparsity rate, the model performance may noticeably decline" — [arXiv](https://arxiv.org/abs/2407.02490)
- Follow-ups: FlexPrefill (B2); SCBench from the same group — [arXiv 2412.10319](https://arxiv.org/abs/2412.10319); MMInference (ICML 2025 poster) — [OpenReview](https://openreview.net/forum?id=me6PfbATWM); SALE, low-bit estimation for sparse prefill (ICML 2026) — [OpenReview](https://openreview.net/forum?id=WyGEJKjvJN); Token Sparse Attention (ICML 2026) — [OpenReview](https://openreview.net/forum?id=BXz2zosn6H)

**B2. FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference — ICLR 2025 (Oral)**
- Links: paper [arXiv 2502.20766](https://arxiv.org/abs/2502.20766), venue record [OpenReview](https://openreview.net/forum?id=OfjIlbelrT), code [bytedance/FlexPrefill](https://github.com/bytedance/FlexPrefill).
- Problem: fixed sparse patterns and fixed budgets (as in MInference) do not adapt to the input; FlexPrefill chooses the pattern per query and sets the per-head budget by cumulative attention — [arXiv](https://arxiv.org/abs/2502.20766)
- Models: Llama-3.1-8B-Instruct (128k), GLM-4-9B-Chat (1M), Yi-9B-200K, Qwen2-7B-Instruct (128k) — [arXiv](https://arxiv.org/abs/2502.20766)
- Datasets: RULER (4k-128k) and InfiniteBench (average 214k tokens, 10 tasks) — [arXiv](https://arxiv.org/abs/2502.20766)
- Hardware and framework: "a single NVIDIA A100 GPU with 80GB of memory"; "a custom pipeline using PyTorch, building on FlashAttention"; Triton kernels, block_size = 128 — [arXiv](https://arxiv.org/abs/2502.20766)
- Baselines and headline: compared against MInference and fixed-budget Vertical-Slash attention on an accuracy-vs-attention-latency trade-off curve; the paper states its method "consistently outperforms MInference and Fixed Budget Vertical-Slash across different attention latencies" (full-attention Llama RULER average 88.70 in Table 1) — [arXiv](https://arxiv.org/abs/2502.20766)
- Reproducible on 4x48 GB (assessment): yes for RULER up to 128k with 7B-9B models; scaled down for the longest InfiniteBench inputs if they exceed 48 GB.
- Stated limitations: no limitations section; the only forward-looking statement is "Future work could explore further optimizations and applications of this adaptive approach in diverse model architectures." — [arXiv](https://arxiv.org/abs/2502.20766)
- Follow-ups: Dynamic Hierarchical Sparse Attention (ICML 2026 spotlight) — [OpenReview](https://openreview.net/forum?id=o3gN27ITWV); Long-Context Attention Benchmark (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=W7sVYFJAEp); Scout Before You Attend (ICML 2026) — [OpenReview](https://openreview.net/forum?id=uCLVPafHqd)

**B3. LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention — MLSys 2025**
- Links: paper [arXiv 2502.14866](https://arxiv.org/abs/2502.14866), venue record [OpenReview](https://openreview.net/forum?id=KQ4pJFFqT1), code [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve).
- Problem: prefill cost grows quadratically and decode memory linearly with context; LServe unifies static sparsity (half of the heads become streaming heads, following DuoAttention) and dynamic query-aware KV page selection in one block-sparse serving engine — [arXiv](https://arxiv.org/abs/2502.14866)
- Models and benchmarks: Llama-3-8B and Llama-2-7B on LongBench (10 tasks) and Needle-in-a-Haystack; DeepSeek-R1-Distill-Llama-8B for a reasoning-model check — [arXiv](https://arxiv.org/abs/2502.14866)
- Hardware and framework: built in CUDA on QServe / TensorRT-LLM lineage; evaluated on an A100 80 GB server and a single L40S 48 GB instance — [arXiv](https://arxiv.org/abs/2502.14866)
- Baseline and headline: "accelerates LLM prefilling by up to 2.9× and decoding by 1.3-2.1× over vLLM, maintaining long-context accuracy" — [arXiv](https://arxiv.org/abs/2502.14866)
- Reproducible on 4x48 GB (assessment): yes — one of the two testbeds is a single 48 GB L40S, and models are 7B-8B.
- Stated limitations: none located in the arXiv text — [arXiv](https://arxiv.org/abs/2502.14866)
- Follow-ups: Tactic (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=tJod11fK1A); LycheeDecode (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=YWCHLdNGVU); Sparse Attention Adaptation for Long Reasoning (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=c5BOcHM6J8)

**B4. ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference — ICML 2025 (spotlight poster)**
- Links: paper [arXiv 2410.21465](https://arxiv.org/abs/2410.21465), venue record [OpenReview](https://openreview.net/forum?id=oa7MYAO6h6), code [ByteDance-Seed/ShadowKV](https://github.com/ByteDance-Seed/ShadowKV).
- Problem: existing KV eviction and sparse-attention methods have "accuracy degradation, inadequate memory reduction, and significant decoding latency overhead"; ShadowKV keeps a low-rank key cache on GPU, offloads the value cache to CPU, and reconstructs a sparse KV set on the fly — [arXiv](https://arxiv.org/abs/2410.21465)
- Models: Llama-3-8B-1M, GLM-4-9B-1M, Llama-3.1-8B, Yi-9B-200K, Phi-3-Mini-128K, Qwen2-7B-128K — [arXiv](https://arxiv.org/abs/2410.21465)
- Benchmarks: RULER, LongBench, Needle In A Haystack; baselines Quest, Loki, InfiniGen — [arXiv](https://arxiv.org/abs/2410.21465)
- Headline: "up to 6× larger batch sizes and boost throughput by up to 3.04× on an A100 GPU without sacrificing accuracy" — [arXiv](https://arxiv.org/abs/2410.21465)
- Reproducible on 4x48 GB (assessment): scaled down — same 8B-9B models, but batch-size and throughput points were measured against an 80 GB A100 memory limit, and the CPU-offload bandwidth depends on the host.
- Stated limitations: no limitations or future-work section located in the arXiv text — [arXiv](https://arxiv.org/abs/2410.21465)
- Follow-ups: StrideKV (ICLR 2027 submission, under review) — [OpenReview](https://openreview.net/forum?id=OWKY1jCUhO); xKV (ICML 2026) — [OpenReview](https://openreview.net/forum?id=XDb69lykA8); RaBitQCache (ICML 2026) — [OpenReview](https://openreview.net/forum?id=zc8ZDMJmKB)

**B5. KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction — NeurIPS 2025 (oral)**
- Links: paper [arXiv 2505.23416](https://arxiv.org/abs/2505.23416), venue record [OpenReview](https://openreview.net/forum?id=JFygzwx8SJ), code [snu-mllab/KVzip](https://github.com/snu-mllab/KVzip).
- Problem: query-aware KV eviction (SnapKV-style) degrades when a compressed cache is reused for different queries; KVzip scores KV pairs by how much the model needs them to reconstruct the context, so one compressed cache serves many queries — [arXiv](https://arxiv.org/abs/2505.23416)
- Setup: LLaMA-3.1-8B, Qwen2.5 and Gemma-3 families; contexts up to 170K tokens; question answering, retrieval, reasoning and code tasks, plus RULER in the appendix; baselines are query-aware eviction methods and head-level DuoAttention; single A100 80 GB (hardware as extracted from the paper) — [arXiv](https://arxiv.org/abs/2505.23416)
- Headline: KV cache size reduced 3-4x and FlashAttention decoding latency about 2x lower with negligible loss; compression costs roughly 2x a normal prefill — [arXiv](https://arxiv.org/abs/2505.23416)
- Reproducible on 4x48 GB (assessment): yes for 8B-class models at moderate context; scaled down for the longest (170K) contexts and largest models.
- Stated limitations (Appendix B, closely paraphrased): the study "primarily adopts an empirical approach and does not include theoretical guarantees concerning compression-induced information loss"; KV eviction "might raise potential concerns regarding privacy leakage"; the approach "involves a compression overhead" that "can be amortized over multiple queries" — [arXiv](https://arxiv.org/abs/2505.23416)
- Follow-ups: Fast KVzip (arXiv comment field says NeurIPS 2026; not checked on OpenReview) — [arXiv 2601.17668](https://arxiv.org/abs/2601.17668); EchoPress (ICLR 2027 submission, under review) — [OpenReview](https://openreview.net/forum?id=kmxFD1i4br); CriticalKV (ICML 2026) — [OpenReview](https://openreview.net/forum?id=vMwlcLyI4b); EntroKV (ICML 2026 spotlight) — [OpenReview](https://openreview.net/forum?id=xhAMjsnWUe); STAR-KV (ICML 2026 spotlight) — [OpenReview](https://openreview.net/forum?id=lJjH1q6RwY)

**B6. Shorter records in this line**
- SnapKV (NeurIPS 2024 poster): selects KV positions per head using an observation window at the end of the prompt; evaluated on 16 LongBench datasets and inputs up to 380K tokens on a single A100 80 GB with a HuggingFace implementation; used as a baseline by KVzip, ThinKV and SCBench — [arXiv 2404.14469](https://arxiv.org/abs/2404.14469), [OpenReview](https://openreview.net/forum?id=poE54GOq2l)
- SCBench (ICLR 2025 Poster): see Key Question 3 — [arXiv 2412.10319](https://arxiv.org/abs/2412.10319)

#### C. Quantization and low-bit inference

**C1. FlatQuant: Flatness Matters for LLM Quantization — ICML 2025 (poster)**
- Links: paper [arXiv 2410.09426](https://arxiv.org/abs/2410.09426), venue record [OpenReview](https://openreview.net/forum?id=uTz2Utym5n), code [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant).
- Problem: W4A4 post-training quantization still loses accuracy after rotation-based pre-transforms because weight and activation distributions stay steep; FlatQuant learns per-layer affine transforms (Kronecker-decomposed) to flatten them — [arXiv](https://arxiv.org/abs/2410.09426)
- Models: LLaMA-2 7B / 13B / 70B, LLaMA-3 8B / 70B — [arXiv](https://arxiv.org/abs/2410.09426)
- Benchmarks: WikiText-2 and C4 perplexity plus zero-shot QA tasks; baselines SmoothQuant, OmniQuant, AffineQuant, QuaRot, SpinQuant — [arXiv](https://arxiv.org/abs/2410.09426)
- Headline: "<1% accuracy drop for W4A4 on LLaMA-3-70B, surpassing SpinQuant by 7.5%"; up to 2.3x prefill and 1.7x decoding speedup over FP16, with latency measured on an RTX 3090; W4A4 RTN WikiText-2 perplexity on LLaMA-2 7B: 5.79 vs FP16 5.47, SpinQuant 6.14, QuaRot 8.56 — [arXiv](https://arxiv.org/abs/2410.09426)
- Reproducible on 4x48 GB (assessment): yes for 7B-13B; scaled down for 70B calibration (layer-wise calibration of a 70B model needs multi-GPU placement or longer wall-clock).
- Stated limitations (quoted): "the full potential of 4-bit quantization has not been thoroughly explored"; "the optimal selection of calibration sets remains an open question"; "we have not examined the integration of FlatQuant with newer data types, such as MXFP4" — [arXiv](https://arxiv.org/abs/2410.09426)
- Follow-ups: STaMP (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=suB4wYViDt); SERQ (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=nFjj8NEBqv); TwinQuant (ICML 2026) — [OpenReview](https://openreview.net/forum?id=ahYRAjx7UG); LiftQuant (ICML 2026 spotlight) — [OpenReview](https://openreview.net/forum?id=1GvXUhLIMP); "INT vs. FP" (ICML 2026) — [OpenReview](https://openreview.net/forum?id=1GIYHWO9S5)

**C2. SpinQuant: LLM Quantization with Learned Rotations — ICLR 2025 (Poster)**
- Links: paper [arXiv 2405.16406](https://arxiv.org/abs/2405.16406), venue record [OpenReview](https://openreview.net/forum?id=ogO6DGE6FZ), code [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant).
- Problem and setup: random rotations before quantization give high variance in accuracy; SpinQuant learns the rotation matrices with Cayley SGD on 800 WikiText-2 samples for 100 iterations; W4A4KV4 gap to full precision of 2.9 points on LLaMA-2 7B zero-shot reasoning — [arXiv](https://arxiv.org/abs/2405.16406)
- Cost: rotation optimization takes about 13 / 18 / 30 minutes for LLaMA-3 1B / 3B / 8B and about 3.5 hours for LLaMA-2 70B — [arXiv](https://arxiv.org/abs/2405.16406)
- Reproducible on 4x48 GB (assessment): yes for 1B-13B; scaled down for 70B.
- Follow-up that reports beating it: FlatQuant (C1) — [arXiv 2410.09426](https://arxiv.org/abs/2410.09426)

**C3. QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving — MLSys 2025**
- Links: paper [arXiv 2405.04532](https://arxiv.org/abs/2405.04532), venue record [OpenReview](https://openreview.net/forum?id=1FfmStySS1), code [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve).
- Problem: on cloud GPUs, existing INT4 weight quantization pays a large dequantization overhead so it is not faster than higher-precision serving; the paper notes "On data center GPUs like A100, a CUDA core operation is as expensive as 50 INT4 tensor core operations" and proposes the QoQ algorithm (W4A8KV4, progressive group quantization, SmoothAttention) with matching kernels — [arXiv](https://arxiv.org/abs/2405.04532)
- Accuracy evaluation: WikiText-2 perplexity, zero-shot accuracy and long-context accuracy (section 6.2 headings) — [arXiv](https://arxiv.org/abs/2405.04532)
- Efficiency evaluation: throughput vs TensorRT-LLM on A100 and L40S: Llama-3-8B 1.2x (A100) and 1.4x (L40S); Qwen1.5-72B 2.4x (A100) and 3.5x (L40S); QServe on L40S exceeds TensorRT-LLM on A100 — [arXiv](https://arxiv.org/abs/2405.04532)
- The paper has an MLSys artifact appendix (hardware and software dependencies, experiment workflow, expected results) — [arXiv](https://arxiv.org/abs/2405.04532)
- Reproducible on 4x48 GB (assessment): yes — L40S 48 GB is one of the paper's two testbeds, and W4 weights for a 72B model fit one 48 GB card.
- Stated limitations: none located in the arXiv text — [arXiv](https://arxiv.org/abs/2405.04532)
- Follow-ups: LServe builds on the same engine (B3) — [arXiv 2502.14866](https://arxiv.org/abs/2502.14866)

**C4. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache — ICML 2024 (Poster)**
- Links: paper [arXiv 2402.02750](https://arxiv.org/abs/2402.02750), venue record [OpenReview](https://openreview.net/forum?id=L057s2Rq8O), code [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI).
- Setup and result: keys quantized per-channel and values per-token at 2 bits; Llama / Llama-2, Falcon, Mistral in HuggingFace Transformers on LM-Eval tasks and LongBench; single A100 80 GB; up to 4x larger batch and 2.35x-3.47x throughput; the authors name reducing quantization overhead in prefill and decode as future work — [arXiv](https://arxiv.org/abs/2402.02750)
- Reproducible on 4x48 GB (assessment): yes (7B-13B models, single GPU).
- Used as a baseline by ThinKV (D3) — [arXiv 2510.01290](https://arxiv.org/abs/2510.01290)

#### D. Efficient decoding for reasoning models and test-time compute

**D1. Certaindex / Dynasor: Efficiently Scaling LLM Reasoning Programs with Certaindex — NeurIPS 2025 (poster)**
- Links: paper [arXiv 2412.20993](https://arxiv.org/abs/2412.20993), venue record [OpenReview](https://openreview.net/forum?id=nn51ewu5k2), code [hao-ai-lab/Dynasor](https://github.com/hao-ai-lab/Dynasor).
- Problem: reasoning programs (self-consistency, MCTS, Rebase, CoT) spend the same compute on easy and hard queries; certaindex is an answer-stabilization signal used by a serving-side scheduler to stop or re-allocate tokens. Users: operators serving multi-call reasoning workloads — [arXiv](https://arxiv.org/abs/2412.20993)
- Workloads: self-consistency on LiveCodeBench and GSM8K with Llama-3.1-8B; MCTS on ASDiv / GSM8K with Llama-2-7B plus a 7B reward model; Rebase on MATH / GSM8K with Llemma 7B / 34B plus a 34B reward model — [arXiv](https://arxiv.org/abs/2412.20993)
- Hardware and framework: A100 80 GB GPUs (one GPU for SC, two for Rebase / MCTS: LLM plus reward model); "Dynasor is implemented as a lightweight scheduling component in SGLang ... only introduces ∼500 lines of code into the core system, and requires no change to the model weights" — [arXiv](https://arxiv.org/abs/2412.20993)
- Baselines and headline: SGLang intra-program scheduler with even allocation and with length-based allocation; certaindex "consistently reduces token usage by 9–52% across workloads compared to both baselines"; abstract: "up to 50% compute savings and 3.3× higher throughput in real workloads with no accuracy drop" — [arXiv](https://arxiv.org/abs/2412.20993)
- Reproducible on 4x48 GB (assessment): yes — 7B-34B models on one or two GPUs; the 34B LLM plus 34B reward model need one 48 GB card each at FP16 with reduced KV headroom, or two each.
- Stated limitations (quoted): the work "did not explore its integration with advanced serving techniques like PD disaggregation or chunked prefill. The potential impact of these integrations on the latency-accuracy trade-off in real-world workloads remains an open area for future research." — [arXiv](https://arxiv.org/abs/2412.20993)
- Follow-ups: HELIOS (MLSys 2026) — [OpenReview](https://openreview.net/forum?id=CV52m9NJFK); "What If We Allocate Test-Time Compute Adaptively?" (ICML 2026) — [OpenReview](https://openreview.net/forum?id=wblvinAlrW); Zero-Overhead Introspection for Adaptive Test-Time Compute (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=GqZYGOYuF2)

**D2. DEER: Dynamic Early Exit in Reasoning Models — ICLR 2026 (Poster)**
- Links: paper [arXiv 2504.15895](https://arxiv.org/abs/2504.15895), venue record [OpenReview](https://openreview.net/forum?id=NpU7ZXafRi), code [iie-ycx/DEER](https://github.com/iie-ycx/DEER).
- Problem: reasoning models over-think; DEER is training-free and, at reasoning transition points, elicits a trial answer and exits the chain of thought when the trial answer's confidence is high — [arXiv](https://arxiv.org/abs/2504.15895)
- Models: "DeepSeek-R1-Distill-Qwen series of models (1.5B, 7B, 14B, and 32B), Qwen3 series of models (1.7B, 4B, 8B, 14B, 32B), QwQ-32B, and DeepSeek-R1"; R1-Distill-Qwen-7B, Qwen3-14B and QwQ-32B are shown in the main table — [arXiv](https://arxiv.org/abs/2504.15895)
- Benchmarks: GSM8K, MATH-500, AMC 2023, AIME 2024, AIME 2025, OlympiadBench, GPQA Diamond, HumanEval, BigCodeBench, LiveCodeBench; 4 sampling rounds per instance on the small AMC / AIME sets — [arXiv](https://arxiv.org/abs/2504.15895)
- Framework and hardware: "We have implemented DEER using both HuggingFace Transformers and the vLLM inference acceleration framework. The experimental results presented in this paper are based on the vLLM implementation"; confidence threshold lambda = 0.95; hardware stated as 8x H100 80 GB — [arXiv](https://arxiv.org/abs/2504.15895)
- Headline: chain-of-thought length reduced by 19.1%-80.1% on average while accuracy improves by 0.3%-5.0% across 11 reasoning models — [arXiv](https://arxiv.org/abs/2504.15895)
- Reproducible on 4x48 GB (assessment): yes — all open models are 32B or smaller and run in vLLM; only the full DeepSeek-R1 row is out of reach.
- Stated limitations: the paper reports that Llama-3.1-Nemotron-Nano-8B-v1 "consistently exhibits low confidence in generating intermediate answers, resulting in a significantly lower early stopping rate" (55% vs 80% for Qwen3-8B and 85% for R1-Distill-Qwen-7B), so the efficiency gain on that model is limited; trial-answer generation adds latency, which the paper addresses with a branch-parallel variant — [arXiv](https://arxiv.org/abs/2504.15895)
- Follow-ups: S-GRPO (NeurIPS 2025 poster) — [OpenReview](https://openreview.net/forum?id=wNMK5o0Vfg); SpecExit (ICML 2026) — [OpenReview](https://openreview.net/forum?id=5RMsiXvUqS); Statistical Early Stopping for Reasoning Models (ICML 2026) — [OpenReview](https://openreview.net/forum?id=TJshRZDdyW); "Does Your Reasoning Model Implicitly Know When to Stop Thinking?" (ICML 2026) — [OpenReview](https://openreview.net/forum?id=xF6klfHhO8); WAVE early exit (ICML 2026) — [OpenReview](https://openreview.net/forum?id=hCY1J253Kf); Conformal Thinking (ICML 2026) — [OpenReview](https://openreview.net/forum?id=noDJPmA3ha)

**D3. ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models — ICLR 2026 (Oral)**
- Links: paper [arXiv 2510.01290](https://arxiv.org/abs/2510.01290), venue record [OpenReview](https://openreview.net/forum?id=M3CeHnZKNC), code: no repository link found in the paper text (unverified whether code exists elsewhere).
- Problem: reasoning models generate tens of thousands of output tokens, so the decode-time KV cache, not the prompt, exhausts GPU memory (the paper's motivating case: GPT-OSS-20B at about 32K generated tokens and batch 32 needs about 50 GB of KV plus 40 GB of weights, more than an 80 GB A100). ThinKV classifies thought types and applies hybrid quantization plus eviction, with a kernel that extends PagedAttention — [arXiv](https://arxiv.org/abs/2510.01290)
- Models: DeepSeek-R1-Distill-Llama 8B / 70B, R1-Distill-Qwen-14B, GPT-OSS 20B / 120B, QwQ-32B, AceReason-Nemotron-14B, MobileLLM-R1-950M — [arXiv](https://arxiv.org/abs/2510.01290)
- Datasets: MATH-500, AIME, GSM8K, LiveCodeBench; calibration on 100 prompts from s1K; maximum generation 32K tokens, 8 responses per question — [arXiv](https://arxiv.org/abs/2510.01290)
- Hardware: one A100 80 GB and one NVIDIA GH200 — [arXiv](https://arxiv.org/abs/2510.01290)
- Baselines: SnapKV, RocketKV, H2O, RaaS, LazyEviction, R-KV, PM-KVQ, KIVI — [arXiv](https://arxiv.org/abs/2510.01290)
- Headline: near-lossless accuracy with less than 5% of the original KV cache and up to 5.8x higher throughput over state-of-the-art baselines — [arXiv](https://arxiv.org/abs/2510.01290)
- Reproducible on 4x48 GB (assessment): scaled down — 8B-32B models fit; the batch sizes at which the full-KV baseline runs out of memory shift on 48 GB cards, and no public code was found.
- Stated limitations (Appendix F, quoted): "While ThinKV demonstrates strong performance for long-output reasoning tasks, it is not directly applicable to settings dominated by long input contexts. Should future LRMs place greater emphasis on long-input contexts, additional exploration will be required." The paper also states "KV cache compression remains an underexplored yet critical direction" as reasoning outputs grow — [arXiv](https://arxiv.org/abs/2510.01290)
- Follow-ups (same line, 2026): BeaconKV (ICML 2026) — [OpenReview](https://openreview.net/forum?id=bgAbffQCrL); "Which Heads Matter for Reasoning? RL-Guided KV Cache Compression" (ICML 2026) — [OpenReview](https://openreview.net/forum?id=fBRIgbTvId); "Information-Aware KV Cache Compression for Long Reasoning" (ICLR 2027 submission, under review) — [OpenReview](https://openreview.net/forum?id=5AVwzeoj2P)

**D4. R-KV: Redundancy-aware KV Cache Compression for Reasoning Models — NeurIPS 2025 (poster)**
- Links: paper [arXiv 2505.24133](https://arxiv.org/abs/2505.24133), venue record [OpenReview](https://openreview.net/forum?id=2jwAjomEDB), code [Zefan-Cai/R-KV](https://github.com/Zefan-Cai/R-KV).
- Setup and result: DeepSeek-R1-Distill-Llama-8B and R1-Distill-Qwen-14B on MATH-500 and AIME 2024, A100 80 GB; about 100% of full-KV accuracy at 10% of the KV cache (prior baselines reach about 60%), 90% memory saving and 6.6x throughput; "The primary throughput improvement from R-KV stems from enabling significantly larger inference batch sizes"; at batch size 1 only "a slight throughput advantage over FullKV" — [arXiv](https://arxiv.org/abs/2505.24133)
- Stated limitations (closely paraphrased): not compatible with paged attention; serving frameworks without KV-compression interfaces require memory reallocation that can offset the gains — [arXiv](https://arxiv.org/abs/2505.24133)
- Reproducible on 4x48 GB (assessment): yes (8B and 14B models, single GPU); maximum batch sizes scale with 48 GB instead of 80 GB.
- Beaten by / compared in: ThinKV (D3) — [arXiv 2510.01290](https://arxiv.org/abs/2510.01290)

**D5. SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning — NeurIPS 2025 (poster)**
- Links: paper [arXiv 2504.07891](https://arxiv.org/abs/2504.07891), venue record [OpenReview](https://openreview.net/forum?id=wCbOKbZ7kf), code [ruipeterpan/specreason](https://github.com/ruipeterpan/specreason).
- Problem: reasoning-model latency is dominated by long thinking; token-level equivalence is not needed as long as the reasoning trajectory is preserved, so a small model speculates whole reasoning steps and the base model scores each step with a short prefill-only verification pass — [arXiv](https://arxiv.org/abs/2504.07891)
- Models: base QwQ-32B and Skywork-OR1-Preview-32B; small DeepSeek-R1-1.5B and Zyphra ZR1-1.5B; an extra R1-70B run on 4x A100 80 GB — [arXiv](https://arxiv.org/abs/2504.07891)
- Datasets: AIME, MATH500, GPQA — [arXiv](https://arxiv.org/abs/2504.07891)
- Hardware and framework: "two NVIDIA A6000-48GB GPUs. We use vLLM 0.8.2 ... enable prefix caching. Both models are served with a tensor parallelism degree of two." — [arXiv](https://arxiv.org/abs/2504.07891)
- Headline: 1.4-3.0x speedup over vanilla reasoning-model inference with accuracy changes of +0.4 to +9.0%; combining with token-level speculative decoding gives a further 8.8-58.0% latency reduction — [arXiv](https://arxiv.org/abs/2504.07891)
- Reproducible on 4x48 GB (assessment): yes — the paper's main testbed is 2x A6000 48 GB with vLLM.
- Stated limitations / future work: the fixed-score verification offers "only a coarse-grained configuration space"; "Future work can explore more sophisticated strategies, such as logprob-based confidence estimates or dynamic thresholds, to enable finer-grained tradeoffs without incurring additional runtime cost" — [arXiv](https://arxiv.org/abs/2504.07891)
- Follow-ups: Speculative Thinking (COLM 2025) — [OpenReview](https://openreview.net/forum?id=4Ns18bSoHo); "Scaling Up, Speeding Up" SD-for-test-time-scaling benchmark (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=DjOmnwX4wJ)

**D6. Shorter records in this line**
- Thinkless (NeurIPS 2025 poster): RL (DeGRPO) trains `<short>` / `<think>` control tokens on DeepSeek-R1-Distill-Qwen-1.5B using the DeepScaleR 40K math set; evaluated on Minerva Algebra, MATH-500, GSM8K; reduces long-chain thinking usage by 50%-90%; trained on a single node with 4 H100; the authors state the validation is on math data only — [arXiv 2505.13379](https://arxiv.org/abs/2505.13379), [OpenReview](https://openreview.net/forum?id=ariVQf0KZx), code [VainF/Thinkless](https://github.com/VainF/Thinkless)
- Controlling Thinking Speed in Reasoning Models (NeurIPS 2025 spotlight): steering vector for fast/slow thinking implemented on vLLM; R1-Distill-Qwen-7B / 32B and others on AIME24, MATH-500, GPQA Diamond, LiveCodeBench v2; max 32,768 tokens, temperature 0.6, top-p 0.95, 8 samples; A100 80 GB; +1.3% accuracy with 8.6% fewer tokens; stated limitations are a shared steering intensity across models and a heuristic sliding window — [arXiv 2507.03704](https://arxiv.org/abs/2507.03704), [OpenReview](https://openreview.net/forum?id=ESELaMThLN), code [D2I-ai/thinking-speed-control](https://github.com/D2I-ai/thinking-speed-control)
- Other 2026 records found in this line (not read in full): Efficient Reasoning with Balanced Thinking (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=cJseWJJ5IM); The Markovian Thinker (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=3As6AQ9ELI); Scalable Chain of Thoughts via Elastic Reasoning (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=E0Qfhma53J); Plan and Budget (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=ctspw4CqbS); PEAR (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=HLc2igXEA3); Parallel-Probe (ICML 2026) — [OpenReview](https://openreview.net/forum?id=d7TW4gTLcq); Re-FORC (ICML 2026) — [OpenReview](https://openreview.net/forum?id=8NNPSNAvTX)

#### E. Parallel / non-autoregressive decoding

**E1. Hogwild! Inference: Parallel LLM Generation via Concurrent Attention — NeurIPS 2025 (spotlight)**
- Links: paper [arXiv 2504.06261](https://arxiv.org/abs/2504.06261), venue record [OpenReview](https://openreview.net/forum?id=roKj4IwaVT), code [eqimp/hogwild_llm](https://github.com/eqimp/hogwild_llm).
- Problem: prior parallel-reasoning schemes fix the collaboration pattern (voting, sub-task splitting); here several instances of one model generate concurrently while attending to each other's partial output through a shared, concurrently updated KV cache, with no fine-tuning — [arXiv](https://arxiv.org/abs/2504.06261)
- Setup: QwQ-32B with 2, 3 and 4 workers; synthetic tasks built from 5 GSM8K questions each, and the LIMO set of 817 math problems; baselines include sequential generation with and without early stopping, and independent parallel attempts; inference-time measurements on an NVIDIA L40S with QwQ-32B-AWQ — [arXiv](https://arxiv.org/abs/2504.06261)
- Reproducible on 4x48 GB (assessment): yes — the timing testbed is a single 48 GB L40S with an AWQ 32B model.
- Stated limitations (quoted): "Our method exhibits reduced robustness when applied to smaller models or longer contexts ... our automatic evaluation metric relies on a proprietary model, which may limit reproducibility." Future work named: fine-tuning or RL for collaboration — [arXiv](https://arxiv.org/abs/2504.06261)
- Follow-ups: none confirmed in this session (parallel-decoding topic search was rate-limited).

**E2. Shorter records in this line**
- PASTA, "Learning to Keep a Promise" (ICML 2025 poster): the model is fine-tuned to emit annotations in an asynchronous-decoding language, which an interpreter executes in parallel; AlpacaEval; geometric-mean speedups 1.21x-1.93x with quality change from +2.2% to -7.1%; H100 GPUs; baselines APAR and Skeleton-of-Thought; no code link found in the paper — [arXiv 2502.11517](https://arxiv.org/abs/2502.11517), [OpenReview](https://openreview.net/forum?id=ZfX43ZZRZR)
- Multiverse (NeurIPS 2025 spotlight): MapReduce-style natively parallel generation; Multiverse-32B is Qwen2.5-32B-Instruct fine-tuned on 1K examples in 3 hours on 8 NVIDIA B200; AIME24 54%, AIME25 46%, also MATH500 and GPQA Diamond; the authors state that only SFT was used and RL is future work — [arXiv 2506.09991](https://arxiv.org/abs/2506.09991), [OpenReview](https://openreview.net/forum?id=r9YDEErKXU), code [Multiverse4FM/Multiverse](https://github.com/Multiverse4FM/Multiverse)

#### F. Diffusion language models (inference speed)

**F1. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding — ICLR 2026 (Poster)**
- Links: paper [arXiv 2505.22618](https://arxiv.org/abs/2505.22618), venue record [OpenReview](https://openreview.net/forum?id=3Z3Is6hnOT), code: github.com/NVlabs/Fast-dLLM (from memory, not found in the extracted text — unverified).
- Problem: open diffusion LLMs (LLaDA, Dream) are slower than autoregressive models because bidirectional attention prevents KV caching and parallel unmasking hurts quality; the paper adds a block-wise approximate KV cache and confidence-thresholded parallel decoding — [arXiv](https://arxiv.org/abs/2505.22618)
- Setup: "multiple open-source Diffusion LLMs (LLaDA, Dream) and four mainstream benchmarks (GSM8K, MATH, HumanEval, MBPP)"; "NVIDIA A100 GPU with a single batch size and no inference speedup frameworks"; metric tokens/s vs task accuracy — [arXiv](https://arxiv.org/abs/2505.22618)
- Headline: up to 27.6x throughput at generation length 1024 with minimal accuracy loss; 8.1x on GSM8K (5-shot) at length 256 from cache plus parallel decoding combined — [arXiv](https://arxiv.org/abs/2505.22618)
- Reproducible on 4x48 GB (assessment): yes — 7B-8B models, batch 1, single GPU.
- Stated limitations (closely paraphrased): at larger batch sizes the cached diffusion model struggles to match LLaMA, because diffusion LLMs "incur higher computational overhead due to full attention operations during decoding" — [arXiv](https://arxiv.org/abs/2505.22618)
- Follow-ups: Fast-dLLM v2 (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=1NZ3DHF9nT); Dynamic-dLLM (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=SdnkB5pGbq); SparseD (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=dwbrZtYP04); CDLM (MLSys 2026) — [OpenReview](https://openreview.net/forum?id=eB8yjR6alL); FlashBlock (ICML 2026) — [OpenReview](https://openreview.net/forum?id=4jfuNNghPS); Quant-dLLM (ICLR 2026 Poster) — [OpenReview](https://openreview.net/forum?id=HD7tuVakmR)

**F2. Shorter records in this line**
- dKV-Cache (NeurIPS 2025 poster): delayed KV caching for diffusion LMs, 2-10x inference speedup on language-understanding, math and code benchmarks; speed measured on an A6000 (LLaDA) and an H20 (Dream); the authors state they studied the algorithm in isolation and left system-level optimization — [arXiv 2505.15781](https://arxiv.org/abs/2505.15781), [OpenReview](https://openreview.net/forum?id=Gppo2JImHs), code [horseee/dKV-Cache](https://github.com/horseee/dKV-Cache)
- Prophet (ICLR 2026 Oral): early-commit decoding triggered by the top-2 confidence gap; LLaDA-8B and Dream-7B on MMLU, ARC-C, HellaSwag, TruthfulQA, GSM8K and others; up to 3.4x fewer decoding steps; future work named: a trained stopping discriminator and integration with KV-cache methods — [arXiv 2508.19982](https://arxiv.org/abs/2508.19982), [OpenReview](https://openreview.net/forum?id=g88nt4ieTG), code [pixeli99/Prophet](https://github.com/pixeli99/Prophet)

#### G. Efficient architectures (hybrids, MoE)

**G1. Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search — NeurIPS 2025 (poster)**
- Links: paper [arXiv 2508.15884](https://arxiv.org/abs/2508.15884), venue record [OpenReview](https://openreview.net/forum?id=WZQXaTNYEB), code [NVlabs/Jet-Nemotron](https://github.com/NVlabs/Jet-Nemotron).
- Problem: designing efficient LM architectures normally requires pre-training from scratch; PostNAS starts from a pre-trained full-attention model, freezes the MLPs, and searches attention-layer placement, the linear-attention block (JetBlock), and hardware-aware hyperparameters — [arXiv](https://arxiv.org/abs/2508.15884)
- Setup: Jet-Nemotron-2B / 4B vs Qwen2.5, Qwen3, Gemma3, Llama3.2 and other linear-attention / SSM models; MMLU(-Pro), math, commonsense, retrieval, coding and long-context tasks; throughput on an NVIDIA H100 (64K context for the headline comparison); "The models are then trained on 350B tokens. We summarize the experimental costs in Appendix A.2." — [arXiv](https://arxiv.org/abs/2508.15884)
- Headline: up to 53.6x generation throughput and 6.1x prefill speedup; 47x vs Qwen3-1.7B-Base at 64K context — [arXiv](https://arxiv.org/abs/2508.15884)
- Reproducible on 4x48 GB (assessment): scaled down — throughput and accuracy evaluation of the released 2B / 4B checkpoints fits one GPU; repeating PostNAS with hundreds of billions of training tokens does not fit this node in practical time.
- Stated limitations: no limitations section was located by keyword search of the arXiv text — [arXiv](https://arxiv.org/abs/2508.15884)
- Follow-ups: Jet-Long (NeurIPS 2026 per OpenReview venue string) — [OpenReview](https://openreview.net/forum?id=oyNkgcQ2fZ)

**G2. The Mamba in the Llama: Distilling and Accelerating Hybrid Models — NeurIPS 2024 (poster)**
- Links: paper [arXiv 2408.15237](https://arxiv.org/abs/2408.15237), venue record [OpenReview](https://openreview.net/forum?id=uAzhODjALU), code [jxiw/MambaInLlama](https://github.com/jxiw/MambaInLlama).
- Setup and result: distills Llama-3-8B / Zephyr into hybrids keeping 50%, 25%, 12.5% or 0% of attention layers, and adds a hardware-aware speculative decoding algorithm for Mamba; distillation takes under 5 days on 8x A100 80 GB; the paper notes naive speculative decoding gave 1.5x on an RTX 3090 but no gain on an H100 until optimized — [arXiv](https://arxiv.org/abs/2408.15237)
- Reproducible on 4x48 GB (assessment): scaled down — inference with released checkpoints fits one GPU; the distillation run was 8x A100 for days.

**G3. MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs — ASPLOS 2025 (venue unverified)**
- Links: paper [arXiv 2411.11217](https://arxiv.org/abs/2411.11217); code: no repository link found in the paper text (unverified).
- Problem: "the large model size makes MoE models inaccessible to individuals without high-end GPUs"; the paper is a batch-inference system with a CPU-GPU-I/O pipelining schedule (CGOPipe) with paged weights and a Hierarchical Roofline Model for choosing the policy — [arXiv](https://arxiv.org/abs/2411.11217)
- Setup: Mixtral 8x7B on a single T4 16 GB or L4, Mixtral 8x22B and DBRX on 2-4 T4s; prior offloading system FlexGen is the main point of comparison (the paper notes FlexGen's data fitting "might take hours or days"); disk offloading is not considered — [arXiv](https://arxiv.org/abs/2411.11217)
- Headline: up to 10.3x higher throughput than prior offloading systems for Mixtral 8x7B on a single T4 — [arXiv](https://arxiv.org/abs/2411.11217)
- Reproducible on 4x48 GB (assessment): yes — the paper targets GPUs smaller than 48 GB; a memory cap on an A6000 would be needed to recreate the memory-constrained regime, and absolute numbers would differ from T4 / L4.
- Stated limitations / future work: the performance model covers a single node without GPU-GPU communication; disk offloading and KV sparsity are left out — [arXiv](https://arxiv.org/abs/2411.11217)
- Note: this paper sits close to the "offload systems" boundary of the scope; it is included because the scope lists MoE inference where throughput is measured.

### Inferences
- Papers whose reported testbed already matches the student's hardware class (48 GB cards, 1-4 of them): SWIFT (4x A6000), SpecReason (2x A6000), LServe and QServe (single L40S as second testbed), Hogwild! (single L40S for timing), dKV-Cache (A6000 for LLaDA).
- Papers whose headline numbers depend on hardware the student lacks: Judge Decoding (8x H100 for 405B), HSD (96 GB H20 for an 8-bit 72B target), MInference and ShadowKV (1M-token contexts and batch scaling on 80 GB A100), SuffixDecoding end-to-end (4x H100), Multiverse and Jet-Nemotron and Mamba-in-the-Llama training (8x B200 / large-scale pre-training / 8x A100).
- Lossless speculative-decoding papers (EAGLE-3, SWIFT, HSD, SuffixDecoding) report no quality metric at all, only speed and acceptance statistics; relaxed-verification, pruning, quantization and early-exit papers each need an accuracy benchmark alongside the speed number, which roughly doubles the evaluation surface to reproduce.
- Several papers report speed without any serving framework (Fast-dLLM "no inference speedup frameworks"; SWIFT, KIVI, SnapKV on HuggingFace transformers at batch 1), while others report inside vLLM / SGLang / TensorRT-LLM (DEER, SpecReason, Certaindex, LServe, QServe, EAGLE-3's batched tables); numbers across these two groups are not comparable.
- Papers with no public code found (Judge Decoding, ThinKV, PASTA, MoE-Lightning) would require re-implementation before reproduction.

### Gaps
- Reviewer-raised weaknesses: not obtained for any paper. OpenReview forum and PDF endpoints returned a bot challenge (HTTP 403) and I did not work around it.
- Code links marked "from memory" (MInference, Fast-dLLM) were not re-checked; "no repository link found" (Judge Decoding, ThinKV, PASTA, MoE-Lightning) means only that the arXiv text has none.
- EAGLE-3: the GPU model used for the main speedup table and the draft-head training cost were not found in the text.
- Hogwild! Inference: headline speed/accuracy numbers were not extracted (results are presented as figures).
- KVzip, LServe, QServe, Jet-Nemotron: exact per-benchmark tables, dataset links, and (for Jet-Nemotron) the Appendix A.2 training cost were not extracted.
- FlexPrefill: a single headline speedup number was not found in the text; the result is given as a performance-vs-latency curve against MInference.
- Dataset/benchmark URLs were not collected; benchmarks are named as the papers name them.
- Follow-up lists come from OpenReview title/abstract search; I did not open those papers, so "builds on or beats" is verified only where a quoted claim is given (HSD over EAGLE-3, FlatQuant over SpinQuant, SuffixDecoding vs EAGLE-3, ThinKV vs R-KV as baseline).
- Early-exit / layer-skipping for standard (non-reasoning) LLMs in 2025-2026 main tracks: no verified paper beyond SWIFT, Kangaroo and LayerSkip (topic search rate-limited).

## Key Question 3: Which standard benchmarks and measurement protocols do these lines share?

### Takeaway
Each line has a de-facto protocol: Spec-Bench-style tasks at batch 1 with speedup ratio and accepted-tokens-per-step for speculative decoding; RULER / InfiniteBench / LongBench / Needle-in-a-Haystack on 7B-9B long-context models for sparse attention and KV pruning; WikiText-2 / C4 perplexity plus zero-shot QA with WxAyKVz notation for quantization; MATH-500 / AIME / GPQA Diamond / LiveCodeBench on DeepSeek-R1-Distill and QwQ-32B for reasoning efficiency; LLaDA / Dream on GSM8K-type tasks at batch 1 for diffusion LMs. The protocols differ sharply in whether a serving framework and batching are involved.

### Cited Findings

Speculative decoding
- Spec-Bench protocol: Vicuna-7B-v1.3 in FP16 on a single RTX 3090 (24 GB), batch size 1, mean speedup over 3 runs, PyTorch 2.0.1, with an additional A100 80 GB analysis; six methods compared (SpS, Medusa, EAGLE, Lookahead, PLD, REST); EAGLE reaches 1.8x-2.4x — [arXiv 2401.07851](https://arxiv.org/abs/2401.07851)
- Spec-Bench is used directly by Kangaroo (Vicuna 7B/13B) — [arXiv 2404.18911](https://arxiv.org/abs/2404.18911); by SuffixDecoding as its non-agentic stress test (it describes 8 MT-Bench categories among the Spec-Bench tasks) — [arXiv 2411.04975](https://arxiv.org/abs/2411.04975); and EAGLE-3 states its five tasks follow "EAGLE and Spec-Bench" (MT-bench, HumanEval, GSM8K, Alpaca, CNN/Daily Mail) — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- Shared metrics: speedup ratio vs vanilla autoregressive decoding and average acceptance length tau (EAGLE-3) — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840); mean generated length M per target forward pass and acceptance rate alpha (SWIFT) — [arXiv 2410.06916](https://arxiv.org/abs/2410.06916); Block Efficiency (tokens per serial target call, "independent of hardware") and Decoding Speed in tokens/s (HSD) — [arXiv 2601.05724](https://arxiv.org/abs/2601.05724)
- Shared task trio outside Spec-Bench: GSM8K, HumanEval and CNN/DailyMail appear in SWIFT, HSD, EAGLE-3 and Judge Decoding (GSM8K, HumanEval, MT-Bench) — [SWIFT](https://arxiv.org/abs/2410.06916), [HSD](https://arxiv.org/abs/2601.05724), [EAGLE-3](https://arxiv.org/abs/2503.01840), [Judge Decoding](https://arxiv.org/abs/2501.19309)
- Lossless methods skip quality evaluation: EAGLE-3 "uses strict speculative sampling acceptance conditions, ensuring no loss in performance. Therefore, we do not evaluate generation quality." — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- Batch size 1 with HuggingFace transformers is the default in SWIFT and Spec-Bench; batched results in a serving engine appear only as an extra study in EAGLE-3 (SGLang v0.4.4 on H100, vLLM) — [SWIFT](https://arxiv.org/abs/2410.06916), [Spec-Bench](https://arxiv.org/abs/2401.07851), [EAGLE-3](https://arxiv.org/abs/2503.01840)

Long-context sparse attention and KV pruning
- RULER and InfiniteBench are the accuracy benchmarks of both MInference and FlexPrefill; Needle In A Haystack is used by MInference, LServe and ShadowKV; LongBench by LServe, ShadowKV, SnapKV and KIVI — [MInference](https://arxiv.org/abs/2407.02490), [FlexPrefill](https://arxiv.org/abs/2502.20766), [LServe](https://arxiv.org/abs/2502.14866), [ShadowKV](https://arxiv.org/abs/2410.21465), [SnapKV](https://arxiv.org/abs/2404.14469), [KIVI](https://arxiv.org/abs/2402.02750)
- The same long-context model set recurs: Llama-3(.1)-8B long-context variants, GLM-4-9B-1M, Yi-9B-200K, Qwen2-7B-128K, Phi-3-Mini-128K — [MInference](https://arxiv.org/abs/2407.02490), [FlexPrefill](https://arxiv.org/abs/2502.20766), [ShadowKV](https://arxiv.org/abs/2410.21465)
- The standard hardware is a single A100 80 GB, with custom PyTorch / Triton kernels built on FlashAttention as the dense baseline — [MInference](https://arxiv.org/abs/2407.02490), [FlexPrefill](https://arxiv.org/abs/2502.20766)
- SCBench protocol: 12 tasks under two shared-context modes (multi-turn and multi-request), analysing methods by four KV-cache stages (generation, compression, retrieval, loading); models include Llama-3.1-8B/70B, Qwen2.5-72B/32B, Llama-3-8B-262K, GLM-4-9B, Codestral Mamba and Jamba-1.5-Mini; HuggingFace or vLLM with FlashAttention-2 — [arXiv 2412.10319](https://arxiv.org/abs/2412.10319)
- KVzip evaluates in the multi-query (query-agnostic) setting that SCBench formalised, against query-aware eviction baselines — [arXiv 2505.23416](https://arxiv.org/abs/2505.23416)

Quantization
- Accuracy protocol: WikiText-2 and C4 perplexity plus zero-shot QA tasks, on LLaMA-2 7B/13B/70B and LLaMA-3 8B/70B, with bit-widths written as W4A4 / W4A4KV4 / W4A8KV4 — [FlatQuant](https://arxiv.org/abs/2410.09426), [SpinQuant](https://arxiv.org/abs/2405.16406), [QServe](https://arxiv.org/abs/2405.04532)
- Calibration protocol example: SpinQuant optimises rotations on 800 WikiText-2 samples for 100 iterations — [arXiv 2405.16406](https://arxiv.org/abs/2405.16406)
- Speed protocol varies: FlatQuant reports prefill/decoding speedup vs FP16 on an RTX 3090; QServe reports serving throughput vs TensorRT-LLM on A100 and L40S; SpinQuant's speed test is limited to an FP8 LLaMA-3 70B on H100 — [FlatQuant](https://arxiv.org/abs/2410.09426), [QServe](https://arxiv.org/abs/2405.04532), [SpinQuant](https://arxiv.org/abs/2405.16406)
- KV quantization protocol (KIVI): LM-Eval tasks and LongBench in HuggingFace Transformers on one A100 80 GB, reporting maximum batch size and throughput — [arXiv 2402.02750](https://arxiv.org/abs/2402.02750)

Reasoning-model efficiency
- Shared benchmarks: MATH-500 and AIME appear in DEER, ThinKV, R-KV, SpecReason, Controlling Thinking Speed and Multiverse; GPQA Diamond in DEER, SpecReason, Controlling Thinking Speed and Multiverse; LiveCodeBench in DEER, ThinKV, Certaindex and Controlling Thinking Speed; GSM8K in DEER, ThinKV, Certaindex and Thinkless — [DEER](https://arxiv.org/abs/2504.15895), [ThinKV](https://arxiv.org/abs/2510.01290), [R-KV](https://arxiv.org/abs/2505.24133), [SpecReason](https://arxiv.org/abs/2504.07891), [Thinking Speed](https://arxiv.org/abs/2507.03704), [Multiverse](https://arxiv.org/abs/2506.09991), [Certaindex](https://arxiv.org/abs/2412.20993), [Thinkless](https://arxiv.org/abs/2505.13379)
- Shared models: DeepSeek-R1-Distill (Qwen 1.5B-32B, Llama 8B/70B) and QwQ-32B — [DEER](https://arxiv.org/abs/2504.15895), [ThinKV](https://arxiv.org/abs/2510.01290), [R-KV](https://arxiv.org/abs/2505.24133), [SpecReason](https://arxiv.org/abs/2504.07891), [Hogwild!](https://arxiv.org/abs/2504.06261)
- Sampling protocol: maximum generation 32K tokens with 8 responses per question (ThinKV); max 32,768 tokens, temperature 0.6, top-p 0.95, 8 samples (Controlling Thinking Speed); 4 sampling rounds per instance on AMC / AIME (DEER) — [ThinKV](https://arxiv.org/abs/2510.01290), [Thinking Speed](https://arxiv.org/abs/2507.03704), [DEER](https://arxiv.org/abs/2504.15895)
- Metric pairs: accuracy with chain-of-thought length / token count (DEER, Thinking Speed, Thinkless); accuracy with latency speedup (SpecReason); accuracy with KV budget, memory saving and batch throughput (R-KV, ThinKV); compute saved and throughput at equal accuracy (Certaindex) — [DEER](https://arxiv.org/abs/2504.15895), [SpecReason](https://arxiv.org/abs/2504.07891), [R-KV](https://arxiv.org/abs/2505.24133), [ThinKV](https://arxiv.org/abs/2510.01290), [Certaindex](https://arxiv.org/abs/2412.20993)
- Framework: vLLM is the stated engine for DEER, SpecReason (v0.8.2, prefix caching on) and Controlling Thinking Speed; SGLang for Certaindex — [DEER](https://arxiv.org/abs/2504.15895), [SpecReason](https://arxiv.org/abs/2504.07891), [Thinking Speed](https://arxiv.org/abs/2507.03704), [Certaindex](https://arxiv.org/abs/2412.20993)

Diffusion language models
- Shared models LLaDA (8B) and Dream (7B); Fast-dLLM uses GSM8K, MATH, HumanEval, MBPP and reports tokens/s on one A100 at batch 1 with "no inference speedup frameworks"; Prophet reports decoding-step reduction on MMLU, ARC-C, HellaSwag, TruthfulQA, GSM8K; dKV-Cache reports speedup on language-understanding, math and code benchmarks — [Fast-dLLM](https://arxiv.org/abs/2505.22618), [Prophet](https://arxiv.org/abs/2508.19982), [dKV-Cache](https://arxiv.org/abs/2505.15781)

Parallel decoding
- No common benchmark across the three papers found: Hogwild! uses LIMO (817 problems) and synthetic GSM8K bundles; PASTA uses AlpacaEval with length-controlled win rate against APAR and Skeleton-of-Thought; Multiverse uses AIME24/25, MATH500, GPQA Diamond — [Hogwild!](https://arxiv.org/abs/2504.06261), [PASTA](https://arxiv.org/abs/2502.11517), [Multiverse](https://arxiv.org/abs/2506.09991)

### Inferences
- GSM8K and HumanEval are the two tasks that cross the most lines (speculative decoding, reasoning efficiency, diffusion LMs), though used for speed in the first and for accuracy in the other two.
- The speculative-decoding line and the diffusion-LM line both default to batch size 1 without a serving engine; the reasoning-efficiency and serving-quantization lines default to vLLM / SGLang / TensorRT-LLM with batching. A reproduction therefore has to pick the same regime as the paper it follows to obtain comparable numbers.
- A single A100 80 GB is the modal testbed for long-context and KV work; on a 48 GB card the same protocols apply at shorter maximum context or smaller batch.

### Gaps
- I did not verify benchmark versions (e.g. LiveCodeBench release, AIME year mix in ThinKV) or collect benchmark URLs.
- The 2026 benchmark papers "Scaling Up, Speeding Up" (speculative decoding for test-time scaling) and "Long-Context Attention Benchmark" were found by title only; their protocols were not read — [OpenReview](https://openreview.net/forum?id=DjOmnwX4wJ), [OpenReview](https://openreview.net/forum?id=W7sVYFJAEp)
- No shared latency-measurement convention (warm-up, number of runs, TTFT vs end-to-end) was found stated across lines; only Spec-Bench states "mean over 3 runs".
- No common benchmark was identified for MoE inference or hybrid-architecture throughput beyond per-paper choices.

## Key Question 4: Which problems do several papers name as still open?

### Takeaway
The recurring author-stated open items are: making speculative decoding pay off under batching and inside production engines; the missing quality guarantee once verification is relaxed; integration of KV compression / sparse attention with paged-attention serving stacks and the overhead of building indices or compressed caches; robustness of KV pruning under multi-turn reuse; and generalisation of reasoning-efficiency methods beyond math benchmarks and across models. Diffusion-LM papers name batched, system-level efficiency as open.

### Cited Findings

Speculative decoding under batching and in serving engines
- Spec-Bench lists as an open challenge "How to apply Speculative Decoding in batched inference scenarios?" and the trade-off between speculation accuracy and drafting efficiency — [arXiv 2401.07851](https://arxiv.org/abs/2401.07851)
- EAGLE-3 reports that in SGLang on H100, EAGLE falls below the no-speculation baseline from batch size 24 and EAGLE-3's gain shrinks to 1.38x at batch 64 (chain drafting only, no tree) — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- The Mamba in the Llama reports that naive speculative decoding gave 1.5x on an RTX 3090 but no speedup on an H100 until a hardware-aware implementation was written — [arXiv 2408.15237](https://arxiv.org/abs/2408.15237)
- Certaindex states that integration with "PD disaggregation or chunked prefill ... remains an open area for future research" — [arXiv 2412.20993](https://arxiv.org/abs/2412.20993)

Quality guarantees when verification or acceptance is relaxed
- Judge Decoding: "An obvious limitation is the loss of the mathematical guarantee to match target quality ... there is no certainty for novel tasks", and new tasks "require careful annotation of data" — [arXiv 2501.19309](https://arxiv.org/abs/2501.19309)
- SpecReason: its score-threshold verification is "coarse-grained"; logprob-based confidence or dynamic thresholds are left as future work — [arXiv 2504.07891](https://arxiv.org/abs/2504.07891)
- KVzip: "does not include theoretical guarantees concerning compression-induced information loss" — [arXiv 2505.23416](https://arxiv.org/abs/2505.23416)

Draft quality and model-size dependence
- Judge Decoding: small or self-speculative drafters "quickly deteriorate when drafting too far into the future", and "Speedups for smaller models such as Llama-8B are hence tougher to achieve" — [arXiv 2501.19309](https://arxiv.org/abs/2501.19309)
- SWIFT: larger LLMs show more layer sparsity; scaling to larger models is left "for future investigation" — [arXiv 2410.06916](https://arxiv.org/abs/2410.06916)
- Hogwild! Inference: "reduced robustness when applied to smaller models or longer contexts" — [arXiv 2504.06261](https://arxiv.org/abs/2504.06261)
- SuffixDecoding: on non-agentic Spec-Bench it is outperformed by model-based drafters, and EAGLE-2/3 fail on SWE-Bench "due to their maximum sequence length limitations" — [arXiv 2411.04975](https://arxiv.org/abs/2411.04975)

Overhead of sparse-index / compression construction
- MInference: at 10k-token contexts index-building overhead grows from 5% to 30% of time, so latency approaches dense FlashAttention; higher sparsity "may noticeably" reduce accuracy — [arXiv 2407.02490](https://arxiv.org/abs/2407.02490)
- KVzip: compression overhead of about 2x a standard prefill, "amortized over multiple queries" — [arXiv 2505.23416](https://arxiv.org/abs/2505.23416)
- KIVI: reducing quantization overhead in prefill and decoding is named as future work — [arXiv 2402.02750](https://arxiv.org/abs/2402.02750)
- QServe: existing INT4 schemes pay dequantization overhead on cloud GPUs because "a CUDA core operation is as expensive as 50 INT4 tensor core operations" — [arXiv 2405.04532](https://arxiv.org/abs/2405.04532)

Integration of KV compression with serving frameworks
- R-KV: not compatible with paged attention; frameworks without KV-compression interfaces need memory reallocation that can offset gains; at batch size 1 the throughput advantage is slight — [arXiv 2505.24133](https://arxiv.org/abs/2505.24133)
- ThinKV builds a kernel extending PagedAttention to address this, and states its method "is not directly applicable to settings dominated by long input contexts" — [arXiv 2510.01290](https://arxiv.org/abs/2510.01290)
- dKV-Cache: algorithm studied in isolation, system-level optimization left for later — [arXiv 2505.15781](https://arxiv.org/abs/2505.15781)

KV pruning under shared-context reuse
- SCBench finds that sub-O(n)-memory methods do well single-turn but degrade in multi-turn use, while O(n) methods hold up in multi-request settings — [arXiv 2412.10319](https://arxiv.org/abs/2412.10319)
- ShadowKV states eviction strategies "often result in information loss and accuracy degradation in tasks such as multi-turn conversations" — [arXiv 2410.21465](https://arxiv.org/abs/2410.21465)
- KVzip positions query-agnostic compression as the response to query-aware eviction failing across queries — [arXiv 2505.23416](https://arxiv.org/abs/2505.23416)

Generalisation of reasoning-efficiency methods
- Thinkless: validated on math datasets only, with a simple SFT warm-up — [arXiv 2505.13379](https://arxiv.org/abs/2505.13379)
- DEER: models that are under-confident on trial answers (Llama-3.1-Nemotron-Nano-8B-v1, 55% early-stop rate) gain little — [arXiv 2504.15895](https://arxiv.org/abs/2504.15895)
- Controlling Thinking Speed: one steering intensity for all models and a heuristic sliding window — [arXiv 2507.03704](https://arxiv.org/abs/2507.03704)
- Multiverse: only SFT explored, RL left for future work; use beyond LLM reasoning underexplored — [arXiv 2506.09991](https://arxiv.org/abs/2506.09991)

Calibration data, hyperparameters and number formats in quantization and layer skipping
- FlatQuant: "the optimal selection of calibration sets remains an open question"; integration "with newer data types, such as MXFP4" not examined; "the full potential of 4-bit quantization has not been thoroughly explored" — [arXiv 2410.09426](https://arxiv.org/abs/2410.09426)
- LayerSkip: requires finetuning or pretraining with its recipe and tuning of p_max, e_scale and R — [arXiv 2404.16710](https://arxiv.org/abs/2404.16710)

Diffusion LMs at batch and system level
- Fast-dLLM: at larger batch sizes the cached diffusion model "struggles to match LLaMA", since diffusion LLMs "incur higher computational overhead due to full attention operations during decoding" — [arXiv 2505.22618](https://arxiv.org/abs/2505.22618)
- Prophet names integration with KV-cache acceleration as future work — [arXiv 2508.19982](https://arxiv.org/abs/2508.19982)

Reproducibility of evaluation
- Hogwild! Inference: "our automatic evaluation metric relies on a proprietary model, which may limit reproducibility" — [arXiv 2504.06261](https://arxiv.org/abs/2504.06261)

### Inferences
- "Works at batch 1 in a research codebase, unclear inside a batched serving engine" is stated in some form in four separate lines: speculative decoding (Spec-Bench, EAGLE-3, Mamba in the Llama), KV compression for reasoning (R-KV), diffusion LMs (Fast-dLLM, dKV-Cache), and reasoning schedulers (Certaindex).
- Two papers in this pool present themselves as answers to an open item stated by an earlier one: ThinKV's PagedAttention-extending kernel relative to R-KV's paged-attention incompatibility, and KVzip's query-agnostic scoring relative to the multi-turn degradation reported by SCBench.
- Papers at ICLR 2025 orals (FlexPrefill) and MLSys 2025 (LServe, QServe) state no limitations in the arXiv text, so their open problems can only be read from later papers that compare against them.

### Gaps
- Reviewer discussions, which often state open problems more bluntly than the authors, were not accessible (OpenReview bot challenge).
- No limitations text was found for EAGLE-3, HSD, FlexPrefill, LServe, ShadowKV, QServe and Jet-Nemotron, so those papers contribute nothing to this section.
- Open problems stated in 2026 follow-up papers (ICML 2026 / ICLR 2026 records listed under Key Question 2) were not read; the list above reflects 2024-2025 papers plus ThinKV, DEER, HSD, Fast-dLLM and Prophet.
- No survey or position paper was consulted to cross-check which open problems the community regards as shared; this section aggregates per-paper statements only.
