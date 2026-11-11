# Harvest: accepted papers whose contribution is answering a question

Lens: 2024-2026 papers in ML systems / efficient LLM inference whose main contribution is a
measurement, characterization, negative result, reproduction or "rethinking" study.
Written 2026-11-11. Working notes, not a ranking.

Marking used below:

- **V** = I downloaded the arXiv HTML and found the quoted text or number myself with grep.
- **F** = returned by a page fetch through a summarizing model; the primary source was opened, but the wording may be paraphrased.
- **S** = seen only in a search-result summary; the primary source was not opened.

## What shape of question got accepted, and at what hardware scale

- **"Does the reported gain survive the metric a deployer uses?"** A published technique class is re-measured under batching, throughput and end-to-end response length instead of per-token speed. Examples: Rethinking KV compression (MLSys 2025, 4x A6000), Speculative Decoding: Performance or Illusion? (MLSys 2026, H100), RLLM serving study (ICLR 2026, 8x A6000 / 8x 4090 / 8x A100-40GB).
- **"What does this new workload look like on an unmodified open engine?"** Reasoning models and agents are run on stock vLLM and the paper reports where time, memory and cache hits go. Examples: Agentic AI Workload Characteristics (IISWC 2026, 2x H100), The Cost of Dynamic Reasoning (HPCA 2026, 1x A100-40GB and 8x A100). The trace papers of this shape (ServeGen NSDI 2026, Copilot in the Wild) need production access and are not reproducible on one node.
- **"A widely used accounting assumption is wrong; here is the threshold under a different cost model."** These are mostly analytical with small measurements. Examples: Kinetics (NeurIPS 2025; cost model on B200 specs, about 14B threshold), Not All Bits Are Equal (ICLR 2026; 1,700 configurations, latency on 1x A100-80GB, 8-bit 4B threshold).
- **"A hidden variable changes the accuracy everyone reports."** Evaluation-validity studies. Examples: numerical nondeterminism (NeurIPS 2025; L40S and A100, 2 and 4 GPUs, 7B-8B models), A Sober Look (five clusters, 1.5B-7B models), Hold Onto That Thought (1x A6000).
- **Scale and what carried the paper.** Outside the trace papers, every study here used 1 to 8 GPUs of A6000 / A100 / L40S / H100 class, open models of 7B to 32B (one goes to 70B), vLLM / SGLang / LMDeploy, and public benchmarks. What they have in place of hardware is a wide grid (models x methods x budgets x benchmarks), one headline number that contradicts a common belief, and usually one unexplained anomaly that the authors explicitly leave open. Those leftover anomalies are the source of the cards below.

## Question papers (16)

### 1. Kinetics: Rethinking Test-Time Scaling Laws (NeurIPS 2025)
https://arxiv.org/abs/2506.05333

- Question: do test-time scaling laws that count tokens or FLOPs rank small and large models correctly once memory access is counted?
- How: Qwen3 0.6B-32B and DeepSeek-R1 distills on AIME24/25 and LiveCodeBench; an analytical cost ("eFLOPs") using B200 specs; block top-k throughput on H200 8-GPU nodes. Not an end-to-end serving measurement (V: "Rather than constructing a full end-to-end serving system, we estimate the overall model execution time using per-layer latency and throughput measurements").
- Finding (V): "computational resources are best spent first on increasing model size up to a critical threshold (empirically, around 14B parameters) before investing in test-time strategies"; "prior scaling laws consistently overestimate the effectiveness of small models enhanced with inference-time strategies"; "in the regime of long CoTs, where the generation length exceeds 4096 tokens, the cost of attention surpasses that of model parameters by a factor of 10-1000×".
- Assumption behind it (V): Figure 3 is "Latency breakdown for different model sizes (Qwen3 series) and context lengths (batch size 4K)"; "The access to the model parameters will be amortized across requests in a batch; Figure 3 shows parameter access time is negligible when the batch size is large. Thus, we only consider the second term (i.e., KV cache loading) in our cost function."
- Left open (V): "our cost analysis assumes a cloud-based serving environment, where computational resources are typically sufficient and large batch sizes are feasible. In contrast, local deployment scenarios ... often face limited VRAM where access to model parameters can dominate inference costs."; "the effects of training and post-training strategies are not fully explored and may influence the performance gaps and robustness to sparse attention mechanisms". (F: critical size is 7B for the DeepSeek series.)

### 2. Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models (ICLR 2026; venue from search listing)
https://arxiv.org/abs/2510.10964

- Question: under a fixed memory budget, should bits go to weights or to KV cache (more thinking)?
- How: Qwen3 0.6B-32B, AIME25 and GPQA-Diamond, 1,700 configurations (GPTQ 4/8-bit weights, R-KV eviction, HQQ KV quantization). V: "All measurements are performed on a single NVIDIA A100 80 GB GPU using the vLLM framework".
- Finding (V): "for models with effective size (parameters × bits per weight) below 8-bit 4B (≈4.2 GB), allocating more memory to model weights yields larger gains, whereas above this threshold, memory is better spent increasing the test-time budget". Also V: "a Qwen3-4B model with 4-bit weights occupies 2.49 GB, but its KV cache for a 32k-token generation requires 4.42 GB".
- Left open (V): "in practice, model weights can be amortized across multiple concurrent generations, fundamentally changing the memory dynamics"; "the inflection point where extra KV cache beats extra model weight may change as models become more sophisticated"; "We evaluate on the Qwen3 family, chosen for its broad size range and fixed architecture".
- Not reconciled: the abstract says 8-bit 4B; Section 5 headings (F) say 8-bit 8B for eviction versus quantization.

### 3. Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference (NeurIPS 2025, oral per neurips.cc listing in search; v1 title "Give Me FP32 or Give Me Death?")
https://arxiv.org/abs/2506.09501

- Question: how much do batch size, GPU count and GPU type change LLM outputs and benchmark scores through floating-point effects?
- How: DeepSeek-R1-Distill-Qwen-7B / Llama-8B, Qwen2.5-7B, Llama-3.1-8B; AIME'24, MATH500, LiveCodeBench; L40S and A100, 2 and 4 GPUs, batch 8/16/32, vLLM.
- Finding (V): "under bfloat16 precision with greedy decoding, a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size".
- Claim for sampling (V): "numerical precision introduces an additional source of variance beyond the intended randomness, and lower-precision formats such as BF16 tend to produce higher output variance"; "Researchers using random sampling with BF16 may need substantially more runs to achieve the same statistical confidence as with higher-precision formats".
- Weak point (V): Table 5's values "reflect the variability of Pass@1 performance across 6 different system configurations (3 batch sizes × 2 GPU counts), not across repeated runs of the same configuration." Table 5, DeepSeek-R1-Distill-Qwen-7B, AIME'24 n=64: BF16 0.3749, FP16 0.5391, FP32 0.7377 (FP32 is the most variable). The authors: "We interpret this as a result of dataset size and sampling dynamics rather than a contradiction of the overall trend."
- Left open: no within-configuration noise baseline for the sampling claim; no test of whether the mean or the length distribution shifts.
- Internal inconsistency flagged by the fetcher (F): 2.2% versus 5.80% of outputs diverging under FP32.

### 4. Hold Onto That Thought: Assessing KV Cache Compression On Reasoning (listed in a NeurIPS 2025 virtual session, probably a workshop; track not confirmed)
https://arxiv.org/abs/2512.12008

- Question: do KV eviction methods tuned on long-prompt tasks hold up on long reasoning generation?
- How: Llama-3.1-8B, R1-Distill-Qwen-7B/14B, R1-Distill-Llama-8B, Nemotron-Nano-8B; 8 benchmarks incl. GSM8K and MATH-500; budgets 128-512; V: "Each model is allowed to generate a maximum of 2048 new tokens via greedy decoding"; V: "Accuracy benchmarks were performed on an HPC cluster using an NVIDIA RTX A6000 48GB GPU."
- Finding (V): "eviction strategies at low budgets can produce longer reasoning traces, revealing a tradeoff between cache size and inference costs"; "KNorm, arguably the lowest performing strategy, tends to cause the greatest elongation of outputs."; "H2O results in significantly longer reasoning traces than SnapKV-D, which occasionally do not terminate".
- Left open: no explanation of the elongation; nothing above 14B; no batched serving; no limitations section. The text says two seeds in one place and three in another.

### 5. Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving (MLSys 2025)
https://arxiv.org/abs/2503.24000

- Question: do KV compression methods deliver in a production-style engine?
- How: LMDeploy, LLaMA-7B and LLaMA-3.1-8B, ShareGPT and LongBench; KIVI, GEAR, H2O, StreamingLLM. V: "The majority of the experiments are conducted on A6000 GPUs" (4x A6000).
- Finding (V): "more than 20% of the samples show at least a 1.5× increase in response length"; Table 4 length increase 1.69 (KIVI-4), 1.70 (GEAR-4), 1.55 (H2O-512), 1.76 (Stream-512).
- Untested speculation left by the authors (V): "The lengthy responses may serve as an implicit way for KV cache compression to compensate for accuracy loss."
- F: generation capped at 1024 tokens; H2O decode speedup 0.69x at TP=2. The fetcher flagged a mismatch between the 20% claim in the text and Table 5.

### 6. Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models (arXiv 2504.04823; venue not confirmed)
https://arxiv.org/abs/2504.04823

- Question: does weight / activation / KV quantization hurt reasoning models more than ordinary LLMs?
- How: DeepSeek-R1 distills 1.5B-70B, QwQ-32B, Qwen3-8B; AIME-120, MATH-500, GSM8K, GPQA, LiveCodeBench; vLLM backend.
- Finding (V): "for various quantization configurations (e.g., W4G128, W8A8KV8, and KV4), the generated output lengths closely align with those of the BF16 models. However, under more aggressive low-bit quantization (e.g., W3G128 and W4A4KV4), LLMs tend to produce longer outputs. This effect is particularly pronounced in smaller models like DeepSeek-R1-Distill-Qwen-1.5B".
- Left open (V): "the underlying mechanisms driving these effects remain incompletely understood. Future work will focus on elucidating these causal relationships"; "We speculate that it is due to the quantization error accumulated along the sequence, which hinders effective reasoning."

### 7. Reasoning Language Model Inference Serving Unveiled: An Empirical Study (ICLR 2026)
https://arxiv.org/abs/2510.18672

- Question: do standard serving optimizations behave the same for reasoning models as for ordinary LLMs?
- How: vLLM 0.8.1 and SGLang 0.4.6.post1; three 8-GPU servers (V: "a server with 8 RTX A6000 GPUs with 48GB VRAM", plus RTX 4090 and A100-PCIE-40GB); DeepSeek-R1 distills 7B-70B versus Qwen2.5; GSM8K, MATH-500, AIME-2024, GPQA.
- Finding (V): "while the 14B or 32B RLLM maintained performance with minimal degradation after KV cache quantization, the 7B RLLM experienced almost complete performance deterioration"; "speculative decoding significantly reduces throughput and degrades the TTFVT metric".
- Left open (V): "surprisingly, some of these techniques have the opposite effect on smaller RLLMs, e.g., 7B. We leave the investigation of this phenomenon to future."

### 8. Speculative Decoding: Performance or Illusion? (MLSys 2026)
https://arxiv.org/abs/2601.11580

- Question: do speculative decoding speedups hold in a production engine at realistic batch sizes?
- How: vLLM v0.10.1.1 on H100 80 GB; Llama3.1-8B, Llama3-70B, Qwen3-8B(-Thinking), GLM-4.5-Air; includes AIME22-24 and GPQA.
- Finding: (F) EAGLE speedup falls from 1.73x to 1.21x from batch 1 to 128; verification is 42%-95% of time. (V) "This assumption typically holds when the system is memory-bound, which is common at small batch sizes, but may break down once the system becomes compute-bound."
- Left open (V): "We leave an analysis that measures overlap with respect to this evolving context for future work."

### 9. Agentic AI Workload Characteristics (IISWC 2026, author-stated)
https://arxiv.org/abs/2605.26297

- Question: what do coding-agent workloads do to an LLM server?
- How: 2x H100 NVL, vLLM v0.20.0 with TP=2, Qwen3.6-27B and Gemma4-31B, Claude Code driven through Harbor.
- Finding (V): Table II, Terminal-Bench 2.0: prefix-cache hit 98.85 / 16.78 / 19.21 % at max concurrency 15 / 20 / 25, generation throughput 30.08 / 16.96 / 10.30 tok/s per request; "reducing average generation throughput by up to 66%"; "Failed agents often carry larger contexts (up to 1.8× in terms of mean) and longer tails".
- Left open (V): "Advanced orchestration techniques, including speculative or overlapped tool execution, dynamic context compaction or summarization, and branching or tree-search workflows, are outside the scope of this study. These techniques may change tool latency, context evolution, and KV-cache behavior, and we leave their characterization to future work."; "Our results assume recomputation under thrashing."

### 10. The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective (HPCA 2026)
https://arxiv.org/abs/2506.04301

- Question: what do agents cost in latency and energy at the infrastructure level?
- How (F): vLLM v0.6.6; Llama-3.1-8B on 1x A100 40 GB and 70B on 8x A100.
- Finding (F): LLM inference is 69.4% of latency; agents use 62.1x-136.5x the GPU energy per query.
- Left open (V): "Although our analysis does not account for LLM request batching"; "A detailed exploration of efficient agent serving under SLA constraints is left as future work."

### 11. ServeGen: Workload Characterization and Generation of LLM Serving in Production (NSDI 2026)
https://arxiv.org/abs/2505.09999

- From the existing catalogue in `research_notes/Reproducible LLM systems paper setups/serving_scheduling.md`; not re-opened in this pass. 3.54B requests; naive ShareGPT + Poisson under-provisions by 50%. Left open: "characterizing prefix caching requires access to the content of requests which we currently opt out of".

### 12. Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference (arXiv 2503.08311; venue not confirmed)
https://arxiv.org/abs/2503.08311

- Question: why does throughput plateau at large batch?
- How (F): 1x H100, vLLM, OPT-1.3B/2.7B and Llama-2-7B/13B.
- Finding (V): "over 50% of the attention kernel cycles stalled due to data access delays for all tested models".
- Left open (F): "we leave as future work the exploration of bottlenecks in larger models". The fetcher flagged 7.49% versus 12.78% for one number.

### 13. Are We Scaling the Right Thing? A System Perspective on Test-Time Scaling (workshop paper, arXiv 2509.19645)
https://arxiv.org/abs/2509.19645

- How: 4x GH200-96GB, vLLM v0.7.3, R1 distills and S1.1 at 1.5B / 7B / 14B, MATH500.
- Finding (V): "scaling to 4 GPUs only improves the latency improvement by an average of 1.7×; for 1.5B models, multi-GPU results are even worse than single-GPU. Such a phenomenon can be potentially attributed to the workload nature of reasoning tasks, emphasizing long sequences instead of large batch sizes."
- Left open: no equal-latency small-versus-large comparison and no hybrid models.

### 14. The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management (NeurIPS 2025 listing; track not confirmed)
https://arxiv.org/abs/2508.21433

- How: SWE-agent on SWE-bench Verified, several models up to Qwen3-Coder 480B.
- Finding (V): "a simple environment Observation Masking strategy halves cost relative to the Raw Agent while matching, and sometimes slightly exceeding, the solve rate of LLM-Summary"; "LLM-Summary consistently leads to longer trajectories, suggesting they mask failure signals that would otherwise prompt earlier termination."; "for Qwen3-Coder 480B, we observe an increase of the mean trajectory length by 15% compared to the Raw Agent and 13% compared to the Observation Masking strategy".

### 15. A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility (arXiv 2504.07086; venue not confirmed)
https://arxiv.org/abs/2504.07086

- Finding on hardware (V): "We evaluated the same model across five different compute clusters, each with varying GPU types and memory configurations. As shown in Figure 10, performance varied by up to 8% for OpenRS-1.5B and 6% for DeepSeek-R1-Distill-7B on AIME'24"; "OpenThinker2-7B scored 53.0% ± 4.6 on A100 versus 57.1% ± 5.2 on H100 for AIME'24"; "even consecutive runs on the same A100 cluster produced variations: Bespoke-Stratos-7B ranged from 19.3% to 23.0% on AIME'24 across runs."
- Finding on length (V): "Longer outputs tend to be more error-prone, even in complete responses not close to the maximum sequence length."; "No, truncation is not the primary cause. While responses approaching the 32,000-token limit are almost always incorrect, the correlation between length and errors persists even for complete responses well below this limit (e.g., 10,000-15,000 tokens)."

### 16. Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression (arXiv 2608.01631; venue not confirmed)

- F from the alphaXiv page only: fixed-trace replay protocol; failure "is tied less to KV memory reduction itself than to losing access to parts of the reasoning trace". Says nothing about output length. Not used as evidence.

## Supporting papers (not question papers)

- ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models (ICLR 2026 oral per iclr.cc listing in search), https://arxiv.org/abs/2510.01290. 1x A100 80GB and 1x GH200, max 32K tokens. V: "Unlike quantization, eviction does not cause an increase in generation length; however, as b→0, accuracy degrades with higher compression." This is the opposite of paper 4.
- M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models, https://arxiv.org/abs/2504.10449. V: "higher accuracy compared to DeepSeek R1 distilled transformer reasoning models under a fixed generation time budget using self-consistency voting"; "Our current hybrid model is only 3× faster than a Transformer of the same size when serving inference with vLLM."
- KVBuffer (arXiv 2605.19049, preprint). V: "The linear attention state is usually larger than the per-token key and value (KV) by two orders of magnitude, e.g., 2MB in Qwen3-Next Gated DeltaNet Layer". L40S and B300, Qwen3.5-4B.
- HALO, "Hybrid Linear Attention Done Right" (arXiv 2601.22156). V: "We propose to select attention layers that, when replaced by RNN layers, exhibit a large drop in recall performance and a small drop in CSR".
- Distilling to Hybrid Attention Models via KL-Guided Layer Selection (arXiv 2512.20569; ICLR 2026 from search). V: "performance on in-context recall benchmarks like RULER ... is highly dependent on the sliding window size"; "A simple yet effective solution is to incorporate a few global (softmax) attention layers".
- HySparse (arXiv 2602.03560). V: "HySparse uses the full attention layer as a precise oracle to identify important tokens". laLTE (arXiv 2510.20787). V: "once a token is dropped, it is no longer available to any future query".
- DeltaS: Reading the Gated Linear Attention State for KV Cache Eviction in Streaming Video (arXiv 2609.27470, preprint). Qwen3.5 backbone, six long-video benchmarks, accuracy on H100s. V: "Our evaluation uses a single hybrid backbone; the effectiveness of DeltaS across other hybrid architectures and model scales remains to be established."
- LLM-42 (arXiv 2601.17768). V: "Enabling deterministic inference in SGLang incurs a substantial throughput penalty, ranging from 24% (in=2048, out=512) to 36% (ShareGPT)."
- Copilot in the Wild (arXiv 2608.00101). V: "Context compaction is triggered in 7.8% of sessions yet affects 44% of total tokens"; Table 6: 7.8% of sessions, 44.2% of total tokens; "Without prompt/response content, we cannot correlate infrastructure efficiency with task completion quality".
- Sample Count Is Not Enough (arXiv 2609.19499). V: "On A100 GPUs, eight serial calls use 4.64 – 4.86× as much gross GPU-device energy". Phi-3-mini and Qwen2.5-1.5B on GSM8K only; no model-size comparison.
- Search summary only (S), not opened: vLLM batch-invariance docs (VLLM_BATCH_INVARIANT), Thinking Machines "Defeating Nondeterminism in LLM Inference", MarginGate (arXiv 2605.30218), AgentSpec (arXiv 2608.24004), SmoothAgent (arXiv 2607.00151), TokenPilot, Continuum (arXiv 2511.02230), CacheScout (arXiv 2608.14624), FastTTS (arXiv 2509.00195), "Latency and Token-Aware Test-Time Compute" (arXiv 2509.09864), "LLM Serving in the Wild" (arXiv 2608.03036), MiniCPM-SALA (arXiv 2602.11761, F), "Understanding and Enhancing Mamba-Transformer Hybrids" (arXiv 2510.26912, F).

## Local measurements used as grounding

- `/home/ryankert_l/kv-fetch/kv-fetch/results/profile/findings.md`: decode step at 8K context. Qwen3.5-4B: batch 1 10.60 ms (gemm 89%, attention 4%); batch 16 17.44 ms (attention 27%); batch 64 36.34 ms (attention 49%, gemm 32%, gdn_recurrent 16%). Qwen3-4B: batch 1 11.10 ms (attention 15%); batch 16 30.98 ms (attention 68%). Qwen3.5-9B batch 16 26.31 ms (attention 18%); Qwen3-8B batch 16 39.47 ms (attention 53%).
- `/home/ryankert_l/kv-fetch/kv-fetch/EXPERIMENTS.md` (H-a): "the oracle shared set at 512 misses 22.7% of attention on Qwen3.5-4B vs 11.6% on Qwen3-4B, and each of Qwen3.5's 8 attention layers misses more than its Qwen3-4B counterpart (18-34% vs 5-20%). Attention in the hybrid's attention layers is less concentrated, not more." 30 AIME problems, one model pair.
- Project memory (kv-fetch-phase0.md): E3 pilot losses "mostly truncation at 16K (not verbatim loops, except SnapKV)"; hybrid H3 losses mostly 32K truncation.

## Cards

Five cards. Order is the order they were built, not a ranking.

### Card A. Where is the "think longer versus bigger model" threshold on a real batched engine at 48 GB?

- **Question.** When cost is GPU time measured on a real batched engine at the concurrency one 48 GB card can hold, does the model size below which "think longer" loses to "use a bigger model" stay near the roughly 14B that Kinetics derives, or fall toward the 8-bit 4B that the memory-budget study reports?
- **Tension.** Two accepted papers give different thresholds from different cost models, and each names the other's regime as its limitation. Kinetics drops weight access because of a batch-4K assumption; Not All Bits counts weights in full because it accounts per generation. A 48 GB card sits between the two: weights and KV are the same order of magnitude.
- **Evidence.** Kinetics quotes (paper 1); Not All Bits quotes (paper 2); local roofline: at 8K and batch 1, GEMM is 89% of a Qwen3.5-4B step and attention 4%, at batch 16 attention is 27% (hybrid 4B) versus 68% (Qwen3-4B); paper 13's 1.7x on 4 GPUs.
- **Why unknown.** Kinetics did not build a serving system; Not All Bits measured latency on one A100 without amortizing weights across a batch; paper 13 has no cross-size equal-cost comparison; M1 compares one hybrid at one size.
- **If the threshold stays near 14B.** The Kinetics conclusion is robust to the cost model; small-model test-time scaling is a poor use of a 48 GB card as well.
- **If it falls toward 4B or lower.** The 14B figure is a property of the batch-4K assumption; the quantity that sets the threshold is the weights-to-KV traffic ratio at feasible batch, which a hybrid changes.
- **First experiment.** Qwen3-4B / 8B / 14B (bf16) and 32B (FP8 or AWQ), plus Qwen3.5-4B / 9B / 27B-FP8; AIME 2024+2025 and a MATH-500 subset; thinking budgets 4K-32K and self-consistency N in {1, 4, 16}; vLLM on one 48 GB card at the largest concurrency that fits. Number: GPU-seconds per problem to reach each accuracy level, and which model size is on the frontier at each cost, compared with the eFLOPs-predicted and memory-budget-predicted frontiers.
- **Closest existing answer.** Not All Bits (memory budget, no batched measurement); paper 13 (latency only); M1 (one hybrid, fixed time budget); 2609.19499 (batching of candidates, tiny models). None measures the crossover.
- **Overlap.** D3: uses the same single-GPU roofline harness, and D3's result (GEMM dominates at batch 1) already hints at the cost side. It is not a bottleneck hunt, and the accuracy side is new.
- **Caveat.** Qwen3.5 and Qwen3 differ in training as well as architecture, so the hybrid part of the comparison is confounded.

### Card B. Is the extra length under lossy optimizations compensation or failure?

- **Question.** When a lossy inference optimization makes a reasoning model generate more tokens, are the extra tokens compensation that recovers accuracy, or the symptom of a run that has already failed?
- **Tension.** Rethinking KV speculates compensation and does not test it; Sober Look finds longer outputs are more error-prone even far from the token limit; Hold Onto That Thought says eviction lengthens traces while ThinKV says eviction does not; the quantization study says the mechanism is not understood.
- **Evidence.** Papers 5, 4, 6, 15; ThinKV; the nondeterminism paper's 9,000-token length swing from numeric noise alone (a noise floor any study must beat); local E3 / H3 losses mostly truncation.
- **Why unknown.** Every paper reports length as a side observation. None runs a paired per-problem analysis with a noise floor, and none caps the perturbed run at the unperturbed length to see whether accuracy drops.
- **If compensation.** Per-token speedups of compression overstate the gain; the right cost metric is tokens (or seconds) to a correct answer, and budget control interacts with compression.
- **If symptom.** Length growth is a cheap online failure signal, which points to abort-and-fall-back designs; truncation-dominated losses are explained.
- **First experiment.** Qwen3-8B and DeepSeek-R1-Distill-Qwen-7B; AIME 2024+2025 and MATH-500; 8 seeds per problem at T=0.6, max 32K. Conditions: full KV bf16; R-KV, H2O, SnapKV at budgets 512 / 1024 / 2048; 4-bit KV; 4-bit weights. (1) noise floor of the per-problem length ratio between unperturbed runs; (2) paired length ratio and correctness per perturbation; (3) rerun the perturbed condition with thinking closed at the control's median length for that problem. Number: accuracy uncapped minus accuracy capped under each perturbation, against the same difference for the control; and P(correct | length ratio > 1.5) versus P(correct | otherwise).
- **Closest existing answer.** Hold Onto That Thought sec. 4.5 and Quantization Hurts Reasoning sec. 4.1 (observations only); Rethinking KV (speculation); ThinKV (opposite observation for eviction); Sober Look sec. 5.1 (length-error correlation without compression).
- **Overlap.** D1: the perturbation is KV eviction on long reasoning, and existing E3 runs already contain per-problem lengths. The question is about the response to lossy optimization in general, not about a better selector.

### Card C. Are the attention layers left in a hybrid intrinsically harder to sparsify?

- **Question.** Do the softmax-attention layers that remain in a hybrid linear-attention model spread their attention over more tokens than the attention layers of a comparable full-attention model as a general property, or is the 22.7% versus 11.6% gap an accident of one model pair and one task?
- **Tension.** KV compression and sparse-attention results (and the Kinetics argument for sparse attention) were established on full-attention models. Hybrid design papers keep exactly the layers whose removal hurts recall most. The one local measurement says hybrid attention is less concentrated. DeltaS evicts on Qwen3.5 without comparing to a full-attention model.
- **Evidence.** Local H-a; HALO; KL-guided layer selection; Kinetics limitation on training and sparse attention; DeltaS limitation.
- **Why unknown.** The local result is 30 AIME problems on one pair. No paper compares attention concentration between a hybrid and a full-attention model.
- **If general.** Post-hoc KV compression numbers do not transfer to hybrids, and the residual KV is closer to incompressible; this explains why HySparse and laLTE train sparsity in.
- **If an accident.** The existing compression literature transfers; the local anomaly is a Qwen3.5 / AIME property. This answer is less interesting but it retires a tempting wrong belief.
- **First experiment.** Qwen3.5-4B versus Qwen3-4B, Qwen3.5-9B versus Qwen3-8B, and one converted hybrid against its own source model if weights are public (availability not checked); AIME, RULER, LongBench. Number: attention mass missed by a per-layer oracle top-k (k = 512, 1024) and attention entropy per attention layer, plus accuracy under H2O / SnapKV at equal budgets on both members of each pair.
- **Closest existing answer.** Local H-a; DeltaS (eviction on Qwen3.5 for video, no comparison); HySparse and laLTE (trained sparsity).
- **Overlap.** D1 directly: this is the H-a finding of the dropped direction, asked as an architecture property instead of a selector design. D2: none.

### Card D. Does long context cause agent failure, or do failing runs just run longer?

- **Question.** Does accumulated context causally lower an agent's chance of solving a task, or do failing runs simply accumulate more context before they stop?
- **Tension.** The IISWC 2026 study reports failed agents carry up to 1.8x the context and leaves compaction to future work; the Copilot trace has no quality signal; The Complexity Trap finds masking keeps the solve rate while summaries lengthen trajectories.
- **Evidence.** Papers 9, 14; Copilot in the Wild.
- **Why unknown.** The correlation is observational. The Complexity Trap changes the whole policy from step 0, so it does not separate the effect of context size at a given state from the effect of the run already going wrong.
- **If causal.** Compaction is a quality tool and should be triggered by context size.
- **If symptom.** Context growth is a failure signal; the serving system can use it for early termination or deprioritization, and compaction is a cost tool only.
- **First experiment.** SWE-agent or mini-SWE-agent with Qwen3-Coder-30B-A3B (FP8 or AWQ) on vLLM; 100 SWE-bench Verified instances. Fork each recorded run at steps 20 and 40 into three arms: continue raw; continue with old observations masked; continue with context padded by stale observations to match the failed-run length. Number: solve-rate difference between arms among forks from the same state, and steps to termination.
- **Closest existing answer.** The Complexity Trap (policy-level comparison); context-length degradation studies outside agents (not opened).
- **Overlap.** None of D1-D4 directly; loosely D4 because both concern agent rollouts.
- **Caveat.** The infrastructure (Docker task environments on a shared node) is heavy and one week is tight. The Complexity Trap's masking result already suggests context size is not a strong cause, so this is partly answered.

### Card E. At temperature above zero, does numeric nondeterminism change anything beyond sampling noise?

- **Question.** When a reasoning model is sampled at temperature above zero, do numeric precision, batch size and GPU type shift its accuracy and output-length distribution by more than seed-to-seed sampling noise?
- **Tension.** The NeurIPS 2025 paper claims numeric precision adds variance beyond the intended randomness, but its Table 5 measures spread across configurations without a within-configuration baseline and has FP32 as the most variable format in one cell. Sober Look attributes up to 8% to hardware, yet reports 19.3% to 23.0% across consecutive runs on the same cluster. Meanwhile determinism costs 24% to 36% of throughput (LLM-42).
- **Evidence.** Papers 3, 15; LLM-42.
- **Why unknown.** Neither paper separates between-configuration variance from within-configuration variance with enough seeds.
- **If it shifts.** Accuracy comparisons across engines, batch sizes and GPUs in systems papers carry an uncontrolled term; the mechanism (which tokens flip, and why reasoning traces amplify it) is worth finding.
- **If it does not.** Batch-invariant kernels are unnecessary for sampled workloads and the throughput cost is avoidable; the hardware effect in Sober Look needs another explanation. This is the less surprising answer.
- **First experiment.** DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B on AIME'24 and MATH-500 at T=0.6-0.7 in vLLM; factors: bf16 versus fp32, batch 1 / 8 / 32, RTX A6000 versus RTX 6000 Ada; at least 64 seeds per configuration on AIME. Number: between-configuration variance of mean Pass@1 against within-configuration seed variance (permutation test), and a two-sample test on the length distributions.
- **Closest existing answer.** Table 5 of paper 3 and Figure 10 / Stage 1-3 of paper 15; both assert an effect without the separation.
- **Overlap.** None of D1-D4.

## Candidates set aside

- Why KV quantization destroys the 7B reasoning model but not 14B / 32B (paper 7's open item). Partly explained by paper 6 (V: "this is due to the huge outliers over biases in the key and value layers of these two models"); the prefix-caching anomaly is probably tied to vLLM 0.8.1.
- Speculative decoding under batching for reasoning models. Explained by paper 8; KVBuffer covers hybrid verification cost.
- Prefix-cache thrashing cliff for agents (paper 9). Crowded (SmoothAgent, Continuum, CacheScout, TokenPilot, all S only) and close to D2 / D4.
- Memory traffic of linear-attention state at decode. Covered by KVBuffer.
- Large-batch bottleneck in larger models (paper 12's open item). Same as D3.

## Not verified

- Venues: Not All Bits (ICLR 2026), ThinKV (ICLR 2026 oral), nondeterminism paper (NeurIPS 2025 oral) are from search listings, not from the papers. Hold Onto That Thought and The Complexity Trap appear in NeurIPS 2025 listings with track unknown. Papers 6, 12, 15, 16 have no confirmed venue.
- Papers 10 and 12: setup details are F; only the quoted sentences are V.
- Paper 11 was taken from the existing catalogue and not re-opened.
- Whether a converted hybrid with public weights and a public source model exists for Card C was not checked.
- Internal inconsistencies inside source papers (papers 2, 3, 4, 5, 12) were reported by the fetcher and not resolved.
- Gap checks were web searches on 2026-11-11; a paper not indexed by the search could have answered any card.
