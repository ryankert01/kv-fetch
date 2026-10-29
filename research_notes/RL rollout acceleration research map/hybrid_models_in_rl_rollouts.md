# RL post-training and rollout generation for hybrid linear-attention / SSM models

Scope: hybrid models (Gated DeltaNet, Mamba/Mamba2, lightning attention, KDA) in RL rollouts. Covers system support, reported problems, techniques that use the fixed-size recurrent state, and gaps. Research date 2026-10-29. Most 2026 sources are arXiv-only preprints, vendor or personal blogs, docs, and GitHub issues/PRs. Each item gives its venue/status. "Search snippet" means I saw the claim only in a search-result summary and did not open the page, so it is weaker evidence.

---

## Q1. Which hybrid models were trained with large-scale RL, and what do their reports say about rollout infra, cost and problems?

### Takeaway
Almost every major hybrid family now ships RL-trained checkpoints: MiniMax-M1, Qwen3-Next/3.5/3.6, Kimi Linear, Kimi K3, Ring-linear/Ling, Nemotron 3/3.5, Falcon-H1R and Jamba Reasoning. Their reports agree on one hybrid-specific pain point, a training-inference numeric mismatch that the recurrent state amplifies. Only Kimi K3's report describes rollout infrastructure that explicitly handles the recurrent (KDA) state: it offloads that state alongside the attention KV during partial rollouts and runs a KDA-aware prefix cache. None of the reports gives a rollout-cost breakdown attributable to the hybrid layers.

### Cited Findings
**MiniMax-M1** (arXiv 2506.13585, June 2025, arXiv tech report only; lightning-attention hybrid)
- Rolled-out token probabilities differed between training and inference modes. Layer-by-layer analysis blamed "high-magnitude activations in the LM head". Raising the LM head to FP32 "improved the correlation between training and inference probabilities from approximately 0.9x to 0.99x." — [MiniMax-M1 §3.2](https://arxiv.org/html/2506.13585v1)
- Secondary sources say the mismatch did not appear in smaller dense softmax-attention models. I did not verify that wording in the paper. — [search summary of MiniMax-M1 and reviews](https://www.themoonlight.io/review/minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attention)
- CISPO clips importance-sampling weights instead of token updates. It "matches DAPO's performance with only 50% of the training steps." — [MiniMax-M1 §3.1](https://arxiv.org/html/2506.13585v1)
- Full RL ran on 512 H800s for 3 weeks at a rental cost of $534,700. M1 uses 25% of DeepSeek-R1's FLOPs at 100K generation length. The paper does not name the rollout engine and gives no separate rollout cost. — [MiniMax-M1 abstract/§1](https://arxiv.org/html/2506.13585v1)
- Other RL-stability fixes: AdamW β2=0.95 and eps=1e-15, because gradients span 1e-18 to 1e-5. Generation halts if 3,000 consecutive tokens each have probability >0.99, to stop repetition. — [MiniMax-M1 §3.2](https://arxiv.org/html/2506.13585v1)

**MiniMax-M2** (blog, 30 Oct 2025): MiniMax dropped the hybrid and went back to full attention.
- "Anyone who read our M1 paper will recall the serious precision issues we hit during RL training." "Linear attention is currently far more sensitive to numerical precision than full attention." "The infrastructure for linear and sparse attention is much less mature." The post also asks "How do you optimize speculative decoding with linear attention backbone?" and stresses high prefix-cache hit rates in real conversations. — [MiniMax blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)

**Qwen3-Next** (Sept 2025, model card/blog; GDN:attention = 3:1, 512-expert MoE)
- Per the model card and blog, GSPO was used to address "the stability and efficiency challenges" that hybrid attention plus high-sparsity MoE create in RL training. No rollout-infra details. — [HF model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking); [Alibaba Cloud blog](https://www.alibabacloud.com/blog/qwen3-next-towards-ultimate-training-%26-inference-efficiency_602580) (both via search snippet)

**Qwen3.5** (blog, 17 Feb 2026; no full tech report found)
- Qwen built "a scalable asynchronous RL framework that supports Qwen3.5 models of all sizes" with "a fully disaggregated training-inference architecture". It improves throughput and consistency "via techniques such as FP8 end-to-end training, rollout router replay, speculative decoding, and multi-turn rollout locking". The post claims a 3×–5× end-to-end speedup. It gives no GPU counts and says nothing GDN-specific about rollouts. — [Qwen3.5 blog (Alibaba Cloud)](https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894)
- The Qwen3.5-Omni tech report (arXiv 2604.15804) reuses the Qwen3.5 hybrid MoE and includes RL stages, but has no rollout-infra detail. — [Qwen3.5-Omni report](https://arxiv.org/html/2604.15804v1) (search snippet)

**Kimi Linear** (arXiv 2510.26692, Oct 2025, tech report; KDA + MLA, 48B-A3B)
- "We noticed that the precision mismatch between training and inference engines may lead to unstable RL learning. Therefore we introduce truncated importance sampling." Under the same RL recipe, Kimi Linear improved "faster and better" than MLA on MATH500/AIME2025; the figure has no numeric values. The paper uses a chunk kernel in prefill and a recurrent kernel in decode. It open-sources the KDA kernel and the vLLM implementation, "requiring no modification to caching or scheduling interfaces". — [Kimi Linear §5.4.2, §5.5.1, §6.3](https://arxiv.org/html/2510.26692v2)

**Kimi K3** (arXiv 2607.24653, July/Aug 2026, tech report; 2.8T total / 104B active; 69 KDA + 24 MLA layers)
- Partial rollouts: "the generation phase pauses as soon as a fraction λ∈(0,1) of trajectories completes", and paused rollouts resume first in the next iteration. Rollout and training "share the same quantization scheme", which removes quantization mismatch. — [Kimi K3 §4.1.2, §4.1.4](https://arxiv.org/html/2607.24653v1)
- External KV cache pool in CPU DRAM: "KDA states are offloaded and prefetched together with the corresponding MLA KV cache blocks". The pool "is released" after each rollout iteration "to avoid contention with training workloads". — [Kimi K3 §5.3.1](https://arxiv.org/html/2607.24653v1)
- KDA-aware prefix cache: recurrent-state checkpoints are saved "only at (a sparse subset of) MLA's hash endpoints", typically at conversation-turn boundaries. Hash blocks are 512 tokens; the physical shared block is forced to 1024–6144 tokens. A hit copies the checkpoint into the request's private state. — [Kimi K3 §5.4.1](https://arxiv.org/html/2607.24653v1)
- Sandboxes: checkpoint/resume latency 133 ms / 49 ms; 51.2M sandboxes created. — [Kimi K3 §5.3.2](https://arxiv.org/html/2607.24653v1)

**Ring-mini-linear-2.0 / Ring-flash-linear-2.0** (Ling team, arXiv 2510.19338, Oct 2025, preprint)
- The paper calls training-inference disparity the "root cause of the prevalent problem of training collapse in RL". It aligned these modules: the linear-attention KV state ("If the KVCache is initialized as BF16 in the inference engine, errors will accumulate progressively"), the LM head (FP32 softmax via a custom GEMM), RMSNorm, RoPE, attention backend, and MoE router. "In extreme cases, the output probability for the same token can be 0 during training and 1 during inference." After alignment it trains on rollout probabilities directly, with no recompute. — [Ring-linear §5.2](https://arxiv.org/html/2510.19338)

**NVIDIA Nemotron 3 / 3.5** (hybrid Mamba-Transformer MoE; NeMo-RL + NeMo-Gym + vLLM)
- Nemotron 3 Nano (arXiv 2512.20848, Dec 2025): "synchronous GRPO with masked importance sampling to mitigate training-inference misalignment". 128 prompts × 16 generations, max generation 49K, MoE router frozen. Rollouts go through NeMo-Gym's vLLM model server, which "carefully preserves token and inference log-prob data". — [Nemotron 3 Nano §3.2](https://arxiv.org/html/2512.20848)
- Nemotron 3 Super (arXiv 2604.12374, Apr 2026): asynchronous GRPO with inference "at most one step behind". In-flight weight updates, and "We do not recompute the KV cache after updating the model weights on the inference workers." Masked IS. 256 prompts × 16; max generation 49K raised to 64K; "thousands of GPUs". — [Nemotron 3 Super §3.2.4–3.2.5](https://arxiv.org/html/2604.12374v1)
- Nemotron 3.5 Super/Lightning (NVIDIA report dated 2026-10-27, not on arXiv at time of reading): router replay plus FP32 LM head "reduce the KL error between the training and inference engines by more than an order of magnitude." GBS 8,192, 16 rollouts per sample, max generation 80K raised to 128K. — [Nemotron 3.5 report §3.2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)

**Falcon-H1R-7B** (arXiv 2601.02346, Jan 2026; parallel Transformer-Mamba hybrid)
- GRPO on "a custom RL framework on top of the verl library", with TIS cap 2, 48K max response, G=16, 256 H100s. G=32 "substantially increased generation time" without gains. The report makes no hybrid-specific RL claims. — [Falcon-H1R §3](https://arxiv.org/html/2601.02346v1)

**Ling-3.0-tiny/flash** (hybrid KDA/MLA MoE, 7.9B and 124B): trained with agentic RL on AReaL + Megatron (see HARTS, Q4). — [HARTS](https://arxiv.org/html/2608.28158)

**OLMo Hybrid 7B** (Ai2, 2026; 75% GDN layers): only SFT/DPO checkpoints are listed, with no RLVR checkpoints. — [HF card](https://huggingface.co/allenai/OLMo-Hybrid-7B)

**Jamba Reasoning 3B** (AI21, Oct 2025): the model card says online RLVR was used and that "full support for training Jamba through VeRL will be available soon". — [model card, seen via search-result mirror](https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B)

### Inferences
- Large-scale RL on hybrids is routine in industry, so "can you RL a hybrid" is settled. What is not settled is how to make hybrid rollouts cheap and consistent. Every report spends its RL-infra text on mismatch fixes (FP32 LM head, router replay, TIS/MIS, aligned state precision), not on throughput techniques specific to the recurrent state.
- Kimi K3 is the one industrial report that treats recurrent state as a first-class object in the RL rollout system (offload with KV, prefix checkpoints). Its pool is cleared every iteration, so it does not persist state across policy versions.
- MiniMax returning to full attention for M2 is the main public negative data point. Its stated reasons (precision sensitivity in RL, immature infra, spec decoding, prefix caching) are exactly the systems gaps below.

### Gaps
- No full Qwen3.5/3.6 tech report with RL-infra detail was found; the blog does not explain "multi-turn rollout locking" or "rollout router replay".
- No hybrid report gives a rollout-time vs. training-time breakdown, or compares rollout cost against an equal-quality dense model at RL scale. MiniMax-M1's FLOPs comparison is analytical.
- I did not open Nemotron-H / Nemotron Nano 2 RL details, or the Kimi K3 report's full RL algorithm section (only §4.1 and §5 excerpts).

---

## Q2. Training-inference mismatch for hybrid/linear-attention kernels: causes, fixes, and hybrid-specific measurements

### Takeaway
The hybrid-specific cause is well documented. Training uses chunkwise-parallel GDN/KDA/SSD kernels; the rollout engine uses a chunked kernel for prefill and a token-by-token recurrent kernel for decode. Small state differences then propagate through every later token, and further through the delta-rule update. Bitwise parity for GDN was demonstrated in 2026 (TorchTitan+vLLM; XoRL+SGLang fork; Megatron's own engine for Nemotron-3-Nano), but it costs about 20–40% sampling throughput, or 2–5× with full batch invariance. One careful study found the accuracy benefit unclear. FP32 recurrent-state caches are the standard practical mitigation; BF16 state visibly breaks generations.

### Cited Findings
**Mechanism**
- Most open-source GDN stacks use FLA's chunked kernel for training and prefill and a recurrent kernel for decode, which are "two different algorithms" on the two sides of the RL loop. — [Yichuan Wang blog, Aug 2026](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)
- XoRL trains GDN 64 tokens at a time while SGLang decode updates one token at a time. "Gated DeltaNet is a recurrent layer. It carries an FP32 state from token to token." The fix replays the 64-row training calculation from the last chunk boundary at decode time, inside CUDA graphs. "We can't shard recurrent layers by heads, because the state runs along the sequence dimension." GDN is called "the largest single cost we can name" for Qwen. — [Ashwinee Panda, "0 train-infer mismatch…", 17 Aug 2026 (blog)](https://kiddyboots216.github.io/mismatch/)
- A theory-oriented tech report says "the effective policy is parameters plus execution path." For DeltaNet, "The state error is not merely observed downstream; it participates in the future update dynamics." It also warns that recomputing the denominator with the learner engine "hides, rather than fixes, the mismatch". The page has no measurements. — [Yifan Zhang et al., 9 Aug 2026, project page/tech report](https://yifanzhang-pro.github.io/Pretraining-RL-Science/)
- Ring-linear lists the linear-attention state in BF16 as a source that accumulates error, alongside the LM head, RMSNorm, RoPE, prefill-vs-decode attention, and MoE routing. — [Ring-linear §5.2.1](https://arxiv.org/html/2510.19338)

**Measurements specific to hybrids**
- TorchTitan RL + vLLM on Qwen3.5-9B/35B-A3B. The generator uses the recurrent kernel for both prefill and decode; the trainer uses recurrent forward and chunked backward. Together with batch-invariant GEMM, `num_splits=1` attention, and a batch-invariant MoE router, logprob diff at step 0 is exactly 0. At off-policy 32, BI gives ~0.035 vs ~0.065 without (no-BI runs away after ~step 100). Throughput cost: about 2–3× on MATH and about 5× on the 64-turn terminal agent. Accuracy gains are mixed; the author recommends BI "as a debugging tool rather than a production default". Prefix caching stays on via `--mamba-cache-mode align`, salted per group on weight sync. The post states that "no open-source stack has hit train/inference KL = 0 for linear attention" before this. — [Yichuan Wang blog](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)
- Qwen3.6-35B-A3B, bitwise-identical forward between XoRL and an SGLang fork: sampling throughput fell from 7,863 to 4,767 tok/s (−39.4%). Removing exact GDN decode recovered about half the added wall time. The full RL step costs ~20% more (374.3 s → 447.0 s). — [Panda blog](https://kiddyboots216.github.io/mismatch/)
- Megatron-LM PR #8646 (draft, opened 28 Oct 2026): in Megatron's in-repo inference engine with FP32 conv/SSM states in batch-invariant mode, "0 of 486,586 generated tokens have an engine log-probability that differs from the trainer's" in a Nemotron-3-Nano GRPO test. — [Megatron-LM PR #8646](https://github.com/NVIDIA/Megatron-LM/pull/8646)
- NeMo-RL Qwen3.5 recipes report `train/gen_kl_error` of ~0.0024 (35B-A3B) and ~0.0014 (9B), with nightly thresholds of 0.004/0.003. — [NeMo-RL Qwen3.5 guide](https://docs.nvidia.com/nemo/rl/nightly/guides/models/qwen/qwen3-5.html)
- Kimi K3 on Miles (colocated LoRA, MXFP4 SGLang rollout): train/rollout KL "flat at the MXFP4 floor (about 2e-3)". — [LMSYS blog, 27 Jul 2026](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)
- Fireworks K3 RL: deterministic forward, batch-invariant RMSNorm, and router replay (R3). Synchronous LoRA train/rollout divergence as low as 0.0008 (k3 estimator). — [Fireworks blog, 12 Oct 2026 (vendor)](https://fireworks.ai/blog/kimi-k3-on-fireworks-fast-rl-at-frontier-scale)

**State precision**
- vLLM PR #63218 (open, Oct 2026): Nemotron Omni silently fell back to a BF16 SSM cache. "bf16 rounding of the recurrent state shows up as runaway generations": 23% of generations hit the token cap with BF16 vs 0.25% with FP32 (Nano Omni), and GPQA-D went 51.9 → 68.4. FP32 state doubles SSM-cache memory and costs a few percent of decode throughput. — [vLLM PR #63218](https://github.com/vllm-project/vllm/pull/63218)
- SGLang's Qwen3-Next defaults to `--mamba-ssm-dtype float32`. — [SGLang Qwen3-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next)
- Nemotron 3 Super stores the SSM cache in FP32 by default, with an FP16 + stochastic-rounding option, and notes that recurrent quantization error accumulates across decode steps while chunked SSD training has no per-token quantization boundaries. — [Nemotron 3 Super report](https://arxiv.org/html/2604.12374v1) and [Super FP4 docs](https://docs.nvidia.com/nemotron/nightly/nemotron/super3/quantization.html) (partly search snippet)
- Qwen3.5 checkpoints store GDN `A_log` and `out_norm` in FP32 while HF code builds them as BF16, according to closed Megatron-LM PR #3634. Miles #975 keeps `A_log` in FP32 so Megatron and SGLang agree. — [Megatron-LM PR #3634](https://github.com/NVIDIA/Megatron-LM/pull/3634) (search snippet); [Yichuan blog citing miles #975](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)

**Implementation divergences found in the wild**
- vLLM #51779 (Aug–Oct 2026): GDN Q/K-norm dtype differs between vLLM kernels and the HF BF16 formula. On Qwen3.5-397B, cached-decode NLL was 0.309 vs 0.409 in the reference. Maintainers concluded it is "a bug of transformers fallback path" and recommend FLA. A trainer using the HF fallback would therefore not match the engine. — [vLLM issue #51779](https://github.com/vllm-project/vllm/issues/51779)
- vLLM #59764 (Oct 2026): on Qwen3.6-35B-A3B (GDN+MoE, v0.30.0) a prompt's logprobs move by up to 0.198 when another sequence shares its engine step. Suspects include GDN chunked prefill over varlen batches and MoE kernels; with `max_num_seqs=1` the difference is 0.0. — [vLLM issue #59764](https://github.com/vllm-project/vllm/issues/59764)
- RL-Kernel PRs (Qwen3-Next parity work) report that transformers and vLLM disagree on where the gated-norm weight multiply happens. — [RL-Align/RL-Kernel PR #549](https://github.com/RL-Align/RL-Kernel/pull/549) (search snippet)
- verl #8178 (Oct 2026, closed as not reproducible in isolation): under Ulysses SP on Qwen3.5-9B, GDN with FLA `cp_context` gave wrong parameter gradients (all-param cosine 0.958; `linear_attn` 0.79) while forward logprobs matched within ~0.005. — [verl issue #8178](https://github.com/verl-project/verl/issues/8178)
- Thinking Machines' batch-invariant ops cover RMSNorm, GEMM and attention but not GDN/linear attention or MoE routing. — [Yichuan blog](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)
- SGLang's Dec 2025 hybrid post lists "Deterministic inference adaption … to support bitwise training-inference consistency" for hybrid linear models as future work. — [PyTorch/SGLang blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)

**Algorithmic fixes used**
- TIS (Kimi Linear; Falcon-H1R cap 2), masked IS (Nemotron 3 Nano/Super), CISPO (MiniMax-M1), GSPO (Qwen3-Next), FP32 LM head (MiniMax-M1, Ring-linear, Nemotron 3.5), router replay (Qwen3.5, Nemotron 3.5, Fireworks K3), and using rollout logprobs directly after alignment (Ring-linear). — sources above.

### Inferences
- The hybrid-specific fix is to make prefill, decode and the trainer forward use the same reduction order: recurrent everywhere, or chunked replay at decode. That trades away exactly the efficiency hybrids are bought for. Current costs are 20–40% sampling throughput (Panda) and up to 5× with full batch invariance (Wang). A cheap parity mode for GDN/KDA, such as fixed-chunk replay that is fast in CUDA graphs, is a clear open systems target.
- Evidence that bitwise parity improves RL outcomes for hybrids is weak; one seed per config in Wang's study. Most production teams use IS correction plus FP32 state/LM head rather than parity.
- The mismatch interacts with async RL. Nemotron 3 Super and Wang keep KV/state across weight swaps, so a rollout's recurrent state mixes policy versions. No one has measured how much worse this is for recurrent state than for attention KV.

### Gaps
- No peer-reviewed paper measures training-inference mismatch for hybrids against a matched dense baseline under the same RL stack. MiniMax-M1 only says it did not appear in smaller dense models (secondary sources).
- No public KL numbers for Qwen3.5/3.6 production RL, Kimi Linear RL, or MiniMax-M1 beyond the correlation figure.
- I did not check whether SGLang's deterministic-inference PR for Qwen3-Next (#12845) landed.

---

## Q3. Do RL frameworks and rollout engines support hybrids, and what limitations are documented?

### Takeaway
Basic support exists everywhere that matters. vLLM V1 and SGLang both serve Qwen3-Next/3.5/3.6, Nemotron-H/3 and Kimi Linear/K3. verl (Megatron/mbridge, FSDP, VeOmni), NeMo-RL, slime/Miles, vime, AReaL (via HARTS) and TorchTitan RL all have hybrid recipes. The documented limitations are concentrated in the features RL rollouts lean on: block-granular or experimental Mamba prefix caching, no same-step state reuse across requests in vLLM, spec-decode/MTP state bugs, coarse block sizes, CP/SP numerics for GDN, and determinism. I found no hybrid-specific documentation for OpenRLHF or ROLL.

### Cited Findings
**vLLM**
- V1 treats hybrids as first-class. Mamba state is "around 2.57 MiB" per sequence for Nemotron-Nano-12B-v2, vs ~64 KiB per 16-token attention block. Attention block size is inflated to match the Mamba page size (e.g., 672 tokens). `FULL_AND_PIECEWISE` CUDA graphs were needed to beat V0. "Prefix caching is also now supported for hybrid models that use Mamba-2, although it is currently in an experimental state." — [PyTorch/vLLM blog, Nov 2025](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)
- `mamba_cache_mode="align"` caches Mamba state only at block-multiple positions at the end of a scheduler step and requires chunked prefill. Newer versions add an opt-in fine-grained "shared prefix checkpoint". Flag names change across versions. — [vLLM APC docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) (search snippet across versions)
- RFC #45702 (Jun–Sep 2026): equal block sizes force coarse prefix granularity (e.g., 1024 tokens). "vLLM does not currently support reusing a Mamba state that is produced by another request in the same scheduler step." — [vLLM RFC #45702](https://github.com/vllm-project/vllm/issues/45702)
- PR #61192 (merged 19 Oct 2026): on hybrids, "reads that start together cannot reuse each other's recurrent state". A warm-up request prefills the shared prefix once: 16 questions went from 106.5 s to 12.0 s, and prefix-cache hit rate from 0.37 to 0.94 (Qwen3_5-arch 27B). The workaround applies to one endpoint, not general n>1 sampling. — [vLLM PR #61192](https://github.com/vllm-project/vllm/pull/61192)
- #57267: `mamba_cache_mode="all"` gives different results within a batch when prefill is split; `align` avoids it. #53504: with MTP on a hybrid, the first repeat of a prompt misses the prefix cache. — [vLLM #57267](https://github.com/vllm-project/vllm/issues/57267), [vLLM #53504](https://github.com/vllm-project/vllm/issues/53504) (search snippets)
- Spec-decode state bugs: #62304, where on Qwen3.5 with n-gram drafts at v0.31.0 "the GDN state misses the last accepted token's update" (clean on v0.32.0, likely fixed by PR #56531). Related: #53912 (MTP + prefix caching corrupts output on hybrids) and #61738 (wide verification failures on GDN, k ≥ 7). — [vLLM #62304](https://github.com/vllm-project/vllm/issues/62304)

**SGLang**
- Separate Mamba pool (per request) and KV pool (per token), plus an elastic VMM-based pool. MambaRadixCache copies state on a match and forks a checkpoint on insert. PD disaggregation transfers Mamba state "atomically" as one contiguous block. Future work listed: page size >1, HiCache, deterministic inference. — [PyTorch/SGLang blog, Dec 2025](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- Tracker #12867 (Nov 2025, auto-closed as inactive Mar 2026): prefix caching "only for page size = 1"; open items include overlap schedule, storing state "every k steps / branch point", spec-decode adaptation, deterministic inference (#12845), and HiCache. No RL items. — [SGLang #12867](https://github.com/sgl-project/sglang/issues/12867)
- Qwen3-Next scheduler: `no_buffer` (default) vs `extra_buffer` (overlap scheduling + branch-point caching; needs FLA and `--page-size 64`). — [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next)
- Unified Radix Cache (Aug 2026): one token-keyed tree with full-attention KV, SWA KV and Mamba checkpoints as components across GPU/host/external tiers. Release notes reportedly make it the default for SWA/Mamba/DSA. — [LMSYS blog](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/); [SGLang releases](https://github.com/sgl-project/sglang/releases) (search snippet)
- Correctness bugs: PR #29633, a silent wrong first token on Qwen3.5/3.6 when an extra_buffer cache hit has prompt length ≡ 1 mod 64. PR #45789: the scheduler discards Mamba fork state when chunked prefill is interrupted, forcing recompute. — [SGLang PR #29633](https://github.com/sgl-project/sglang/pull/29633), [SGLang PR #45789](https://github.com/sgl-project/sglang/pull/45789) (search snippets)

**RL frameworks**
- verl: v0.7.0 adds Qwen3-Next on Megatron; v0.7.1 adds Qwen3.5 via mbridge (PR #5381); v0.8.0 adds Qwen3.5 MTP SFT/RL (#5898), FSDP GRPO (#5682), Ulysses SP (#6482), and VeOmni demos up to 122B-A10B. No GDN-specific rollout features are listed. — [verl v0.8.0 notes](https://newreleases.io/project/github/verl-project/verl/release/v0.8.0); [verl v0.7.x mirror](https://sourceforge.net/projects/verl.mirror/files/v0.7.1/)
- verl #8362: the SGLang rollout ignores `rollout.logprobs_mode`, so logprobs differ between backends (tested on Qwen3-0.6B). — [verl #8362](https://github.com/verl-project/verl/issues/8362) (search snippet)
- NeMo-RL: Qwen3.5 9B/35B-A3B/397B-A17B GRPO recipes with vLLM rollouts. GDN `conv1d`, `in_proj_a/b` are excluded from FP8 rollout quantization ("last dim is 4"). Missing FLA makes Automodel ~2× slower with no error. Automodel CP works only for the MoE LLM. An async 1-off recipe exists. — [NeMo-RL Qwen3.5 guide](https://docs.nvidia.com/nemo/rl/nightly/guides/models/qwen/qwen3-5.html)
- Megatron-Bridge selects the cuDNN GDN kernel only on a Megatron-Core that declares `gdn_kernel_backend`; otherwise it falls back to FLA. So the trainer kernel can silently differ by version. — [Megatron-Bridge Qwen3.5 recipe docs](https://docs.nvidia.com/nemo/megatron-bridge/nightly/apidocs/bridge/bridge.recipes.qwen_vl.gb300.qwen35_vl.html) (search snippet)
- slime lists Qwen3Next. Its vLLM-based fork vime lists Qwen3.5/3.6/Qwen3Next but documents no hybrid-specific handling. — [vime repo](https://github.com/vllm-project/vime); [slime DeepWiki mirror](https://backiee.wasmer.app/https_deepwiki_com/THUDM/slime) (secondary)
- Miles ran Kimi K3 RL on day 0 (colocated Megatron trainer + SGLang rollout). — [LMSYS K3 blog](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)
- AReaL + Megatron power HARTS's hybrid agentic RL (Ling-3.0). AReaL "drops and recomputes the cache" on weight update, while TorchTitan RL keeps it. — [HARTS](https://arxiv.org/html/2608.28158); [Yichuan blog](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)
- A community verl fork supports Qwen3.5 async RL on Megatron and suggests `rollout_corr/log_ppl_diff ≲ 5e-4` as a health check. — [IanYHWu/verl-may-2026-async](https://github.com/IanYHWu/verl-may-2026-async) (search snippet)

### Inferences
- "Supported" means the model runs. The RL-relevant features (prefix sharing inside a GRPO group, spec decoding, determinism, CP) are the least mature and keep producing correctness bugs in the latest releases (Aug–Oct 2026).
- Same-step state reuse is the key hybrid limitation for GRPO. The n samples of a prompt arrive together, so with attention KV they share blocks automatically, but on hybrids the recurrent prefix may be recomputed per sample unless the engine forks state explicitly.
- When the trainer and the engine come from different stacks (HF/FLA/cuDNN/Megatron vs vLLM/SGLang kernels), GDN numerics diverge in ways that are hard to notice (#51779, #8178, RL-Kernel). Shared-kernel designs (TorchTitan RL, Megatron's own engine) are the reaction to this.

### Gaps
- I found no OpenRLHF or ROLL documentation or issue about hybrid models.
- I did not confirm how vLLM's internal `n>1` fan-out (ParentRequest) interacts with Mamba prefix caching. No issue explicitly reports "prompt prefilled n times for GRPO on hybrids".
- I did not find a "ReplaySSM/RecoverSSM" entry in verl/slime configs; ReplaySSM is engine-level (see Q6).

---

## Q4. Does any work exploit the fixed-size recurrent state (cheap fork, migration, swap, cross-iteration checkpointing of partial rollouts)?

### Takeaway
Partially, and mostly on the trainer side or inside serving engines, not as an RL-rollout systems contribution. LongStraw and HARTS exploit captured or handed-off recurrent state to share prompt or prefix computation across GRPO group members and rollout trees during training. Kimi K3 offloads KDA state with KV during partial rollouts. SGLang and Kimi K3 copy-on-write state for prefix hits. I found no paper that uses the O(1) state to make rollout fork, migration, preemption-swap or cross-iteration partial-rollout persistence cheap and evaluates it in an RL loop. That remains the clearest open niche.

### Cited Findings
- State-size facts: the SGLang blog says the SSM state "is of fixed size and is updated in place". Memory is O(1) vs O(N) for attention. — [SGLang blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/). Nemotron-Nano-12B-v2 state is ~2.57 MiB per sequence; "For a 128k sequence, the KV cache can be nearly 200 times larger than the Mamba state." — [vLLM blog](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/). Kimi K3 KDA state blocks are ~54 MB vs MLA KV blocks ~27 KB, and "KDA state cannot be sharded by token position". — [LMSYS K3 blog](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)
- **LongStraw** (arXiv 2607.14952, Jul 2026, preprint): for long-context GRPO, it captures the shared prompt without autograd, "keeps only the state the architecture requires", and restores it for each group member. That is the compact recurrent state for Qwen3.6-27B (48 GDN + 16 FA layers) plus CP-sharded KV pages. Old/reference branches are scored without a graph, and responses are replayed one at a time. Trainer side, beyond 2M tokens. — [LongStraw abstract](https://arxiv.org/abs/2607.14952)
- **HARTS** (arXiv 2608.28158, v1 28 Aug 2026, v2 26 Oct 2026, preprint, no venue): a prefix-sharing training system for agentic RL rollout trees on hybrid-attention models. It uses "boundary-state handoff and bounded replay": the KDA kernel exports boundary states during its chunkwise recurrence, and branches replay fewer than one chunk from the nearest chunk anchor. It reports 4.77–10.45× update throughput vs trajectory-expanded training on SWE-bench (Ling-3.0 7.9B/124B hybrid KDA/MLA MoE, 8–64 GPUs). Without replay, symmetric KL at handoff depth 4–5 exceeds baseline-repeat error by 32–36%. Built on AReaL + Megatron; rollouts use partial rollout with staleness 1. — [HARTS abstract](https://arxiv.org/abs/2608.28158); [HARTS HTML](https://arxiv.org/html/2608.28158)
- **Kimi K3** partial rollouts: KDA states are offloaded to and prefetched from a CPU-DRAM pool together with MLA KV blocks. The pool is released after each rollout iteration, so state does not persist across policy versions. — [Kimi K3 §5.3.1](https://arxiv.org/html/2607.24653v1)
- **Keeping stale caches across weight updates**: Nemotron 3 Super does "not recompute the KV cache after updating the model weights". The TorchTitan RL study keeps the cache across weight swaps, while AReaL drops and recomputes. — [Nemotron 3 Super §3.2.4](https://arxiv.org/html/2604.12374v1); [Yichuan blog](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)
- **Cheap state copy in engines**: SGLang copies state on a radix hit and forks a checkpoint on insert. Its spec decoding gives each draft token its own Mamba slot. — [SGLang blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/). K3 on SGLang uses "copy-on-write, snapshot, and donate moves", "race-free by construction". — [LMSYS K3 blog](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)
- **Migration and preemption (non-hybrid baseline)**: RLBoost (arXiv 2510.19225, v3 Apr 2026) migrates in-flight generations at token granularity between preemptible instances via re-prefill. It is not hybrid-specific. — [RLBoost](https://arxiv.org/html/2510.19225v3) (search snippet)
- **Swap vs recompute**: a July 2026 blog argues that for SSM state, swapping is almost always preferable to recompute because the buffer is small and fixed. This is opinion; no system or measurement. — [ecitis blog](https://blog.ecitis.org/hybrid-ssm-attention) (search snippet, low-authority)
- **ReplaySSM** (Tri Dao blog, ~Jun 2026; arXiv 2610.13778 per search): represents state as a checkpoint plus a buffer of recent inputs, cutting state writes. It allows 3.0–3.3× more concurrent requests under spec decoding at fixed memory. It argues "Decoding is the bottleneck for RL post-training and inference serving" but does not study forking or parallel sampling. — [ReplaySSM blog](https://tridao.me/blog/2026/replayssm/)
- vLLM RFC #39647 proposes deferring the n>1 split until after prefill so children share one prefill (attention KV, not hybrid-specific). — [vLLM RFC #39647](https://github.com/vllm-project/vllm/issues/39647) (search snippet)

### Inferences
- A hybrid request is mostly fixed-size state plus a small attention KV (e.g., 3:1 GDN:attention). So per-request migration and swap costs are dominated by the attention layers' KV, plus a constant multi-MB recurrent state at turn or chunk boundaries. This shifts the swap-vs-recompute and migrate-vs-re-prefill trade-off toward moving state, especially for long reasoning and agentic rollouts, where re-prefill costs O(L) per recurrent layer. No RL system has quantified this.
- Fork for n samples is the cheapest case: one prefill, then n copies of an O(1) state plus shared attention KV. Engines currently do it only through prefix-cache machinery with block or step granularity restrictions (vLLM same-step limitation), not as an explicit fork primitive in RL frameworks.
- Persisting partial-rollout state across policy versions conflicts with exact on-policy-ness. The state was produced by old weights and cannot be "patched" like KV positions. Nemotron and TorchTitan already accept stale KV/state in async RL, so the open question is the accuracy effect for recurrent vs attention state, which nobody has measured.
- Trainer-side and rollout-side state reuse (LongStraw/HARTS vs engine prefix caches) are designed separately. A system that hands rollout-time recurrent checkpoints to the trainer (boundary states at turn or fork points) would connect the two. I found no such work.

### Gaps
- No paper evaluates hybrid-aware preemption (swap state vs recompute), migration of hybrid rollouts across GPUs, or long-tail rebalancing that ships recurrent state, inside an RL loop.
- I could not open the ReplaySSM arXiv paper (2610.13778) to check for RL experiments.
- I did not open Hypic (arXiv 2607.01299) or LinearKV (arXiv 2608.11231), position-independent caching for hybrids, for RL relevance.

---

## Q5. Prefix caching for hybrids (Marconi, vLLM align mode, SGLang Mamba radix cache) and interaction with GRPO groups and multi-turn agentic RL

### Takeaway
Hybrid prefix caching exists and is improving, but only at coarse checkpoint granularity. Marconi frames the problem: in-place state updates give all-or-nothing reuse. vLLM stores state at block-aligned step ends; SGLang uses radix copy/fork, branch points and a unified tree; Kimi K3 checkpoints at turn boundaries. For RL, multi-turn agentic rollouts reuse well (Fireworks reports ~99% hit rate on K3 multi-turn rollouts). GRPO group sharing is the weak spot because samples start in the same step. Caches must also be salted or scoped per policy version.

### Cited Findings
- **Marconi** (MLSys 2025; arXiv 2411.19379): SSM state updates in place, so reuse is all-or-nothing. It admits only states likely to be reused and uses FLOP-aware eviction. Reports up to 34.4× higher token hit rate and 71.1% (617 ms) lower TTFT vs vLLM/SGLang extended to hybrids. — [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html) (search snippet); [arXiv](https://arxiv.org/html/2411.19379v3)
- vLLM PR #37898 implements Marconi-style admission (mentioned in RFC #45702 comments). — [vLLM RFC #45702](https://github.com/vllm-project/vllm/issues/45702)
- vLLM align mode and the same-step limitation: see Q3. — [RFC #45702](https://github.com/vllm-project/vllm/issues/45702), [PR #61192](https://github.com/vllm-project/vllm/pull/61192)
- SGLang's MambaRadixCache keeps two LRU lists. KV must be evicted leaf-to-root, but Mamba states can be evicted from any node. Benchmark: Qwen3-Next-80B FP8 with 50 groups × 10 prompts sharing a 10K system prompt. — [SGLang blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- SGLang Unified Radix Cache: session-aware configuration cut TTFT by 2.9–16.6% vs HiRadixCache LRU on SWE-bench runs. — [LMSYS URC blog](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/) (search snippet)
- Kimi K3 checkpoints recurrent state at a sparse subset of MLA hash endpoints, usually turn boundaries. "a checkpoint is either hittable in every group or in none." — [Kimi K3 §5.4.1](https://arxiv.org/html/2607.24653v1)
- Fireworks K3 RL rollouts reach ~99% cache hit rate in long-context multi-turn rollout benchmarks. "Cache reuse is scoped to the correct adapter and policy version." — [Fireworks blog](https://fireworks.ai/blog/kimi-k3-on-fireworks-fast-rl-at-frontier-scale)
- The TorchTitan RL + vLLM study runs `--mamba-cache-mode align` with the prefix cache "salted per group on weight sync". — [Yichuan blog](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/)
- LMCache treats the recurrent state as an opaque page so hybrid prefix reuse works without model-specific code. — [LMCache docs](https://docs.lmcache.ai/mp/hybrid_models.html) (search snippet)
- The MiniMax-M2 post: "In real-world applications, the cache-hit rate for conversations is very high", so a new architecture must handle prefix caching well. — [MiniMax blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)
- Trainer-side prefix sharing for hybrid rollout trees is covered by HARTS, with related work Tree Training, AReaL-DTA, psRL and Prefix Grouper (mostly attention-only), and by LongStraw for the GRPO group prompt. — [HARTS](https://arxiv.org/html/2608.28158); [LongStraw](https://arxiv.org/abs/2607.14952)

### Inferences
- GRPO pattern (one prompt, n samples, same step): with attention-only models the engine shares prompt KV blocks automatically. With hybrids in vLLM, the prompt's recurrent state may be computed once per sample unless a warm-up or fork path is used (#45702, #61192). That wastes compute that grows with prompt length × n.
- Multi-turn agentic RL suits hybrid prefix caching well, because reuse points are turn boundaries, matching Kimi K3's checkpoint placement and Fireworks' ~99% hit rate. Checkpoint memory (tens of MB per request for K3) then becomes a capacity problem, which is where Marconi-style admission and eviction applies.
- Every weight update invalidates prefix caches semantically. RL systems either salt or scope by version (TorchTitan, Fireworks) or knowingly reuse stale state (Nemotron/TorchTitan async). Cross-iteration hybrid caching policy is unexplored.

### Gaps
- I found no measurement of prefill redundancy within GRPO groups for hybrid models in verl, NeMo-RL or slime, and no doc stating whether their rollout path dedups the group prefix for Mamba/GDN layers.
- I did not find Marconi follow-ups that evaluate RL rollout traces specifically.

---

## Q6. Speculative decoding / MTP for hybrid models in rollouts

### Takeaway
Spec decoding for hybrids is now engineered: per-draft state slots in SGLang, STree tree verification, ReplaySSM input-replay rollback, KVBuffer, and MTP heads in Qwen3-Next/3.5 and Nemotron 3. Qwen3.5's RL framework and Kimi K3 RL rollouts (on Fireworks) use it. But the only RL-rollout study with hybrid targets finds smaller gains than on dense models, and the authors name linear-attention targets as future work. vLLM still had hybrid spec-decode correctness bugs in Oct 2026.

### Cited Findings
- Why standard spec decoding fails on SSMs: state updates in place, so rejected tokens cannot be rolled back, and the EAGLE-tree mask is incompatible. SGLang gives each draft token its own Mamba cache slot and promotes the last accepted one; for top-k >1 trees it traces parents. Qwen3-Next-80B FP8 at batch 1: 257 → 307 → 325 tok/s, with accept lengths 2.71 / 3.41 / 4.23. — [SGLang blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- **STree** (NeurIPS 2025; arXiv 2505.14969): tree-based speculative decoding for SSMs and hybrids, using the structure of accumulated state-transition matrices. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a7efd7734bb2e79997f8d1137ece3f09-Abstract-Conference.html)
- **ReplaySSM** (Tri Dao blog ~Jun 2026; NVIDIA + Dao AI Lab; arXiv 2610.13778 per search): "Rollback becomes a buffer operation." Spec decoding runs 1.87–1.96× over standard decoding and up to 2.14× over vLLM's baseline speculative path, on Nemotron-3 and Qwen3.5 (4B to 550B), with FP32 SSM state and MTP drafters. Per a secondary summary, public vLLM v0.31 exposes it only for Mamba-2 behind an experimental flag, and the same flag enables a "RecoverSSM" path for Kimi-K3. — [ReplaySSM blog](https://tridao.me/blog/2026/replayssm/); [PyTorch-KR summary](https://discuss.pytorch.kr/t/replayssm-ssm-nvidia/12174) (secondary)
- **KVBuffer** (arXiv 2605.19049, May 2026, preprint): buffers recent K/V so drafts are verified "without storing temporary states". Up to 45.17% lower linear-attention decode latency and 5× more max requests with 4-token verification, implemented in SGLang for Qwen3-Next. — [KVBuffer abstract](https://arxiv.org/abs/2605.19049)
- **Online Draft Co-Training for spec decoding in RL** (arXiv 2609.07108, Sep 2026, tech report): hybrid targets show weaker gains. Qwen3.5-35B-A3B with DFlash: accept length 4.58 but only 1.50× rollout / 1.46× E2E. Nemotron-3.5-Lightning-30B-A3B with DSpark: 1.19× / 1.16×. Qwen3.5-122B-A10B: 1.72× / 1.35×. Dense Qwen3-8B: up to 2.23×. The authors say for linear-attention models "the relative speedup from speculation is inherently smaller" and list linear-attention models, "where its current benefits are limited", as future work. — [arXiv 2609.07108](https://arxiv.org/html/2609.07108)
- **NeMo-RL speculative rollouts** (arXiv 2604.26779, Apr 2026, preprint): 1.8× rollout throughput at 8B under sync RL and a projected 2.5× E2E at 235B with async RL. The abstract reports no hybrid targets. — [arXiv 2604.26779](https://arxiv.org/abs/2604.26779)
- Qwen3.5's RL framework lists "speculative decoding" among its throughput techniques. — [Qwen3.5 blog](https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894). verl v0.8.0 adds Qwen3.5 MTP SFT/RL (#5898). — [verl v0.8.0](https://newreleases.io/project/github/verl-project/verl/release/v0.8.0). vime PR #485 adds a rollout spec-decode benchmark with an MTP example on Qwen3.5-4B (GPU tests not yet run). — [vime PR #485](https://github.com/vllm-project/vime/pull/485) (search snippet)
- Kimi K3 RL rollouts on Fireworks: KDA state advances "only through the accepted prefix" via ReplaySSM. On a 16×GB300, 128-trajectory, 50-turn synthetic rollout workload, EAGLE3 gave +49.37% complete-run throughput and ReplaySSM +65.05% (including the freed cache capacity). Acceptance was up to 12% higher than the SGLang DSpark baseline. — [Fireworks blog](https://fireworks.ai/blog/kimi-k3-on-fireworks-fast-rl-at-frontier-scale). On SGLang, ReplaySSM gives "a roughly 32x cut in draft-window memory" for K3 (512 KB → 16 KB). — [LMSYS K3 blog](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)
- Nemotron 3 Super adds an "MTP healing" stage after RL (only the MTP heads are trained). Nemotron 3.5 uses a GenRM with MTP inference to speed RLHF. — [Nemotron 3 Super §3.2](https://arxiv.org/html/2604.12374v1); [Nemotron 3.5 report §3.5](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)
- vLLM correctness: #62304 (n-gram on Qwen3.5, GDN state misses an accepted token's update, v0.31.0; clean on v0.32.0), #53912 (MTP + prefix caching corrupts hybrid output), #61738 (wide verification on GDN). — [vLLM #62304](https://github.com/vllm-project/vllm/issues/62304)

### Inferences
- Hybrids get less from speculation in rollouts for two reasons. Decode per token is already cheaper, being memory-bound on a fixed state rather than a growing KV. And verification must materialize per-draft states or replay inputs. ReplaySSM/KVBuffer-style techniques are the enabling primitives, but nobody has put them into an RL rollout study with a matched dense baseline.
- Spec-decode state bugs are themselves a source of off-policy data. A wrong state update changes the sampled distribution without any visible error. RL frameworks do not check for this.

### Gaps
- No RL paper isolates spec decoding on a hybrid policy whose MTP head drifts during RL (Nemotron's "MTP healing" hints at the drift but gives no rollout-time acceptance curve).
- I could not find whether Qwen3.5's RL framework uses MTP (bundled heads) or a separate drafter.

---

## Q7. Overall: is "RL rollouts for hybrid models" an open systems research niche?

### Takeaway
Yes, with a caveat. The niche is open for rollout-side systems work that exploits the O(1) recurrent state: fork for n samples, swap and migrate under preemption or long-tail rebalancing, cross-iteration partial-rollout state, and cheap train/rollout parity for chunked-vs-recurrent kernels. It is filling quickly on adjacent fronts: trainer-side prefix sharing (HARTS, LongStraw), engine prefix caching (Marconi, SGLang URC, vLLM align/partial hits, Kimi K3), spec decoding (ReplaySSM, STree, KVBuffer), and mismatch studies (Wang, Panda, Zhang). Industry (Moonshot, NVIDIA, Qwen, Fireworks) ships pieces without papers on rollout-side state management.

### Cited Findings
- Published hybrid-specific RL systems: HARTS (trainer-side, Aug–Oct 2026, preprint) and LongStraw (trainer-side, Jul 2026, preprint). — [HARTS](https://arxiv.org/abs/2608.28158); [LongStraw](https://arxiv.org/abs/2607.14952)
- Industrial hybrid RL rollout infra appears only inside model reports or vendor blogs: Kimi K3 (KDA offload, KDA prefix cache), Fireworks K3 (ReplaySSM, cache scoping, delta weight sync), and Qwen3.5 (async disaggregated RL with spec decoding, no details). — [Kimi K3](https://arxiv.org/html/2607.24653v1); [Fireworks](https://fireworks.ai/blog/kimi-k3-on-fireworks-fast-rl-at-frontier-scale); [Qwen3.5 blog](https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894)
- Engine maintainers list hybrid prefix-cache granularity, same-step reuse, spec-decode adaptation and determinism as open or recently fixed items. — [vLLM RFC #45702](https://github.com/vllm-project/vllm/issues/45702); [SGLang #12867](https://github.com/sgl-project/sglang/issues/12867)
- The authors of the RL spec-decoding study list linear-attention targets as future work. — [arXiv 2609.07108](https://arxiv.org/html/2609.07108)
- MiniMax cites immature linear-attention infra as a reason to abandon hybrids for M2. — [MiniMax blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)

### Inferences
These are candidate research directions, ranked by how open they look.
1. **Hybrid-aware rollout scheduling, preemption and migration.** Swap or ship the fixed recurrent state plus the small attention KV instead of re-prefilling, for long-tail rebalancing (cf. RLBoost re-prefill) and partial rollouts. Nothing hybrid-specific has been published.
2. **Explicit fork for GRPO groups and agentic branching on the rollout side.** One prefill, n copies of the state, shared attention KV. This addresses the vLLM same-step limitation, and the fork-point states could be handed to a HARTS/LongStraw-style trainer.
3. **Cross-iteration state reuse under async RL.** Measure the accuracy cost of stale recurrent state vs stale KV, which Nemotron 3 Super and TorchTitan already accept implicitly. Design version-aware checkpoint policies; Kimi K3 currently clears its pool each iteration.
4. **Cheap train/rollout parity for chunked-vs-recurrent kernels.** Current costs are 20–40% sampling throughput (Panda) and 2–5× with full BI (Wang). Benefits are unproven, so a study pairing a cheaper parity kernel with RL-outcome evaluation would add to both lines of work.
5. **Spec decoding for hybrid rollouts.** Gains are smaller (1.19–1.72× rollout vs up to 2.23× dense in 2609.07108), and MTP drift during RL is unstudied. This is more incremental given ReplaySSM/STree/KVBuffer.
- Risks for a PhD project: fast-moving engines (vLLM 0.30–0.32 alone changed hybrid spec-decode and prefix-cache behavior), large-lab competition (NVIDIA/Moonshot/Qwen), and the need for multi-GPU RL runs to show end-to-end effect. Defensible contributions are probably mechanism plus measurement on open models (Qwen3.5/3.6, Nemotron 3 Nano, Kimi Linear) in an open stack (verl/slime/NeMo-RL + vLLM/SGLang).

### Gaps
- I found no survey of RL rollout efficiency that treats hybrids as a category. The taxonomy arXiv 2609.25463 surfaced in search but was not opened.
- I did not check ICML/NeurIPS/MLSys/OSDI 2026 accepted-paper lists for unpublished-on-arXiv hybrid RL systems work; venue status for 2026 items is arXiv-only unless stated.
