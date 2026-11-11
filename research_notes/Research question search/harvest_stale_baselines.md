# Harvest: headline gains that may not survive a current baseline

Date: 2026-11-11. Lens: accepted 2024-2026 results measured against an old engine version or a weak
baseline, never independently reproduced, or reported by later papers as shrinking. Question form:
how much of paper P's gain remains against a current engine with today's defaults, and what absorbed
the rest?

Reference engine for "current": vLLM 0.32 (the version already profiled in this repo).
Hardware assumed: up to 4 idle 48 GB cards (never GPU 0), no NVLink, open models up to about 30B.

Dropped directions referred to below: D1 decode-time KV page selection; D2 state checkpoints and
prefix caching for hybrids; D3 profiling vLLM 0.32 for single-GPU bottlenecks; D4 RL rollout
acceleration.

Five cards. Cards are not ranked. "Verified" means the primary source was opened this session and
the quote was read there.

---

## Card 1. KV compression throughput for reasoning models

**Question.** On two 48 GB cards with a current vLLM baseline that uses FP8 weights and the FP8 KV
cache, how much of the 5.4x to 6.6x throughput gain reported for KV compression on reasoning models
(DiffKV, R-KV, ThinKV) remains when throughput is counted as correctly solved problems per second,
and is the rest absorbed by the stronger memory baseline or by longer generations?

**Tension.** Stale or weak baseline (FP16 weights, unquantized KV, old vLLM commit) plus a later
paper reporting that the gains shrink in production engines.

**Evidence.**
- DiffKV (SOSP'25 per our catalogue), https://arxiv.org/html/2412.03131 (verified):
  "DiffKV achieves a remarkable 5.4× higher throughput over vLLM on the QwQ-32B thinking model";
  "DiffKV sustains a batch size of 15.9, far exceeding vLLM's batch size of 2.7.";
  "model weights are stored in FP16 precision";
  "We measure the throughput and latency on NVIDIA L40 GPUs, each with 48 GB of memory".
  The vLLM baseline version is not stated in the paper text I could read. vLLM's own fp8 KV cache
  is not a baseline.
- DiffKV repository, https://github.com/zyqCSL/DiffKV (verified, README): "DiffKV is built on top
  of vLLM (commit" 1db83e3. That commit is vLLM PR #2246 ("[Docs] Update installation instructions
  to include CUDA 11.8 xFormers"). The PR number places it around the v0.2.x releases (my
  inference from PR numbering; the commit page showed no date).
- R-KV (NeurIPS'25 per our catalogue), https://arxiv.org/html/2505.24133 (verified):
  "a 90% memory saving and a 6.6×" throughput; single A100 80G; throughput derived from maximum
  batch size; Appendix D says benefits "might be less pronounced" in serving frameworks with paged
  attention.
- ThinKV (arXiv 2510.01290 v2, venue not shown), https://arxiv.org/html/2510.01290 (verified):
  "pure quantization can inflate generation length by up to 5.1×"; "substantial generation-length
  inflation negates most of the compression gains, yielding only a modest 1.1× improvement in
  throughput"; headline "up to 5.8× higher inference throughput over SoTA baselines".
- Rethinking Key-Value Cache Compression Techniques for LLM Serving (MLSys'25),
  https://arxiv.org/html/2503.24000 (verified): with FlashAttention and PagedAttention "the
  performance gains from KV cache compression diminish further."; "show negative computational
  efficiency in certain scenarios of batch size, sequence length, and tensor parallelism.";
  response length grows 1.55x-1.76x (Table 4). Scope: LLaMA and Mistral, LMDeploy, at most 1024
  generated tokens, four A6000 cards. No reasoning models.
- vLLM docs, https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html (verified):
  FP8 KV cache "can significantly reduce its memory footprint" (flag `kv_cache_dtype`).
- Own measurement (this repo, E3/H3): accuracy losses under KV page selection were "mostly
  truncation at 16K" / 32K caps, which is the same length-inflation symptom.

**My arithmetic (inference, not a source).** QwQ-32B in FP16 is about 65 GB. On 2x48 GB at 0.9
utilisation that leaves about 21 GB for KV. FP8 weights (about 33 GB) leave about 53 GB, roughly
2.5x more. FP8 KV doubles tokens per GB. The product is about 5x more KV tokens with no
compression algorithm at all, which is the same size as the 5.4x headline. Whether capacity
converts to throughput at that batch size is the open part.

**Why unknown.** Each paper compares against FP16 weights and uncompressed KV. The MLSys'25
re-measurement used non-reasoning models with at most 1024 output tokens, where memory is not the
bottleneck. ThinKV shows length inflation for quantization but uses its own baselines and reports
tokens per second. Nobody reports solved problems per second against FP8+fp8 on a current engine.

**If most of the gain remains.** KV compression is a real serving lever for reasoning models even
after FP8; the interesting follow-up is why eviction does not inflate length while quantization
does (ThinKV's claim) and what the limit is.
**If most is absorbed.** The field's throughput numbers for reasoning-model KV compression are
mostly a memory-accounting artefact; the useful finding is the decomposition (weights, KV dtype,
length inflation, residual) and a metric (correct answers per GPU-second) that later papers should
report. This answer is also publishable as a measurement paper.

**First experiment.** QwQ-32B (and one hybrid, Qwen3.5-27B-FP8, as a contrast) on two 48 GB cards,
MATH-500 and AIME, 16K generation cap. Arms: vLLM 0.32 FP16 weights + bf16 KV; FP8 weights;
FP8 weights + fp8 KV; DiffKV from its repository at the paper's thresholds (needs Ada cards per
its README). Measured: sustained batch size, output tokens per request, accuracy, and correct
answers per GPU-second.

**Closest existing answer.** Rethinking KV Cache Compression (MLSys'25): shows the shrink for
non-reasoning models in LMDeploy; no FP8 baseline, no long outputs. ThinKV: length inflation for
quantization only. No paper found that re-measures DiffKV or R-KV.

**Overlap with dropped directions.** D1: same family (KV reduction for long reasoning); D1 asked
whether a selector beats Quest on accuracy, this asks about the serving-throughput accounting.
Partly D3 (capacity numbers already measured for some models).

---

## Card 2. Tool-call KV retention versus the default prefix-cache LRU

**Question.** With vLLM's default automatic prefix caching as the baseline (freed blocks stay
cached until the LRU reuses them), how much of the 1.1x-3.7x gain reported for keeping KV across
tool calls (InferCept, Continuum) remains for thinking-model agent loops on a 48 GB card, and at
what KV-memory pressure does the default LRU start losing paused agents' prefixes?

**Tension.** The paper's description of the baseline contradicts the engine's design document;
a later measurement study shows the failure mode exists but tests no fix.

**Evidence.**
- InferCept (ICML'24), https://arxiv.org/abs/2402.01869 (verified, abstract page only; PDF text
  not extractable here): "causing unnecessary recomputation of already computed contexts, which
  accounts for 37-40% of total model forwarding time."; "InferCept improves the overall serving
  throughput by 1.6x-2x and completes 2x more requests per second." Baseline vLLM version not
  recovered.
- Continuum (arXiv 2511.02230 v2, no venue), https://arxiv.org/html/2511.02230v2 (verified):
  "Continuum reduces delay by 1.12x to 3.66x and improves throughput by 1.10x to 3.22x on
  multi-turn agentic workloads."; "We use the stable release of vllm 0.10.2 with default setting.";
  "vLLM or SGLang will evict a request's KV cache as soon as decoding finishes, implicitly assuming
  the request is complete." Does not say whether prefix caching was on. A100/B200/H100.
- vLLM prefix caching design doc, https://docs.vllm.ai/en/latest/design/prefix_caching.html
  (verified): "When a request is finished, we free all its blocks if no other requests are using
  them (reference count = 0)."; "Blocks 2, 3 and 4 are added to the free queue in the reverse
  order (but block 2 and 3 are still cached)."; "When the head block (least recently used block)
  of the free queue is cached, we have to evict the block." No pinning, no TTL.
- IISWC'26 agentic workload characterisation, https://arxiv.org/html/2605.26297 (verified;
  vLLM v0.20.0, Qwen3.6-27B, TP2, 2x H100): "Context thrashing first appears as a sharp
  prefix-cache hit-rate drop, from about 98% to 16–26%, shifting the workload toward prefill.";
  "reducing average generation throughput by up to 66%". No mitigation evaluated.
- Own measurement (this repo, K0): a perfect prefix-cache policy cut prefill time only 2.6-5.5%
  against the best vLLM setting on Mooncake traces, where outputs are 2-3% of input. Noted then:
  "Only untested niche: thinking-model tool loops (long outputs)."

**Why unknown.** Two published results point opposite ways: K0 and the design doc say the default
LRU already retains nearly everything; IISWC'26 says hit rate collapses to 16-26% under
concurrency. The two differ in output length and memory pressure, and neither sweeps that axis.
Continuum's baseline may or may not have had prefix caching on.

**If a large gain remains.** There is a specific regime (long thinking outputs, many paused
agents, small KV pool) where LRU is the wrong policy, and the pressure threshold tells you what
a policy must know (tool duration, context growth).
**If little remains.** Retention papers measured against a baseline the engine no longer has, and
the remaining problem is admission (how many agents to run), not eviction. This answer is less
exciting but closes the K0 niche.

**First experiment.** Qwen3-8B and Qwen3.5-9B in thinking mode on one 48 GB card; replay
mini-swe-agent SWE-bench trajectories and BFCL v4 with recorded tool latencies; sweep concurrent
agents 4 to 64. Measured: prefix-cache hit rate after each tool return, recomputed prefill tokens
as a share of GPU time, and job completion time, each against an infinite-memory lower bound
(same trace at a concurrency where nothing is evicted).

**Closest existing answer.** IISWC'26 study (documents thrashing on H100, no fix, no pressure
sweep on small cards). Continuum (fix, but baseline unclear). K0 in this repo (short outputs).
No independent re-evaluation of Continuum or InferCept found.

**Overlap with dropped directions.** D2, strongly: it is the one niche K0 left untested. Slight
D4 (preemption and watermark interact with retention).

---

## Card 3. SpecReason under concurrent load

**Question.** Does SpecReason's 1.4x-3.0x single-request latency gain become a throughput gain,
shrink to parity, or turn into a loss when the small and base models share 48 GB cards under
concurrent load on a current vLLM?

**Tension.** Never measured under batching; the token-level analogue is known to shrink with
batch size; the paper's own stronger-hardware run already shrank.

**Evidence.**
- SpecReason (NeurIPS'25), https://arxiv.org/html/2504.07891v2 (verified): "SpecReason achieves
  1.4−3.0× speedup over vanilla LRM inference while improving accuracy by 0.4−9.0%"; "We run our
  evaluations on two NVIDIA A6000-48GB GPUs."; "We use vLLM ... 0.8.2 as the underlying inference
  engine and enable prefix caching."; "Inference is performed sequentially: the small and base
  models take turns, avoiding kernel-level interference."; on 4x A100 with R1-70B the speedup is
  1.5x instead of 1.9x because "the smaller model R1-1.5B sees only a modest 1.1× TPT improvement
  on stronger hardware, which narrows the performance gap."
- Speculative Decoding: Performance or Illusion? (MLSys 2026), https://arxiv.org/html/2601.11580
  (verified; vLLM v0.10.1.1, H100): larger batches "systematically reduces the relative speedup of
  speculative decoding."; EAGLE on GSM8K with Llama3.1-8B goes from 1.73x at batch 1 to 1.21x at
  batch 128. Token-level only; step-level methods not covered.

**Why unknown.** SpecReason's gain is a ratio of per-token times between a 1.5B and a 32B model
at batch 1. Under load the base model's per-token cost per request falls with batching, the two
models compete for the same memory and SMs, and the base model must re-read each speculated step
(prefill work that scales with concurrency). The sign of the net effect on requests per second
is not derivable from the paper.

**If the gain holds.** Step-level speculation is a capacity technique, not only a latency one,
and the mechanism (cheap judge passes ride on prefix caching) is worth isolating.
**If it shrinks or reverses.** It joins token-level speculative decoding as a low-load technique;
the crossover concurrency and its cause (memory split, verification prefill, interference) is the
finding.

**First experiment.** QwQ-32B (FP8) with R1-1.5B or Qwen3-1.7B on two 48 GB cards, using the
authors' released code if it runs on vLLM 0.32; AIME and MATH-500, 8192-token budget; closed-loop
concurrency 1, 4, 16, 64. Measured: correct answers per second and accuracy for vanilla base
model versus SpecReason at each concurrency.

**Closest existing answer.** None found for SpecReason under batching. Nearest: the MLSys'26
paper above (token-level), "Scaling Up, Speeding Up" (ICLR 2026, arXiv 2509.04474; token-level SD
for test-time scaling; not opened in full), SSR (arXiv 2505.15340) and SparseSpec.

**Overlap with dropped directions.** None. (SpecReason is on the literature-map shortlist.)

---

## Card 4. NanoFlow's optimal-throughput gap on a current engine

**Question.** On a single 48 GB card, what fraction of NanoFlow's compute-bound optimal
throughput does vLLM 0.32 reach on a mixed prefill-plus-decode workload, and is the remaining gap
prefill/decode interference or something else?

**Tension.** Baseline four engine generations old; two later papers report NanoFlow losing to
plain chunked prefill.

**Evidence.**
- NanoFlow (OSDI'25), https://arxiv.org/html/2408.12757v2 (verified): baseline vLLM
  "v0.5.3.post1 (commit ID 38c4b7e)"; "vLLM, DeepSpeed-FastGen, and TensorRT-LLM achieve only
  22.0%, 22.9%, and 37.8% of the optimal throughput, respectively."; NanoFlow 68.5% of optimal;
  "8×A100 80GB SXM GPUs interconnected via NVLink".
- vLLM v0.6.0 performance update, https://vllm.ai/blog/2024-09-05-perf-update (verified):
  "vLLM achieves 2.7x higher throughput and 5x faster TPOT (time per output token) on Llama 8B
  model"; "1.8x higher throughput and 2x less TPOT on Llama 70B model." (versus v0.5.3).
- MuxWise (ASPLOS'26), https://arxiv.org/html/2504.14489 (verified; SGLang 0.4.10post2):
  "NanoFlow performs worse than the original chunked-prefill."; "NanoFlow outperforms
  chunked-prefill only on ShareGPT."
- Bullet (ASPLOS'26), https://arxiv.org/html/2504.19516 (verified; vLLM v0.8.5): NanoFlow
  "5.2% lower SLO compliance and 20.0% lower throughput compared to Bullet."; chunked prefill
  "inevitably compromises throughput".
- Own measurement (this repo, D3): decode at 0.89-0.95 of the hardware roofline on vLLM 0.32;
  prefill GEMMs at about 0.70-0.72 of bf16 peak.

**Why unknown.** The 22% figure is against v0.5.3.post1. 22% times 2.7x is about 59%, before
five more minor versions of work. No paper recomputes the bound for a current engine, and none
does it on a mid-range card where Bullet says the SM budget changes the trade-off. D3 measured
decode-only and prefill-only shares, not the mixed case.

**If a large gap remains.** There is headroom specifically in mixed steps, which names the
mechanism (interference in chunked prefill) and where to look.
**If the gap is small.** NanoFlow's premise no longer holds on current engines; a clean negative
with the bound recomputed. Given D3's numbers this is the more likely answer and it is the boring
one.

**First experiment.** Qwen3-8B and Qwen3-30B-A3B-FP8 on one 48 GB card; NanoFlow's cost model
recomputed for the card; ShareGPT, LMSYS-Chat and a constant 1024-in/512-out workload at
saturation. Measured: tokens per second as a fraction of the bound, split into decode-only steps,
prefill-only steps and mixed steps.

**Closest existing answer.** MuxWise and Bullet (NanoFlow loses to chunked prefill under SLOs on
A100s; neither reports distance to the bound). D3 in this repo (single-phase rooflines).

**Overlap with dropped directions.** D3, strongly: it is D3's method applied to mixed workloads.

---

## Card 5. Speculative decoding when batch size is capped by KV memory

**Question.** When the batch on a 48 GB card is capped by KV memory rather than by compute (long
reasoning outputs), does draft-based speculative decoding (MTP or EAGLE-3) raise or lower served
requests per second on vLLM 0.32, and does the sign differ between dense and hybrid models?

**Tension.** Two accepted claims disagree on the regime: speculative decoding loses at large
batch because the engine is compute-bound; reasoning workloads are said to be memory-bound, where
spare compute should make speculation free. A measured anomaly here says the draft state itself
halves capacity on hybrids.

**Evidence.**
- Speculative Decoding: Performance or Illusion? (MLSys 2026), https://arxiv.org/html/2601.11580
  (verified): larger batches "systematically reduces the relative speedup of speculative
  decoding."; reasoning runs (Qwen3-8B thinking) only at batch 1-16; no hybrid models.
- SparseSpec (arXiv 2512.01278, no venue listed), https://arxiv.org/html/2512.01278 (verified):
  long reasoning chains "shift the performance bottleneck of inference from compute-bound to
  memory-bound"; KV loading "accounting for over 70% of the end-to-end latency." at batch 128 on
  H100; baseline vLLM v0.11.0; up to 2.13x.
- NanoFlow (OSDI'25), https://arxiv.org/html/2408.12757v2 (verified): "end-to-end LLM serving is
  compute bound for most common workloads and LLMs."
- Own measurement (this repo): with MTP on, the `--watermark` gain on hybrid rollouts fell from
  -17% step time to -1% (1 GPU) and -7% (TP2); "MTP halves engine concurrency on hybrids"
  (per-GDN-group speculative state blocks).
- EAGLE-3 (arXiv 2503.01840): 1.38x at batch 64 in SGLang v0.4.4 (not verified this session;
  number taken from the repo's literature catalogue).

**Why unknown.** The MLSys'26 study stops at batch 16 for reasoning and uses H100s with dense
models. SparseSpec is on H100 with dense Qwen3 and its own self-speculation, not the engine's
shipped MTP/EAGLE path. On a 48 GB card the memory cap arrives at much smaller batch, and on
hybrids the draft state costs capacity. No one has measured where the two effects cross.

**If speculation raises capacity.** Memory-bound reasoning serving has idle compute that
speculation can use; the size of the gain bounds what any compute-for-memory trade can buy.
**If it lowers capacity.** The draft's memory footprint is the binding cost in this regime, which
points at the state layout rather than the acceptance rate.

**First experiment.** Qwen3-8B (EAGLE-3 head) and Qwen3.5-9B (MTP) on one 48 GB card, MATH-500
and AIME with thinking on, 8K-16K caps, concurrency swept until the KV pool is full. Measured:
completed requests per second with and without speculation, the engine's admitted batch size,
and acceptance rate.

**Closest existing answer.** SparseSpec (same regime claim, H100, dense, self-speculation). The
MLSys'26 paper (batch effect, compute-bound regime). D3 notes in this repo ("hybrid MTP beats
dense EAGLE3, no GDN verify bottleneck seen") cover per-step speed, not capacity.

**Overlap with dropped directions.** D3 (speculative decoding was profiled there at the step
level) and D4 (the MTP capacity anomaly was found in rollout runs).

---

## Checked, not carded

- **KV offload versus recompute break-even.** py-kvcache (arXiv 2609.11744; vLLM v0.22, H100):
  required storage rate "from 23.2 GB/s at 1k tokens to 10.4 GB/s at 8k and 3.5 GB/s at 80k.";
  "an 8k prompt requires a 77.8% prefix reuse for breaking even,". HCache (EuroSys'25, arXiv
  2410.05004): "up to 5.73X reduction in TTFT" versus token recomputation. Dropped: prefill rates
  measured here on 48 GB cards (about 6.3K tok/s for Qwen3-8B at 8K) put the break-even load rate
  near 1-1.5 GB/s, so loading wins on this hardware by arithmetic. A Red Hat benchmark
  (2026-11-02) and "Can I Buy Your KV Cache?" (arXiv 2606.13361) also exist.
- **Learning to Rank scheduling** (NeurIPS'24, arXiv 2408.15792; vLLM v0.4.1; 2.8x p90 latency).
  Dropped as crowded: JITServe (NSDI'26, arXiv 2504.20068) re-measured it ("outperforming LTR by
  1.3×–1.7×"), plus PARS (arXiv 2510.03243) and others.
- **Token-level speculative decoding at batch, dense models.** Answered by the MLSys'26 paper.
- **Hybrid long-context capacity claims.** Already measured here (1.16-2.5x).
- **CacheBlend, prefill/decode disaggregation, structured output (XGrammar), Jenga,
  ServerlessLLM, LoRA serving.** Either adopted upstream or already answered; not pursued.

## Not verified

- DiffKV: vLLM baseline version is not in the paper text I could read; the repository names commit
  1db83e3 but I could not confirm its date or release. Per-model speedups (Figure 17) not read.
- InferCept: baseline version and hardware (PDF not extractable; abstract page only).
- Venue labels SOSP'25 (DiffKV) and NeurIPS'25 (R-KV) come from the repo's literature catalogue.
- ThinKV venue unknown. Continuum and SparseSpec are arXiv-only as far as I could see.
- EAGLE-3 batched numbers (catalogue only).
- "Scaling Up, Speeding Up" (arXiv 2509.04474) seen in search results, not opened in full.
- Whether SpecReason's released code runs on vLLM 0.32.
- The memory arithmetic in Card 1 is mine and assumes 0.9 GPU memory utilisation.
