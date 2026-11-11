# Harvest: attention and GPU kernels — the gap between algorithmic savings and realised speed

Date: 2026-11-11. Working notes for the question search. Read-only work: no GPU jobs were run.
Numbers marked "derived" are my own arithmetic from quoted numbers, not statements from a paper.

## 1. Reading list

Depth: S = relevant sections read in full text (arXiv HTML), A = abstract only (via Semantic Scholar batch endpoint), X = seen only in a web-search snippet (not verified).

| id | short name | depth |
|---|---|---|
| 2501.01005 | FlashInfer | S |
| 2402.05099 | Hydragen | S |
| 2408.11049 | MagicDec | S |
| 2503.08311 | Mind the Memory Gap | S |
| 2503.18773 | BitDecoding | S (partly) |
| 2512.01278 | SparseSpec / PillarAttn | S |
| 2605.24168 | Inference Time Context Sparsity: Illusion or Opportunity? | S |
| 2610.04313 | MOIRA | S |
| 2611.09722 | FOSCAR | S |
| 2610.11201 | PageWeaver | S (partly) |
| 2605.01910 | SANTA | S (partly) |
| 2604.22312 | Guess-Verify-Refine top-k (GVR) | S (sec 2.4) |
| 2605.19893 | SSV: Sparse Speculative Verification | S (intro) |
| 2606.26666 | PersistentKV | A + abstract-level grep |
| 2610.33719 | ZipFusion | A + grep |
| 2604.07815 | AsyncTLS | A + intro |
| 2607.27735 | SparseSpec-L | A |
| 2607.21535 | Windowed-MTP | A |
| 2603.14224 | Self-Indexing KVCache | A |
| 2603.01399 | Quasar | A |
| 2607.17283 | Lossless but Not Free (SD on consumer hardware) | A |
| 2603.03251 | Speculative Speculative Decoding | A |
| 2502.10424 | QuantSpec | A |
| 2606.08382 | STAR-KV | A |
| 2605.02568 | StreamIndex | A |
| 2610.28441 | SSA | A |
| 2605.27740 | UNIQUE | A |
| 2604.00235 | MAC-Attention | A |
| 2610.34907 | TMA characterisation | A |
| 2608.02947 | ATFlash | A |
| 2402.15220, 2402.14808 | ChunkAttention, RelayAttention | exist, not read |
| 2407.08608, 2406.10774, 2502.02770, 2502.14866 | FlashAttention-3, Quest, Twilight, LServe | downloaded, not read |
| 2604.05438 | Top-K Retrieval with Fixed-Size Linear-Attention Completion | X |

Local records used (the user's own): `/home/ryankert_l/gemm-kernel/README.md`, `/home/ryankert_l/kv-fetch/kv-fetch/results/profile/findings.md`, `/home/ryankert_l/kv-fetch/kv-fetch/results/profile/NOTEBOOK.md`, project memory note for Phase 0.

## 2. Extracted claims (exact quotes)

**FlashInfer 2501.01005.** "For query tile size 1, we use CUDA Cores template since tensor core instruction m (minimum rows) is 16, and use Tensor Cores for other query tile sizes". Parallel generation: "Peak speedups occur at n=4, where ITL decreases by 13.73% for the 8B model and 17.42% for the 70B model ... Smaller values of n do not benefit from composable formats due to insufficient increase in block size. For larger n, the computation ceases to be dominated by attention processes". Table 5 (kernel latency in us, suffix 128): prefix 1024: composable 45.17 vs single 46.52 (BS=16), 87.86 vs 130.49 (BS=64); prefix 8192: 88.67 vs 226.57 (BS=16), 125.76 vs 931.75 (BS=64); prefix 32768: 217.42 vs 945.67, 254.54 vs 4090. Derived: at prefix 1024 / BS=16 the composable format reads about 6x fewer tokens but is 1.03x faster; at 8192 / BS=16 bytes predict about 14 us, measured 88.67 us.

**Hydragen 2402.05099.** "Hydragen throughput is always within 70% of the no-attention ceiling"; "GPUs with a higher ratio of compute to memory bandwidth benefit more".

**MagicDec 2408.11049.** "The high ratio of peak FLOPS to memory bandwidth in modern GPUs causes the increase in KV loading time with batch size to outweigh the increase in computation time (see Fig. 1). As a result, although compute-bound linear layers increase verification cost, it is mitigated by the KV bottleneck." "Because verification and decoding share the same KV budget, their KV cache loading costs are equivalent."

**Mind the Memory Gap 2503.08311.** "over 50% of the attention kernel cycles stalled due to data access delays"; "DRAM read average throughput remains below 65% across all models during the decoding phase"; replication gain "is primarily due to overlapping GPU gaps during CPU computations ... the CPU time is reduced by an average of 78% across both models when using two replicas".

**BitDecoding 2503.18773.** "Results on Bandwidth-constrained GPU (RTX 4090) ... BitDecoding achieves roughly 4x (4-bit) and over 7x (2-bit) speedups over FlashDecoding-v2". Evaluated "across Blackwell, Hopper, Ada, and Ampere GPU architectures".

**FOSCAR 2611.09722.** "instruction-issue throughput is a scarce resource on modern GPUs". "FlashInfer BF16 exceeds 90% of peak on both H100 and B200." "Among integer 4-bit baselines, BitDecoding is highest at 70.0% on H100 and 36.1% on B200, while its INT2 kernel reaches only 30.7% and 14.5%". "From H100 to B200, peak HBM throughput grows 2.39x, while peak instruction-issue throughput grows only 11%". "reconstruction alone can consume 30% and 60% of H100's peak hardware budget of 9.99 lane-slots/byte". "At peak HBM throughput, INT2 reconstruction exceeds B200's hardware peak of 4.65 lane-slots/byte, and INT4 reconstruction alone consumes 86% of that peak." Footnote assumption: "128 lane-slots per SM cycle". H100: INT2/INT4 geo-mean 3.36x/2.83x over FlashInfer BF16; B200: 2.67x/2.30x, and "1.16x and 1.00x over TensorRT-LLM NVFP4". Evaluated at batch 1, 8, 64; single-token decode only (no multi-row verification found in the text).
Derived for RTX 6000 Ada: 142 SMs x 128 lane-slots x 1.1 to 2.67 GHz / 926 GB/s = about 21 to 52 lane-slots/byte, i.e. 4 to 11 times B200's budget. The clock range is the user's own measurement ("a memory-bound kernel holds 2.67 GHz while cuBLAS SGEMM runs at 1.19 GHz").

**SparseSpec 2512.01278.** "kernel optimized for verification (full) attention achieves near 85% bandwidth but less than 50% for draft (sparse) attention; kernels optimized for sparse attention deliver over 80% bandwidth, but degrade to 50% on full attention". "with k=16, alpha=0.75, s=0.05, the attention latency is reduced by 6.78x" (formula). Table 2 Qwen3-8B: attention 17.1 ms -> 5.2 ms ("reduces the execution time of Attention by 3.29x"), GEMM 7.2 -> 8.9 ms. Acceptance "6.16 out of 8 tokens". Derived: formula at measured acceptance gives (6.16+1)/(8*0.05+1) = 5.1x, measured 3.29x; verification is about 71% of the remaining attention bytes.

**2605.24168 (context sparsity).** "A common belief is that efficiency gains require block-structured sparsity [58]; without it, sparsity is unlikely to translate into real speedups. We present an argument against this view." "This is possible because the KV cache's vector dimension provides sufficient contiguous memory to make such sparsity effective on modern hardware." Table 1 (H100, FP16, Hq=32, Hkv=8, D=128, page 16, 128K): "We generally break-even at 10x sparsity for every batch"; B=1 speedups at 100x/200x/500x sparsity: 10.25 / 11.05 / 11.14 (dense 0.19 ms; derived floor about 17 us). With an indexer: "MHA: break-even at 2x sparsity with 4.17x speedup at 100x sparsity. GQA: break-even at 10x, with 2.81x speed up at 100x". Hybrids: "hybrid architectures such as Qwen3.5 and Gemma3 exhibit significantly greater robustness to context sparsity, maintaining performance even at up to 50x sparsity with little to no degradation in quality"; "restricting attention to just 128 retrieved tokens corresponds to approximately 250x sparsity while still preserving strong performance"; "suggesting that the inclusion of SSM or linear-attention layers may inherently improve resilience to sparse context retrieval". Selection is "exact oracle top-k".

**MOIRA 2610.04313.** "matches dense accuracy while reading about 30% of the pages and reduces ... TPOT by 2.2-2.5x relative to dense FA3"; "Selecting per query head, as Twilight does, needs 1.51x the pages at the same output error"; "FFD scans a 2-bit copy of every key at every step, so keeping 12.5-21.2% of the blocks makes its attention only 1.08-1.20x faster than dense"; "Quest's rule selects per query head, so FlashInfer's paged kernel attends over 32 lists instead of 8 (8.30 vs. 4.89 ms with the CUDA-core kernel, plus a 2.16 ms host plan)"; "Quantized KV caches, tensor parallelism and latent attention are not covered".

**GVR 2604.22312.** "this Top-K stage runs once per decode query and becomes a meaningful latency bottleneck even when the indexer and attention kernels are already highly optimized". Traffic model: indexer N*d_i*2B, radix top-k R*N*4B (R about 3), sparse attention K*d*2B. Atomics reduce "effective SIMT utilization well below the memory bandwidth ceiling".

**SSV 2605.19893.** "speculative decoding ... mitigates the bandwidth bottleneck by increasing arithmetic intensity, while sparse attention ... directly reduces the volume of accessed KV cache." "During verification, the query batch is larger than single-token decoding but still too small to amortize repeated kernel launches and intermediate materialization."

**PageWeaver 2610.11201.** "a small support does not necessarily yield efficient GPU work". B300: "neither online window beats the faster existing backend on any of the eight tested inputs".

**SANTA 2605.01910** (RTX 6000 Ada). 1.50x kernel speedup at 32k while reading "<= 1.56% of value rows"; 1.25x end to end. Derived: keys are still read in full (bound 2x); the value path costs about 17% of a dense pass for under 1% of the bytes.

**PersistentKV 2606.26666** (RTX 3060). "reduces launch fan-out from 16.00 to 2.00 launches per decode step"; 1.044-1.080x on B8 workloads, 1.403x on a B1 trace.

**AsyncTLS 2604.07815.** "While token-level sparse attention offers superior accuracy, its indexing overhead is costly; block-level methods improve efficiency but sacrifice precision."

**User's local records.** Decode step at 0.89-0.92 of the memory roofline; attention "0.99-1.02 of the KV-read bound"; fp8 KV step-time ratio "0.70 at 32K x 16, 0.79 at 8K x 64"; "H-a REFUTED (oracle@512 misses 22.7% on Qwen3.5 vs 11.6% Qwen3-4B; every layer more diffuse)"; E2: "token selections touch ~3x more pages".

## 3. Tensions considered and dropped

1. **Why does selection cost far more than its byte share (indexer / top-k)?** 2605.24168 Table 2, FFD in MOIRA and SANTA all show a selection stage costing 5-20x its byte share. Dropped: GVR (2604.22312) already gives the traffic model (R*N*4B per query head for radix top-k, multi-pass re-reads, atomics) and a fix; StreamIndex, AsyncTLS and Self-Indexing KVCache work on the same problem. Also sits inside D1.
2. **Does the ranking of low-bit kernels invert between HBM and GDDR6 cards?** Dropped as predictable: BitDecoding already reports about 4x / over 7x on an RTX 4090, and the derived issue budget on Ada is 4-11x B200's. Its facts are used in card 1.
3. **Shared-prefix cascade bound (Hydragen 32x vs FlashInfer 13-17%).** Dropped as a card: explained by Amdahl plus a small-list floor; the floor part is card 3.
4. **Is large-batch decode bandwidth-bound or stall-bound (Mind the Memory Gap vs MagicDec)?** The paper says DRAM is saturated yet also reports average DRAM read below 65% and attributes the replication gain to CPU gaps. Dropped: this is D3 (profiling), and the user's own roofline measurement already answers it for their stack.
5. **Power-cap coupling between kernels on a 300 W card** (clock set by average power, so isolated kernel timings may not add). Dropped: no literature tension, pure profiling (D3 shape); decode is mostly clock-insensitive.
6. **Low-rank vs low-bit KV at equal realised speed.** Dropped: predictable (low-bit wins at equal bytes; FOSCAR removes most of the issue cost).
7. **Certified sparse verification (make verification read less, losslessly).** Dropped: predictable negative; SSV covers the sparse-target case.
8. **GQA group agreement / per-KV-head vs per-query-head selection.** Dropped: MOIRA answers it at page level (1.51x); D1.
9. **O(N) selection ceiling and temporal reuse of selections.** Dropped: GVR, SSV reuse layers and SparseSpec's stride reuse already exploit it; D1.
10. **GDN state traffic / low-rank state.** Dropped: D2 and ReplaySSM; Windowed-MTP (2607.21535) covers the draft-head cost beside a hybrid target.

## 4. Kept cards

1. **Do low-bit KV dequantisation and multi-token speculative verification spend the same spare compute?** (same-saving-twice). FOSCAR's issue ceiling vs MagicDec/SSV's "verification rides on idle compute". Not covered by FOSCAR (single-row decode), Quasar (quantised weights) or QuantSpec (no kernel analysis).
2. **Is the token-vs-page crossover set by a contiguous-bytes quantum, so the best block size grows as bits per token fall?** (contradiction + untested assumption). 2605.24168 attributes token-level success to the 256-byte row; at INT2 that row is 32 bytes. Overlaps D1.
3. **What sets decode attention time for short lists, and where is the knee below which reading more tokens is free?** (unexplained anomaly). FlashInfer Table 5, 2605.24168 B=1 saturation, SparseSpec's 5.1x predicted vs 3.29x measured. Partly D3-shaped.
4. **Why do hybrids tolerate 50-250x oracle top-k when their attention mass is more spread out?** (contradiction with the user's H-a). Builds on residue (b); overlaps D1.

## 5. Search log

- Semantic Scholar keyword search: rate-limited (HTTP 429) for most queries; one query returned BitDecoding, FOSCAR, OSCAR. Batch-by-id endpoint worked and was used for abstracts.
- arXiv API: HTTP 429 throughout; full texts fetched from arxiv.org/html instead.
- WebSearch (6 of 6 used): (1) SD with low-bit KV, verification compute-bound; (2) sparse attention indexer / top-k overhead; (3) GPU gather bandwidth vs run length, token vs page, quantised KV; (4) latency floor for short KV lists in sparse draft kernels; (5) why hybrids are robust to top-k sparse attention; (6) multi-token verification kernel over INT4/2-bit KV, issue bound.
- Closest existing answers: card 1 FOSCAR 2611.09722 and Quasar 2603.01399; card 2 2605.24168 and AsyncTLS 2604.07815; card 3 PersistentKV 2606.26666 and SSV 2605.19893; card 4 2605.24168 and 2604.05438 (snippet only).
