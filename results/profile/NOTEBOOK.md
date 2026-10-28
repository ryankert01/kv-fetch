# Profiling notebook (agent-written; newest last)

Harness: scripts/profile/harness.py (queue in results/profile/harness/, runs in results/profile/runs/),
analysis scripts/profile/findings.py -> results/profile/findings.md after every job. GPU 6 only while
GPU 0 is in use by others. Selection rule: EXPERIMENTS.md, "Profiling hybrid models on vLLM".

## 2026-10-29 00:30 -- start of night 1
Pilot (results/profile/pilot0, vision tower loaded, superseded by the harness base runs):
- Qwen3.5-4B decode C=2048: b=128 38.9 ms/step, b=256 348 ms/step (9x). Hypothesis: preemption or
  block-pool exhaustion from per-request Mamba state blocks. Job probe_b256_qwen35_4b counts preemptions.
- Hybrid vs dense at long context: C=32K b=4 15.1 vs 30.2 ms (Qwen3.5-4B vs Qwen3-4B); prefill 32K
  2.95 vs 4.63 s. KV capacity 955K vs 227K tokens.
- Decode GEMMs at b=16 run near the weight-read bound (9.7 ms vs ~9.1 ms ideal): little headroom there.
Queued: configs/harness_plan_night1.yaml (24 jobs: base sweeps, serving, 8 ablations, scaling).

## 00:50 -- base_qwen35_4b (Ada, text-only)
- b=256 collapse confirmed as preemption: C=2048 b=256 323 ms/step with 39 preemptions during generation,
  although 256 x 2.7K tokens = 692K < the reported 955K-token capacity. Hypothesis: the Mamba groups'
  state/checkpoint blocks share the pool (each GDN state page = one 528-token attention page per layer),
  so a hybrid request needs about twice the blocks its tokens suggest. probe_b256 scans b=128..256.
- Decode roofline: GEMMs 0.89-1.12 of the weight-read bound, attention 0.99-1.02 of the KV-read bound:
  no kernel headroom in the big classes. GDN recurrent looks faster than read+write of fp32 state (its
  writes drain from L2 during later kernels); share 6% at b=16, 16% at b=64. State I/O is ReplaySSM's
  topic (upstream 0.32 has ReplaySSM for Mamba2 only), so it fails rule (c) unless the problem differs.
- GPU idle between decode steps ~1.5-1.9 ms regardless of batch: 14% of the step at b=1 (qualifies as
  overhead), 9% at b=16, 5% at b=64. Host CPU is shared (load ~127 on 96 cores), so part may be the machine.
  Queued nsys captures (configs/harness_plan_night1_f1.yaml): hybrid b=1 and b=16, dense b=1.
- Prefill: GEMMs at 0.70-0.72 of the bf16 peak; attention 24% of a 32K prefill; GDN chunk kernels 7-8%.
- Whole decode step vs the memory roofline (weights + KV + state read and write, at 926 GB/s): 0.89 at
  b=1 and b=16, 0.92 at b=64 (C=8192). Decode is close to bandwidth-optimal for the bytes it moves; the
  headroom left is mostly the inter-step gap. Bigger decode wins need fewer bytes (quantisation, KV or
  state compression, speculative decoding), not faster kernels.

## 00:47 -- base_qwen3_4b (dense pair)
- Inter-step GPU idle: dense 0.93-0.96 ms/step vs hybrid 1.52-1.57 ms at the same points (C=8192, b=1
  and 16). The hybrid pays ~0.6 ms more host-side work per decode step -- a hybrid-specific overhead
  (candidates: Mamba align-mode bookkeeping, several KV-cache groups' block tables, GDN metadata).
  The nsys captures queued at 00:50 (hybrid b=1/b=16, dense b=1) should show where it goes.
- Hybrid / dense decode tokens/s: 0.97 at 2K x b=1, 1.87 at 8K x 16, 2.00 at 32K x 4 (KV 4.5x smaller);
  max batch at 32K: 16 vs 4. Prefill ratio 0.87 at 1K, 0.91 at 2K, 1.08 at 8K, 1.52 at 32K: the hybrid
  is slower on short prompts. Queued kernel profiles of 1K/2K/4K prefills for both (follow-up 2).

## 00:58 -- correction, and a new lead
- CORRECTION: the "1.5 ms GPU idle between decode steps" (and the hybrid's extra 0.6 ms) was a parsing
  artefact. vLLM's GPU annotation covers the model forward only; lm_head and sampling run after it. The
  nsys capture (nsys_gap_qwen35_4b_b1, scripts/profile/nsys_steps.py) shows the GPU idle only 0.36
  ms/step under nsys, and an eager 1.36 ms lm_head kernel per step (248K vocabulary, tied embedding).
  parse_trace.py now gives each step the window from the previous forward's end to its own end; with it,
  decode GPU idle is ~1% for both models, and the hybrid's extra time is its 1.6x larger lm_head.
- Decode summary after the fix (C=8192): whole step 0.89-0.95 of the memory roofline, GPU busy 99-100%,
  GEMMs (incl. lm_head) 0.78-0.96 of the weight-read bound. No decode overhead candidate qualifies.
- New lead: small prefill steps. A 1-request cache-hit recompute (<= 528 tokens) step: hybrid window
  39 ms with the GPU busy 65%; dense 24 ms, busy 90%. Hypothesis: the GDN chunked-prefill path launches
  many Triton kernels per layer outside CUDA graphs (qwen_gdn_attention_core is a graph splitting op), so
  small or mixed prefill steps are launch-bound on this loaded host. Widened prof_short_prefill_* to
  256-4096 tokens; queued nsys of one 512-token prefill per model (follow-up 3).
- 00:55: nsys on the dense Qwen3-4B never opens its capture range (no CUDA activity in the report, twice,
  with both stop modes); the hybrid captures work. Not pursued: the decode-gap question is settled by the
  corrected torch-profiler windows (GPU idle ~1% on both). Small-prefill comparisons use the torch traces.

## 00:59 -- small prefill is launch-bound on the hybrid (nsys_prefill512_qwen35_4b)
- One 512-token prefill: 66 ms window, GPU busy 27 ms (41%), 414 kernels launched outside CUDA graphs
  (~17 per GDN layer: conv1d, post-conv, cumsum, kkt, 16x16->64x64 inverse, recompute_w_u, chunk h,
  chunk o, plus elementwise/index ops) and 33 piecewise graph launches. GDN core is a graph splitting op,
  so its chunked path always runs eagerly. Caveat: the host CPU is oversubscribed (load ~127/96 cores),
  which inflates launch cost; nsys adds more. Queued unprofiled prefill scans 16-2048 tokens, both models.
- Why it could matter end to end: any step with prefill tokens (cache-hit recompute, short turns, chunked
  prefill mixed with decode) takes this path; if it has a launch floor, mixed steps stall every decoding
  request in the batch. P4 chat (TPOT p99) will show whether it reaches serving.
- Prior-work check for the small-prefill lead (rule c), 01:05: vLLM 0.32 selects a FlashInfer GDN prefill
  kernel on SM90 / SM100 / SM120 (qwen_gdn_linear_attn._resolve_gdn_prefill_backend); SM86 / SM89 (our
  A6000 / RTX 6000 Ada) fall back to the FLA Triton chunk path measured here. Upstream is also reworking
  CUDA-graph capture for hybrids ("breakable" graphs, VLLM_USE_BREAKABLE_CUDAGRAPH, off by default in
  0.32; PR 64266 open, which still leaves GDN as an eager break). So the launch-bound path may be specific
  to pre-Hopper GPUs, and neighbouring work is active: likely fails rule (c) unless it also shows on the
  FlashInfer path. We have no Hopper GPU to check.

## 01:04 -- small-prefill floor confirmed unprofiled (scan_small_prefill_*)
| prompt tokens | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B ms | 23.8 | 23.5 | 29.1 | 23.9 | 24.8 | 41.9 | 88.6 | 168.8 |
| Qwen3-4B ms | 12.4 | 12.4 | 12.7 | 13.5 | 20.2 | 40.0 | 77.5 | 153.8 |
- The hybrid has a ~24 ms floor up to 256 tokens, ~11 ms above the dense model (1.8-2.3x for <= 128
  tokens). ~400 eager launches x ~27 us is in line with ordinary Python/Triton launch cost, so probably
  not just the loaded host. Affects: short turns after a prefix-cache hit, and -- if mixed steps take the
  chunk path for their prefill part -- every decoding request in a step that carries any prefill tokens.
- Candidate fixes to test later (not built): for short prefills dispatch GDN to the single-launch
  fused_recurrent kernel (it takes T > 1; spec-decode verify already does), or capture the chunk path in
  CUDA graphs per token-count bucket. Rule (c): the FlashInfer GDN prefill on SM90+ may not have the
  floor; upstream "breakable" graphs leave GDN eager.
- Gate for pursuing it: P4 must show the floor in serving (TPOT of mixed steps, chat TTFT), else drop.

## 01:20 -- queued to test the small-prefill lead
- micro_gdn_prefill_paths (scripts/profile/gdn_prefill_micro.py): 24 layers of the GDN chunk path vs the
  single-kernel fused recurrent update for T = 1..2048 tokens, wall (incl. launches) and GPU time, plus
  agreement of outputs and final states. If recurrent wins below some T, routing short prefills there is
  the obvious fix (vLLM already sends multi-token spec-decode verify through it).
- serve_multiturn_4b_pair: `vllm bench serve` prefix_repetition (64 shared 4K prefixes, 128-token new
  suffix, 256 output), concurrency 16 and 64, hybrid vs dense: every prefill is short and follows a hit.

## 03:50 -- P4 hybrid serving (serve_qwen35_4b), and a hybrid-only extra prefill step
Qwen3.5-4B, GPU 6, `vllm serve` defaults on this GPU: max_num_batched_tokens 2048 (48 GB card), 905K-token
pool. Random prompts, fixed lengths, ignore_eos; one run per cell.

| workload | c | setting | out tok/s | TTFT p50 ms | TPOT p50 ms | ITL p99 ms |
|---|---|---|---|---|---|---|
| reasoning 1K/8K | 16 | prefix on / off | 1037 / 1042 | 564 / 824 | 15.3 / 15.2 | 18 / 18 |
| reasoning 1K/8K | 64 | on / off | 2068 / 2078 | 2397 / 922 | 30.7 / 30.6 | 41 / 40 |
| reasoning 1K/8K | 256 | on / off | 1698 / 2057 | 54390 / 43894 | 93.8 / 73.0 | 261 / 281 |
| chat 4K/512 | 16 | on / off | 617 / 634 | 1649 / 1071 | 22.8 / 23.1 | 180 / 180 |
| chat 4K/512 | 64 | on / off | 958 / 850 | 1215 / 1283 | 70.9 / 73.4 | 222 / 219 |
| chat 4K/512 | 256 | on / off | 714 / 761 | 77338 / 74185 | 220 / 207 | 266 / 287 |

- **Saturation collapse.** At c=256 throughput is at or below c=64 for both workloads. Engine log
  (reasoning, prefix on): 256 running at 4068 tok/s, then the pool fills (99%) and the running set
  drains 256 -> 186 -> 133 -> 103 -> 34 with 222 waiting, KV only 33% used and 148 tok/s. Preempted
  requests are recomputed (V1 has no swap) 2048 tokens per step (prompt throughput counts none of it).
  Same pattern with prefix caching off. Cause candidates: lockstep growth of identical requests
  (range_ratio 0) plus 528-token blocks makes many requests need a block on the same step; recompute of
  up to 9K tokens per victim at the 2048 budget. Generic (dense preempts too, at 1/4 the load), and
  partly an artefact of identical synthetic lengths; not pursued yet. Prefix on vs off differences are
  within run-to-run scatter except possibly reasoning c=256 (1698 vs 2057); one run each, not claimed.
- **Align-mode prefill split (new, hybrid-only).** The trace step annotations show that with prefix
  caching on every hybrid prefill is cut at its last 528-token block boundary and the tail runs as its
  own engine step: 1024 -> 528 + 496 (43.9 + 32.0 ms; dense: one 55.8 ms step), 4096 -> 3696 + 400
  (263 + 41 ms; dense 265), 32768 -> 4 x 7920 + 1056 + 32 (the 32-token tail: 31.5 ms). Source:
  `Scheduler._mamba_block_aligned_split` (align mode writes the state only at chunk ends, so chunk ends
  must be block aligned, and the prompt is stopped at its last cacheable boundary). In serving the 2048
  budget becomes 1584 (3 x 528) per chunk, so a 4K chat prompt takes ~4 steps where the dense model takes 2.
  Each extra step pays the hybrid small-step floor (~24-31 ms, above).
- **Prior work for the split (rule c).** vLLM 0.32 already avoids it for Mamba2 (`num_prefill_checkpoint_blocks=1`,
  alignment 1, ops/ssd_checkpoint.py) and FlashKDA (alignment 16, kda_checkpoint.py): the backend exports
  the block-boundary state from inside the final chunk. GDN (qwen_gdn_linear_attn.py) has no exporter, so
  Qwen3-Next / Qwen3.5 always split. Porting it to GDN is a real fix but an engineering one (an existing
  upstream mechanism, new backend); FLA's chunk kernel keeps states only at 64-token chunk ends and
  528 is not a multiple of 64, so it is not a one-liner. Upstream PR search still to do.
- **micro_gdn_prefill_paths is invalid**: the two kernels' outputs disagree (rel err 1-90, one nan), so
  the inputs do not match what one of them expects (head mapping or state layout), and its chunk timings
  (0.4 ms per layer at T=1) are 10x the in-engine ones. Not used. The in-engine traces already answer the
  question it was for: at a 256-token prefill the GDN kernels are 1.9 ms of GPU time per step, the step is
  31 ms with the GPU busy 50%: the floor is launch / host time, not GDN math, so a faster GDN kernel alone
  will not remove it; fewer launches (graph capture, fusion) or fewer steps (no split) would.
- Queue: multi-turn pair moved ahead of the dense serving run (now prio 3; dense c=256 reasoning dropped, it
  only thrashes at ~25 requests of capacity); prefix on/off prefill scans around block multiples queued
  (f7). Harness: jobs now run in their own process group and a timeout kills the group (a killed serving
  job used to leave its server holding the GPU).
