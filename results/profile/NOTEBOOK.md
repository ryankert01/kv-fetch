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

## 03:38 -- P4 hybrid serving (serve_qwen35_4b), and a hybrid-only extra prefill step
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

## 03:45 -- multi-turn gate (serve_multiturn_4b_pair) and upstream search: the two prefill leads are taken
prefix_repetition: 64 shared 4096-token prefixes, 128-token suffix, 256 output, prefix caching on.

| model | c | out tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p99 ms | prefix hit rate |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 16 | 588 | 605 / 5158 | 23.6 / 31.2 | 147 | |
| Qwen3-4B | 16 | 444 | 875 / 5628 | 31.7 / 39.0 | 210 | |
| Qwen3.5-4B | 64 | 1269 | 299 / 15742 | 40.0 / 75.2 | 185 | 70% |
| Qwen3-4B | 64 | 656 | 996 / 16598 | 90.5 / 104.9 | 332 | 61% |

- The hybrid wins every metric (1.3x / 1.9x output throughput); the dense model's 229K-token pool cannot
  hold the 64 x 4K prefixes (hit rate 61 vs 70%). So the small-step floor does not decide this workload;
  capacity does. The gate as stated ("does the floor show in serving") gets a no for hybrid vs dense; how
  much the hybrid would gain without the floor is not measurable without a fix.
- Upstream (gh search, vLLM PRs, 03:40):
  - align split: #52789 (merged 2026-08, FlashKDA, "9-25% TTFT") introduced internal prefill checkpoints;
    #60050 (merged 2026-10-14) does it for GDN on the FlashInfer prefill backend (Qwen3.5-35B-A3B GB200,
    TTFT -10 to -20%); #60659 (open, updated 2026-10-28) does it for the Triton/FLA GDN backend, i.e.
    exactly our Ada case, no kernel changes; #63670 (open draft) removes the split altogether.
  - small-step floor: #61779 + #61952 (open, 2026-10-16/27) capture the GDN core in breakable CUDA graphs
    of mixed batches (A100: mixed-step host time 99 -> 55 ms, TPOT -9 to -10%); SGLang issue 35851 is the
    same per-layer break cost.
  - preemption at hybrid block transitions: issue #62752 / PR #62878 (open, 2026-10-21) budget the Mamba
    state transition so decode does not preempt avoidably (CPU reproducer, no serving numbers).
  - preempt-to-CPU instead of recompute: #32398 (OffloadingConnector preemptions-only mode, closed
    unmerged 2026-03) and #43839 (preemption_mode="offload", closed unmerged 2026-05); #43797 (merged)
    skips decode-phase blocks in CPU offload. So in 0.32 a preempted decoding request is recomputed in full.
- Verdict under rule (c): the align split and the small-step floor both fail (fixes merged or open
  upstream within the last two weeks). The saturation collapse is the only lead left open; f8 measures it
  with spread lengths and preemption counters.

## 03:50 -- what the align split costs at batch 1 (f7: split_scan_*)
Unprofiled prefill latency, median of 5, LLM class (8192-token budget), GPU 6.

| prompt tokens | 256 | 512 | 528 | 600 | 1024 | 1056 | 1100 | 2048 | 2112 | 4096 | 4224 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid, prefix on (align) ms | 25.5 | 42.4 | 48.3 | 62.2 | 89.8 | 96.9 | 92.4 | 170.9 | 178.5 | 333.8 | 346.2 | 646.7 |
| hybrid, prefix off ms | 25.3 | 42.9 | 48.9 | 51.3 | 80.2 | 81.7 | 89.4 | 155.4 | 171.9 | 326.1 | 323.4 | 646.3 |
| dense, prefix on ms | 19.6 | 38.9 | 48.7 | 49.0 | 78.5 | 80.1 | 82.6 | 158.8 | 152.9 | 326.7 | 333.1 | 717.9 |
| on / off | 1.01 | 0.99 | 0.99 | 1.21 | 1.12 | 1.19 | 1.03 | 1.10 | 1.04 | 1.02 | 1.07 | 1.00 |

- Up to one block (<= 528) nothing is split and on = off. Past it the split adds 3-23 ms (~10% at
  0.6-2K tokens, 2-7% at 4K). At 8192 on (7920 + 272, two steps) and off (one 8192 step) are equal;
  not explained, within run-to-run scatter of a few ms. Without prefix caching the hybrid matches the dense model
  from 512 tokens up (0.97-1.12x) and is 1.29x slower only at 256 tokens (the small-step floor).
- Same size as upstream's claim for the merged FlashInfer version (#60050: TTFT -10 to -20% at 8K on a
  35B MoE where a second full pass costs more); the FLA version (#60659) is open. Confirms the lead is
  real and already taken. Not pursued.

## 05:37 -- the collapse with spread lengths, and its counters (f8: serve_thrash_qwen35_4b)
Qwen3.5-4B, `vllm serve` (2048-token budget, 256 seqs), GPU 6, workload reasoning_var: input 1024 and
output 8192, each uniform in +-50% (`--random-range-ratio 0.5`), 2 prompts per concurrency slot.
Counters are /metrics deltas over each run; ms/step = run time / `vllm:iteration_tokens_total_count`;
running and "gen < half its max" from the server's 10-s stats lines (serve_timeline).

| setting | c | out tok/s | TTFT p50 s | TPOT p50 ms | preemptions | steps | ms/step | mean running | gen < half max |
|---|---|---|---|---|---|---|---|---|---|
| prefix on (default, align) | 64 | 1744 | 0.37 | 30.0 | 0 | 23499 | 26.9 | 52 | 19% |
| prefix off | 64 | 1763 | 0.32 | 29.7 | 0 | 23501 | 26.6 | 52 | 20% |
| prefix on (default, align) | 256 | 1576 | 160 | 92.2 | 3474 | 36551 | 73.9 | 133 | 87% |
| prefix off | 256 | 2020 | 147 | 69.2 | 800 | 36175 | 58.3 | 132 | 59% |

- The collapse is not an artefact of identical lengths: with spread lengths prefix-on loses more
  (1576 vs 1698 tok/s with fixed lengths), prefix-off about the same (2020 vs 2057).
- At c=256 prefix caching on preempts 4.3x as often as off at the same ~133 running requests and the
  same number of engine steps, each 27% longer; output throughput is 22% lower. At c=64 (no preemption)
  the two are equal, so the align-mode bookkeeping itself costs nothing; the difference is all in how
  saturation is handled.
- The c=64 drop against fixed lengths (1744 vs 2068) is the drain at the end of the run (running falls
  64 -> 3 over the last 3 minutes as the longest outputs finish), not the scheduler.
- vLLM's counters do not see recomputation: prompt and iteration token counts equal the dataset's
  (535,270 prompt + 4,231,042 output tokens in both settings), so the work redone after the 3474 / 800
  preemptions is invisible. Prefix-cache queries and hits count first prefills only (0 hits).
- Likely mechanism for the on/off gap (to be measured, f10): in align mode a decoding request crossing a
  528-token boundary needs a new state block in each of the three Mamba groups as well as the attention
  block, 4 blocks instead of 1, and the old state blocks are freed only two steps later. Upstream issue
  #62752 / PR #62878 (open) describes this for speculative decoding on a CPU reproducer, without serving
  numbers.
- f10: the same load offline with the scheduler instrumented (scripts/profile/thrash_trace.py): the
  trigger of every preemption, recomputed tokens, prefix-cache savings on readmission; dense Qwen3-4B at
  matched pressure (c=80) for whether the on/off gap is hybrid-only.

## 06:45 -- an existing knob removes the collapse (f9: serve_thrash_knobs_qwen35_4b)
Same server and load as f8 at c=256 (reasoning_var, prefix caching on unless noted). Rows 1-2 from f8.

| setting | out tok/s | TTFT p50 / p99 s | TPOT p50 ms | e2e p50 s | preemptions | steps | ms/step | mean running | KV >= 97% | gen < half max |
|---|---|---|---|---|---|---|---|---|---|---|
| default (watermark 0) | 1576 | 160 / 698 | 92.2 | 1143 | 3474 | 36551 | 73.9 | 133 | 84% | 87% |
| prefix off | 2020 | 147 / 538 | 69.2 | 890 | 800 | 36175 | 58.3 | 131 | 76% | 59% |
| `--watermark 0.1` | 2128 | 141 / 540 | 61.7 | 825 | 128 | 39644 | 50.5 | 123 | 21% | 41% |
| `--max-num-seqs 128` | 2057 | 484 / 696 | 57.7 | 893 | 790 | 40394 | 51.4 | 116 | 35% | 28% |

- Keeping 10% of the blocks free when admitting waiting / preempted requests cuts preemptions 27x and
  gives +35% output and -33% TPOT against the default, with no TTFT cost; it also beats prefix caching
  off. Capping running requests at 128 helps throughput as much but not preemptions, and triples the
  median TTFT (requests queue instead).
- So the saturation collapse fails rule (c): an existing knob removes it. What is left is a
  configuration finding (vLLM's default watermark 0 costs this hybrid 26% of its throughput at
  saturation, prefix caching on), worth an upstream issue, not a paper.
- What remains against pure decode: ~123 running at ~5K mean context would take ~40-45 ms per decode
  step (decode sweep: 39.1 ms at 2K x 128, 37.3 ms at 8K x 64), against 50.5 ms per step here. The rest is
  the new requests' prefill chunks inside the 2048-token budget and the end-of-run drain. Not chased.
- f10 (running) still says why prefix caching on preempts 4.3x as often at watermark 0, and whether the
  dense model shows the same on/off gap (moved ahead of the hybrid prefix-off trace).

## 07:33 -- why prefix caching on thrashes: a readmission ping-pong (f10: thrash_trace_qwen35_4b_on)
The f8 load offline (in-process LLMEngine, 256 in flight, 512 requests, 2048-token budget, watermark 0,
prefix caching on = align mode), scheduler instrumented (scripts/profile/thrash_trace.py, summarised by
thrash_report.py). It reproduces the server run: 3484 preemptions (server: 3474), 1480 output tok/s (1576).

| | |
|---|---|
| preemptions | 3484, of 326 requests (up to 61 times one request) |
| trigger: a decode at a 528-token block boundary cannot get its blocks | 1930 (55%) |
| trigger: a readmitted request's recompute chunk cannot get its blocks | 1484 (43%) |
| trigger: a first prefill chunk | 70 |
| victim is the triggering request itself | 1503 |
| victim's context (mean) | 2273 tokens, 1842 computed |
| readmitted requests preempted again | 3158 of 3484, median 2 steps after readmission |
| prefix-cache hit on readmission | 16% of what the request had computed (369 of 2260 tokens) |
| tokens: decode / new prefill / recompute | 4.14M / 0.53M / 5.13M (recompute = 9.7x new prefill) |
| steps with recompute | 4941 of 35997, 147 ms each vs 65 ms decode-only; 26% of the time |

- Mechanism. Preemption is LIFO (`running[-1]`), and a preempted request goes back to the head of the
  queue. Admission checks that the whole sequence fits (`scheduler_reserve_full_isl`) but reserves
  nothing; the readmitted request then recomputes in 1584-token chunks (align mode), each needing 3 new
  blocks, while ~115 decoding requests take blocks at every 528-token boundary (4 blocks each in align
  mode: attention + one state block per Mamba group). Within ~2 steps a chunk or a boundary crossing
  fails, and the victim is the newest running request, i.e. the one just readmitted. Its work is lost and
  it is readmitted again. The long-running requests keep their blocks; short new ones starve.
- Cost: 5.1M recomputed tokens is more than all decode tokens; the recompute steps take 26% of the
  time. Removing the extra 82 ms in those steps alone would save ~15% of the run.
- Why the knobs help (f9): watermark 0.1 makes readmission wait for 194 free blocks, enough to finish the
  recompute, so the ping-pong stops (128 preemptions). Prefix caching off helps less: recompute chunks of
  2048 and 1 block per boundary crossing narrow the window (800 preemptions).
- Next in f10: the dense model at matched pressure, on and off.

## 08:02 -- the dense model at matched pressure does not ping-pong (f10: thrash_trace_qwen3_4b_on)
Qwen3-4B, same load and scheduler settings, 80 in flight / 160 requests (in-flight demand over capacity
about the hybrid's at 256: 232K-token pool, 16-token blocks). Prefix caching on.

| | hybrid, 256 in flight | dense, 80 in flight |
|---|---|---|
| mean running / in flight | 115 / 256 (45%) | 35 / 80 (44%) |
| preemptions per request | 6.8 | 2.7 |
| triggers | 55% decode at a block boundary, 43% recompute chunk | 422 of 424 decode at a block boundary |
| free blocks when an allocation fails (median) | 2 (a boundary crossing needs 4) | 0 (needs 1) |
| victim is the triggering request / victim mid-recompute | 1503 / 2045 of 3484 | 9 / 2 of 424 |
| readmitted, then preempted again: median steps later | 2 | 106 |
| recompute tokens per output token | 1.24 | 0.67 |
| share of time in steps with recompute | 26% | 7% |

- Both preempt under this pressure (watermark 0), but only the hybrid loses the recompute itself: a
  dense readmission finishes its recompute and decodes for ~100 steps before it is chosen again; a
  hybrid readmission is usually thrown out mid-recompute within 2 steps. That fast ping-pong is the
  hybrid-specific part: 528-token blocks, 4 blocks per decode boundary crossing in align mode, and
  1584-token chunks of 3 blocks each, so the pool runs out in coarse steps.
- Still fails rule (c) as a research target (watermark 0.1 removes it), but it is a clean hybrid-only
  scheduler pathology for an upstream issue: default settings + prefix caching + saturation.

## 08:04 -- prior work for the speculative-decoding lead (before f11 has numbers)
- In 0.32: ReplaySSM (`--use-replayssm`, cache recent SSM inputs and skip the per-step full-state store)
  exists for Mamba2 only (NemotronH is the one model class with `supports_replayssm`); RecoverSSM for
  Kimi KDA. Qwen3.5's GDN runs the default path (all k + 1 states kept for rollback).
- Upstream for GDN (gh search, 08:04): #58863 RecoverSSM for Qwen GDN (open, updated 2026-10-28);
  #49887 ReplaySSM for Gated DeltaNet spec decode (closed 2026-10-12); #56466 ReplaySSM + prefix caching
  for GDN spec decode (WIP); #53929 adaptive verification for Qwen GDN (open); #55504 (merged) bounds
  GDN spec-decode query widths. So the state-rollback cost of GDN speculative decoding is being worked
  on upstream right now; f11 can size it, but as a research target it likely fails rule (c) too.

## 08:34 -- dense prefix on = off (f10: thrash_trace_qwen3_4b_off)
- Qwen3-4B at 80 in flight, prefix caching off: 423 preemptions (on: 424), 740 tok/s (on: 751),
  readmission to next preemption median 108 steps (on: 106), recompute 1.04M tokens (on: 0.86M, the
  prefix cache saves a little on readmission). So the prefix-caching penalty at saturation (4.3x the
  preemptions, -22% output in serving) is the hybrid's alone: it comes from align mode, not from prefix
  caching as such.
- f11: spec_qwen35_4b failed twice on `import pyarrow` (missing from venv-vllm32; installed with uv at
  08:32) and was requeued behind spec_qwen3_4b, which runs now.

## 08:48 -- speculative decoding: works on the hybrid, no GDN verify bottleneck visible (f11: spec_*)
LLM class, defaults (Qwen3.5: prefix caching on = align mode), 90 AIME problems with the chat template
as prompts (cycled), 1024 output tokens each, ignore_eos, greedy. Qwen3.5-4B with its own MTP layer;
Qwen3-4B with a community EAGLE3 head. Step = wall / `vllm:iteration_tokens_total` count.

| model | k | b=1 step x / speedup | b=16 step x / speedup | b=64 step x / speedup | acceptance (b=1 / 16 / 64) |
|---|---|---|---|---|---|
| hybrid, MTP | 1 | 1.27 / 1.55 | 1.26 / 1.51 | 1.58 / 1.19 | 0.97 / 0.94 / 0.94 |
| hybrid, MTP | 2 | 1.44 / 2.02 | 1.50 / 1.74 | 2.12 / 1.22 | 0.96 / 0.88 / 0.88 |
| hybrid, MTP | 3 | 1.61 / 2.31 | 1.73 / 1.86 | 2.57 / 1.18 | 0.92 / 0.81 / 0.80 |
| dense, EAGLE3 | 1 | 1.09 / 1.51 | 1.18 / 1.28 | 1.82 / 0.82 | 0.65 / 0.62 / 0.60 |
| dense, EAGLE3 | 2 | 1.13 / 1.82 | 1.28 / 1.37 | 2.23 / 0.76 | 0.53 / 0.44 / 0.44 |
| dense, EAGLE3 | 3 | 1.19 / 1.85 | 1.33 / 1.28 | 2.53 / 0.69 | 0.40 / 0.33 / 0.33 |

(k = 0 step: hybrid 10.16 / 13.01 / 20.87 ms, dense 9.80 / 11.92 / 17.72 ms at b = 1 / 16 / 64.)
- The hybrid gains more (2.3x at b=1, 1.9x at b=16, still 1.2x at b=64, where the dense EAGLE3 setup
  loses) because its MTP layer is accepted far more often. Acceptance is inflated for both by ignore_eos
  (greedy text keeps going, often repetitively, after the answer); the comparison is between drafters,
  not a clean acceptance measurement.
- The hybrid's step cost does grow faster with k at small batch: +2.1-2.7 ms per draft token at b=1 vs
  +0.6-0.9 ms for dense. Most of that is its drafter, not GDN verification: each MTP draft step runs a
  full decoder layer and the 248K-vocabulary lm_head (1.36 ms per step at b=1 in the nsys capture),
  against EAGLE3's one layer and 32K draft vocabulary. At b=64 the per-draft-token extra is the same
  size for both (12-33 ms vs 15-27 ms): verifying 64 x (k + 1) tokens is compute-bound either way.
- Verdict: no hybrid-specific verify bottleneck at this resolution (kernel traces would be needed to
  split the MTP layer from GDN verify), and upstream is on the GDN state side already (08:04 entry).
  A drafter-side observation remains: a sparse or truncated lm_head for MTP drafting (upstream #59740,
  "context aware sparse LM head", open) would cut most of the small-batch overhead. Not hybrid-specific.

## 09:24 -- the hybrid with prefix caching off (f10: thrash_trace_qwen35_4b_off)
Same load as the prefix-on trace. Reproduces the server again: 779 preemptions (server 800), 1928 tok/s (2020).

| hybrid, 256 in flight | prefix on (align) | prefix off |
|---|---|---|
| output tok/s | 1480 | 1928 |
| preemptions | 3484 | 779 |
| triggers | 55% decode boundary, 43% recompute chunk | 770 of 779 decode boundary |
| readmitted, preempted again: median steps later | 2 | 44 |
| recompute tokens per output token | 1.24 | 0.41 (dense: 0.67 / 0.81) |
| share of time in steps with recompute | 26% | 11% |

- Without align mode the hybrid behaves like the dense model: preemptions come from decode boundary
  crossings, and a readmitted request finishes its recompute before it is chosen again. The 2-step
  ping-pong (victims thrown out mid-recompute) is align mode's: 1584-token chunks of 3 blocks and 4 blocks
  per decode boundary crossing.
- Harness ends here (runner until 09:30; the 21 jobs still pending were not reached). Morning report:
  results/profile/morning_report_2026-10-29.html.

## 09:45 -- day 2: the 9B pair looks like the 4B pair (d2_base_qwen35_9b, d2_base_qwen3_8b)
User (09:3x): keep looking, with bigger models (plan configs/harness_plan_day2.yaml; runner until 21:00, one
card while GPU 0 is busy). engine_profile.py now counts a hybrid request's state blocks (3 for Qwen3.5) in its
capacity check, so the 9B skips points the 4B ran (C=2K x 128, 8K x 64).

| | Qwen3.5-9B | Qwen3-8B | hybrid / dense |
|---|---|---|---|
| capacity (blocks x block size) | 673,728 tokens (528-token blocks, 3 state blocks / request) | 171,344 | |
| prefill 1K / 2K / 8K / 32K ms | 174 / 312 / 1218 / 5470 | 157 / 309 / 1293 / 7342 | 1.11 / 1.01 / 0.94 / 0.75 |
| decode step 8K x 1 / 8K x 16 / 32K x 4 ms | 18.6 / 25.3 / 23.8 | 18.7 / 38.5 / 38.5 | 1.00 / 0.66 / 0.62 |
| cache-hit recompute step (decode_8192_b1 profile) | 57 ms, busy 86% | 30 ms, busy 93% | 1.9 |

- Kernel shares: hybrid prefill 85% GEMM, GDN chunk 4.5% at 8K and 4% at 32K, attention 15% at 32K; decode
  93% GEMM at b=1, at 8K x 16 GEMM 75% / attention 18% / GDN recurrent 4%. Decode at the weight-read
  bound (18.6 ms for ~17 GB). Nothing near the selection rule; the hybrid-specific kernels shrink with size.
- The one hybrid-only cost that grows: a prefix-cache hit recomputes up to one 528-token block (align
  mode) where the dense model recomputes < 16 tokens, 57 vs 30 ms here. That is the prefix-caching line
  stopped on 10-28; noted, not pursued.
