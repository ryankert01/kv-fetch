# Experiment plan: a page selector for tiered KV on long reasoning

Agreed with the user on 2026-10-24. Target: NeurIPS 2027 (deadline around May 2027).

## Claim under test

On long reasoning, at the same GPU memory, tiered KV (full cache in host memory, a small GPU cache)
driven by a cheap page selector

1. is clearly more accurate than eviction (SnapKV, R-KV, RaaS, LazyEviction, ForesightKV-style),
2. is more accurate than existing offload selectors (Quest, ArkVale, InfiniGen, ShadowKV, ClusterKV), and
3. matches keep-all sparse attention (LessIsMore, SeerAttention-R) while holding far less KV on GPU.

The main battleground is small budgets (512 to 1K tokens per KV head), where published methods
start to lose accuracy.

Why this angle (results/compare_explore.json, explore split, 38 Qwen3-4B thinking traces):
- Eviction has a ceiling. Even evicting by true future attention misses about as much attention
  as SnapKV (8.5% vs 8.6% at 1K), because evicted pages revive.
- Per-step re-selection has headroom. True top-B misses 5.0% at 1K, 42% less than the best eviction.
- The selector is the bottleneck. Quest misses 8.0%.
- Prefetch timing barely matters for coverage. Skipping late pages costs about 0.4 points.

## Rules

- Development and tuning use the explore split only (even sample index). Final numbers use the
  confirm split and benchmarks never looked at during development (HMMT Feb 2025, GPQA-Diamond).
  Hyperparameters are frozen before either is opened.
- Selectors may only use what a tiered system can observe: the current query (through page key
  summaries or a GPU-resident key sketch), and attention on pages they actually attended (including
  the pinned sink/recent pages and the prefill window). Full attention is read only by the
  upper bounds.
- Training-free first. A small learned gate (SeerAttention-R style, frozen base model) is the
  fallback if E1/E2 fail their gates.
- GPUs: never GPU 0; only free cards among 1-6; yield cards when others are short.
- Every result records model, context length, page size, w and git commit (kvfetch.provenance).

## Stages and gates

| Stage | What | Data / compute | Gate to pass |
|---|---|---|---|
| E1 | Offline screen of training-free selectors | existing reasoning traces, CPU | at 1K, close at least half of the Quest-to-top-B gap (missing attention at most 6.5%), at no more than 1.5x Quest's scoring cost |
| E2 | Head-to-head with published selectors | ~20 explore traces re-traced with full keys at the traced layers (about 130 MB each) | at 512 and 1K, 20-30% less missing attention than the best published selector |
| E2b | Page-level head-to-head at vLLM block granularity (added 2026-10-24, see Status) | E2's replayed keys, GPU | at 512 and 1K, one page set per layer: ours misses at least 20% less than the best of Quest, ArkVale, ShadowKV-page, and no more than InfiniGen-page at the same GPU metadata bytes |
| E3 | Online accuracy (main result) | sparse decoding in HF, batched; Qwen3-4B main, Qwen3-8B scale check | pilot: at 1K, at least 5 points above the best baseline on AIME (avg@8, paired by problem) |
| E4 | System cost | analytic model first, then a prototype | GPU memory at equal accuracy; PCIe bytes/step; tokens/s vs batch size |
| E5 | Mechanism and final numbers | confirm split + held-out benchmarks | revival statistics, eviction ceiling, the blind decision-point thresholds (evaluated once) |

### E1 candidates (scripts/compare_selectors.py, config `selectors:`)

- Quest bound with the current query; Quest plus a recency share of the budget.
- Observed attention history: last observed mass, peak observed mass.
- Hybrids: a share of the budget by Quest, the rest by observed history.
- One set shared by all KV heads of a layer (LessIsMore-style nomination counts).
- Upper bounds: true top-B per KV head, and the best set shared across KV heads.

Metrics: missing attention (1 - coverage) at 512/1K/2K per layer, revived attention captured
(mass on pages that re-enter the top-B after at least 16 dormant steps), and scoring cost in bytes
read per page per step.

### E3 design notes

- Benchmarks: AIME24, AIME25, MATH-500 L5 for development; HMMT Feb 2025 and GPQA-Diamond held out.
- avg@8 per problem: about 2 points standard error over AIME's 60 problems; comparisons are paired.
- Budgets 512 / 1K / 2K (4K for reference). Methods: full attention, SnapKV, R-KV, RaaS, Quest,
  LessIsMore, SeerAttention-R (released gates if available), the best E2 baseline, ours.
- Needs batched decoding (about 12 sequences per 48 GB card at 16K) to fit in days, not weeks.

## Status

- E1 (results/selectors_explore.json): coverage criterion passed (best candidate misses 6.1% at 1K,
  63% of Quest's gap to true top-B closed); scoring cost 1.51x Quest, just over the 1.5x limit.
- E2 (results/baselines_explore.json): failed as written. Token-level selectors beat every page
  selector on coverage, even the page ceiling (InfiniGen best case 3.6% vs true top-B pages 5.0% at
  1K), but their selections touch about 3x more pages.
- Decision (user, 2026-10-24): page granularity, so the selector drops into vLLM. A 16-token page is
  one vLLM block; one page set per layer shared by all KV heads is a sub-block-table that vLLM's
  paged kernels and block transfers read unchanged (a block holds all KV heads).
- E2b replaces E2 as the head-to-head: every method selects pages, token methods are adapted
  (InfiniGen's speculated attention summed per page), per KV head and shared per layer. Its gate
  narrows E2's comparison class to page-level methods after seeing E2; that is a degree of freedom
  spent, so the confirm split and the held-out benchmarks remain the real test.
- E2b (results/pages_explore.json, 38 traces): failed as written. With one page set per layer, our
  best misses 16.6% less than the best page baseline (Quest, vote) at 1K and 13.0% at 512 (needed
  20%); it beats Quest on every trace and is no worse than InfiniGen-page at equal metadata bytes
  (6.5% vs 6.7% at 1K). The bar was near-unreachable: the shared-set ceiling itself is only 27% (1K)
  and 21% (512) below Quest, and we close about 65% of that gap. At page granularity, missing
  attention cannot show a large margin over Quest; per-head sets fetch 2.0-2.8x whole blocks per
  budget page, so the vLLM-native shared set wins at equal block traffic.

- E3 pilot (started 2026-10-24, scripts/run_e3.py, config `e3:`): AIME24 + AIME25 explore problems
  (30), avg@8 with Qwen3's thinking-mode sampling, one page set per layer for every method.
  Baselines for the pilot: Quest (vote) and SnapKV-style eviction (shared); oracle page sets as the
  ceiling if time allows; R-KV, LessIsMore and the rest wait for the main E3 table. Sparse decoding
  (kvfetch/sparse_decode.py) keeps the full cache on the GPU (accuracy only, no speed claims), was
  checked against HF's logits (results/reasoning/e3_checks/logits.txt) and against E2b's offline
  replay on the engine's own queries and keys (replay.txt: identical Quest sets, ours Jaccard >= 0.998).
- E3 pilot (results/e3_explore.json, 30 problems, avg@8, 16K generation cap): failed as written. At
  1K ours 43.8% vs Quest 40.4% (+3.3 points, CI95 [0.0, 6.7]; needed +5), full attention 48.3%,
  SnapKV-shared 8.3%. The bar was out of reach for any shared selector: the oracle page set scores
  44.2%, only +3.8 over Quest, and ours is within 0.4 of it. At 512 the selector matters: ours 33.3%
  vs Quest 22.5% (+10.8, CI95 [5.0, 17.5], 12 wins / 1 loss), oracle 38.3%, SnapKV-shared 0.8%.
  Most of the loss is truncation at 16K (Quest@512 70% vs full 41%); accuracy on finished samples is
  73-80% for every non-SnapKV run. Truncated Quest/ours outputs are not verbatim loops (1-3%, as
  full); SnapKV's often are (26% at 512).
- E3 ablations (added after seeing the 1K and 512 results, so exploratory): at 512 the observed
  history adds nothing (ours vs ours-nohist +0.0, CI95 [-3.8, 3.8]); the mean-key mix carries the
  gain (ours-nohist vs quest-sum +10.0, CI95 [4.6, 15.8]); vote vs summed shares for Quest bounds
  does not matter (+0.8). The "importance history" part of the method is not supported online.
- E3 caveats (raised by a mock NeurIPS review, 2026-10-25; overall 2/6). Budgets count attended KV
  only: page summaries stay on the GPU for every token (48.25 B vs 32 B per token per KV head for
  Quest), about 1.6K (ours) vs 1.1K (Quest) tokens of KV per layer at 17K context. At equal total
  GPU memory ours@512 compares with Quest@1K, which wins (40.4% vs 33.3%). The SnapKV-shared variant
  does not protect the prompt's pages and is a weak eviction baseline (offline it misses only 0.6
  points more attention than Quest, online it collapses). Not yet run: a per-head oracle (to test
  whether the 1K gap is the shared-set ceiling), mean_mix values other than 0.5, R-KV / RaaS /
  LessIsMore online, a 32K generation cap. The 30 problems are the ones whose traces chose the
  selector in E1/E2b. Truncated-output repetition: scripts/e3_repetition.py
  (results/e3_repetition_explore.json).

## Timeline

| When | Work |
|---|---|
| Nov 2026 | E1, E2, gate decisions; fallback (learned gate) if needed |
| Dec 2026 - Jan 2027 | batched sparse decoding; E3 pilot, then the Qwen3-4B main table |
| Feb - Mar 2027 | E3 on Qwen3-8B and held-out benchmarks; E4 prototype |
| Apr 2027 | E5 on the confirm split; writing |
| May 2027 | submission |

# Hybrid models (pivot agreed with the user on 2026-10-26)

Pure-transformer KV selection is crowded (the E3 mock review scored originality 2/6). Hybrid models
(linear-attention or SSM layers plus a few softmax-attention layers) are where new models are going.
Their prefix caching (Marconi, vLLM `mamba_cache_mode align`, SGLang's Mamba radix cache) and SSM-state
I/O / speculative rollback (ReplaySSM, arXiv 2610.13778) are taken; decode-time page selection and
tiering for their attention layers on long reasoning is, as far as a non-exhaustive search found, open.
vLLM cannot yet offload hybrid KV (issue #38230, open as of Aug 2026).

Why it matters (Qwen3.5-4B config: 32 layers, 24 Gated DeltaNet + 8 full attention, 4 KV heads x 256):
KV is 32 KB/token (Qwen3-4B: 144 KB) and the GDN state 48 MB/sequence (fp32). At 16K tokens the KV is
~91% of per-sequence memory and ~84% of per-step state+KV traffic (~91% once ReplaySSM halves state
traffic). Three GDN layers sit between consecutive attention layers and read no KV: a natural window to
prefetch the next attention layer's pages. Models with very few attention layers (Nemotron-3: 4 of 42)
are a weaker case (KV passes the Mamba state only after ~5K tokens), so Qwen3.5 is the main target.

## Hypotheses and how each is tested

- H-a (concentration): in hybrids the attention layers act as global retrieval anchors, so attention
  is more concentrated on few pages than in a transformer. Test (H1): the shared-set oracle's missing
  attention at 512 tokens, Qwen3.5-4B vs Qwen3-4B, same problems, same pipeline.
- H-b (eviction tolerance): the recurrent state keeps a coarse summary of evicted tokens, so eviction
  hurts less. Tests: (offline, H1) next-token KL of teacher-forced sparse forwards vs missing attention,
  both models; (online, H3) prompt-protected SnapKV-style eviction at 1K relative to full attention.
  Holds if the online loss on Qwen3.5-4B is at most half the loss on Qwen3-4B (both vs their own full).
- H-c (cross-layer prefetch): a query predicted from the residual stream three layers before an
  attention layer (its input norm, q projection, q norm and RoPE applied to that hidden state) selects
  nearly as well as the real query, so pages can be fetched during the GDN layers. Holds if it costs at
  most 1.0 point of missing attention at 512 tokens (Quest and ours) on Qwen3.5-4B.
- H-d (exploratory, GDN prior): the GDN write strength (beta) and decay mark what the recurrent state
  stored and forgot; as a page prior it improves selection. Mixing weight chosen on AIME24 problems,
  reported on AIME25 problems; holds if missing attention at 512 drops by at least 5% (relative).

## Stages and gates (fixed before running; configs/hybrid.yaml)

| Stage | What | Gate |
|---|---|---|
| H0 | separate venv (transformers 5.19, fla-core 0.5.2; E3's .venv untouched); the E3 engine extended to hybrid models (GDN conv/recurrent state per slot, CUDA graphs, presence penalty); checked against HF's own generation logits | logits match HF within E3's tolerance; sparse methods with a budget >= context equal full |
| H1 | offline screen on sample 0 of the 30 AIME explore problems (full attention, thinking mode), first 16K generated tokens, 8 attention layers per model (Qwen3-4B: same relative depths), exact attention from one replay forward | (i) Qwen3.5's oracle misses no more than Qwen3-4B's at 512; (ii) Quest's relative gap to the oracle at 512 on Qwen3.5 is at least Qwen3-4B's. (i)+(ii): selector track. (i) only: systems track (H4 first, Quest-class selection). Not (i): attention in hybrids is diffuse; report and rethink |
| H2 | cheap page summaries (the mock review's main objection: summaries cost 6-9% of KV on the GPU): mean-key only, fp8 / int8 bounds, larger summary pages; per-step page churn | pick the cheapest format with GPU metadata <= 2% of KV bytes whose missing attention at 512 is within 0.5 points of full-precision ours (chosen on AIME24, reported on AIME25); it becomes "ours" in H3 |
| H3 | online accuracy, Qwen3.5-4B: AIME explore 30 problems, avg@8, thinking-mode sampling (T 1.0, top-p 0.95, top-k 20, presence penalty 1.5, model card), 32K generation cap; full, Quest, ours, prompt-protected SnapKV, oracle at 512 and 1K; Qwen3-4B prompt-protected SnapKV at 1K and 512 under E3's settings (for H-b) | ours@512 >= max(Quest@512, SnapKV@512) + 5 points (paired by problem), with ours' metadata <= 2% of KV |
| H4 | systems: page churn from H1 vs PCIe bandwidth in the GDN window (measured step and copy times on these GPUs, batch 1-64); a host-resident KV prototype if time allows | stretch: tiered throughput >= 0.9x all-on-GPU where both fit, larger max batch at 32K |

Unchanged rules: explore split only; the confirm split and held-out benchmarks (HMMT Feb 2025,
GPQA-Diamond) wait for E5; never GPU 0; every result records model, context, page size, w and commit.
Audits (independent agents, no access to my conclusions) review the engine before H3 runs, the H1/H2
analysis code before its numbers are used, and every reported number against the result files.

## Hybrid status

- H0 (2026-10-26): kvfetch/sparse_decode.py serves hybrid models (KV for the 8 attention layers only,
  per-slot GDN conv/recurrent state through HF's linear-attention cache protocol, presence penalty,
  prompt-protected SnapKV, H2 summary formats). Checks (results/hybrid/h0_checks/, both models under
  .venv-hybrid): every method with a budget above the context gives HF's logits (mean |dlogit| 0.023-0.032
  Qwen3.5, 0.048-0.060 Qwen3-4B; max up to 5.3, within HF's own eager-vs-SDPA spread of ~6 on Qwen3.5)
  with slots retiring and refilling mid-decode; CUDA graphs on vs off give identical tokens (full, quest,
  snapkv; greedy and sampled). An independent audit found no blocker; it found that SnapKV-style
  eviction trimmed only when a page joined, so on most steps it attended one page over budget (~3% more
  tokens at 512, ~1.5% at 1K). This was also in E3 (SnapKV collapsed there regardless); fixed (trim every
  step), in the offline replica too. Old and new engines give bit-identical tokens on Qwen3-4B otherwise.
- H1/H2 analysis audit (independent agent, 2026-10-26, before any Qwen3.5 sequence was analysed): no
  blocker. Fixed: offline SnapKV renormalised the GQA-group mean instead of each query head (the
  engine's softmax over attended tokens) and evicted at step 0 (dense prefill online); int4 bounds now
  rounded to bf16 as the engine scores them; 2-page units break ties toward the later page in both
  (they tied on the same unit bound); merge marks gates incomplete unless both models have all 30
  problems with all layers. Change to H1's design: each problem is cut to the shorter of the two
  models' generated lengths (missing attention at a fixed budget grows with context; Qwen3.5's 32K
  samples would otherwise be compared at 16K against Qwen3-4B's shorter finished ones). The fidelity
  run adds a budget-above-context row (the sparse path's numerical floor). After the fixes every
  offline replica selects the engine's pages on every step (scripts/check_h1_replicas.py,
  results/hybrid/h1_checks/replicas.txt: 90 of 90 method x seed x prompt-length cases, 0 steps differ).
- H1 + H2 (2026-10-26, results/h1_explore.json, complete: 30 problems x 8 attention layers per model,
  each problem cut to the shorter model's length, mean 11,917 generated tokens):
  - Gate (i) FAILED, H-a refuted: the oracle shared set at 512 misses 22.7% of attention on Qwen3.5-4B vs
    11.6% on Qwen3-4B, and each of Qwen3.5's 8 attention layers misses more than its Qwen3-4B counterpart
    (18-34% vs 5-20%). Attention in the hybrid's attention layers is less concentrated, not more.
    Gate (ii) passed: Quest's gap to the oracle is 28.7% of Quest's miss on Qwen3.5 vs 19.0%. By the plan
    ("not (i): report and rethink") the selector track's premise fails; H3 continues as scheduled to
    measure what this means online.
  - The selector ranking flips between the models: on Qwen3.5 mean-key logits (25.5%) beat Quest bounds
    (31.8%) and the E3 mix (26.2%); on Qwen3-4B mean-key is worst (19.5%) and the mix (13.1%) beats Quest (14.3%).
  - H-c FAILED: the query predicted from 3 layers earlier costs +6.7 points (ours) / +6.8 (Quest) on
    Qwen3.5 (Qwen3-4B: +3.8 / +3.7); even one GDN layer of lookahead costs +3.4 (Qwen3-4B: +0.2).
  - H-d FAILED: the GDN prior (beta, weight 0.5 chosen on aime24) lowers aime25 missing attention by 0.3%
    (needed 5%); its Spearman correlation with page mass is 0.09 (beta) / -0.01 (forgot).
  - H2: mean8 chosen (fp8 mean keys, 1.56% of KV): 24.5% vs the mix's 25.2% on aime24 (aime25 26.6 vs
    27.2). It is ShadowKV-style landmark scoring in fp8 with summed head shares: little novelty alone. 1.56% is the
    format's size: the engine still stores fp32 means and casts them to fp8 every step, and still allocates
    and writes the unused bf16 min/max bounds, so its summaries take ~12.5% of KV.
  - H-b offline (teacher-forced fidelity, 10 sequences <= 8K tokens): Qwen3.5's oracle@512 gives
    next-token KL 0.011 at 22.9% missing; Qwen3-4B 0.035 at 14.1% (0.108 at 26.1%, oracle@256). Per unit
    of missing attention the hybrid's output moves less: 7.4-7.5x lower KL over three pairs with matched
    missing attention (confounded: 8 perturbed layers vs 36). H3's online test decides H-b.
  - Page churn at 512: 6.7 new pages per step per layer (Qwen3.5, mean8) vs 5.3 (Qwen3-4B, mix).
- H3 (2026-10-26, results/h3_explore.json; 30 explore problems x 8 samples, 32K cap for Qwen3.5, 16K for
  Qwen3-4B as in E3; complete, every run 8 samples on all 30 problems):
  - Gate PASSED: ours (mean8, 1.56% of KV) @512 30.0% vs quest@512 19.2% (snapkv@512 0.0%): +10.8 points
    paired by problem, CI95 [+4.2, +18.3], 9 wins / 1 loss of 30. At 1K: ours 41.25% vs quest 33.8%, +7.5
    [+1.7, +14.6]. Full attention 49.6%; even oracle@512 loses 13.8 points [-21.3, -6.25] and ours@512
    5.8 points to it; ours@1024 is 8.3 points below full.
    oracle@1024 39.2%: ours@1024 is +2.1 points above it [-0.8, +5.0] (minimum missing attention per step
    is not maximum accuracy).
  - Losses are mostly unfinished reasoning: 43.8% of full-attention samples hit the 32K cap, 63.3% of
    ours@512 and 75.4% of quest@512; accuracy among finished samples is 78-92% for every method (full 88%).
  - H-b FAILED online: prompt-protected SnapKV@1K loses 40.0 points on Qwen3.5 vs 38.8 on Qwen3-4B
    (ratio 1.03, needed <= 0.5); both fail by not finishing (89.6% vs 43.8% capped on Qwen3.5, 82.9% vs
    35.8% on Qwen3-4B). The offline KL advantage does not survive long generation. The caps differ by
    design (each model's pre-registered setting, as are the sampling settings) and both SnapKV@1K runs
    sit near the floor (9.6%, 14.2%), so the ratio is approximate; a fair test needs one decoding setting
    and a budget that does not collapse (2-8K).
- H4 (2026-10-26; costs results/h4_costs_qwen35_4b.json measured on an idle RTX 6000 Ada at 5ae4a8f;
  residency results/h4_residency_qwen35_4b.json, 10 full-length Qwen3.5 sequences, mean 22,082 generated
  tokens, plus 3 ours@512 trajectories as a cross-check; two audit rounds, no blocker; this is a cost
  model from measured parts, not an end-to-end prototype):
  - Fetch paths (64 KB pages): a Triton gather reading pinned host memory over PCIe (UVA, in-graph, no
    host sync) reaches ~25 GB/s (one page 14 us); with 16 looping programs it slows a concurrent GDN layer
    by 7-16%, with one program per block by 2.7-2.9x. The staged CPU path (index readback, CPU gather, copy)
    reaches ~4.7 GB/s (one page 51 us). Summed layer times match measured whole steps within -2% to +3% (whole steps measured where all KV fits
  in 30 GiB: batch 64 at 8K, 32 at 16K, 8 at 32K; the rest, including the 32K batch-32 all-on-GPU
  reference, are uncalibrated sums). Misses come from full-attention trajectories; on the selector's own
  trajectories c4 misses 26% more pages, and the engine's slower ours attention layer dilutes fetch cost:
  both make 0.94x optimistic.
  - Misses per step per attention layer at 512 (exact selection, LRU GPU cache of c x budget): 6.8 (c1),
    3.5 (c2), 1.7 (c4); on the selector's own trajectories 7.5 / 4.1 / 2.1. The H-c lookahead prefetch
    is never better than exact + c4 (its predicted query, page scores and top-k cost 0.1-5.5 ms per layer); the one-step-stale selection
    (lag1) hides every fetch but raises missing attention from 31.8% to 35.6%.
  - Stretch gate PASSED as modelled: exact selection, c4, zero-copy gather keeps >= 0.94x the all-on-GPU
    throughput wherever both fit (batch 1-160, 8-32K), and at 32K the tiered batch reaches 160
    (host-memory bound; GPU bound 262) vs 33 all-on-GPU: 1,219 vs 971 tokens/s at the all-on-GPU
    measured batch 32 (1.26x); with its step time interpolated to its max batch 33, ~980 tokens/s (1.24x). The gain is
    capped by the engine's per-step cost at large batch (GDN layers 2.1 ms each, the engine's ours@512
    attention layer, fp32 means cast to fp8 every step, 8.6 ms at batch 160 vs Quest's 5.0 ms).

# Prefix caching for hybrid models (direction set with the user on 2026-10-28)

Why: the hybrid track drifted from decode-time KV on long reasoning (H0-H4) to prefix-cache checkpoints
(2026-10-27 pilot, scratchpad ckpt/) without restating the problem. The user then fixed it: the topic
is prefix caching itself; the headline metric is capacity at no accuracy loss; it must be built on
vLLM; the venue, hybrid-specificity and the size of the win may be relaxed if they conflict.

## Claim under test

In vLLM, a hybrid model's prefix cache spends most of its bytes on recurrent-state checkpoints.
Storing them in a format whose error stays at the prefix-cache noise floor, keeping them only where
hits land, and holding the rest in host memory caches several times more prefix tokens per GPU GiB
than vLLM's dense setting, without the misses of its retention-0 default and without changing accuracy.

The tension it resolves (Qwen3.5-4B, 528-token blocks; KV 32 KiB/token; GDN checkpoint 49.1 MiB fp32,
13.3 MiB fp8 per-row incl. scales and conv):
- dense (fork default, vLLM `prefix_cache_retention_interval=None`): 127 KiB per cached token, 75% of it
  checkpoints. Hits land anywhere.
- semantic only (vLLM main default 0): ~38 KiB per cached token for an 8K prompt, but no checkpoint in
  the generated region: in a 5-turn tool loop each turn recomputes the previous output (pilot: 316 vs
  115 ms prefill per turn).
- fp8 checkpoints at dense spacing: 58 KiB per cached token (2.2x dense); every 4th block: 38 KiB (3.3x).

"No accuracy loss" is defined against the rechunk floor (the difference a hit already causes by
splitting the prefix computation). Pilot (81 checkpoints, 30 traces; thresholds below were set after
seeing it, so K1's end-to-end test is the blind one): KL vs exact, once / chained every 528 tokens over
2,048 tokens, as multiples of the floor: fp8-e4m3 per-row 1.02-1.05 / 1.13, int8 per-row 2.36-2.41 /
3.75, int4-g32 16.6 / 34.1. Offline bar: once <= 1.1x, chained <= 1.25x.

Prior work to beat or build on: Marconi and Sparse Prefix Caching (exact checkpoints, placement and
eviction), SGLang INT8 checkpoints (PR 28185, separate pool, not combinable with HiCache) and HiCache
Mamba offload (PR 20457), DASC (head/channel pruning, GPU only), vLLM PR 52962 (draft FP8/INT8 ReplaySSM
checkpoints) and issue #63565 (the offload tier keeps only the replay boundary). Watch PR 52962.

## Stages and gates (fixed before running)

| Stage | What | Gate |
|---|---|---|
| K0 | headroom, CPU only: trace-driven simulator of vLLM's hybrid prefix cache (d0d681e retention semantics: semantic boundaries, periodic, dense; LRU over free blocks; host tier at the measured 2.0 ms per fp32 checkpoint and 25 GB/s; recompute time from the measured fit, results bench/recompute_probe.json). Workloads: Mooncake FAST'25 toolagent and conversation traces (512-token hash blocks, used as blocks; a follow-up that extends an earlier prompt is assumed to carry its output in the next blocks) and the pilot's agent loop. At equal GPU cache memory (10, 20, 30 GiB): vLLM main default, dense, best periodic interval, each with vLLM's host offload as it is today; vs fp8 checkpoints + placement + host tier with an oracle (future hits known) and a simple online rule | the oracle cuts total prefill time by >= 20% vs the best existing configuration on both Mooncake traces at one or more memory sizes; report hit rate, cached tokens per GiB, and the gain split into precision / placement / tier. Fail: capacity does not limit these workloads; report and rethink |
| K1 | no-loss, GPU (blind): kvfetch/sparse_decode.py with every GDN recurrent state rounded through fp8-e4m3 per-row at every 528-token block during generation (about 60 round trips per 32K trace, far harsher than real hits); H3's full-attention settings, AIME explore 30 problems avg@8; int8 per-row as contrast | fp8 vs full attention, paired by problem: point estimate >= -2 points and 95% CI lower bound >= -5 points |
| K2 | vLLM implementation (fork): fp8 checkpoint pool apart from the fp32 running state, retention-policy hook, host tier that keeps every retained checkpoint; trace replay on GPUs 1-3 | >= 70% of K0's modelled gain end to end at the same memory |
| K3 | online policy: per candidate boundary choose drop / GPU / host and format from bytes vs expected recompute saved | within 10 points of the K0 oracle's gain |

Unchanged rules: never GPU 0; every script takes a config and records provenance; audits by independent
agents before numbers are used. The pilot scripts and results move from the scratchpad into the repo
with K0.

## Prefix-caching status

- K0 (2026-10-28; scripts/k0_simulate.py, configs/prefix_k0.yaml; results/prefix/k0_v2.json, v1 = before
  the audit): FAILED. The oracle (checkpoints exactly where later requests hit, fp8, host tier with
  junctions from host lookups and decode blocks stored) cuts gated prefill time vs the best existing vLLM
  setting (retention 0, periodic or dense, each with host offload) by 5.5 / 4.5 / 2.6% on toolagent and
  3.8 / 4.2 / 3.4% on conversation at 10 / 20 / 30 GiB per instance. With infinite memory and every
  reusable boundary checkpointed it would be 14.0 / 5.6 / 2.9% and 15.8 / 7.4 / 4.8%
  (results/prefix/k0_v2_ceiling.json): no placement, precision or tier passes on these traces.
  - Why: only 57.5% / 38.0% of prompt tokens are reusable at all; 94-97% of hits extend an earlier
    prompt and outputs are 2-3% of input tokens (means 182 / 343), so retention 0's replay boundary
    already sits at the hit; under it checkpoints are 15-23% of GPU cache bytes; host offload recovers
    most of what capacity costs. Against vLLM's default without offload the oracle is 18-25% faster, but
    that gain is the host tier, which vLLM already has.
  - Audit (independent agent, no blocker; it reproduced v1 exactly): output aliasing let several
    follow-ups claim one output block, vLLM's default offload_prompt_only was not modelled, periodic
    retention kept prefill states vLLM never exports (only chunk ends exist), output blocks should be
    (L+O-1)//B -- all fixed in v2 (cuts moved by at most 0.5 points). Not changed: sequential replay
    ignores concurrency (-0.1 to -0.4 points as a bound); cost-model sensitivity <= 0.6 points; 4 instances
    give up to 6.1%, 16 instances 0.9-4.7%.
  - Not covered by these traces: long outputs (thinking models in tool loops), where retention 0
    recomputes every step's output (pilot: 316 vs 115 ms per turn). There dense or periodic retention is
    the existing fix; what is left is the bytes they spend.
