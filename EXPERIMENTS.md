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

## Timeline

| When | Work |
|---|---|
| Nov 2026 | E1, E2, gate decisions; fallback (learned gate) if needed |
| Dec 2026 - Jan 2027 | batched sparse decoding; E3 pilot, then the Qwen3-4B main table |
| Feb - Mar 2027 | E3 on Qwen3-8B and held-out benchmarks; E4 prototype |
| Apr 2027 | E5 on the confirm split; writing |
| May 2027 | submission |
