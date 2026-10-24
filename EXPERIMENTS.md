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

## Timeline

| When | Work |
|---|---|
| Nov 2026 | E1, E2, gate decisions; fallback (learned gate) if needed |
| Dec 2026 - Jan 2027 | batched sparse decoding; E3 pilot, then the Qwen3-4B main table |
| Feb - Mar 2027 | E3 on Qwen3-8B and held-out benchmarks; E4 prototype |
| Apr 2027 | E5 on the confirm split; writing |
| May 2027 | submission |
