# RESULTS — Phase 0 (definitions and infrastructure)

Branch `phase0`. Environment: `.venv` (torch 2.8.0+cu128, transformers 4.57.1, Python 3.11),
RTX A6000 / RTX 6000 Ada (48 GB), shared machine — capped at 3 GPUs from the gate onward.
Every result JSON carries `provenance` (model, context length, page size, w, git commit +
dirty flag, resolved config). Reproduce any step with the command listed in its section.

> **Blinding note.** §7 says `GRADUAL_SHARE_MIN` / `LEAD_TIME_MIN` are fixed *before seeing
> data*. Timing the small eval necessarily computed Phase-1-style statistics (class shares,
> rise times, lead times). They are **deliberately not summarized here**; they sit in
> `results/small_eval/*.json` under `tasks.*.metrics` if you choose to look. Everything below
> is infrastructure validation, cost, or task design.

---

## 1. Trace size estimate → §9.3; storage feasibility for Figures A and B

`python scripts/estimate_trace_size.py configs/trace_size.yaml` → `results/trace_size.json`

Dense = m for every page; top-K = 512 indices + fp16 masses + fp32 residual per (t,l,h).
M_w is **not stored**: in the dense format it is an exact window sum of m.

| scenario | page | heads | dense / sample | top-512 / sample | dense total | top-512 total |
|---|---|---|---|---|---|---|
| spec target: 128K / 2K steps / 32 L / 8 KV | 16 | 8 kv | 8.1 GB | 1.0 GB | — | — |
| small eval (Llama-3.2-3B, 4 L, 40 samples) | 16 | 24 query | 97 MB | 48 MB | 4.3 GB | 2.4 GB |
| large 32K (8B, all L, 400 samples) | 16 | 8 kv | 2.1 GB | 1.0 GB | 863 GB | 414 GB |
| large 32K | 16 | 32 query | 8.5 GB | 4.0 GB | 3.4 TB | 1.6 TB |
| large 128K (8B, all L, 400 samples) | 16 | 8 kv | 8.1 GB | 1.0 GB | 3.2 TB | 452 GB |
| large 128K | 16 | 32 query | 32.5 GB | 4.0 GB | 12.9 TB | 1.8 TB |

Free disk today: `/home` ≈ 96 GB, `/share2` ≈ 112 GB, `/` ≈ 254 GB.

* Small eval: trivially fits in dense, query-head format.
* **Large eval as specified (§5: all layers, 2K steps, 100+ samples × tasks) does not fit in
  either format.** At 128K even top-512 at KV-head granularity is ~450 GB.
* **top-512 cannot represent B = 10 % at 128K / page 16** (needs 820 pages); K must be ≥ 832.
* A single 128K / 2K-step all-layer query-head trace is 32.5 GB — it cannot sit in host RAM
  per sample either; the tracer would need to stream to disk or reduce online.

**Does top-K + residual preserve our metrics?** `python scripts/topk_fidelity.py configs/small_eval.yaml --trace-dir traces/small_eval/meta-llama__Llama-3.2-3B-Instruct__32768 --K 128 256 512 1024`
→ `results/topk_fidelity.json`. Each of the 40 dense small-eval traces is rebuilt from its
top-K pages, the small-eval analysis is rerun, and only the *difference* is reported (no metric
levels). Mass coverage over all pages looks reassuring (top-512 holds 96–97 % at 32K), but
76–78 % of mass is on pinned pages, so the decision-relevant number is eligible mass:

| K at 32K (page fraction) | ≈ same fraction at 128K | eligible mass kept | \|Δ recall@5 %\| mean / worst | pages changing class (query / kv view) | worst class-share shift | worst Δ lead median |
|---|---|---|---|---|---|---|
| 128 (6.3 %) | **512** | 0.58 | 0.045 / 0.059 | 9 % / 13 % | 8.8 pts | 2 steps |
| 256 (12.5 %) | 1024 | 0.71 | 0.022 / 0.029 | 6 % / 7 % | 6.5 pts | 1 |
| 512 (25 %) | 2048 | 0.84 | 0.006 / 0.008 | 2.4 % / 3 % | 2.4 pts | 1 |
| 1024 (50 %) | 4096 | 0.95 | < 0.001 | 0.2 % / 0.6 % | 0.4 pts | 0–1 |

**Recommendation for §9.3: do not use top-512 + residual for Phase 1–2.** At 128K it would
move recall by ~0.05 and class shares by up to ~9 points — the same order as any sensible
decision margin. Faithful top-K needs ≳ 25 % of pages, which saves only ~2× over dense.
Store **dense** traces and cut volume elsewhere: a layer subset (as the small eval does), fewer
samples for the trajectory study than for accuracy (Figure A needs task scores, not traces),
KV-head granularity if per-query-head analysis turns out unnecessary, or online metric
computation for the large sweep. Caveat: this is a 32K proxy matched by page fraction; at
128K attention is likely more spread, which would make top-K worse, not better.

---

## 2. Tracing harness → infrastructure for every figure

`python scripts/trace_smoke.py configs/smoke_2k.yaml` → `results/smoke_2k.json`

Mechanism: an attention function registered through HF's `AttentionInterface` (no kernels,
no monkey-patching). It delegates the attention **output** to HF SDPA, so generations are
bit-identical to an untraced run, and for traced layers recomputes `softmax(qKᵀ·scale)` in
fp32 for the decode query (and the last 32 prompt queries) and reduces it to page mass.
The matching mask function is registered too — without it HF silently skips mask creation
for unknown implementations and SDPA's `is_causal` is top-left aligned, which is wrong for
chunked prefill.

Validated on 2K RULER prompts, 32 decode steps, all layers, prefill chunk boundary placed
inside the observation window:

| model (GQA group) | fp32 isolation max\|Δm\| | worst / bound | fp32 cross-path max\|Δm\| | tokens identical | row-sum err (fp16) | bf16 top-5 % overlap vs eager | round-trip full / top-K |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B (4) | 1.0e-6 | 0.05 | 7.0e-6 | yes | 3.7e-4 | 0.994 | ✓ / ✓ |
| Qwen2.5-1.5B (6) | 6.5e-4 | 0.12 | 1.7e-3 | yes | 3.8e-4 | 0.903 | ✓ / ✓ |
| Qwen3-1.7B (2) | 3.0e-6 | 0.12 | 3.2e-5 | yes | 3.6e-4 | 0.990 | ✓ / ✓ |

(Run at clean commit `9c5cfa8` on an RTX A6000. Isolation and cross-path columns are the max
over decode steps and the prefill observation window.)

* **Isolation**: the tracer wraps HF's own `eager_attention_forward`, so recompute and HF's
  eager weights see identical q/k in one forward. Bound: `1e-5 + 5e-7·max|logit|` per row.
* **Qwen2.5 is not a bug**: its layer-0 head 1 has logits of **~9.4K–23K** (q/k bias).
  fp32's relative ε then means ~1e-3 absolute probability noise in *any* implementation; all
  other layers agree to ≤ 3e-6. Consequence: bf16 eager attention is unreliable for this
  family (bf16 logit resolution at 9.5K is 64) — our fp32 recompute matches SDPA's fp32
  accumulation, not eager's.
* Tracing all layers costs ~20 % per decode step at 2K; 4 traced layers cost less.

---

## 3–4. Metric library and trajectory classifier

`kvfetch/metrics/`, `python -m pytest` → 16 tests, all hand-built toy traces with exact
binary-fraction values. Conventions fixed in code and `configs/base.yaml`:

| quantity | convention |
|---|---|
| step | s = 0 is the last prompt query; s ≥ 1 is the query of generated token s (each step produces one token) |
| M_w[t] | Σ m[t+1 … t+w] (exact, derived) |
| eligible pages | full prompt pages minus the sink page; the decode region (pages touching generated tokens) is pinned. B = ceil(frac × eligible) |
| recall@B,k | predictor score at t (info ≤ t) vs. top-B of M_w[t+k], mass-weighted |
| lead time | resident set *for* step t = top-B(score[t−1]); lead = t_true − t_pred; t_true = first step in top-B of m. Last-step predictor ⇒ exactly −1 on a new page (tested). Censored if top-B already at step 0; "missed" if never predicted. Modes `first` (spec) and `contiguous` |
| churn | \|S(t) Δ S(t−1)\| / B |
| trajectory | on m relative to the per-step top-B cutoff (not M_w — a window sum turns every jump into a w-step ramp), with hysteresis; classes stable-low / stable-high / oscillating / sudden-jump / gradual-rise / **other** (e.g. high-then-falls). Each onset after a deep-low period also gets an onset kind + rise time |
| heads | traces store query heads; analyses run per query head and per KV group (`kv_mean`) |

Known-value tests include the §10 example: a page rising linearly gives lead −1 under the
last-step predictor and exactly h−1 under h-step linear extrapolation; a 20-step ramp is
gradual-rise with rise time 19; a step is sudden-jump with rise time 1.

---

## 5. Small-eval runner → the ≤ 1 h unit test for Figures A/B iteration

`python scripts/run_small_eval.py configs/small_eval.yaml` → `results/small_eval/meta-llama__Llama-3.2-3B-Instruct__32768.json`,
traces in `traces/small_eval/…` (dense, query heads, 4.3 GB).

Model is **provisional** (Llama-3.2-3B-Instruct) until §9.4 is decided; rerun with
`--set small_eval.model=… --set small_eval.context_length=…`.

| stage | wall |
|---|---|
| model load | 3 s |
| retrieval-head probe (4 × niah_single_2, all layers, 8 steps) | 39 s |
| trace RULER multi-key: 20 × (32K prefill + 256 steps) | 306 s (6.4 s prefill, 33 ms/step) |
| trace long-generation: 20 × (32K prefill + 256 steps) | 238 s (5.5 s prefill, 24 ms/step) |
| analysis (16 CPU workers, overlapped with tracing) | 23 s after the last trace |
| **total** | **10.3 min — within the 60 min budget** |

* Pipeline: probe → traced layers {3, 8, 14, 24} = early / mid / late at 0.1 / 0.5 / 0.9 depth
  + the strongest retrieval layer (layer 8, head 8: 0.67 of attention on the needle's pages
  while emitting the answer). Metrics per spec: recall@5 % for k ∈ {1, 4}, lead-time median,
  class shares, churn — computed for the last-step and EMA reference predictors only.
* Single-threaded analysis was 53 s / trace (35 min for 40 traces), which would have pushed
  a larger iteration model over the hour; it now runs in a process pool alongside tracing.
* Prefill used 8K chunks in this run; single-chunk (flash) prefill is now the default up to
  32K, ~2× faster prefill.
* RULER multi-key score 95 % (string_match_all). Long-generation is trace-only.

**Infrastructure facts from the run (not trajectory statistics):**

* **Pinned pages hold 76–78 % of all attention mass** (sink page + decode region). Recall
  and lead time are therefore defined over the remaining ~23 %; counting the sink would
  make any predictor look excellent. Pinned pages still count toward the Figure A budget.
* **Standard RULER does not put the jump in the decode window.** For the three strongest
  retrieval heads in traced layers, the queried needle is already in the top 5 % at the
  32nd-to-last prompt token in 95 % of samples, at decode step 0 in 100 %, and during the
  answer in 100 %. The answer always ends at step 5 (stop token would win), after which the
  needle's mass drops from 0.60 to 0.03; **251 of 256 decode steps are a forced continuation**
  with stop tokens masked. As configured, the "sudden-jump" task tests the *end-of-prefill*
  signal (stored: `obs_mass`, last 32 prompt queries), not decode-time fallback. Options for
  you (§9.2 / small-eval tasks): multi-query in one answer (RULER `niah_multiquery` with many
  keys — each answer item is a fresh jump, natural decode), or multi-turn re-reference where
  the next question's tokens are fed as decode steps.

---

## 6. Model gate → §4, §9.4 (which models generate the traces behind every figure)

`python scripts/run_model_gate.py configs/model_gate.yaml --claim --skip-done` (work-queue
workers, one per GPU) → `results/model_gate/<model>__<len>.json`;
`python scripts/summarize_model_gate.py configs/model_gate.yaml` → `results/model_gate/summary.md`.

RULER NIAH re-implemented from `NVIDIA/RULER` (same templates, needle format, 40 depths,
singularization, length fitting, `string_match_all`). Chat templates, greedy, 128 new tokens,
50 samples per task. **Pass rule** (config): the needle group mean (s1–s3) **and** the
multi-key group mean (mk1–mk3) must both be ≥ 85.6, RULER's own bar.

GATE_TABLE_PLACEHOLDER

---

## Infrastructure notes for later phases

* **128K needs chunked prefill** (memory). The chunked path passes an explicit mask, so SDPA
  drops from flash to the mem-efficient kernel (~2× slower prefill at 32K). Fix before the large
  eval: lower-right causal bias for chunks.
* The tracer keeps a host array `[T, L, H, P]` per sample. Fine for the small eval (100 MB);
  a 128K / 2K-step / all-layer trace is 8–33 GB, so it would need streaming to disk.
* Gate-result provenance: the first four gate jobs ran in workers started before the
  provenance fix (`00f1503`), which read HEAD when the job *finished*. Their `git.commit` was
  corrected to `66c01a9`, the commit those workers loaded, and each carries a
  `provenance_note`. Every later result captures the commit at process start and embeds the
  resolved config.

---

## Decisions I made that you may want to revisit

| decision | where | why |
|---|---|---|
| Step 0 = last prompt query; one step per generated token | `tracing/schema.py` | every recorded step produced an output token; end-of-prefill is a first-class step |
| Sink page + decode region pinned, excluded from metrics; B = ceil(frac × eligible) | `definitions.pinned` | the sink alone holds most mass and would inflate recall |
| Lead time: set *for* step t uses info ≤ t−1; truth = top-B of instantaneous m | `lead_time` | makes "negative = stall" literal; the last-step predictor scores exactly −1 |
| Classify on m with hysteresis, not M_w; explicit `other` class | `trajectory` | M_w smears every jump into a w-step ramp; high-then-falls fits none of the five classes |
| Traces store query heads; analyses report query and `kv_mean` views | `definitions.head_views` | spec says keep per-head; tiers act per KV group |
| RULER decoded to a fixed 256 steps with stop tokens masked; `natural_stop_step` recorded | `small_eval.tasks` | fixed shapes and timing; post-stop steps can be dropped in analysis |
| Long-generation = summarize a 32K prefix of a PG19 test book | `longgen` | stand-in "gradual" task until §9.2 |
| Retrieval layer = layer whose best head puts most mass on the queried needle while answering | `small_eval.layers.probe` | cheap, matches "strongest retrieval heads" |
| Gate pass = both group means ≥ 85.6 | `gate.pass_threshold` | RULER's convention; see §6 for how it plays out |
| Regex sentence splitter instead of nltk | `tasks/ruler.py` | nltk refuses to load data from a group-writable home dir; only affects needle insertion points |

---

## Open items (§9) — what I need from you before Phase 1

1. **`GRADUAL_SHARE_MIN`, `LEAD_TIME_MIN`.** Also fix *which denominator* "gradual-rise share"
   uses. The analysis reports three: all eligible pages, ever-high pages, and onsets that
   follow a deep-low period (`gradual_share_of_onsets`). I suggest the last: it is "of the
   events that would cause a stall, how many give warning". The classifier thresholds
   (`floor_ratio`, `sudden_max_rise_steps`, `exit_ratio`, `oscillating_min_transitions`)
   directly set that split, so fix them together with the decision values. Hardware input
   for `LEAD_TIME_MIN` (Llama-3.1-8B, page 16): a page is 64 KiB per layer, 2 MiB across 32
   layers. At ~24 GB/s PCIe 4.0 and a ≥ 20 ms batch-1 step, one step moves ~240 full pages vs.
   B = 410 pages (5 % at 128K). Since lead 0 already includes one full step of transfer time,
   at batch 1 extra lead only matters for churn bursts above ~60 % of B per step. It matters
   much more at batch > 1 or on slower links. Pick the regime the paper targets.
2. **Custom multi-turn re-reference task.** Same mechanism also fixes the RULER problem in §5:
   feed each new question's tokens as decode steps so the jump lands inside the traced window.
3. **Trace granularity.** Recommendation: dense, with volume cut by layer subset / sample
   count / KV-head view (§1). Top-512 + residual changes recall by ~0.05 and class shares by
   up to ~9 points at the 128K-equivalent page fraction.
4. **Reference and iteration models.** See §6. The three 8B reference candidates were not
   gated: they were held to stay within 3 GPUs (≈ 12 GPU-hours at 16K/32K; 128K for Qwen
   additionally needs YaRN). Release with `rm results/model_gate/*8B*__*.claim
   results/model_gate/*7B*__*.claim` and start workers.
