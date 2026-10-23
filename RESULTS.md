# RESULTS — Phase 0 (definitions and infrastructure)

**Visual, plain-language version: [`report.html`](report.html)** (open in a browser; regenerate with
`python scripts/make_report.py configs/report.yaml`). Published as a private artifact:
https://claude.ai/artifact/XiS93psCJ1vP6fAPbUC55u (republish with `--artifact <path>` and the Artifact tool).

Branch `phase0`. Environment: `.venv` (torch 2.8.0+cu128, transformers 4.57.1, Python 3.11),
RTX A6000 / RTX 6000 Ada (48 GB), shared machine — after the first hour of the gate, limited to devices 1, 2, 3 (+6 later).
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
levels). Mass coverage over all pages looks reassuring (top-512 holds 96–99 % at 32K), but
44–78 % of mass sits on pinned pages (model-dependent, §5), so the decision-relevant number is
eligible mass. Llama-3.2-3B traces:

| K at 32K (page fraction) | ≈ same fraction at 128K | eligible mass kept | \|Δ recall@5 %\| mean / worst | pages changing class (query / kv view) | worst class-share shift | worst Δ lead median |
|---|---|---|---|---|---|---|
| 128 (6.3 %) | **512** | 0.58 | 0.045 / 0.059 | 9 % / 13 % | 8.8 pts | 2 steps |
| 256 (12.5 %) | 1024 | 0.71 | 0.022 / 0.029 | 6 % / 7 % | 6.5 pts | 1 |
| 512 (25 %) | 2048 | 0.84 | 0.006 / 0.008 | 2.4 % / 3 % | 2.4 pts | 1 |
| 1024 (50 %) | 4096 | 0.95 | < 0.001 | 0.2 % / 0.6 % | 0.4 pts | 0–1 |

Qwen3-4B traces (`results/topk_fidelity_qwen3_4b.json`). Attention here is much more
concentrated:

| K at 32K (page fraction) | ≈ same fraction at 128K | eligible mass kept | \|Δ recall@5 %\| mean / worst | pages changing class (query / kv view) | worst class-share shift | worst Δ lead median |
|---|---|---|---|---|---|---|
| 128 (6.3 %) | **512** | 0.91 | 0.005 / 0.008 | 4 % / 7 % | 4.7 pts | 3 steps |
| 256 (12.5 %) | 1024 | 0.95 | 0.002 / 0.004 | 3 % / 3.5 % | 3.5 pts | 1 |
| 512 (25 %) | 2048 | 0.98 | < 0.001 | 0.8 % / 1.3 % | 1.1 pts | 1 |
| 1024 (50 %) | 4096 | 0.99 | < 0.001 | < 0.2 % | 0.1 pts | 0–1 |

**Recommendation for §9.3:**

* **Phase 1 (trajectory classes, rise and lead times — the decision point): store dense.**
  At the 128K-equivalent of top-512, class shares move by 4.7 (Qwen3) to 8.8 (Llama) points
  and lead medians by 2–3 steps. That is the same order as any sensible decision margin.
  Getting within ~1–2 points needs K ≥ 25 % of pages, a saving of only ~2× over dense.
* **Phase 2 recall comparisons: top-K is model-dependent.** It is fine on Qwen3-4B (|Δ recall|
  < 0.01 at the 512-equivalent), but not on Llama-3.2-3B (~0.05).
* Cut dense volume elsewhere: a layer subset (as the small eval does), fewer trajectory samples
  than accuracy samples (Figure A needs task scores, not traces), KV-head granularity if
  per-query-head analysis turns out unnecessary, or online metric computation for large sweeps.
* Caveat: these are 32K proxies matched by page fraction. At 128K attention is likely more
  spread out, which would make top-K worse, not better.

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

`python scripts/run_small_eval.py configs/small_eval.yaml [--set small_eval.model=Qwen/Qwen3-4B]`
→ `results/small_eval/<model>__32768.json`; traces in `traces/small_eval/…` (dense, query heads).
The config default is Llama-3.2-3B (chosen before the gate). It was rerun on Qwen3-4B, the
gate's recommendation; switch the default once §9.4 is decided.

| stage | Llama-3.2-3B (GPU to itself) | Qwen3-4B (GPU shared with an 8B gate job) |
|---|---|---|
| model load | 3 s | 15 s |
| retrieval-head probe (4 × niah_single_2, all layers, 8 steps) | 39 s | 51 s |
| trace RULER multi-key: 20 × (32K prefill + 256 steps) | 306 s (6.4 s prefill, 33 ms/step) | 607 s (10.8 s, 76 ms/step) |
| trace long-generation: 20 × (32K prefill + 256 steps) | 238 s (5.5 s, 24 ms/step) | 597 s (10.7 s, 75 ms/step) |
| analysis (16 CPU workers, overlapped with tracing) | 23 s after last trace | 25 s after last trace |
| **total** | **10.3 min** | **21.7 min (upper bound)** |
| traces | 4.3 GB | 5.6 GB |

Both runs are within the 60 min budget.

* Pipeline: probe → traced layers = early / mid / late at 0.1 / 0.5 / 0.9 depth + the
  strongest retrieval layer. Llama-3.2-3B: {3, 8, 14, 24}, with layer 8 head 8 putting 0.67 of
  its attention on the needle's pages while answering. Qwen3-4B: {4, 18, 29, 32}, with layer
  29 head 11 at 0.90. Metrics per spec: recall@5 % for k ∈ {1, 4}, lead-time median, class
  shares, churn, computed for the last-step and EMA reference predictors only.
* Single-threaded analysis was 53 s / trace (35 min for 40 traces), which would have pushed
  a larger iteration model over the hour; it now runs in a process pool alongside tracing.
* The Llama run used 8K prefill chunks. Single-chunk (flash) prefill is now the default up to
  32K, ~2× faster prefill.
* RULER multi-key score: 95 % (Llama-3.2-3B), 100 % (Qwen3-4B). Long-generation is
  trace-only.

**Infrastructure facts from the runs (not trajectory statistics):**

* **Pinned pages hold a large, model-dependent share of attention mass:** 76–78 % for
  Llama-3.2-3B and 44–45 % for Qwen3-4B (sink page + decode region). Recall and lead time are
  defined over the remainder; counting the sink would make any predictor look excellent.
  Pinned pages still count toward the Figure A budget.
* **Standard RULER does not put the jump in the decode window (both models).** For the
  strongest retrieval heads in traced layers, the queried needle is already in the top 5 % at
  the 32nd-to-last prompt token in 95 % (Llama) / 100 % (Qwen3) of samples, and at decode
  step 0 and throughout the answer in 100 %. The answer ends at step 5–6 (Llama) / 9–10
  (Qwen3), when a stop token would win. After that the needle's mass drops from 0.60 → 0.03
  (Llama) / 0.66 → 0.06 (Qwen3). So **~96–98 % of the 256 decode steps are a forced
  continuation** with stop tokens masked. As configured, the "sudden-jump" task tests the
  *end-of-prefill* signal (stored: `obs_mass`, last 32 prompt queries), not decode-time
  fallback. Options for you (§9.2 / small-eval tasks):
  * multi-query in one answer (RULER `niah_multiquery` with many keys): each answer item is a
    fresh jump, with natural decode;
  * multi-turn re-reference, where the next question's tokens are fed as decode steps.

---

## 6. Model gate → §4, §9.4 (which models generate the traces behind every figure)

`python scripts/run_model_gate.py configs/model_gate.yaml --claim --skip-done` (work-queue
workers, one per GPU) → `results/model_gate/<model>__<len>.json`;
`python scripts/summarize_model_gate.py configs/model_gate.yaml` → `results/model_gate/summary.md`.

RULER NIAH re-implemented from `NVIDIA/RULER` (same templates, needle format, 40 depths,
singularization, length fitting, `string_match_all`). Chat templates, greedy, 128 new tokens,
50 samples per task. **Pass rule** (config): the needle group mean (s1–s3) **and** the
multi-key group mean (mk1–mk3) must both be ≥ 85.6, RULER's own bar.

**Iteration-model candidates (final):**

| model | len | s1 | s2 | s3 | mk1 | mk2 | mk3 | needle avg | multi-key avg | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 16K | 100 | 98 | 100 | 96 | 76 | 26 | 99.3 | 66.0 | no |
| Llama-3.2-1B-Instruct | 32K | 100 | 94 | 98 | 92 | 82 | 8 | 97.3 | 60.7 | no |
| Llama-3.2-3B-Instruct | 16K | 100 | 100 | 100 | 98 | 100 | 98 | 100 | 98.7 | **yes** |
| Llama-3.2-3B-Instruct | 32K | 100 | 100 | 100 | 98 | 94 | 60 | 100 | 84.0 | no (−1.6) |
| Qwen2.5-1.5B-Instruct | 16K | 100 | 100 | 100 | 98 | 98 | 56 | 100 | 84.0 | no |
| Qwen2.5-1.5B-Instruct | 32K | 100 | 100 | 100 | 94 | 86 | 44 | 100 | 74.7 | no |
| Qwen2.5-3B-Instruct | 16K | 100 | 100 | 100 | 100 | 100 | 68 | 100 | 89.3 | **yes** |
| Qwen2.5-3B-Instruct | 32K | 100 | 100 | 100 | 98 | 98 | 64 | 100 | 86.7 | **yes** (+1.1) |
| Qwen3-1.7B | 16K | 100 | 100 | 100 | 100 | 86 | 88 | 100 | 91.3 | **yes** |
| Qwen3-1.7B | 32K | 100 | 100 | 100 | 98 | 74 | 46 | 100 | 72.7 | no |
| **Qwen3-4B** | 16K | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 98.7 | **yes** |
| **Qwen3-4B** | 32K | 100 | 100 | 100 | 100 | 98 | 96 | 100 | **98.0** | **yes** |

(s = niah_single, mk = niah_multikey; 50 samples each, so a task near 60 % has ±7 pts of
sampling noise.)

* Only `niah_multikey_3` (UUID keys in a haystack of UUID needles) separates the models; every
  other task is ≥ 86 for every model ≥ 3B.
* **Qwen3-4B is the only candidate that clears 32K with margin.** It scores 96 on mk3, where
  the next best scores 64. Llama-3.2-3B (84.0) and Qwen2.5-3B (86.7) sit within noise of the
  bar at 32K.
* **Recommendation for §9.4 (iteration model): Qwen3-4B, small eval at 32K.** It follows the
  spec's rule ("longest length it passes"). Its family partner Qwen3-8B has the same GQA
  shape (8 KV heads, group 4), which helps the §4 guard. Caveats: Qwen3 is natively 32K, so
  a 128K Qwen3-8B reference needs YaRN, while Llama-3.1-8B is native 128K. Qwen2.5-3B has only
  2 KV heads (group 8) and the large-logit heads described in §2.
* Wall time per job ranged 4–39 min; Qwen3-4B @ 32K took 37 min (6.4 s prefill per sample).

**Reference candidates (§4: the reference model must pass too):**

| model | len | s1 | s2 | s3 | mk1 | mk2 | mk3 | needle avg | multi-key avg | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 16K | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 99.3 | yes |
| Llama-3.1-8B-Instruct | 32K | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | yes |
| Qwen2.5-7B-Instruct | 16K | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 99.3 | yes |
| Qwen2.5-7B-Instruct | 32K | 98 | 100 | 100 | 100 | 98 | 90 | 99.3 | 96.0 | yes |
| Qwen3-8B | 16K | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | yes |
| Qwen3-8B | 32K | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 99.3 | yes |

All three pass at 16K and 32K, near ceiling, so the gate does not separate them. The choice
rests on the §4 guard and on 128K:

| pairing (iteration → reference) | for | against |
|---|---|---|
| **Qwen3-4B @ 32K → Qwen3-8B** (recommended) | same family and GQA shape (8 KV heads, group 4); both pass 32K with margin | Qwen3 is natively 32K, so a 128K reference needs YaRN and its own 128K gate |
| Llama-3.2-3B @ 16K → Llama-3.1-8B | native 128K reference, 100 % at 32K | small eval capped at 16K (fails 32K by 1.6 pts, within noise; a re-gate with more samples could flip it) |
| Qwen3-4B @ 32K → Llama-3.1-8B | best small model + native-128K reference | cross-family: e.g. pinned-mass share is 44 % (Qwen3-4B) vs 77 % (Llama-3.2-3B), so the §4 agreement check is at risk for reasons unrelated to trajectories |

None of the references has been gated at 128K yet; whichever you pick needs that before the
large eval. Phase 1's guard itself only needs 32K.

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
   layers. At ~24 GB/s PCIe 4.0 and a ≥ 20 ms batch-1 step, one step moves ~230 full pages vs.
   B = 410 pages (5 % at 128K). Since lead 0 already includes one full step of transfer time,
   at batch 1 extra lead only matters for churn bursts above ~56 % of B per step. It matters
   much more at batch > 1 or on slower links. Pick the regime the paper targets.
2. **Custom multi-turn re-reference task.** Same mechanism also fixes the RULER problem in §5:
   feed each new question's tokens as decode steps so the jump lands inside the traced window.
3. **Trace granularity.** Recommendation: dense for Phase 1, with volume cut by layer subset,
   sample count or KV-head view (§1). At the 128K-equivalent page fraction, top-512 +
   residual moves class shares by 5–9 points and lead medians by 2–3 steps. For Phase 2
   recall it is acceptable on Qwen3-4B (Δ < 0.01) but not on Llama-3.2-3B (~0.05).
4. **Reference and iteration models.** See §6. Iteration: Qwen3-4B @ 32K is the
   recommendation. Reference: gated at 16K/32K only. Neither length is the 128K the large eval
   uses, so whichever reference you pick needs a 128K gate before Phase 3 (Qwen candidates
   with YaRN).
