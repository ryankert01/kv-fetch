# Harvest: new architectures and decoding paradigms (does acceleration transfer)

Date: 2026-11-11. Read-only literature work. No GPU jobs were run, nothing was posted, nothing was
committed. This is the only file written.

How to read this file: section 1 is the reading list, section 2 the extracted claims, section 3
the tensions that were considered and dropped, section 4 the five kept question cards, section 5
the searches and what was not covered.

"Verified" means I read the sentence or number in this session, either in the arXiv abstract
(arXiv API, raw text) or in the paper's arXiv HTML through WebFetch. WebFetch returns text through
an extraction model and only the first 100,000 characters of each page, so body quotes should be
checked against the PDF before they are cited anywhere. "Abstract" means only the abstract was read.

Scope: GDN/Mamba hybrids, MoE at inference, built-in MTP heads, diffusion LMs. The question asked
of every paper was: which acceleration technique transfers, which breaks, and is the stated reason
tested.

## 1. Reading list

All ids below were returned by the arXiv API (id_list query) in this session unless marked.

Deep read (HTML body, relevant sections):

| id | short name | what it is |
|---|---|---|
| 2611.12127 | Lost in Rounding | bf16 round-to-nearest recurrent state makes hybrids reason longer |
| 2611.14246 | SinkState | low-rank "linear attention sink" in GDN/Mamba states; on-the-fly SVD |
| 2611.04055 | HybridSpec | speculative decoding for GDN-MoE hybrids, single request |
| 2610.38981 | Why SD works better on sparse MoE | routing overlap explains SD speedups above MoESD's prediction |
| 2609.23314 | ValueDiff | value-geometry KV eviction for sink-suppressed models |
| 2611.09709 | RidgeFill | late-layer cache fill; splice tests of which cache component carries the error (first 100k chars only) |
| 2610.39853 | dLLM-KVQ | KV cache quantization for diffusion LMs |
| 2609.38169 | STEPQuant | recurrent-state bit allocation by error size and memory lifetime |
| 2610.39931 | GGA + pure-write window | low-bit GDN state; recent writes kept apart from the old state |
| 2610.06750 | Balancing Memory Pathways | hybrids rely more on attention than on the state; auxiliary loss shifts it |
| 2609.04434 | What Attention Recalls and Recurrence Controls | split-prefill: what flows through KV vs state |
| 2610.14754 | GateSight | GDN prefill signals predict which KV entries to keep |
| 2610.13778 | ReplaySSM | checkpoint + input buffer instead of per-token state writes |
| 2609.39185 | Low-Discrepancy Dither | rounding rule for quantized recurrent state |
| 2505.22618 | Fast-dLLM | approximate KV cache + confidence-threshold parallel decoding |

Abstract only, hybrids: 2611.01864 (LoReS, "Quantizing Recurrent Language Models Is Easier Than You
Think"), 2610.29545 PACE, 2609.38166 LeapQuant, 2609.33051 SketchSSM, 2608.30386 DASC, 2611.01909
Who Asks Who Answers, 2603.22473 Component Ablation, 2606.15378 Rethinking Efficient Attention in
Hybrids, 2610.26830 Competing Learning, 2609.15545 Token Before the Value, 2611.06133 Long-Context
Decoding in Hybrids, 2611.04313 Coarse Semantic Blocks, 2610.41420 SparseGDN, 2611.01022
Jet-QSparse, 2609.27470 DeltaS, 2607.02303 HOLA, 2510.20787 (hybrid sparse attention + learnable
eviction), 2611.08133 Hybrid State Transfer, 2611.12062 KV translation between hybrids, 2610.41994
DAMI, 2611.08376 Muir, 2610.42148 Flide, 2610.38095 Nemotron 3.5, 2604.12374 Nemotron 3 Super,
2606.15007 Nemotron 3 Ultra, 2505.06708 Gated Attention, 2606.22874 SpotAttention, 2605.06221
UniPrefill (last two: listing only).

Abstract only, MoE and speculative decoding: 2610.34440 (expert bytes, kernel bandwidth), 2505.19645
MoESD, 2505.16056 (local routing consistency of 20 MoE LLMs), 2604.23150 (expert activation
patterns, multi-node), 2610.38331 Soft-MTP, 2611.00588 TVP, 2611.14399 REAL, 2611.13017
Multi-Future Drafting, 2611.12356 DForkTree, 2611.11371 D3Tree, 2611.06425 D3D, 2606.11552,
2606.10820 K-Forcing, 2611.10634 CVSA.

Abstract only, diffusion: 2610.06996 MaskAhead, 2605.20813 PulseCol, 2609.11998 Fixed State Long
Reach, 2608.06628 LLaDA-Hybrid, 2607.04206 Sangam, 2606.07571 BiCache, 2606.02544 SimSD,
2603.25702 S2D2, 2605.14305 FeF-DLLM, 2605.18817 MRP, 2608.27514, 2607.01775 Set Diffusion,
2602.08404 TEAM, 2610.19855, 2610.40847, 2509.23094 d2Cache, 2607.16339 LaCache.

Anchors (existence checked by id): 2412.06464 Gated Delta Networks, 2411.13676 Hymba, 2403.19887
Jamba, 2504.03624 Nemotron-H, 2406.07522 Samba, 2501.08313 MiniMax-01, 2412.07067 MoE-CAP,
2401.04088 Mixtral, 2502.09992 LLaDA, 2508.15487 Dream 7B, 2503.09573 Block Diffusion, 2505.15781
dKV-Cache, 2506.17298 Mercury, 2410.13212 AsymKV, 2502.15075 Quantize What Counts, 2502.04420
KVTuner, 2504.15364 KeyDiff, 2603.27467 TurboAngle.

Seen only as a web-search snippet, not checked by id: 2606.15453 (ST-MoE prefetching), 2607.08780
(Sticky Routing), 2605.27081, 2610.41508 HiState, 2611.04794 Bevy, 2610.39216 PDScope. Olmo-Hybrid:
no id found, not used.

Own-project numbers used (read in `/home/ryankert_l/kv-fetch/kv-fetch/EXPERIMENTS.md` lines 212-236):
H1 oracle@512 misses 22.7% of attention on Qwen3.5-4B vs 11.6% on Qwen3-4B; H-b offline "next-token
KL 0.011 at 22.9% missing; Qwen3-4B 0.035 at 14.1%", "7.4-7.5x lower KL over three pairs with
matched missing attention (confounded: 8 perturbed layers vs 36)"; H-b failed online (ratio 1.03);
H-d: GDN beta prior has Spearman 0.09 with page mass.

## 2. Extracted claims

### 2.1 Recurrent-state precision (hybrids)

- 2611.12127, abstract: "With bf16 RTN state, five of six models reason longer (geometric-mean
  length 1.07x to 1.27x...)"; "On Qwen3.5-9B, cap hits rise from 118 to 223 and accuracy falls from
  85.6% to 70.8%"; "85% of them are exact repetition loops"; "Same-width bf16 SR removes the length
  increase in four of five affected models... so the failure is mostly rounding bias, not missing
  resolution"; "SR on 25% of heads (an intermediate-decay band) removes the effect, while random
  heads and the top teacher-forced-KL octiles do not; in Nemotron-3 and Qwen3.5-9B the effect is
  spread across heads."
- 2611.12127, body: toy model h <- round(e^(-r) h), u = 2^-9: RTN freezes decay for r below about u,
  distorts it for u < r < about 4u, is nearly unbiased above. The toy "does not predict which heads
  matter". Nemotron-Nano-9B-v2 decay octiles: teacher-forced-KL share q0 49%, q1 37%, q2 15%. SR on
  q1+q2 (864 heads, 25%, decay 0.57u to 4.1u) recovers F_log 1.03 [0.86, 1.23]; SR on q0 alone
  -0.12; top-2 KL octiles 0.59; random 864 heads 0.06. The band test fails on Qwen3.5-9B (F_log
  0.35). "Why their freezing does not change behaviour remains open." "The Falcon-H1R null is
  unexplained." The paper does not relate the effect to state rank, a dominant subspace, or state
  norm. Fixes tested: SR, FP32 on head subsets, FP16 RTN.
- 2609.39185, abstract: "Round-to-nearest behaves differently: because it discards small updates,
  its error keeps growing, so it can look best in short evaluations yet falls far behind over long
  generations." Body: Lemma 1 works on the combined update "δt = (at−1)ĥt−1+wt", so decay and write
  are not separated. Dither on the top 25% of entries by memory keeps "75–87% of the full gain".
  With rounding every 8th write the dither advantage "mostly disappeared on the hybrids". The paper
  "measures KL rather than generation length or task accuracy". Models are 130M to 1.4B.
- 2609.38169 (STEPQuant), body: "longer-lived state units tend to exhibit larger accumulated
  errors"; Spearman about 0.80 between gate half-life and accumulated state error over 2304 heads
  under INT6; more bits go to units with retention near 1. "uniform quantization substantially
  increases output length"; Kimi 4-bit average AIME output 63.40K tokens. Round-to-nearest only;
  stochastic rounding and state rank are not discussed.
- 2611.01864 (LoReS), abstract: "decay-based mixed-precision to protect slowly decaying heads or
  channels, and stochastic rounding to limit error accumulation under repeated requantization."
- 2610.29545 (PACE), abstract: "PACE rounds the state only once every few generated tokens";
  "Rounding less often reduces the error that accumulates in the state"; "the large entries of the
  gated-delta state gather in columns"; "It also keeps the heads that retain rounding error longest
  at eight bits."
- 2609.38166 (LeapQuant), abstract: "per-window quantization, which leaps over a window of tokens
  and quantizes the state only once at its end"; "retains the state's largest outliers as a few
  high-precision Compensator Tokens".
- 2610.39931, body: the cause is given only as a possibility ("this feedback offers a possible
  explanation for how stalled reasoning can persist"); "We have not yet fully characterized how
  information written by tokens is retained and read during state recurrences". Pure-write window
  m = 8 only, stored as FP8 pairs. STEP@44+PW-S8 beats STEP@55 by 3.29 (4B) and 1.57 (9B) points at
  5.78-5.79 vs 5.68 bits per value. STEP@44: 36.66%/30.67% of outputs hit the 100K cap; GGA@44+PW-S8:
  4.61%/2.44%. Rounding mode not stated.
- 2610.13778 (ReplaySSM), body: window L = 16; "A shared checkpoint is requantized once per window,
  so its error falls as L grows"; quality gain attributed to separate read and update precisions
  ("Separate precisions break the feedback loop"), all arms use SR. Shared NVFP4+SR loses
  5.88/13.05/10.92 points (MMLU-Pro/GPQA-Diamond/Math-L5); NVFP4 read with FP16 update loses
  0.15/2.36/0.10.
- 2611.14246 (SinkState), body: "the queries, keys, and values of linear layers concentrate in a
  stable, low-dimensional subspace, a phenomenon we call the linear attention sink"; on PG-19 the
  top-1 direction captures 39-74%, top-8 73-88%; "semantics-independent"; 2 to 15 of up to 128
  components hold 99% of the state. "The recurrent states are kept in FP32 for all models", so the
  paper has no precision result.
- 2611.08133 (HST), abstract: "More accurate REC reconstruction need not improve continuation."
- 2604.12374 is cited by 2611.12127 for "FP16 SSM cache raises verbosity up to 37%" (not read in
  2604.12374 itself; its abstract does not say it).

### 2.2 Division of labour and KV eviction in hybrids

- 2609.04434, abstract: "Exact retrieval survives only through attention (64-98% of full accuracy)
  and collapses to zero through recurrence. Output language and persona reverse the pattern".
  Table 1 (rec-only / kv-only): Qwen3.5-4B KV retrieve 0.00 / 0.89, language following 0.70 / 0.01,
  persona 0.49 / 0.14. "The recurrent state preserves the semantic field of seen words rather than
  their identities." Limitation: "Split-prefill resets one cache while preserving the other, which
  produces a partly out-of-distribution combination." Whole-context ablation only; no selective
  eviction.
- 2610.06750: "we find a substantially greater reliance on attention than on the recurrent state".
  Qwen3.5-4B QA: full 30.2%, attention-only 20.2%, recurrent-only 5.1%. With a LoRA (rank 16) and an
  auxiliary loss that "masks attention to the preceding context", recurrent-only rises from 5.0% to
  22.8% while attention-only goes 25.0% to 23.7%. No KV-eviction experiment.
- 2611.01909, abstract: "linear attention constructs the query, while softmax attention performs
  the lookup".
- 2610.14754 (GateSight): utility of a KV entry is "how much removing it changes the attention
  output over the model's own continuations" (attention output, not logits). "useful entries tend
  to arrive where the recurrent state updates more strongly"; top decile of slow-head beta*||v||
  reaches the top utility decile 18.2% of the time (1.82x chance); per-layer correlations +0.06 to
  +0.19. The attention output gate's mean "differs by more than an order of magnitude across
  layers" and is used only as a per-layer budget. No test of whether the state still carries
  evicted tokens. ValueDiff is not a baseline.
- 2609.27470 (DeltaS), abstract: keeps chunks "inducing larger normalized state change"; "the two
  memories of hybrid architectures can work cooperatively."
- 2607.02303 (HOLA), abstract: "the cache stores associations that should not be forced through
  that state", yet the keep rule is "tokens with large beta * ||e||, the prediction residual
  actually committed to the state".
- 2609.23314 (ValueDiff): BOS attention mass 0.408 (Qwen3-4B), 0.034 (Qwen3.5-4B), 0.031
  (Qwen3.5-9B), 0.591 (Llama-3.1-8B). sigma_V/sigma_K: Llama-3.1-8B 0.33, Llama-3.2-3B 0.40,
  Qwen3-4B 0.97, Qwen2.5-7B 1.34, Qwen3.5-4B 1.36, Qwen3.5-9B 1.49, GPT-OSS-20B 1.67. "weaker sinks
  co-occur with greater value-vector dispersion relative to key-vector dispersion" (no mechanism
  given). MATH-500 at 25% budget, Qwen3.5-4B: ValueDiff 84.3, KeyDiff 65.2, TOVA 39.3, SnapKV 24.7.
  "On key-dominant non-suppressed controls (σV/σK<0.5; Llama), KeyDiff and KeyNorm lead." Qwen2.5
  has a high ratio without sink suppression and degrades under every method.
- 2505.06708 (Gated Attention), abstract: "applying a head-specific sigmoid gate after the Scaled
  Dot-Product Attention"; "applying query-dependent sparse gating scores to modulate the SDPA
  output"; "this sparse gating mechanism mitigates 'attention sink'".
- 2611.09709 (RidgeFill), abstract and intro: the cache-fill error is in "the keys on Llama-3.1-8B,
  half of it in their scale" and in "the values and the recurrent state on Qwen3.5-9B"; Qwen3-8B
  needs keys and values fixed together. Splice tables (section 6.6) were beyond the 100k-character
  cut and were not read.
- 2502.15075, abstract: "key projections systematically have larger spectral and Frobenius norms
  than value matrices"; "for any given memory budget, prioritizing precision for keys over values
  strictly reduces quantization error". 2410.13212, abstract: "the transformer's output loss is
  more sensitive to the quantization of key matrices". 2502.04420, abstract: "why key cache is
  generally more important than value cache"; Llama-3.1-8B is nearly lossless at 3.25 bits,
  "sensitive models like Qwen2.5-7B-Instruct" need 4.0. 2603.27467, abstract: "K-dominated versus
  V-dominated layers and negative-transfer layers where increased precision degrades quality".

### 2.3 MoE and speculative decoding on hybrids

- 2611.04055 (HybridSpec): verification time "grew by 71% from length 4 to 16, almost entirely in
  GDN and MoE layers, compared with 4%-6% for a dense Qwen3-4B target"; the default SGLang verifier
  "updates the recurrent state at every token and stores a full copy after each update"; latency at
  length 16 "has a correlation of 0.999" with the number of distinct (layer, expert) pairs. Single
  request, A100.
- 2610.38981: routing overlap between consecutive tokens on Qwen3-30B: uniform 0.063, independence
  0.119, empirical 0.431. Qwen3.5-35B is "a notable exception" to the batch-size pattern and is not
  identified as a GDN hybrid. Future work: "understanding which architectural factors determine
  temporal overlap".
- 2610.34440, abstract: "the per-slot kernel has the higher effective bandwidth (r = 0.32 to 0.75).
  Thus deduplication is faster only from 5 to 8 tokens per pass, or never."
- 2610.13778: conventional speculation "on Qwen3.5-122B-A10B at B=512 falls below standard
  decoding"; ReplaySSM+SD is 1.85x and 2.10x over conventional speculation at B=512.
- 2505.16056, abstract: "domain-specialized experts contribute more to routing consistency than
  vocabulary-specialized ones"; "settings like shared experts that decrease expert combination
  space can lead to low local routing consistency".

### 2.4 Diffusion LMs

- 2610.39853: decoding is "confidence-threshold parallel decoding with threshold 0.9", generation
  length 256, block 32; "single-run inference evaluations"; no step counts reported. Total error is
  written as reuse error plus quantization error; the interaction is not measured. "the reused
  cache may already differ from full recomputation before quantization is applied". Full attention
  vs prefix cache vs dual cache, LLaDA: HumanEval 42.07 / 42.07 / 35.98, MBPP 29.80 / 26.60 / 25.80;
  Dream: HumanEval 39.63 / 43.29 / 37.20. Several 4-bit cells match or beat the unquantized cache
  (LLaDA dual GSM8K 78.62 vs 77.94).
- 2505.22618 (Fast-dLLM): with vs without cache, Dream length 256: HumanEval 49.4 -> 53.7, MBPP
  56.6 -> 53.2. Step counts are given by threshold (9, 12, 20 at 0.7, 0.8, 0.9 on one prompt), not
  by cache setting. Evidence that the cache is close is a cosine heatmap described as "consistently
  close to 1".
- 2509.23094 (d2Cache), abstract: its cache "can enable quasi left-to-right generation and mitigate
  premature overconfidence in tokens at the end of the sequence" and "yields consistent
  improvements in generation quality".
- 2606.07571 (BiCache), abstract: AR-style prefix caching on DLMs "causes model accuracy to
  collapse to near zero"; "shared prefix KVs remain stable and reusable in shallow layers".
- 2610.06996 (MaskAhead), abstract: 9.5x KV memory at 1.2-point F1 loss; quantized variant 20.1x at
  2.3 points.

## 3. Tensions considered and dropped

1. What determines expert-routing overlap between consecutive tokens (asked as future work by
   2610.38981). Dropped: 2505.16056 already studies it over 20 models (domain vs vocabulary
   experts, shared experts, load-balance trade-off) and 2604.23150 reports domain-specific
   activation and prefill-decode correlation on 100k traces.
2. Why Qwen3.5-35B is an exception in 2610.38981 while HybridSpec reports GDN+MoE verification
   growth and the user's vLLM numbers show no GDN verify bottleneck. Dropped: the likely answer is
   engine-specific (per-token state copies in SGLang's verifier), which is D3 territory.
3. Does deduplicating experts in a verify pass pay (2610.38981 says overlap helps, 2610.34440 says
   dedup is faster "only from 5 to 8 tokens per pass, or never"). Dropped: kernel and engine
   question, D3 territory, and 2610.34440 already gives the model.
4. Sparse-attention self-draft on hybrids (SparseSpec transfer). Dropped: bounded by the attention
   share of step time; 2611.06133 gets 1.26x at 64k. Predictable.
5. Acceptance limit of product (block-diffusion) drafters. Dropped: derived in 2611.13017 ("no
   product-ranked tree can hold two futures that disagree in m positions without 2^{m-1}-1
   chimeras").
6. Tokens per denoising step vs accepted length as the same saving. Dropped: too vague; S2D2 and
   SimSD already treat them as one thing.
7. Low-rank KV (Palu) vs hybridisation as the same saving. Dropped: Jet-QSparse and 2611.06133
   show the savings stack on hybrids; predictable.
8. Pattern reuse across denoising steps (PillarAttn transfer to dLLMs). Dropped: done by PulseCol.
9. Mask queries as lookahead for eviction in block diffusion. Dropped: done by MaskAhead.
10. Why the built-in MTP head on hybrids beats EAGLE-3 on dense models (own table: acceptance 0.92
    vs 0.40). Dropped: cannot separate co-training from architecture without training a head on a
    matched dense model; more than a week.
11. Speculative rollback of recurrent state. Dropped: HybridSpec, ReplaySSM, and D2.
12. Does the 2610.06750 auxiliary LoRA convert into a larger KV-eviction budget. Folded into card 2
    as a follow-up arm; alone it is a "does gain G survive condition C" question.
13. Staleness error and quantization error "add" in dLLM caches. Folded into card 5; alone the
    noise-floor answer is predictable.
14. Speculative decoding changes the number of state rounding events per committed token. Folded
    into card 1 as the writeback-interval arm; the earlier search already has a step-decode vs
    chunked-scan question.

## 4. Kept cards (ranked)

### Card 1. What does round-to-nearest writeback destroy in a recurrent state: the decay or the writes, and in which heads?

- Tension (contradiction between explanations). Lost in Rounding explains the failure with a decay
  toy model (rounding freezes or distorts slow decay) and finds the behavioural effect in an
  intermediate-decay band, with the slowest octile inert (-0.12) although it holds 49% of
  teacher-forced KL. The dither paper explains RTN failure by lost writes ("discards small
  updates") and finds the gain in long-memory entries. STEPQuant, LoReS and PACE all give their
  extra bits to the slowest-decaying heads. PACE, LeapQuant and ReplaySSM round once per window.
  LeapQuant and SinkState point at a few large directions. No paper separates the decay term from
  the write term, and the only paper that measures behaviour (length, loops) disagrees with the
  KL-based papers on which heads matter. The decay-band account also fails on Qwen3.5-9B.
- Expert guess: RTN absorbs small increments; slow heads accumulate most error; any unbiased or
  deferred rounding fixes it. Confidence low, because the slowest heads were shown to be inert for
  behaviour in the one model family where it was tested.
- A testable consequence nobody checked: the toy model's bands scale with r times the number of
  tokens per writeback, so writing back every 8 to 16 tokens should move the 0.57u-4.1u band out of
  the distorted region and remove the length increase under plain RTN. If the write term is the
  cause, the interval should matter much less than the size of the entry a write lands on.
- Design (my inference, not from a paper): if the decay term is the cause, keep the per-head decay
  as an FP32 scalar outside the stored matrix (the GDN and Mamba-2 decay is a scalar per head), so
  the matrix is only rounded on writes; no random numbers needed. If the write term is the cause,
  split off the large directions (SinkState, Compensator Tokens) and round only the residual. Either
  way it says which heads the 4-bit allocators should protect.
- First experiment. Offline: record FP32 per-token (decay, gate, k, v) streams for Qwen3.5-9B and
  Nemotron-Nano-9B-v2 on 50 long reasoning traces; replay each head with bf16 and FP8 RTN in four
  arms (round everything; exact decay, rounded writes; rounded decay, exact writes; writeback every
  1, 4, 16 tokens); measure signed drift of the state norm and readout, by decay octile and by
  top-8 subspace vs residual. Online: the two arms that differ most, plus FP32, bf16 RTN and bf16
  SR, on 150 competition-math problems; cap hits and length. Stop if neither exact-decay nor
  exact-write removes at least half of what SR removes offline, or if the online length ratio of
  bf16 RTN to FP32 on this setup is inside the seed spread.
- Cost: about 3 GPU-days. Caveat: SR already fixes bf16 for free, so the payoff is at 8 bits and
  below, and in knowing which measure (KL or drift) to allocate bits by.
- Overlap: earlier-search questions on state error in low-energy directions and on state magnitude
  vs bf16; residue (a). Differs by asking which term and which heads, using the Nov 2026 papers.

### Card 2. For the same tokens, are the KV entry and the recurrent state redundant or does each need the other?

- Tension (untested assumption, opposite predictions). GateSight, DeltaS and HOLA all keep in the
  KV cache the tokens that write most strongly to the state. The division-of-labour papers say the
  state carries what attention does not (language 0.70 rec-only vs 0.01 kv-only) and attention
  carries what the state cannot (retrieval 0.00 rec-only), which predicts that strongly written,
  non-exact content could be dropped from the KV cache for free. 2611.01909 says the two are in
  series (state builds the query, attention answers), which predicts no compensation at all. Own
  data: per unit of missing attention the hybrid's next-token KL is 7.4-7.5x lower than the dense
  model's offline, but the advantage did not survive online.
- Expert guess: little compensation on stock models; evicting a KV entry costs about what the
  attention change predicts. Confidence medium. It could be wrong because persona and language
  tokens are read by attention yet delivered by the state, and because a rank-16 LoRA moves
  recurrent-only QA from 5.0% to 22.8%, so the capacity exists.
- First experiment. Qwen3.5-4B and 9B, teacher-forced, 100 reasoning traces and 100 long-prompt QA
  items. For chunks of 64 tokens, four conditions: intact; chunk masked from attention only; chunk's
  writes removed from all GDN states only (write gate zeroed for those tokens); both. Measure
  continuation NLL. The interaction term (both minus sum of singles) is the answer: negative means
  redundant, positive means in series. Split by token type (system prompt, problem statement,
  thinking, tool output). Stop if the interaction is under 10% of the larger single effect for all
  token types.
- Outcomes. Redundant for some class: evict that class first, a hybrid-only budget, and GateSight's
  attention-output utility over-counts. In series: KV eviction on hybrids has no hybrid-specific
  slack; put effort into quantization and low-rank instead, and the auxiliary-loss LoRA becomes the
  only way to buy slack.
- Cost: about 2 GPU-days. Overlap: D1 (dead; this explains its hybrid result rather than proposing
  a selector), earlier Q23, and the sibling card "Where an attention error goes in a hybrid" in
  harvest_own_projects.md (that one tracks decay of one error over time; this one asks whether the
  state holds a second copy).

### Card 3. What determines whether a cache's error budget sits in the keys or the values?

- Tension (contradiction). Three KV-quantization papers state, two of them as theory, that keys
  need more bits than values (2502.15075 "strictly reduces quantization error", 2410.13212,
  2502.04420). On Qwen3.5 the eviction signal is in the values (ValueDiff, sigma_V/sigma_K 1.36 to
  1.49) and the cache-fill error is in "the values and the recurrent state" (RidgeFill), while on
  Llama-3.1-8B both papers find keys. Qwen3-8B needs both. Qwen2.5-7B has a value-heavy ratio
  (1.34) without any sink suppression and is the "sensitive" model in KVTuner. dLLM-KVQ asserts
  keys matter more under cache reuse and gives no number. Candidate causes: the sink (a key-side
  mechanism), key normalisation, the attention output gate (scales the value path by a
  layer-dependent factor), projection norms.
- Expert guess: follows the sink; sigma_V/sigma_K predicts it. Confidence medium. Qwen2.5 and
  Qwen3-4B (sink rate 0.548, ratio 0.97) do not fit that story.
- First experiment. Teacher-forced splice on Llama-3.1-8B, Qwen2.5-7B, Qwen3-4B, Qwen3-8B,
  Qwen3.5-4B, Qwen3.5-9B, LLaDA-8B, Dream-7B: quantize keys only or values only at 8, 4, 2 bits and
  low-rank keys only or values only at matched bytes; measure NLL change per layer. Regress the
  key-to-value damage ratio on sigma_V/sigma_K, sink rate, spectral norm ratio of the projections,
  a key-norm flag, and mean gate. Stop if sigma_V/sigma_K already ranks all models (Spearman 0.9 or
  more): then the answer is published and only a rule is missing.
- Outcomes. One statistic predicts it: an offline rule for splitting bits or rank between K and V
  on any new model, including gate-aware value precision on hybrids. Nothing model-level predicts
  it: allocation must be per layer (TurboAngle's K- and V-dominated layers), and "more for keys" is
  a property of two model families.
- Cost: about 2 GPU-days. Overlap: sibling card 5 in harvest_kv_compression.md (bias vs error size
  for keys and values); this one asks which architecture feature sets the split.

### Card 4. Is the attention output gate the sink's replacement, so that missing attention is harmless when the gate is closed?

- Tension (unexplained anomaly, builds on residue b). In a dense model a head with nothing to
  retrieve puts its mass on the first token; that mass is always kept, so sparse attention loses
  nothing. Gated attention removes the sink (BOS mass 0.408 to 0.034) and replaces it with a
  "head-specific", "query-dependent sparse" gate. If a no-op is now expressed as a closed gate, the
  attention pattern in those steps is free to be diffuse and missing mass there costs nothing. That
  would explain both own results: more missing attention (22.7% vs 11.6%) and 7.4-7.5x less KL per
  unit of it. GateSight uses the gate only as a per-layer average.
- Expert guess: partly yes; gate-weighted missing mass closes most of the gap to the dense model.
  Confidence medium. Against: the offline KL advantage did not help online, the 8-vs-36-layer
  confound may explain the KL ratio alone, and the gate may act as a smooth scale rather than an
  on/off signal.
- First experiment. Qwen3.5-4B, existing teacher-forced traces: per head and step, record gate
  magnitude, attention entropy, oracle@512 missing mass, and the logit effect of the sparse
  approximation. Test whether (a) diffuse steps are closed-gate steps, (b) gate-weighted missing
  mass predicts per-step KL better than raw missing mass, (c) the KL-per-missing-mass ratio against
  Qwen3-4B falls toward 1 after gate weighting and matching the number of perturbed layers. Stop if
  gate magnitude and attention entropy correlate below 0.2 in every layer.
- Outcomes. Yes: coverage targets and selectors on hybrids should be gate-weighted, and gate-closed
  heads can skip the KV read at long context (the gate is known before attention is computed). No:
  the diffuseness is real, and selection on gated models has a floor that dense-model results do
  not show.
- Cost: about 1 GPU-day. Overlap: residue (b) and D1. The gate gives no eviction rule (it is per
  step, not per token), so this does not revive D1; it changes how its result is read.

### Card 5. When an approximate or quantized cache changes a diffusion LM's accuracy, is that representation error or a changed commit schedule?

- Tension (untested assumption). Fast-dLLM and dLLM-KVQ both decode with a confidence threshold, so
  any change to the cache can change which tokens are committed at which step. Neither reports step
  counts per cache setting. Accuracy moves both ways under caching (Dream HumanEval 49.4 to 53.7;
  LLaDA HumanEval 42.07 to 35.98 with dual cache), 4-bit caches sometimes beat the unquantized
  cache, and d2Cache says its cache choice changes commit order and improves quality. dLLM-KVQ
  writes the total error as reuse error plus quantization error without measuring whether they
  add. The parallel with hybrids: an error that looks small per step changes how long generation
  runs.
- Expert guess: mostly small representation noise; step counts move by a few percent; many of the
  table differences are single-run noise (HumanEval has 164 problems). Confidence medium. It could
  be wrong because a stale cache is a biased view (it shows a more masked context), and bias in
  confidence acts on a threshold rule.
- First experiment. LLaDA-8B and Dream-7B with the Fast-dLLM code, GSM8K and a 500-problem MATH
  subset, 3 seeds: full recomputation, prefix cache, dual cache, each with and without 4-bit and
  2-bit cache quantization. Log tokens committed per step, positions, confidences. Then replay each
  cached run under the full-recomputation commit schedule and the reverse. Stop if step counts and
  commit order differ by under 2% and schedule replay changes accuracy by less than the seed spread.
- Outcomes. Schedule: recalibrate the threshold with the cache, report steps next to every cache or
  quantization result, and expect lower bit widths to work once the commit rule is corrected.
  Representation: bits and refresh interval are the levers and should be set together.
- Cost: about 2 GPU-days. Overlap: none with D1-D4; related in spirit to residue (a).

## 5. Searches and coverage

Prior-answer searches (arXiv listing search through WebFetch unless marked):

- "recurrent state rounding hybrid linear attention precision": 2 results, PACE 2610.29545 and
  LeapQuant 2609.38166.
- WebSearch "recurrent state bf16 round-to-nearest bias ... decay stagnation small writes absorbed":
  found 2609.39185 (dither). Nothing separates decay from write or tests GDN reasoning length by
  term.
- "hybrid recurrent state KV cache eviction attention": 3 results (HiState, DeltaS, DASC).
  WebSearch on whether the state compensates for evicted tokens: found HOLA 2607.02303 and DeltaS;
  no measurement of compensation.
- "key value cache quantization keys values sensitivity asymmetric": AsymKV and TurboAngle.
  WebSearch on why keys are more sensitive and sink, QK-norm, gated attention: 2502.15075,
  KVTuner; nothing on sink-free or gated models.
- "mixture-of-experts routing consecutive tokens overlap locality inference": 0 results; "MoE
  expert locality batch": 14 results, 6 relevant. WebSearch found 2505.16056 and others. This
  closed the MoE overlap question (dropped tension 1).
- "diffusion language model KV cache staleness error quantization": 0 results. WebSearch found
  Fast-dLLM, d2Cache, LaCache; none measures steps under approximation or the interaction.
- 5 of 6 WebSearch calls used.

Not done:

- arXiv API keyword search returned HTTP 429 for most of the session; only id_list queries worked.
  Semantic Scholar returned 429. So citation-graph checks for follow-ups were not possible.
- Pages longer than 100,000 characters were read only up to that point: HybridSpec appendix
  (batched results), RidgeFill section 6.6 splice tables and limitations, ValueDiff appendices F-G
  (output length), STEPQuant limitations, ReplaySSM appendix E.
- Not read beyond the abstract: every MTP paper, every block-diffusion drafter paper, LLaDA-Hybrid,
  Fixed State Long Reach, BiCache, MaskAhead, TEAM, MoESD, Nemotron reports. No card rests on them.
- Mercury and Gemini Diffusion: closed models, nothing testable on this hardware; not pursued.
- Jamba, Hymba, Samba, MiniMax-01, MoE-CAP: existence checked, not used in a card.
- The Qwen3.5 gate's exact form (per head or per dimension, where it is applied) was taken from the
  2505.06708 abstract and the repo's own notes, not from the model code.
