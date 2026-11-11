# Harvest: unexplained results in the user's own projects

Date: 2026-11-11. Read-only. No GPU jobs, nothing posted, nothing committed.

Scope: results in the user's own project folders that the project notes do not explain,
turned into research questions and checked against 2025-2026 papers.

How the papers were read: the arXiv API and Semantic Scholar returned 429 from this machine
for the whole session. Searching was done through arxiv.org listing pages (WebFetch) and 6
WebSearch calls (the full budget). Every paper page was read through WebFetch, which passes
the page to a small model that extracts quotes. Quotes below are as that extraction returned
them. "sections" = HTML full text queried for specific sections; "abs" = abstract page only;
"listing" = seen only in a search listing or search snippet (not opened, not used as verified
evidence).

## 1. Own-project results used (source = file path)

kv-fetch (`/home/ryankert_l/kv-fetch/kv-fetch/EXPERIMENTS.md`, `RESULTS.md`, `results/profile/NOTEBOOK.md`, `results/profile/findings.md`)

- E3, Qwen3-4B, AIME 30 problems avg@8, 16K cap: full 48.3%; budget 1K: ours 43.8%, Quest 40.4%,
  oracle shared page set 44.2%, SnapKV-shared 8.3%; budget 512: ours 33.3%, Quest 22.5%, oracle 38.3%.
  "Most of the loss is truncation at 16K (Quest@512 70% vs full 41%); accuracy on finished samples
  is 73-80% for every non-SnapKV run."
- compare_explore: "Even evicting by true future attention misses about as much attention as SnapKV
  (8.5% vs 8.6% at 1K), because evicted pages revive."
- H1: "the oracle shared set at 512 misses 22.7% of attention on Qwen3.5-4B vs 11.6% on Qwen3-4B".
  "The selector ranking flips between the models".
- H-c: "even one GDN layer of lookahead costs +3.4 (Qwen3-4B: +0.2)."
- H-b offline: "Qwen3.5's oracle@512 gives next-token KL 0.011 at 22.9% missing; Qwen3-4B 0.035 at
  14.1%"; "7.4-7.5x lower KL over three pairs with matched missing attention (confounded: 8 perturbed
  layers vs 36)". Online: "prompt-protected SnapKV@1K loses 40.0 points on Qwen3.5 vs 38.8 on
  Qwen3-4B ... The offline KL advantage does not survive long generation." (My note: both runs sit
  near the accuracy floor, so this online pair is weak. The better online pair is H3 vs E3 below.)
- H3, Qwen3.5-4B, 32K cap: full 49.6%; ours@512 30.0%, Quest 19.2%, oracle@512 35.8%; ours@1K 41.25%,
  "oracle@1024 39.2%: ours@1024 is +2.1 points above it [-0.8, +5.0] (minimum missing attention per
  step is not maximum accuracy)"; "43.8% of full-attention samples hit the 32K cap, 63.3% of ours@512
  and 75.4% of quest@512; accuracy among finished samples is 78-92% for every method (full 88%)."
- Checkpoint pilot: fp8 per-row 1.02x noise floor; "States: 99% energy in ~12 singular values, but
  r=32 still 2.8x."
- R2: stale KV after 32 GRPO steps is 1.55-1.74x the kernel-path floor and carries 12-16% of the
  policy change.
- Spec decode table: hybrid MTP k=3 speedup 2.31 / 1.86 / 1.18 at batch 1 / 16 / 64 (acceptance
  0.92 / 0.81 / 0.80); dense EAGLE3 1.85 / 1.28 / 0.69 (acceptance 0.40 / 0.33 / 0.33).

subspec_v2 (`/home/ryankert_l/subspec_v2/reports/2026-04-01_acceptance_length_investigation.md`)

- Cascade variant has "systematically 17-38% lower acceptance than SubSpec at identical
  d/K/temperature configs": d=4 K=3 4.4 vs 4.0; d=8 K=3 7.1 vs 5.9; d=12 K=6 9.3 vs 7.6;
  d=24 K=3 15.6 vs 9.6; d=32 K=6 18.4 vs 12.2. SubSpec's 4-bit draft "uses KV entries written by
  the target model"; the cascade draft "uses KV entries written by itself". The report blames
  kernel-path numerics; the confirming run ("debug_logits: true") was listed as a next step and not
  done. The cause is therefore unverified.
- `2026-03-29_final_comprehensive_report.md`: fused kernel 1.15-1.37x in microbench, -2% end to end
  ("attention is <1% of model forward time").

SparseSpec (`/home/ryankert_l/SparseSpec/`): TriForce chain 447.47 tok/s, spine tree 623.49 tok/s.

gemm-kernel (`/home/ryankert_l/gemm-kernel/README.md`)

- "Dense FMA is a power virus on a 300 W board. An RTX 6000 Ada starts a GEMM at its 2505 MHz boost
  bin, hits the power cap within ~15 ms, and settles near 1100 MHz."
- "a memory-bound kernel holds 2.67 GHz while cuBLAS SGEMM runs at 1.19 GHz."
- Ladder at 4096^3: `20_cublas` bf16 107.55 TFLOP/s at 0.93 GHz, 79.5% SOL; `21_wmma` bf16
  109.39 TFLOP/s at 1.73 GHz, 43.6% SOL.
- "`21_wmma` matches cuBLAS in wall-clock time while being *half* as efficient per clock. It is
  simply a cooler kernel, so the card lets it run 1.8x faster."
- Peaks: "bf16 mma m16n8k16 -> fp32 343.8 [burst] 329.3 [sustained] 2.34 [SM GHz]".

Other folders: Palu has code and a readme only. mambo's parity result is not an inference-speed
result. The adaptfm folder is absent; only a memory note exists (depth-pruned AWQ IFEval 0.20 vs
threshold 0.814).

## 2. Reading list

Sections read (12):

| id | short name | what was taken |
|---|---|---|
| 2512.01278 | SparseSpec | "PillarAttn achieves an average acceptance token length of 6.16 out of 8 tokens"; sparsity 0.05; "attention ... accounts for more than 77% of end-to-end execution time"; does not report attention recall of the draft |
| 2506.04108 | Rectified Sparse Attention (ReSA) | sparse decoding methods "suffer from KV cache misalignment, where approximation errors accumulate and degrade generation quality"; rewrites KV every f=32 tokens, keeps tokens; "ReSA does not rely on output logits to make per-token accept / reject decisions"; math 7B: dense 60.72, sparse 57.72, ReSA 60.52; no experiment separating KV error from wrong tokens |
| 2601.03043 | Lil (Less is Less) | sparse attention methods "consistently increase output length by up to 90% on reasoning-intensive datasets"; "the model repeats content when trying to rebuild context"; information loss never measured directly, no per-step localisation; Guardian early stop |
| 2606.00206 | Quantized Reasoning Models Think They Need to Think Longer | "in up to 52% of the quantized models' failures, models reach the right answer in intermediate reasoning steps but do not output it"; position-level KL vs full-precision entropy Spearman 0.92 (MATH-500, 3-bit AWQ); marker logit penalty cuts CoT 12-23%; no sparse attention or eviction |
| 2610.23818 | Acceptance Is Not Speed | relaxed verification "accepts 1.34x as many tokens per round but produces 1.54x longer completions"; "They do not show why an unlikely token makes a reasoning model think more rather than less."; "the tenth of cases with the largest increase holds 59% of the net extra tokens at the median"; temperature 1.2 also lengthens (1.27x on GSM8K, GPT-OSS-20B) |
| 2602.02711 | Dynamic Mixed-Precision Routing (DMR) | "decoding a step at low precision turns a success into a failure at only about 1% of the steps"; keeping low-precision KV: "decreasing task success by 6.6 percentage points (68.2% to 61.6%)"; W3 decoding after a top-tier prefill: success "increases from 29.6% to 79.2% on ALFWorld"; router AUC 0.738 / 0.627 / 0.745 |
| 2606.03928 | VaSE | at matched budget (GSM8K, Qwen3-4B, K=512) sampling instead of top-k raises SnapKV from 64.3% to 70.9%; resampling the projection each step lifts CurDKV from 78.6% to 87.6%; "We attribute this to improved diversity of the retained KV pairs" (diversity never measured); evicting large-range value states drops GSM8K to 14.3%; "the model tends to enter endless loops of self-reflection and never reaches a conclusion." |
| 2609.17943 | ASPIRE | "the selected sparse context becomes increasingly stale as drafting proceeds"; recall "decreases quickly as token lag" grows; per-request accepted length on LongBench (draft length 20) "from 2.56 to 20.00 with mean 11.51"; no analysis of which tokens are rejected; does not say whether draft-written KV is overwritten |
| 2603.12038 | Slow-Fast Inference | "within-sentence support stability"; dense steps at punctuation, forced after 64 steps; Qwen3-4B-Thinking GPQA 64.14 -> 63.70; retained ratio "approximately 15%"; output length not reported |
| 2502.10424 | QuantSpec | draft reads upper 4 bits of the shared KV, recent 2G tokens in full precision; acceptance ">90%"; sparse-KV drafts degrade faster with speculation length; no ablation of draft-written vs target-written KV |
| 2509.18344 | SubSpec | draft KV "the target model then overwrites during verification"; sharing "enhances alignment by ensuring that both models operate on an identical contextual history"; ablation: tau 23.07 -> 25.14 when the shared KV is added (Qwen2.5 7B, MT-Bench) |
| 2611.01909 | Who Asks, Who Answers | "linear attention constructs the query, while softmax attention performs the lookup"; removing the Address block "reduces ... reader attention from 0.83 to 0.06"; does not measure attention entropy or sinks |

Abstract page read (14):

| id | short name | what was taken |
|---|---|---|
| 2611.12127 | Lost in Rounding | "With bf16 RTN state, five of six models reason longer."; Qwen3.5-9B: "cap hits rise from 118 to 223 and accuracy falls from 85.6% to 70.8%"; "the failure is mostly rounding bias, not missing resolution"; stochastic rounding removes most of it; top teacher-forced-KL octiles do not localise it |
| 2609.04098 | Minima (GDN NVFP4) | the delta-rule recurrence "holds injected noise at a flat plateau over 32K tokens" and "forgets a state impulse within hundreds of steps" |
| 2605.06763 | Louver | "missing even one critical key can lead to sharp error spikes"; recall guarantees as the lever |
| 2508.07101 | LessIsMore | "selection errors accumulate over long generation horizons"; "critical tokens are largely shared across attention heads" |
| 2505.21600 | R2R | "only a small fraction of tokens genuinely diverge reasoning paths between LLMs and SLMs" |
| 2610.31433 | Rate-Controlled KV Re-selection | drift-chosen refresh timing beats random and periodic timing by 2.0-5.0 points (multimodal) |
| 2607.27735 | SparseSpec-L | "extending the speculation horizon can reduce rather than improve speedup" |
| 2610.13202 | S3R | sparse self-speculative RL rollouts, 1.35-1.39x |
| 2610.14754 | GateSight | GDN gate/decay/write signals predict KV utility in hybrids |
| 2605.16928 | RTPurbo | "only a small subset of attention heads truly requires full long-context processing" |
| 2608.21719 | PowerSlider | "compute-bound prefill loses throughput almost linearly with GPU frequency"; decode "sustains it down to 0.57x nominal" |
| 2608.03228 | SAKI-RoPE | "any low-rank index calibrated on a long prefix drops the attention-sink token" |
| 2609.28029 | Tensor decomposition of KV | "The head and layer modes are nearly full-rank"; post-RoPE keys lose 41% to 64% of pre-RoPE compressibility |
| 2608.20359 | SSR | partial-CoT answer as drafter; not about rejections |

Listing or snippet only (not opened; not used as verified evidence): 2605.11999 (Illusion of
Power Capping in LLM Decode), 2501.08219, 2609.29707, 2512.22066, 2610.22877 (BS-Spec),
2610.02800 (BitNest), 2606.18967, 2608.00531 (CURE), 2603.27027 (TAPS), 2512.23765, 2611.05345,
2610.07822, 2608.29748, 2602.07223 (SpecAttn), 2603.01399 (Quasar), 2606.02011, 2611.14246
(SinkState), 2611.06084, 2608.12149, 2608.30310, 2606.07684, 2603.13281, 2602.12029, 2610.19026,
2606.06467, 2608.23843, 2610.02815, 2610.06927, 2607.27269.
Anchors known from earlier project notes, not re-opened today: Palu 2407.21118, ThinKV 2510.01290,
R-KV 2505.24133, SeerAttention-R 2506.08889.

## 3. Tensions considered and dropped

| tension | why dropped |
|---|---|
| MTP engine runs 21.6 requests where the no-MTP engine runs 47 | vLLM block-allocation artifact (extra state blocks per request), known upstream |
| Sparse kernel 1.15-1.37x in microbench, -2% end to end; "attention is <1% of forward time" vs SparseSpec's 77% | Predictable from bytes read: weights dominate at batch 1, KV at large batch x long context |
| Hybrid MTP acceptance 0.80-0.92 vs dense EAGLE3 0.33-0.40 | Different models and head types; joint training is the expected cause (high-confidence prediction) |
| Depth-pruned AWQ IFEval 0.20 | Collapse without healing is expected; no FP16-pruned score to separate causes |
| Tree vs chain drafts in SparseSpec and subspec_v2 | Known from tree-draft literature |
| GDN state: 99% energy in 12 singular values but r=32 is 2.8x noise | D2 was killed; 2611.14246 and 2611.06084 (listing only) store states as a sink plus low-rank factors, which is the likely explanation |
| Stale KV across RL updates carries only 12-16% | D4 killed; not anomalous |
| Page-mean keys vs low-rank keys vs delta coding: same redundancy? | 2609.28029 already separates token-mode and feature-mode structure at reconstruction level; the selection use is D1 |
| Is the correct answer already present before the cap in capped samples? | Answered for quantization by 2606.00206 (52% of failures); for sparse attention Lil's Guardian result points the same way. Kept only as a free first check on stored outputs for cards 1-3 |
| Write/read asymmetry of a quantized copy (subspec_v2 gap, DMR 29.6% -> 79.2%) as its own card | Merged into card 1 as the quantized arm; DMR already shows the effect at agent-step granularity |
| What sets the timescale on which the needed KV set changes (Slow-Fast vs ASPIRE vs SparseSpec k=8) | Close to D1; drift-triggered refresh (2610.31433) and Slow-Fast already address it |
| Per-step KL lower in the hybrid but no online advantage, as a separate card | Folded into cards 2 and 5; the SnapKV online pair is a floor effect |

## 4. Kept cards (ranked)

### Card 1. Token channel or state channel
Question: when sparse (or low-bit) decoding makes a reasoning model run long, does the damage enter
through the few tokens a full pass would have rejected, or through the KV entries written under the
approximation?

- Contradiction. ReSA (2506.04108) rewrites KV every 32 tokens, keeps every token, and reports
  57.72 -> 60.52 (dense 60.72); it blames "KV cache misalignment" but never separates KV error from
  wrong tokens. SparseSpec (2512.01278) and ASPIRE fix tokens (and KV) and get 6.16 of 8 accepted
  at a 5% budget. Lil (2601.03043) blames information loss then reconstruction. 2606.00206 puts
  quantization damage at token level (markers at high-entropy positions). DMR (2602.02711) finds
  keeping low-precision KV costs 6.6 points. Own: kv-fetch loses 15-20 points at budget 512 with no
  repair of either kind, almost all by not finishing; subspec_v2 loses 17-38% of acceptance length
  when the 4-bit draft reads its own KV (cause unverified).
- Expert guess: both, with KV rewrite recovering over half at mild sparsity and less at 4%
  budgets. Confidence low.
- Search: listings "speculative decoding rejected tokens reasoning", "self-speculative decoding
  draft"; WebSearch on rejected positions in sparse drafts. Closest: ReSA Appendix B (speed only),
  DMR Appendix F (agents, quantization).
- First experiment: Qwen3-4B, AIME 30x8, budget 512, 2x2: tokens {sparse, corrected to the
  full-attention choice under shared sampling noise} x KV {sparse-written, rewritten densely every
  32 tokens}. Add a 4-bit-weights pair. Measure cap hits, length, accuracy. About 5 GPU-days.

### Card 2. Persistent error or per-step error size
Question: is reasoning-length inflation caused by error that repeats in the same direction over many
steps rather than by its per-step size, so that randomising the error at equal size removes it?

- Evidence: Lost in Rounding (2611.12127): "the failure is mostly rounding bias, not missing
  resolution". VaSE (2606.03928): sampling instead of top-k at the same budget, 64.3% -> 70.9%.
  Own: oracle@1024 39.2% vs ours@1024 41.25%; 3x lower per-step KL in the hybrid at budget 512 with
  no smaller online loss. Against: 2610.23818, temperature 1.2 (unbiased noise) lengthens 1.27x;
  Louver says recall is the lever.
- Expert guess: per-step size governs; dithering lowers recall and does not help. Confidence low.
- Search: WebSearch "length inflation ... stochastic OR bias ... per-step KL" found no paper that
  separates the two. Closest: 2611.12127 (recurrent state only), VaSE (eviction only, diversity not
  measured).
- First experiment: Qwen3-4B AIME, budget 512: deterministic selector vs dithered selection with
  budget raised until teacher-forced missing attention matches; int4 KV with round-to-nearest vs
  stochastic rounding. About 5 GPU-days.

### Card 3. Do sparse-attention errors and quantization errors fall at the same positions
- Evidence: 2606.00206: quantization KL tracks full-precision entropy, Spearman 0.92. Louver:
  "missing even one critical key can lead to sharp error spikes". QuantSpec: sparse-KV drafts
  degrade faster with speculation length than quantized ones. No listed paper stacks sparse
  attention with low-bit weights in one draft or measures rejection overlap.
- Expert guess: large overlap at high-entropy positions. Confidence medium. Doubt: retrieval misses
  need not be high-entropy; Lost in Rounding found top-KL octiles do not localise its effect.
- First experiment: teacher-forced along full-model traces, Qwen3-8B, AIME and LiveCodeBench:
  per-position mismatch under 5% sparse attention, 4-bit weights, both. About 2 GPU-days.

### Card 4. Where an attention error goes in a hybrid
- Evidence: own H-b (KL 0.011 at 22.9% missing vs 0.035 at 14.1%; no online advantage), Minima
  (2609.04098) "forgets a state impulse within hundreds of steps", Lost in Rounding (state bias
  lengthens reasoning in five of six hybrids), 2611.01909 (recurrent layers build the query).
- Expert guess: absorbed within hundreds of steps. Confidence medium.
- First experiment: single-step attention knock-out, teacher-forced KL decay over 2K tokens,
  Qwen3.5-4B vs Qwen3-4B. About 1 GPU-day.

### Card 5. What sets joules per FLOP of a GEMM on a power-capped card
- Evidence: gemm-kernel README numbers above. cuBLAS bf16 107.55 TFLOP/s at 0.93 GHz; the
  register-only mma loop sustains 329.3 at 2.34 GHz; a kernel at 43.6% per-clock efficiency ties
  cuBLAS in wall-clock.
- Expert guess: data movement is most of the energy, but cuBLAS is near the minimum, so little
  headroom. Confidence low.
- Closest: PowerSlider 2608.21719 (serving level), 2605.11999 (listing only).
- First experiment: kernel sweep with board power and SM clock logged. Under 1 GPU-day.
- Close to killed direction D3.
