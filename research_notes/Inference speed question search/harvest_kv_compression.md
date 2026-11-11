# Harvest: research questions in KV-cache compression

Date: 2026-11-11. Read-only literature work. No GPU jobs were run, nothing was posted.

How to read this file: section 1 is the reading list, section 2 the extracted claims
(quoted from the source), section 3 the tensions that were considered and dropped, section 4
the five kept question cards, section 5 the searches and what was not covered.

"Verified" means the sentence or number was read in the paper's arXiv HTML or abstract page in
this session. Everything else is marked "not verified".

## 1. Reading list

Body sections read (arXiv HTML), 24 papers:

| arXiv id | Short name | What it is |
|---|---|---|
| 2611.07025 | LowRAM | Low-rank keys, values predicted from keys plus sparse residuals |
| 2410.21465 | ShadowKV | Per-context low-rank pre-RoPE keys, values offloaded |
| 2407.21118 | Palu | Low-rank projection of K and V, Fisher rank search |
| 2503.18893 | xKV | Cross-layer shared SVD basis |
| 2410.13212 | AsymKV | Layer-wise asymmetric K/V bit widths |
| 2502.04420 | KVTuner | Layer-wise mixed-precision K/V search, sensitivity analysis |
| 2511.01815 | KVTC | PCA transform coding with DP bit allocation |
| 2407.11550 | Ada-KV | Head-wise adaptive eviction budget |
| 2410.19258 | HeadKV | Head budget from offline retrieval-reasoning score |
| 2610.18026 | ChampionKV | Head-budget eviction, "abnormal heads" |
| 2610.23939 | PeekKV | Head budget from one decode token |
| 2412.12706 | More Tokens, Lower Precision | Token-count vs precision trade-off |
| 2610.05685 | BreadthKV (Spend Bytes on Breadth) | Precision-count trade-off on reasoning traces |
| 2611.07256 | TriAttention | Skip / 4-bit / 8-bit tiles per layer and head |
| 2611.07605 | The Gating Mechanism of KV Cache Eviction | Centroids inside the softmax for evicted mass |
| 2610.35020 | Attention under Bounded Key-Query Similarity | Theory: uniform sampling near-optimal |
| 2410.18517 | KVSharer | Layer-wise KV sharing between dissimilar layers |
| 2405.14366 | MiniCache | Cross-layer merging in deep layers |
| 2607.12550 | JoLT | Joint Tucker-rank and residual-bit allocation |
| 2609.24298 | KV-COBRA | Per-head co-optimised bit and rank |
| 2611.03992 | FlagRot | Selective rotation for 2-bit caches, 15 models incl. one hybrid |
| 2605.02905 | eOptShrinkQ | Spiked-matrix model: shared low-rank part plus per-token residual |
| 2610.12133 | Attic-KV | Query-agnostic eviction by rehearsing generated QA |
| 2506.06266 | Cartridges | Trained KV prefix by self-study |

Partly read (some sections or tables): 2604.11501 v2 (Attention-Weighted Value Projection; v1 was
"Quantization Dominates Rank Reduction for KV-Cache Compression"), 2502.01941 (KVFundaBench),
2505.23416 (KVzip), 2610.36950 (DCKV, hybrid Qwen3.5-2B), 2610.18642 (MultiAlloc),
2512.05916 (KQ-SVD), 2608.02691 (OptR), 2610.03027 (TaSQ), 2609.35800 (HeadGuard).

Abstract only (existence checked on arXiv): 2605.03562 HeadQ (HTML returned 404), 2608.04074 NOVA-KV
("Spend Bits Where Queries Look"), 2605.06675 RateQuant, 2605.08317 RDKV, 2606.08382 STAR-KV,
2603.04427 Thin Keys Full Values, 2502.13063 Cramming 1568 Tokens, 2508.17032 Learned Structure in
Cartridges, 2603.27819 KVSculpt, 2607.08032 (rate-distortion view of memory compaction),
2611.12127 Lost in Rounding, 2611.01864 (quantizing recurrent LMs), 2607.28699 WitCert,
2605.18053 (Protection Is Nearly All You Need), 2510.00231 Pitfalls of KV Cache Compression,
2407.01527 (KV Cache Compression, But What Must We Give in Return), 2610.20686 RoLaQ, 2610.33625
RAQ-MLA, 2610.17202 MLAfication, 2611.12062 (KV translation between hybrids), 2610.11214 (KV
quantization and linear attention), 2610.31098 RIFT, 2610.15384 (E8 lattice), 2608.05326 QEvict,
2608.02901 AnchorKV, 2602.08005 ResidualKV, 2610.17970 AlignPred, 2610.35023 BellowsKV.

Corrections to ids: 2412.19442 is "A Survey on Large Language Model Acceleration based on KV Cache
Management", not the "what must we give in return" benchmark (that one is 2407.01527).

Not verified (recalled, not opened): Haris and Onak "Compression Barriers", Slim Attention, QuantSpec,
LESS, PM-KVQ, ARKV 2603.08727 (seen only in a search summary).

Model configs read from Hugging Face config.json (hidden / heads / KV heads / layers):
Qwen2.5-7B-Instruct 3584/28/4/28, Qwen2.5-3B-Instruct 2048/16/2/36, Qwen2.5-14B-Instruct 5120/40/8/48,
Qwen2.5-32B-Instruct 5120/40/8/64, Mistral-7B-Instruct-v0.3 4096/32/8/32, Qwen3-8B 4096/32/8/36.

## 2. Extracted claims (verified unless marked)

Keys versus values, rank versus bits
- ShadowKV 2410.21465: "pre-Rotary Position Embedding (RoPE) keys are exceptionally low-rank compared to the layer inputs, post-RoPE keys, values, key weight matrix, and value weight matrix"; "Pre-RoPE keys share low-rank subspaces within sequences but differ across sequences"; "We set the chunk size to 8, the rank to 160"; "accuracy increases with the rank up to approximately 160, after which it stabilizes near full-rank performance".
- Palu 2407.21118: "we observe that the value is generally allocated a higher rank than the key". Table 1, 50% compression, G-LRD: Llama-2-7B Wiki2 ppl 5.47 to 6.01; Llama-3-8B-Inst 8.28 to 10.11.
- xKV 2503.18893: "We set the rank for key and value to be 1:1.5 if value compression is applied."
- JoLT 2607.12550: "values are about 2.4x harder to compress than keys because their feature spectrum is flat"; "The surrogate of Equation (2) charges every unit of discarded Frobenius mass equally, whether it remains as structured truncation error or as quantization noise, and it does not model the asymmetric effect of key errors through the softmax. In the free zone this approximation is accurate, but at higher compression on LLaMA-2 it makes extra key rank appear cheaper than key residual bits of the same byte cost."; "LLaMA-2 jumps sharply to +27.28%" at 4x.
- KV-COBRA 2609.24298: "each head has a different optimal mix of rank truncation and quantization"; "Replacing our damped proportional update with an exact DP solver for the sum-distortion minimization reduces sum D* by ~12-16% but actually degrades PPL by ~0.5 on average. The surrogate itself is the proxy at fault, not the solver"; "V-heavy splits dominate the joint K+V sweep (Tables 4, 5), reflecting V's flatter spectrum".
- 2604.11501 v1 abstract: "At matched storage budgets across five models (124M-14B, MHA and GQA), we find that quantization consistently outperforms rank reduction by 4-364 PPL"; "INT4 matches FP16 accuracy ... while rank-32 at identical storage collapses to 0.4%"; "projection damage exceeds quantization damage by 3 x 2^(2b) per direction under the softmax Fisher metric". v2 (the current version) withdraws the comparison: "Historical weight-perturbation experiments favor coefficient rounding over the value projections tested, but do not establish a comparison at equal cache storage."
- AsymKV 2410.13212: "the loss derived from the key matrix's quantization will be magnified relative to that of the value matrix, owing to the multiplication of the query and activation function applied specifically to the key matrix".
- KVTuner 2502.04420, Table 2 (word perplexity, HF KIVI-HQQ), KV8 / K8V2 / K2V8: Llama3-8B-Instruct 9.95 / 10.04 / 31.92; Qwen2.5-7B-Instruct 9.56 / 9.45 / 1866.33 (and 220.83 at K4V8); Qwen2.5-3B 10.60 / 11.36 / 147.03; Llama2-7B-chat 11.60 / 11.67 / 13.86. "The 5-bit K8V2 precision pair achieves performance equal to or better than the higher 6-bit K4V8 precision pair".
- KVTC 2511.01815, App. B.4: "for long-context retrieval tasks, the value cache could be compressed more than the key cache ... On the other hand, stronger compression of values shows a noticeable degradation on the GSM8K and MMLU tasks."; "the DP assigns zero bits to a substantial number of trailing principal components"; "we report Frobenius-norm reconstruction error as a proxy for downstream task accuracy. While convenient, this metric does not guarantee task-level gains".
- eOptShrinkQ 2605.02905: "the key-value (KV) cache in transformer attention heads admits a natural decomposition into a low-rank shared context component and a full-rank per-token residual"; "Even a simple rank-1 SVD at 2.04 bits matches standalone TQprod at 3.00 bits"; "the Lloyd-Max quantizer systematically shrinks each coordinate toward zero".
- FlagRot 2611.03992: "On Qwen2.5-7B, the key-projection biases of the first and last layers add a constant offset of several hundred to a few low-frequency channels, which then sets the scale of every group that contains them; the uncentered per-token 4-bit caches reach a KL of thousands"; centering "changes all logits of a query by the same amount, so it leaves the unquantized softmax weights unchanged". Table 13 (KL x 10^3, INT2, per-token RTN uncentered / centered RTN / KIVI per-channel 2.5 bits): Llama-3.1-8B 115 / 39.8 / 18.5; Qwen3-8B 83.9 / 54.9 / 29.8; Llama-2-7B 37.3 / 32.1 / 13.5; GLM-4-9B 275 / 25.8 / 15.2; Qwen3.8-27B (hybrid, KV cache in 16 of 64 layers) 27.0 / 22.5 / 19.6; geometric mean 75.4 / 43.4 / 18.7. Table 10 KV heads: Llama-2-7B 32, GLM-4-9B 2, Qwen2.5-7B 4, Llama-3.1-8B and Qwen3-8B 8. "The hybrid Qwen3.8-27B keeps a KV cache in only 16 of its 64 layers, and there KIVI-only already matches BF16".
- HeadQ 2605.03562 (abstract): "For keys, the visible object is score error modulo constant shifts"; "Fisher/score-space error predicts attention KL far better than raw key MSE; same-budget counterexamples, null-space interventions, query-PCA controls, and wrong-sign HeadQ falsify storage-MSE alternatives. Matched Pythia checkpoints localize the main anomaly to a small-model low-entropy route-flip boundary."
- LowRAM 2611.07025: "within each attention head, values are largely predictable from their corresponding keys"; "giving 4x near-lossless value compression"; "Removing residual corrections drops recovered performance from 99.0% to 64.9% (34.1 points)"; "Random selection scores less than 40%, with or without sinks, whereas MSE-ranked selection recovers 98%. Forcing residuals for attention sinks reduces attention-output error for both random and MSE-ranked selection but does not positively affect accuracy." (ablation on RULER NIAH-Multikey-1, 4K tokens, Llama-3.1-8B-Instruct); "Although QJL debiases the inner product estimations, we found that its variance is amplified by Softmax, thus degrading both attention quality and downstream accuracy."; "It is also unclear how much of the learned parametrisation persists across contexts ... and whether hard-to-predict tokens are stable across contexts".
- Lost in Rounding 2611.12127 (abstract): "Same-width bf16 SR removes the length increase in four of five affected models ... so the failure is mostly rounding bias, not missing resolution."

Heads
- KVTuner: "Lemma 1. Only attention heads with sparse and concentrated patterns demonstrate consistent robustness to low-precision KV cache quantization."; "Highly sparse streaming heads are generally more robust to KV cache quantization than retrieval heads"; "the sensitivity of LLMs and intermediate layers to KV cache quantization is the model property and independent of input prompts".
- Ada-KV 2407.11550: "most heads retain nearly all attention weights with a small cache (e.g., top 5%), while dispersed heads require larger cache proportions."
- ChampionKV 2610.18026: "assigning larger budgets to heads with dispersed attention and smaller budgets to concentrated heads".
- TriAttention 2611.07256: "useful interactions differ in their precision requirements: some tolerate low-precision scores while remaining sensitive to removal". It ranks sensitivity separately for 4-bit and for skipping but reports no correlation between the two.
- More Tokens, Lower Precision 2412.12706: "head-level KV allocation methods such as Ada-KV and HeadKV ... perform well under high precision and a low number of preserved tokens. However, when precision is reduced to as low as 2-bit, these methods experience a more pronounced performance degradation compared to KV pruning methods compatible with KIVI, such as SnapKV and PyramidKV." (confound: a different quantizer was used for those two methods).
- PeekKV 2610.23939: "within a single input, the heads playing the leading roles shift between the prefill and decode stages"; "the signal at the last prefill token (prefill@last) attains an average rank correlation of only 0.58 with the reference ranking, whereas the signal at the first decode token (decode@1) ... reaches 0.77".
- BreadthKV 2610.05685: "keeping more tokens at low precision beats eviction alone in 17 of 18 settings"; "The best bit-width depends on the model and budget, and the offline attention error we measured does not predict it reliably"; "a joint rate-distortion allocator (RDKV) is 1.0 point less accurate".
- RateQuant 2605.06675 (abstract): "all current quantizers assign the same bit-width to every attention head, ignoring the large variation in head importance".

Token selection, query-agnostic compression, trained caches
- Attic-KV 2610.12133: "At a 3% keep ratio, rereading the whole context keeps 31.5 of 96.5 points on RULER"; "Whole-document rehearsal keeps 5.7-7.1% on seven of the eight tasks: the budget lands on the answer no more often than on any other token"; "in natural text nearly every token is unpredictable, and a full reread spreads the budget evenly. The answer-aware oracle keeps 48.4% on average."; anchors are "the keys that stand out from the chunk's average key".
- KVzip 2505.23416: scores KV pairs by attention received while the model reconstructs the context; "consistently maintains inference accuracy even when evicting up to 70% of the KV cache".
- Cartridges 2506.06266: "match ICL performance while using 38.6x less memory"; next-token training "yields Cartridges that memorize the corpus perfectly using 107x less memory than the KV-cache. However, the resulting Cartridges are not general - they degrade the LM's ability to respond to diverse questions beyond regurgitating the corpus".
- Cramming 2502.13063 (abstract): "the maximum attainable lossless compression ratio is typically not higher than x10"; "vectors with compression ratios up to x1500 exist"; "the compression limits are determined not by the length of the input but by the amount of uncertainty to be reduced".
- 2508.17032 (abstract): "Cartridge keys act as stable, shareable retrieval routers for the compressed corpora and ... most of the learned compression occurs within the Cartridge value vectors".
- Gating Mechanism 2611.07605: "what eviction unavoidably discards ... is recovered by parameter-free centroids read inside the same softmax far better than by any memory read outside it"; "We find that discarded entries still receive substantial attention mass".
- 2610.35020: "we show that the simple uniform sampling algorithm is nearly optimal for a typical range of parameters"; "in practice, the key-query cosine similarity is small in absolute value once the attention sink is excluded".

Cross-layer
- KVSharer 2410.18517: "when the KV caches of two layers differ significantly, sharing one layer's KV cache with another during inference does not lead to significant performance degradation"; authors leave it unexplained.
- xKV: "adjacent layers exhibit surprisingly low token-level similarity"; "their dominant singular vectors remain highly aligned".

## 3. Tensions considered and dropped

1. GQA sharing ratio sets a floor on key bits (was a kept candidate, now dropped). Trigger: KVTuner Table 2, where the K2V8 perplexity blow-up orders by query-heads-per-KV-head (MHA Llama-2 about 1.2x, ratio 4-5 about 1.4-3.2x, Qwen2.5-3B ratio 8 13.9x, Qwen2.5-7B ratio 7 195x). FlagRot (2611.03992, 2 Nov 2026) explains the outlier: Qwen2.5-7B's key-projection biases put a constant offset of several hundred on a few channels, and subtracting the key mean fixes it. In FlagRot's Table 13 the model with the fewest KV heads (GLM-4-9B, 2 KV heads) goes from KL 275 to 25.8 with centering and is below the cohort mean with per-channel scales (15.2 vs 18.7), while MHA Llama-2-7B sits at 32.1 and 13.5. So after the offset is removed there is no visible relation between KV-head count and low-bit damage. The anomaly was an artifact. What is left (a regression over FlagRot's 12 published per-model numbers) is a re-analysis, not a research question.
2. Is the attention KV of a GDN hybrid more or less compressible per entry than a dense model's. Suggestive numbers exist: FlagRot's hybrid Qwen3.8-27B has a KV cache in 16 of 64 layers yet its best INT2 KL (14.2) is about the cohort mean (12.4), i.e. roughly 2x the damage per attention layer; at task level "KIVI-only already matches BF16"; DCKV reports ">55 times fewer bytes than bf16 at a loss of at most 1.8 points" on Qwen3.5-2B with no dense comparison. Dropped as a stand-alone card because it sits on top of the earlier Q23 residue and D2, and at least six recent preprints work on eviction in these hybrids (scoop risk noted in the earlier verification). The hybrid is used as one arm in cards 1, 2 and 5 instead.
3. KVSharer: sharing between dissimilar layers works better than between similar ones. A strong researcher would guess this is attention-layer dropping in disguise (the layers whose KV can be overwritten are layers the model does not need). Medium-high confidence, so dropped.
4. Cross-layer redundancy (MiniCache vs xKV vs KVSharer disagreement). Predictable: one residual stream seen through different projections; subspace alignment without token-level similarity is what that predicts. AlignPred and BellowsKV already build on it.
5. Token-count vs precision trade-off. Saturated by BreadthKV, RDKV, TriAttention, MultiAlloc, QEvict, ARKV.
6. A summary slot or centroid for evicted attention mass. Done by 2611.07605 (centroids inside the softmax) and earlier merging work.
7. Transform coding with reverse water-filling for the cache. Done by KVTC, DCKV, NOVA-KV, RateQuant.
8. Static vs dynamic head importance (KVTuner "model property" vs PeekKV prefill/decode shift). A strong researcher would predict quantization sensitivity is static and eviction importance is dynamic. Kept only as a sub-measurement of card 2.
9. Why random eviction is so strong on reasoning traces vs theory (2610.35020). Overlaps D1 and the earlier search (Random Attention); stochastic sparse attention already exists (2605.01910, abstract only).
10. Which local error metric predicts downstream accuracy (LowRAM Fig. 3, KVTC, BreadthKV, KV-COBRA all say MSE-type errors do not). Overlaps the earlier residue (a) (one offline predictor of reasoning-length inflation); HeadQ and NOVA-KV already give attention-aware metrics for the first link of the chain.
11. Layer-budget shape (PyramidKV vs others), arithmetic-task fragility (KVFundaBench), eviction beating the full cache: task-dependent and predictable, or covered by earlier Q08.
12. Decode staleness (RIFT), decode-time head shift as a selection signal (PeekKV): these are D1 territory.

## 4. Kept cards

### Card 1. Is the low-rank part of a context's key cache the part attention cannot see?

Question: Is the low-rank component of a context's keys mostly invisible to attention (common to all tokens, or in directions the queries do not use), so that "keys are low-rank" and "keys need many bits" describe two different components?

Type: contradiction.

Evidence: ShadowKV keeps rank 160 of 1024 per context with near full accuracy; 2604.11501 v1 claims rank reduction loses to quantization by "4-364 PPL" at matched storage and that "projection damage exceeds quantization damage by 3 x 2^(2b) per direction", then v2 withdraws the equal-storage comparison; JoLT's allocator fails on LLaMA-2 because "it makes extra key rank appear cheaper than key residual bits of the same byte cost"; KV-COBRA's exact solver lowers summed distortion 12-16% and makes perplexity worse; KVTuner shows keys need 4+ bits (K2V8 31.92 vs K8V2 10.04 on Llama-3-8B); FlagRot shows that removing only the key mean, which softmax cannot see, cuts INT2 KL from 115 to 39.8 on Llama-3.1-8B and from 275 to 25.8 on GLM-4-9B; eOptShrinkQ says the cache is "a low-rank shared context component and a full-rank per-token residual" and that "a simple rank-1 SVD at 2.04 bits matches standalone TQprod at 3.00 bits".

Expert prediction (medium): only the mean is invisible. The rest of the low-rank part is routing signal (token type, position) that queries read, and bit fragility is softmax amplification of the residual.

Why it could be wrong: a rank-1 mean already accounts for a 3-10x change in low-bit damage and about one bit, so common-mode structure may explain much more of both effects than assumed. ShadowKV's subspace changes from sequence to sequence, which is what a context-specific common component would look like. Nobody found has measured query energy in the per-context low-rank subspace against the residual. HeadQ ran "null-space interventions" but its body could not be opened.

Outcome A (mostly invisible): low-rank key methods mostly save bytes on something that need not be stored. Build: drop the common-mode and query-null component analytically at write time (no SVD, no stored basis, no reconstruction), then quantize the near-isotropic residual with a plain low-bit quantizer. It also sets a bound: no further rank gain on keys.

Outcome B (visible): the low-rank part is real signal; rank and bits are commensurable in a query-weighted metric. Build: fix the allocators (JoLT, KV-COBRA) by measuring both errors in score space, and treat per-context key rank as the main knob.

First experiment: Llama-3.1-8B, Qwen3-8B, Qwen3.5-9B (attention layers). 50 contexts of 4K-16K tokens (LongBench, RULER, reasoning traces), teacher-forced, hooks on pre- and post-RoPE keys. Per head and context split K into token mean, top-r per-context subspace (r from 8 to 160) and residual. Measure (i) the share of real query energy in each part, (ii) attention KL and next-token KL when each part is zeroed, quantized to 2 or 4 bits, or replaced by noise of equal norm, (iii) the same with a fixed calibration basis. Stop if zeroing the low-rank part beyond the mean costs as much KL per unit energy as zeroing the residual. About 2 GPU-days.

Prior-answer search: arXiv API queries on "KV cache" with low-rank + quantization, with "stochastic rounding / unbiased / coherent error", with GQA + sensitivity; one web search on truncation error vs quantization noise. Closest: HeadQ 2605.03562 (score error modulo constant shifts, abstract only), FlagRot 2611.03992 (centering), eOptShrinkQ 2605.02905 (spiked model), NOVA-KV 2608.04074 (attention-aware transform, abstract only). None reports how much of the low-rank key energy the queries read. Scoop risk is high; this sub-area gets several papers a month.

Overlap with dropped directions: none with D1-D4.

### Card 2. Are the heads that tolerate low-bit KV the heads that tolerate eviction?

Question: Across attention heads, is sensitivity to key and value quantization the same property as sensitivity to token eviction, or the opposite one?

Type: contradiction.

Evidence: KVTuner Lemma 1 says only sparse, concentrated heads are robust to low-bit KV; Ada-KV and ChampionKV say dispersed heads are the ones that need a large eviction budget; so by these two, diffuse heads are fragile to both. TriAttention says some interactions "tolerate low-precision scores while remaining sensitive to removal". HeadQ localises its key-quantization anomaly to a "low-entropy route-flip boundary", i.e. sparse heads. 2412.12706 reports head-level eviction allocators degrade more at 2 bits. BreadthKV finds low precision plus more tokens wins in 17 of 18 settings but that offline attention error does not predict the best bit width and a joint rate-distortion allocator is 1.0 point worse than a simple rule.

Expert prediction (low to medium): complementary. Diffuse heads average out value noise but lose mass under eviction; sparse heads survive eviction but a key error can flip their route.

Why it could be wrong: KVTuner's published lemma says the opposite. Key and value sensitivity may have opposite signs with respect to head entropy, and KVTuner measures relative attention-output error, not downstream effect.

Outcome A (same heads): the two savings overlap; the diffuse heads bound both. One sensitivity score serves both allocators, and head-level eviction plus low bits should not be stacked.

Outcome B (opposite heads): choose the compression family per head: evict in sparse heads, keep every token at 1-2 bits in diffuse heads. Gains multiply. Needs a per-head-group page layout in the serving kernel.

First experiment: Llama-3.1-8B and Qwen3-8B, then Qwen3.5-9B. 50 prompts (RULER 8K, LongBench subset, teacher-forced reasoning traces). For every KV head apply four single-head interventions: keys to 2 bits, values to 2 bits, keep the top 10% of tokens by that head's attention, keep a random 10%. Record next-token KL. Report Spearman correlation across heads between eviction KL and each quantization KL, and regress each on head entropy. Check stability across prompts and between prefill and decode positions. Stop if a published table already has this, or if correlations are within +-0.2 on all models (then head-level structure is not worth a design). About 3 GPU-days.

Prior-answer search: arXiv API ("KV cache" + quantization + eviction; + heads), one web search on per-head quantization vs eviction sensitivity. Closest: TriAttention 2611.07256 (two sensitivity rankings, no comparison), RateQuant 2605.06675 (per-head bit allocation only), ARKV 2603.08727 (layer-level tiers, not verified).

Overlap: uses head entropy like the earlier Q23 residue, but asks whether two sensitivities coincide, not whether hybrids are more diffuse. Not D1.

### Card 3. Is "value not predictable from key" an importance signal?

Question: Are the tokens whose values cannot be predicted from their keys the tokens that later queries read, or only the tokens with the largest reconstruction error?

Type: unexplained anomaly.

Evidence: LowRAM: random residual selection "scores less than 40%" while MSE-ranked selection "recovers 98%" (NIAH-Multikey-1 only); removing residuals drops recovered performance "from 99.0% to 64.9%"; forcing residuals on sinks lowers attention-output error but not accuracy; the authors do not know "whether hard-to-predict tokens are stable across contexts". Attic-KV: "in natural text nearly every token is unpredictable, and a full reread spreads the budget evenly"; whole-context rehearsal keeps 31.5 of 96.5 RULER points at 3%; its own anchors are keys that stand out from the chunk mean. 2508.17032: in trained caches keys act as routers and the compression sits in the values.

Expert prediction (medium): mostly a benchmark artifact. In NIAH the needle is an outlier for a predictor fitted on the haystack; on natural text unpredictability will track what gets asked only weakly.

Why it could be wrong: if keys are addresses and values are content, a value that the key does not imply is the information the token adds, which is close to what a fact is. Attic's key-outlier rule works on natural text. LowRAM's 34-point drop is an average over its benchmark suite, not only NIAH.

Outcome A (importance signal): a query-agnostic eviction score from one ridge regression per head, with no reread (KVzip) and no generated QA (Attic).

Outcome B (independent): predictability and importance are two axes, so the savings multiply: evict unimportant tokens, store keys only for important and predictable ones, store full entries for the rest. It also tells LowRAM users that the NIAH number does not transfer.

First experiment: Llama-3.1-8B-Instruct. Contexts from SQuAD multi-question (KVzip's setting), LongBench QA and RULER NIAH 4K. Per head, closed-form ridge map from keys to values fitted per context. Per-token residual norm compared with (a) oracle attention from held-out question and answer tokens, (b) the KVzip score, (c) key distance from the chunk mean, (d) value norm: Spearman and overlap at 10%. Then evict to 10-30% by residual and compare with KVzip and SnapKV on multi-query QA. Stop if overlap with the oracle on natural text is within 1.2x of chance. About 1.5 GPU-days.

Prior-answer search: arXiv API ("KV cache" + "from keys" / "key-to-value" / "values from keys") returned only LowRAM. Closest: LowRAM's own selection ablation (one synthetic task), Attic-KV.

Overlap: prefill-time, query-agnostic scoring; not D1 (decode-time page selection).

### Card 4. What bounds the usable content of a trained KV slot?

Question: What limits how much question-answerable context one trained KV slot can hold: the capacity of the slot, or the coverage of the queries it was trained on?

Type: missing limit.

Evidence: Cramming: one input vector can hold up to 1568 tokens (x1500), while practical encoders stay "not higher than x10". Cartridges: next-token training memorises the corpus at "107x less memory" but the result is "not general"; self-study reaches 38.6x. Attic-KV: reconstruction-style rehearsal keeps 31.5 of 96.5 points at 3% ("Rehearse everything, remember nothing"). So decodable density is 3 to 40 times higher than usable density and nobody has said which one is the limit. 2508.17032 says keys act as routers and values carry the content.

Expert prediction (low to medium): coverage, not capacity. With enough varied training queries, usable density approaches decodable density.

Why it could be wrong: decoding verbatim is sequential and driven by the growing suffix; answering a question is random access. A slot is read through one key per head per layer, so the number of separately addressable items per slot may be small whatever the training set.

Outcome A (capacity): a ceiling of some number of facts per slot holds for every method that reduces token count (eviction, merging, Cartridges, KVSculpt). Budget slots by fact count, stop chasing more than about 40x in token count, and get the rest from bytes per slot.

Outcome B (coverage): densities far above 40x are reachable; the work moves to cheap compaction (amortised or closed-form, as in KVSculpt) and query synthesis.

First experiment: synthetic random-access task. N random name-to-code facts; train an m-slot KV prefix on a frozen Qwen3-4B or Llama-3.1-8B with context distillation on queries that cover every fact (so coverage is complete by construction), and separately with next-token reconstruction. Sweep N/m from 1 to 64. Measure exact-match recall on held-out phrasings and verbatim reconstruction accuracy. Also train values only with keys frozen. Either curve shape answers the question; stop early if a paper with this curve turns up. About 3 GPU-days.

Prior-answer search: one web search (Cartridges capacity per slot), arXiv API ("KV cache" + capacity + compaction/distillation/learned). Closest: Cramming 2502.13063 (input embeddings, sequential reconstruction), 2508.17032, 2607.08032 (abstract mentions "a layer-agnostic lower bound"; body not read, could contain a bound).

Overlap: adjacent to D2 only in that both concern stored prefixes; this is a lossy trained prefix on dense models, not hybrid state checkpoints.

### Card 5. Is low-bit damage from bias or from error size, and does it differ for keys and values?

Question: Does low-bit value-cache damage come from the bias of the rounding rather than its size, as reported for recurrent state, while key damage comes from variance?

Type: paper A's explanation vs paper B's data.

Evidence: Lost in Rounding: same-width stochastic rounding of recurrent state removes the reasoning-length increase, "so the failure is mostly rounding bias, not missing resolution". LowRAM on keys: "Although QJL debiases the inner product estimations, we found that its variance is amplified by Softmax". eOptShrinkQ: the MSE quantizer "systematically shrinks each coordinate toward zero", yet a rank-1 removal at 2.04 bits matches the unbiased quantizer at 3.00 bits. 2604.11501 v2 section 4.1 gives a second-order formula in which a deterministic logit perturbation and zero-mean rounding noise have different costs, with no experiment. KVTuner: values tolerate 2 bits on perplexity (K8V2 10.04 vs 9.95); KVTC: value compression hurts GSM8K and MMLU.

Expert prediction (medium): values, bias matters in diffuse heads because attention averages out zero-mean noise; keys, variance matters.

Why it could be wrong: a KV entry is written once and its error is frozen, so unbiased rounding only helps across tokens inside one attention read, not across steps. Recurrent state is rewritten every step, so its bias accumulates. The recurrent-state result may not transfer, and the two KV data points say debiasing does not help.

Outcome A (bias dominates for values): choose quantizer family by tensor, not only bit width: unbiased or dithered for values, MSE-optimal for keys, and fewer value bits in diffuse heads.

Outcome B (size dominates): MSE-optimal everywhere, and the bias effect is specific to state that is rewritten each step. In a hybrid the two caches then need different quantizers.

First experiment: Llama-3.1-8B, Qwen3-8B, Qwen3.5-9B. Values, then keys, at 2 and 3 bits with nearest rounding, stochastic rounding on the same grid, Lloyd-Max, and Lloyd-Max with the shrink rescaled away. Per head split attention-output error into mean and variance terms; teacher-forced next-token KL; then closed loop on 200 GSM8K / MATH-500 problems for accuracy and generated length. Stop if nearest and stochastic rounding differ by under 5% in KL and within noise in accuracy for both tensors. About 2 GPU-days.

Prior-answer search: arXiv API ("KV cache" + stochastic rounding / dither / unbiased quantization) returned WitCert, QJL work, DASHQuant, and the recurrent-model quantization paper; none compares biased and unbiased rounding for keys and values separately. Closest: Lost in Rounding 2611.12127 (recurrent state only), LowRAM appendix (keys, QJL only).

Overlap: near the earlier residue (a) and Q25. It differs in asking which part of the error does the damage for each tensor, not for a length predictor.

## 5. Searches and coverage

- arXiv API (export.arxiv.org) over HTTPS; rate-limited (HTTP 429) for part of the session. Queries run are listed on each card. Semantic Scholar citation lookups for KVTuner and AsymKV failed with "Too Many Requests".
- Web search: 3 of the 6 allowed calls used.
- Not covered: bodies of HeadQ, NOVA-KV, RateQuant, RDKV, STAR-KV, Thin Keys, KVSculpt, 2607.08032; MLA-cache quantization papers beyond abstracts; KV compression for vision-language and video models; system papers on compressed-cache kernels; no citation-graph search worked.
