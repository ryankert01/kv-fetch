# Harvest: composition across techniques, and theory of limits

Date: 2026-11-11. Read-only literature work. No GPU jobs, nothing posted, nothing committed.

Scope. Part A: papers that stack two or more acceleration techniques and report (or assume) how the gains combine. Part B: bounds and "limits" papers for speculative decoding (SD) and KV cache compression. The aim was questions, not methods: places where two papers disagree, where a paper assumes something nobody measured, or where one quantity might predict the gain of several techniques.

How to read the evidence marks. "Read" means I fetched arxiv.org/abs and arxiv.org/html in this session and the quote is copied from that text. "Abstract only" means the paper has no HTML version or I did not open the body. "Not opened" means I only saw the title in a search listing.

## 1. Reading list (all ids checked on arxiv.org in this session unless marked)

Theory of limits, KV side
- 2502.15955 Compression Barriers for Autoregressive Transformers (Haris, Onak; Feb 2025). Read. "Any algorithm for attention-based token generation must use $\Theta(nd)$ space, where $n$ is the number of tokens generated so far and $d = \Omega(\log n)$ is the dimension of the KV embeddings." Sparsity assumptions give sublinear space.
- 2607.01520 The risk of KV cache compression (Jul 2026). Read. Compression as "sparse approximation of a measure, strictly generalizing token eviction"; "sharp separation between algorithms with and without access to the future-query distribution"; "Random sampling already provides a strong query-agnostic baseline, consistent with its minimax optimality." Experiments: Qwen3-32B, LongBench-v2 long subset, about 95% compression; full KV 37.04 (+/- 4.65); authors say "standard errors overlap and further validation is warranted".
- 2610.35020 Attention under Bounded Key-Query Similarity (Chen, Feng, Indyk, Kapralov, Kochetkova, Prokhorov; 26 Oct 2026). Read. "the simple uniform sampling algorithm is nearly optimal for a typical range of parameters, supporting its surprising effectiveness on real datasets"; "the empirical (a, b) values for widely used models lie in or close to the uniform sampling regime". How (a, b) was measured: Figure 3, "endpoints a and b computed as 0.001 and 0.999 quantiles of cosine similarities distribution. Initial-token (attention sink) is excluded."; Appendix F uses query head 0 and KV head 0 of layers 4/8/16/24 only. The regime condition renders as "a < 0 < b and b − a ≤ 1 + b 2" in HTML (ambiguous between (1+b)/2 and 1+b^2; check the PDF before using it).
- 2609.03430 Random Attention (Sep 2026, v2 28 Sep). Read (abstract, sections 5.2, 5.3, Appendix D, Appendix H). "The selection signal, in other words, contributes almost nothing beyond uniform random selection." Policy: pin the prompt, i.i.d. uniform score per head for the rest. Caveats the authors state themselves: the policy is "a soft recency window" (survival "about 0.94^n at K = 1024 and r = 64"); "Random Attention never recovers a passcode announced 57 compression rounds before the question, while R-KV recovers it 84 % of the time"; shared draw across heads scores 0.871 / 0.788 vs 0.874 / 0.789 on Qwen3-4B MATH500 (cross-head diversity not load-bearing there); "Text-level redundancy is itself inferred, not measured"; "eviction generally lengthens generation relative to full attention". Metric is task accuracy, not token agreement.
- 2502.07861 BalanceKV (Feb 2025). Read Table 1 and Table 5. Uniform sampling of prefill KV at rate 0.25 is within about 0.1 to 1 point of SnapKV on LongBench-E averages (e.g. 48.76 vs 48.77 on one model, 49.88 vs 50.22 on another); in the low-error regime uniform loses a few points (TriviaQA 86.3 vs 91.6 at rate 0.8).
- 2605.18053 Protection Is (Nearly) All You Need (May 2026). Abstract only. "protection dominates; scoring differences are secondary once boundaries are guarded; per-head allocation gives a further modest gain."
- 2609.16617 Divergence Timing and Cumulative Disagreement under KV-Cache Eviction (Sep 2026). Abstract only. "post-divergence exposure accounts for 85-90% of four aggregate mismatch gaps"; SnapKV at 50% "enters divergence later and less often" than recency retention "while post-divergence total variation (TV) remains high".
- 2603.04427 Thin Keys, Full Values (Feb 2026). Abstract only. "selection requires only $O(\log N)$ dimensions"; "composes with GQA and quantization for up to 16× combined key cache compression".
- 2611.10634 Coverage-Verified Sparse Attention (7 Nov 2026). Abstract only. "The fraction of softmax mass on the selected tokens, which we call coverage, satisfies an exact algebraic equality with per-step output error".

Theory of limits, SD side
- 2512.11718 Speculative Decoding Speed-of-Light (Dec 2025). Read. Entropy-based bound for deterministic drafters with negligible drafter cost and i.i.d. acceptance ("Although the i.i.d. assumption is unreasonable"); "about × 2 difference between the theoretical bound and EAGLE-3 is caused by a speculation error"; "current deterministic algorithms are approaching the fundamental limits under the assumptions studied".
- 2606.12370 Bebop (Jun 2026). Abstract only. "the MTP acceptance rate is fundamentally bounded by the fluctuation of model entropy".
- 2605.18810 D-PACE (May 2026). Read Table 1: DFlash and D-PACE drafters of Qwen3-4B lose only about 6 to 7% of emitted length from T=0 to T=1 (e.g. 4.85 vs 4.50).
- 2610.28355 Beyond the Point Mass / D-PAL (Oct 2026). Abstract only. "from 4.50 to 4.57 at T=0 and from 4.20 to 4.28 at T=1".
- 2608.02438 XPress (Aug 2026). Abstract only. Restoring causality in a block drafter "raises acceptance length by about 30% on average (up to +56%)" on Qwen3-8B.
- 2607.06763 Trees from Marginals (no HTML). Abstract only. "their independence assumption causes acceptance rates to degrade sharply as the speculative budget grows".
- 2604.26412 KVShot (Apr 2026). Abstract only. "long-range decay persists even in TTT-trained drafters"; "KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal".
- 2609.02897 Margins, Not Windows / AdaptiveSpec. Abstract only. Margin rule plus tree policy "operate on orthogonal axes and compound in effect".
- 2610.00321 CAST, 2611.14399 REAL loss, 2611.00588 TVP, 2610.29210 Target Probability Reuse, 2607.08690 Training-free Relaxed SD. Abstracts only.
- 2403.10444 Block Verification, 2511.15898 Global Resolution, 2602.16961, 2601.05724, 2505.12398. Not opened (listing only).

Composition
- 2512.01278 SparseSpec (Nov 2025). Read. "instead of using a static sparsity pattern (e.g., sliding window attention), a dynamic sparse attention mechansim that adapts to the context dynamics is necessary to achieve a high acceptance rate"; "PillarAttn achieves an average acceptance token length of 6.16 out of 8 tokens"; sparsity 0.05; baselines are EAGLE-3, N-gram, Streaming. No random or uniform selection baseline.
- 2602.07223 Vegas (Feb 2026). Abstract and a keyword scan. Verification-guided top-K; no random-selection baseline found.
- 2607.27735 SparseSpec-L (Jul 2026). Read in part. "negative draft entropy obtains an acceptance AUC of 0.638 on RULER QA1. Tokens in the low-entropy bin are accepted at 86.3%, compared with 62.1% in the high-entropy bin."
- 2609.17943 ASPIRE (Sep 2026). Abstract only. Per-request acceptance estimates, intra-draft refresh layer.
- 2605.17613 VeriCache (May 2026). Read. "25–40 accepted tokens per verification round for VeriCache vs. only 2–3 for typical small-model drafters"; "VeriCache + Eagle reaches 4.35 × ideal speedup vs. 3.50 × for VeriCache alone and 1.78 × for Eagle alone (Appendix C)"; Appendix C: "Under chain-independence of the two drafters, the expected accepted draft length per cycle becomes γ(x, c) · x · [1 + γ_e(d_e) · (d_e − 1)]". Sequence-level KL "grows roughly linearly in t under KVzip 4× (left) and TurboQuant k4v3 (right); at temperature 0.5".
- 2610.31098 RIFT (Oct 2026). Read in part. "decode staleness"; for training masks, "the attention-based masking scheme outperforms every random and sliding-window control at temperatures 0 and 1 and at all three considered KV budgets".
- 2608.30252 Strong Drafts Need Compact Memories, 2606.24957 Dustin. Abstracts only.
- 2510.26843 CAS-Spec (NeurIPS 2025). Read in part. Cascades of self-drafts; its simulation assumes "α(M_t, M_d2) = α(M_d1, M_d2)".
- 2604.05417 MetaSD (multi-drafter bandit). Abstract only.
- 2605.01106 Component-Aware Self-Speculative Decoding in Hybrid Language Models (May 2026). Abstract only. Falcon-H1 alpha = 0.68 at k=2, Qwen3.5 alpha = 0.038; "perplexity degradation from a companion ablation study predicts speculative viability without running speculative decoding".
- 2505.22179 SpecMQuant (May 2025). Read in part. "the increased computational load of tree-style draft verification undermines the memory access benefits from 4-bit weight quantization".
- 2603.01399 Quasar. Abstract only. "Being orthogonal to existing drafting strategies".
- 2607.04244 Quantize the Target, Quantize the Drafter (Jul 2026). Read in part. "In our experiments, SWA often increases the mean acceptance length and reduces decoding latency, especially for long-context inputs."
- 2607.20434 Break Through the Compression Bottleneck (May 2026). Abstract only. "first mathematical proof that low-rank decomposition and quantization are non-orthogonal".
- 2609.18005 A Calibrated Instrument (no HTML). Abstract only. "A model's certainty about a token predicts how likely it is to differ from the full model's choice, but not how much that difference affects judged quality."
- 2607.20981 Beyond Independent Optimization (survey, multimodal edge). Abstract only.
- 2606.00206 Quantized Reasoning Models Think They Need to Think Longer. NOT read in this session (seen in a web search summary and in the earlier verification notes).

Not confirmed, so not cited with an id: SpecTr, QuantSpec, MagicPIG, "One-Step Limits of Randomized Speculative Decoding".

## 2. Tensions considered and dropped

- SD x weight quantization: why 4-bit targets gain less from SD. Answered by SpecMQuant itself (verification-to-decoding time ratio rises on W4 models). Predictable.
- What bounds a block drafter's acceptance at temperature 0 (entropy vs missing causality vs stale inputs). The pieces are already measured: T=0 vs T=1 gap is about 7% (D-PACE, D-PAL), restoring causality gives about +30% (XPress), decay persists in causal drafters (KVShot). An expert would say "both, with staleness left over" with fair confidence.
- Are T=0 rejections located where the target is near-tied? AdaptiveSpec's margin rule already exploits this; expert confidence high.
- Does a KV compressor beat the query-agnostic minimax bound (2607.01520)? Real queries are not worst case; predictable.
- Does the key rank needed grow with log n (Haris-Onak d = Omega(log n) vs Thin Keys O(log N))? A fixed logit error gives a multiplicative attention error that does not depend on n; predictable "no".
- Depth pruning x quantization, layer skip after pruning. Predictable sub-additivity.
- Thinking-token eviction x sparse-KV drafting on the same cache. Arithmetic on bytes read; predictable.
- "Reasoning traces restate what they need" (Random Attention) vs N-gram drafts accepting fewer than 2 of 8 tokens on reasoning (SparseSpec). Looks like a contradiction but resolves trivially: restated values are not verbatim n-grams.
- Speed-of-Light bound vs VeriCache's 25 to 40 accepted tokens. Not a violation: the bound depends on target entropy, which is low on the agentic traces used.
- Space-time tradeoff of recomputing KV from token ids. Too close to dropped direction D2.
- Bebop's RL-stage entropy claim. RL framing is dropped direction D4; only the entropy-acceptance sentence was used.

## 3. Kept cards (summary; full cards returned in structured output)

C1. What decides whether attention-score selection beats signal-free sampling for a sparse-KV self-drafter, head by head, and does a key-query statistic predict it?
- Theory (2610.35020, 2607.01520) and one empirical paper (2609.03430) say uniform or random KV sampling is near-optimal; every sparse self-speculation system (SparseSpec, Vegas, SparseSpec-L, ASPIRE, RIFT) is built on top-K selection and none has a random baseline. The theory's regime was measured with quantile clipping on one head per layer. Random Attention's metric is task accuracy after a longer trace; a drafter needs next-token agreement.

C2. Do a compressed-KV self-drafter and a small or MTP drafter get rejected at the same tokens?
- VeriCache's composed 4.35x is an ideal number computed "Under chain-independence of the two drafters". The HTML shows no experimental support for that assumption. Correlation in either direction changes the composed length, and positive correlation helps the cascade rather than hurting it.

C3. Is reasoning-length inflation under a lossy variant set by its teacher-forced divergence hazard, on one curve across families, or by what happens after the first divergence?
- Builds on residue (a) of the earlier search. New inputs: the 2609.16617 decomposition (85 to 90% post-divergence), VeriCache's linear KL growth, and the fact that the same hazard number is the acceptance rate when the lossy variant is used as a drafter, which sets up a trust-or-verify decision rule.

C4. When two lossy techniques are stacked, do their logit perturbations add, and do they point the same way?
- "Orthogonal" is claimed in at least four abstracts without a measurement; one 2026 paper proves non-orthogonality for low-rank plus quantization of weights. A one-step, teacher-forced test gives "orthogonal" a checkable meaning.

C5 (weakest, heavy overlap with earlier Q23). Are the attention layers of a GDN hybrid in a different key-query regime from dense layers once the sink and normalisation are removed?
- Builds on residue (b). New inputs: the sink-excluded (a, b) statistic of 2610.35020 and 2605.01106's alpha = 0.038 for Qwen3.5 without its attention layers.

## 4. Prior-answer searches

arXiv listing search (HTML scrape; the arXiv API and Semantic Scholar search returned 429 for most of the session): "speculative decoding" "random eviction"; "self-speculative" "uniform sampling" KV; "speculative decoding" "sparse attention" draft acceptance; "speculative decoding" "KV cache" compression draft; "KV cache" eviction "per head" random; "speculative decoding" heterogeneous drafters; "speculative decoding" drafters "complementary" errors acceptance; "speculative decoding" correlated rejections cascade; "speculative decoding" "acceptance" "perplexity" predicts; "divergence" "generation length" reasoning quantization predict; LLM compression "non-orthogonal" quantization; "KV cache" quantization eviction "compose" error additive; "speculative decoding" limits "block" drafter "independence"; "speculative decoding" "epistemic".
WebSearch (3 of 6 used): sparse self-speculative draft acceptance random vs top-k ablation; cascade SD two drafters rejection correlation independence; per-token KL predicts reasoning length inflation across quantization and eviction.
Closest papers per card are listed in the structured output.

## 5. Not done

- Semantic Scholar citation lists for 2609.03430, 2610.35020 and 2512.11718 returned nothing in the last attempt, so papers citing them after October 2026 are not covered.
- Bodies not opened: Vegas ablation tables, ASPIRE, Dustin, Strong Drafts Need Compact Memories, MetaSD, HCSpec (ACL 2026, seen only in a web result), 2606.00206, 2611.10634.
- SpecTr, Block Verification and the multi-draft optimal-transport line were not read; no card depends on them.
- The regime inequality in 2610.35020 should be checked against the PDF.
