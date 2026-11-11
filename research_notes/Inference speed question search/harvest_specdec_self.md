# Harvest: self-speculation, sparse-KV drafts, lossy acceptance

Date: 2026-11-11. Read-only literature pass. No GPU jobs were run. Numbers marked "my arithmetic" are mine, not the papers'.

Sources were read from arxiv.org/html text dumps (body) or the arXiv API (abstract only). "Body" means I read the relevant sections; "abs" means abstract only. Figures were not read, only captions and the text around them.

## 1. Reading list

Anchors
- Draft & Verify, 2309.08168 (abs)
- LayerSkip, 2404.16710 (abs)
- Kangaroo, 2404.18911 (abs)
- SWIFT, 2410.06916 (abs)
- TriForce, 2404.11912 (body: sections 3.1, 3.2, Table 2)
- MagicDec, 2408.11049 (body: sections 3.3, 4.3, 4.4)
- QuantSpec, 2502.10424 (body)
- LongSpec, 2502.17421 (abs)
- SparseSpec, 2512.01278 (body; plus the local repo README and rejection sampler)
- Judge Decoding, 2501.19309 (body: intro, "Rejected tokens")
- AutoJudge, 2504.20039 (abs)
- SpecReason, 2504.07891 (abs)
- Speculative Thinking, 2504.12329 (abs)
- Reward-Guided SD, 2501.19324 (abs)
- R-Stitch, 2507.17307 (abs)
- SpecDec++, 2405.19715 (abs)
- The Sparse Frontier, 2504.17768 (abs)
- Speculative Decoding: Performance or Illusion?, 2601.11580 (body)
- Hold Onto That Thought (KV compression on reasoning), 2512.12008 (abs)

2025-2026
- Vegas (v1 title "SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding"), 2602.07223 (abs)
- ASPIRE, 2609.17943 (body)
- Coverage-Verified Sparse Attention (CVSA), 2611.10634 (body, incl. Appendix P)
- Acceptance Is Not Speed, 2610.23818 (body, incl. Appendix C, F)
- Nucleus Speculative Decoding, 2610.07822 (body: setup, results text)
- Revisiting Lossy Verification, 2607.26627 (abs; body downloaded, not read)
- Practical Investigation of Training-free Relaxed SD, 2607.08690 (abs; body downloaded, not read)
- When Is a Draft Accepted?, 2606.30265 (body)
- RIFT, 2610.31098 (body)
- S3R, 2610.13202 (body: budget sweep, Appendix E.2)
- EfficientRollout, 2606.18967 (body, incl. Appendix B.2, B.3)
- SparseSpec-L ("A Sparse Glimpse of the Whole"), 2607.27735 (body)
- VeriCache, 2605.17613 (body: section 4.3, discussion)
- Cassandra, 2605.26558 (body: section IV-C only)
- BudgetDraft, 2606.00144 (abs)
- Dustin, 2606.24957 (abs)
- SSV, 2605.19893 (abs)
- STS, 2605.15508 (abs)
- MARS, 2601.15498 (abs)
- JuDi, 2601.04766 (abs)
- PAD, 2511.00351 (abs)
- AdaptiveSpec ("Margins, Not Windows"), 2609.02897 (abs)
- Mentored Decoding, 2609.30474 (abs)
- TSS, 2609.26100 (abs)
- SSR: Self-Speculation for Faster Reasoning Models, 2608.20359 (skimmed body: it drafts the answer from a partial reasoning trace, not a sparse/quantized draft)
- A Calibrated Instrument, 2609.18005 (abs)
- Multi-Drafter SD with Alignment Feedback (MetaSD), 2604.05417 (abs)
- Quantization Inflates Reasoning, 2606.25519 (abs)
- RATIO, 2609.39801 (abs)
- Joint Subspace Sparse Attention, 2610.19026 (abs)
- Beyond KV Reconstruction (MLA drafts), 2607.27269 (abs)
- Strong Drafts Need Compact Memories, 2608.30252 (abs)
- When Hidden States Drift (KVShot), 2604.26412 (abs)

## 2. Extracted claims (exact quotes)

Sparse self-draft acceptance at short range
- ASPIRE 2609.17943: "At γ=1 both variants achieve near-identical acceptance (∼95%), since the context has not yet gone stale. As draft length grows, the gap widens steadily: without refresh, acceptance drops to 60.3% at γ=10, whereas with refresh it remains at 76.6%, a 16.4 pp gap." (Qwen3-8B, ρ=7%, temperature 0.7.) My arithmetic: a constant 0.95 per-token rate gives 0.95(1-0.95^10)/(0.05*10) = 0.762, so the with-refresh curve looks like a constant 5% per-token loss. This assumes "acceptance" means accepted/drafted.
- S3R 2610.13202 (s=0.05, T=1.0, Qwen3-8B): "acceptance falls from 95.8% for the first proposed token to 67.9% for the sixth". Budget sweep: "Increasing s from 0.05 to 0.50 raises acceptance from 80.4% to 96.7%, while speedup falls from 1.370× to 1.058×."
- SparseSpec 2512.01278: "PillarAttn achieves an average acceptance token length of 6.16 out of 8 tokens"; "We set the sparsity ratio to 0.05, as performance saturates with further increases in selected tokens."; "We set the temperature as 0.65 following common practice."; "For long output of RLMs, sparsity patterns shift significantly based on semantic context during generation."; "with k=16, α=0.75, s=0.05, the attention latency is reduced by 6.78×". My arithmetic: 6.16 of 8 matches an iid per-token rate of about 0.94.
- TriForce 2404.11912, Table 2 (4K budget, 122K prompt, Llama2-7B-128K): acceptance 0.9234 at T=0.0, 0.9137 at T=0.6, 0.9004 at T=1.0, 0.8902 at T=1.2; small-model baseline 0.4649 -> 0.2664. "utilizing only 1K tokens could theoretically achieve a 97.6% acceptance rate with Top-K selection method".
- VeriCache 2605.17613: "25–40 accepted tokens per verification round for VeriCache vs. only 2–3 for typical small-model drafters"; "at 4× compaction, the acceptance rate stays above ∼0.8 even at draft length 30"; "Existing compressors optimize direct-serving accuracy; a compressor designed to maximize acceptance length at large draft horizons—a different objective—could push VeriCache’s throughput further."
- SparseSpec-L 2607.27735: "negative draft entropy obtains an acceptance AUC of 0.638 on RULER QA1. Tokens in the low-entropy bin are accepted at 86.3%, compared with 62.1% in the high-entropy bin."; "all methods exhibit stable acceptance rates and increasing speedups at longer contexts" (10K to 60K); MagicDec (SnapKV) "When forced to a longer horizon (k=10), its acceptance rate degrades below that of StreamingLLM due to permanent information loss."; LayerSkip "0.80× speedup with 41.4% acceptance".
- RIFT 2610.31098: retained attention "drops from 67.0% to 25.4% for SnapKV, while increasing from 43.5% to 68.7% for SWA as generation progresses"; "decode staleness is exacerbated in hybrid-attention architectures".

What predicts acceptance
- EfficientRollout 2606.18967: "target-output entropy and token-level agreement between ℳp and its weight-quantized copy Q(ℳp) show a strong negative correlation (r=−0.99)" (across RL steps, not per token). "lowering temperature from 1.0 to 0.6 raises α from 97.6% to 99.0%".
- When Is a Draft Accepted? 2606.30265: "the mean top token probability is ≈0.85 for Qwen3-1.7B and ≈0.87 for Qwen3-4B, and the median is even higher at ≈0.97 and ≈0.99"; "On low margin target distributions, even a small draft-target divergence may be sufficient to cause the greedy criterion to reject".
- CVSA 2611.10634: coverage "satisfies an exact algebraic equality with per-step output error"; "chronic cells ... are 1.6% of cells ... account for only 11.1% of failure events, while intermittent cells failing on 5–80% of steps account for 86.4%, and the set of failing heads overlaps by a Jaccard index of only 0.33 between consecutive steps."

Sparse versus quantized self-drafts
- MagicDec 2408.11049: "model compression alone is unable to reach 90% acceptance rate, while KV compression achieved significantly higher rates under similar memory constraints."
- EfficientRollout App. B.2: "Under sparsity budgets that are cheap enough to yield speedup in practice, e.g., retaining 256 tokens, the resulting drafter remains relatively weak: with γ=5, block efficiency τ typically stays around 3–4. In contrast, a 4-bit weight-quantized drafter works well in our regime. Even with the simplest asymmetric RTN (round-to-nearest) quantization, we achieve block efficiency τ=5.2 at γ=5." Table 4: idealized draft/target latency 0.360 (W4), 0.573 (W8).
- QuantSpec 2502.10424: "maintains high acceptance rates (>90%)"; "can be combined with sparse KV methods ... which we leave for future work."
- Cassandra 2605.26558: "using value pruning and mantissa truncation together demonstrates a robust acceptance rate compared to using either method alone."
- Performance or Illusion 2601.11580: "different SD methods achieve higher acceptance at different token positions. This complementarity suggests that adaptively combining multiple SD methods can unlock additional speedup" (n-gram vs EAGLE only).

Lossy acceptance and length
- CVSA App. P: "on MATH-500 (R1-Distill-Llama-8B, 200 problems, seed-paired sampling) CVSA-c4 reaches 64.0% against dense 80.5%. ... refreshing the selection every step (M=1) recovers two-thirds of the gap, to 75.0% ... cap-outs fall from 31 to 13 of 200 problems and generation-length inflation from +61% to +20% above dense. A residual 5.5-point gap remains".
- Acceptance Is Not Speed 2610.23818: "relaxed verification on average accepts 1.34× as many tokens per round but produces 1.54× longer completions"; λ median "0.94 for standalone drafters to 1.23 for EAGLE heads"; "The relaxed-only tokens are 7% to 24% of the committed draft tokens. For the four rules other than spec_casc_tok they have a median target probability of 0.01 to 0.09 (shared tokens: 0.78 to 0.93)"; "At T=1.2, GSM8K completions are 1.27× longer ([1.10,1.47]) ... with accuracy unchanged (0.96 against 0.97)"; "At Qwen3-8B’s recommended sampler (temperature 0.6), mentored_dec, cactus, and spec_casc_tok stop inflating on GSM8K and LiveCodeBench"; at that sampler the gentle rules "no longer lengthen completions (λ 1.01 to 1.04), save 4 to 7% of the rounds, and keep accuracy"; "A draft token outside the target’s top-p/top-k support is rejected regardless of α." No sparse or quantized self-drafts in the study.
- Nucleus SD 2610.07822: "over half of the rejected tokens lie inside the target’s own nucleus"; "NSD’s accuracy difference relative to SD ranges from −1.58 to +4.14 percentage points, with NSD achieving 0.40 percentage points higher accuracy on average"; "Speedup Ratio is the ratio of each method’s token throughput to target model autoregressive (AR) throughput" (not corrected for completion length); T=1.0, target top-p 1.0. Drafters: DSpark and Nemotron diffusion self-speculation; no sparse-KV or quantized self-draft.
- Quantization Inflates Reasoning 2606.25519 (abs): "INT4/INT3 quantization can preserve accuracy but increase reasoning-token usage".
- Hold Onto That Thought 2512.12008 (abs): "eviction strategies at low budgets can produce longer reasoning traces".
- Judge Decoding 2501.19309: "even in instances where the draft model produces entirely accurate solutions, the target model frequently rejects numerous tokens".

Budget and context length
- The Sparse Frontier 2504.17768 (abs): "longer sequences tolerate higher sparsity, suggesting that fixed-budget methods in production are suboptimal".
- BudgetDraft 2606.00144 (abs): "naive sparse/full speculative decoding suffers from the sparse/full mismatch as context length grows, causing the acceptance rate to drop quickly".

Draft quality versus stand-alone quality
- Beyond KV Reconstruction 2607.27269 (abs): "attention-function errors that may be tolerable for standalone generation but substantially lower draft-token acceptance".
- ASPIRE: "attention at the last layer is specialized for next-token prediction and less representative of overall page importance".

## 3. Tensions considered

Kept (cards 1-5 below)
1. The ~5% per-token loss of sparse self-drafts at zero staleness shows up in three papers with different models, budgets and temperatures. Two readings (target is near-undecided there; a needed entry was missing) predict different things and the published numbers split.
2. MagicDec ranks KV compression above weight compression as a self-draft; EfficientRollout ranks W4 above sparse. Nobody has checked whether the two drafts fail at the same positions.
3. Nucleus SD says in-nucleus rejections are free to accept; Acceptance Is Not Speed says relaxed rules run 1.54x longer; CVSA shows unverified sparse attention costs 16.5 points with about 95% token agreement. Which tokens do the damage?
4. Fixed ratio (SparseSpec 5%, ASPIRE 7%) versus fixed count (TriForce 4K, RIFT 8K) budgets; Sparse Frontier and BudgetDraft point in opposite directions; nobody separates prompt length from generated length.
5. Selector ranking as a draft versus as a stand-alone compressed cache: SnapKV falls below StreamingLLM at k=10; VeriCache and the MLA-draft paper both say the two objectives differ.

Dropped
- Temperature dependence of same-model draft acceptance. Answered: mild decline in TriForce Table 2 (0.9234 -> 0.8902 from T=0 to 1.2) and EfficientRollout (99.0% -> 97.6% from T=0.6 to 1.0). The temperature data is used as evidence in card 1 instead.
- Is the gain of relaxed rules just an implicit sampler change? Answered by Acceptance Is Not Speed, Appendix C: at the recommended sampler the gentle rules save only 4 to 7% of rounds.
- Staleness versus error accumulation inside a draft window. ASPIRE's refresh result already separates them (with refresh the decay matches a constant per-token rate). Folded into card 1.
- One quantity predicting reasoning-length inflation across causes. This is residue (a) of the earlier search with relaxed acceptance added. Not re-proposed as a card; card 3 is the per-token intervention version and says so.
- Sparse self-speculation on hybrid (GDN plus attention) models. Shape is "does the gain survive a condition", and it sits on D1, Q23 and residue (b); the repo's own H1 run already measured hybrid diffuseness.
- Certified lossless verify-skipping from coverage bounds. An expert would be confident the bounds are too loose; "When Is a Draft Accepted?" already gives a log(2) ceiling on KL certificates.
- Sparse verification (Dustin, SSV) versus sparse drafting claiming the same attention saving. Reduces to a budget-versus-agreement curve, which S3R's sweep already sketches (96.7% at s=0.50).
- Why layer-skip self-drafts are far worse (41.4%) than sparse or quantized ones. Predictable: late layers sharpen the next-token distribution.
- Layer-wise budget allocation; which layer to refresh from. ASPIRE's layer sweep answers the second; the first is predictable.
- Draft-side predictors of rejection (draft entropy AUC 0.638). This is a design consequence of card 1, not a separate question.
- Anything specific to RL rollouts (D4), SD under concurrent load (Q22), memory-capped batch (Q02), cost ratio crossover (Q01).

## 4. Prior-answer searches

arXiv API queries (abstract field, sorted by date):
- "speculative decoding" + sparse + entropy + acceptance: no hits.
- "speculative decoding" + "rejected tokens": Nucleus SD 2610.07822, Carryover Drafting 2609.14717, ReTrace 2608.29748, Oilbird 2608.03839, D-cut 2607.14647 and others; none on sparse self-drafts.
- "self-speculative" + acceptance + (margin OR coverage): SparseSpec-L 2607.27735 only.
- "speculative decoding" + quantized + sparse + draft: QuantSpec 2502.10424, 2510.09942.
- "speculative decoding" + (complementarity OR "multiple drafters" OR "heterogeneous drafters"): MetaSD 2604.05417.
- "length inflation": Acceptance Is Not Speed 2610.23818, JSSA 2610.19026, several on-policy-distillation papers.
- reasoning + ("longer reasoning" OR "reasoning length") + (quantization OR "KV cache" OR "sparse attention"): 2606.25519, 2609.39801, 2512.12008.
- ti:"Sparse Frontier": 2504.17768.
- "sparse attention" + "context length" + ("scaling law" OR sublinear OR "fixed budget"): nothing relevant.
- "speculative decoding" + acceptance + ("target entropy" OR "predict acceptance" OR "acceptance prediction"): SpecDec++ 2405.19715.
- "speculative decoding" + ("sparse attention" OR "sparse KV"): 20 hits, all in the reading list or checked by title (this is how the Vegas id was found).
- "speculative decoding" + (lossy OR relaxed) + ("self-speculative" OR sparse OR quantized): VeriCache, LongSpec, HyCo 2610.21436; none applies a relaxed rule to a sparse or quantized self-draft.
- "KV cache" + compression + acceptance + draft: RIFT, 2608.30252, 2607.27269, 2604.26412.
- "speculative decoding" + KV + (eviction OR SnapKV OR H2O): Dustin, SpecExtend 2505.20776, a survey.

WebSearch (2 of 6 used):
- per-token rejection of sparse self-drafts against target entropy and attention coverage: no direct paper; closest SparseSpec-L.
- relaxed verification on sparse-KV self-drafts and length inflation: no direct paper.

## 5. Kept cards (short form; full cards are in the structured output)

1. What determines which tokens a fresh sparse-KV self-draft gets wrong: a near-undecided target, or a missing entry? Prediction: mostly the target. Confidence low.
2. Do a sparse-KV self-draft and a 4-bit-weight self-draft fail at the same positions? Prediction: large overlap. Confidence medium.
3. When unverified sparse attention makes a reasoning model longer and less accurate, is the damage done by tokens outside the dense nucleus or by individually plausible ones? Prediction: outside the nucleus. Confidence medium.
4. Does the KV count needed for fixed acceptance grow with context, and differently for generated reasoning than for prompts? Prediction: near-constant for prompts, near-linear for reasoning. Confidence low.
5. Do KV selectors rank the same as drafts (accepted run length) and as stand-alone compressed caches (task accuracy)? Prediction: mostly yes with a recency exception. Confidence medium.

## 6. Not covered

- Figures (only captions and surrounding text were read).
- Bodies of 2607.26627 and 2607.08690 (downloaded, abstracts only), BudgetDraft, Dustin, SSV, Vegas, MARS, JuDi, PAD, AdaptiveSpec, layer-skip papers.
- Citation graphs (Semantic Scholar was not queried in the second half of the session).
- The local subspec_v2 repo.
