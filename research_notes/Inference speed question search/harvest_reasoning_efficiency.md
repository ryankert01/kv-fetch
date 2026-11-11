# Harvest: reasoning-model efficiency (how long models think and what it buys)

Date: 2026-11-11. Read-only literature work. No GPU jobs, no posts, no commits.
Sources: arXiv API (keyword and id-list queries), arXiv HTML pages read with WebFetch. Semantic Scholar returned 429 and was not used. WebSearch was not used (0 of 6).

"Read" below means the HTML full text was opened in this session and the quoted text or number was seen there. "Abstract" means only the arXiv abstract was read. Long pages were read up to the first 100,000 characters.

## 1. Reading list

All ids were confirmed through the arXiv API in this session.

### Anchors
| id | short title | depth |
|---|---|---|
| 2504.15895 | DEER, Dynamic Early Exit in Reasoning Models | read |
| 2504.05419 | Reasoning Models Know When They're Right | read |
| 2505.17813 | Don't Overthink It (shorter chains preferred) | abstract |
| 2502.12067 | TokenSkip | abstract |
| 2502.09601 | CoT-Valve | abstract |
| 2412.06769 | Coconut | abstract |
| 2506.02536 | Answer Convergence as a Signal for Early Stopping | abstract |
| 2505.24133 | R-KV | abstract |
| 2510.01290 | ThinKV | abstract |
| 2504.13367 | ThoughtTerminator | abstract |
| 2412.18547 | Token-Budget-Aware LLM Reasoning | abstract |
| 2501.19393 | s1 | abstract |
| 2504.09858 | Reasoning Models Can Be Effective Without Thinking | abstract |
| 2503.01141 | Token complexity of CoT compression | abstract |
| 2503.16419 | Stop Overthinking (survey) | abstract |
| 2502.03373 | Demystifying Long CoT | abstract |
| 2512.01278 | SparseSpec (PillarAttn) | read |
| 2506.19143 | Thought Anchors | abstract |
| 2512.12895 | Wait, Wait, Wait... Why Do Reasoning Models Loop? | abstract |

### 2025-2026
| id | short title | depth |
|---|---|---|
| 2506.08390 | On Reasoning Strength Planning (pre-allocated length direction) | read |
| 2509.24248 | SpecExit | read |
| 2512.12008 | Hold Onto That Thought (KV compression on reasoning) | read |
| 2604.06613 | The Detection-Extraction Gap | read |
| 2606.00206 | Quantized Reasoning Models Think They Need to Think Longer | read |
| 2606.13603 | Beyond the Commitment Boundary | read |
| 2606.26472 | EpiKV | read (first 100K chars) |
| 2608.04771 | ReCo | read |
| 2608.20359 | SSR | read |
| 2608.28859 | Halt Vector | read |
| 2609.03633 | </think> Doesn't Stop Reasoning | read |
| 2609.06473 | Steering Under Compression | read |
| 2609.09989 | Stable Answers, Unfinished Reasoning | read |
| 2610.05685 | BreadthKV | read |
| 2610.13570 | Pareto Frontiers of Reasoning Termination | read |
| 2610.30644 | Speculative Acceptance Does Not Decay with Position | read |
| 2611.02278 | Concise-9B (concise self-training of Qwen3.5-9B) | read |
| 2611.12127 | Lost in Rounding (bf16 recurrent state, hybrids) | read |
| 2509.26522 | Entropy After </Think> | abstract |
| 2510.19176 | Zero-Step Thinking | abstract |
| 2510.24941 | Decorative vs true thinking steps | abstract |
| 2512.07993 | SkipKV | abstract |
| 2505.13866 | RPC | abstract |
| 2601.18383 | DynTS | abstract |
| 2601.16986 | Crystal-KV | abstract |
| 2601.03066 | Functional importance of reasoning tokens | abstract |
| 2603.05488 | Reasoning Theater | abstract |
| 2604.23351 | Solution hides in hidden states | abstract |
| 2605.27965 | Shape of Overthinking | abstract |
| 2605.11746 | When Reasoning Traces Become Performative | abstract |
| 2605.11467 | ProFIL (probe-filtered RL) | abstract |
| 2606.25519 | Quantization Inflates Reasoning | abstract |
| 2606.30852 | When Does Learning to Stop Help | abstract |
| 2606.02835 | Thinking Past the Answer | abstract |
| 2606.15070 | ASAG | abstract |
| 2606.03928 | VaSE | abstract |
| 2607.21433 | Token Budget Saturation | abstract |
| 2607.11089 | OS-Pruner | abstract |
| 2607.09786 | Length penalties make reasoning less monitorable | abstract |
| 2607.11317 | Calibrated e-CUSUM decoding for quantized reasoning models | abstract |
| 2608.15797 | KV-Rescue | abstract |
| 2608.21860 | ChainPrune | abstract |
| 2608.27963 | SABER | abstract |
| 2609.39801 | RATIO | abstract |
| 2610.20641 | AbridgeKV | abstract |
| 2610.23818 | Acceptance Is Not Speed (lossy SD length inflation) | abstract |
| 2610.25908 | CALM | abstract |
| 2610.25970 | GLOBE | abstract |
| 2611.05919 | Routes of Reasoning (MoE routers) | abstract |
| 2611.10845 | Inference for Reasoning Models | abstract |
| 2611.11901 | Adaptive Entropy-Sparing | abstract |
| 2507.08806 | Think Clearly | abstract |
| 2506.19830 | Lookahead Reasoning | abstract |
| 2602.03708 | SemanticSpec | abstract |
| 2603.11504 | LongFlow | abstract |
| 2510.10964 | Not All Bits Are Equal | abstract |
| 2504.04823 | Quantization Hurts Reasoning? | abstract |
| 2606.23181 | DART | id confirmed, not read |

## 2. Extracted claims (exact quotes and numbers)

### How much of the thinking comes after the answer
- 2604.06613 (read). Post-commitment share on MATH-500: Qwen3-32B-Think 75%, 32B-NoThink 61%, 8B-Think 73%, 8B-NoThink 52%, GPT-OSS-120B 64%; abstract: "52--88% of chain-of-thought tokens are produced after the answer is recoverable from a partial prefix". Commitment is the earliest checkpoint where 8 free continuations agree at 0.75 or more. At a 10% prefix Qwen3-32B-Think: 70% recoverable by free continuation, 34% correct under forced extraction. BAEE truncates 70-78% of serial generation "while improving accuracy by 1–5 pp"; total tokens rise about 3-5x. Post-commitment entropy rises for Think models (about 1.38-1.44x the pre-commitment value) and falls for NoThink models; not explained.
- 2606.13603 (read). gemma-4-26B-A4B-it reaches its final answer after 13-23% of CoT tokens, gpt-oss-20b after 43-68%; "up to 87% of generated reasoning tokens" after stabilisation in some settings. Number-token corruption at 20%: answer preserved in "only 61% of runs" before the boundary and 95% after. Hedging words appear at similar rates before and after the boundary. Probe exit on MATH-500 "recovers 89% of full-CoT accuracy" saving 39%.
- 2611.02278 (read). "every 2,048 thinking tokens we close the trace's thinking and let the model state its answer"; "54% of thinking tokens on the hard set and 64% on the confirmation set come after that point". "Stopping each trace there would keep every answer at 0.35–0.44×the base model's tokens, but finding that point takes hindsight". Stopping after three agreeing forced answers costs the base model 2.4-9.2 points. Concise-9B: 0.66x tokens, accuracy -0.20 [-0.77, +0.36] points; "86% [83, 90] of the thinking tokens it saves come after the answer has formed"; post-answer thinking 7,506 -> 3,692 tokens, pre-answer 5,137 -> 4,538. Olmo-3-7B-Think saves only 1-4%. 59% of wrong traces also have an early fixed point.
- 2610.13570 (read). Overthinking ratio: GPT-OSS-20B 41.1%, Qwen3-4B-Thinking 70.5%, Qwen3-30B-A3B-Thinking 74.4%, R1-Distill-Qwen-7B 66.7%, 14B 64.8%, 32B 68.6%. "Greater length redundancy does not imply greater accuracy degradation." Oracle budget to match full thinking: dynamic routing 71.4%, fixed budgets 35.0%, dynamic early exit 14.2%. Within-question mean pairwise absolute difference of the ratio: 20.23 pp.
- 2609.09989 (read). "agreement establishes that the current answer persists under a fixed probing procedure, not that the reasoning has terminated". Of 1,500 trajectories, 736 "agreed on an answer they went on to abandon"; 84.2% of those "still ended correct". 155 of 216 non-terminal stops "commit a wrong answer to a trajectory that would have reached the right one".
- 2504.15895 (read). "accuracy improvements of 0.9 to 4.8 points while reducing sequence length by 19.1% to 42.9%".
- 2504.05419 (read). Probes "achieving ROC-AUC scores above 0.7"; early exit "reducing the number of inference tokens by 24% without compromising performance"; models "do not efficiently use this internal knowledge during inference".
- 2607.21433 (abstract). Converged traces 90.3% accurate, non-converged 6.6%; probe AUC 0.608.
- 2608.27963 SABER (abstract). "reduces reasoning token consumption by 30.2%--39.8% on average".

### What controls length inside the model
- 2506.08390 (read). Linear probe on question activations: "a correlation of 0.84 between predicted and actual values"; "the changes in the logits of the end-of-reasoning token are the result of planning, rather than the cause of planning." Steering at -0.2 on GPQA: tokens 1022.10 vs 6998.01, accuracy 5.66 vs 33.92.
- 2608.28859 (read). Halt vector at layer 18 of R1-Distill-Qwen-7B. Steering strength vs length rho = -0.82. Saturation about 1,650 think tokens against a median of about 3,700 (factor 2.24). "the cut tracks each problem's own removable slack at 0.70". A random direction makes generation about 10% longer. "Satisfying a single scalar leaves the off-axis dimensions unconstrained, and they move". No test under quantization or KV compression.
- 2609.06473 (read). Steering under INT8 and NF4 on Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-it. Vectors extracted separately per precision: "cosine similarity 0.989-0.998 for INT8, 0.945-0.990 for NF4". "shortening is a step function with only 12-30% shortening (model-dependent) before discontinuous failure". Applying the FP16 vector to the quantized model "is deferred to follow-up work". No projection of quantized activations on the direction; no reasoning models; no KV methods.
- 2609.03633 (read). After an injected </think>, "reasoning-like generation continues into the answering phase". End-of-thinking regeneration rate under DEER: QwQ-32B 33.9%, R1-Distill-14B 16.3%, R1-Distill-1.5B 2.2%, Qwen3-14B 1.3%. Authors' hypothesis: "subsequent tokens assign it insufficient attention". Attention bias on the injected token: 16.3% -> 3.8%, answer tokens 1,252 -> 667.

### Lossy methods lengthen thinking
- 2606.00206 (read). "quantization noise can shift the sampled token toward an overthinking marker such as "Wait" or "But"". "models reach the right answer in intermediate reasoning steps but do not output it as a final answer" in "up to 52%" of failures. Position-level KL vs full-precision entropy: Spearman 0.92. Marker penalty: CoT length -12 to -23%. No measurement of the stop decision.
- 2611.12127 (read). bf16 recurrent state, six hybrid models, 590 problems: geometric-mean length x1.07 to x1.27 on five of six. Qwen3.5-9B cap hits 118 -> 223, accuracy 85.6% -> 70.8%. Same-width stochastic rounding restores Qwen3.5-9B to 87.6% and removes the length increase in four of five affected models. Head octile with 49% of teacher-forced KL had no length effect; stochastic rounding on the octiles with 37% + 15% removed essentially all of it. Falcon-H1R-7B unaffected (x1.008), unexplained. Extra length not located relative to the answer.
- 2608.04771 (read). "a smaller cache leads the model to generate more tokens, partly canceling the saving." R-KV at 25% on MATH-500: R1-Distill-Qwen-7B 3,268.7 -> 4,538.1 tokens (+38.8%); R1-Distill-Llama-8B +79.0%; 399 of 500 problems longer. No mechanism.
- 2610.05685 (read). "SnapKV-D outputs are 1.1–2.0× as long as FullKV's." Hypothesis, not tested: "eviction deletes the record of a check along with the result." "wait" per thousand tokens on Qwen3 AIME24 at B=512: 6.4 SnapKV-D, 3.7 FullKV. Derailed runs "rarely loop word for word". Offline distortion picks the wrong method on two of three models. Loop-detector rescue recovered 15% of capped runs.
- 2606.26472 EpiKV (read, first 100K chars). "the generations that finish are wrong about as often as without eviction". Table 3, MATH-500, K=1024, n=495, errors / cap hits / terminated wrong, Qwen-7B: none 104/55/49, RaaS 149/99/50, H2O 172/124/48. Example where the model "reaches the right answer and cannot stop". "Accuracy follows the tokens the model is still reading when it decides to stop." Protecting large-magnitude value states "changes neither loops nor accuracy". Greedy decoding, two R1 distills. No systematic location of loop onset relative to the first correct answer in the part read.
- 2608.15797 KV-Rescue (abstract). "An evicted 7B model and a full-context 1.5B model make complementary errors, and an oracle choice between their answers recovers 79% of the accuracy gap to the full-KV 7B model."
- 2606.03928 VaSE (abstract). Evicting large-magnitude value states "causes catastrophic failure where models enter repetitive reasoning loops".
- 2512.12008 (read). "At lower budgets, eviction occurs more frequently, resulting in a higher likelihood of critical token eviction, resulting in longer reasoning."
- 2610.23818 (abstract). Relaxed verification "on average accepts $1.34\times$ as many tokens per round but produces $1.54\times$ longer completions".
- 2512.12895 (abstract). Looping in unperturbed models attributed to errors in learning; "generations remain much longer than necessary at high temperature".

### Drafting on reasoning traces
- 2512.01278 SparseSpec (read). "average acceptance token length of 6.16 out of 8 tokens"; "the critical tokens of the KV-Cache differs dramatically over time". No breakdown by trace phase.
- 2610.30644 (read). Second half minus first half acceptance, within problem: same-family greedy +0.031 [+0.026, +0.036], cross-family greedy +0.069 [+0.063, +0.076]. Number and operator tokens accepted at ">=0.93 in all configurations"; reflection tokens 0.747 (same family, greedy), 0.367 (cross family). No split by reasoning phase, no entropy by position, one target (R1-Distill-Qwen-7B), independent drafts only, gamma = 5.
- 2509.24248 SpecExit (read). Qwen GSM8K: tokens -54%, latency 140.3 -> 75.8 (-46%). No acceptance by phase.
- 2608.20359 SSR (read). "The speedup therefore comes from hiding draft latency behind ongoing reasoning, not from reducing total FLOPs."

## 3. Tensions considered and dropped

| tension | why dropped |
|---|---|
| Lossy speculative decoding lengthens reasoning | Answered by 2610.23818. |
| Decision tokens bound speculative decoding gains | Predictable; 2606.00206 (rho 0.92 with entropy) and 2610.30644 cover it. |
| Oracle bound on early exit as a number | Computed in 2610.13570. Kept only the "why is the bound not reached" form (card 3). |
| Self-consensus versus confidence as exit signal | Settled in 2609.09989. |
| Shortening sometimes raises accuracy | Mostly cap and noise effects (Q on R-KV noise in the earlier search); predictable. |
| How much thinking is "needed": 52-88% versus 19-43% | Definitional. Prefix self-consistency measures whether the rest can still solve the problem, not whether the work is done. A number, not a design. |
| Concise models erode speculative-decoding and KV gains | "Does gain survive condition" shape; the useful part is folded into card 5. |
| Latent reasoning (Coconut line) versus token reasoning | Needs training; no one-week signal on this hardware. Only abstracts read. |
| CoT compression (TokenSkip, CoT-Valve) versus intrinsic token complexity | Training-heavy; 2503.01141 already frames the limit. |
| Is "Wait" productive or harmful | Largely answered by 2606.00206 and 2512.12895. |
| Does the answer phase attend to the tail | Covered by Crystal-KV and DynTS. |
| Why an injected </think> is ignored: attention to the token (2609.03633) or halt state (2608.28859) | A strong researcher would guess the halt state predicts it; medium-high confidence; smaller speed stake (it bounds early-exit savings, 1,252 -> 667 answer tokens). Recorded as a possible follow-up to card 1. |
| Which Mamba/GDN heads carry the bf16 length effect | 2611.12127 is two days old and already per-head. Residual (why Falcon-H1R is unaffected) not developed. |
| Parallel versus sequential scaling, keeping thinking across turns, length fixed by prompt, early exit on quantized models | Earlier search (Q03, Q24, Q12, Q27). |
| Oracle minimal KV budget, thought-structure selectors | Inside killed D1. |

## 4. Kept cards

Overlap with the earlier search is stated in each card. Cards 1 and 2 build on residue (a) of the earlier Q05 and share a harness.

### Card 1. Signed or unsigned?
Question: When weight quantization, KV quantization, KV eviction or bf16 state rounding makes a reasoning model think longer, is the cause a persistent signed shift of the model's state (removable by unbiased rounding or one correction vector along its own length-control direction), or does any disturbance of enough size lengthen thinking whatever its direction?
- Tension: 2606.00206 explains inflation as noise that tips sampling toward markers at high-entropy positions. 2611.12127 removes inflation with stochastic rounding, which keeps the error size and removes the bias. 2608.28859 shows a random-direction push lengthens by about 10%, and that a one-axis halt direction controls length (rho -0.82). 2609.06473 shows such directions survive quantization (cosine 0.945-0.998) but never measures whether quantization moves the state along them.
- Expert guess: persistent offsets lengthen, fresh zero-mean noise of the same norm mostly does not, and direction matters only weakly. Low confidence.
- Experiment: activation-injection arms (fresh noise, fixed random offset, fixed offset along plus and minus halt direction) at matched norms, plus real methods with round-to-nearest versus stochastic rounding; projection shift versus measured length ratio; corrective offset.

### Card 2. What does the stop decision read?
Question: When KV eviction makes a reasoning model think longer, does the extra thinking come after the answer has already formed, and does the model then fail to stop because it lost the derivation it re-reads or because it lost the record that it already checked?
- Tension: EpiKV says finished runs are as accurate as without eviction and the loss is non-termination; KV-Rescue reads the loss as missing information; BreadthKV guesses a lost check record; VaSE blames large-norm values and EpiKV reports that fix does nothing; ReCo finds 79.8% of problems longer, not only loops.
- Expert guess: mostly after the answer; the tail needs the derivation more than its own check records. Low confidence on the second half.
- Experiment: locate answer formation by forced readouts under full and evicted cache; then masked forks of full-cache traces after the answer forms (mask derivation, mask post-answer tokens, mask random).

### Card 3. Why is the hindsight stop point out of reach?
Question: Why do online early-exit rules stop so far short of the hindsight stop point: when an answer has formed, is whether it will later change already decided by the prefix (so a better signal can find it) or decided by later sampling (so no signal can)?
- Tension: hindsight 0.35-0.44x tokens (2611.02278) and oracle exit at 14.2% of budget (2610.13570) against 19-43% real savings; 736 of 1,500 trajectories abandon an agreed answer and 84.2% of those still end correct (2609.09989); probe AUCs sit at 0.6-0.7 across papers.
- Expert guess: mostly decided by the prefix. Low confidence.
- Experiment: 16 forks per prefix at the first fixed point; distribution of per-prefix flip probability; between-prefix share of variance; probe against it.

### Card 4. What sets the length of the post-answer tail?
Question: What sets how long a reasoning model keeps thinking after its answer has formed: a length budget fixed at the prompt that the tail fills, or a verification pass whose length scales with the derivation?
- Tension: total length is predictable from the question before any reasoning (0.84, 2506.08390), yet the time at which the answer forms varies across samples of one question (20.23 pp, 2610.13570). Budget-fill predicts a negative within-question correlation between time-to-answer and tail length; proportional verification predicts a positive one. The tail share ranges from 13-23% (gemma) to 75% (Qwen3-32B) with no explanation, and training halves the tail while barely touching the part before it (2611.02278).
- Expert guess: positive within-question correlation. Low confidence.
- Experiment: 16 samples per problem, forced readouts, within-problem covariance of time-to-answer and tail on three model families.

### Card 5. Is the tail easier or harder to draft? (weakest of the five)
Question: Is the thinking a model produces after its answer has formed easier or harder for a draft to predict than the derivation before it, and does the answer differ between small independent drafts and sparse-attention self-drafts?
- Tension: 2604.06613 measures higher entropy after commitment in Think models (about 1.38-1.44x); 2610.30644 measures slightly higher acceptance in the second half (+0.03 to +0.07). Different models; position is not phase; no measurement exists for sparse self-drafts.
- Expert guess: slightly easier for independent drafts, no real difference for sparse self-drafts. Medium confidence. 2610.30644 weakens this card.
- Experiment: teacher-forced acceptance of three draft types on existing traces, split at answer formation.

## 5. Prior-answer queries run (arXiv API unless noted)

- "speculative decoding" AND acceptance AND ("reasoning trace" OR "thinking phase" OR "reasoning phase") -> 2610.23818, 2608.20359
- "speculative decoding" AND overthinking -> 2509.24248
- "speculative decoding" AND "early exit" AND reasoning -> 2509.24248
- "speculative decoding" AND reasoning AND ("post-answer" OR commitment OR "redundant reasoning") -> nothing relevant
- draft AND acceptance AND "chain-of-thought" AND position -> 2610.30644
- "reasoning strength" AND (quantization OR quantized OR compression) -> nothing relevant
- reasoning AND steering AND quantization AND length -> 2609.06473
- "stochastic rounding" AND reasoning AND length -> 2611.12127
- quantization AND reasoning AND bias AND "stochastic rounding" AND KV -> nothing
- "end-of-thinking" / termination mechanism -> 2609.03633, 2608.28859
- KV AND reasoning AND eviction AND longer -> 2610.20641, 2603.11504, 2512.12008, 2512.07993
- "KV cache" AND reasoning AND (re-derive OR recompute OR re-verification OR loop) -> 2606.03928
- epiphany AND "KV cache" -> 2606.26472; KV AND rescue AND reasoning AND eviction -> 2608.15797
- reasoning AND KV AND eviction AND answer AND verification AND stop -> nothing
- "reasoning models" AND loop AND (repetition OR looping) AND why -> 2512.12895, 2607.11317
- length AND "reasoning models" AND quantization -> 2606.00206, 2606.25519, 2610.05685, 2609.39801
- "post-commitment" OR ("after the answer" AND "reasoning tokens" AND verification) -> 2604.06613, 2605.11467, 2605.11746
- "early exit" AND reasoning AND (aleatoric OR irreducible OR branching OR resampling) AND answer -> 2608.27963, 2610.25970
- "chain-of-thought" AND resampling AND counterfactual -> nothing (2506.19143 found by id)
- "reasoning length" AND "within-problem"; thinking tokens AND variance AND same prompt -> nothing
- thinking tokens AND (verification phase OR self-verification) AND length AND (proportional OR budget) -> nothing

## 6. Not done

- Semantic Scholar citation scans (429). WebSearch not used.
- Latent-reasoning and CoT-compression papers read at abstract level only.
- EpiKV beyond the first 100,000 characters (its Appendix E.1 loop analysis was not read and may locate loop onset; check before starting card 2).
- KV-Rescue, VaSE, Thought Anchors, "Why Do Reasoning Models Loop", SABER, GLOBE: abstracts only.
- Non-arXiv venues (OpenReview-only papers, blog posts) were not searched.
