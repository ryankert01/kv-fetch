# Question harvest: speculative decoding with a separate or attached drafter

Date: 2026-11-11. Read-only literature work; no GPU jobs were run.
Scope: draft models, EAGLE family, block and semi-autoregressive drafters, trees versus chains, acceptance theory, lossy verification, batch behaviour.
Result: four question cards kept, fourteen tensions dropped.

"Verified" below means I read the sentence or number in the arXiv HTML or the arXiv API abstract in this session. Everything else is marked unverified.

## 1. Reading list

Body sections read (arXiv HTML):

| id | short name | what I read |
|---|---|---|
| 2610.24993 | drafter recipe comparison (six recipes, Qwen3-8B) | abstract, Table 1, entropy and predecessor analysis, concurrency |
| 2610.23818 | Acceptance Is Not Speed (relaxed verification lengthens completions) | abstract, Tables 1 and 3 text, temperature control, guard appendix, limitations |
| 2512.11718 | Speed-of-Light bound for deterministic drafting | abstract, Table 1, experiments, limitations |
| 2610.11750 | fixed-tree ceiling, ExitTrain, TEV | abstract, ceiling statement, limitations |
| 2503.01840 | EAGLE-3 | Table 1 |
| 2602.06036 | DFlash | Tables 1-2, thinking-mode rows, SGLang concurrency |
| 2608.30427 | DBloom | block-length analysis, EOS remark |
| 2610.07822 | Nucleus SD | abstract, metric definition, rescued-token analysis |
| 2610.10929 | Speedbumps (speculative rejection attacks) | sections VI-C, VI-E1 |
| 2611.10432 | D-REAL | abstract, baseline +1 epoch result |
| 2501.19309 | Judge Decoding | abstract, motivation |
| 2402.01528 | Decoding Speculative Decoding | abstract |
| 2611.04055 | HybridSpec | verification latency numbers |
| 2610.38981 | MoE verification better than predicted | efficiency numbers |
| 2601.11580 | Speculative Decoding: Performance or Illusion? (vLLM study) | abstract, section 8 oracle definition |
| 2604.14682 | Acceptance Dynamics Across Cognitive Domains | abstract, setup, entropy section |
| 2610.41177 | TraceSpec | context-length finding |

Abstract read (arXiv API): 2611.14399 REAL Loss; 2611.13017 Multi-Future Drafting; 2611.12464 Phrase-Level SD; 2611.05611 Hybrid Spark; 2611.00588 TVP; 2610.40287 energy accounting; 2610.34440 MoE expert bytes; 2610.29210 target probability reuse; 2610.29013; 2610.28355 D-PAL; 2610.12939; 2609.38795; 2609.27396 DPara; 2609.22156 Limits of Speculation; 2608.26112 TreeGraft; 2608.20359 SSR; 2607.21804 ADSD; 2607.12696 EcoSpec; 2607.12422 PEFT-BD negative result; 2607.05147 DSpark; 2606.30265 When Is a Draft Accepted; 2606.25097; 2606.18394 JetSpec; 2606.07710 WhiFlash; 2606.02091 DFlare; 2604.09557 SPEED-Bench; 2604.05417 MetaSD; 2602.05774 VSD; 2602.03708 SemanticSpec; 2601.15498 MARS; 2510.22876 batch SD correctness; 2408.11049 MagicDec; 2406.14066 SmartSpec; 2402.12374 Sequoia; neighbours outside the area: 2606.00206, 2611.12127, 2609.03633, 2608.28859.

Title seen only (exists in arXiv API listing, not read): 2611.13442 ForeSpark; 2611.12356 DForkTree; 2611.11371 D3Tree; 2610.38968 TableTune; 2610.07350 TLAR; 2610.02800 BitNest; 2605.29727 Bastion; 2605.18810 D-PACE; 2604.12989 DDTree.

Seen only in web search results, not opened: 2602.23881 LK Losses; 2504.07891 SpecReason; 2506.19830 Lookahead Reasoning. Not cited as evidence.

Not checked, so not cited: SpecTr, BASS, Medusa, Hydra original papers.

## 2. Extracted claims (all verified unless stated)

Drafter convergence and plateau
- 2610.24993: "On MT-Bench, the maximum-minus-minimum speedup gap falls from 1.36 in published results to 0.85 after remeasurement and 0.23 after retraining, in speedup multiples. The corresponding GSM8K gaps are 4.60, 2.84, and 1.53."
- 2610.24993 Table 1, Qwen3-8B, budget 16, target T=1 with greedy draft (1|0), high-entropy tasks, accepted length: DFlash 3.19, DFlare 3.29, D-PACE 3.34, Domino 3.56, DSpark 3.55, DCore 3.37; EAGLE-3 tree 4.33; EAGLE-3 chain 2.88. At 0|0: 4.03 to 4.83. At 1|1: DFlash 3.09, Domino 4.09, DSpark 4.18. Targets run "with thinking disabled".
- 2610.24993: "Across Eagle-3, DFlash, and DSpark, accepted prefixes tend to have lower target entropy than the rejected position ending the path." No oracle or ceiling is computed in the paper.
- 2610.24993: "After a non-argmax predecessor, block-parallel acceptance falls 0.38–0.43 below the rate after an argmax predecessor, against 0.08–0.09 for semi-AR recipes".
- 2611.10432: "Baseline +1 epoch yields minimal macro-average changes (Δ≤0.03 tokens) across all configurations, indicating that standard token-level training has largely plateaued"; D-REAL adds "up to 0.23 tokens".
- 2606.02091: "We further scale training data from 800K to 2.4M samples"; improvement over DFlash "roughly 11%, 8%, and 5%".

Ceilings
- 2512.11718: bound "E[X] ≤ (μ+ μ(2))log(P)/μ^2 + O(1)"; "it remains unclear how close existing algorithms are to the fundamental limits"; "Generally, this suggests that about ×2 difference between the theoretical bound and EAGLE-3 is caused by a speculation error. This suggests that there is still significant room for improvement"; and "This suggests that current deterministic algorithms are approaching the fundamental limits under the assumptions studied".
- 2512.11718 limitations: "The analysis depends on Assumption 2, which states that the distributions of acceptance probabilities are independent and identically distributed across prefixes. This is not exact for language models"; "the analysis is focused on deterministic drafting and does not address stochastic speculation strategies".
- 2512.11718 Table 1 (T=1.0, 80 samples per dataset), mean μ: Llama-3 8B 0.683, Llama-3 70B 0.153, DeepSeek-R1-distill-Llama-8B 0.554, Qwen3-8B 0.319. Per task: L3-8B MT-bench 1.088, GSM8K 0.636; R1-distill MT-bench 0.633, GSM8K 0.233, HumanEval 0.530.
- 2503.01840 Table 1, T=1, EAGLE-3 mean accepted length: Llama-3.1 8B 4.92, Llama-3.3 70B 5.66, R1-distill 8B 4.89. So a 4.5 times difference in μ between the 8B and 70B targets goes with a 15% difference in accepted length.
- 2610.11750: "one plus target coverage sharply bounds the expected output-block length, including the bonus token, of any exact path verifier". The bound is conditional on the tree the drafter actually built; the paper does not compute the best tree the target would allow. "Our systems results are limited to batch-one decoding with DDTree".
- 2601.11580: its upper bound is a different object. "The oracle speedup represents an upper bound in which the accepted length at each decoding step is known in advance, and the system proposes exactly that many tokens." It removes wasted verification; it does not ask how long an accepted run the target permits.
- 2606.30265 (abstract): certificates for greedy and relaxed acceptance in terms of target margin; "relaxed and tree-based criteria substantially enlarge the region of certified acceptance, especially on decoding steps with low target model distribution margin".
- My own arithmetic, not from any paper: plugging Table 1 values into the abstract's formula with a natural log gives numbers far above the "×2" the paper reports, so I do not trust my reading of the constants and do not use the arithmetic as evidence.

Thinking versus answer text
- 2602.06036 Tables 1-2, Qwen3-8B, T=0, accepted length: thinking disabled MATH-500 7.87, AIME25 7.08; thinking enabled MATH-500 5.82, AIME25 5.74 (the thinking drafter is "trained on target-model outputs with reasoning traces").
- 2610.24993, 2608.30427, 2610.41177 all evaluate with thinking disabled.
- 2606.07710 (abstract): "current approaches rely on static drafting paradigms, utilising either autoregressive drafting models for reasoning or diffusion-based parallel drafting models for structured outputs. We empirically find that drafting accuracy fluctuates dramatically within a single sequence".
- 2611.13017 (abstract): "no product-ranked tree can hold two futures that disagree in m positions without 2^{m-1}-1 chimeras ... in practice chimeras make up 27-33% of a 64-node tree".
- 2604.14682 (abstract; TinyLlama-1.1B drafting Llama-2-7B-Chat, 64-token generations): "the entropy-acceptance correlation is consistently negative but weak across all domains (rho in [-0.20, -0.15]). Counterintuitively, chat produces the highest entropy yet the highest acceptance rate. We attribute this divergence to the lexical predictability of RLHF-aligned register."

Lossy verification and completion length
- 2610.23818: "relaxed verification on average accepts 1.34× as many tokens per round but produces 1.54× longer completions".
- 2610.23818: "Relaxation lengthens completions in 248 of the 360 cells (Tables 1 and 3): λ has median 1.08 and mean 1.54 ... Its median varies more with the rule (0.97 for spec_casc_tok to 1.39 for cactus) and the drafter (0.94 for standalone drafters to 1.23 for EAGLE heads) than with the task (1.04 on GSM8K to 1.14 on HumanEval)".
- 2610.23818: "The relaxed-only tokens are 7% to 24% of the committed draft tokens. For the four rules other than spec_casc_tok they have a median target probability of 0.01 to 0.09 (shared tokens: 0.78 to 0.93) and a mean target rank of 37 to 234."
- 2610.23818: "A lossless control agrees: raising the sampling temperature from 1.0 to 1.2 ... lengthens GPT-OSS-20B completions by 1.27× on GSM8K and 1.07× on LiveCodeBench, and T=1.5 inflates GSM8K by 3.9× with accuracy 0.31"; but "Qwen3-8B behaves differently: its completions barely lengthen at T=1.2 or 1.5 (λ 1.02 to 1.04 ...), but its acceptance falls faster".
- 2610.23818: "At Qwen3-8B's recommended sampler (temperature 0.6), mentored_dec, cactus, and spec_casc_tok stop inflating on GSM8K and LiveCodeBench, but spec_casc_opt still inflates by 1.5×".
- 2610.23818: "They do not show why an unlikely token makes a reasoning model think more rather than less."
- 2610.23818: "On GSM8K, 50% (GPT-OSS-20B) and 82% (Qwen3-8B) of the extra tokens come after it [the first reference answer]"; a per-token guard "cut the eight cases from 13 364 to 10 534 tokens ...; on all 30 cases it moved the mean from 9 270 to 10 363 tokens"; "A guard judges one draft token at a time and cannot see whether a run will ever close its trace".
- 2610.23818 does not cite 2606.00206 and does not test token class or position.
- 2610.07822: "over half of the rejected tokens lie inside the target's own nucleus"; "The tokens rescued by NSD are highly plausible under the target: 94.1% rank within its top five, and 86.7% lie in the core"; "Speedup Ratio is the ratio of each method's token throughput to target model autoregressive (AR) throughput", so completion length is not in its metric.
- 2606.00206 (abstract): "Positions with high KL divergence correlate strongly with high next-token entropy, and at these positions quantized models disproportionately sample overthinking markers such as \"wait\", \"but\", and \"alternatively\"."
- 2611.12127 (abstract): same-width stochastic rounding "removes the length increase in four of five affected models", "so the failure is mostly rounding bias, not missing resolution".
- 2609.03633 (abstract): an injected end-of-think token "does not always induce a clean answering phase".

Acceptance versus what the target writes
- 2501.19309: "even powerful draft models such as GPT-4o, as well as human text cannot achieve high acceptance rates under the standard verification scheme".
- 2402.01528: "the draft model's capability in language modeling does not correlate strongly with its performance in speculative decoding".
- 2610.10929: "Speedbump-P reduces speed-up by 12.2–36.8% under greedy decoding"; "Excluding SpS/D, the configurations recover 70–93% of their accuracy loss between β=0 and β=1, but lose 66–95% of their MAL reduction"; "indicating that most of the degradation comes from changing what the target generates. This conclusion is limited to one penalty on one task"; "The SpS/P suffixes transfer well because they act on the target's output rather than on the drafter".
- 2607.21804 (abstract): "On the GSM8K dataset, our attack increases the mean sample time by 62.3% while preserving the task quality."

Batch and system side (used only to decide what to drop)
- 2610.24993: "Under 1|0 on the larger target, budget-8 non-AR recipes retain about 1.43–1.50× speedup at C=64, compared with 0.98–1.10× for budget 16".
- 2610.40287 (abstract): "The saving shrinks as the batch grows, because batching has already shared the weight read".
- 2611.04055: verification latency "grew by 71% from length 4 to 16 ... compared with 4%-6% for a dense Qwen3-4B target".
- 2610.38981: "the MoE L2 hit rate rises from 5.72% in AR decoding to 27.79% under SD verification".

## 3. Tensions considered and dropped

1. Do speedups survive batch or concurrency. Dropped: the roofline argument predicts the answer, 2610.40287 states the mechanism, 2601.11580 measured it on vLLM. Same flaw as the earlier search.
2. Which training loss is best. Dropped: 2610.29013 and 2610.24993 already retrain under one protocol and the gaps collapse.
3. Tree versus chain shape. Dropped: Sequoia, 2512.11718 Lemma 1, MFD and DDTree cover it; the remaining question is the ceiling (Card 2).
4. Adaptive draft length. Dropped: DSpark, SmartSpec, TableTune, WhiFlash and the oracle in 2601.11580 already address it.
5. Why block drafters lose more at T=1. Dropped: explained in 2610.24993 (non-argmax predecessor).
6. TraceSpec context-length trend. Dropped: the paper explains it as a composition effect.
7. Quantized self-drafting. Dropped: BitNest exists; expected answer is predictable.
8. Verification kernels for hybrid models. Dropped: HybridSpec; close to killed directions D2 and D3.
9. MoE verification cost disagreement (2610.38981, 2610.34440, 2611.04055, EcoSpec, 2609.22156). Dropped: different engines and kernels; the likely answer is "depends on the kernel"; close to D3.
10. Do all drafters fail on the same tokens at T=0. Dropped: complementarity is already shown (TreeGraft +15.1%, WhiFlash, Oracle_Combine in 2601.11580).
11. Rejections caused by floating-point noise. Dropped: my estimate is under 2% of rejections; predictable.
12. Is relaxed verification the same saving as lowering the temperature. Dropped as a standalone question: Nucleus SD's 1.85× average over lossless SD is far larger than the T=1 to T=0 gain, so "not the same" is predictable. The useful part is inside Card 1.
13. Sampled versus deterministic drafts relative to the bound. Folded into Card 2's experiment.
14. "Draft model quality does not track acceptance" (2402.01528). Too loose as a question; used as evidence in Card 4.

## 4. Kept cards

### Card 1. What sets the length change under lossy verification: a surprisal dose, or where and what the admitted tokens are?
- Type: unexplained anomaly. Builds on residue (a) of the earlier search, but with a causal token-level intervention.
- Evidence: 2610.23818 (1.34× tokens per round, 1.54× longer; rule medians 0.97 to 1.39; standalone drafters 0.94 versus EAGLE heads 1.23; GPT-OSS-20B lengthens with temperature, Qwen3-8B does not yet still inflates under relaxed rules; 82% of extra GSM8K tokens come after the first correct answer; per-token guard fails off its tuning cases; "They do not show why..."). 2610.07822 (rescued tokens 94.1% top-five; throughput is per token). 2606.00206 (quantization: markers at high-entropy positions).
- Expert guess: mostly a dose effect of target surprisal; admissions inside the nucleus are harmless. Medium confidence.
- Why the guess may be wrong: Qwen3-8B is flat under temperature but inflates under relaxed rules; the drafter type matters more than the task; most extra tokens come after the answer is already on the page.
- Existing answers searched: arXiv API queries on "length inflation", lossy or relaxed speculative decoding with reasoning, reasoning-length with perturbation, injection or intervention, termination and end-of-think. Closest: 2610.23818 (documents, does not explain), 2606.00206 (quantization only), 2611.12127 (bias versus resolution), 2609.03633, 2608.28859. No token-injection dose-response study found.
- First experiment: no drafter needed. A logits processor on Qwen3-8B (thinking on) and GPT-OSS-20B overrides the sampled token at a controlled rate with a token of chosen target rank, with arms by position (before or after the first correct answer), by class (marker words versus rank-matched other words), an arm that never overrides when the close-of-thinking token has high probability, an arm that only suppresses the closing token, and a temperature arm matched in mean realized surprisal. GSM8K and AIME24, 200 problems per arm. About 3 GPU-days. Stop if length is a single smooth function of mean realized surprisal across all arms within noise (that is answer A, and it is still a result).
- Answer A (dose): one offline number bounds length inflation for any rule and drafter; build a surprisal-neutral rule that admits more draft tokens while lowering the residual sampling temperature.
- Answer B (position or class): build a verifier that is strict only around termination decisions and marker tokens and loose elsewhere.

### Card 2. Is the drafter plateau the target's own ceiling or a shared training limit?
- Type: missing limit, with a contradiction inside one paper.
- Evidence: 2512.11718 says both "about ×2 difference ... caused by a speculation error ... significant room for improvement" and "approaching the fundamental limits", under an i.i.d. assumption it calls "not exact"; 2610.24993 shows six recipes converge (gap 1.36 to 0.23, accepted length 3.19 to 3.56); 2611.10432 says training "has largely plateaued"; EAGLE-3 gets 5.66 on a 70B target and 4.92 on an 8B target although μ differs 4.5 times; 2610.11750's ceiling is conditional on the drafter's tree; 2601.11580's oracle removes wasted verification only.
- The object nobody has computed: at T=1 with a deterministic draft, a drafted token is accepted with probability equal to the target's probability of that token. So the best chain is the target's most probable continuation and the best N-node tree is the N most probable target prefixes. Both can be computed from the target alone, per context, with no i.i.d. assumption.
- Expert guess: drafters reach 50 to 70% of this ceiling on high-entropy text, more on chat, less on large low-entropy targets. Low to medium confidence.
- Existing answers searched: arXiv API queries on ceiling, headroom, fundamental limit, oracle, optimal draft tree, target entropy; Semantic Scholar citations of 2512.11718 (one survey); one web search. Closest: 2512.11718, 2610.11750, 2601.11580, 2606.30265, 2608.30427.
- First experiment: Qwen3-8B with released DFlash, DSpark and EAGLE-3 checkpoints, MT-Bench and GSM8K, thinking off then on. At about 300 sampled anchors per task compute the target's best chain and best-first tree for budgets 8, 16 and 60 by extra target forward passes; compare with the realized accepted length at the same anchors; label each rejection by the target's top probability at that position. Also compare the 1|1 accepted length with the deterministic ceiling. About 1.5 GPU-days. Stop if drafters already use more than about 85% of the ceiling on every task: then the result is "saturated" and the design shifts to sampled or coupled drafts.
- Answer A (near ceiling): more drafter training is wasted on that text; work on sampled or coupled drafting, target-shaped trees, or lossy rules.
- Answer B (far below): the six recipes share a training limit; find which positions (confident target, drafter wrong) carry the loss and train for those.

### Card 3. Why is thinking text harder to draft than answer text?
- Type: untested assumption plus unexplained anomaly.
- Evidence: 2602.06036 (MATH-500 5.82 with thinking versus 7.87 without; AIME25 5.74 versus 7.08, even with a drafter trained on reasoning traces); all 2026 comparisons disable thinking; 2512.11718 Table 1 (R1 distill has lower μ than its Llama base on GSM8K, 0.233 versus 0.636, but higher on HumanEval); 2606.07710 asserts AR drafters suit reasoning and block drafters suit structured output; 2611.13017 chimeras; 2604.14682 finds entropy explains little of acceptance.
- Expert guess: mostly irreducible; self-talk has more near-tied positions. Medium confidence.
- Why the guess may be wrong: at T=0 there is no irreducible part by definition, yet the gap is measured at T=0; the weak entropy correlation in 2604.14682; the R1-distill row suggests thinking lowers μ on math, which should make drafting easier, not harder.
- Existing answers searched: arXiv API query on speculative decoding with thinking or reasoning trace and acceptance; one web search. Closest: 2602.06036 (numbers only), 2604.14682 (domains, no thinking models), 2602.03708 and 2608.20359 (methods, no diagnosis).
- First experiment: same instrument as Card 2. Qwen3-8B thinking on, the same MATH-500 problems, split positions into think and answer segments; for each segment report the target top-probability distribution, the chain ceiling, the realized accepted length of both released DFlash drafters and EAGLE-3 at T=0 and T=1, and the share of rejections at near-tied positions. About 1 GPU-day. Stop if the two segments have the same ceiling-use ratio (the gap is then fully explained by target entropy).
- Answer A (irreducible): lossless drafting has a low ceiling where most tokens are; the lever is step-level or lossy acceptance, which links to Card 1.
- Answer B (reducible): thinking-specific drafter data, or routing between AR and block drafters by segment.

### Card 4. Can acceptance be raised from the prompt side without changing answer quality?
- Type: paper A's explanation predicts P, paper B's data shows not-P.
- Evidence: 2501.19309 and 2610.07822 say most rejections are of valid continuations, so acceptance has slack unrelated to quality; 2604.14682 attributes high acceptance in chat to register, not entropy; 2610.10929 says holding the target's output fixed gives up 66–95% of the acceptance change and that the effect "act[s] on the target's output"; 2607.21804 claims a 62.3% slowdown "while preserving the task quality".
- Expert guess: only small gains at equal quality. Low confidence.
- Existing answers searched: arXiv API queries on system prompt, prompt tuning or soft prompt with acceptance; fine-tune the target, draft-friendly, co-train; paraphrase, surface form, style; one web search. Closest: 2610.10929 and 2607.21804 (downward only), 2501.19309, 2610.07822, 2601.15498. No study of the upward direction found.
- First experiment: Qwen3-8B with a public EAGLE-3 or DFlash drafter. (1) hand-written style and format system prompts; (2) a soft prompt optimised for expected accepted length with a KL penalty to the clean target distribution, which is the Speedbumps objective with the sign flipped. Measure accepted length, accuracy, completion length, and time per solved problem. About 4 GPU-days. Stop if the penalised optimum keeps less than a third of the unpenalised gain, or if gains come from repetitive output.
- Answer A (steerable): ship a per-drafter acceptance-tuned prefix; draftability becomes a training target for the target model.
- Answer B (pinned): the slack is reachable only from the verification side; supports the ceiling view in Card 2.

## 5. Searches run

- arXiv API (export.arxiv.org), about 40 queries across: speculative decoding 2026 listings; acceptance ceiling, oracle, fundamental limit; lossy or relaxed verification and reasoning length; thinking or reasoning-trace acceptance; prompt or soft-prompt and acceptance; batch speculative decoding; MoE verification.
- Semantic Scholar citations endpoint for 2512.11718: one citing paper, a survey.
- Web search: 3 of 6 calls used (thinking-versus-answer acceptance split; prompt optimisation for draft-friendly output; oracle upper bound versus trained drafters).

## 6. Not reached

- Bodies of MagicDec, SmartSpec, Sequoia and the batch-correctness paper were read at abstract level only.
- SpecTr, BASS, Medusa, Hydra were not checked and are not cited.
- The batch and goodput sub-area produced no card: every tension there had a predictable answer.
- The Card 2 ceiling for trees assumes verification by sampling without replacement over deterministic children; I did not check every tree verifier against that assumption.
