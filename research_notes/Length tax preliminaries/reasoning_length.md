# Tutor pack: how reasoning models decide length

Subskill id: `reasoning_length`. Written 2026-11-13 15:17 CST. Read-only work; no GPU job was run. A second reader examined and corrected the pack on 2026-11-13; see the Verification log at the end.

Scope: thinking traces and the end-of-thinking decision; where extra tokens go (before or after the answer first appears); length caps and truncated answers; early-exit rules and the hindsight stop point; why probes of correctness mostly measure problem difficulty; per-token cost versus lasting damage as two models of length inflation.

Notation in this file:

- **L** is the number of tokens one run generates. **EoT** is the end-of-thinking token (`</think>` for Qwen and DeepSeek style models; for GPT-OSS it is the switch from the analysis channel to the final channel).
- **λ** is always length inflation: mean length of the lossy arm divided by mean length of the exact arm. "Exact" means lossless decoding of the unchanged model (the speculative decoding papers say "strict" or "lossless").
- Three sources use the letter λ for other things. I rename them here: **τ** is DEER's confidence threshold, **w** is the share of same-problem pairs in the AUROC formula, **β** is the logit penalty. Always ask "which λ?" when the learner quotes a source.
- **p** is the target model's next-token distribution and **q** is the drafter's, as in the sibling pack `specdec_exact.md`.

How to read the tags:

- **[checked]**: I read the fact in this run in a saved copy of the arXiv HTML page (the copies were saved on 11 Nov 2026 in the scratch folder). No network request was made in this run. Each arXiv id was confirmed against the title and the arXiv stamp inside the saved page.
- **[from notes]**: taken from project notes (`reports/length-tax.html` or `research_notes/Inference speed question search/`). I did not re-read the source in this run.
- **[unverified]**: I could not trace it to a source text. (Not used for any number below; everything not checked is listed in Section 7.)
- No tag: a textbook fact, a derivation done here, or a toy model defined here.

---

## 1. Core ideas

1. **A trace has two parts and one switch.** A reasoning model writes a thinking block, then an EoT token, then the answer, so L = L_think + L_answer. EoT is an ordinary vocabulary token, so "when to stop thinking" is decided by the same next-token sampling as every other token.

2. **Length as a stop chance per token (the toy model of this file).** Let h_t = P(EoT at token t | no EoT before t). Then P(L_think > t) = ∏_{s=1..t} (1 − h_s) and E[L_think] = Σ_{t≥0} P(L_think > t). With a constant h the length is geometric with mean 1/h, so anything that lowers h_t makes traces longer.

3. **Length inflation and the rounds it costs.** λ = L̄_relax / L̄_strict (mean generated tokens of the relaxed and the lossless arm). The number of target passes is R ≈ L/(ℓ̄ + 1), with ℓ̄ = mean accepted draft tokens per round, so R_relax / R_strict ≈ λ · (ℓ̄_strict + 1)/(ℓ̄_relax + 1). A relaxed rule saves rounds only if (ℓ̄_relax + 1)/(ℓ̄_strict + 1) > λ, and no rule can pay for a λ above (N_draft + 1)/(ℓ̄_strict + 1), where N_draft is the draft length. [checked: 2610.23818 Sec 2.2, paragraph "Verifier rounds", Eq 4-5]

4. **Inflation is common, and it is thinking.** Over 360 cells (5 relaxed rules × 6 datasets × 12 model pairs, temperature 1.0, one run per case) outputs get longer in 248 cells; λ has median 1.08 and mean 1.54; 114 cells need more target rounds than lossless. On GPT-OSS-20B the analysis channel gets at least 92% of the extra characters in 25 of 30 cells; a model with no thinking trace (Llama-3.1-8B-Instruct) lengthens the answer itself. [checked: 2610.23818 Sec 3.3, 3.4, 3.7]

5. **A few runs carry most of the extra length.** On the two primary pairs, the tenth of cases with the largest increase holds 59% of the net extra tokens (median over cells). It is mostly not exact looping on GPT-OSS-20B (at most 2.9% of tokens sit in a repeated 50-gram), but Qwen3-8B does loop under one rule (spec_casc_opt: 15% of its AIME24 tokens). [checked: 2610.23818 Sec 3.7]

6. **Where the extra tokens go depends on the task.** On GSM8K, 50% [37, 59] (GPT-OSS-20B) and 82% [65, 96] (Qwen3-8B) of the extra tokens come after the answer first appears. On AIME24 the relaxed GPT-OSS-20B runs reach the answer later (median token 3,865 against 1,721) and 59% of the extra tokens come before it (41% [28, 54] after). For Qwen3-8B on AIME24 the estimate is too wide to use (−16% [−177, 49] after, from 33 pairs). All four shares use only pairs where both runs are correct and not cut off. "First appears" means the first time the reference answer shows up as a standalone number; this is a proxy, because an intermediate result can equal the answer. [checked: 2610.23818 Sec 4; App F, Table F2]

7. **The hindsight stop point.** Every 2,048 thinking tokens, close the thinking and read out a forced answer; call the readouts r_1 … r_k and the trace's final answer a. The answer is "fixed" at the first r_j such that r_j, …, r_k all equal a (at the end of the trace if r_k ≠ a, or if the trace is too short to have a readout). Tokens after that point are slack: 54.0% of all thinking tokens on the hard set and 63.8% on the confirmation set for Qwen3.5-9B. [checked: 2611.02278 Table 4]

8. **The hindsight point is a bound, not a rule.** It uses the final answer of the trace, which a running system does not have. Rules that can run lose accuracy: "stop when 3 readouts in a row agree" costs 2.4 to 9.2 points on Qwen3.5-9B. On the hard set this rule uses about the same tokens as the bound (0.45× against 0.44× of the full trace), so its stops are in the wrong places, not only late. [checked: 2611.02278 Table 5]

9. **DEER, a typical early-exit rule.** At a transition point (a marker such as "Wait", or a high-entropy first token after a blank line) DEER asks for a trial answer A = (a_1 … a_n) and computes C = (∏_{i=1..n} max_{v ∈ V} p(v | P, T, I, a_<i))^{1/n}. Here P is the prompt, T the thinking so far, I the answer-inducing prompt, V the vocabulary and a_<i the trial-answer tokens before step i. It exits if C > τ, with τ = 0.95. The paper reports 19.1% to 42.9% shorter CoT with accuracy up 0.9 to 4.8 points; a later reimplementation by another group (greedy decoding, R1-Distill-Qwen-7B, same threshold 0.95) loses 6.7 points on AIME24. [checked: 2504.15895 Sec 3.2 Eq 4, Sec 4.2; 2608.28859 Table 3 and its text]

10. **Writing EoT does not always stop the thinking.** After an injected EoT, reasoning-like text can go on inside the answer phase until the model writes EoT again. For R1-Distill-14B with DEER on MATH this is not seen after a natural EoT, and it reaches about 40% of samples when the exit removes 90-100% of the trace [checked: 2609.03633 Sec 4.1 (definition), Sec 4.2, Fig 2]. "Never after a natural EoT" is a result for that setting, not a law: under relaxed speculative decoding, cactus on Qwen3-8B reopens a think block that the model had closed in 11 of 30 AIME24 outputs [checked: 2610.23818 Sec 3.7]. In the same study, 515 runs with a forced closing token still reach the budget, because the model starts reasoning again or rewrites its solution [checked: 2610.23818 App F].

11. **A length cap hides inflation.** With a cap C we only see min(L, C). The measured ratio E[min(L_lossy, C)] / E[min(L_exact, C)] is pulled toward 1; for a geometric length E[min(L, C)] = (1 − (1 − h)^C)/h. The budget-hit share rises in 208 of 360 cells (median 11% against 4% lossless) and the authors call their λ a lower bound [checked: 2610.23818 Sec 3.3, 3.7, Limitations]. In this project's own AIME log, the full Qwen3-4B model hits the 16,384 cap in 41% of samples [from notes: reports/length-tax.html].

12. **A cut-off run is scored wrong, so length turns into accuracy loss.** In 2610.23818 a run that is cut off before it closes its thinking is scored wrong. The authors took the 1,132 relaxed runs that were still thinking at a threshold near the budget and forced the closing token there. Correct answers rose from 126 to 439 [checked: 2610.23818 Sec 4, Table 4; App F]. Under KV eviction on MATH-500 the extra errors are almost all cap hits: errors 104 → 172, cap hits 55 → 124, finished-but-wrong 49 → 48 (H2O against no eviction, R1-Distill-Qwen-7B, cache of 1,024 entries, 495 problems, greedy) [from notes: `harvest_reasoning_efficiency.md`, which cites 2606.26472 Table 3].

13. **Pooled AUROC is a mixture.** For a score S, AUROC(S) = P(S_a > S_b) + ½·P(S_a = S_b) over pairs where run a is correct and run b is incorrect. AUROC_pooled = w · AUROC_within + (1 − w) · AUROC_between, where w is the share of such pairs whose two runs come from the same problem (the source writes λ for w). A score that depends only on the question is tied inside each problem, so its AUROC_within is exactly ½. [checked: 2609.03436 Sec 4.2, Eq 1]

14. **Measured: the probe signal is problem difficulty.** A hidden-state probe at thinking token 4 has pooled AUROC 0.849 [0.735, 0.919] (1,024 traces, 8 per problem). Inside problems it is at chance: 0.496 [0.466, 0.527] with the pre-registered weighting and 0.515 [0.481, 0.562] with exact pair weighting (a different sample set: 22 problems with both outcomes, 256 traces each, 5,632 samples). The pooled value itself falls to 0.586 at token 16 and about 0.55 from token 32 on. A score that never reads the trace (the pass rate of the other attempts on the same problem) reaches 0.873 on 192,315 DeepSeek-R1 generations. [checked: 2609.03436 Sec 4.2-4.4]

15. **Length follows the problem too.** Across problems, longer traces are less often correct; part of this is difficulty, because hard problems give long traces. Inside one problem, when two traces disagree in correctness, the shorter one is the correct one in 60% [54, 65] of 387 pairs (three traces per hard-set problem, base model). The effect is small: on the development suite Qwen3.5-9B gives 54% [48, 60], an interval that includes 50% [checked: 2611.02278 Sec 5.1]. A linear probe on the question's activations predicts thinking length with correlation 0.84 before any thinking token is written; this is a correlation between questions, on R1-distilled Qwen models [from notes: `harvest_reasoning_efficiency.md` and `verification.md`, which cite 2506.08390].

16. **Where a small error turns into length: hesitation points.** Under 3-bit AWQ, the KL divergence between the full-precision and the quantized next-token distribution is largest where the full-precision model is itself unsure (correlation ρ = 0.92 with its entropy), and the tokens there are hesitation words such as "Wait", "But", "Alternatively". Subtracting a constant β > 0 from the logits of a set S of 50 such tokens, z′_t(v) = z_t(v) − β·1[v ∈ S], cuts CoT length by 12% to 23% on average. Accuracy is kept or improved on average; a few cases lose a little (five models, three quantization methods, five benchmarks). On R1-Distill-Qwen-1.5B (MATH-500), going from BF16 to 3-bit AWQ raises errors from 72 to 265, and errors where the trace reached the correct answer but did not commit from 19 to 139. The paper calls its account an intuition for one failure mode, not a full causal explanation. [checked: 2606.00206 abstract, Sec 4.2, 4.3, 5.1, 6]

17. **Which tokens does a relaxed rule add?** Lossless verification already accepts every draft token x with q(x) ≤ p(x), so a relaxed rule only adds tokens the drafter over-rates, q(x) > p(x). In traced GPT-OSS-20B runs these relaxed-only tokens are 7% to 24% of the committed draft tokens. For four of the five rules their median target probability is 0.01 to 0.09 (tokens accepted under both rules: 0.78 to 0.93). The fifth rule, spec_casc_tok, adds tokens near the top of the target's ranking (median probability 0.29). The paper says its logs "do not show why an unlikely token makes a reasoning model think more rather than less". [checked: 2610.23818 Sec 3.7, Table B2]

18. **Dose is not the whole story, and exact sampling is not a zero.** Exact decoding at temperature 1.2 lengthens GPT-OSS-20B on GSM8K by 1.27× [1.10, 1.47] with no relaxation at all, but Qwen3-8B barely lengthens at 1.2 or 1.5 (λ 1.02 to 1.04) [checked: 2610.23818 App C]. With the same target, Llama-3.1-8B-Instruct inflates 3.30× with its EAGLE-3 head and 1.10× with a standalone Llama-3.2-1B drafter (mean of λ over all cells of each pair) [checked: 2610.23818 Sec 3.6, Table 3]. Numerical noise alone (GPU count, GPU type, batch size) can change the response length by up to 9,000 tokens under greedy bf16 decoding of R1-Distill-Qwen-7B [checked: 2506.09501 abstract].

19. **Two models of inflation.** *Per-token cost*: h_t depends only on whether step t is lossy (h_lossy < h_exact), so after a switch back to exact the remaining length has the exact-run distribution. *Lasting damage*: each lossy token sets a hidden flag D = 1 with chance d, and a flagged trace has a low stop chance h_D in every mode, so flagged runs stay long after the switch. With one flag, the fully lossy mean is E[L] = 1/a + (x/a)·(1/h_D), where x = (1 − h_exact)·d is the chance per token to get flagged and a = h_exact + x is the chance per token to leave the healthy state. (Toy model of this file. None of the sources read for this pack reports a switch-back test.)

20. **Hints toward lasting damage, none decisive.** The heavy tail (idea 5). A per-token guard (lossless verification for eight positions after clusters of hesitation markers, and for whole rounds late in a run) cut its eight tuning cases from 13,364 to 10,534 mean tokens, but on all 30 cases the mean went from 9,270 to 10,363; the authors write that a guard "cannot see whether a run will ever close its trace" [checked: 2610.23818 App F]. In an unchanged R1-Distill-Qwen-7B, 13% of rollouts never reach saturated probe confidence before a 16k cap [checked: 2608.28859 abstract]. Keeping the recurrent state of a hybrid model in bf16 and not in FP32 lengthens outputs ×1.07 to ×1.27 in five of six models, and raises hits of the 32K cap from 118 to 223 of 590 items on Qwen3.5-9B [checked: 2611.12127 abstract, Table 1].

---

## 2. Level ladder

- **Level 2.** Can say what the thinking block and the EoT token are, and that the length of a run is a sampled outcome. Can state λ, the hindsight stop point and the idea of an early-exit rule. Knows that a run cut off at the cap is scored wrong. Follows the AUROC mixture formula when it is shown.
- **Level 3.** Can compute the round ratio, the slack share, a capped length ratio and a pooled AUROC from given numbers. With a hint, explains why a cap understates λ, why a question-only score has within-problem AUROC ½, and what the two inflation models predict after a switch back to exact.
- **Level 4.** Derives the capped mean and the AUROC identity without help. Explains why the hindsight point is a bound and why agreement rules fail (answers that are later revised). Separates "extra tokens after the answer" from "answer reached later" by task. Names the channel (token, stored KV, or both) for a given lossy method and designs arms that separate them. Reads a table critically (median against sum, proxy definitions, runs lost to the cap).
- **Level 5.** Designs the switch-back experiment with a zero-dose control, endpoints that survive the cap, and a sample size. Finds the flaw in "the probe lets us stop early" and in "early exit removes the inflation". Can explain each choice to someone else.

---

## 3. Misconceptions

1. **"A longer trace means the model worked harder, so it is more often right."** Across problems the sign is the other way: long traces come from hard problems. Inside one problem the shorter of two disagreeing traces is right 60% of the time on the hard set (54% on a second suite, where the interval includes 50%), and more than half of the thinking tokens come after the answer is fixed. [checked: 2611.02278 Sec 5.1, Table 4]
2. **"My probe predicts correctness with AUROC 0.85, so it can tell which trace will fail."** A pooled AUROC is mostly a between-problem number. The same probe is at chance inside problems (0.496 or 0.515, depending on the weighting); it has learned which problems are hard. [checked: 2609.03436 Sec 4.4]
3. **"The hindsight stop point is an early-exit rule."** It needs the final answer of the trace, so it can only be computed afterwards. Rules that can run lose 2.4 to 9.2 points. [checked: 2611.02278 Table 5]
4. **"Appending `</think>` ends the reasoning."** The model can keep reasoning in the answer phase and write EoT a second time. 515 forced runs still reach the budget in one study. [checked: 2609.03633 Sec 4.2; 2610.23818 App F]
5. **"With the same cap in both arms, the measured λ is fair."** Both arms are right-censored, the lossy arm more, so λ is understated. The cap also turns length into wrong answers, so the accuracy drop is partly a length effect. [checked: 2610.23818 Limitations, Sec 4]
6. **"The lossy model reasons worse, so the extra tokens are extra work before the answer."** On GSM8K half or more of the extra tokens come after the answer first appears. On AIME24 most come before (59% on GPT-OSS-20B; the Qwen3-8B estimate is too wide to use). The task decides. [checked: 2610.23818 Sec 4; App F, Table F2]
7. **"The extra length is just a loop."** On GPT-OSS-20B at most 2.9% of tokens are in a repeated 50-gram (1.9% lossless). Loops are real for Qwen3-8B under one rule only. [checked: 2610.23818 Sec 3.7]
8. **"If two models give the same mean λ they are the same; and a heavy tail proves lasting damage."** Exercise 2 shows two models with the same λ = 1.5 and opposite answers after a switch back. It also shows that the per-token model puts 84% of the net extra tokens in the top tenth of cases, just from sampling noise.

---

## 4. Question bank

Numbers marked "made up" are chosen for the exercise and come from no paper.

### reasoning_length-Q1
- **Level:** 3 (level 2 if the tutor first gives R ≈ L/(ℓ̄ + 1)). **Type:** predict (numeric).
- **Question:** "A lossless speculative setup accepts 3.0 draft tokens per round on average. A relaxed rule accepts 4.2, but the outputs become 1.4 times longer (λ = 1.4). The draft length is 6. (a) By what factor does the number of target rounds change? (b) Is the relaxed rule faster? (c) What is the largest λ that any relaxed rule could pay for at this draft length?" (Numbers made up.)
- **Answer key:** Rounds R ≈ L/(ℓ̄ + 1). (a) R_relax/R_strict = λ·(ℓ̄_strict + 1)/(ℓ̄_relax + 1) = 1.4 × 4.0/5.2 = 1.077, so 7.7% more rounds. (b) No. The acceptance gain is 5.2/4.0 = 1.30, which is below λ = 1.4. (c) The best case accepts all 6 draft tokens: (6 + 1)/(3 + 1) = 1.75. [formula checked: 2610.23818 Sec 2.2, Eq 4-5]
- **Likely wrong answer:** "Acceptance went up 30%, so it is about 30% faster." (Ignores that there are more tokens to generate.)
- **Follow-up:** "Which ℓ̄_relax would exactly break even at λ = 1.4?" (1.4 × 4.0 − 1 = 4.6.)

### reasoning_length-Q2
- **Level:** 2. **Type:** explain why (numeric).
- **Question:** "In one study, a run that is cut off before it closes its thinking is scored wrong. 1,132 lossy runs were still thinking at a threshold a little below the length budget. Scored as they are, 126 of them are correct. When the closing token is forced at the threshold, 439 are correct: 335 runs change from wrong to correct and 22 from correct to wrong. (a) What is the accuracy of these runs before and after? (b) What does this say about why long runs are wrong? (c) What is still not explained?"
- **Answer key:** (a) 126/1,132 = 11.1% and 439/1,132 = 38.8%; net change 335 − 22 = +313. (b) At least 335 runs already held a usable answer and only lacked room to state it. So part of the accuracy loss under a cap is a length effect, not wrong reasoning. (c) 1,132 − 439 = 693 runs (61%) are still wrong. There the reasoning had not reached the answer, or had reached a wrong one, or the model went on reasoning after the forced token. The numbers do not say how many runs fall in each group. [checked: 2610.23818 Sec 4, Table 4; App F]
- **Likely wrong answer:** "Runs that hit the cap are wrong because the reasoning failed."
- **Follow-up:** "Does the forced close also give the speed back?" (No. The tokens before the cap are already paid for. And 515 forced runs still reach the budget because the model resumes reasoning [checked: 2610.23818 App F].)

### reasoning_length-Q3
- **Level:** 3. **Type:** explain why (numeric).
- **Question:** "For Qwen3.5-9B on a hard problem set, a trace has 16,057 thinking tokens on average: 7,379 before the answer is fixed and 8,677 after. The median trace has its answer fixed at 0.35 of its thinking. (a) What share of all thinking tokens comes after the fixed point? (b) The median trace suggests 65% after. Why is the token share lower?"
- **Answer key:** (a) 8,677/16,057 = 54.0%. (The two parts add up to 16,056 because the means are rounded.) (b) The token share weights each trace by its length. Long traces have a smaller share after the fixed point: for traces that hit the cap the share is 36%, against 65% for correct traces and 59% for wrong ones. By the definition, a trace whose last readout is not its final answer is fixed "at the end", so it has no slack at all. [checked: 2611.02278 Table 4]
- **Likely wrong answer:** "The two numbers contradict each other; one must be a mistake." Or 7,379/16,057 = 46% given as the slack.
- **Follow-up:** "Wrong traces have 59% of their tokens after the fixed point. What does that tell you about using 'the answer stopped changing' as a sign of a correct answer?" (It is not a sign of a correct answer: wrong answers are fixed early too.)

### reasoning_length-Q4
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "Toy model. In exact decoding, thinking ends with chance 1/1,000 per token. In lossy decoding the chance is 1/1,500. Runs are cut at 2,000 tokens. (a) What is the true λ without a cap? (b) What mean lengths and what λ do you measure with the cap? (c) What share of runs hits the cap in each arm?" (Numbers made up.) *Tutor note:* at level 3, give the formula E[min(L, C)] = (1 − (1 − h)^C)/h if the learner asks for it. Deriving it is level 4.
- **Answer key:** (a) Means are 1/h: 1,000 and 1,500, so λ = 1.5. (b) E[min(L, C)] = Σ_{t=0..C−1} (1 − h)^t = (1 − (1 − h)^C)/h. Exact: 1,000 × (1 − 0.1352) = 864.8. Lossy: 1,500 × (1 − 0.2635) = 1,104.8. Measured λ = 1,104.8/864.8 = 1.277. (c) (1 − h)^C: 13.5% and 26.3%. The cap hides 45% of the extra length (0.277 is left of 0.5).
- **Likely wrong answer:** "The cap is the same for both arms, so the ratio stays 1.5."
- **Follow-up:** "What does the measured ratio do when the cap goes to 500 and to 8,000?" (1.08 and 1.49. A tight cap pushes the ratio to 1.)

### reasoning_length-Q5
- **Level:** 3. **Type:** find the flaw.
- **Question:** "A colleague trains a linear probe on the hidden state at thinking token 4 of R1-Distill-Qwen-7B. On 1,024 traces (128 problems, 8 traces each) it predicts final correctness with AUROC 0.85. He proposes: 'Kill the traces with a low probe score at token 4 and sample again. This saves the tokens of traces that will fail.' Find the flaw, and name the number that would test it."
- **Answer key:** The 0.85 is pooled over problems. A score reaches it by knowing only how hard the problem is. Then all traces of one problem get about the same score, and a new sample of the same problem gets the same low score again. The test is the within-problem AUROC: use only (correct, incorrect) pairs from the same problem. In the source it is at chance at token 4 (0.496 [0.466, 0.527] with the pre-registered weighting, 0.515 [0.481, 0.562] with exact pair weighting) and at chance at all ten positions from token 4 to 512, while a difficulty score that never reads the trace reaches 0.981 on the same pooled set. So the plan kills hard problems, not bad traces. Probing later does not help: the pooled AUROC itself falls to 0.586 at token 16 and about 0.55 from token 32 on. One probe variant built after the fact reaches 0.556 [0.522, 0.588] inside problems at token 32; this is small and far from 0.85. [checked: 2609.03436 Sec 4.4]
- **Likely wrong answer:** "Token 4 is too early. Train the probe at token 512 and it will work."
- **Follow-up:** "With one trace per problem, what is w, and what can the pooled AUROC tell you about the within-problem signal?" (w = 0. Nothing.)

### reasoning_length-Q6
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "Use AUROC_pooled = w·AUROC_within + (1 − w)·AUROC_between. (a) w = 0.2, within = 0.60, between = 0.95: what is the pooled AUROC? (b) A probe has pooled 0.849 and within 0.50 with w = 0.1: what is its between-problem AUROC? (c) Could a pooled 0.849 at w = 0.1 come only from within-problem signal, with between = 0.50?" (The values of w are made up.)
- **Answer key:** (a) 0.2 × 0.60 + 0.8 × 0.95 = 0.88. (b) (0.849 − 0.1 × 0.50)/0.9 = 0.888. (c) No: it would need within = (0.849 − 0.9 × 0.50)/0.1 = 3.99, and an AUROC cannot exceed 1. [formula checked: 2609.03436 Eq 1]
- **Likely wrong answer:** (a) "The average, 0.775." (c) "Yes, if the probe is very good inside problems."
- **Follow-up:** "In Exercise 1, score D has within-problem AUROC 0.85, but the pooled value only moves from 0.925 to 0.954. Why?" (w is 0.003 there: with 8 traces per problem almost every pair is a between-problem pair.)

### reasoning_length-Q7
- **Level:** 3. **Type:** predict. **Tied to Part B.**
- **Question:** "You decode the first 500 thinking tokens with a lossy method and then switch to exact decoding. You look only at runs that are still thinking at token 500 and measure how many more tokens they need. What does the per-token cost model predict? What does the lasting damage model predict? Which statistic would you look at?"
- **Answer key:** Per-token cost: the stop chance depends only on the current step, so after the switch the remaining length has the same distribution as in an exact run. The total cost is only what was paid during the first 500 tokens. Lasting damage: some runs were pushed into a state that does not close, and they stay long after the switch. The effect sits in a few runs, so look at the upper tail (share of runs above a long threshold, or the cap-hit share), not at the median. In the toy of Exercise 2 the mean remaining length is 1.33× the exact one while the median is only 1.05×.
- **Likely wrong answer:** "Both predict a return to normal, because the model is exact again."
- **Follow-up:** "In the lasting damage case, where is the damage stored once decoding is exact again?" (In the text already written, which stays in the context. For methods that write approximate KV entries, also in the cache. This is Part C.)

### reasoning_length-Q8
- **Level:** 2. **Type:** explain why.
- **Question:** "The Concise-9B paper defines the point where the answer of a trace is 'fixed'. (a) State the definition. (b) Why can a running system not stop at that point? (c) What happens to accuracy with a rule that can run, such as 'stop when 3 readouts in a row agree'?"
- **Answer key:** (a) Read out a forced answer every 2,048 thinking tokens. The answer is fixed at the first readout from which all later readouts equal the final answer of the trace. (b) The definition uses the final answer and all later readouts. A running system has neither, so the point is known only in hindsight; it is an upper bound on what early exit can save. (c) The 3-agree rule loses 2.4 points (confirmation set) to 9.2 points (hard set) on Qwen3.5-9B, because early agreement is common on answers that the model later revises. [checked: 2611.02278 Table 5]
- **Likely wrong answer:** "Stop when the answer stops changing; that is the fixed point." (You cannot know it has stopped changing without seeing the future.)
- **Follow-up:** "The grid is 2,048 tokens wide. Assume the answer does not change and change back between two readouts that agree. Does the grid make the measured slack too large or too small?" (Too small. The true fixed point is at or before the readout that detects it, up to 2,047 tokens earlier. A trace that is too short to have a readout counts as no slack, which also lowers the number. Without the assumption there is one case the other way: a change and change back between two late readouts is missed, and then the slack is too large. The paper treats its number as a lower bound.) [checked: 2611.02278 Sec 5.2, Limitations]

### reasoning_length-Q9
- **Level:** 4. **Type:** apply to a new case (numeric).
- **Question:** "DEER asks for a trial answer of 4 tokens. The largest next-token probabilities at the 4 steps are 0.99, 0.98, 0.97 and 0.90. The threshold is 0.95. (a) Exit or continue? (b) The same with 0.80 in place of 0.90. (c) Why a geometric mean and not the plain product of the probabilities? (d) Give one reason why a high value does not mean that the full trace would end with the same answer."
- **Answer key:** C = (∏ p_i)^{1/n}. (a) Product 0.8470, fourth root 0.959 > 0.95: exit. (b) Product 0.7529, fourth root 0.931: continue. (c) The product falls with the answer length: 12 tokens at 0.99 each give a product of 0.886 but a geometric mean of 0.99. The mean is a per-token confidence, so one threshold works for short and long answers. (This is a derivation of this file. The paper itself compares the geometric mean with the arithmetic mean: the geometric mean is pulled down more by one low token.) (d) Any one of: early agreement is common on answers that are later revised [checked: 2611.02278 Sec 5.2, Table 5]; C is the model's own confidence in a trial answer, and a model can be sure of a wrong answer; a reimplementation of DEER by another group loses 6.7 points on AIME24 [checked: 2608.28859 Table 3]. A learner may also say "the confidence follows how easy the problem is, not how good this trace is", by analogy with the probe result of Q5. Accept it as a hypothesis; no source read here tests it for DEER. [formula checked: 2504.15895 Sec 3.2, Eq 4]
- **Likely wrong answer:** "(a) Continue, because one token is below 0.95."
- **Follow-up:** "If three of the four probabilities are 1.0, what is the lowest fourth probability that still exits?" (0.95⁴ = 0.8145.)

### reasoning_length-Q10
- **Level:** 4. **Type:** find the flaw. **Tied to Part A.**
- **Question:** "A colleague says: 'On GPT-OSS-20B and GSM8K, exact sampling at temperature 1.2 gives λ = 1.27 [1.10, 1.47]. The relaxed rule mentored_dec at its loosest setting gives λ = 1.16 at temperature 1.0 on the same task, which is inside that interval. So length inflation is a function of one number, the dose of unlikely tokens. The method does not matter.' Find the flaws."
- **Answer key:** (1) One matched point is not a curve. The claim needs several doses per method, with all methods falling on one curve. (2) The dose was never measured; only λ was matched. Dose must be defined and measured in the same way in both arms, for example the surprisal −log p(x_t) of each committed token under the exact target at temperature 1. The two methods also differ in shape. Temperature adds a little surprisal at every position. A relaxed rule commits a small share of far-tail tokens (7% to 24% of committed draft tokens are relaxed-only; for four of the five rules their median target probability is 0.01 to 0.09). The mean can match while the shape differs. (3) In the same paper the two do not move together on another model: Qwen3-8B barely lengthens at temperature 1.2 or 1.5 (λ 1.02 to 1.04) but inflates under relaxed rules (1.18× on average with its EAGLE-3 head). This speaks against "one number". It is not a proof, because the dose was not measured there either. (4) With the same target and the same five rules, the drafter moves the mean λ of Llama-3.1-8B-Instruct from 1.10× (standalone Llama-3.2-1B) to 3.30× (EAGLE-3 head). Either the two drafters give very different doses, or something besides the dose matters (which tokens, and where in the trace). Only a dose measurement can tell. Note that the EAGLE-3 head reads the target's hidden states and the standalone drafter reads only tokens, so "the drafter that sees less does more harm" does not fit either. (5) The two methods change the stop decision in different ways. Temperature reshapes the whole distribution at every step, the EoT probability included. A relaxed rule changes the committed distribution only where q differs from p, and it moves it toward q. A level-4 answer has (1), (2) and one of (3) to (5). [checked: 2610.23818 App C, Sec 3.6, 3.7, Table B2]
- **Likely wrong answer:** "The flaw is that temperature 1.2 is not a lossy method." (True, but that is the point of using it as a dose control.)
- **Follow-up:** "Which single measurement would you add to both arms first?" (Score every committed token under the exact target with teacher forcing. This gives a dose histogram per run.)

### reasoning_length-Q11
- **Level:** 4. **Type:** apply to a new case. **Tied to Part C.**
- **Question:** "Three lossy methods: (i) relaxed speculative acceptance, with a separate drafter and a dense target; (ii) sparse attention at every step, where the model reads only part of the cache; (iii) an 8-bit KV cache. For each one, through which channel can harm reach later steps: the emitted token, the stored KV entries, or both? Then design the arms that separate the two channels for method (ii) or (iii)."
- **Answer key:** (i) Token only. The target computes its own KV for every committed token in its verification pass, so its cache is exact given the text. (This holds when the drafter has its own cache; check it for self-speculative systems, see `specdec_exact.md`.) (ii) Both. The token comes from approximate logits, and the keys and values written at this position in deeper layers come from hidden states that were computed with the sparse read. (iii) Both. The read is approximate, and the entries stay approximate for the rest of the run. Design, a 2×2 at a switch point s, with exact decoding after s: text before s is {from an exact run, from the lossy run} and the cache at s is {rebuilt exactly from that text, left as the lossy method wrote it}. "Lossy text + rebuilt cache" is the token-only arm. "Exact text pushed through the lossy method with teacher forcing + its approximate cache" is the KV-only arm. "Exact text + rebuilt cache" is the baseline. "Lossy text + lossy cache" is the method as it really runs (both channels). Compare the remaining length after s.
- **Likely wrong answer:** "Quantized KV only changes the stored state, so the token is not a channel." Or: "In relaxed speculative decoding both channels are active."
- **Follow-up:** "For method (i), which cells of the 2×2 cannot be built, and what does that tell you about the inflation in 2610.23818?" (The two approximate-cache cells. The target builds its own cache in a dense, exact verification pass, also when the drafter is an EAGLE-3 head on the target. So all of that inflation goes through the emitted tokens.)

### reasoning_length-Q12
- **Level:** 3. **Type:** apply to a new case (numeric).
- **Question:** "Medians for GPT-OSS-20B, lossless → relaxed, over pairs where both runs are correct and not cut off. GSM8K: first appearance of the answer at token 102 → 126; total length 183 → 264. AIME24: first appearance 1,721 → 3,865; total length 3,822 → 7,020. From these medians, what share of the extra tokens comes after the answer on each task? What does the difference between the tasks mean? Why is your number only rough?"
- **Answer key:** GSM8K: extra length 81, extra before the answer 24, so 57/81 = 70% after. AIME24: extra length 3,198, extra before the answer 2,144, so 67% before and 33% after. Meaning: on the easy task the model reaches the answer at almost the same place and then keeps checking; on the hard task it reaches the answer much later. Rough because a difference of medians is not the median (or mean) of the per-pair differences. The paper's own per-pair figures are 50% [37, 59] after on GSM8K and 41% [28, 54] after on AIME24: same direction, different size. Also, "first appearance" is a proxy. [checked: 2610.23818 App F, Table F2]
- **Likely wrong answer:** "On both tasks the model thinks about twice as long, so it is the same effect."
- **Follow-up:** "For Qwen3-8B on AIME24 the paper gives −16% [−177, 49] after the answer, from 33 pairs. What can you conclude?" (Not the size. The interval is 226 points wide. Its upper end is 49%, so the data fit "at least half of the extra tokens come before the answer", the same direction as GPT-OSS-20B, and nothing more exact than that. A negative share means that the part after the answer got shorter while the part before it grew by more than the whole increase.)

### reasoning_length-Q13
- **Level:** 4. **Type:** connect two ideas. **Tied to Part B.**
- **Question:** "Fact 1: the tenth of cases with the largest increase holds 59% of the net extra tokens. Fact 2: a per-token guard (exact verification for eight positions after clusters of hesitation markers) cut the mean length on its 8 tuning cases but not on all 30 cases. Which of the two inflation models do these facts favor? Why is neither fact a proof?"
- **Answer key:** Both fit lasting damage: a few runs enter a state that does not close, and a guard that judges one token at a time cannot see that state. Not a proof, because: (a) concentration also comes from sampling noise. In the toy of Exercise 2 the per-token model puts 84% of the net extra tokens in the top tenth, and a third of the cases are more than 2× longer with no lossy step at all. (b) The guard was tuned on 8 cases, where two runs that hit the budget held 61% of all tokens; a failure to transfer to 30 cases can be overfitting. It is also one run per case, and on these 30 problems λ of one rule moves from 0.96 to 1.60 between seeds. (c) Neither fact compares the remaining length after a switch back to exact, which is the only direct test. [checked: 2610.23818 Sec 3.7, App C, App F]
- **Likely wrong answer:** "A heavy tail means lasting damage. Per-token cost would stretch every run by the same factor."
- **Follow-up:** "In Exercise 2, with lossy decoding for the whole run, which statistic does separate the two models?" (The quantiles. Per-token cost moves the median and the 90th percentile by the same 1.50. Lasting damage moves the median by 1.03 and the 90th percentile by 1.18; almost all of its extra length is above the 90th percentile.)

### reasoning_length-Q14
- **Level:** 5. **Type:** apply to a new case. **Tied to Part B.**
- **Question:** "Design the experiment that decides between per-token cost and lasting damage for one lossy method on one reasoning model. Say which arms you run, what you measure, what each model predicts, and which controls you need. Assume a length cap that many runs hit."
- **Answer key:** A level-5 answer has these parts.
  1. *Arms.* For each problem: exact for the whole run; lossy up to token s and exact after; several values of s. Fully lossy is useful only as a reference.
  2. *Endpoint.* The remaining length after s, for runs still thinking at s. Not the total length, which mixes in the cost already paid before s.
  3. *Matched control.* The runs still thinking at s are a selected group, and length follows the problem. So compare inside each problem: continuations of a lossy prefix against continuations of an exact prefix of the same problem at the same s. Even then the two groups are selected a little differently, because the lossy arm loses fewer runs before s. Report how many runs each arm lost before s, and keep s small against the typical length.
  4. *Zero-dose control.* Continue two different exact prefixes of the same problem and compare them. This is the noise floor. Without it any difference can be sampling or numerical noise (up to 9,000 tokens under greedy bf16 [checked: 2506.09501 abstract]).
  5. *Several continuations per prefix.* Lasting damage is a property of the prefix. If a prefix is broken, most of its continuations are long. A high between-prefix variance (against within-prefix variance) is the signature. Per-token cost predicts no prefix effect.
  6. *Cap-safe statistics.* Use the share of runs with remaining length above a threshold, or survival curves, with the same cap in all arms. Do not use raw means.
  7. *Size.* The toy in Q16 needs about 600 runs per arm for a 1.33× effect. Pairing inside problems reduces this.
  8. *Links.* Record the surprisal of the prefix tokens under the exact target (Part A). For methods that write approximate KV, decide at s whether the cache is kept or rebuilt (Part C).
- **Likely wrong answer:** "Run lossy-then-exact and exact on the whole dataset and compare the mean total length."
- **Follow-up:** "Why do you continue each lossy prefix several times and not once?" (One continuation cannot separate 'this prefix is broken' from 'this sample was long by chance'.)

### reasoning_length-Q15
- **Level:** 5. **Type:** find the flaw.
- **Question:** "A colleague argues: 'More than half of all thinking comes after the answer is fixed (54% to 64%). Under relaxed speculative decoding, 50% to 82% of the extra tokens on GSM8K come after the answer first appears. So add an early-exit rule to the lossy system and the length inflation is gone at no cost.' Find the flaws."
- **Answer key:** (1) The 54% to 64% is the hindsight bound. Rules that can run lose 2.4 to 9.2 points; on the hard set the 3-agree rule spends 0.45× of the tokens, about the same as the bound's 0.44×, and still loses 9.2 points [checked: 2611.02278 Table 5]. (2) The GSM8K numbers do not hold on hard tasks. On AIME24 the relaxed run reaches the answer later (1,721 → 3,865) and few cut-off runs contain the answer at all (8 of 33 and 9 of 53, against 15 of 26 and 147 of 201 on GSM8K). An exit rule cannot return an answer that was never reached [checked: 2610.23818 Table F2]. (3) "First appears" needs the reference answer; a running system does not have it. (4) An injected EoT is not always obeyed (idea 10). (5) Picking the bad trace inside a problem is not solved: the one probe study read for this pack is at chance inside problems (0.496 or 0.515, depending on the weighting) [checked: 2609.03436 Sec 4.4]. (6) The exit rule hides the effect; it does not explain it. The exit decision is also computed from the lossy model's own outputs. A level-5 answer sorts by task: slack after the answer on easy tasks, a real delay on hard ones.
- **Likely wrong answer:** "The only flaw is that early exit costs a little accuracy."
- **Follow-up:** "On GSM8K most relaxed runs that are cut off already contain the answer. On AIME24 few do. What does a forced close buy in each case?" (GSM8K: it can recover the answer in most cut-off runs, 15 of 26 and 147 of 201. AIME24: in few, 8 of 33 and 9 of 53.)

### reasoning_length-Q16
- **Level:** 5. **Type:** apply to a new case (numeric). **Tied to Part B.**
- **Question:** "Lasting damage toy. Exact decoding stops with chance h0 = 1/1,000 per token. Each lossy token breaks the trace with chance d. A broken trace stops with chance h_D = 1/10,000 per token, in every mode. A trace that is not broken stops with chance h0 in both modes; at a lossy token the stop check comes first, then the break. (a) Find d so that a fully lossy run has λ = 1.5. (b) Now decode lossy for 500 tokens, then exact. Among runs still thinking at token 500, about what share is broken, and what is the mean remaining length relative to exact? (c) About how many runs per arm do you need to see this difference in means (5% two-sided, 80% power)? (d) Why is a yes/no endpoint better when there is a cap?" (Numbers made up.)
- **Answer key:** (a) Let x = (1 − h0)·d ≈ d. A healthy trace leaves the healthy state with chance a = h0 + x per token, so it spends 1/a tokens there and breaks with probability x/a. Mean length = 1/a + (x/a)/h_D. Set this to λ/h0: 1 + x/h_D = λ(h0 + x)/h0, so x = (λ − 1)/(1/h_D − λ/h0) = 0.5/(10,000 − 1,500) = 5.88 × 10⁻⁵. (b) Still thinking and healthy: (1 − h0)^500·(1 − d)^500 ≈ e^{−0.5}·e^{−0.029} = 0.589. Still thinking and broken ≈ d·∫₀^500 e^{−(h0+d)k} dk ≈ 5.88 × 10⁻⁵ × 388 = 0.0228 (a broken trace almost never stops before token 500). Share broken ≈ 0.0228/(0.589 + 0.0228) = 3.7% (exact sum: 3.64%). Mean remaining = 0.9636 × 1,000 + 0.0364 × 10,000 = 1,327.5, a ratio of 1.33. (c) n per arm = (z_{α/2} + z_β)²·(σ₀² + σ₁²)/δ², with z_{α/2} = 1.96, z_β = 0.84, δ = difference of the means = 327.5, σ₀ = sd of the exact remaining length ≈ 1,000, σ₁ = sd in the lossy-then-exact arm = 2,728 (mixture of two geometrics). n = 7.84 × (1.00 + 7.44) × 10⁶/327.5² ≈ 617. (d) A cap censors the long broken runs and shrinks the difference in means. The share of runs above a threshold below the cap is not touched by the cap: P(remaining > 5,000) is 0.67% against 2.85%, about 570 runs per arm.
- **Likely wrong answer:** "(b) The ratio is 1.5, as in the fully lossy run." Or: "(b) 500 lossy tokens out of 1,000, so half of the effect, 1.25."
- **Follow-up:** "What does the per-token cost model give for (b)?" (Exactly 1.0. A geometric length has no memory.)

### reasoning_length-Q17
- **Level:** 4. **Type:** connect two ideas.
- **Question:** "Fact 1 (quantization): the quantized and the full-precision model differ most at positions where the full-precision model has high entropy, and the tokens there are hesitation words. Fact 2 (relaxed speculative decoding): tokens accepted only by the relaxed rule have a median target probability of 0.01 to 0.09. Connect the two facts to the question 'which tokens need verifying?' Then say what speaks against a simple answer."
- **Answer key:** Both facts say that the harm enters at a small share of positions. The natural hypothesis: verify exactly at branch points, that is, where the target's entropy is high, and where the draft token is a far-tail token or a hesitation marker. Support: a penalty on 50 markers cuts CoT by 12% to 23%; boosting them gives up to +445% CoT; a penalty on random tokens has no consistent effect [checked: 2606.00206 abstract, Sec 5.2, App B.4]. Against: (1) a guard built on this idea did not hold on 30 cases (mean 9,270 → 10,363) [checked: 2610.23818 App F]. (2) Fact 1 is about where two distributions differ and fact 2 about which tokens were committed. Neither shows that these tokens cause the later length, and both papers say so. (3) If the damage is lasting, the token to verify is the one that starts the state that does not close. It may not look special when it is emitted.
- **Likely wrong answer:** "Verify every low-probability token. Fact 2 proves that they cause the inflation."
- **Follow-up:** "The rule spec_casc_tok adds 7% extra tokens with median target probability 0.29 and mean target rank 0.9. Its mean λ is 0.96. What does this suggest?" (Extra tokens near the top of the target's ranking look harmless; the inflation goes with far-tail tokens. This is a comparison across rules, not a causal test. The rule also differs in a second way: it is the only one that can reject tokens that lossless verification accepts. So its low λ may not come from the kind of extra tokens alone.) [checked: 2610.23818 Sec 2.3, 3.6, 3.7]

---

## 5. Exercises

All three were run in the scratch folder with Python 3.11.5 and numpy 1.24.3 on CPU. The outputs below are the real outputs of these scripts.

### Exercise 1. A pooled AUROC that is all difficulty

**Task.** Simulate 128 problems with pass rates drawn from Beta(0.5, 0.5) and 8 attempts each. Build four scores for each attempt: (A) the true pass rate of its problem; (B) the pass rate plus noise; (C) the leave-one-out pass rate, that is, the share of the other 7 attempts that are correct; (D) the pass rate plus 0.15 for a correct attempt plus noise. For each score compute w, the pooled, the within-problem and the between-problem AUROC, and check AUROC_pooled = w·AUROC_within + (1 − w)·AUROC_between. Count ties as ½.

**Expected result.** w = 0.003. Pooled AUROC is about 0.9 for all four scores. Within-problem AUROC is exactly 0.500 for A, about 0.50 for B, exactly 0.000 for C and 0.85 for D. The identity holds to rounding error.

**How the tutor checks.** The learner gives the table and then answers without code: (1) "Why is the within-problem value of C exactly 0?" (With k correct attempts out of n, a correct attempt sees (k − 1)/(n − 1) and an incorrect one sees k/(n − 1). The correct attempt always has the lower score.) (2) "Score D can really tell a good trace from a bad one. Why does the pooled AUROC move so little, from 0.925 to 0.954?" (w = 0.003: almost all pairs are between problems.) (3) "What would you report for a new probe?" (The within-problem AUROC and the trace-blind baseline next to the pooled value.)

```python
# Exercise 1: a pooled probe AUROC can be all problem difficulty.
import numpy as np
rng = np.random.default_rng(0)

P, n = 128, 8                                  # problems, attempts per problem
p = rng.beta(0.5, 0.5, size=P)                 # pass rate of each problem
Y = (rng.random((P, n)) < p[:, None]).astype(int)   # 1 = correct attempt
prob = np.repeat(np.arange(P), n)              # problem id of each attempt
y = Y.ravel()

def wins(sp, sn):
    """wins + half the ties over all (correct, incorrect) pairs, and the pair count"""
    d = sp[:, None] - sn[None, :]
    return (d > 0).sum() + 0.5 * (d == 0).sum(), d.size

def decompose(s):
    w_all, n_all = wins(s[y == 1], s[y == 0])
    w_in = n_in = 0
    for i in range(P):
        m = prob == i
        a, b = wins(s[m & (y == 1)], s[m & (y == 0)])
        w_in += a
        n_in += b
    w = n_in / n_all
    pooled = w_all / n_all
    within = w_in / n_in
    between = (w_all - w_in) / (n_all - n_in)
    return w, pooled, within, between

k = Y.sum(1)                                   # correct attempts per problem
mixed = int(((k > 0) & (k < n)).sum())
print(f"problems {P}, attempts {P*n}, accuracy {y.mean():.3f}, "
      f"problems with both outcomes {mixed}")

loo = ((k[:, None] - Y) / (n - 1)).ravel()     # pass rate of the OTHER attempts
scores = {
    "A true pass rate (question only)": np.repeat(p, n),
    "B pass rate + noise (trace-blind)": np.repeat(p, n) + 0.1 * rng.standard_normal(P * n),
    "C leave-one-out pass rate":         loo,
    "D pass rate + 0.15*Y + noise":      np.repeat(p, n) + 0.15 * y + 0.1 * rng.standard_normal(P * n),
}
print(f"{'score':36s} {'w':>6s} {'pooled':>7s} {'within':>7s} {'between':>8s} {'identity gap':>13s}")
for name, s in scores.items():
    w, pooled, within, between = decompose(s)
    gap = pooled - (w * within + (1 - w) * between)
    print(f"{name:36s} {w:6.3f} {pooled:7.3f} {within:7.3f} {between:8.3f} {gap:13.1e}")
```

Output:

```
problems 128, attempts 1024, accuracy 0.511, problems with both outcomes 75
score                                     w  pooled  within  between  identity gap
A true pass rate (question only)      0.003   0.925   0.500    0.926       1.1e-16
B pass rate + noise (trace-blind)     0.003   0.912   0.503    0.913       1.1e-16
C leave-one-out pass rate             0.003   0.892   0.000    0.895       1.1e-16
D pass rate + 0.15*Y + noise          0.003   0.954   0.851    0.954       1.1e-16
```

### Exercise 2. Per-token cost against lasting damage

**Task.** Exact decoding stops with chance h0 = 1/1,000 per token. Model PC (per-token cost): a lossy step stops with chance h1 = h0/1.5. Model LD (lasting damage): each lossy token breaks the trace with chance d; a broken trace stops with chance h_D = 1/10,000 per token in every mode. (a) Derive d so that LD also has λ = 1.5 (see Q16). (b) Simulate fully lossy runs for both models. Report the ratio of means, of medians and of 90th percentiles to exact, the share of runs above a cap of 8,000, and the ratio of means under that cap. (c) Per-case view: pair each lossy run with one exact run; report the share of cases more than 2× longer and the share of the net extra tokens held by the top tenth of cases. Do the same for exact against exact. (d) Switch to exact after token 500. For runs still thinking at 500, report the remaining length relative to exact.

**Expected result.** d = 5.89 × 10⁻⁵. Both models have mean ratio 1.50. PC moves the median and the 90th percentile by 1.50; LD moves them by 1.03 and 1.18. Under the cap PC still shows 1.495 but LD shows only 1.23. Exact against exact: one third of the cases are more than 2× longer. Top-tenth share: 0.84 for PC and 1.41 for LD. After the switch: PC remaining ratio 1.00; LD 1.33 in the mean and 1.05 in the median; P(remaining > 5,000) is 0.67% for exact and PC and 2.9% for LD.

**How the tutor checks.** The learner gives the numbers and then answers without code: (1) "Both models have λ = 1.5. Which numbers in part (1) already differ, and why?" (The quantiles and the cap hits. PC stretches every run; LD leaves most runs alone and makes a few very long.) (2) "Why does the cap hide about half of the extra length in LD and almost none in PC?" (LD's extra tokens sit in runs far above 8,000.) (3) "One third of the cases double in length with no lossy step. What does that say about a per-case ratio from one run per arm?" (It is mostly noise. A geometric length is noisier than real traces, but the point stands: you need a zero-dose control.) (4) "Which line of the output decides between the models?" (Part (3), the remaining length after the switch.)

```python
# Exercise 2: per-token cost versus lasting damage. Same mean inflation,
# different answer to "what happens after we switch back to exact?"
import numpy as np
rng = np.random.default_rng(1)
N = 2_000_000
h0 = 1 / 1000          # exact decoding: chance per token that thinking ends
lam_target = 1.5       # both models are tuned to this mean inflation

# Model PC (per-token cost): while a step is lossy the stop chance is h1.
h1 = h0 / lam_target
# Model LD (lasting damage): each lossy token breaks the trace with chance d.
# A broken trace stops with chance hd per token, in lossy AND in exact mode.
hd = 1 / 10000
x = (lam_target - 1) / (1 / hd - lam_target / h0)   # x = (1-h0)*d, derived in the text
d = x / (1 - h0)
print(f"h0={h0:.6f} h1={h1:.6f} hd={hd:.6f} d={d:.3e}")

geom = lambda h, size: rng.geometric(h, size=size).astype(np.int64)

def run_pc(s):
    """lossy for tokens 1..s, exact afterwards (s = None means lossy for ever)"""
    g1 = geom(h1, N)
    if s is None:
        return g1
    return np.where(g1 <= s, g1, s + geom(h0, N))

def run_ld(s):
    e = geom(h0, N)                # token at which a healthy trace would stop
    k = geom(d, N)                 # token after which the damage happens
    limit = np.inf if s is None else s
    broken = (k < e) & (k <= limit)
    return np.where(broken, k + geom(hd, N), e)

exact = geom(h0, N)
C = 8000                           # length cap
def row(name, L):
    q50, q90 = np.percentile(L, [50, 90])
    e50, e90 = np.percentile(exact, [50, 90])
    capped = np.minimum(L, C).mean() / np.minimum(exact, C).mean()
    print(f"{name:14s} mean ratio {L.mean()/exact.mean():5.3f} | median ratio {q50/e50:5.3f} | "
          f"p90 ratio {q90/e90:5.3f} | cap hits {np.mean(L > C):6.2%} | ratio under cap {capped:5.3f}")

print(f"\n(1) lossy for the whole run, cap C={C} (exact run: cap hits {np.mean(exact > C):.2%})")
pc_full, ld_full = run_pc(None), run_ld(None)
row("per-token cost", pc_full)
row("lasting damage", ld_full)

print("\n(2) per-case view: run each case once per arm and compare with its exact run")
for name, L in [("exact vs exact", geom(h0, N)), ("per-token cost", pc_full), ("lasting damage", ld_full)]:
    diff = np.sort(L - exact)[::-1]
    top = f"{diff[:N // 10].sum() / diff.sum():5.2f}" if diff.mean() > 50 else "  n/a"
    print(f"{name:14s} net extra per case {diff.mean():7.1f} | cases more than 2x longer {np.mean(L > 2 * exact):6.2%} | "
          f"top-tenth sum / net sum {top}")

s = 500
print(f"\n(3) switch back to exact after token s={s}; runs still thinking at s only")
ex_rem = exact[exact > s] - s
for name, L in [("per-token cost", run_pc(s)), ("lasting damage", run_ld(s))]:
    rem = L[L > s] - s
    print(f"{name:14s} mean remaining {rem.mean():7.1f} (exact {ex_rem.mean():6.1f}) ratio "
          f"{rem.mean()/ex_rem.mean():5.3f} | median remaining ratio {np.median(rem)/np.median(ex_rem):5.3f} | "
          f"P(remaining > 5000) {np.mean(rem > 5000):6.2%} (exact {np.mean(ex_rem > 5000):5.2%})")

# closed form for the lasting-damage arm
a = (1 - h0) * (1 - d)
ks = np.arange(1, s + 1)
p_healthy = a ** s
p_broken = np.sum(a ** (ks - 1) * (1 - h0) * d * (1 - hd) ** (s - ks))
pb = p_broken / (p_healthy + p_broken)
mean_rem = (1 - pb) / h0 + pb / hd
var0 = (1 - h0) / h0 ** 2
second = (1 - pb) * (2 - h0) / h0 ** 2 + pb * (2 - hd) / hd ** 2
var_ld = second - mean_rem ** 2
delta = mean_rem - 1 / h0
n_arm = (1.96 + 0.84) ** 2 * (var0 + var_ld) / delta ** 2
print(f"\n(4) closed form: P(broken | still thinking at s) = {pb:.4f}; mean remaining = {mean_rem:.1f}; "
      f"ratio = {mean_rem * h0:.3f}")
print(f"    sd of remaining length: exact {var0 ** 0.5:.0f}, lasting damage {var_ld ** 0.5:.0f}")
print(f"    runs per arm to see the difference in means (5% two-sided, 80% power): {n_arm:.0f}")
p0, p1 = (1 - h0) ** 5000, (1 - pb) * (1 - h0) ** 5000 + pb * (1 - hd) ** 5000
pbar = (p0 + p1) / 2
n_bin = (1.96 * (2 * pbar * (1 - pbar)) ** 0.5 + 0.84 * (p0 * (1 - p0) + p1 * (1 - p1)) ** 0.5) ** 2 / (p1 - p0) ** 2
print(f"    P(remaining > 5000): exact {p0:.4f}, lasting damage {p1:.4f}; runs per arm for this yes/no test: {n_bin:.0f}")
```

Output:

```
h0=0.001000 h1=0.000667 hd=0.000100 d=5.888e-05

(1) lossy for the whole run, cap C=8000 (exact run: cap hits 0.03%)
per-token cost mean ratio 1.502 | median ratio 1.501 | p90 ratio 1.502 | cap hits  0.49% | ratio under cap 1.495
lasting damage mean ratio 1.502 | median ratio 1.025 | p90 ratio 1.177 | cap hits  2.78% | ratio under cap 1.226

(2) per-case view: run each case once per arm and compare with its exact run
exact vs exact net extra per case     0.6 | cases more than 2x longer 33.33% | top-tenth sum / net sum   n/a
per-token cost net extra per case   501.9 | cases more than 2x longer 42.91% | top-tenth sum / net sum  0.84
lasting damage net extra per case   501.8 | cases more than 2x longer 35.22% | top-tenth sum / net sum  1.41

(3) switch back to exact after token s=500; runs still thinking at s only
per-token cost mean remaining   998.0 (exact  999.6) ratio 0.998 | median remaining ratio 1.000 | P(remaining > 5000)  0.67% (exact 0.67%)
lasting damage mean remaining  1328.6 (exact  999.6) ratio 1.329 | median remaining ratio 1.049 | P(remaining > 5000)  2.88% (exact 0.67%)

(4) closed form: P(broken | still thinking at s) = 0.0364; mean remaining = 1327.5; ratio = 1.328
    sd of remaining length: exact 999, lasting damage 2728
    runs per arm to see the difference in means (5% two-sided, 80% power): 617
    P(remaining > 5000): exact 0.0067, lasting damage 0.0285; runs per arm for this yes/no test: 569
```

### Exercise 3. The hindsight stop point against agreement rules (pen and paper)

**Task.** Seven traces. Each letter is one readout: the answer the model would give if thinking were closed at that point. Readouts come every 2,048 thinking tokens. The reference answer is A.

| trace | readouts | final answer |
|---|---|---|
| T1 | A A A A A A | A |
| T2 | B B A A A A A A | A |
| T3 | C B A A B A A A | A |
| T4 | D D D D | D |
| T5 | C D B A C D | B |
| T6 | B A A A A | A |
| T7 | B B B B A A A | A |

For each trace find (a) the hindsight fixed point; (b) where "2 readouts in a row agree" stops and which answer it returns; (c) the same for 3 in a row. A rule that never fires runs to the end and returns the final answer. Then give the total tokens relative to the full traces and the number of correct answers for each rule.

**Expected result.** Full traces: 44 readouts, 5 of 7 correct. Hindsight: 24 readouts (0.545×), 5 of 7. 2-agree: 21 readouts (0.477×), 3 of 7. 3-agree: 32 readouts (0.727×), 4 of 7.

**How the tutor checks.** The learner gives the table and then answers: (1) "Which traces make 2-agree wrong, and what do they share?" (T2 and T7. Early agreement on an answer that the model later revises.) (2) "Why does the hindsight rule never lose accuracy?" (By definition it returns the final answer of the trace.) (3) "T4 is wrong and is fixed at the first readout. What does that tell you?" (Stability is not correctness. In the paper wrong traces have 59% of their tokens after the fixed point.) (4) "Why is T5 fixed at the end?" (Its last readout is not its final answer. In a real trace this happens because the last readout comes before the end of the thinking; the toy keeps it for the definition.)

```python
# Exercise 3: the hindsight stop point against k-agree rules, on seven toy traces.
# One readout = the answer the model gives if thinking is closed at that point.
# Readouts come every 2,048 thinking tokens; a trace with m readouts has 2,048*m tokens.
GRID = 2048
REF = "A"                                         # reference answer
traces = {                                        # (readouts, final answer of the full trace)
    "T1": (list("AAAAAA"),   "A"),
    "T2": (list("BBAAAAAA"), "A"),
    "T3": (list("CBAABAAA"), "A"),
    "T4": (list("DDDD"),     "D"),
    "T5": (list("CDBACD"),   "B"),
    "T6": (list("BAAAA"),    "A"),
    "T7": (list("BBBBAAA"),  "A"),
}

def fixed_point(r, final):
    """first readout j such that r[j:] all equal the final answer; the end if there is none"""
    if r[-1] != final:
        return len(r), final
    j = len(r)
    while j > 1 and r[j - 2] == final:
        j -= 1
    return j, final

def k_agree(r, final, k):
    """stop at the first readout that completes k equal readouts in a row"""
    for j in range(k, len(r) + 1):
        if len(set(r[j - k:j])) == 1:
            return j, r[j - 1]
    return len(r), final                          # the rule never fires: run to the end

rules = {"full trace": lambda r, f: (len(r), f),
         "hindsight (fixed)": fixed_point,
         "2 agree": lambda r, f: k_agree(r, f, 2),
         "3 agree": lambda r, f: k_agree(r, f, 3)}
full = sum(len(r) for r, _ in traces.values())
print(f"{'rule':18s} " + " ".join(f"{t:>5s}" for t in traces) + "   tokens  ratio  correct")
for name, rule in rules.items():
    stops, ok = [], 0
    for r, f in traces.values():
        j, ans = rule(r, f)
        stops.append(f"{j}{ans}")
        ok += ans == REF
    tot = sum(int(x[:-1]) for x in stops)
    print(f"{name:18s} " + " ".join(f"{x:>5s}" for x in stops) + f"  {tot * GRID:7d}  {tot / full:.3f}  {ok}/{len(traces)}")
```

Output (each cell is the stopping readout and the returned answer):

```
rule                  T1    T2    T3    T4    T5    T6    T7   tokens  ratio  correct
full trace            6A    8A    8A    4D    6B    5A    7A    90112  1.000  5/7
hindsight (fixed)     1A    3A    6A    1D    6B    2A    5A    49152  0.545  5/7
2 agree               2A    2B    4A    2D    6B    3A    2B    43008  0.477  3/7
3 agree               3A    5A    8A    3D    6B    4A    3B    65536  0.727  4/7
```

The arithmetic in the answer keys of Section 4 was also checked with a short script (`checks.py` in the scratch folder: round ratio, forced-close shares, slack shares, capped geometric means, AUROC mixtures, DEER confidence, Table F2 medians).

---

## 6. Diagnostic picks

- **Level 2-3: reasoning_length-Q4** (capped geometric lengths). One numeric answer shows whether the learner can turn a stop chance into a mean length and sees which way the cap moves λ, or only says "same cap, fair comparison".
- **Level 4-5: reasoning_length-Q14** (design of the switch-back experiment). A level-4 answer gives the arms and the two predictions. A level-5 answer also has the matched control inside problems, the zero-dose control, several continuations per prefix, and statistics that survive the cap. It maps directly onto Part B and touches Parts A and C.

---

## 7. Sources

Read in this run, in saved copies of the arXiv HTML pages (no network request):

| id | title (as on the saved page) | what I read |
|---|---|---|
| 2610.23818 | Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding | Sections 2.1-2.3, 3.1-3.4, 3.6, 3.7, 4, 5, Limitations, Appendix C, Appendix F, Table F2; captions of Tables B1, F1, 4. Other tables not read cell by cell. |
| 2611.02278 | Concise-9B: Removing the Slack from a Reasoning Model's Thinking | Main text and Tables 4, 5, 6. Appendices not read. |
| 2504.15895 | Dynamic Early Exit in Reasoning Models (DEER), v3 | Abstract, Section 3 (method, Eq 4-5), Section 4.2 and nearby result text. Result tables not read cell by cell. |
| 2609.03436 | It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories | Abstract and Sections 4.2-4.4. The budget (truncation) part was read only in the abstract. |
| 2606.00206 | Quantized Reasoning Models Think They Need to Think Longer, but They Do Not | Main text and Appendix B.4. Other appendices not read. |
| 2609.03633 | </think> Doesn't Stop Reasoning: Analysis of Spurious CoT Termination | Abstract, introduction, Sections 4.1 and 4.2 (Fig 2). |
| 2608.28859 | The Halt Vector: Internalizing a Causal Steering Intervention for Efficient Reasoning | Abstract, introduction, Table 3 and the text on the DEER comparison, Limitations. |
| 2506.09501 | Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference (v2) | Abstract and introduction only (read by the examiner). |
| 2611.12127 | Lost in Rounding: bf16 Recurrent State Makes Hybrid Reasoning Models Think Longer | Abstract, contributions, Table 1 caption and the text near it (read by the examiner). |

All five suggested ids match their titles. The task gave no title for 2606.00206; its title is in the table.

Cited from project notes only (not opened in this run):

- 2606.26472 (EpiKV): errors, cap hits and finished-but-wrong counts under KV eviction (idea 12). The notes are in `harvest_reasoning_efficiency.md`. No copy of the paper is on disk.
- 2506.08390 (On Reasoning Strength Planning): correlation 0.84 between a probe on the question and the thinking length (idea 15). The notes are in `harvest_reasoning_efficiency.md` and `verification.md`. No copy of the paper is on disk.
- `reports/length-tax.html`: this project's AIME log, Qwen3-4B, 41% of samples at the 16,384 cap (idea 11); the wording of Parts A, B and C.

Things I could not verify, or that a reader should treat with care:

- Every [from notes] number in the list above.
- All [checked] facts were read in page copies saved on 11 Nov 2026, not fetched again. A later arXiv version may differ.
- Numbers from tables reached me through a text extraction of the HTML. I read the prose statements of the key numbers; I did not compare every table cell.
- In 2610.23818 the "3.30× against 1.10×" of idea 18 is the mean of λ over all cells of each pair (Table 3 caption, read by the examiner).
- The DEER loss of 6.7 points on AIME24 is from another group's reimplementation (2608.28859: greedy decoding, one model, against a never-exit baseline), not from the DEER paper.
- The toy models (ideas 2 and 19, Q4, Q16, Exercise 2) are mine. Geometric lengths are noisier than real traces, so the toy numbers show directions, not sizes.
- "None of the sources read for this pack reports a switch-back test" covers only the papers in the table above. I did not search for other work.

---

## 8. Verification log

Examiner pass, 2026-11-13 15:32 CST (time from `date`). CPU only, no GPU job. No network request was made: every source was read again in the page copies saved on 11 Nov 2026 in the scratch folder, or in the project notes.

**What was checked (48 items).** 20 core ideas, 8 misconceptions, 17 questions with their follow-ups, 3 exercises. The level ladder and the diagnostic picks were read too.

- Each question was solved before its key was read. Every number in the keys was computed again with a separate script. No arithmetic error was found.
- The code of Exercises 1 to 3 was copied out of this file and run again (Python 3.11.5, numpy 1.24.3, CPU). All three outputs are identical to the pasted outputs. Exercise 3 was also done by hand.
- Each paper fact was looked up again in the saved copy: the number, the setting, and what the paper claims.

**What changed.** Nothing was deleted.

1. **Q10 key (was wrong).** Point (5) said that a relaxed rule does not change the EoT probability. It does, wherever q differs from p. Point (4) said the drafter contrast shows "what the drafter could not see". The data do not show this: the drafter that inflates more is the EAGLE-3 head, which reads the target's hidden states. Both points are rewritten. The question now names the rule (mentored_dec, λ = 1.16) and gives the interval, so the premise is the paper's own.
2. **Q12 follow-up key (was wrong).** The old key said "Nothing" and "many of these runs are near the budget". The pairs exclude cut-off runs, so the second part had no support. New key: same direction as GPT-OSS-20B, size unknown.
3. **Q2 and idea 12 (were ambiguous).** The 1,132 runs are relaxed runs still thinking at a threshold near the budget, not all runs that hit the budget. The question now says that a cut-off run is scored wrong in that study. Key (c) now also names runs that reason on after the forced token.
4. **Q5, idea 14, misconception 2, Q15 point (5).** The within-problem AUROC is 0.496 under the pre-registered weighting and 0.515 under exact pair weighting; both are at chance. The pooled and the within-problem numbers use different sample sets. Added: the pooled AUROC itself falls to 0.586 at token 16, and one after-the-fact probe reaches 0.556 inside problems. "No probe is known" was too strong and is now "the one study read here".
5. **Q8 follow-up (was ambiguous).** "Too small" holds only if the answer does not change and change back between readouts. The assumption is now in the question and the opposite case is in the key.
6. **Q9 (c) and (d).** The paper compares the geometric mean with the arithmetic mean, not with the product. The product argument stays, marked as a derivation of this file. "Confidence follows problem ease" is now marked as a hypothesis.
7. **Idea 6 and misconception 6.** "At least 59% before the answer on AIME24" mixed two models. It now says 59% for GPT-OSS-20B and "too wide to use" for Qwen3-8B, with the intervals.
8. **Idea 10 and misconception 4.** "Does not happen after a natural EoT" is a result for one setting (R1-Distill-14B, DEER, MATH). Added the counter-case from 2610.23818 (11 of 30 outputs reopen a closed think block).
9. **Ideas 5, 15, 16, 17, 18, 20: setting added.** Primary pairs (5). The 60% is small and is 54% [48, 60] on a second suite (15). "Accuracy kept" is "kept or improved on average" (16). The 0.01 to 0.09 median is for four of the five rules, on traced GPT-OSS-20B runs (17). The 3.30× and 1.10× are means over all cells of a pair (18). The guard is described as in the paper (20).
10. **Citations corrected.** 2610.23818: Eq 4-5 are in Sec 2.2, not 2.3 (idea 3, Q1); Table F2 is in App F. 2609.03633: the result is in Sec 4.2, Fig 2. 2608.28859: the DEER loss is in Table 3, on AIME24, greedy decoding, R1-Distill-Qwen-7B.
11. **Tags raised from [from notes] to [checked].** 2506.09501 (9,000 tokens; abstract) and 2611.12127 (×1.07 to ×1.27 in five of six models, cap hits 118 → 223 of 590; abstract and Table 1). Both were read in saved copies and their ids match their titles.
12. **Small additions.** Q11: the fourth cell of the 2×2 and the EAGLE-3 case. Q13: one run per case, and λ moves from 0.96 to 1.60 between seeds. Q14: the two arms are selected differently before s. Q16: the question now says that a healthy trace stops with chance h0 in both modes. Q17: spec_casc_tok can also reject tokens that lossless accepts; "+445%" is "up to +445%". Q1: level 3, not 2 (the ladder puts the round ratio at level 3). Q4: a tutor note on the formula. Q3: a rounding note. Exercise 3: a note on trace T5.

**What stays unverified.**

- 2606.26472 (EpiKV): errors 104 → 172, cap hits 55 → 124, finished-but-wrong 49 → 48 (idea 12). Only the project notes were read. No copy of the paper is on disk.
- 2506.08390: correlation 0.84 (idea 15). Only the project notes were read. The notes say it is a correlation between questions.
- The 41% cap-hit share of Qwen3-4B on AIME (idea 11) is from `reports/length-tax.html`. The run log behind it was not opened.
- The arXiv ids were matched to titles inside the saved copies, not on arxiv.org. A later arXiv version may differ.
- Table numbers were read through a text extraction of the HTML. Cells that the prose does not repeat (parts of Table F2 of 2610.23818, Tables 4 and 5 of 2611.02278) were read only there.
- The DEER loss of 6.7 points is one reimplementation by another group, on one model.
- Q9 (d): "DEER's confidence follows problem ease" is a hypothesis. No source read here tests it.
- "No source reports a switch-back test" (idea 19) covers only the nine papers in Section 7. Nobody searched for other work.
- The toy models (ideas 2 and 19, Q4, Q7, Q16, Exercise 2) are not fitted to any data. They show directions, not sizes.

**How far to trust the pack.** The numeric keys and the exercise outputs can be used as they are. The paper facts tagged [checked] were read twice, by two readers, in saved copies. Treat the [from notes] numbers as hearsay, and treat the design keys (Q11, Q14) as one good answer, not the only one.
