# Tutor pack: experiment design and statistics for length effects

Subskill id: `experiment_stats`. Written 2026-11-13 15:20 CST. Examined and corrected 2026-11-13 by a second pass (see Section 8, Verification log). No GPU job was run. CPU-only numpy code was run in the scratch folder.

Scope: paired comparisons on the same prompts and seeds; the noise floor from numerical nondeterminism and why every arm needs a noise-only control; right-censoring by a length cap; heavy tails (mean versus median, share of the top decile); bootstrap intervals for a ratio of means; matched-dose controls; intervention designs (switch back to the exact method at token t; token-only versus KV-only repair) and what each outcome would mean.

Symbols used in this file. Case i is one (prompt, seed) pair. **x_i** is the number of generated tokens of the exact arm on case i. **y_i** is the number of generated tokens of the lossy arm on the same case. **n** is the number of cases. **λ = (Σ_i y_i) / (Σ_i x_i)** is the length inflation, a ratio of mean lengths. **C** is the length cap (maximum generated tokens). **B** is the number of bootstrap resamples. "Exact" means the lossless reference method (dense attention, strict verification, bf16 weights). "Lossy" means the method under test.

How to read the tags:

- **[checked]**: the fact was read in the paper text in this run, and read again by the examiner. For all four arXiv papers the text was a copy of the arXiv HTML page saved in the scratch folder. The title of each id was confirmed in that copy. No page was fetched again today.
- **[checked, local data]**: I recomputed the number in this run from the stored sample files in `results/reasoning/e3_shards/` (CPU only, read only).
- **[checked, local code]**: I read it in the repository code in this run.
- **[from notes]**: taken from `research_notes/Inference speed question search/verification.md` or `reports/length-tax.html`. The examiner confirmed that the notes say it. It is a plan or a judgment of the notes, not a paper fact.
- **[unverified]**: from memory only.
- No tag: a textbook fact or a derivation done here. Each one was also checked by the simulations in Section 5.

---

## 1. Core ideas

**Design and noise**

1. **The estimand.** The length inflation is λ = (Σ_i y_i) / (Σ_i x_i), computed on the same cases for both arms. It is the number that enters cost: the length-corrected speedup is S_relax ≈ (L̄_AR / L̄_relax) · S, where S is the acceptance-based speedup of the paper's Eq. 1 (it ignores length), L̄_AR is the mean length of plain decoding and L̄_relax is the mean length under the relaxed rule [checked: 2610.23818 Sec 2.2, Eq. 3].

2. **Why both arms use the same prompts.** Model the logs as log x_i = a_i + e_i and log y_i = a_i + δ + e'_i. Here a_i is the difficulty of prompt i (standard deviation τ across prompts), e_i and e'_i are independent run noise (standard deviation σ), and δ is the effect. The paired estimate, the mean of (log y_i − log x_i) over n prompts, has variance 2σ²/n. With different prompts in the two arms the variance is 2(τ² + σ²)/n. So pairing cuts the needed n by the factor (τ² + σ²)/σ².

3. **What a shared seed buys, and what it does not.** The kv-fetch engine draws one uniform number per step for each (problem, sample) and samples by inverse CDF, so every method sees the same random numbers [checked, local code: `kvfetch/sparse_decode.py`, function `sample_token`]. Two arms emit the same tokens until the first step where they pick different tokens. After that the contexts differ and the two runs are close to independent; only the prompt and the prefix before the split are still shared [from notes: verification.md F38]. In the stored E3 data, 0 of the 93 seed-paired pairs that finished in both arms have the same length. Any leftover link is small: inside a problem, the correlation of log length between the two runs of a seed is 0.18, 0.17, −0.11 and 0.00 in four lossy runs, while shuffled seeds give 0 ± 0.08 [checked, local data].

4. **Numerical noise alone makes runs diverge.** Floating-point addition is not associative: (a + b) + c ≠ a + (b + c). So batch size, GPU count and GPU type change the last bits of the logits. With greedy decoding in bf16, over 90% of the MATH500 examples of DeepSeek-R1-Distill-Qwen-7B diverged across 12 such settings; in FP32 only 2.2% did [checked: 2506.09501 Sec 3.2]. "Diverged" means that at least two of the 12 outputs differ. It is not the share for one pair of settings. The paper does not report that share; it can only be lower or equal.

5. **How large that noise is.** The paper ran 12 runtime settings (2 GPU types × 2 GPU counts × 3 batch sizes) with greedy decoding. For DeepSeek-R1-Distill-Qwen-7B in bf16 on AIME'24 (cap 32,768), the standard deviation of one example's output length across the 12 settings, averaged over examples, was 9,190 tokens, and the standard deviation of accuracy was 9.15 points. On MATH500 the values were 2,774 tokens and 1.04 points. In FP32 they were 0 and 0 on AIME'24, and 139 tokens and 0.12 points on MATH500 [checked: 2506.09501 Sec 3.1, Tables 3 and 4]. Note: the abstract speaks of "9,000 tokens difference in response length". The only matching number in the tables is this per-example standard deviation (Table 4), and Sec 3.2 points to that table. So I read the 9,000 as this standard deviation. It is not a difference between mean lengths.

6. **The noise-only control (zero dose).** Any change, even in the last bit, flips some near-tie between two tokens. After that the run is a new sample. So "the lengths changed" proves nothing. Every experiment needs an arm that uses no lossy method but still lets runs diverge: the exact method with other seeds, or the exact method with the same seeds and a different batch order or concurrency [from notes: verification.md F17 arm 0, F38 controls]. Call its ratio λ₀. A lossy arm's λ counts only by how far it lies outside the spread of λ₀ around 1.

7. **Size of the floor in practice.** In 2610.23818 the seed-to-seed standard deviation of λ had a median of 0.08 over 28 GPT-OSS-20B cells (largest 0.27, on AIME24 with 30 cases) and a median of 0.03 on Qwen3-8B [checked: 2610.23818 App. C]. In the toy A/A test of Exercise 1 the standard deviation of λ₀ is 0.075 with 200 prompts and 0.18 with 30 prompts.

8. **A replicate should not fall inside a 95% interval 95% of the time.** Take two independent estimates with the same standard error s. Their difference has standard deviation s√2. So the second estimate lies inside the first one's interval (± 1.96 s) with probability 2Φ(1.96/√2) − 1 = 0.83, where Φ is the standard normal CDF. Two replicates both lie inside with probability about 0.72 (simulated here). In 2610.23818 the seed-0 interval contained both new seeds in 21 of 28 cells, which is 0.75 [checked: 2610.23818 App. C]. That is about what this simple model predicts. Caveat: the match is loose. The new seeds reuse the same cases, which should push the share above 0.72 (if only half of the variance is run noise, the model gives 0.91). They also ran on another server [checked: 2610.23818 App. C], which can push it down. So 0.75 fits the model but does not test it.

9. **Every lossy arm also carries lossless side changes. Each needs its own twin.** A nucleus gate at T = 1 is itself a sampler truncation, so it needs an exact arm with top-p 0.95 [from notes: verification.md F07]. A relaxed rule may close the reasoning earlier, so it needs an early-close control [from notes: verification.md F01]. 2610.23818 ran two such twins under the lossless rule: other draft lengths (λ from 0.96 to 1.06 on GPT-OSS-20B) and temperature 1.2 (GSM8K length 1.27 times, interval [1.10, 1.47]) [checked: 2610.23818 App. C].

**Censoring by the cap**

10. **Right-censoring and the ceiling.** With a cap C we observe min(L, C), not the length L. The capped ratio λ_C = Σ_i min(y_i, C) / Σ_i min(x_i, C) can never be larger than C divided by the mean capped length of the exact arm. E3 example: the exact arm has mean 12,385 tokens at C = 16,384, so the ceiling is 16,384 / 12,385 = 1.32. The measured 1.094 (interval [1.052, 1.150]) for oracle@512 already uses 29% of that room [checked, local data; the same values are in verification.md F07].

11. **Direction of the bias.** The function min(·, C) never stretches a difference. So if y_i ≥ x_i for every case, the capped difference of means is at most the true difference. The ratio usually moves toward 1 as well, because the extra tokens sit in the long runs. 2610.23818 says that cut-off runs make its λ a lower bound [checked: 2610.23818 Sec 3.3 and Limitations]. This is not a theorem. If the extra tokens sit in short runs and the long runs do not change, the cap shrinks the denominator and λ_C goes up (Exercise 2, third scenario; question Q16).

12. **What stays valid under a fixed cap.** All runs share the same cap. So the survival curve S(t) = share of runs not yet finished at token t is exact for every t < C. A quantile q is exact when q < 1 − (cap-hit rate). The mean capped length equals the area under S(t) from 0 to C. Report for each arm: the cap-hit rate as its own outcome, the survival curve, and the capped mean. E3: the exact arm hits the cap in 41% of runs, so its median (14,008) is valid. Oracle@512 hits it in 57%, so its median is the cap itself [checked, local data].

13. **Dropping capped runs is selection on the outcome.** "Keep only cases that finished in both arms" removes mostly the runs with the largest increase. The result can move up or down. In E3 it gives 1.149 on 93 of 240 pairs, against 1.094 on all pairs [checked, local data]. In Exercise 2 (tail scenario, cap 4,096) it gives 1.10 when the true ratio is 1.46. 2610.23818 uses it only as a robustness check: it "moves λ by at most 0.1 in 45 of the 60 cells" [checked: 2610.23818 Sec 3.7]. A low-truncation subset picked after seeing the data (E3: 11 problems, 1.246, interval [1.132, 1.416]) is post hoc and cannot be the main result [checked, local data; flagged in verification.md F07].

**Heavy tails and intervals**

14. **Mean and median answer different questions.** Over the 360 cells of 2610.23818, λ has median 1.08 and mean 1.54 [checked: 2610.23818 Sec 3.3]. Inside a cell the same thing happens. On the 60 cells of the two primary model pairs, the tenth of cases with the largest increase holds 59% of the net extra tokens (median over cells), and the 90th percentile of the per-case ratio has a median of 2.2 [checked: 2610.23818 Sec 3.7]. Top-decile share = (sum of d_i over the 10% of cases with the largest d_i) / (Σ_i d_i), with d_i = y_i − x_i. It can be above 100% when some d_i are negative. Cost follows the mean. "What happens to a typical request" follows the median. Report both.

15. **Ratio of means is not mean of ratios.** Take an A/A test where each run has independent lognormal noise with log standard deviation σ. Then log(y_i / x_i) is normal with mean 0 and variance 2σ², so the expected value of y_i / x_i is exp(σ²). For σ = 0.5 that is 1.28, although nothing changed. Use Σy / Σx for cost. Use the geometric mean exp(mean of log(y_i / x_i)) or the median for the typical case. Do not use the plain mean of per-case ratios.

16. **Paired bootstrap for λ.** Draw n case indices with replacement. Keep both arms of each drawn case. Compute λ* = Σ y / Σ x on the drawn cases. Repeat B times. The 2.5% and 97.5% quantiles of the B values of λ* are the 95% percentile interval. 2610.23818 uses B = 10,000 over cases [checked: 2610.23818 Sec 3.1]. Resample the unit that is independent. With several seeds per problem, resample problems, and seeds inside them if seeds should count as random (the "two-level" bootstrap of 2610.23818 App. C [checked]). E3: resampling the 240 samples gives [1.068, 1.122]; resampling the 30 problems gives [1.052, 1.150] [checked, local data].

17. **What the interval does not cover.** It covers the sampling of cases only. It does not cover the bias from the cap, or a choice made after seeing the data. With few heavy-tailed cases it is also too narrow: in Exercise 1 at n = 30 the A/A interval excludes 1 in 8.9% of trials, not 5%. 2610.23818 App. F shows the tail risk: a guard tuned on 8 AIME24 cases cut their mean length from 13,364 to 10,534 tokens, but on all 30 cases it moved the mean from 9,270 up to 10,363. On the eight tuning cases, two runs that reached the 32,768 cap held 61% of all output tokens (2 × 32,768 / (8 × 13,364) = 0.61) [checked: 2610.23818 App. F].

18. **Power.** The half-width of the interval shrinks like 1/√n, where n counts independent units (problems), not seeds per problem. E3 half-widths on 11 problems × 8 seeds run from 0.06 (oracle@1024) to 0.18 (quest@512) [checked, local data]. The notes conclude that about 400 runs per arm are needed to tell 1.05 from 1.15, and about 960 samples per arm to tell "closes one third" from "closes two thirds" of a 28-point cap-hit gap [from notes: verification.md F07, F38]. Fix the thresholds and a "partial" band (for example 1.05 to 1.15) before the run [from notes: verification.md F07].

**Interventions**

19. **Matched dose (Part A).** "Dose" has two parts: how often a method changes a token, and how unlikely the changed token is under the full model [from notes: length-tax.html]. A matched control must equal both. Two ways: the same substitution rate with the same rank profile, or the same mean realized surprisal s_t = −ln p(x_t), where p is the full model's next-token distribution and x_t is the emitted token. Temperature alone is a weak match. It samples from p's own tail, while a relaxed rule admits tokens that the drafter likes and the target does not: for four of the five relaxed rules, the tokens that only the relaxed rule accepts have median target probability 0.01 to 0.09 and mean target rank 37 to 234. The fifth rule, spec_casc_tok, admits mild tokens (median probability 0.29, mean rank 0.9) [checked: 2610.23818 Sec 3.7]. Reading: if random substitutions at matched dose give the same λ as the method, dose is enough. If the method gives more, something besides dose matters.

20. **Switch-back (Part B) and the 2×2 repair design (Part C).**
    - *Switch-back.* Run the lossy method up to token t, then the exact method. Pure per-token cost (memoryless) predicts extra tokens only from the lossy phase: at most t in any run, and about t · (1 − 1/λ) on average for runs that last longer than t, where λ is the always-on inflation. (Derivation: t lossy tokens make t/λ tokens of progress, so t − t/λ tokens are extra.) Lasting damage (hysteresis) predicts that some runs stay long after the switch. The two can give the same mean (Exercise 3; there the match comes from the toy's parameters, it is not a law). So compare the distribution: the share of runs whose extra tokens exceed t, and the upper quantiles. Memoryless means a per-token budget or gate is enough. Hysteresis means rollback or prevention is needed [from notes: length-tax.html, verification.md F01].
    - *2×2.* Four arms with mean lengths L_D, L_U, L_T, L_S. D: exact token choice, exact KV. U: lossy token choice, lossy KV. T: lossy token choice, KV rewritten exactly (KV-only repair; only the token channel is left). S: exact token choice that reads lossy-written KV (token-only repair; only the state channel is left) [from notes: verification.md F07, F38]. Token-channel share = (L_T − L_D) / (L_U − L_D). State-channel share = (L_S − L_D) / (L_U − L_D). Interaction = L_U − L_T − L_S + L_D. The two shares sum to 100% only when the interaction is 0. Reading: T ≈ D means the stored KV carries the harm and a rewrite of the KV is enough. This is the ReSA idea: every few tokens, re-encode the recent tokens with dense attention to refresh the KV cache, and keep the tokens [checked: 2506.04108 abstract and Sec 2.2]. S ≈ D means the emitted token carries it and tokens must be verified. Neither means both channels matter. If T ≈ D and S ≈ D but U is long, the harm needs both channels at once (pure interaction); then either repair is enough.

---

## 2. Level ladder

- **Level 2.** Can say why both arms use the same prompts and seeds, what a cap does to a mean, and why mean and median differ for long-tailed lengths. Knows that a noise-only arm is needed, but cannot yet say how large the noise is.
- **Level 3.** Computes λ, the ceiling C / (mean capped exact length), the top-decile share and the 2×2 shares from given numbers, and can write a paired bootstrap. With a hint, explains why dropping capped runs is biased and why the mean of per-case ratios is not λ.
- **Level 4.** Without help: picks the resampling unit, sizes the noise floor and the sample count, says which statistics stay valid under the cap, builds a matched-dose control, and states what each outcome of the switch-back and the 2×2 design would mean.
- **Level 5.** Finds hidden flaws in a full design: survivor selection at the switch point, a lossless side change inside a lossy arm, a post hoc subset, a mean that cannot separate two mechanisms. Designs the Part A, B and C experiments with decision rules fixed in advance, and can explain each control to someone else.

---

## 3. Misconceptions

1. **"Greedy decoding is deterministic, so one run per arm is enough."** Wrong. In bf16 the result depends on batch size, GPU count and GPU type; over 90% of MATH500 examples diverged across 12 runtime settings in one test [checked: 2506.09501 Sec 3.2]. And any lossy method changes the logits, so its greedy run diverges at the first near-tie whether or not the method does harm.

2. **"The same seed means the two arms share their randomness for the whole run."** Wrong. They share it only until the first different token. After that the same uniform number is applied to a different distribution in a different context. Pairing then helps only through the shared prompt.

3. **"The cap only removes a few long runs, so the ratio is still about right."** Wrong. The capped ratio cannot exceed C / (mean capped exact length). With 41% of exact runs at the cap, that ceiling is 1.32, and 37% of the E3 pairs are capped in both arms and add a ratio of exactly 1 [checked, local data].

4. **"Censoring always makes the ratio too small."** Mostly true, not always. If the extra tokens are in short runs, the cap removes unchanged long tokens from both arms and the ratio rises (Q16, Exercise 2).

5. **"Keep only the runs that finished in both arms; then the cap does not matter."** Wrong. That selects on the outcome. It removes the runs with the largest increase and can even push the ratio below 1 (Exercise 2, tail scenario at cap 1,500: 0.989).

6. **"The bootstrap interval is the full uncertainty."** Wrong. It covers case sampling only. It misses cap bias and post hoc choices. It also misses anything that all cases of one run share, such as one server, one batch order or one software build. With 30 heavy-tailed cases it is also too narrow (Exercise 1).

7. **"Matching how often tokens change matches the dose."** Wrong. A changed token with full-model probability 0.3 adds 1.2 nats of surprisal; one with probability 0.05 adds 3.0 nats. The control must also match how unlikely the tokens are.

8. **"If the mean length after switch-back equals the per-token prediction, there is no lasting damage."** Wrong. A rare trigger with a constant hazard can give the same mean as a per-token cost (Exercise 3: 1.058 against 1.060). So an equal mean proves nothing. A mean far from the per-token prediction would speak against a pure per-token cost. An equal mean does not speak for it. The tail of the distribution separates the two.

---

## 4. Question bank

Each question is self-contained. The tutor asks it as written, waits, and uses the follow-up only after a wrong or partial answer.

### experiment_stats-Q1

- **Level:** 2 to 3 (level 2: says that the cap bounds the ratio; level 3: gets both numbers). **Type:** predict (numeric).
- **Question:** "Every run is cut off at a cap of 16,384 tokens. With this cap, the exact arm has a mean length of 12,385 tokens. A lossy arm shows a ratio of mean lengths of 1.094. What is the largest ratio any lossy arm could show in this setup, and what share of the possible increase does 1.094 use?"
- **Answer key:** The lossy mean cannot exceed the cap. Largest ratio = 16,384 / 12,385 = 1.323. Share used = (1.094 − 1) / (1.323 − 1) = 0.094 / 0.323 = 29%. So 1.094 is not "9% more thinking"; it is 29% of everything this setup can show. (Numbers are the stored E3 values [checked, local data].)
- **Likely wrong answer:** "There is no limit. 1.094 means the model thinks 9% longer."
- **Follow-up:** "Suppose the lossy method made every run endless. What ratio would this table show?" (1.323. So a true ratio of infinity reads as 1.32 here.)

### experiment_stats-Q2

- **Level:** 2 to 3 (level 2: says that only the prompt term cancels; level 3: gets 3.56). **Type:** predict (numeric).
- **Question:** "Log length has a standard deviation of 0.8 across prompts, and 0.5 across repeated runs of one prompt. Each of two arms runs 200 prompts, one run per prompt. The two runs of a prompt split early, so their run noise is independent. You estimate the log ratio as the difference of the two mean log lengths. How many times larger is the variance of this estimate if the two arms use different prompts instead of the same prompts?"
- **Answer key:** Same prompts: the prompt term cancels. Variance = 2 × 0.5² / 200 = 0.0025, standard error 0.050. Different prompts: variance = 2 × (0.8² + 0.5²) / 200 = 2 × 0.89 / 200 = 0.0089, standard error 0.094. Factor = 0.89 / 0.25 = 3.56. So unpaired needs 3.56 times more runs for the same precision.
- **Likely wrong answer:** "Pairing with the same seed removes all the noise, so the paired variance is about zero."
- **Follow-up:** "After the first different token the two runs are independent samples for the same prompt. Which standard deviation does pairing remove then: the 0.8 or the 0.5?" (Only the 0.8. The 0.5 stays in both arms.)

### experiment_stats-Q3

- **Level:** 3. **Type:** find the flaw.
- **Question:** "A paper tests quantized reasoning models with greedy decoding (temperature 0) on six benchmarks of at most 100 questions each. It gives one length number per model and benchmark, with no interval and no repeated run. On average over the six benchmarks, reasoning length grows by 6.0% with one 4-bit quantizer and by 8.2% with another. A reader concludes that the second quantizer disturbs reasoning more. What is missing before you can believe that?" (Setup and numbers: arXiv 2606.25519, Qwen3-4B at INT4, HQQ +6.0% and AWQ +8.2%; 100 questions per benchmark, 30 for AIME25; vLLM at temperature 0 [checked: 2606.25519 Sec 3 and Table 2]. The text has no word about repeated runs or intervals [checked by text search]; "one run" is my reading. The "reader" is invented; the paper is not quoted as making this claim.)
- **Answer key:** A noise-only arm and intervals. (1) Greedy bf16 runs are not repeatable across batch size or GPU setup, so one run is one sample. (2) Any weight change flips near-ties, and then each answer is a new sample; a change in length for one question means nothing. (3) With 100 long-tailed questions the ratio of an unchanged replicate may be off by several percent (Exercise 1: standard deviation 0.075 with 200 prompts). Needed: the bf16 baseline run at least three times with different request order and concurrency, a paired bootstrap interval over questions for each ratio, and the two quantizers compared on the same questions. Extra fact for the tutor: per benchmark the order is not stable. AWQ is higher on three of the six benchmarks and HQQ on the other three (GSM8K: −2.7% against +5.5%; MBPP: +36.4% against +19.3%) [checked: 2606.25519 Table 2].
- **Likely wrong answer:** "Nothing. Greedy decoding is deterministic, so one run is exact, and 8.2% is larger than 6.0%."
- **Follow-up:** "You rerun the unquantized bf16 model, still greedy, in 12 runtime settings: 2 GPU types, 2 GPU counts, 3 batch sizes. For a reasoning model on MATH500, about what share of questions gives the same tokens in all 12 runs?" (Under 10%: over 90% of examples diverged across the 12 settings [checked: 2506.09501 Sec 3.1, 3.2]. For only two settings the share of identical answers is higher. The paper does not report it, so do not quote 90% for one pair.)

### experiment_stats-Q4

- **Level:** 3. **Type:** predict (numeric).
- **Question:** "One cell has 10 cases. The exact arm takes 1,000 tokens on every case. The lossy arm takes 800, 950, 1,000, 1,050, 1,080, 1,100, 1,300, 1,600, 2,350 and 4,200 tokens. Compute the ratio of means, the median per-case ratio, and the share of the net extra tokens held by the one case with the largest increase. Which number do you give to the person who pays for tokens?"
- **Answer key:** Lossy sum = 15,430, exact sum = 10,000, so λ = 1.543. Median per-case ratio = (1.08 + 1.10) / 2 = 1.09. Net extra = 5,430 tokens; the largest case adds 3,200; share = 3,200 / 5,430 = 58.9%. The payer gets 1.54. The median 1.09 describes a typical request. (The toy numbers copy two patterns in 2610.23818. Across its 360 cells, λ has mean 1.54 and median 1.08. Inside a cell, on the 60 primary-pair cells, the top-decile share is 59% [checked: 2610.23818 Sec 3.3, 3.7].)
- **Likely wrong answer:** "About 1.09. The 4,200 run is an outlier and should be dropped."
- **Follow-up:** "Drop the 4,200 case. What is the ratio of means on the other nine? Did the token bill change?" (11,230 / 9,000 = 1.248. The bill still contains the 3,200 extra tokens. Dropping the case changes the report, not the cost.)

### experiment_stats-Q5

- **Level:** 3. **Type:** explain why.
- **Question:** "A colleague says: 'Capped runs are censored, so I keep only the cases that finished in both arms. That ratio is free of the cap.' In a simulation with 200,000 cases, 10% of lossy runs become 5 times longer and the true ratio of means is 1.46. With a cap of 4,096, the both-finished ratio is 1.10. Why is it so far off?"
- **Answer key:** The filter selects on the outcome. The filter removes 60% of the runs that became 5 times longer. These are the long ones, and they hold 78% of all extra tokens. What remains is mostly the unaffected 90%. The cases are not missing at random. Report instead: the cap-hit rate per arm, the survival curves, the quantiles q with q < 1 − (cap-hit rate), and the capped ratio as a bound; or raise the cap, or use a task where the exact arm rarely hits it. (Numbers from Exercise 2.)
- **Likely wrong answer:** "The sample got smaller, so the estimate is noisier."
- **Follow-up:** "There are 200,000 cases. Can sampling noise move the ratio from 1.46 to 1.10? Then which runs did the filter remove?" (No. It removed the long, damaged runs.)

### experiment_stats-Q6

- **Level:** 3. **Type:** find the flaw (numeric).
- **Question:** "You run the exact method twice with different seeds on 200 prompts. The mean of the per-prompt ratios y_i / x_i is 1.31. The ratio of total tokens is 1.05 with a 95% interval [0.91, 1.22]. A student says the second run thinks 31% longer. What is wrong, and what do you predict for the mean of the inverse ratios x_i / y_i?"
- **Answer key:** Nothing differs between the two runs, so the true ratio is 1. The mean of ratios is biased upward: with lognormal run noise of log standard deviation σ in each run, the expected value of y_i / x_i is exp(σ²); σ = 0.5 gives 1.28. By symmetry the mean of x_i / y_i is also well above 1, about 1.3 (theory 1.28; this simulated test gave 1.25). Accept any answer that says "also above 1, about the same size". Both runs cannot be about 30% longer than the other, so this statistic is not a length effect. The ratio of totals, 1.05, has an interval that contains 1. (Numbers from Exercise 1, part a.)
- **Likely wrong answer:** "The mean of x_i / y_i will be 1 / 1.31 = 0.76, so the second run really is longer."
- **Follow-up:** "Two prompts have (x, y) = (1,000, 2,000) and (2,000, 1,000). Compute the mean of y/x, the mean of x/y, and the ratio of totals." (1.25, 1.25, and 1.00.)

### experiment_stats-Q7

- **Level:** 3. **Type:** apply to a new case (numeric).
- **Question:** "The exact arm hit the cap in 99 of 240 runs. The runs are 30 problems × 8 seeds. Five problems hit the cap in all 8 runs and ten problems never did. A student reports the 95% interval as ±6 points, from sqrt(p(1 − p) / 240). Compute that standard error, compute it again as if each problem were all-or-nothing, and say which is closer to the truth."
- **Answer key:** p = 99 / 240 = 0.4125. Naive: sqrt(0.4125 × 0.5875 / 240) = 0.032, so ±6.2 points. All-or-nothing: only 30 independent units, sqrt(0.4125 × 0.5875 / 30) = 0.090, so ±17.6 points. Half of the problems are all-or-nothing, so the truth is closer to the second. Measured on the stored data, the standard error over problems is 0.069, so ±13.6 points [checked, local data]. The right tool is a bootstrap over problems. (Counts are the stored E3 values [checked, local data]. For the paired difference oracle@512 minus exact, the problem bootstrap gives +15.8 points, interval [8.3, 23.3] [checked, local data].)
- **Likely wrong answer:** "±6 points. 240 runs are 240 observations."
- **Follow-up:** "You add 8 more seeds for the five problems that always hit the cap. How much do you learn about the cap-hit rate?" (Almost nothing. They will hit the cap again. New problems are what add information.)

### experiment_stats-Q8

- **Level:** 4. **Type:** find the flaw. **Tied to Part A.**
- **Question:** "A student wants to test whether dose alone explains the longer reasoning under a relaxed speculative rule, which gives λ = 1.50. As a control he raises the temperature of the exact model until the share of emitted tokens that are not the top-1 token equals that share in the relaxed arm. The control gives λ = 1.05. He concludes: 'The dose is matched and the control is not longer, so dose does not explain it.' Find the flaw."
- **Answer key:** Only half of the dose is matched: how often, not how unlikely. Temperature samples from the target's own distribution, only flattened (probability proportional to p^(1/T)). So its non-top-1 tokens are mostly tokens the target itself rates fairly likely. The relaxed rule admits tokens the drafter prefers: for four of the five rules in 2610.23818, median target probability 0.01 to 0.09 and mean target rank 37 to 234 [checked: 2610.23818 Sec 3.7]. A fair control matches the rate and the rank profile (random substitutions drawn to the same profile [from notes: verification.md F07]), or the mean realized surprisal −ln p(x_t). It also needs a noise-only arm and intervals. Only if such a control stays short does "more than dose" follow.
- **Likely wrong answer:** "The control is fine. Same rate of non-top-1 tokens means same dose."
- **Follow-up:** "In the relaxed arm a changed token has full-model probability about 0.05. In the temperature arm a non-top-1 token has probability about 0.3 (an example value, not from a paper). How many nats of surprisal does each one add?" (−ln 0.05 = 3.0 and −ln 0.3 = 1.2. Same rate, 2.5 times less surprisal.)

### experiment_stats-Q9

- **Level:** 4. **Type:** apply to a new case (numeric). **Tied to Part A.**
- **Question:** "Define dose as (share of tokens substituted) × (−ln p of the substituted token under the full model). Arm X substitutes 10% of tokens, each with full-model probability 0.05. Arm Y substitutes tokens with full-model probability 0.001. At what substitution rate does Y have the same dose as X? Then: at that matched dose X gives λ = 1.10 and Y gives λ = 1.40, each with an interval of ±0.04. What does this say about a single dose curve?"
- **Answer key:** X: 0.10 × (−ln 0.05) = 0.10 × 2.996 = 0.300 nats per token. Y: each substitution adds −ln 0.001 = 6.908 nats, so the rate is 0.300 / 6.908 = 4.3%. Same dose, different λ, and the gap of 0.30 is far outside ±0.04. So one curve in this dose measure does not hold: rare, very unlikely tokens cost more than frequent mild ones. That looks like a trigger effect. Caution: it may only show that the dose scale is wrong. Test three or more dose levels per arm. Then check whether another dose measure (for example rate × surprisal squared, or the rate of tokens below a probability threshold) puts both arms on one curve.
- **Likely wrong answer:** "Y also needs 10%. Dose is the substitution rate."
- **Follow-up:** "At 10% in both arms, which arm adds more surprisal per token, and by what factor?" (Y, by 6.908 / 2.996 = 2.3.)

### experiment_stats-Q10

- **Level:** 4. **Type:** predict (numeric). **Tied to Part B.**
- **Question:** "Exact runs need 1,000 tokens on average. With the lossy method always on, the mean is 1,400. Now you run the lossy method for the first 200 tokens and then switch to the exact method. Assume a pure per-token cost: on average a lossy-phase token gives 1/1.4 of a token of progress, the waste differs from run to run, and nothing is left behind after the switch. Every run lasts longer than 200 tokens. Predict the mean total length. Then give a hard upper bound for the extra length of any single run."
- **Answer key:** Progress during the lossy phase = 200 / 1.4 = 142.9 tokens of exact work. Remaining = 1,000 − 142.9 = 857.1. Total = 200 + 857.1 = 1,057. So λ = 1.057 and the mean extra is 200 × (1 − 1/1.4) = 57 tokens. No run can have more than 200 extra tokens, because extra tokens arise only in the lossy phase. The bound is reached by a run that made no progress at all in those 200 tokens. (Exercise 3 gives 1.058.)
- **Likely wrong answer:** "1,000 + 0.4 × 200 = 1,080." (This uses 0.4 extra tokens per lossy-phase token. The 0.4 is per token of progress.)
- **Follow-up:** "With the method always on, 1,400 tokens are generated and 400 are extra. What share of generated tokens is extra? Apply that share to 200 tokens." (400 / 1,400 = 28.6%; 57 tokens.)

### experiment_stats-Q11

- **Level:** 4. **Type:** connect two ideas. **Tied to Part B.**
- **Question:** "A toy simulation has two models. In model M each bad token wastes 8 tokens and then the run goes on as before. In model H each bad token has a small chance to break the run for good, which makes it 5 times longer. Both give λ = 1.40 when the lossy method is always on. After a switch to the exact method at token 200 they give λ = 1.058 and 1.060. Connect this with what you know about heavy tails: which statistic of the switch-back runs separates the two models, and what does each model predict for it?"
- **Answer key:** The mean is the wrong tool. H has a constant hazard, so its expected damage grows in proportion to the exposure t, like a per-token cost. In this toy the size matches too. That is not a law: with all runs equally long, H's mean extra is about 45% above the per-token prediction, and with a very wide length spread about 25% below (checked in the scratch script `hslope.py`). So a mean that equals the per-token prediction does not rule out lasting damage. The tail can separate the models. The learner needs the statistic and the direction, not the numbers. Share of runs with more than 200 extra tokens: M gives 0, H gives 1.6%. Shift of the 99th percentile of length: M 43 tokens, H 1,115 tokens. Share of the net extra tokens in the top tenth of runs: M 0.14, H 1.00. In M every run pays a little. In H a few runs pay everything. (Numbers from Exercise 3.)
- **Likely wrong answer:** "Switch later. With a larger t the two means will separate."
- **Follow-up:** "At t = 800 the simulation gives 1.218 and 1.224. Did a later switch separate the means?" (No. Look at the tail, not the mean.)

### experiment_stats-Q12

- **Level:** 5. **Type:** find the flaw. **Tied to Part B.**
- **Question:** "Here is a design. Task: AIME with a cap of 16,384 tokens; the exact arm hits the cap in 41% of runs. Run the lossy arm, and at token 2,000 switch to the exact method. Take the runs that are still going at token 2,000 in each arm. Compare the mean number of tokens after token 2,000 between the switched arm and the exact arm. If the two means are equal, conclude 'no lasting damage'. Find at least three flaws."
- **Answer key:** (1) Censoring: the remaining length is capped at 14,384 and many runs hit it, so the cap pushes the two means together. (2) Survivor selection: "still going at 2,000" is a different set of cases in each arm. A lossy prefix delays easy cases past 2,000, so the switched arm keeps easier cases and its mean remaining length drops; that hides damage. (3) No noise floor for the remaining length: there is no exact-to-exact branch. (4) "Equal" is the wrong null: a pure per-token cost already predicts about 2,000 × (1 − 1/λ) extra tokens after the switch, because the lossy prefix made less progress. (5) The mean cannot separate a rare trigger from a per-token cost; the tail must be reported. Better design: a task with exact cap-hit under 10%; branch from a fixed prefix (same lossy prefix, continue exact with several seeds; same exact prefix, continue exact with several seeds); several values of t; outcomes are the share of runs with extra above t and the survival curves; thresholds fixed in advance.
- **Likely wrong answer:** "Only the cap is a problem. Raise it and the design is fine."
- **Follow-up:** "One problem is finished by the exact arm at token 1,800 and by the lossy arm at token 2,500. In which arm's 'still going at 2,000' set does it appear? Is it an easy or a hard problem, and what does it do to that arm's mean remaining length?" (Only in the switched arm. Easy. It lowers that mean, so damage looks smaller.)

### experiment_stats-Q13

- **Level:** 4. **Type:** predict (numeric). **Tied to Part C.**
- **Question:** "Four arms have these mean lengths. D: exact token choice, exact KV: 1,000. U: lossy token choice, lossy KV: 1,400. T: lossy token choice, KV rewritten exactly: 1,320. S: exact token choice that reads lossy-written KV: 1,060. Compute the share of the gap carried by the token channel alone, the share carried by the state channel alone, and the interaction. Which channel must a fix address?"
- **Answer key:** Gap = 1,400 − 1,000 = 400. Token channel alone (arm T): (1,320 − 1,000) / 400 = 80%. State channel alone (arm S): (1,060 − 1,000) / 400 = 15%. Interaction = 1,400 − 1,320 − 1,060 + 1,000 = 20 tokens, 5% of the gap. Seen as repairs: KV-only repair (U to T) closes 80 / 400 = 20%; token-only repair (U to S) closes 340 / 400 = 85%. The token channel carries the harm, so tokens must be verified; rewriting KV is not enough. Note: arm S pays for full attention, so it is a diagnostic arm, not a product [from notes: verification.md F38].
- **Likely wrong answer:** "KV-only repair closes 80%, because T is at 80%."
- **Follow-up:** "In arm T, which part is still lossy: the token choice or the stored KV? So does 1,320 show what is left after repairing the KV, or what the KV repair removed?" (The token choice is still lossy. 1,320 is what is left. The repair removed only 80 tokens.)

### experiment_stats-Q14

- **Level:** 5. **Type:** find the flaw. **Tied to Part C.**
- **Question:** "A 2×2 run on 30 problems × 8 seeds gives this result. KV-only repair closes 45% of the cap-hit gap between the exact and the lossy arm. Token-only repair closes 50%. Each share has a 95% interval of about ±25 points. The student writes: 'The two channels carry equal harm, and they add up, because 45 + 50 is about 100.' What is wrong, and what would you change?"
- **Answer key:** (1) ±25 points cannot tell one third from two thirds, so "equal" is not shown. (2) "They add up" is not shown either. If the two shares were independent, the sum would have an interval of about ±35 points (sqrt(25² + 25²)), so it could be 60 or 130. They are not independent: both use arms U and D. So the interval of the sum must come from the joint bootstrap, and it is usually wider. (3) Each share is a ratio whose denominator, the gap, is itself noisy; all four arms must be resampled together by problem. (4) Ceiling: 5 problems hit the cap in every exact run, so they cannot show any increase. 10 problems never hit it in the exact arm and show little (16% cap-hit in the stored oracle@512 run). So the whole gap rests on few problems [checked, local data; the counts 5 and 10 are also in verification.md F38]. (5) No noise floor: an exact-versus-exact replicate must show how much the cap-hit rate moves with nothing changed. Changes: about 960 samples per arm, a task where the exact arm rarely hits the cap, length ratio and survival curves next to cap-hit rate, and decision bands fixed in advance. Example: "one channel dominates" needs one repair to close at least 80% and the other at most 20%. "Additive" needs the sum of the two repair shares between 85% and 115%, because the sum is 100% plus the interaction share. Anything else is reported as "mixed".
- **Likely wrong answer:** "It is fine. Shares that sum to about 100% prove the channels are additive."
- **Follow-up:** "Suppose each share is known to ±25 points and the two estimates were independent. What is the interval of their sum? Can it tell 'additive' from 'strong interaction'?" (About ±35 points: from 60% to 130%. No.)

### experiment_stats-Q15

- **Level:** 4. **Type:** apply to a new case (numeric).
- **Question:** "On 11 problems × 8 seeds (88 runs per arm) the 95% interval of λ has a half-width of 0.10. You must tell λ = 1.05 from λ = 1.15, so you want a half-width of 0.05. First: how many runs per arm does the 1/√n rule ask for? Second: you can run 32 seeds on the same 11 problems, or 8 seeds on 44 problems. Which choice does the rule describe, and why?"
- **Answer key:** Halving the half-width needs 4 times the units: 4 × 88 = 352 runs per arm (the notes round up to 400 [from notes: verification.md F07]). The rule holds for 44 problems × 8 seeds. The independent unit is the problem. The variance of one problem's estimate is b² + w²/m, where b² is the spread of the effect between problems, w² the noise within a problem and m the number of seeds. More seeds shrink only w²/m. More problems shrink everything.
- **Likely wrong answer:** "352 runs, any way you like. 11 problems × 32 seeds is the easier setup."
- **Follow-up:** "Suppose you run unlimited seeds on the same 11 problems. Does the interval for λ on new problems of this kind shrink to zero?" (No. b²/11 remains.)

### experiment_stats-Q16

- **Level:** 4. **Type:** predict (numeric).
- **Question:** "In the exact arm, half of the runs take 500 tokens and half take 5,000 tokens. The lossy method makes each short run 750 tokens and does not change the long runs. Compute the ratio of means without a cap, and with a cap of 2,000 tokens. Did the cap push the ratio toward 1?"
- **Answer key:** No cap: (750 + 5,000) / (500 + 5,000) = 5,750 / 5,500 = 1.045. Cap 2,000: (750 + 2,000) / (500 + 2,000) = 2,750 / 2,500 = 1.10. The cap pushed the ratio away from 1. The difference of means is the same in both (125 tokens); the cap only made the denominator smaller. So "the capped ratio is a lower bound" holds when the extra tokens sit in runs near the cap. It must be checked, for example with the survival curves.
- **Likely wrong answer:** "The cap always hides inflation, so the capped ratio is below 1.045."
- **Follow-up:** "What is the difference of the two arms' mean lengths without the cap and with it? And the exact arm's mean in each case?" (125 and 125; 2,750 and 1,250.)

### experiment_stats-Q17

- **Level:** 5. **Type:** apply to a new case (numeric).
- **Question:** "On the same 400 prompts and seeds you measure two ratios against the bf16 exact baseline. One: the exact method again with a different batch size gives λ₀ = 1.00, 95% interval [0.94, 1.06]. Two: 4-bit KV quantization gives λ = 1.12, interval [1.05, 1.19]. A reviewer says: 'The intervals overlap between 1.05 and 1.06, so the quantization effect is inside the noise.' Is the reviewer right? Give a number. Then name one better analysis. Last, take two explanations for longer reasoning. 'Chaos': any change makes the run diverge, and divergence alone makes reasoning longer. 'Information loss': the lossy method removes something the model needs. What does this result say about the two?"
- **Answer key:** Not right. Overlap of two intervals is not a test of the difference. Standard errors: 0.06 / 1.96 = 0.031 and 0.07 / 1.96 = 0.036. If independent, the difference has standard error sqrt(0.031² + 0.036²) = 0.047, so z = 0.12 / 0.047 = 2.55, two-sided p about 0.011. The two ratios share the baseline and the prompts, so they are not independent; 2.55 is a rough number. Better: both arms ran on the same prompts, so bootstrap the ratio Σ(quantized lengths) / Σ(rerun lengths) over prompts, with all three runs of a prompt kept together. Meaning: the zero-dose arm diverges but does not get longer (λ₀ near 1). That is expected: an exact rerun is a new sample from the same distribution. Quantization shifts the mean beyond that floor. So divergence alone (chaos) does not explain the increase. Something the quantizer does to the model explains it. This fits information loss, but it does not prove that mechanism. One arm is still missing before the claim is solid: a seed-resampled exact replicate, to confirm the size of the floor.
- **Likely wrong answer:** "Yes. Overlapping 95% intervals mean no significant difference."
- **Follow-up:** "Two independent estimates are 1.00 and 1.10, each with standard error 0.03. Do their 95% intervals overlap? What is z for the difference?" (Yes: [0.94, 1.06] and [1.04, 1.16]. z = 0.10 / (0.03 × √2) = 2.36.)

### experiment_stats-Q18

- **Level:** 4. **Type:** connect two ideas.
- **Question:** "You run an exact arm and a lossy arm in vLLM in bf16. Each request has the same seed in both arms, and requests are batched continuously. Idea one: a shared seed keeps the two arms identical until the lossy method changes a token. Idea two: bf16 results depend on batch composition. Put the two ideas together. What can you no longer conclude from the position of the first different token, and which extra arm or setting fixes it?"
- **Answer key:** With changing batch composition, even two exact runs with the same seed can diverge at some point. With greedy decoding, over 90% of MATH500 examples diverged across 12 runtime settings (batch size, GPU count, GPU type), and sampling is affected too [checked: 2506.09501 Sec 3.2, 3.3]. The paper did not test vLLM continuous batching itself, so measure the rate in your own setup. So the first different token between the arms may come from numerics, not from the lossy method. "The method changed a token at position k" is not shown. Fix one: a zero-dose arm, the exact method rerun with the same seeds under a different batch order, and compare its first-divergence positions and its λ₀ with the lossy arm. Fix two: remove the numerical noise, with a fixed batch composition, or FP32, or LayerCast (bf16 weights, FP32 compute; divergence under 3.4% [checked: 2506.09501 Sec 4]). Then zero dose gives almost identical runs (FP32 still diverged on 2.2% of examples), and almost every divergence is caused by the method. Keep the zero-dose arm to confirm it.
- **Likely wrong answer:** "The first different token is where the lossy method acted. The seed guarantees it."
- **Follow-up:** "Rerun the exact arm twice with the same seeds, once with 8 and once with 32 concurrent requests. Do you expect every pair of runs to be token-identical?" (No.)

---

## 5. Exercises

All three are CPU-only and need only numpy. I ran each one in the scratch folder on 2026-11-13 (Python 3.11.5, numpy 1.24.3). The code and the output below are copied by script from the files I ran. Run times were 23 s, 0.2 s and 1.8 s. A learner's own numbers will differ a little if the seed or the code layout differs. The examiner ran the three code blocks again from this file with the same versions. The output was the same, digit for digit (run times 23 s, 0.2 s, 2.0 s).

### Exercise 1. The noise-only control and the paired bootstrap

**Task (about 25 minutes).** Simulate an A/A test. Each prompt has a typical length drawn from a lognormal distribution (median 1,500, log standard deviation 0.8). Each run multiplies it by independent lognormal noise (log standard deviation 0.5). (a) For one test with 200 prompts print the ratio of means, the mean of ratios, the median of ratios, the geometric mean, and a 95% paired bootstrap interval for the ratio of means. (b) Repeat the test 1,000 times for n = 30, 200 and 800 prompts, once with no effect and once with a true effect of 1.10. For each row print the standard deviation of λ, how often the paired interval excludes 1, and the same for an unpaired bootstrap that resamples the two arms separately.

**Solution code.**

```python
# Exercise 1: the noise-only (A/A) control and the paired bootstrap. numpy only.
import numpy as np

def two_runs(n, rng, effect=1.0, sig_prompt=0.8, sig_noise=0.5):
    """Two runs on the same n prompts. Each prompt has its own typical length (base).
    Each run multiplies it by independent noise. effect = true ratio of mean lengths."""
    base = rng.lognormal(np.log(1500.0), sig_prompt, n)
    x = base * rng.lognormal(0.0, sig_noise, n)            # exact arm
    y = effect * base * rng.lognormal(0.0, sig_noise, n)   # second arm
    return x, y

def boot_paired(x, y, rng, B=1000):
    """Percentile bootstrap over prompts. The two arms of a prompt stay together."""
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return np.quantile(y[idx].sum(axis=1) / x[idx].sum(axis=1), [0.025, 0.975])

def boot_unpaired(x, y, rng, B=1000):
    """Wrong for this design: resamples the two arms separately."""
    n = len(x)
    ix = rng.integers(0, n, size=(B, n))
    iy = rng.integers(0, n, size=(B, n))
    return np.quantile(y[iy].sum(axis=1) / x[ix].sum(axis=1), [0.025, 0.975])

rng = np.random.default_rng(0)

# Part (a): one A/A experiment with 200 prompts
x, y = two_runs(200, rng)
r = y / x
lo, hi = boot_paired(x, y, rng, B=10000)
print("Part (a): one A/A experiment, n = 200 prompts, true effect = none")
print(f"  ratio of means   sum(y)/sum(x)      = {y.sum()/x.sum():.3f}")
print(f"  mean of ratios   mean(y/x)          = {r.mean():.3f}   (theory exp(0.5^2) = {np.exp(0.25):.3f})")
print(f"  median of ratios median(y/x)        = {np.median(r):.3f}")
print(f"  geometric mean   exp(mean(log y/x)) = {np.exp(np.log(r).mean()):.3f}")
print(f"  95% paired bootstrap interval for the ratio of means = [{lo:.3f}, {hi:.3f}]")

# Part (b): repeat the experiment 1000 times, with no effect and with a true effect of 1.10
print("Part (b): 1000 repeated experiments per row")
print("  effect    n   mean_lam  SD_lam  central95%_of_lam   paired:width  excl_1   unpaired:width  excl_1")
for effect in (1.0, 1.1):
    for n in (30, 200, 800):
        lam, wp, wu, ep, eu = [], [], [], 0, 0
        for _ in range(1000):
            x, y = two_runs(n, rng, effect)
            lam.append(y.sum() / x.sum())
            lo, hi = boot_paired(x, y, rng)
            wp.append(hi - lo); ep += (lo > 1.0) or (hi < 1.0)
            lo, hi = boot_unpaired(x, y, rng)
            wu.append(hi - lo); eu += (lo > 1.0) or (hi < 1.0)
        lam = np.array(lam)
        print(f"  {effect:6.2f}  {n:3d}   {lam.mean():8.3f}  {lam.std(ddof=1):6.3f}  "
              f"[{np.quantile(lam, 0.025):.3f}, {np.quantile(lam, 0.975):.3f}]     "
              f"{np.mean(wp):12.3f}  {ep/10:5.1f}%   {np.mean(wu):14.3f}  {eu/10:5.1f}%")
```

**Real output.**

```
Part (a): one A/A experiment, n = 200 prompts, true effect = none
  ratio of means   sum(y)/sum(x)      = 1.051
  mean of ratios   mean(y/x)          = 1.313   (theory exp(0.5^2) = 1.284)
  median of ratios median(y/x)        = 1.060
  geometric mean   exp(mean(log y/x)) = 1.048
  95% paired bootstrap interval for the ratio of means = [0.906, 1.218]
Part (b): 1000 repeated experiments per row
  effect    n   mean_lam  SD_lam  central95%_of_lam   paired:width  excl_1   unpaired:width  excl_1
    1.00   30      1.023   0.183  [0.702, 1.434]            0.639    8.9%            1.147    0.1%
    1.00  200      1.000   0.075  [0.862, 1.157]            0.275    5.8%            0.459    0.5%
    1.00  800      1.001   0.037  [0.931, 1.069]            0.140    5.6%            0.230    0.1%
    1.10   30      1.113   0.199  [0.769, 1.540]            0.691   13.2%            1.238    0.5%
    1.10  200      1.101   0.078  [0.963, 1.268]            0.296   29.6%            0.495    5.1%
    1.10  800      1.100   0.040  [1.023, 1.178]            0.155   75.8%            0.254   30.3%
```

**Expected result.**

- Part (a): the mean of ratios is near exp(0.25) = 1.28 although nothing changed. This single test shows a ratio of means of 1.05 by chance; its interval [0.91, 1.22] contains 1.
- Part (b), no effect: the standard deviation of λ₀ is 0.18 at n = 30, 0.075 at n = 200 and 0.037 at n = 800. It shrinks like 1/√n. At n = 30 the paired interval excludes 1 in 8.9% of trials, more than the nominal 5%, and the mean of λ₀ is 1.02, a small bias of the ratio estimator.
- Part (b), effect 1.10: the paired interval detects the effect in only 13% of trials at n = 30, 30% at n = 200 and 76% at n = 800.
- The unpaired bootstrap gives intervals 1.6 to 1.8 times wider. It almost never raises a false alarm (0.1% to 0.5%), and it loses most of the power (30% against 76% at n = 800).

**How the tutor checks it.** Ask the learner to show the line where the indices are drawn, and confirm that x and y use the same index array in the paired version. Ask: "A paper reports λ = 1.08 on 200 prompts with no noise-only arm. Using your table, how surprised are you?" (Not at all: the standard deviation of λ₀ is 0.075.) Ask: "Why is the mean of ratios 1.28?" (Core idea 15.) Ask: "How many prompts would this toy need to detect 1.10 in most trials?" (About 800.) A level-4 learner also explains why the unpaired interval is wider (the prompt term, Core idea 2).

### Exercise 2. What a length cap does to the ratio of means

**Task (about 20 minutes).** Draw 200,000 exact lengths from a lognormal distribution (median 1,000, log standard deviation 0.7). Build three lossy arms: (i) every run 1.3 times longer, with extra lognormal noise (log standard deviation 0.3); (ii) 10% of runs 5 times longer, same noise; (iii) only runs under 1,000 tokens become 1.5 times longer. For caps of infinity, 8,192, 4,096, 2,048 and 1,500 print: the cap-hit rate of each arm, the capped ratio of means, the ceiling C / (mean capped exact length), the ratio on cases that finished in both arms, and the ratio of medians.

**Solution code.**

```python
# Exercise 2: what a length cap does to the ratio of means. numpy only.
import numpy as np
rng = np.random.default_rng(1)
n = 200_000                                   # large, so sampling noise is small
x = rng.lognormal(np.log(1000.0), 0.7, n)     # exact arm, true lengths (no cap)
noise = rng.lognormal(0.0, 0.3, n)

scenarios = {
    # every run is 1.3x longer (pure scale effect)
    "scale: every run x1.3": 1.3 * x * noise,
    # 90% of runs unchanged, 10% become 5x longer (regime switch, heavy tail)
    "tail: 10% of runs x5": np.where(rng.random(n) < 0.10, 5.0, 1.0) * x * noise,
    # only short runs get longer; long runs are unchanged
    "body: runs under 1000 x1.5": np.where(x < 1000.0, 1.5, 1.0) * x,
}

for name, y in scenarios.items():
    print(f"\n{name}   true ratio of means (no cap) = {y.mean()/x.mean():.3f}")
    print("    cap   hit_exact  hit_lossy  lam_capped  ceiling  lam_both_finished  median_y/median_x")
    for C in (np.inf, 8192, 4096, 2048, 1500):
        xc, yc = np.minimum(x, C), np.minimum(y, C)
        both = (x < C) & (y < C)
        ceiling = C / xc.mean()
        print(f"  {C:6.0f}   {np.mean(x >= C):8.3f}  {np.mean(y >= C):9.3f}  "
              f"{yc.mean()/xc.mean():10.3f}  {ceiling:7.2f}  "
              f"{y[both].mean()/x[both].mean():17.3f}  {np.median(yc)/np.median(xc):17.3f}")
```

**Real output.**

```
scale: every run x1.3   true ratio of means (no cap) = 1.358
    cap   hit_exact  hit_lossy  lam_capped  ceiling  lam_both_finished  median_y/median_x
     inf      0.000      0.000       1.358      inf              1.358              1.296
    8192      0.001      0.007       1.345     6.44              1.344              1.296
    4096      0.022      0.065       1.292     3.29              1.307              1.296
    2048      0.152      0.273       1.183     1.84              1.242              1.296
    1500      0.279      0.423       1.131     1.50              1.210              1.296

tail: 10% of runs x5   true ratio of means (no cap) = 1.464
    cap   hit_exact  hit_lossy  lam_capped  ceiling  lam_both_finished  median_y/median_x
     inf      0.000      0.000       1.464      inf              1.464              1.105
    8192      0.001      0.028       1.342     6.44              1.240              1.105
    4096      0.022      0.088       1.207     3.29              1.102              1.105
    2048      0.152      0.243       1.081     1.84              1.014              1.105
    1500      0.279      0.360       1.042     1.50              0.989              1.105

body: runs under 1000 x1.5   true ratio of means (no cap) = 1.122
    cap   hit_exact  hit_lossy  lam_capped  ceiling  lam_both_finished  median_y/median_x
     inf      0.000      0.000       1.122      inf              1.122              1.226
    8192      0.001      0.001       1.122     6.44              1.123              1.226
    4096      0.022      0.022       1.125     3.29              1.134              1.226
    2048      0.152      0.152       1.139     1.84              1.193              1.226
    1500      0.279      0.279       1.155     1.50              1.267              1.226
```

**Expected result.**

- The noise has mean exp(0.3² / 2) = 1.046, so the uncapped ratio in scenario (i) is 1.3 × 1.046 = 1.36, not 1.30.
- Scenarios (i) and (ii): the capped ratio falls toward 1 as the cap gets tighter (1.36 to 1.13, and 1.46 to 1.04). The tail scenario loses more, because all its extra tokens are in long runs.
- Scenario (iii): the capped ratio rises (1.12 to 1.16). The cap is not always a lower bound.
- The both-finished ratio is no fix. In scenario (ii) it is below the capped ratio at every finite cap and reaches 0.989 at cap 1,500. In scenario (iii) it overshoots (1.27).
- The ratio of medians does not move at all, because under 50% of runs hit the cap in each arm. It is a different quantity from the ratio of means (1.11 against 1.46 in the tail scenario).
- The capped ratio always stays under the ceiling.

**How the tutor checks it.** Ask the learner to point at one row where the capped ratio is above the true ratio and to say why (Core idea 11). Ask: "In the tail scenario at cap 2,048, the lossy arm hits the cap in 24% of runs and the exact arm in 15%. Which reported number shows the effect most clearly?" (The cap-hit rates themselves: 9 points apart, while the capped ratio shows only 1.08.) Ask: "At which cap-hit rate would the median stop being valid?" (50%.) A level-5 learner adds a survival-curve comparison without being asked.

### Exercise 3. Switch back to the exact method at token t (ties to Part B)

**Task (about 30 minutes).** Build two toy models that have the same inflation, 1.40, when the lossy method is always on. Model M: after each progress token a bad token appears with chance 0.05 and wastes 8 tokens; then the run goes on. Model H: each bad token breaks the run for good with a small chance h; a broken run is 5 times longer. Choose h so that H also gives 1.40. Then apply the lossy method only up to token t (t = 50, 200, 800, infinity) and print for each model: λ, the mean extra tokens, the mean extra divided by t, the share of runs whose extra tokens exceed t, the shift of the 99th percentile of length, and the share of the net extra tokens held by the top tenth of runs.

**Solution code.**

```python
# Exercise 3: switch back to the exact method at token t. numpy only.
# Two toy models with the SAME mean inflation when the lossy method is always on.
import numpy as np
rng = np.random.default_rng(2)
n = 4000
r = 0.05      # chance that a progress token is followed by a bad token (lossy method on)
c = 8         # model M: tokens wasted by one bad token (a local detour)
K = 5         # model H: a broken run becomes K times longer
W = np.minimum(rng.lognormal(np.log(800.0), 0.6, n), 6000).astype(int)  # tokens the exact method needs
rows = np.arange(n)

# Model M (per-token cost): a bad token wastes c tokens, then the run goes on as before.
bad = rng.random((n, W.max())) < r
cum = np.cumsum(1 + c * bad, axis=1)          # tokens used after j progress steps while lossy
def length_M(t):
    if np.isinf(t):
        return cum[rows, W - 1]
    done = np.minimum((cum <= t).sum(axis=1), W)              # progress steps finished by token t
    used = np.where(done > 0, cum[rows, np.maximum(done - 1, 0)], 0)
    return used + (W - done)                                   # the rest runs at the exact speed

# Model H (lasting damage): a bad token breaks the run with chance h, for good.
u = rng.random(n)
def length_H(t, h):
    exposed = np.minimum(W, t)                                 # tokens generated while lossy was on
    broken = u < 1.0 - (1.0 - r * h) ** exposed
    return np.where(broken, K * W, W)

lamM = length_M(np.inf).mean() / W.mean()
hs = np.linspace(0.0, 0.02, 4001)                              # pick h so both models match
h = hs[np.argmin([abs(length_H(np.inf, k).mean() / W.mean() - lamM) for k in hs])]

def top_decile_share(extra):
    k = len(extra) // 10
    return np.sort(extra)[-k:].sum() / extra.sum()

print(f"exact arm: mean length {W.mean():.0f}, 99th percentile {np.quantile(W, 0.99):.0f}; model H uses h = {h:.4f}")
print("switch_t  model  lambda  mean_extra  mean_extra/t  runs_with_extra>t  q99_shift  top10%_share")
for t in (50, 200, 800, np.inf):
    for name, L in (("M", length_M(t)), ("H", length_H(t, h))):
        extra = L - W
        a = "      -     " if np.isinf(t) else f"{extra.mean()/t:12.2f}"
        b = "        -        " if np.isinf(t) else f"{np.mean(extra > t):17.3f}"
        print(f"{t:8.0f}  {name:>5}  {L.mean()/W.mean():6.3f}  {extra.mean():10.1f}  {a}  {b}  "
              f"{np.quantile(L, 0.99) - np.quantile(W, 0.99):9.0f}  {top_decile_share(extra):12.2f}")
```

**Real output.**

```
exact arm: mean length 951, 99th percentile 3235; model H uses h = 0.0015
switch_t  model  lambda  mean_extra  mean_extra/t  runs_with_extra>t  q99_shift  top10%_share
      50      M   1.013        12.3          0.25              0.000          3          0.20
      50      H   1.016        15.6          0.31              0.004        129          1.00
     200      M   1.058        55.0          0.28              0.000         43          0.14
     200      H   1.060        57.4          0.29              0.016       1115          1.00
     800      M   1.218       207.4          0.26              0.000        221          0.13
     800      H   1.224       213.1          0.27              0.050       3945          1.00
     inf      M   1.399       379.8        -               -               1291          0.25
     inf      H   1.399       379.0        -               -               6670          1.00
```

**Expected result.**

- The two models have almost the same λ at every switch point (1.058 against 1.060 at t = 200). In this toy the mean cannot separate them.
- Mean extra divided by t is near 0.4 / 1.4 = 0.29 in both models (0.25 to 0.31). This matches the per-token prediction t · (1 − 1/λ). For model M this must hold. For model H it holds because the damage grows in proportion to exposure and because of the toy's length spread (log standard deviation 0.6). With all runs equally long, H gives about 45% more than the prediction. With a log standard deviation of 1.0 it gives about 25% less (scratch script `hslope.py`). So the lesson is "an equal mean does not rule out lasting damage". It is not "the means are always equal".
- The share of runs with extra tokens above t is exactly 0 in model M and 0.4% to 5% in model H.
- The 99th percentile moves by 43 tokens in M and by 1,115 tokens in H at t = 200.
- The top tenth of runs holds 13% to 25% of the net extra tokens in M and all of them in H.

**How the tutor checks it.** Ask: "You may report one number from a switch-back run at t = 200. Which one, and what value would make you say 'lasting damage'?" (The share of runs with more than t extra tokens; any value clearly above the noise-only arm's value.) Ask: "Why can model H give the same mean as model M?" (Constant hazard: expected damage grows in proportion to exposure, like a per-token cost. Accept this. The exact match also needs the toy's length spread.) Ask: "Model H breaks 1.6% of runs at t = 200. How many runs do you need to see at least 10 broken runs?" (About 10 / 0.016 = 625.) A level-5 learner notes that with a cap the broken runs of H would be censored, so the right outcome becomes the cap-hit rate after the switch.

---

## 6. Diagnostic picks

- **Level 2-3: experiment_stats-Q3** (greedy runs with no repeat, six benchmarks, +6.0% and +8.2%). The answer shows at once whether the learner holds "greedy is deterministic" and whether he asks for a noise-only arm without being told. A level-3 answer also names paired intervals over questions.
- **Level 4-5: experiment_stats-Q12** (flaws in a switch-back design on AIME). A level-4 answer finds the cap and the missing noise control. A level-5 answer also finds the survivor selection at the switch point, says that "equal means" is the wrong null, and proposes the branch design. It maps directly onto Part B.

---

## 7. Sources

Papers, read in this run from copies of the arXiv HTML pages saved in the scratch folder (no new fetch today; the title in each copy matches the id). The examiner read every cited passage again in these copies:

| id | title (as shown on the page) | what I read |
|---|---|---|
| arXiv 2506.09501 (v2, 24 Oct 2025) | Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference (Yuan, Li, Ding, Xie, Li, Zhao, Wan, Shi, Hu, Liu) | abstract; Sections 1 to 4 in full, with Tables 3, 4 and 5. Figures were not viewed; numbers tied to Fig. 3 and Fig. 5 come from the running text. Appendices not read. |
| arXiv 2610.23818 | Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding | abstract; Sections 1 to 5 in full; Limitations; Appendix C (seeds, draft length, temperature, Table C1) and Appendix F in full. Appendices B and D: captions only. Table B1 was not read line by line. |
| arXiv 2606.25519 (v2, 29 Jun 2026) | Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models | Section 3 (setup), Table 2, the definition of CTIR, and a text search for repeated runs, seeds and intervals (none found). Not read in full. |
| arXiv 2506.04108 (v2, 5 Jun 2025) | Rectified Sparse Attention | abstract and Section 2.2 only. |

Read directly on disk in this run:

| source | what I read |
|---|---|
| `research_notes/Inference speed question search/verification.md` | sections F07, F38, F01, F17 and the critic's note on the missing zero-dose control. Not the whole file. |
| `reports/length-tax.html` | full text |
| `kvfetch/sparse_decode.py` | function `sample_token` and the `Job` dataclass only |
| `results/reasoning/e3_shards/` | all 240 sample files of each of the runs `full`, `oracle_512`, `oracle_1024`, `quest_512`, `ours-nohist_512` (fields `n_gen`, `stopped`, `score`); recomputed cap-hit rates, paired length ratios and bootstrap intervals over problems |
| `results/e3_explore.json` | `summary` block (matches the recomputed means and cap-hit rates) |
| `research_notes/Length tax preliminaries/specdec_exact.md` | format only |

Statistics reference: none was opened. The textbook facts used here (variance of a paired difference, the lognormal mean exp(σ²), capped mean = area under the survival curve, quantiles under a fixed cap, percentile and cluster bootstrap, standard error of a difference) were derived in the text and checked by the simulations in Section 5 and by a short arithmetic script. The examiner derived and recomputed each of them again (`arith.py` in the examiner's scratch folder). Standard books for these topics are Efron and Tibshirani, "An Introduction to the Bootstrap", and Klein and Moeschberger, "Survival Analysis" [unverified: named from memory, not opened].

Scratch files (code and raw outputs): `/tmp/claude-1025/-home-ryankert-l-kv-fetch-kv-fetch/b232f66b-c1cf-477e-b08d-10c5771f3def/scratchpad/expstats/run2/`. Examiner's scratch files: `/tmp/claude-1025/-home-ryankert-l-kv-fetch-kv-fetch/b232f66b-c1cf-477e-b08d-10c5771f3def/scratchpad/exam_stats/` (`arith.py`, `e3.py`, `hslope.py`, `ex1.py` to `ex3.py`, and a copy of this file before the changes).

---

## 8. Verification log

Examined 2026-11-13 15:40 CST by a second pass that acted as a skeptical examiner. No GPU job. No network fetch. Nothing was committed or posted.

**What was checked (49 items: 20 core ideas, 8 misconceptions, 18 questions, 3 exercises).**

- Every question was solved before its key was read. Every number in a key was recomputed by a separate script.
- The three exercise codes were copied out of this file and run again (Python 3.11.5, numpy 1.24.3, CPU). The output matches the pasted output digit for digit. Each "Expected result" line was checked against that output.
- Every [checked, local data] number was recomputed from `results/reasoning/e3_shards/` with a new script. All agree. Bootstrap ends differ by at most 0.002, which is resampling noise.
- Every paper fact was read again in the saved page copy of its paper. The title of each of the four ids was confirmed in the copy.
- Every [from notes] claim was found in `verification.md` (F07, F38, F01, F17) or in `length-tax.html`.

**Wrong keys that were corrected.**

1. Q3 follow-up, Q18 key, Core idea 4, Misconception 1. The "over 90% diverged" figure of 2506.09501 counts an example when at least two of 12 runtime settings differ. The old text used it as the rate for one change (batch 16 against 32). The paper gives no rate for one pair. The follow-up of Q3 now asks about all 12 settings.
2. Core idea 17. The "two capped runs held 61% of the output tokens" is on the eight tuning cases, not on all 30 cases.
3. Q14 key. The old example band ("both shares between 30% and 70% means the channels interact") was wrong: two shares of 50% add to 100%, which is the additive case. Interaction shows in the sum of the two repair shares (100% plus the interaction share). The new band says so.
4. Q14 key, point 4. "Only half of the problems can show anything" was wrong. The 10 problems that never hit the cap in the exact arm do hit it in lossy arms (16% of runs under oracle@512). Only the 5 always-capped problems cannot move.
5. Q11 key, Misconception 8, Core idea 20, Exercise 3. "The mean cannot separate M and H because H has a constant hazard" was too strong. The equal means in the toy depend on its length spread. With other spreads H's mean is 45% above or 25% below the per-token prediction (`hslope.py`). The correct lesson is kept: an equal mean does not rule out lasting damage.
6. Q9 key. "Check whether any monotone rescaling puts the arms on one curve" cannot work: the two arms have the same dose value. It now says to try another dose measure.
7. Q5 key. "Quantiles below the cap-hit rate" is now "quantiles q with q < 1 − cap-hit rate". "The 5× runs are the ones removed" is now the measured share: 60% of them, holding 78% of the extra tokens.
8. Misconception 6. "The interval misses seed noise" was dropped. With one run per case, the case bootstrap already contains run noise. It misses what all cases of a run share.

**Ambiguous questions that were rewritten.**

9. Q2: now says one run per prompt, independent run noise, and which estimator is meant.
10. Q10: the old wording made every run have exactly 57 extra tokens, so "largest extra of a single run" had two answers (57 or 200). It now asks for a hard upper bound and says the waste differs between runs.
11. Q15: the target half-width (0.05) is now in the question.
12. Q17: "chaos" and "information loss" are now defined in the question. The key no longer says the result proves information loss.
13. Q3: the question now matches the paper (six benchmarks, average growth, no repeated run reported). The ranking claim is given to an invented reader, not to the paper.
14. Q6: the key now accepts "about 1.3, the same size" (the learner cannot know σ). The simulated value is 1.25.
15. Q1 and Q2: level changed from 2 to "2 to 3", because the ladder puts these computations at level 3.

**Smaller fixes.**

16. Core idea 14 and Q4: the 59% and the 2.2 are from the 60 primary-pair cells, not from all 360 cells.
17. Core idea 19 and Q8: the low-probability figures hold for four of the five rules; spec_casc_tok is the exception. "Mostly rank 2 to 5 tokens" had no source and was removed. "Arm R in the notes" does not exist in the notes and was replaced by the F07 wording.
18. Core idea 1: S is the acceptance-based speedup of Eq. 1.
19. Core idea 3: "independent" softened; a measured check of leftover seed coupling was added.
20. Core idea 5: the reading of the abstract's "9,000 tokens" is marked as a reading.
21. Core idea 8: caveat added (same cases, other server).
22. Core idea 13: "removes exactly" is now "removes mostly".
23. Core idea 20: derivation of t · (1 − 1/λ) added; the pure-interaction case of the 2×2 added.
24. Q7: the measured problem-level standard error (0.069) added; "much closer" is now "closer".
25. Q12: two wordings made exact ("pushes the means together", "about" instead of "up to").
26. Q14: the ±35 points for the sum is marked as valid only for independent shares.
27. Q17: the z of 2.55 is marked as rough, because the two ratios share the baseline.
28. Q18: zero dose in FP32 gives "almost identical" runs, not identical ones.
29. Exercise 2: "at every cap" is now "at every finite cap".
30. Tags: 2606.25519 and 2506.04108 moved from [from notes] to [checked]. The Sources section lists what was read.

**Deleted.** No item had to be deleted.

**What stays unverified.**

- 2606.25519: that each model was run exactly once. The paper reports no repeated run and no interval, but it does not say "once". The paper was not read in full.
- 2506.09501: the divergence rate for one pair of settings, and for vLLM continuous batching. The paper reports neither. Figures and appendices were not viewed; the 90% and 2.2% come from the running text of Sec 3.2.
- 2610.23818: Tables B1 and B2 were not read line by line. The figures used here come from the running text of Sec 3.7 and App. C and F.
- The planning numbers from the notes (about 400 runs per arm; about 960 samples per arm; the band 1.05 to 1.15) were found in the notes but not recomputed. The simple 1/√n rule in Q15 gives 352.
- The two book names in the Sources section [unverified].
- The 0.3 in the follow-up of Q8 is an example value, not a measurement.
- All numbers from Exercises 1 to 3 are toy numbers. They show a direction. They are not estimates for a real model.
