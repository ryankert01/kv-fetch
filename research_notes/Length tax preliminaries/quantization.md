# Tutor pack: weight and KV quantization as a perturbation

Subskill id: `quantization`. Written 2026-11-13 15:09 CST. Read-only work; no GPU job was run.

Scope: round-to-nearest versus error-compensating methods (GPTQ, AWQ, HQQ): what each one minimises and what error is left; why the same bit width gives different shifts of the output distribution; how quantization changes sampling at uncertain positions (hesitation tokens) and reasoning length.

Notation in this file. **w** is a weight, **ŵ** its quantized value, **δ = ŵ − w** the weight error (a row vector for one output neuron). **x** is the input vector of a linear layer and **A = E[x xᵀ]** its second-moment matrix. GPTQ writes H = 2XXᵀ for the same thing up to a constant factor. **b** is the bit width and **s** the grid step. **p** is the full-precision (BF16) next-token distribution and **p'** the quantized model's. **ε** is the logit error (a vector over the vocabulary). Note: papers use "s" both for the grid step (KIVI, HQQ) and for the AWQ channel scale. This file writes the AWQ scale as **c**.

How to read the tags:

- **[checked]**: read in the source itself. The first pass used a fetch tool that returns extracted answers. The verification pass (see "Verification log" at the end) downloaded each page and read the raw text for the [checked] facts. All formulas were also re-derived here.
- **[from notes]**: taken from `research_notes/Inference speed question search/` or `reports/length-tax.html`; not re-read in this run.
- **[unverified]**: from memory only, or the source did not state it.
- **[mine]**: my own derivation, reading or toy model. Not a claim of any paper.
- No tag: textbook fact or a derivation written out here.

---

## 1. Core ideas

1. **The uniform grid.** With b bits, an asymmetric min-max quantizer uses zero-point z = min(X), step s = (max(X) − min(X)) / (2^b − 1), code Q(X) = round((X − z)/s) and reconstruction X' = Q(X)·s + z [checked: KIVI 2402.02750 Sec 3.1]. AWQ writes the symmetric form Q(w) = Δ·Round(w/Δ) with Δ = max(|w|) / 2^(N−1), N = number of bits [checked: 2306.00978 Eq. 1]. The grid is set per group of weights (for example 128 weights, "g128"), so one outlier in a group makes the step larger for the whole group.

2. **Round-to-nearest (RTN).** RTN picks, for each weight alone, the nearest grid point. It minimises Σ (ŵ − w)² for the given grid and uses no data. The error lies in [−s/2, +s/2]. If it is spread evenly over that interval, its mean is 0 and its variance is s²/12. One bit less doubles s (exactly: the factor is (2^b − 1)/(2^(b−1) − 1), which is 15/7 = 2.14 from 4 to 3 bits), so the mean squared error grows about 4 times.

3. **The error is frozen, not random.** After quantization, δ is a fixed matrix. The quantized model is a different deterministic function. The same prompt gives the same logits every time, and averaging over samples or seeds does not remove the error. "Noise" only means that δ looks random across weights and that the logit error looks random across positions.

4. **Stochastic rounding (SR).** Let f be the fractional part of (w − z)/s. SR rounds up with probability f and down with probability 1 − f. Then E[ŵ] = w for every w (no bias), and the error variance is s²·f(1 − f). Averaged over f uniform in [0, 1] this is s²·(1/2 − 1/3) = s²/6, twice the RTN value s²/12. SR trades a larger error for an error that is zero-mean given the value.

5. **What matters is the output error of the layer.** For one output neuron with weights w and error δ, the output error on input x is δᵀx. Its mean square over inputs is E[(δᵀx)²] = δᵀ A δ with A = E[x xᵀ]. RTN minimises δᵀδ. That is the same objective only when A is a multiple of the identity. In LLM layers A has a few very large directions, so the direction of δ matters more than its length.

6. **Compensation formula (Optimal Brain Surgeon).** Fix the error on coordinate q to e = quant(w_q) − w_q and let the other weights move freely. Minimising δᵀAδ subject to δ_q = e gives δ = e · A⁻¹e_q / [A⁻¹]_qq (e_q is the q-th unit vector) and a minimum output error of e² / [A⁻¹]_qq. Without compensation the error is e²·A_qq. Since A_qq·[A⁻¹]_qq ≥ 1 for a positive definite A, one compensation step never increases the error on the data that defined A (rounding the other weights later adds new error). GPTQ states the same update as δ_F = −(w_q − quant(w_q)) / [H_F⁻¹]_qq · (H_F⁻¹)_{:,q}, with F the set of weights not yet quantized [checked: 2210.17323 Eq. 2].

7. **GPTQ.** Objective: argmin over Ŵ of ‖WX − ŴX‖₂², one linear layer at a time, where X holds the layer inputs for a small calibration set [checked: 2210.17323 Eq. 1]. It rounds the columns in one fixed order and, after each column, applies the update of idea 6 to the columns not yet rounded. It uses H = 2XXᵀ plus a damping term of 1% of the mean diagonal, a Cholesky form of the inverse, blocks of 128 columns, and 128 calibration segments of 2,048 tokens from C4. The grid is the standard per-row asymmetric min-max grid, fixed before rounding. The inputs X of each block come from the already quantized earlier blocks, not from the full-precision model [checked: 2210.17323 Sec 4 and Algorithm 1 for the method; Sec 5 "Setup" for calibration data, grid and block inputs].

8. **What GPTQ leaves.** (a) The grid is discrete, so a residual stays; the last columns have no free weights to absorb their error. (b) The search is greedy in a fixed order, so the result is not the best grid point (Question Q4 shows a two-weight case). (c) The weights move farther from the originals than under RTN. The error is pushed into directions that the calibration inputs rarely excite. On inputs unlike the calibration set it can be larger than RTN's (Exercise 1). So the calibration text matters: for Qwen3-4B with INT3 GPTQ, changing the calibration text from Wikitext to the model's own math reasoning traces raises accuracy from 28% to 65% and cuts the CoT from 30k to 12k tokens [checked: 2606.25519 Sec 7.2, Fig. 5]. (d) The objective is layer output error, not the next-token distribution and not generated length.

9. **AWQ.** AWQ scales each input channel of W by a factor c > 0 and divides the input by c, which leaves the full-precision layer unchanged. Objective: c* = argmin over c of ‖Q(W·diag(c))·(diag(c)⁻¹·X) − W·X‖. The search space is one number per layer: c = s_X^α, where s_X is the mean activation magnitude per input channel and α is taken from a grid of 20 values in [0, 1] [checked: 2306.00978 Eq. 4-5]. Error analysis: Err(Q(w)x) = Δ·RoundErr(w/Δ)·x, and after scaling Err(Q(w·c)(x/c)) = Δ'·RoundErr(wc/Δ')·x·(1/c), so the error ratio is (Δ'/Δ)·(1/c) [checked: 2306.00978 Eq. 2-3]. AWQ "does not rely on any backpropagation or reconstruction" [checked: 2306.00978, abstract]. After the scale search it also clips the weight range to lower the MSE [checked: 2306.00978, text after Eq. 5].

10. **What AWQ leaves.** The weights are still rounded to nearest, on a rescaled grid. No weight is moved to cancel another weight's error. Protecting channels with large inputs makes Δ' larger when a scaled channel sets the group maximum, and then every other weight in the group gets a larger error. The search is over one α per layer. My reading [mine]: AWQ shapes δ using only the diagonal of A (the size of each input channel); GPTQ also uses the off-diagonal terms (correlated inputs).

11. **HQQ.** HQQ uses no calibration data. It minimises a robust norm of the weight error: argmin over z, s of φ(W − Q⁻¹(Q(W))), with Q_{z,s}(W) = round(W/s + z) = W_q, Q⁻¹_{z,s}(W_q) = s·(W_q − z), and φ an l_p norm with p < 1 (p = 0.7 in the post). In practice the step s is fixed and only the zero-point z is optimised. The solver adds a variable W_e and alternates two steps: W_e ← shrink_lp(W − Q⁻¹(Q(W)), β) with shrink_lp(x, β) = sign(x)·relu(|x| − |x|^(p−1)/β), then z ← the mean of W_q − (W − W_e)/s over the quantization group; β grows by a factor κ each iteration (β = 1, κ = 1.01, 20 iterations) [checked: HQQ blog post, Badri and Shaji, 2023]. What is left: HQQ knows nothing about the inputs, so it cannot shape δ by A. The p < 1 norm accepts a large error on a few outlier weights to fit the bulk better.

12. **Same bit width, different result.** The bit width fixes only the size of the grid step. The chain from there to length has five links: weight error → layer output error (δᵀAδ) → logit error ε → change of the distribution → changed tokens → length. Methods differ at the first link, and the later links are not linear. Measured average CoT inflation (CTIR, weight-only, greedy) on Qwen3-4B at INT4: RTN +42.5%, AWQ +8.2%, GPTQ +12.0%, HQQ +6.0%, ParoQuant +4.7%. At INT3: AWQ +276.1%, GPTQ +48.7%, HQQ +292.1%, ParoQuant +24.5%. On Qwen3-30B at INT4: RTN +43.0%, AWQ +14.4%, GPTQ +3.7%, HQQ +3.9%; at INT3: AWQ +32.8%, GPTQ +19.0%, HQQ +60.1% [checked: 2606.25519 Table 2]. The order of AWQ, GPTQ and HQQ changes with model and bit width. Only "RTN is worst at INT4" holds on both models. Three cautions when reading these numbers. (i) Each average is the mean of six per-benchmark CTIRs, and one benchmark can dominate it: for AWQ INT3 on Qwen3-4B, MBPP alone is +957.4% and AIME25 is +45.1% [checked: Table 2]. (ii) Calibration differs by method. The calibrated methods use the BF16 model's own math reasoning traces ("we adopt Math-CoT as the default calibration corpus for all main-paper PTQ experiments"), while RTN and HQQ use no data [checked: 2606.25519 Sec 3, Sec 7.2]. (iii) All methods, RTN too, use group size 128 [checked: Sec 3], so RTN's grid is not coarser than the others'.

13. **From logit error to distribution change.** Let p' = softmax(z + ε). Then KL(p ‖ p') = log E_p[e^ε] − E_p[ε] ≈ ½·Var_p(ε) = ½·(Σ_v p(v)ε(v)² − (Σ_v p(v)ε(v))²) for small ε. Two results follow. A constant added to all logits changes nothing. And the same ε gives a KL near zero where p is almost one-hot and a large KL where p is spread out. So a correlation between per-position KL and entropy is expected from the softmax alone; it does not show that the quantization error is larger at uncertain positions (Exercise 2).

14. **Margins and greedy flips.** With two candidate tokens and logit margin m = z_(1) − z_(2), the top probability is p = 1/(1 + e^(−m)) and dp/dm = p(1 − p). A margin error δm moves p by about p(1 − p)·δm: 0.25·δm at p = 0.5 and 0.0099·δm at p = 0.99. Under greedy decoding the token changes only if δm < −m. If δm across positions looks like N(0, σ²), the flip rate at margin m is Φ(−m/σ), with Φ the standard normal CDF. The flip count is very non-linear in σ.

15. **Zero-mean logit noise flattens peaked positions on average.** For a tail token u and the top token 1, the odds p'(u)/p'(1) = (p(u)/p(1))·e^(ε_u − ε_1). If the two errors are independent N(0, σ²), then E[e^(ε_u − ε_1)] = e^(σ²) > 1. The median odds do not change, but the mean does, so over many positions unlikely tokens are sampled more often. This looks like a small rise in temperature at peaked positions (Exercise 2e). For a 50/50 position the effect is symmetric and the mean does not move.

16. **Hesitation tokens, what was measured.** Setup: R1-Distill models and QwQ-32B, GPTQ and AWQ at 3 and 4 bits (g128), FlatQuant W4A4KV4 and W8A8KV8, sampling at T = 0.6, top-p = 0.95 [checked: 2606.00206 Sec 3, App. A]. Per-position KL: D_KL(p_t ‖ q_t) = Σ_v p_t(v)·log(p_t(v)/q_t(v)), with p_t from BF16 and q_t from 3-bit AWQ, both run on the same prefixes on MATH-500. It correlates with the BF16 entropy at Spearman ρ = 0.92 "across all positions" [checked: 2606.00206 Sec 4.3, Sec 6, Fig. 4c]. Each KL value is then attached to the token that the quantized model sampled at that position, and averaged per token (only tokens seen at least 50 times) [checked: Sec 4.3]. The tokens with the highest average KL include "Wait", "But", "Alternatively", "if", "maybe", "verify", "think"; the lowest are math and formatting tokens [checked: Fig. 4a-b]. At high-entropy positions (entropy above the model's 80th percentile) it is about 2 to 4 times more likely that at least one of the 50 marker tokens is among the 20 most probable tokens. This holds for BF16 and for 3-bit models [checked: App. B.2, Fig. 7]. Direction: the paper reports it only for free-running traces. The number of high-KL tokens divided by the number of low-KL tokens is 0.57 in BF16 traces and 1.15 in 3-bit AWQ traces [checked: Sec 4.3], and the abstract says quantized models "disproportionately sample overthinking markers". The paper does not report, at matched prefixes, whether q_t(marker) is larger than p_t(marker). In a free-running trace a marker can also be more frequent because the trace has already gone wrong. The two readings are not separated.

17. **Hesitation tokens, the intervention.** Logit penalty: z'_t(v) = z_t(v) − λ if v ∈ S, else z_t(v), with S a list of 50 marker tokens and λ > 0 [checked: 2606.00206 Eq. 1]. The token-list tests below were run on quantized R1-Distill-Qwen-1.5B and averaged over six quantization configurations and five benchmarks; no BF16 run is among them. Penalising markers cuts CoT length by 12 to 23% with accuracy kept or slightly better; penalising random tokens does almost nothing; penalising low-KL tokens raises length by up to 41% [checked: 2606.00206 Sec 5, Fig. 5]. Boosting markers (λ = −3.0) raises CoT length by 395% and lowers accuracy by 31%; boosting the high-KL tokens gives +294% and −18%; boosted random tokens stay near neutral; boosted low-KL tokens reach only +10% at λ = −4.0 [checked: 2606.00206 App. B.4, Fig. 9]. So the identity of the changed token matters for length, not only how many tokens change. The paper calls its margin argument "intuition for one specific failure mode, not a complete causal explanation" [checked: Sec 6].

18. **Length, what was measured.** CTIR(M_q; D) = (ℓ̄(M_q; D) − ℓ̄(M_org; D)) / ℓ̄(M_org; D) × 100%, where ℓ̄(M; D) is the mean number of CoT tokens of model M over dataset D, M_org is the BF16 model and M_q the quantized one [checked: 2606.25519 Eq. 1-3]. That paper uses temperature 0 and one run per configuration, so its flips are argmax flips at small margins (idea 14), not sampling [checked: Sec 3]. Length can grow with accuracy flat: Qwen3-30B GPTQ INT4 loses 0.2 points on the reasoning tasks at CTIR +3.7% [checked: Tables 1-2]. In the other paper, 3-bit AWQ takes R1-Distill-Qwen-1.5B on MATH-500 from 5.2K to 23.4K CoT tokens (4.5 times) and from 85.6% to 47.0% accuracy; 3-bit GPTQ gives 6.8K tokens and 71.6%; 139 of 265 errors (52%) under 3-bit AWQ reach the right answer in the trace and do not output it, against 19 of 72 (26%) for BF16 [checked: 2606.00206 Sec 4.1-4.2]. Not every study finds inflation: one abstract says "quantized models do not exhibit increased output lengths" [from notes: 2504.04823, abstract only].

19. **KV quantization is a different perturbation.** The weights stay exact. The error sits in stored keys and values. Each entry is written once, so its error is frozen, but each new token gets a new error. A new token also reads rounded entries before it writes its own keys and values in the next layers, so new entries inherit some error from old ones. A key error Δk_i changes the attention logit of cached token i by ε_i = qᵀΔk_i/√d (q is the query, d the head dimension) and then passes through a softmax, so idea 13 applies with the attention weights a in place of p. A value error passes linearly: Δo = Σ_i a_i·Δv_i. If the Δv_i are independent, zero-mean, with variance σ_v² per coordinate, then Var(Δo) = σ_v²·Σ_i a_i². Σ a_i² is 1/n for uniform attention over n tokens and 1 for one-hot attention. A shared offset μ in all values passes in full, because Σ a_i = 1. KIVI: keys per channel, values per token. Relative attention-score error for keys: 47.00 per token against 9.60 per channel. Relative attention-output error for values: 3.55 per token against 49.89 per channel (Llama-2-13B, mean over layers and heads) [checked: 2402.02750 Table 2]. The table caption does not state the bit width; the text around it is about 2 bits [unverified: bit width of that table]. KIVI keeps the newest tokens in full precision (residual length R = 128 by default) and passes exact keys and values to the next layers during prefill [checked: 2402.02750 Sec 3.3, 4.1].

20. **Rounding bias needs repeated rounding.** If a state is rounded again at every step (s ← round(s + u)) and the update u is smaller than half a grid step, RTN drops the update every time and the state never moves. SR keeps it on average (Exercise 3). Weights and KV entries are rounded once, so this mechanism does not apply to them; for a write-once store RTN has half the error variance of SR. Reported for bf16 recurrent state in hybrid models: same-width stochastic rounding removes the length increase in four of the five affected models, "so the failure is mostly rounding bias, not missing resolution" [from notes: 2611.12127].

---

## 2. Level ladder

- **Level 2 (can follow along).** Can say what RTN, GPTQ, AWQ and HQQ each use (grid only; calibration inputs and a weight update; calibration activation sizes and a channel scale; a robust norm on weights, no data). Knows that reported length inflation at the same bit width differs by method.
- **Level 3 (can do with help).** Can write the layer objective δᵀAδ, do one GPTQ compensation step on a 2-weight example, and compute how a logit shift changes a two-token distribution. With a hint, sees why the KL is larger at uncertain positions.
- **Level 4 (can do independently).** Can name the error each method leaves and predict when a method fails (inputs unlike the calibration set, lower bit width). Derives KL ≈ ½·Var_p(ε) and the flip rate Φ(−m/σ). Can design a dose measurement and a switch-back experiment for a quantized model, with paired seeds and a decision about the KV cache.
- **Level 5 (can teach it and handle edge cases).** Separates frozen error from fresh noise and bias from variance, for weights, KV entries and rewritten state. Finds the confounds in the published comparisons (calibration data, greedy single runs, direction of the shift not measured at matched prefixes). Can explain why the positions with the largest KL need not be the positions that most need verifying.

---

## 3. Misconceptions

1. **"The same bit width means the same damage."** The bit width fixes only the grid step. The damage depends on where the error points relative to A, then on the softmax, then on which tokens flip. At INT4 on Qwen3-4B, RTN inflates CoT by +42.5% and HQQ by +6.0% [checked: 2606.25519 Table 2].

2. **"GPTQ puts the weights closer to the originals than RTN."** It puts them farther away. In Exercise 1 the weight MSE is 2.2 times larger than RTN's while the output error on calibration-like inputs is 4.2 times smaller. On inputs unlike the calibration set the output error is 2.2 times larger than RTN's.

3. **"Quantization noise is like sampling noise; more samples average it out."** The error is a fixed change of the function. A greedy run of the quantized model is fully reproducible. It can look like a higher temperature at peaked positions (idea 15), but only as an average over positions, never over repeated runs of one position.

4. **"KL is large at uncertain positions because quantization is less accurate there."** A logit error of the same size at every position already gives this pattern, because KL ≈ ½·Var_p(ε) is near zero where p is one-hot (Exercise 2a: Spearman 0.62 with identical noise everywhere). The ρ = 0.92 result does not, by itself, say anything about where the logit error is large.

5. **"Accuracy is unchanged, so the quantized model is free."** Time is tokens times time per token. RTN INT4 on Qwen3-30B keeps the average accuracy on the reasoning tasks (75.6 against 74.8 for BF16) but adds +43.0% CoT tokens [checked: 2606.25519 Tables 1-2]. The 4-bit kernel gives about 1.2 to 1.4 times more tokens per second [checked: 2606.25519 Sec 4]. 1.43 / 1.3 = 1.10, so the trace takes about 10% longer than in BF16. Above +30% inflation a 1.3 times kernel gain is gone (Question Q2). Do not use RTN INT4 on Qwen3-4B as the example here: it inflates by +42.5%, but it also loses 11.3 accuracy points (72.5 to 61.2) [checked: Table 1]. The paper's own example: BBH with Qwen3-4B takes 285 s in BF16 and 292 s in INT4 [checked: Sec 4].

6. **"The paper showed that, at the same position, quantization raises the probability of 'Wait'."** It showed three things. Markers are among the tokens with the highest KL (the KL is attached to the token that the quantized model sampled). Quantized traces contain relatively more high-KL tokens (ratio to low-KL tokens 0.57 in BF16, 1.15 under 3-bit AWQ). Penalising markers shortens the trace [checked: 2606.00206 Sec 4.3, Sec 5]. It did not compare q_t(marker) with p_t(marker) at matched prefixes. So "more markers in the quantized trace" is measured. "Quantization moves probability onto markers at a given position" is still a hypothesis to test.

7. **"Stochastic rounding fixed the bf16 state, so it will fix 4-bit weights or a 4-bit KV cache."** The fix works for a state that is rounded again at every step. Weights and KV entries are rounded once; SR then gives a frozen error with twice the variance (ideas 4 and 20).

8. **"KV quantization is weight quantization applied to another tensor."** With quantized weights every computation is perturbed, including the current step and the KV it writes. With a quantized KV cache the current step uses exact weights and only its reads of older tokens are perturbed. An entry's own rounding error can be removed later by rewriting that entry. Entries that later tokens wrote after reading blurred entries keep what they inherited. The two cases load the token channel and the cache channel of Part C differently.

---

## 4. Question bank

Each entry: level, type, question, answer key, likely wrong answer, follow-up.

### quantization-Q1
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "A group of weights has minimum −0.6 and maximum 0.9. You quantize it with 4-bit asymmetric min-max round-to-nearest. What are the grid step, the largest possible error of one weight, and the RMS error if the error is spread evenly? By what factor does the RMS error grow at 3 bits?"
- **Answer key:** Step s = 1.5 / (2⁴ − 1) = 0.1. Largest error s/2 = 0.05. RMS = s/√12 = 0.0289. At 3 bits s = 1.5/7 = 0.214, RMS = 0.0619. Factor 15/7 = 2.14 (not exactly 2), so the MSE grows 4.6 times.
- **Likely wrong answer:** "Step 1.5/16 = 0.094, and one bit less doubles the error exactly."
- **Follow-up:** "How many grid points does a 4-bit code have, and how many gaps are there between the lowest and the highest point?"

### quantization-Q2
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "A 4-bit kernel gives 1.3 times more tokens per second than BF16. With RTN the reasoning trace gets 42.5% longer; with AWQ it gets 8.2% longer. For each method, is the time to finish the trace shorter or longer than BF16, and by how much? At what length increase does the kernel gain vanish?"
- **Answer key:** Time ratio = length ratio / speed ratio. RTN: 1.425 / 1.3 = 1.096, so 9.6% slower than BF16. AWQ: 1.082 / 1.3 = 0.832, so 16.8% faster. Break-even at +30%. (The inflation numbers are from 2606.25519 Table 2, Qwen3-4B INT4; the paper reports a 1.2 to 1.4 times kernel gain [checked: Sec 4]. Prompt and answer tokens are ignored here. Accuracy is not equal in the two cases: on the reasoning tasks BF16 scores 72.5, AWQ INT4 73.2 and RTN INT4 61.2 [checked: 2606.25519 Table 1].)
- **Likely wrong answer:** "Both are faster, because 4-bit is 1.3 times faster per token."
- **Follow-up:** "The user waits for the whole trace. How many tokens does the RTN model write for every 1,000 BF16 tokens, and how long does that take at 1.3 times the speed?"

### quantization-Q3
- **Level:** 2. **Type:** explain why.
- **Question:** "GPTQ and RTN use the same grid. After GPTQ, the weights are on average farther from the original weights than after RTN. Why can the layer output still be closer to the original output?"
- **Answer key:** The output error of a neuron is δᵀAδ, not δᵀδ. When inputs are correlated, errors on two weights can cancel in the output. GPTQ rounds one weight, then shifts the weights not yet rounded so that the output error on calibration inputs shrinks. The shift moves them away from their original values, into directions the inputs rarely excite.
- **Likely wrong answer:** "It cannot. RTN is nearest by definition, so nothing can beat it."
- **Follow-up:** "Two inputs are always equal, x₁ = x₂. The weights are 0.4 and 0.3 and the grid is the integers. What does the output depend on, and which rounded pair keeps it closest?"

### quantization-Q4
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "One neuron has two weights w = (0.4, 0.3). The grid is the integers. The inputs have A = E[x xᵀ] = [[1, 0.9], [0.9, 1]]. (a) Compute the output error δᵀAδ for round-to-nearest. (b) Do GPTQ: round w₁ first, apply the compensation update to w₂, then round w₂. Give the final weights and the output error. (c) Is the GPTQ result the best grid point?"
- **Answer key:** (a) RTN gives (0, 0), δ = (−0.4, −0.3), error = 0.16 + 0.09 + 2·0.9·0.12 = 0.466. (b) e = 0 − 0.4 = −0.4. A⁻¹ = (1/0.19)·[[1, −0.9], [−0.9, 1]], so the update on w₂ is e·[A⁻¹]₂₁/[A⁻¹]₁₁ = (−0.4)·(−0.9) = +0.36. w₂ becomes 0.66 and rounds to 1. Final (0, 1), δ = (−0.4, +0.7), error = 0.16 + 0.49 − 2·0.9·0.28 = 0.146. The weight error δᵀδ rose from 0.25 to 0.65. (c) No. (1, 0) gives δ = (0.6, −0.3) and error 0.36 + 0.09 − 0.324 = 0.126. The fixed order and greedy rounding miss it.
- **Likely wrong answer:** "GPTQ also gives (0, 0), because 0.3 rounds to 0", or the update has the wrong sign (w₂ = −0.06).
- **Follow-up:** "The inputs are almost equal, so the output is close to (w₁ + w₂)·x. The true sum is 0.7. Rounding w₁ down lost 0.4 of it. In which direction must w₂ move to bring the sum back?"

### quantization-Q5
- **Level:** 3. **Type:** explain why.
- **Question:** "AWQ multiplies a weight column by c > 1 before rounding and divides the matching input by c. Why does this lower the output error from that column, and what is the price?"
- **Answer key:** The rounding error of the scaled weight has about the same absolute size (set by the step Δ'), but it is multiplied by x/c, so its output error shrinks by (Δ'/Δ)·(1/c) [checked: 2306.00978 Eq. 2-3]. Price: if the scaled column now sets the group maximum, Δ' > Δ and every other weight in the group gets a larger error. So AWQ searches α in c = s_X^α to balance the two. No weight is moved to cancel another weight's error.
- **Likely wrong answer:** "Scaling makes the weight land exactly on the grid", or "there is no price because the scaling is undone".
- **Follow-up:** "The step of a group is max|w| divided by 2^(N−1). You scale the column that holds the largest weight by 2. What happens to the step, and to the error of the unscaled columns?"

### quantization-Q6
- **Level:** 3. **Type:** explain why.
- **Question:** "HQQ uses no calibration data. What does it minimise, over which variable, and which kind of error can it therefore not reduce?"
- **Answer key:** It minimises an l_p norm with p < 1 (0.7) of the weight error W − Q⁻¹(Q(W)), over the zero-point z with the step fixed [checked: HQQ post]. The p < 1 norm tolerates a few large errors on outliers to fit most weights better. It never sees activations, so it cannot make the error small in the directions where the inputs are large (it cannot shape δ by A) and it cannot cancel errors across correlated inputs.
- **Likely wrong answer:** "It minimises the layer output error, like GPTQ, only faster."
- **Follow-up:** "To compute the layer output error you need X. Where would HQQ get X from?"

### quantization-Q7
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "At two positions there are only two candidate tokens. At position 1 the logit margin of the top token is 0.2. At position 2 it is 4.6. Quantization lowers both margins by 0.5. Compute the top token's probability before and after at each position. Under greedy decoding, where does the token change?"
- **Answer key:** p = 1/(1 + e^(−m)). Position 1: 0.550 → 0.426, a change of 0.124; the margin is now −0.3, so greedy flips. Position 2: 0.990 → 0.984, a change of 0.006; no flip. The same logit error moves the probability about 20 times more at the uncertain position.
- **Likely wrong answer:** "Both drop by the same amount, since the logit error is the same."
- **Follow-up:** "What is the slope of the sigmoid at p = 0.99? Compare p(1 − p) at 0.55 and at 0.99."

### quantization-Q8
- **Level:** 3. **Type:** connect two ideas.
- **Question:** "A paper reports that the per-position KL between the BF16 and the 3-bit model correlates with the BF16 entropy at Spearman 0.92. Suppose the logit error had exactly the same size at every position. Would you still expect a positive correlation? Why?"
- **Answer key:** Yes. KL(p ‖ p') ≈ ½·Var_p(ε). At a one-hot position the variance under p is near zero whatever ε is; at a flat position it is close to the full variance of ε. So the softmax alone produces the correlation. The result does not show that quantization is less accurate at uncertain positions. To test that, compare the logit error itself (or the pre-softmax hidden-state error) across positions.
- **Likely wrong answer:** "No. Equal error gives equal KL, so the correlation shows that quantization damages uncertain positions more."
- **Follow-up:** "p puts 0.999 on one token. You add +1 to the logit of some other token. Roughly how much probability moves?"

### quantization-Q9
- **Level:** 4. **Type:** predict (numeric).
- **Question:** "The logit error over three tokens is ε = (−0.2, +0.4, 0). Compute ½·Var_p(ε) for p = (0.5, 0.3, 0.2) and for p = (0.98, 0.01, 0.01). What does the ratio tell you?"
- **Answer key:** First p: E[ε] = −0.10 + 0.12 = 0.02; E[ε²] = 0.5·0.04 + 0.3·0.16 = 0.068; Var = 0.068 − 0.0004 = 0.0676; ½·Var = 0.0338 (exact KL 0.0354). Second p: E[ε] = −0.196 + 0.004 = −0.192; E[ε²] = 0.0392 + 0.0016 = 0.0408; Var = 0.0408 − 0.0369 = 0.0039; ½·Var = 0.0020 (exact KL 0.0024). Ratio about 17 (15 with exact KL). The same logit error is about 15 times more visible at the uncertain position.
- **Likely wrong answer:** "KL ≈ ½·Σ ε² = 0.10 at both positions."
- **Follow-up:** "Add 5 to every logit. Does the distribution change? What does your formula give for that ε?"

### quantization-Q10
- **Level:** 4. **Type:** predict (numeric).
- **Question:** "Greedy decoding. In a trace, 10% of positions are uncertain with top-1 to top-2 margin 0.5; the rest have margin 5. Across positions the quantization error on the margin looks like N(0, σ²). How many tokens flip per 1,000 at σ = 0.5? And at σ = 0.25, about one more bit?"
- **Answer key:** Flip rate at margin m is Φ(−m/σ). σ = 0.5: uncertain positions Φ(−1) = 0.159, so 100 × 0.159 = 15.9 flips; confident positions Φ(−10) ≈ 10⁻²³, so 0. σ = 0.25: Φ(−2) = 0.0228, so 2.3 flips. Half the error gives 7 times fewer flips. The dose is not proportional to the error size, and all flips sit at the uncertain positions.
- **Likely wrong answer:** "Half the noise, half the flips: about 8."
- **Follow-up:** "At σ = 0.25 the margin 0.5 is how many standard deviations away? What fraction of a normal distribution lies beyond that?"

### quantization-Q11
- **Level:** 4. **Type:** find the flaw (numeric part).
- **Question:** "A student runs BF16 with seed 1 and the INT4 model with seed 2 at temperature 0.6. In 30% of the problems the first token where the two traces differ is 'Wait', 'But' or 'Alternatively'. The student concludes that quantization causes hesitation. What is wrong, and what would two BF16 runs with different seeds show at a position with p = (0.5, 0.3, 0.2)?"
- **Answer key:** Two runs of the same model with independent seeds differ at that position with probability 1 − Σp² = 1 − (0.25 + 0.09 + 0.04) = 0.62. First divergence happens at high-entropy positions for any two sampled runs, and at high-entropy positions it is 2 to 4 times more likely that a marker is among the 20 most probable tokens [checked: 2606.00206 App. B.2, Fig. 7]. The needed controls: BF16 against BF16 with different seeds, and paired sampling (shared random numbers). With the best pairing the two models differ with probability TV(p, p'); a shared uniform number through the inverse CDF gives more than that (Exercise 2d: 0.33 against a bound of 0.09; shared Gumbel noise gives 0.12).
- **Likely wrong answer:** "Nothing is wrong if the sample is large enough."
- **Follow-up:** "Run the BF16 model twice with seeds 1 and 2. At which kind of position do those two traces first differ?"

### quantization-Q12
- **Level:** 4. **Type:** find the flaw.
- **Question:** "On R1-Distill-Qwen-1.5B, MATH-500, 3-bit AWQ writes 23.4K CoT tokens and 3-bit GPTQ writes 6.8K, against 5.2K for BF16. A colleague concludes: error compensation protects reasoning length, so always use GPTQ. Give two reasons not to conclude this."
- **Answer key:** (1) In that paper the calibration sets differ: GPTQ was calibrated on reasoning traces that the model generated on NuminaMath problems, AWQ on Pile text [checked: 2606.00206 App. A.1]. Method and calibration data are confounded. Calibration text alone can move length a lot: for Qwen3-4B with INT3 GPTQ, Wikitext against math reasoning traces gives 30k against 12k CoT tokens [checked: 2606.25519 Sec 7.2, Fig. 5]. (2) The order is not stable in the other paper, where the calibrated methods share one calibration set: on Qwen3-4B at INT4, AWQ +8.2% is below GPTQ +12.0%; at INT3 HQQ and AWQ are far above GPTQ; on Qwen3-30B INT4, GPTQ is lowest [checked: 2606.25519 Table 2]. Also: one model, and the second paper uses one greedy run per cell.
- **Likely wrong answer:** "The gap is 4.5 times against 1.3 times. It is too large to be anything but the method."
- **Follow-up:** "GPTQ fits A = E[x xᵀ] from calibration inputs. What happens to its output error on inputs that look unlike the calibration set (Exercise 1, last column)?"

### quantization-Q13
- **Level:** 4. **Type:** apply to a new case.
- **Question:** "GPTQ minimises the output error of each layer on 128 calibration segments of web text. You then run the model on a 10,000-token math reasoning trace. Name three separate reasons why the next-token distribution can still shift."
- **Answer key:** Any three of: (1) the grid is discrete, so a residual error stays, and the last columns have nothing left to absorb it; (2) greedy fixed order is not the optimum; (3) A was estimated on web text; reasoning traces excite other directions, where GPTQ has put its error; (4) the objective is per layer, so errors add up through depth, and the squared output error of a layer is not the KL at the softmax; (5) small logit errors matter only at small-margin positions, and long reasoning traces have many of them in sequence. Note for the tutor: the question describes the original GPTQ setup (web text). In the two length papers GPTQ was calibrated on reasoning traces, so reason (3) is weaker for their tables [checked: 2606.00206 App. A.1; 2606.25519 Sec 7.2].
- **Likely wrong answer:** "After compensation the layer output error is zero on the calibration data, so only out-of-distribution inputs matter."
- **Follow-up:** "In the two-weight example (Q4), what was the output error after compensation and rounding? Was it zero?"

### quantization-Q14
- **Level:** 4. **Type:** connect two ideas (Part A).
- **Question:** "A paper takes quantized versions of one small reasoning model. Adding +3 to the logits of 50 hesitation markers makes the reasoning trace 395% longer on average. Adding +3 to 50 random tokens changes almost nothing. Part A defines the dose as how often a token is changed and how unlikely the changed token is. What does this result say about that definition?"
- **Answer key:** Two interventions with a similar count of forced, unlikely tokens give very different length. So the dose as defined (rate and unlikeliness) cannot be the only predictor; the identity or role of the changed token matters. A dose curve must at least be split by token class (marker against other), or the dose must weight tokens by their effect on length. Caveat: the two boosts were not matched on dose in the paper [checked: 2606.00206 App. B.4]; random tokens at +3 may be sampled far less often than markers at +3, because markers already sit near the top at uncertain positions. A clean test matches the realised flip rate. One more arm helps here: boosting the 50 lowest-KL tokens (math and formatting tokens, which are sampled often) raises length by only about 10% even at +4 [checked: App. B.4]. So tokens that are sampled often do not inflate length either, which supports "identity matters". Setting of all these numbers: quantized R1-Distill-Qwen-1.5B, mean over six quantization configurations and five benchmarks; the boost was not run on BF16 [checked: App. A.3, B.4].
- **Likely wrong answer:** "It confirms the dose idea: more boost, more length."
- **Follow-up:** "How often is a random token actually sampled after a +3 boost, compared with a marker that was already in the top 20? Were the two arms given the same dose?"

### quantization-Q15
- **Level:** 4. **Type:** apply to a new case (Part A).
- **Question:** "You add a 4-bit RTN model as one more arm in the Part A experiment, next to sparse attention, random token swaps and a higher temperature. How do you measure its dose without generating anything, and on which side of the 'could not see the context' question does it sit?"
- **Answer key:** Dose: run both models teacher-forced on BF16 traces. Per position record TV(p_t, p'_t) (the smallest possible flip probability), or the argmax flip rate for greedy, and for flipped tokens the BF16 log-probability of the token the quantized model picks. Split by position entropy and by token class. Side: weight quantization hides no context; every past token is still read, with slightly wrong weights. It belongs with temperature and token swaps as a "sees everything" arm. A quantized KV cache is in between: all tokens are read, but blurred. If quantization at matched dose inflates length as much as sparse attention, missing context is not needed to explain the tax.
- **Likely wrong answer:** "Generate with both models and count the tokens that differ."
- **Follow-up:** "After the first differing token the two traces have different prefixes. What are you comparing at the second differing token?"

### quantization-Q16
- **Level:** 4. **Type:** apply to a new case (Parts B and C).
- **Question:** "Part B: you generate the first 2,000 reasoning tokens with the INT4 model, then switch to BF16 for the rest. At the switch you must decide what to do with the KV cache. What are the two options, and what does the difference between them measure?"
- **Answer key:** Option 1: keep the KV cache that the INT4 model wrote. The BF16 model then reads the INT4 text and INT4-written keys and values. Option 2: run the BF16 model over the 2,000-token prefix again (a fresh prefill) and continue. Now only the text carries over. Remaining length in option 2 against a BF16 run from its own prefix measures harm through the tokens (Part B's lasting damage through text). Option 1 minus option 2 measures harm that travels through the stored KV (Part C). Controls: paired seeds, and a BF16-to-BF16 switch with a re-prefill to measure pure numerical noise. Note for weight quantization the KV channel exists even when no token differs, because the INT4 model writes different KV for the same text.
- **Likely wrong answer:** "Just swap the weights. The cache is a function of the text, so it does not matter."
- **Follow-up:** "The same 2,000 tokens are fed to the BF16 model and to the INT4 model. Are the keys and values at layer 20 identical?"

### quantization-Q17
- **Level:** 4. **Type:** predict (numeric) (Part C).
- **Question:** "An attention head reads 100 cached values. Each value has an independent zero-mean rounding error with standard deviation σ per coordinate. Head A attends uniformly. Head B puts 0.9 on one token and spreads 0.1 evenly over the other 99. What is the standard deviation of the output error for each head? What changes if every value also carries the same offset μ?"
- **Answer key:** Var(Δo) = σ²·Σa_i². Head A: Σa² = 100·0.01² = 0.01, so std = 0.10σ. Head B: 0.81 + 99·(0.1/99)² = 0.8101, so std = 0.90σ. The shared offset passes in full in both heads: Σa_i·μ = μ. So zero-mean value noise is averaged away by diffuse heads and not by focused heads; a bias is never averaged away. For Part C: this harm sits in the cache. Rewriting the few entries that focused heads read removes the rounding error of those entries. It does not remove error that later tokens already took in when they read the blurred entries and then wrote their own keys and values.
- **Likely wrong answer:** "Both are σ/10, because 100 entries are averaged", or "0.01σ for head A".
- **Follow-up:** "Head B takes 90% of its output from one value. How much of that value's error reaches the output?"

### quantization-Q18
- **Level:** 4. **Type:** predict (numeric) (Part B).
- **Question:** "A toy model, not from a paper. Every generated token is a hesitation marker with probability h. Each marker adds a re-check segment of 100 tokens, and the tokens in that segment follow the same rule. Without any marker the trace needs 2,000 tokens. What is the expected length at h = 0.002, 0.004 and 0.008? What happens at h = 0.01? In this model, what happens to the remaining length after you switch back to the exact model?"
- **Answer key:** Each token causes on average R = 100h more tokens, each of which does the same, so the expected length is 2,000 / (1 − R). h = 0.002: 2,500. h = 0.004: 3,333 (+33%). h = 0.008: 10,000 (5 times the base, +400%). h = 0.01: R = 1, the expected length is infinite. Every run still ends by itself with probability 1, but some runs are extremely long and hit any token cap. Length is very non-linear in the marker rate. After the switch h returns to its old value at once, so the rest has normal length: the toy model has only a per-token cost (one extra marker costs 100/(1 − R) tokens, 125 at R = 0.2). If real traces stay long after the switch, something that this model lacks keeps h high. [mine]
- **Likely wrong answer:** "Linear: 2,000 + 2,000·h·100, so 2,400, 2,800, 3,600, and 4,000 at h = 0.01."
- **Follow-up:** "The 400 extra tokens at h = 0.002 are also generated tokens. How many markers do they contain on average, and what do those add?"

### quantization-Q19
- **Level:** 5. **Type:** apply to a new case (edge case).
- **Question:** "For bf16 recurrent state in hybrid models, stochastic rounding at the same width removed the extra reasoning length in most models. A colleague proposes stochastic rounding for 4-bit weight quantization, and for a 4-bit KV cache, to remove their length inflation. Predict the result for each and explain."
- **Answer key:** Weights: rounded once, so SR produces one frozen error matrix, with error variance s²/6 instead of s²/12. Nothing is averaged over steps. Expect no gain, probably a loss [mine: a prediction, not a measured result]. The bf16 result works because the state is rounded again at every step: with RTN an update below half a grid step is lost every time (a bias that adds up), and SR keeps it on average [2611.12127, from notes; mechanism as in Exercise 3a]. KV cache: each entry is also written once. SR helps only if a read averages many entries, and even there RTN error is already near zero-mean across entries and has the lower variance (Exercise 3b). So no gain is expected unless the KV quantizer has a real shared bias (for example a floor, or shrinkage toward zero in an MSE-optimal codebook). A level-5 answer adds: SR would help weights only if redrawn at every forward pass (dither), which turns frozen error into fresh noise at twice the variance and costs the kernel speed; whether fresh noise is better for length than frozen error is exactly an open question.
- **Likely wrong answer:** "Yes for both. SR removes the bias, and bias was shown to be the cause."
- **Follow-up:** "How many times is one weight rounded during a 10,000-token generation? How many times is one coordinate of a recurrent state rounded?"

### quantization-Q20
- **Level:** 5. **Type:** find the flaw (adversarial).
- **Question:** "Per-position KL between the BF16 and the quantized model is largest at high-entropy positions. So if you may send only some positions to the BF16 model, you should send the high-entropy ones. But one paper reports that routing the low-entropy predictions to BF16 recovers most of the lost accuracy, and routing the high-entropy ones recovers little. Reconcile the two, and say what is missing before you can decide which tokens need verifying."
- **Answer key:** KL measures how much the distribution moves, not what a changed token costs. At a high-entropy position the BF16 model itself was close to indifferent, so many flips are harmless for the answer (they may still cost length, for example a marker). At a low-entropy position (a digit, an operator) a flip is rare but is a hard error. So accuracy harm and length harm can sit at different positions. Flaws in the comparison: the low-entropy arm is about 75% of all tokens (threshold at the top 25% entropy), so the two arms are not matched in cost; and length per arm is not reported [from notes: 2606.15682]. Missing: a matched-budget comparison that reports both accuracy and length per arm. The research question "which tokens need verifying" needs the harm per verified token, for each kind of harm.
- **Likely wrong answer:** "The two papers contradict each other; one of them must be wrong."
- **Follow-up:** "The model is 50/50 between 'So' and 'Thus' at one position and 99.9% sure of the digit '7' at another. A flip at which position changes the final answer?"

### quantization-Q21
- **Level:** 5. **Type:** find the flaw (edge case).
- **Question:** "On Qwen3-4B at INT4, HQQ (no calibration data, weight-space objective) inflates reasoning by +6.0%, GPTQ (output-error objective, calibrated on the model's own math reasoning traces) by +12.0%, RTN by +42.5%. At INT3, GPTQ gives +48.7% and HQQ +292.1%. These are averages over six benchmarks, one greedy run each. A reviewer says: 'This shows that layer output error does not matter for length.' Another says: 'It shows that it does.' What can you actually conclude, and what would you measure?"
- **Answer key:** Safe: RTN is clearly worst at INT4 (on Qwen3-30B too, +43.0%). At INT3, GPTQ, the method that moves weights to cancel output error, inflates far less than HQQ (+292.1%) and AWQ (+276.1%), which tune a zero-point or a channel scale and then round each weight to nearest. Not safe, three points. (i) The 6-point gap between HQQ and GPTQ at INT4. It comes from one greedy run on about 100 questions per benchmark [checked: 2606.25519 Sec 3], and numerical noise alone can move the length of single runs by thousands of tokens [from notes: 2506.09501]. (ii) Reading the INT3 gap as an effect of the objective alone. GPTQ saw reasoning traces during calibration and HQQ saw no data, so "output-error objective" and "has seen reasoning inputs" are confounded [checked: 2606.25519 Sec 7.2]. AWQ also picks its scale by layer output error and still reaches +276.1% [checked: 2306.00978 Eq. 4-5; 2606.25519 Table 2], so "uses output error" is not enough. (iii) Single benchmarks drive the INT3 averages: on MBPP alone AWQ is +957.4% and HQQ +1051.3% [checked: Table 2]. A possible reading [mine]: at INT4 most of RTN's damage comes from a badly placed grid (outliers stretch the step), which HQQ's zero-point fit, AWQ's scaling and clipping, and GPTQ's weight updates all repair in different ways; at INT3 the grid is too coarse for placement alone, and cancelling errors across weights starts to matter. One more data point: ParoQuant gives +24.5% at INT3, below GPTQ [checked: Table 2]; how ParoQuant works was not read [unverified]. To test: for each method measure, teacher-forced on BF16 reasoning traces, the layer output error, the final-logit KL split by entropy, and the flip rate at marker positions; then check which of them ranks CTIR across methods and bit widths, with several seeds and with the calibration text held fixed.
- **Likely wrong answer:** "HQQ is simply the better method; calibration data is unnecessary."
- **Follow-up:** "The same table has an INT3 row. Which of the two methods wins there, and by how much?"

---

## 5. Exercises

All three were run in the scratch folder with Python 3.11.5 and numpy 1.24.3 on CPU. The outputs below are the real outputs of these scripts.

### Exercise 1. RTN, GPTQ and an AWQ-like scale on one toy layer

**Task.** Build a layer with 16 inputs and 8 outputs. Make the inputs strongly correlated (three shared directions plus weak independent noise). Quantize the weights to 3 bits on a per-row min-max grid with (i) RTN, (ii) GPTQ with the update of core idea 6, (iii) an AWQ-like search over one channel-scale exponent α. Report the weight MSE and the relative output error ‖(Ŵ − W)X‖² / ‖WX‖² on the calibration inputs, on new inputs of the same kind, and on isotropic inputs that the calibration set never showed.

**Expected result.** GPTQ has the largest weight MSE and the smallest output error on calibration-like inputs. On isotropic inputs the order reverses and RTN is best. The AWQ-like method lies between the two in every column.

**How the tutor checks it.** The learner must (a) predict the order of the three methods in each column before running; (b) explain the reversal in the last column with δᵀAδ (GPTQ has put its error where the calibration A is small; isotropic inputs weight all directions equally, so the output error is proportional to the weight MSE); (c) say what this means for a model calibrated on web text and run on reasoning traces.

```python
import numpy as np
rng = np.random.default_rng(0)
d, rows, n, bits = 16, 8, 512, 3

# Correlated inputs: 3 strong shared directions plus weak independent noise.
mix = rng.normal(size=(3, d))
def inputs(m):
    return rng.normal(size=(m, 3)) @ mix * 2.0 + 0.3 * rng.normal(size=(m, d))
Xcal, Xtest = inputs(n), inputs(4000)
Xiso = rng.normal(size=(4000, d))            # inputs the calibration set never showed
W = rng.normal(size=(rows, d))

def make_grid(Wm):                            # per-row asymmetric min-max grid, fixed in advance
    lo = Wm.min(1, keepdims=True); hi = Wm.max(1, keepdims=True)
    return lo, (hi - lo) / (2**bits - 1)
def quant(w, lo, s):                          # round to nearest grid point
    return np.clip(np.round((w - lo) / s), 0, 2**bits - 1) * s + lo

def rtn(Wm):
    lo, s = make_grid(Wm)
    return quant(Wm, lo, s)

def gptq(Wm, X, damp=0.01):
    lo, s = make_grid(Wm)                     # same grid as RTN
    H = 2 * X.T @ X
    H += damp * np.mean(np.diag(H)) * np.eye(d)
    Hinv = np.linalg.inv(H)
    Wq = Wm.copy()
    for j in range(d):                        # fixed left-to-right order
        q = quant(Wq[:, j:j+1], lo, s)[:, 0]
        e = (Wq[:, j] - q) / Hinv[j, j]
        Wq[:, j:] -= np.outer(e, Hinv[j, j:]) # move the error onto the weights not yet rounded
        Wq[:, j] = q
        Hinv = Hinv - np.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]
    return Wq

def awq_like(Wm, X):
    sx = np.abs(X).mean(0)                    # mean activation size per input channel
    best = None
    for a in np.linspace(0, 1, 20):
        sc = sx**a
        Wq = rtn(Wm * sc) / sc                # round the scaled weights, then undo the scale
        err = np.sum(((Wq - Wm) @ X.T)**2)
        if best is None or err < best[0]:
            best = (err, a, Wq)
    return best[2], best[1]

def report(name, Wq):
    wmse = np.mean((Wq - W)**2)
    out = lambda X: np.sum(((Wq - W) @ X.T)**2) / np.sum((W @ X.T)**2)
    print(f"{name:9s} weight MSE {wmse:.4f} | relative output error: calib {out(Xcal):.5f}  "
          f"held-out {out(Xtest):.5f}  isotropic {out(Xiso):.5f}")

ev = np.linalg.eigvalsh(Xcal.T @ Xcal / n)
print("eigenvalues of E[x x^T], largest 4:", np.round(ev[::-1][:4], 2), " smallest:", round(ev[0], 3))
Wr = rtn(W); Wg = gptq(W, Xcal); Wa, a = awq_like(W, Xcal)
report("RTN", Wr); report("GPTQ", Wg); report("AWQ-like", Wa)
print("AWQ-like alpha chosen:", round(a, 3))
print("share of weights where GPTQ and RTN pick different grid points:", round(np.mean(np.abs(Wg - Wr) > 1e-9), 3))
```

Output:

```
eigenvalues of E[x x^T], largest 4: [83.3  45.01 20.72  0.12]  smallest: 0.064
RTN       weight MSE 0.0213 | relative output error: calib 0.02070  held-out 0.02085  isotropic 0.01950
GPTQ      weight MSE 0.0464 | relative output error: calib 0.00491  held-out 0.00450  isotropic 0.04243
AWQ-like  weight MSE 0.0347 | relative output error: calib 0.00777  held-out 0.00794  isotropic 0.03119
AWQ-like alpha chosen: 0.526
share of weights where GPTQ and RTN pick different grid points: 0.172
```

Reading: GPTQ changes only 17% of the grid choices, yet the weight MSE is 2.2 times RTN's (0.0464 / 0.0213), the output error on calibration-like inputs is 4.2 times smaller (0.0207 / 0.0049), and on isotropic inputs it is 2.2 times larger (0.0424 / 0.0195). This toy layer is not an LLM; the "AWQ-like" method is the scale search only, without AWQ's clipping.

### Exercise 2. The same logit noise at every position

**Task.** Make 4,000 positions over a vocabulary of 50 tokens. The positions differ only in how peaked they are (logits = scale × standard normal, scale between 0.5 and 8). Add independent logit noise with the same standard deviation 0.3 at every position. (a) Compute the Spearman correlation between the clean entropy and KL(clean ‖ noisy). (b) Compare the KL with ½·Var_p(ε). (c) By entropy quartile, report the mean KL, the mean TV and the greedy flip rate. (d) Sample one token per position from the clean and from the noisy model with independent seeds, with one shared uniform number (inverse CDF), and with shared Gumbel noise; compare with clean against clean. (e) At one peaked position, redraw the noise many times and compare the mean odds of a tail token with e^(σ²).

**Expected result.** (a) A clearly positive correlation although the noise is the same everywhere. (b) The approximation matches. (c) KL, TV and flip rate all rise with entropy. (d) With independent seeds the clean-against-clean difference rate is as large as clean-against-noisy; pairing brings it down toward the TV bound, and Gumbel pairing gets much closer than a shared uniform number. (e) The mean odds grow by e^(σ²).

**How the tutor checks it.** The learner must state before running whether (a) is zero or positive, and after running explain in one sentence why the published ρ = 0.92 does not prove that quantization is less accurate at uncertain positions. For (d) the learner must say which pairing to use when measuring a dose and why "same seed" in a real sampler is not the best pairing.

```python
import numpy as np
rng = np.random.default_rng(0)
V, T, sigma = 50, 4000, 0.3

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
def rank(a):                                             # average ranks, so ties are handled
    a = np.asarray(a); r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(len(a))
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]
def spearman(a, b):
    return np.corrcoef(rank(a), rank(b))[0, 1]

# Positions differ only in how peaked they are. The logit noise has the same size everywhere.
scale = np.exp(rng.uniform(np.log(0.5), np.log(8.0), size=(T, 1)))
z = scale * rng.normal(size=(T, V))
eps = sigma * rng.normal(size=(T, V))
p, q = softmax(z), softmax(z + eps)

H = -(p * np.log(p)).sum(1)                              # entropy of the clean model
KL = (p * (np.log(p) - np.log(q))).sum(1)                # KL(clean || noisy)
half_var = 0.5 * ((p * eps**2).sum(1) - (p * eps).sum(1)**2)
TV = 0.5 * np.abs(p - q).sum(1)
flip = p.argmax(1) != q.argmax(1)                        # greedy token changes
srt = np.sort(z, 1); margin = srt[:, -1] - srt[:, -2]    # top-1 minus top-2 logit

print("(a) Spearman(entropy, KL) =", round(spearman(H, KL), 3))
print("(b) mean KL =", f"{KL.mean():.5f}", " mean 0.5*Var_p(eps) =", f"{half_var.mean():.5f}",
      " Spearman(KL, 0.5*Var) =", round(spearman(KL, half_var), 3))
print("(c) by entropy quartile (low to high):")
edges = np.quantile(H, [0, .25, .5, .75, 1])
for i in range(4):
    m = (H >= edges[i]) & (H <= edges[i+1])
    print(f"    H in [{edges[i]:.2f}, {edges[i+1]:.2f}]  mean KL {KL[m].mean():.5f}  mean TV {TV[m].mean():.4f}"
          f"  greedy flip rate {flip[m].mean():.4f}  median margin {np.median(margin[m]):.2f}")
print("    overall greedy flip rate", round(flip.mean(), 4), " mean TV", round(TV.mean(), 4))
print("    Spearman(margin, flip) =", round(spearman(margin, flip.astype(float)), 3))

# (d) Two samplers, same position. Independent seeds versus one shared uniform number.
u1, u2 = rng.uniform(size=T), rng.uniform(size=T)
def draw(dist, u):
    return (dist.cumsum(1) < u[:, None]).sum(1).clip(max=V - 1)
print("(d) clean vs clean, independent seeds: tokens differ at", round(np.mean(draw(p, u1) != draw(p, u2)), 3),
      " (theory 1 - sum p^2 =", round(np.mean(1 - (p**2).sum(1)), 3), ")")
print("    clean vs noisy, independent seeds: tokens differ at", round(np.mean(draw(p, u1) != draw(q, u2)), 3))
print("    clean vs noisy, shared uniform (inverse CDF): tokens differ at", round(np.mean(draw(p, u1) != draw(q, u1)), 3))
g = rng.gumbel(size=(T, V))                              # shared Gumbel noise: token = argmax(log prob + g)
print("    clean vs noisy, shared Gumbel noise        : tokens differ at",
      round(np.mean((np.log(p) + g).argmax(1) != (np.log(q) + g).argmax(1)), 3))
print("    lower bound for any pairing = mean TV =", round(TV.mean(), 4))

# (e) One peaked position, fresh noise many times: what happens to a tail token on average?
zc = np.array([6.0, 2.0, 0.0, 0.0, 0.0]); pc = softmax(zc)
for s in (0.3, 1.0):
    qs = softmax(zc + s * rng.normal(size=(200000, 5)))
    print(f"(e) sigma={s}: p(top) {pc[0]:.4f} -> mean {qs[:,0].mean():.4f};  p(token 2) {pc[1]:.4f} -> mean {qs[:,1].mean():.4f}"
          f";  mean odds ratio (token 2 : top) / clean = {np.mean(qs[:,1]/qs[:,0])/(pc[1]/pc[0]):.3f}  (theory e^(sigma^2) = {np.exp(s*s):.3f})")
```

Output:

```
(a) Spearman(entropy, KL) = 0.621
(b) mean KL = 0.03341  mean 0.5*Var_p(eps) = 0.03346  Spearman(KL, 0.5*Var) = 0.994
(c) by entropy quartile (low to high):
    H in [0.00, 1.33]  mean KL 0.01407  mean TV 0.0473  greedy flip rate 0.0260  median margin 2.14
    H in [1.33, 2.60]  mean KL 0.03465  mean TV 0.1005  greedy flip rate 0.1450  median margin 0.71
    H in [2.60, 3.47]  mean KL 0.04127  mean TV 0.1138  greedy flip rate 0.2780  median margin 0.37
    H in [3.47, 3.84]  mean KL 0.04366  mean TV 0.1176  greedy flip rate 0.4390  median margin 0.19
    overall greedy flip rate 0.222  mean TV 0.0948
    Spearman(margin, flip) = -0.513
(d) clean vs clean, independent seeds: tokens differ at 0.732  (theory 1 - sum p^2 = 0.736 )
    clean vs noisy, independent seeds: tokens differ at 0.737
    clean vs noisy, shared uniform (inverse CDF): tokens differ at 0.326
    clean vs noisy, shared Gumbel noise        : tokens differ at 0.118
    lower bound for any pairing = mean TV = 0.0948
(e) sigma=0.3: p(top) 0.9749 -> mean 0.9727;  p(token 2) 0.0179 -> mean 0.0194;  mean odds ratio (token 2 : top) / clean = 1.092  (theory e^(sigma^2) = 1.094)
(e) sigma=1.0: p(top) 0.9749 -> mean 0.9433;  p(token 2) 0.0179 -> mean 0.0393;  mean odds ratio (token 2 : top) / clean = 2.723  (theory e^(sigma^2) = 2.718)
```

Reading: (a) identical noise everywhere still gives Spearman 0.62 between entropy and KL. The KL levels off near ½σ² = 0.045 at flat positions, which is why the value is not higher. (d) With independent seeds, clean against clean (0.732) is almost the same as clean against noisy (0.737): unpaired runs cannot show the effect of quantization at one position. A shared uniform number still gives 0.326, about 3.4 times the TV bound of 0.095; shared Gumbel noise gives 0.118. (c) Flips sit at small margins: Spearman(margin, flip) = −0.51. The flip variable has only two values, so the rank function must average tied ranks; without that the number is wrong (an earlier version of this exercise printed −0.36). (e) Fresh noise is redrawn here to show the average; a real quantized model has one fixed error per position.

### Exercise 3. Rounding once and rounding again

**Task.** Grid step 1/16. (a) A state starts at 1.0 and gets the update u = 0.01 one thousand times, with rounding after every update. Compare RTN and stochastic rounding (SR) with the exact result. (b) Round 200,000 random values once with RTN, SR and floor. Report the mean and RMS error of one value, and the RMS error of a uniform average over n = 1, 10, 100, 1,000 values. (c) Read 100 rounded values with uniform attention and with focused attention (0.9 on one entry).

**Expected result.** (a) Exact 11.0; RTN stays at 1.0 because 0.01 is less than half a step (0.03125); SR ends near 11. (b) One value: RTN RMS 1/√12 = 0.289 steps, SR 1/√6 = 0.408 steps, floor has mean −0.5. Averages over n: RTN and SR both fall like 1/√n and RTN stays below SR; floor stays at 0.5. (c) Focused attention keeps about 0.9 of the single-value error.

**How the tutor checks it.** The learner must predict (a) before running and explain it with "half a grid step". Then answer: for a store that is written once (weights, KV entries), which of RTN and SR has the smaller error, and why does the bf16-state result not transfer? Then: which heads feel value noise most?

```python
import numpy as np
rng = np.random.default_rng(0)
step = 1 / 16                                   # grid step (4 fractional bits)

def rtn(x):   return np.round(x / step) * step
def floor(x): return np.floor(x / step) * step
def sr(x):                                      # stochastic rounding: up with probability = fractional part
    y = x / step; f = y - np.floor(y)
    return (np.floor(y) + (rng.uniform(size=np.shape(x)) < f)) * step

# (a) A state that is rewritten every step: s <- round(s + u), with u smaller than half a grid step.
u, steps, trials = 0.01, 1000, 200
s_rtn = 1.0
s_sr = np.full(trials, 1.0)
for _ in range(steps):
    s_rtn = rtn(s_rtn + u)
    s_sr = sr(s_sr + u)
print(f"(a) exact final state {1.0 + u*steps:.3f} | RTN {s_rtn:.3f} | SR mean {s_sr.mean():.3f}, std over {trials} trials {s_sr.std():.3f}")

# (b) Values that are written once, then read as an attention-weighted mean over n entries.
print("(b) RMS error of one rounded value, in grid steps: theory RTN 1/sqrt(12) = %.4f, SR 1/sqrt(6) = %.4f" % (12**-0.5, 6**-0.5))
N = 200000
v = rng.uniform(-4, 4, size=N)
for name, fn in (("RTN", rtn), ("SR", sr), ("floor", floor)):
    e = (fn(v) - v) / step
    line = f"    {name:5s} single value: mean error {e.mean():+.4f}, RMS {np.sqrt((e**2).mean()):.4f} | RMS error of a uniform read over n entries:"
    for n in (1, 10, 100, 1000):
        m = e[: (N // n) * n].reshape(-1, n).mean(1)
        line += f"  n={n}: {np.sqrt((m**2).mean()):.4f}"
    print(line)

# (c) Same store, but the read is focused: weight 0.9 on one entry, 0.1 spread over 99 others.
a = np.full(100, 0.1 / 99); a[0] = 0.9
print("(c) sum of squared attention weights: uniform over 100 = %.4f, focused = %.4f" % (100 * 0.01**2, (a**2).sum()))
for name, fn in (("RTN", rtn), ("SR", sr)):
    e = ((fn(v) - v) / step)[: (N // 100) * 100].reshape(-1, 100)
    print(f"    {name:5s} RMS read error: uniform {np.sqrt((e.mean(1)**2).mean()):.4f}, focused {np.sqrt(((e @ a)**2).mean()):.4f}")
```

Output:

```
(a) exact final state 11.000 | RTN 1.000 | SR mean 11.078, std over 200 trials 0.733
(b) RMS error of one rounded value, in grid steps: theory RTN 1/sqrt(12) = 0.2887, SR 1/sqrt(6) = 0.4082
    RTN   single value: mean error +0.0006, RMS 0.2888 | RMS error of a uniform read over n entries:  n=1: 0.2888  n=10: 0.0909  n=100: 0.0286  n=1000: 0.0091
    SR    single value: mean error -0.0014, RMS 0.4088 | RMS error of a uniform read over n entries:  n=1: 0.4088  n=10: 0.1303  n=100: 0.0408  n=1000: 0.0127
    floor single value: mean error -0.5001, RMS 0.5774 | RMS error of a uniform read over n entries:  n=1: 0.5774  n=10: 0.5082  n=100: 0.5009  n=1000: 0.5002
(c) sum of squared attention weights: uniform over 100 = 0.0100, focused = 0.8101
    RTN   RMS read error: uniform 0.0286, focused 0.2608
    SR    RMS read error: uniform 0.0401, focused 0.3665
```

Reading: (a) is the only case where SR beats RTN. In (b) and (c), the write-once cases, RTN is better than SR at every n; only a shared bias (floor) survives averaging. The values here are random, so RTN errors are independent across entries. Real keys and values are correlated across tokens, and then RTN errors can be correlated too; that case is not covered here.

---

## 6. Diagnostic picks

- **Level 2-3: quantization-Q4** (two-weight GPTQ step). One short computation shows whether the learner holds the objective δᵀAδ and the direction of the compensation, or only the slogan "GPTQ is better than RTN". Part (c) separates level 3 from level 4.
- **Level 4-5: quantization-Q19** (stochastic rounding for weights and for a KV cache). A level-4 answer says the error is frozen because the tensor is rounded once. A level-5 answer adds the factor 2 in variance, the averaging inside one attention read, the difference from a state that is rounded at every step, and what redrawn dither would test. It maps onto Part C (what is stored and read again).

---

## 7. Sources

Fetched in the first pass through a fetch tool that returns extracted answers. In the verification pass (2026-11-13) each page below was downloaded once and the raw page text was read for the facts tagged [checked]:

| id | title (as shown on the page) | what I read |
|---|---|---|
| arXiv 2210.17323 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar, Ashkboos, Hoefler, Alistarh) | ar5iv HTML: Eq. 1-3, the description of the method (fixed order, lazy blocks, Cholesky, damping), calibration setup, block inputs from the already quantized model, grid and grouping, Tables 3-4 (a few perplexity rows; not cited as numbers in this file) |
| arXiv 2306.00978 | AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration (Lin et al.) | ar5iv HTML: Eq. 1-5, Table 1, Table 4, the statements on clipping and on no backpropagation or reconstruction |
| no arXiv id confirmed; https://dropbox.github.io/hqq_blog/ | Half-Quadratic Quantization of Large Machine Learning Models (Badri and Shaji, Mobius Labs, November 2023) | the blog post: problem statement, operators, half-quadratic solver, solver settings, Llama-2-7B perplexity table. The older address mobiusml.github.io/hqq_blog/ returned 404. |
| arXiv 2606.25519 | Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models (Lian et al.) | arXiv HTML (v2): Sec 3, Sec 4, Eq. 1-3, Table 1 (Qwen3-4B, Phi-4 and Qwen3-30B rows), Table 2 (all cells), Sec 6, Sec 7.1-7.3, App. A.1, App. E |
| arXiv 2606.00206 | Quantized Reasoning Models Think They Need to Think Longer, but They Do Not (Lotfi, Kirichenko, Li, Liu) | arXiv HTML (v1): Sec 3, 4.1-4.3, 5.1, 6, 7, Eq. 1, Fig. 2-5, 7 and 9 as described in the text, App. A.1-A.3, App. B.2, B.4, App. C (the four 50-token lists). The first pass read only the first 100,000 of 114,679 characters; the verification pass read the rest. |
| arXiv 2402.02750 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al.) | ar5iv HTML: Sec 3.1-3.3, 4.1, Tables 1-3 |

Read directly on disk:

| source | what I read |
|---|---|
| `research_notes/Inference speed question search/harvest_weight_compression.md` | Sec 1 source list and the "Reasoning length under quantization" and "Which items and tokens fail" blocks |
| `research_notes/Inference speed question search/harvest_reasoning_efficiency.md` | the "What controls length inside the model" and "Lossy methods lengthen thinking" blocks, Cards 1-2 |
| `research_notes/Inference speed question search/harvest_kv_compression.md` | Card 5 (bias or error size, keys and values) and the lines found by keyword |
| `research_notes/Inference speed question search/harvest_specdec_self.md`, `harvest_own_projects.md`, `harvest_kernels_gap.md` | only the lines found by a keyword search on quantization |
| `reports/length-tax.html` | full text |
| `research_notes/Length tax preliminaries/specdec_exact.md` | header, question format, sources section (to match the format) |

Cited from those notes only, not re-read in this run [from notes]: 2611.12127 (Lost in Rounding, bf16 recurrent state), 2606.15682 (ReQAT, entropy routing), 2504.04823 (Quantization Hurts Reasoning?, abstract only), 2506.09501 (numerical nondeterminism).

Not opened and not used as evidence: `questions_ranked.md`, the two `.json` files. `verification.md` was searched by keyword in the verification pass, only to cross-check the four [from notes] papers.

Points I could not verify in this run:

- 2606.00206 Fig. 4c: the text does not say which model the ρ = 0.92 is for, nor how many positions were used. It says each KL value is attached to "the token that the quantized model sampled at that position", so the shared prefixes are most likely the quantized model's own traces; the paper does not say so in one sentence.
- 2606.00206 App. C: the 50-marker list was read in the verification pass. Every entry has a leading space. The list also holds common words ("or", "but", "could", "might", "still", "another", "different", "error"), so "marker" is a broad class.
- 2606.25519: Sec 3 says 100 questions per benchmark (30 for AIME25), while its Table 5 lists full dataset sizes. The paper does not reconcile the two.
- 2606.00206 calls the 3-bit GPTQ trace "about 32% longer"; 6.8K / 5.2K gives +31% from the rounded numbers.
- How ParoQuant works (2606.25519 describes it as pairwise Givens rotations plus channel-wise scaling) was not read in its own paper.
- KIVI 2402.02750 Table 2: the bit width is not stated in the caption.
- The reading of AWQ as "diagonal of A only", the reading of the INT4 against INT3 pattern in Q21, and the toy model in Q18 are mine, not from a paper.
- 2611.12127, 2606.15682, 2504.04823 and 2506.09501 are still cited from the notes only [from notes]. Question Q20 rests on 2606.15682 and core idea 20 and Q19 rest on 2611.12127.

---

## Verification log

Verification pass finished Fri Nov 13 15:28 CST 2026 (time from `date`). A second reader checked this pack as a skeptical examiner. No GPU job was run. Nothing was posted, committed or pushed.

**What was checked (52 items):** 20 core ideas, 8 misconceptions, 21 questions, 3 exercises. Also the level ladder, the diagnostic picks and the six source ids.

**How.** Every numeric key was recomputed with CPU Python. Every formula was derived again. All three exercise scripts were run again from the code in this file; the pasted outputs match. Six sources were downloaded once each and the raw page text was read: GPTQ 2210.17323 (ar5iv), AWQ 2306.00978, KIVI 2402.02750, 2606.25519, 2606.00206, and the HQQ blog post. Each arXiv id was confirmed against the paper title on the page.

**Corrections made.**

1. Core idea 16 and misconception 6 (wrong). The old text said the paper 2606.00206 "measures divergence, not direction". The paper does report a direction in free-running traces: the ratio of high-KL to low-KL tokens goes from 0.57 (BF16) to 1.15 (3-bit AWQ), and the abstract says quantized models "disproportionately sample overthinking markers". What is missing is the comparison of q_t(marker) with p_t(marker) at matched prefixes. Both items are rewritten. The misconception is now "at the same position, quantization raises the probability of 'Wait'".
2. Core idea 16 (added). Each KL value is attached to the token that the quantized model sampled. The Fig. 7 result is "at least one of the 50 markers is in the top 20", with the entropy threshold at the 80th percentile.
3. Misconception 5 (wrong example). The old example for "accuracy unchanged" was RTN INT4 on Qwen3-4B. That model loses 11.3 accuracy points. The example is now RTN INT4 on Qwen3-30B (75.6 against 74.8, CTIR +43.0%). Question Q2 now states the accuracy of both models.
4. Question Q18 (wrong wording). 10,000 tokens is 5 times the base of 2,000 (+400%), not "4 times". At h = 0.01 the mean is infinite, but each run still ends with probability 1.
5. Exercise 2 (wrong code). The `rank` function ignored ties, so "Spearman(margin, flip) = -0.36" was not a Spearman value. The function now averages tied ranks. The real value is −0.513 (checked against scipy). Line (a) did not change. The reading now says "about 3.4 times" instead of "three times" for 0.326 / 0.0948.
6. Core ideas 8 and 12, questions Q12, Q13 and Q21 (missing confound). In 2606.25519 the calibrated methods use the model's own math reasoning traces (Sec 7.2), and RTN and HQQ use no data. Calibration text alone moves GPTQ INT3 on Qwen3-4B from 30k to 12k CoT tokens (Fig. 5). The averages in Table 2 are driven by single benchmarks (MBPP +957.4% for AWQ INT3). All methods use group size 128, RTN too.
7. Question Q21 (ambiguous key). The old key called GPTQ "the method that uses output error"; AWQ also searches its scale by output error. The old [mine] reading said GPTQ's "group grids" repair RTN; RTN has the same groups. The key is rewritten. The question now states GPTQ's calibration data and that the numbers are single greedy runs. ParoQuant INT3 (+24.5%) is added.
8. Misconception 8, core idea 19 and question Q17 (ambiguous). "The error can be removed by rewriting the entry" is true only for that entry's own rounding error. Entries that later tokens wrote after reading blurred entries keep what they inherited. This matters for Part C.
9. Core idea 17 and question Q14 (missing setting). The penalty and boost tests were run on quantized R1-Distill-Qwen-1.5B only (mean over six quantization configurations and five benchmarks), not on BF16. Q14 now says so. The low-KL boost arm (+10% at λ = −4.0) and the high-KL boost arm (+294%) are added.
10. Core idea 19 (minor). The key logit error is qᵀΔk/√d. The KIVI Table 2 caption does not state a bit width, so "2-bit" is now tagged [unverified].
11. Core idea 7 (minor). GPTQ's calibration data and grid are in Sec 5 "Setup", not Sec 3-4. Added: block inputs come from the already quantized model (this was listed as not read before).
12. Core idea 6 (minor). "Compensation never increases the error" holds for one compensation step; later rounding adds new error.
13. Tags (minor). Bare citations in Q2, Q5 and Q11 now carry a tag and a section. Q19's prediction for weights is tagged [mine]. The sources section and the list of unverified points are updated.

No item was deleted.

**Confirmed without change.** Numeric keys of Q1, Q2, Q4, Q7, Q9, Q10, Q11, Q17 and the three lengths in Q18. The derivations in core ideas 2, 4, 6, 13, 14, 15, 19 and 20. All CTIR and accuracy numbers from 2606.25519 Tables 1-2. All numbers from 2606.00206 (5.2K, 23.4K, 6.8K, 85.6%, 47.0%, 71.6%, 19 of 72, 139 of 265, ρ = 0.92, 12-23%, 41%, 395%, 31%). GPTQ Eq. 1-2, 1% damping, blocks of 128, 128 × 2,048 C4 segments, per-row asymmetric min-max grid. AWQ Eq. 1-5, grid of 20 values for α, clipping. HQQ: p = 0.7, β = 1, κ = 1.01, 20 iterations, the shrink operator, the zero-point update. KIVI Table 2 numbers, the quantizer formula, exact keys and values passed on during prefill.

**Still unverified.**

- 2611.12127 (stochastic rounding of bf16 state), 2606.15682 (entropy routing), 2504.04823 (abstract quote) and 2506.09501 (length noise of single runs): cited from the notes only. Core idea 20, Q19, Q20 and part of Q21 rest on them.
- 2606.00206: which model the ρ = 0.92 belongs to, how many positions, and whose trace gives the shared prefixes (most likely the quantized model's).
- KIVI Table 2: the bit width.
- ParoQuant: its method was not read.
- 2606.00206 says "about 32% longer" for 3-bit GPTQ; the rounded numbers give +31%.
- 2606.25519 Sec 3 says 100 sampled questions per benchmark; its Table 5 lists full dataset sizes.
- Items tagged [mine] are readings or toy models of this pack, not results: AWQ as "diagonal of A only", the INT4 against INT3 reading in Q21, the toy model in Q18, the prediction in Q19.
