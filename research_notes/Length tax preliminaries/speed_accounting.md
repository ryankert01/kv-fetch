# Tutor pack: speed accounting, from acceptance to wall-clock

Subskill id: `speed_accounting`. Written 2026-11-13 15:20 CST. Read-only work; no GPU job was run. A second pass checked and corrected this file on 2026-11-13 15:33 CST; see the Verification log at the end.

Scope: the speedup formula of speculative decoding as a function of acceptance, draft length and draft/verify cost ratio; how a length ratio enters end-to-end time and the break-even point where a lossy method stops paying; share-of-step-time (Amdahl) limits; batch-size regimes and fixed per-call kernel costs; reading a published timing table and computing a ceiling for a new design from it.

This pack follows `specdec_exact.md` in the same folder. That pack covers the accept/reject rule and the three "acceptance" numbers. This pack starts where a round has a cost in milliseconds.

Notation in this file:

| symbol | meaning |
|---|---|
| α | per-token acceptance probability in the i.i.d. model (Leviathan). SparseSpec uses α for accepted/drafted; this file writes that as ā. |
| γ, k, N | draft length (tokens drafted per round). Leviathan writes γ, SparseSpec k, Acceptance Is Not Speed N_draft. |
| τ | mean tokens emitted per round. τ = ℓ̄ + 1, where ℓ̄ is the mean number of accepted draft tokens. |
| c | time of one draft step / time of one plain target step. |
| v | time of one verification pass / time of one plain target step. |
| o | fixed overhead per round / time of one plain target step. |
| S | speedup per token over plain decoding. |
| g | per-token speed gain of a lossy method over the exact method it replaces. |
| λ | length ratio: mean completion length of the lossy method / mean completion length of the exact method. |
| f | share of step time that a method can touch. |
| r | factor by which the method speeds up that share. |

How to read the tags:

- **[checked]**: read in the arXiv page itself. The writer used a fetch tool that returns extracted answers. The verification pass then downloaded the raw HTML of all five papers and read every [checked] fact there. All formulas were also re-derived, and Leviathan's Table 1 was reproduced by code (Exercise 1a).
- **[checked, local file]**: I read it directly on this machine.
- **[from notes]**: taken from `research_notes/Inference speed question search/` or `reports/length-tax.html`; not re-read in this run.
- **[unverified]**: from memory only.
- **"my arithmetic"**: a number I computed from checked numbers. The paper does not state it.
- No tag: textbook fact or a derivation written out here.

---

## 1. Core ideas

1. **The unit is time per finished request.** Time = (number of tokens L) × (time per token). A method can change both factors. Every speedup in this file is a ratio of two such products for the same task.

2. **One round, general form.** A round runs γ draft steps and one verification pass and emits τ tokens on average. In units of one plain target step, the round costs γc + v + o. So S = τ / (γc + v + o). Acceptance Is Not Speed writes the same thing as S ≈ (ℓ̄ + 1)·C_AR / (N_draft·C_drafter + C_verify) ≈ (ℓ̄ + 1)/(1 + N_draft·c_rel), with c_rel = C_drafter/C_AR and C_verify ≈ C_AR "under memory-bound conditions" [checked: 2610.23818 Sec 2.1, Eq. 1]. Eq. 1 has no fixed per-round term o. The paper's appendix fits the measured time per round as c0 + c1 × (tokens committed in the round). On its first server c0 is 12 to 14 ms and c1 is 2.5 to 3.2 ms per token. On its second server c0 is 7.1 to 7.6 ms and c1 is at most 0.08 ms per token [checked: 2610.23818 App. C]. So a real round has a fixed part, and on some servers a part that grows with the tokens it commits.

3. **Leviathan's formula is the special case v = 1, o = 0, i.i.d. acceptance.** Then τ = (1 − α^(γ+1))/(1 − α) and S = (1 − α^(γ+1)) / ((1 − α)(γc + 1)) [checked: 2211.17192 Def 3.7, Theorem 3.8]. The proof is three lines: a round costs T(cγ + 1), it gives τ tokens, and plain decoding costs T per token.

4. **Two ceilings that hold for every γ.** Because τ ≤ γ + 1 and τ < 1/(1 − α): S < (γ + 1)/(γc + 1) < 1/c (for c < 1), and S < 1/(1 − α). So S < min(1/c, 1/(1 − α)). A perfect draft cannot beat 1/c, and a free draft cannot beat 1/(1 − α).

5. **When any gain exists.** At γ = 1 the formula gives S = (1 + α)/(1 + c). This is above 1 exactly when α > c. So a gain exists for some γ whenever α > c, and it is at least (1 + α)/(1 + c) [checked: 2211.17192 Corollary 3.9]. The best γ is found numerically; it grows with α and falls with c [checked: 2211.17192 Sec 3.5]. In Leviathan's own tests c was "always less than 0.05" [checked: 2211.17192, text near Def 3.7]; its appendix table lists c = 0.02, 0.04 and 0.11 for T5-small, T5-base and T5-large [checked: 2211.17192 App. A.3, Table 4]. The sentence and the table do not agree for T5-large (0.11 is not below 0.05). Both are in the paper.

6. **Speculation buys time with extra work.** The total number of arithmetic operations grows by the factor (1 − α)(γĉ + γ + 1)/(1 − α^(γ+1)), where ĉ is the draft/target ratio of operations per token [checked: 2211.17192 Def 3.10, Theorem 3.11]. Example: α = 0.8, γ = 5, c = ĉ = 0 gives 3.69× speed for 1.63× operations [checked: 2211.17192 Table 1; reproduced in Exercise 1a]. The time formula assumes γ + 1 target evaluations run in parallel "without increasing the walltime" [checked: 2211.17192 Sec 3.3]. The paper says the method "is not helpful" when no extra compute is available [checked: 2211.17192 Sec 6].

7. **The verify cost v depends on the regime.** Write f_c for the share of a plain step that grows in proportion to the number of tokens processed (compute-bound matrix products). The rest (reading weights once, reading the KV cache once, fixed costs) does not grow. A verification pass handles γ + 1 tokens, so v = (1 − f_c) + f_c(γ + 1) = 1 + f_c·γ. At batch 1 on a large GPU f_c is near 0 and v ≈ 1. If f_c = 1 the "speedup" is 1 over the operations factor of fact 6, which is below 1.

8. **A self-speculative draft has a floor on c.** If the draft only makes a share f of the step cheaper (for example attention with a KV budget s), then c = (1 − f) + f·s ≥ 1 − f. Together with fact 4: S < 1/c ≤ 1/(1 − f). Amdahl's law enters speculative decoding through c. This is why raising the KV budget from 5% to 50% raised acceptance from 80.4% to 96.7% but cut the speedup from 1.370× to 1.058× [from notes: S3R 2610.13202].

9. **SparseSpec's cost model.** T_base = T_GEMM(B) + T_Attn(M). One round: (kā + 1)·T_spec = k·T_draft + T_verify, with T_draft = T_GEMM(B) + T_Attn(M·s) and T_verify = T_GEMM((k+1)·B) + T_Attn(M). Here B is the number of concurrent requests, M the KV memory read by a full step, s the draft's KV budget, and ā = accepted/drafted. The KV bytes read per token fall by the factor (kā + 1)/(ks + 1) [checked: 2512.01278 Sec 3.2]. If GEMM time per pass does not grow with the tokens in the pass, GEMM time per token grows by (k + 1)/(kā + 1), because there are k + 1 passes for kā + 1 tokens. The paper's own approximation has the same factor: T_spec = (k + 1)/(kā + 1) · T_GEMM((2k + 1)/(k + 1) · B) + (ks + 1)/(kā + 1) · T_Attn(M) [checked: 2512.01278 Sec 3.2]. Here (2k + 1)/(k + 1) · B is the mean number of tokens per pass.

10. **Amdahl's law.** If a method speeds up a share f of the time by a factor r, the whole becomes 1/((1 − f) + f/r) times faster. The ceiling is 1/(1 − f) as r → ∞. Always compute f from a time breakdown at the operating point you care about (model, batch, context length).

11. **SparseSpec's Table 2 as a worked case.** Qwen3-8B, AIME, H100, milliseconds. vLLM: CPU 3.2, attention 17.1, GEMM 7.2, others 1.2, total 28.7. SparseSpec: CPU 0.5, attention 5.2, GEMM 8.9, others 1.4, total 16 [checked: 2512.01278 Table 2]. My arithmetic: f = 17.1/28.7 = 0.596, so attention-only work is capped at 2.47×. Attention fell 3.29× (the byte model with 6.16 accepted of 8 predicts 5.11×). GEMM rose 1.236× (the model of fact 9 predicts 9/7.16 = 1.257×). The total gain is 28.7/16 = 1.79×. The caption does not say what one "time" unit is (per step or per token) [checked]; only the ratios are used here. Three cautions. (1) The 6.16 of 8 is an average "across different models and datasets", and it does not count the bonus token [checked: 2512.01278 Sec 5.3, Fig. 12]. It is not the measured acceptance of the Table 2 run, so 5.11× and 1.257× are rough. (2) The paper reads its table the same way: attention 3.29× faster, GEMM up by 1.7 ms, "consistent with our estimation in § 3.2" [checked: 2512.01278 Sec 5.2]. (3) Sec 3.1 of the same paper says attention is "more than 77%" of the time (its Fig. 2). Table 2 gives 59.6%. These are two different measurements; use the table for arithmetic [checked: 2512.01278 Sec 3.1, Table 2].

12. **Length enters as a plain division.** Let the exact method take t_e per token and write L_e tokens; the lossy method takes t_l per token and writes L_l. With g = t_e/t_l and λ = L_l/L_e, the end-to-end speedup of lossy over exact is g/λ. The lossy method wins only if λ < g. Break-even is λ* = g. Acceptance Is Not Speed states this as S_relax ≈ (L̄_AR/L̄_relax)·S and "a method wins end to end only if its acceptance gain exceeds its length inflation" [checked: 2610.23818 Sec 2.2, Eq. 3].

13. **For a relaxed check, g is the ratio of tokens per round.** With the same draft and the same round cost, g = (ℓ̄_relax + 1)/(ℓ̄_strict + 1). The number of verifier rounds is R ≈ L/(ℓ̄ + 1), so R_relax/R_strict ≈ λ·(ℓ̄_strict + 1)/(ℓ̄_relax + 1) = λ/g [checked: 2610.23818 Sec 2.2, Eq. 4-5].

14. **The gain of any relaxed rule has a ceiling set by the exact baseline.** A round accepts at most N draft tokens, so g ≤ (N + 1)/(ℓ̄_strict + 1) [checked: 2610.23818 Sec 2.2]. Any λ above this value is a sure loss. With N = 6: GPT-OSS-20B on GSM8K has ℓ̄_strict = 2.57, ceiling 1.96; Qwen3-8B on GSM8K has 1.51, ceiling 2.79 [ℓ̄ values checked: 2610.23818 Table B1; ceilings my arithmetic]. For SparseSpec (k = 8, 6.16 accepted on average, see fact 11) the ceiling is 9/7.16 = 1.257. A better exact method leaves less room for a lossy one. The ceiling holds at a fixed draft length N. A lossy method that also drafts longer needs the draft cost c for its bound (see Q12).

15. **For an unchecked method, g is the per-token speedup, with its Amdahl ceiling.** Sparse attention without a check, or quantization, has no rounds. Its break-even is λ* = 1/((1 − f) + f/r) ≤ 1/(1 − f). Turned around: a length ratio λ can only be paid back if f > 1 − 1/λ. For λ = 1.61 (the CVSA case with reused selection [from notes: 2611.10634 App. P]) that needs f > 0.38 even with a free kernel.

16. **Published numbers: the gain and the tax.** Over 360 cells, relaxed verification accepts 1.34× as many tokens per round on average, and completions are 1.54× longer on average. But the median λ is 1.08, the median round ratio is 0.93 and its mean is 1.10. 114 of 360 cells need more rounds than exact checking [checked: 2610.23818 abstract, Sec 3.3, Sec 3.4]. Note 1.54/1.34 = 1.15 is not the mean round ratio (1.10): a mean of ratios is not a ratio of means.

17. **Which average is right depends on the question.** For the total time of a workload, use the ratio of sums: Σ L_lossy / Σ L_exact, divided by g. For "does a typical request get faster", use the median of the per-request ratio. With a heavy tail the two disagree (Exercise 3e). Lengths are also cut at the token budget, so a measured λ is, "if anything, understated" [checked: 2610.23818 Limitations, paragraph "Budget censoring"; Sec 3.3 also says cut-off runs make λ a lower bound].

18. **Rounds are not time.** In the same paper, wall-clock time at batch 1 is worse than the round count on the primary pairs: median ratio 1.01 against 0.96, and eleven cells save rounds but not time. The reason given: on their server a round's cost grows with the tokens it commits. On a second server rounds and time agree [checked: 2610.23818 Sec 3.4, App. C]. In the notation here, v and o are constants on one server and not on the other. So measure time per round on your own server before you read rounds as time.

19. **A kernel call has a fixed cost.** Model the time of one attention call as t(n) ≈ a + b·n, with n the tokens (bytes) read. Sparsity lowers only b·n, so the kernel speedup can never pass (a + b·n)/a. Published case (H100, 128K context, 32 query heads over 8 KV heads; "S× sparsity" means each query head attends to 1/S of the tokens): at batch 1 dense FlashInfer takes 0.19 ms and the sparse kernel is 10.25×, 11.05×, 11.14× faster at 100×, 200×, 500× sparsity. At batch 16 (3.08 ms dense) the same three are 20.09×, 37.32×, 76.77× [checked: 2605.24168 Table 1]. My arithmetic: the batch-1 times are 18.5, 17.2, 17.1 µs, a floor near 17 µs. The batch-16 fit gives a ≈ 13 µs (Exercise 2). The paper reports the plateau and gives no reason for it [checked: no explanation found in the page text]. Table 1 does not include the cost of choosing the tokens; with an indexer, the batch-1 GQA speedup at 100× falls to 1.65× [checked: 2605.24168 Table 2].

20. **Fewer bytes is not less time in the short-list regime.** FlashInfer's shared-prefix kernel at batch 16 with a 1,024-token prefix reads 6× fewer tokens and is 1.03× faster (45.17 against 46.52 µs). With an 8,192-token prefix it reads 13× fewer and is 2.56× faster (88.67 against 226.57 µs) [checked: 2501.01005 Table 5; token counts my arithmetic]. The paper itself says "these speedups do not always yield proportional end-to-end gains" [checked: 2501.01005, text near Table 5].

**The screening recipe (facts 10 to 20 put together).** To get a ceiling for a new design from a published table: (1) compute each row's share of the total; (2) mark the rows the design touches, this gives f; (3) take r from a kernel table at the same batch and list length, with the fixed cost a included; (4) g = 1/((1 − f) + f/r), measured against the best exact system, not against plain decoding; (5) divide by the expected λ; (6) if g/λ ≤ 1 even with r = ∞ and λ from the most friendly published number, stop.

---

## 2. Level ladder

- **Level 2.** Can state S = τ/(γc + 1) and Amdahl's law and compute them when all numbers are given. Knows that a longer completion costs time.
- **Level 3.** Computes the break-even length ratio and the round ratio from tokens per round. Computes a share f from a time table and its ceiling. With a hint, explains why higher acceptance can lower the speedup.
- **Level 4.** Derives the ceilings 1/c, 1/(1 − α), 1/(1 − f) and (N + 1)/(ℓ̄_strict + 1) without help. Reads a published timing or kernel table, pulls out the fixed cost, and gives a ceiling for a new design against the best exact baseline. Picks the right average (ratio of sums or median) for the question asked.
- **Level 5.** Finds accounting errors in papers and plans (mean of ratios, censored lengths, rounds read as time, a batch-1 number used at batch 64, a formula example that does not match its own formula). Builds the cost model for selective verification or KV-only repair before running anything, and says which measured outcome of Parts A, B, C would make each design pay.

---

## 3. Misconceptions

1. **"Higher acceptance means higher speed."** Acceptance is only the numerator. A draft that is accepted more often usually costs more (c rises), and S < 1/c. With a relaxed check the output also gets longer, so g must beat λ (facts 8, 12).

2. **"A kernel that is r times faster makes decoding r times faster."** Only the share f gets faster. With f = 0.6 and r = 10 the total is 2.17×, and no r passes 2.5×. In the SubSpec notes a fused kernel was 1.15-1.37× faster alone and about 2% slower end to end, because attention was under 1% of the forward pass [from notes: `reports/length-tax.html`, Question 4].

3. **"Reading S times fewer bytes makes attention S times faster."** Each call has a fixed cost. Also, the sparsity is counted per query head, and 4 query heads share one KV head there, so the bytes read can fall by less than S (my reading; the paper gives no byte count). At batch 1 in 2605.24168 Table 1, going from 100× to 500× sparsity saves 8% of the time; at 2× and 4× sparsity the sparse kernel is slower than dense (0.32×, 0.63×) [checked].

4. **"Verification always costs one target step."** That holds when the pass is memory-bound and compute is idle (v ≈ 1). When part of the step scales with tokens processed, v = 1 + f_c·γ, and the best γ shrinks fast (Exercise 1c). Leviathan states the assumption openly (fact 6).

5. **"Compare the lossy method with plain autoregressive decoding."** The honest baseline is the best exact method. A relaxed rule can gain at most (N + 1)/(ℓ̄_strict + 1) over it. For SparseSpec at k = 8 that is 1.257 (fact 14).

6. **"Mean gain 1.34 and mean length 1.54, so relaxed checking needs 15% more rounds everywhere."** Both are means over cells with a heavy tail. The median cell needs 7% fewer rounds, and 246 of 360 cells (360 − 114) do not need more rounds [checked: 2610.23818 Sec 3.4]. And 1.54/1.34 is not the mean of the per-cell ratio (fact 16).

7. **"Fewer rounds means less time."** A round's cost can depend on what it commits and on fixed overheads. The paper that counts rounds finds time worse than rounds on one of its two servers, and equal on the other (fact 18). The safe statement: fewer rounds may or may not mean less time; it depends on the server.

8. **"If the runs hit the token cap, the length ratio is still fine to use."** A capped run has an unknown true length, so λ is biased low, and the time "saved" by the cap is paid in lost answers. In the kv-fetch AIME log the full model itself hits the 16,384 cap in 41% of samples [from notes: `reports/length-tax.html`]. Report cap rates and a cost per solved problem next to λ.

---
## 4. Question bank

Each entry: level, type, question, answer key, likely wrong answer, follow-up. All numbers in the answer keys were run in `checks.py` in the scratch folder. Questions tied to the research question say so in the first line.

### speed_accounting-Q1
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "One round drafts 5 tokens. A draft step costs 0.2 of a plain target step. The verification pass costs 1.2 target steps. On average the round emits 3.0 tokens. What is the speedup over plain decoding?"
- **Answer key:** Round cost = 5 × 0.2 + 1.2 = 2.2 target steps. Plain decoding would need 3.0 steps for 3.0 tokens. Speedup = 3.0/2.2 = 1.36×.
- **Likely wrong answer:** "3×, because each round gives 3 tokens for one target pass."
- **Follow-up:** "How much time do the five draft steps take, in target steps? Is that time free?"

### speed_accounting-Q2
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "A lossy method is 1.5 times faster per token than the exact method. Its completions are 1.2 times longer. How much faster is it per finished request? At what length ratio does it stop paying?"
- **Answer key:** Time = tokens × time per token. Ratio = 1.5/1.2 = 1.25× faster. It stops paying at λ = 1.5: break-even length ratio equals the per-token gain.
- **Likely wrong answer:** "1.5× faster; the length only costs output tokens, not speed." Or "1.5 − 0.2 = 1.3×."
- **Follow-up:** "The exact method writes 1,000 tokens at 10 ms each. How many milliseconds does the lossy method need for its 1,200 tokens?" (1,200 × 6.67 = 8,000 ms against 10,000 ms.)

### speed_accounting-Q3
- **Level:** 2. **Type:** explain why (numeric).
- **Question:** "A draft step costs 0.25 of a target step. Suppose every draft token is accepted. What is the speedup with a draft length of 4, with a draft length of 20, and what is the limit for a very long draft? Why is there a limit?"
- **Answer key:** With α = 1 a round emits γ + 1 tokens for γc + 1 steps. γ = 4: 5/2 = 2.5×. γ = 20: 21/6 = 3.5×. Limit: 1/c = 4×. Every emitted token still had to be drafted once, and drafting one token costs c. The verification pass is the only cost that is shared, and a long draft only spreads that one pass thinner.
- **Likely wrong answer:** "With perfect acceptance the speedup grows without limit as the draft gets longer."
- **Follow-up:** "With a draft of 1,000 tokens, all accepted, how many target-step units did the drafting alone cost?" (250, for 1,001 tokens.)

### speed_accounting-Q4
- **Level:** 3. **Type:** apply to a new case (numeric).
- **Question:** "With the exact check a system emits 3.5 tokens per round. With a relaxed check it emits 4.9 tokens per round, at the same cost per round. The relaxed completions are 1.25 times longer. How many verifier rounds does the relaxed system need, relative to the exact one? Is it faster end to end?"
- **Answer key:** Gain g = 4.9/3.5 = 1.40. Round ratio = λ/g = 1.25/1.40 = 0.893, so 10.7% fewer rounds. End to end it is g/λ = 1.12× faster. Break-even would be λ = 1.40.
- **Likely wrong answer:** "It is 1.4× faster" (ignores λ), or "it is slower, because the output is longer" (ignores g).
- **Follow-up:** "The exact run writes 3,500 tokens. How many rounds is that? How many tokens does the relaxed run write, and how many rounds is that?" (1,000 rounds; 4,375 tokens; 893 rounds.)

### speed_accounting-Q5
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "In a decode step, attention takes 60% of the time. You replace the attention kernel with one that is 10 times faster. How much faster is the step? What is the most you could ever get from attention work alone?"
- **Answer key:** 1/((1 − 0.6) + 0.6/10) = 1/0.46 = 2.17×. Ceiling as the kernel time goes to zero: 1/0.4 = 2.5×.
- **Likely wrong answer:** "About 6× (0.6 × 10)" or "close to 10×."
- **Follow-up:** "After the change, how many time units does the non-attention part take, if the old step was 100 units?" (Still 40.)

### speed_accounting-Q6
- **Level:** 3. **Type:** find the flaw (numeric).
- **Question:** "A paper's abstract says: over 360 settings, a relaxed check accepts on average 1.34 times as many tokens per round, and completions are on average 1.54 times longer. A colleague concludes: 'So relaxed checking needs 1.54/1.34 = 1.15 times as many rounds. It is 15% slower in general.' What is wrong with this step?"
- **Answer key:** Two things. (1) 1.54/1.34 is a ratio of two means. The quantity that matters per setting is λ/g, and the mean of a ratio is not the ratio of the means. The paper's own mean round ratio is 1.10, not 1.15 [checked: 2610.23818 Sec 3.4]. (2) The length ratio has a heavy tail: its median is 1.08 and the median round ratio is 0.93, so the typical setting needs 7% fewer rounds. 114 of 360 settings need more. "In general" is wrong; "in about a third of the settings" is right.
- **Likely wrong answer:** Accepts the 15% and only argues about whether rounds equal time.
- **Follow-up:** "Two settings. In the first, gain 1.2 and length ratio 1.0. In the second, gain 1.5 and length ratio 2.1. Compute the mean gain, the mean length ratio, their quotient, and then the two round ratios one by one." (Means 1.35 and 1.55, quotient 1.15; round ratios 0.83 and 1.40: one setting wins, one loses.)

### speed_accounting-Q7
- **Level:** 3. **Type:** apply to a new case (numeric).
- **Question:** "A table gives a sparse attention kernel at batch size 1. The dense kernel takes 190 µs. The sparse kernel is 10.25 times faster when each query head attends to 1/100 of the tokens, and 11.14 times faster at 1/500. How long does the sparse kernel take in each case? What does the step from 1/100 to 1/500 buy, and what does that tell you about the kernel?"
- **Answer key:** 190/10.25 = 18.5 µs and 190/11.14 = 17.1 µs. Reading 5 times fewer tokens saves 8% of the time. So about 17 µs of the call does not depend on the bytes read: a fixed cost per call. The speedup at batch 1 cannot pass about 190/17 ≈ 11× however sparse the read is. (Source of the numbers: [checked: 2605.24168 Table 1].)
- **Likely wrong answer:** "At 1/500 it should be about 5 times faster than at 1/100; the table shows the kernel is badly tuned."
- **Follow-up:** "If the time were proportional to the tokens read, what would the time at 1/500 be, starting from 18.5 µs at 1/100? What is the gap to the measured 17.1 µs?" (3.7 µs. The measured 17.1 µs is 13.4 µs above that. Tutor note: 13.4 µs is a gap, not the fixed cost. A line t = a + b/S through the two points gives a fixed cost a ≈ 16.7 µs and b ≈ 185 µs.)

### speed_accounting-Q8
- **Level:** 3. **Type:** explain why (numeric).
- **Question:** "At batch size 1 on a large GPU, verifying 9 tokens in one pass costs about the same as one normal decode step. On another setup, 30% of the step time grows in proportion to the number of tokens in the pass. What does the 9-token verification pass cost there, in decode steps? If a round emits 7.16 tokens and each of the 8 draft steps costs 0.43, what is the speedup in each setup?"
- **Answer key:** At batch 1 the step is set by reading the weights and the cache once; 9 tokens ride on the same read, so v ≈ 1. In the second setup v = 0.7 + 0.3 × 9 = 3.4. Speedups: 7.16/(8 × 0.43 + 1) = 1.61× and 7.16/(3.44 + 3.4) = 1.05×.
- **Likely wrong answer:** "Verification is one forward pass, so it always costs one step: 1.61× in both."
- **Follow-up:** "In the second setup, how long does a pass over 2 tokens take compared with a pass over 1 token?" (1.3 steps.)

### speed_accounting-Q9
- **Level:** 4. **Type:** apply to a new case (numeric).
- **Question:** "A paper gives this time breakdown in milliseconds. Baseline: CPU 3.2, attention 17.1, GEMM 7.2, others 1.2. Their system: CPU 0.5, attention 5.2, GEMM 8.9, others 1.4. They say attention became 3.29 times faster. Why is the whole system only 1.79 times faster? And what is the most that any further attention work can add on top of their system?"
- **Answer key:** Totals 28.7 and 16.0, ratio 1.79. Attention was 17.1/28.7 = 59.6% of the baseline, so even free attention gives at most 28.7/11.6 = 2.47×. With attention at 5.2 and the other rows unchanged the total would be 1.71×. GEMM got slower (7.2 to 8.9, because there are more passes per emitted token) and CPU got faster (3.2 to 0.5), which together give 1.79×. On top of their system attention is now 5.2/16 = 32.5%, so further attention work gives at most 16/10.8 = 1.48×. (Numbers: [checked: 2512.01278 Table 2].)
- **Likely wrong answer:** "Something is lost in the kernel; a better attention kernel could still give about 3×."
- **Follow-up:** "Set the attention row of their system to zero. What is the new total, and what is 16 divided by it?" (10.8 ms; 1.48.)

### speed_accounting-Q10
- **Level:** 4. **Type:** connect two ideas (numeric).
- **Question:** "A self-speculative draft uses the full model with sparse attention. In a normal step, attention takes 60% of the time, and the draft reads 5% of the KV cache. Assume time follows bytes. What is the cost ratio c of a draft step? What does Leviathan's condition 'α > c' say here, and what is the highest speedup this design can ever reach? Which law is hiding inside c?"
- **Answer key:** c = (1 − f) + f·s = 0.40 + 0.60 × 0.05 = 0.43. A gain needs per-token acceptance above 0.43; with lower acceptance no draft length helps. Ceiling: S < 1/c = 2.33×, and c can never fall below 1 − f = 0.40, so S < 2.5× for any KV budget. That is Amdahl's law: the draft only shrinks the attention share. Example: k = 8 and 7.16 tokens per round give 7.16/(8 × 0.43 + 1) = 1.61×.
- **Likely wrong answer:** "c = 0.05, because the draft reads 5% of the cache. The ceiling is 20×."
- **Follow-up:** "In a draft step, how long do the MLP and projection layers take compared with a normal step?" (The same time: 0.40 of a normal step. Only the attention part got cheaper.)

### speed_accounting-Q11
- **Level:** 4. **Type:** predict (numeric).
- **Question:** "A model has 36 layers, 8 KV heads of dimension 128, and stores keys and values in 16-bit floats. Its weights take 16 GB. Assume a decode step reads all weights once and the whole KV cache of every request once, and that time follows bytes. Case 1: batch 1, context 4,096 tokens. Case 2: batch 128, context 4,096 tokens each. In each case, what share of the bytes is KV cache, and what is the ceiling for a method that only makes attention cheaper?"
- **Answer key:** KV per token = 36 layers × 2 (K and V) × 8 × 128 × 2 bytes = 147,456 bytes. Case 1: 4,096 × 147 kB = 0.60 GB against 16 GB of weights: share 3.6%, ceiling 1/(1 − 0.036) = 1.04×. Case 2: 128 × 0.60 = 77.3 GB against the same 16 GB: share 83%, ceiling 5.8×. The same sparse method is nearly useless at batch 1 and strong at large batch. (A level-5 answer adds: at batch 128 the matrix products are no longer a pure weight read, so the real share is lower than the byte share. For comparison, SparseSpec's own time table has attention at 60% of the baseline time [checked: 2512.01278 Table 2].)
- **Likely wrong answer:** "Attention is the bottleneck of long-context decoding, so the gain is large in both cases."
- **Follow-up:** "At batch 1, how many gigabytes does one step read in total, and how many of them belong to the cache?" (16.6 GB in total; 0.60 GB is cache.)
- **Tutor note:** In case 2 the cache (77 GB) plus the weights (16 GB) is 93 GB. That does not fit on one 80 GB card. The case is a byte-count exercise. Give credit if the learner points this out.

### speed_accounting-Q12
- **Level:** 4. **Type:** find the flaw (numeric). **Tied to Part C.**
- **Question:** "SparseSpec drafts 8 tokens with sparse attention, then runs one full-attention pass. The pass rewrites the KV entries and rejects bad tokens; on average 6.16 of 8 draft tokens survive, plus one bonus token. A colleague proposes: 'Keep the full pass for the KV rewrite, but never reject a token. Verification is the expensive part, so this will be about twice as fast.' Using only these numbers and the same draft length of 8, what is the most this change can gain, and what must the length ratio stay below?"
- **Answer key:** The passes are the same in both designs: 8 sparse steps and 1 full pass per round. Only the tokens kept per round change: 9 instead of 7.16. Gain ≤ 9/7.16 = 1.257. So at k = 8 the design pays only if λ < 1.257. The claim "verification is the expensive part" is also wrong: the full pass is still there, and it is only about 1 of the 4.44 time units of a round (with c = 0.43 from Q10). (Level-5 addition, needs c: with no rejection the draft can be longer. The cost per token is then (kc + 1)/(k + 1), which falls toward c. With c = 0.43, SparseSpec costs 4.44/7.16 = 0.620 per token. So the gain is 1.39 at k = 32 and at most 0.620/0.43 = 1.44 for any k. Still far from 2×.) This is the largest speed prize that "the harm travels through the KV, not the token" can buy at k = 8. If the harm travels through the token, the rejections are needed and the prize is zero.
- **Likely wrong answer:** "About 2×, because the verification pass is removed from the critical path."
- **Follow-up:** "List the GPU passes of one round in SparseSpec and in the new design. Which pass disappeared?" (None.)

### speed_accounting-Q13
- **Level:** 4. **Type:** apply to a new case (numeric). **Tied to Part B.**
- **Question:** "A relaxed check gives 1.30 times as many tokens per round as the exact check. Assume the per-token cost model: each wrong token that gets through adds m extra tokens to the completion, and nothing else. Wrong tokens are a fraction ρ of all emitted tokens. First: write λ as a function of ρ and m. Then: for ρ = 0.07 and for ρ = 0.24, how large may m be before the relaxed check stops paying? Last: what changes in this calculation if the damage is lasting instead?"
- **Answer key:** L = L_e + ρ·m·L, so λ = 1/(1 − ρm). Break-even λ = g gives ρm = 1 − 1/g = 0.231. For ρ = 0.07: m ≤ 3.3 tokens. For ρ = 0.24: m ≤ 0.96 tokens. (The 7% to 24% range is the published share of relaxed-only tokens among committed draft tokens in the traced runs [checked: 2610.23818 Sec 3.7, Table B2]. It is a share of committed draft tokens, not of all emitted tokens, so it only gives the size of ρ.) Under lasting damage there is no fixed m: a single wrong token can send the run to the cap, λ is set by the probability of a derailed run times its length, and lowering ρ helps only in proportion. The two models can have the same mean λ and need different repairs (Exercise 3e).
- **Likely wrong answer:** "λ = 1 + ρm, so m ≤ 0.30/0.07 = 4.3." (Forgets that the extra tokens are also emitted by the relaxed method and contain wrong tokens too; and uses g − 1 instead of 1 − 1/g.)
- **Follow-up:** "With ρ = 0.07 and m = 4.3, compute λ from L = L_e + ρ·m·L. Is it above or below 1.30?" (1/(1 − 0.301) = 1.43, above.)

### speed_accounting-Q14
- **Level:** 4. **Type:** connect two ideas (numeric).
- **Question:** "Two exact systems draft 6 and 8 tokens per round. System X accepts 2.57 of 6 on average. System Y accepts 6.16 of 8. For each, what is the largest gain any relaxed acceptance rule can give at the same draft length? A published temperature control lengthened one model's completions by 1.27 times. Against which system could a rule with that length ratio still win?"
- **Answer key:** Ceiling = (N + 1)/(ℓ̄_strict + 1). X: 7/3.57 = 1.96. Y: 9/7.16 = 1.257. A length ratio of 1.27 can be paid back on X (if the rule reaches a gain above 1.27 of the possible 1.96). On Y it cannot: even accepting everything gives 1.257 < 1.27. The better the exact baseline, the less a lossy rule can win and the smaller the length tax it can afford. (The 1.27 is GPT-OSS-20B on GSM8K when the sampling temperature goes from 1.0 to 1.2 with exact checking [checked: 2610.23818 Sec 3.7, App. C].)
- **Likely wrong answer:** "Y drafts more tokens, so relaxing has more to gain there."
- **Follow-up:** "In system Y, how many of the 9 possible tokens per round are already emitted with the exact check?" (7.16 of 9.)

### speed_accounting-Q15
- **Level:** 4. **Type:** connect two ideas. **Tied to Part A.**
- **Question:** "Part A asks whether the length tax depends only on the 'dose' (how often tokens change and how unlikely the changed tokens are), or also on the method. Suppose the answer is 'only the dose'. Explain how you would then find the break-even setting of a new lossy method without generating a single long completion. Then say what breaks in this plan if sparse attention is worse than random token swaps at the same dose."
- **Answer key:** If λ is one function of dose, measure that curve once (any cheap perturbation). For a new method, at each setting: (1) measure the dose by teacher forcing on stored exact traces, no generation; (2) compute the per-token gain g from a time breakdown and kernel tables (f, r, fixed costs); (3) the method pays where g(setting) > λ(dose(setting)); the break-even setting is where the two curves cross. If sparse attention is worse at equal dose, λ depends on the method (on what context was hidden), the teacher-forced dose no longer predicts λ, and each method needs its own generation runs at each setting. The offline screen is lost, but the design target changes to "what the draft may not hide".
- **Likely wrong answer:** "Measure acceptance offline and plug it into the speedup formula." (That gives g only. It says nothing about λ.)
- **Follow-up:** "Your offline number says the method is 1.4 times faster per token. Which second number do you still need before you can say it is faster per request, and can teacher forcing give it?" (The length ratio λ. Teacher forcing cannot give it directly. It can only give it through the dose curve, and only if Part A says λ depends on the dose alone.)

### speed_accounting-Q16
- **Level:** 5. **Type:** find the flaw (edge case, numeric).
- **Question:** "An experiment has a cap of 16,384 tokens. The exact arm solves 48.3% of problems with a mean length of 10,000 tokens. The lossy arm is 1.5 times faster per token, solves 33.3%, and has a mean length of 11,000 tokens. In both arms some runs stop at the cap. The report says: 'λ = 1.10, gain 1.5, so the lossy arm is 1.36 times faster end to end.' Give two reasons this claim is wrong, and compute one number that is fairer."
- **Answer key:** (1) Censoring: runs that hit the cap have an unknown true length, so both means are too low. If the lossy arm hits the cap more often, its mean is cut more and λ = 1.10 is an underestimate. (2) The arms do not deliver the same thing: accuracy fell by 15 points. Some of that loss may be answers cut off at the cap. If so, the cap turned extra length into lost answers and hid the time. A fairer number is cost per solved problem: exact 10,000/0.483 = 20,700 token-times; lossy 11,000/(1.5 × 0.333) = 22,000. Ratio 0.94: the lossy arm is about 6% slower per solved problem. Also report the cap rate per arm and, if possible, a longer cap or a survival-style median.
- **Likely wrong answer:** "The claim is fine as long as both arms use the same cap."
- **Follow-up:** "A run that would need 40,000 tokens is cut at 16,384. What length enters the mean? What does it do to λ if such runs are more common in the lossy arm?" (16,384 enters the mean. The lossy mean is cut more than the exact mean, so the measured λ is too low.)

### speed_accounting-Q17
- **Level:** 5. **Type:** apply to a new case (adversarial, numeric). **Tied to the design goal "which tokens need verifying".**
- **Question:** "You plan selective verification on top of a sparse self-draft. A sparse step costs 0.43 of a full step. A round is 8 sparse steps. In a verified round one full pass follows, and 7.16 tokens are emitted. In an unverified round no full pass runs and all 8 drafted tokens are kept. You verify a fraction φ of the rounds. Give the time per token as a function of φ. What is the gain over the exact system (φ = 1) at φ = 0.25 and at φ = 0? What does this say about the length ratio your selector may cause?"
- **Answer key:** Time per token = (8 × 0.43 + φ)/(φ × 7.16 + (1 − φ) × 8). φ = 1: 4.44/7.16 = 0.620. φ = 0.25: 3.69/7.79 = 0.474, gain 1.31. φ = 0: 0.43, gain 1.44. So even with no verification at all the gain is capped at 1.44, because the draft steps themselves (3.44 of the 4.44 units per round) stay. The selector must keep λ below 1.31 at φ = 0.25 to break even. Skipping checks saves little: at most 1.44×. So the room for extra length is small. A full answer also notes that unverified rounds leave approximate KV entries behind, so φ changes Part C's channel too, and that the model assumes v = 1.
- **Likely wrong answer:** "Verifying a quarter of the rounds cuts verification cost by 4, so the system is several times faster."
- **Follow-up:** "In one exact round, how many of the 4.44 time units are draft steps? Can skipping the check remove them?" (3.44 of 4.44. No.)

### speed_accounting-Q18
- **Level:** 5. **Type:** find the flaw (adversarial).
- **Question:** "A paper measures a relaxed acceptance rule at batch size 1 and reports: 'in the median setting it needs 7% fewer verifier rounds than exact checking.' You want to turn it on in a vLLM server at batch 64. Give two separate reasons why the server may get slower, and one reason why it may gain more than 7%."
- **Answer key:** Slower: (1) Rounds are not time. The time of one round can grow with the tokens it commits. The same paper found this on one of its two servers: median time ratio 1.01 against round ratio 0.96. On its second server rounds and time agree [checked: 2610.23818 Sec 3.4, App. C]. Which case holds on your server at batch 64 is not known, so measure it. (2) A batch server's throughput follows the sum of all tokens, so the mean matters, not the median. The length ratio has a heavy tail (mean round ratio 1.10, and 114 of 360 settings need more rounds). Long runs also hold a batch slot and KV memory longer and hit the cap more. Possible gain: the 7% is a median over many rules, drafters and tasks, and your own setting may be on the good side. The median length ratio runs from 0.94 for standalone drafters to 1.23 for EAGLE heads, and from 0.97 for one rule (spec_casc_tok) to 1.39 for another (cactus) [checked: 2610.23818 Sec 3.3]. So a level-5 answer asks which model pair, rule and sampler the server uses. Two answers that sound right but are wrong: (a) "a fixed overhead per round makes fewer rounds save more". A fixed cost per round gives exactly the round ratio, not more. (b) "the verify pass costs more at batch 64, so relaxed gets slower". Both arms pay the same pass, so the ratio between them does not change.
- **Likely wrong answer:** "7% fewer rounds means about 7% more throughput; batch size does not change the count of rounds."
- **Follow-up:** "Ten requests: nine get 10% shorter in rounds, one gets 3 times longer. All were equal before. What happens to the total number of rounds?" (9 × 0.9 + 3 = 11.1 against 10: 11% more.)

### speed_accounting-Q19
- **Level:** 5. **Type:** find the flaw (edge case, numeric).
- **Question:** "A paper models the KV bytes read per emitted token as falling by (kα + 1)/(ks + 1), where k is the draft length, α is accepted/drafted and s is the draft's KV budget. It gives the example 'with k = 16, α = 0.75, s = 0.05 the attention latency is reduced by 6.78×'. Its measured table, at k = 8 with 6.16 of 8 accepted and s = 0.05, shows attention going from 17.1 ms to 5.2 ms. Check the example against the formula. Then compare formula and measurement at k = 8, and give three possible reasons for the gap." (Any three of the four reasons in the key count.)
- **Answer key:** Example: (16 × 0.75 + 1)/(16 × 0.05 + 1) = 13/1.8 = 7.22, not 6.78. The value 6.78 comes out with α = 0.70 (12.2/1.8), so the example and the formula do not match. The paper prints exactly k = 16, α = 0.75, s = 0.05 and 6.78× [checked: 2512.01278 Sec 3.2, raw page]. The mismatch is in the paper. At k = 8: (6.16 + 1)/(0.4 + 1) = 5.11 predicted, 17.1/5.2 = 3.29 measured. Reasons: the formula counts bytes, and time is not bytes: (1) fixed cost per call and per short list; (2) a kernel tuned for one of the two shapes (sparse draft, full verify with k + 1 query tokens) runs the other at a lower share of peak bandwidth (near 85% against under 50%) [checked: 2512.01278 Sec 4.2]; (3) the sparse path has extra work (top-K choice, index handling) that is not KV reading; (4) the 6.16 of 8 is an average over models and datasets [checked: 2512.01278 Sec 5.3], not the acceptance of the Table 2 run. If that run accepted fewer tokens, the prediction is lower. But acceptance alone would have to be 3.6 of 8 to give 3.29×, so it cannot be the whole gap. The honest reading: the 5.1× is a ceiling, and the paper itself calls the result "consistent with our estimation" [checked: 2512.01278 Sec 5.2]. It is not proof of one specific cause. Not a valid reason: "α is a mean over rounds". The round cost is fixed and the tokens add up linearly, so a mean is the right input.
- **Likely wrong answer:** "The formula is fine and the gap must be the kernel's fixed launch cost."
- **Follow-up:** "Test your one cause. Assume 36 layers, one attention call per layer per pass, and 17 µs fixed per call. How many milliseconds per emitted token are fixed in the baseline (1 pass per token) and in the speculative system (9 passes per 7.16 tokens)? Add that to the byte model. How much of the gap between 3.35 ms and 5.2 ms does it close?" (0.61 ms and 0.77 ms. Byte part: (17.1 − 0.61)/5.11 = 3.22 ms; plus 0.77 gives 3.99 ms. That closes about 0.65 of the 1.86 ms gap, roughly a third. The 36 layers and the 17 µs are assumptions taken from other sources, so this is a size check only.)

---

## 5. Exercises

All three were run in the scratch folder with Python 3.11.5 and numpy 1.24.3 on CPU. The outputs below are the real outputs of these scripts. Each takes under 30 minutes; parts (d) and (e) of Exercise 1 can be done with pen and paper.

### Exercise 1. The speedup formula, its ceilings, and the break-even length ratio

**Task.** (a) Code the tokens-per-round, speedup and operations formulas (facts 3 and 6) and reproduce six rows of Leviathan's Table 1 (c = ĉ = 0). (b) For six (α, c) pairs find the best draft length by search, and print the two ceilings 1/c and 1/(1 − α) and the γ = 1 value (1 + α)/(1 + c). (c) Let the verify pass cost v = 1 + f_c·γ for f_c = 0, 0.1, 0.3, 1 with α = 0.8 and c = 0.05. (d) Compute the round ratio and the end-to-end ratio for three (tokens per round, λ) cases. (e) Compute the gain ceiling (N + 1)/(ℓ̄_strict + 1) for four exact systems.

**Expected result.** (a) matches the paper: for example α = 0.8, γ = 5 gives 1.63× operations and 3.69× speed [checked: 2211.17192 Table 1]. (b) The best γ falls from 8 to 3 when c goes from 0.05 to 0.30 at α = 0.8; S always stays under both ceilings; with α = 0.40 < c = 0.43 the best is γ = 1 and S = 0.979 < 1. (c) The best speedup falls 3.09, 2.11, 1.44, 0.88 and the best γ falls 8, 5, 3, 1. (d) Gain 1.40 with λ = 1.25 wins (1.12×); with λ = 1.54 it loses (0.91×). (e) Ceilings 1.96, 2.79, 4.12 and 1.257.

**How the tutor checks.** The learner shows the six rows of (a) next to the paper's values. Then, without code: "In (b), which of the two ceilings is the tight one for α = 0.9, c = 0.43, and what does that tell you about where to spend effort: a better draft or a cheaper draft?" (1/c = 2.33 is far below 1/(1 − α) = 10: the draft is too expensive, not too wrong.) And: "In (e), why is the last line the smallest?" (The exact system already emits 7.16 of 9 possible tokens.)

```python
# Exercise 1: speedup formula, ceilings, best draft length, and the length-tax break-even.
# alpha = per-token acceptance probability (i.i.d. model), gamma = draft length,
# c = time of one draft step / time of one target step, v = time of the verify pass / target step.
import numpy as np

def tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)

def speedup(alpha, gamma, c, v=1.0):
    return tokens(alpha, gamma) / (gamma * c + v)

def ops_factor(alpha, gamma, c_hat):
    return (1 - alpha) * (gamma * c_hat + gamma + 1) / (1 - alpha ** (gamma + 1))

print("(a) Leviathan Table 1 rows, c = c_hat = 0:  alpha gamma  ops  speed")
for a, g in [(0.6, 2), (0.7, 3), (0.8, 2), (0.8, 5), (0.9, 2), (0.9, 10)]:
    print(f"    {a:.1f} {g:3d}   {ops_factor(a, g, 0):.2f}x  {speedup(a, g, 0):.2f}x")

print("(b) best gamma and the two ceilings 1/c and 1/(1-alpha)")
gammas = np.arange(1, 201)
for a, c in [(0.8, 0.05), (0.8, 0.30), (0.9, 0.05), (0.9, 0.43), (0.95, 0.43), (0.40, 0.43)]:
    s = speedup(a, gammas, c)
    g = int(gammas[np.argmax(s)])
    print(f"    alpha={a:.2f} c={c:.2f}: best gamma={g:3d}  S={s.max():.3f}"
          f"   1/c={1/c:.2f}  1/(1-alpha)={1/(1-a):.2f}  gamma=1: {(1+a)/(1+c):.3f}")

print("(c) verify pass that costs more than one step: v = 1 + fc*gamma")
for fc in [0.0, 0.1, 0.3, 1.0]:
    s = speedup(0.8, gammas, 0.05, v=1 + fc * gammas)
    print(f"    fc={fc:.1f}: best gamma={int(gammas[np.argmax(s)]):2d}  S={s.max():.3f}")

print("(d) length tax: end-to-end ratio = gain / lambda, break-even lambda = gain")
for ts, tr, lam in [(3.5, 4.9, 1.25), (3.5, 4.9, 1.54), (7.16, 9.0, 1.20)]:
    gain = tr / ts
    print(f"    tokens/round {ts} -> {tr}: gain={gain:.3f}  lambda={lam}:"
          f" rounds ratio={lam/gain:.3f}  end-to-end={gain/lam:.3f}x")

print("(e) ceiling of the gain of any relaxed rule: (N+1)/(l_strict+1)")
for name, N, l in [("GPT-OSS-20B GSM8K", 6, 2.57), ("Qwen3-8B GSM8K", 6, 1.51),
                   ("Qwen3-8B LongBench-v2", 6, 0.70), ("SparseSpec k=8", 8, 6.16)]:
    print(f"    {name:22s} N={N} l_strict={l}: max gain = {(N+1)/(l+1):.3f}")
```

Output:

```
(a) Leviathan Table 1 rows, c = c_hat = 0:  alpha gamma  ops  speed
    0.6   2   1.53x  1.96x
    0.7   3   1.58x  2.53x
    0.8   2   1.23x  2.44x
    0.8   5   1.63x  3.69x
    0.9   2   1.11x  2.71x
    0.9  10   1.60x  6.86x
(b) best gamma and the two ceilings 1/c and 1/(1-alpha)
    alpha=0.80 c=0.05: best gamma=  8  S=3.092   1/c=20.00  1/(1-alpha)=5.00  gamma=1: 1.714
    alpha=0.80 c=0.30: best gamma=  3  S=1.554   1/c=3.33  1/(1-alpha)=5.00  gamma=1: 1.385
    alpha=0.90 c=0.05: best gamma= 13  S=4.674   1/c=20.00  1/(1-alpha)=10.00  gamma=1: 1.810
    alpha=0.90 c=0.43: best gamma=  4  S=1.506   1/c=2.33  1/(1-alpha)=10.00  gamma=1: 1.329
    alpha=0.95 c=0.43: best gamma=  6  S=1.685   1/c=2.33  1/(1-alpha)=20.00  gamma=1: 1.364
    alpha=0.40 c=0.43: best gamma=  1  S=0.979   1/c=2.33  1/(1-alpha)=1.67  gamma=1: 0.979
(c) verify pass that costs more than one step: v = 1 + fc*gamma
    fc=0.0: best gamma= 8  S=3.092
    fc=0.1: best gamma= 5  S=2.108
    fc=0.3: best gamma= 3  S=1.440
    fc=1.0: best gamma= 1  S=0.878
(d) length tax: end-to-end ratio = gain / lambda, break-even lambda = gain
    tokens/round 3.5 -> 4.9: gain=1.400  lambda=1.25: rounds ratio=0.893  end-to-end=1.120x
    tokens/round 3.5 -> 4.9: gain=1.400  lambda=1.54: rounds ratio=1.100  end-to-end=0.909x
    tokens/round 7.16 -> 9.0: gain=1.257  lambda=1.2: rounds ratio=0.955  end-to-end=1.047x
(e) ceiling of the gain of any relaxed rule: (N+1)/(l_strict+1)
    GPT-OSS-20B GSM8K      N=6 l_strict=2.57: max gain = 1.961
    Qwen3-8B GSM8K         N=6 l_strict=1.51: max gain = 2.789
    Qwen3-8B LongBench-v2  N=6 l_strict=0.7: max gain = 4.118
    SparseSpec k=8         N=8 l_strict=6.16: max gain = 1.257
```

### Exercise 2. Find the fixed cost per call in two published kernel tables

**Task.** (a) Take Table 1 of arXiv 2605.24168 (speedup of a sparse decode kernel over dense FlashInfer; batch 1 and batch 16; sparsity 2× to 500×). Turn the speedups into times. Print time × S, which would be constant if time followed the tokens read. Fit t = a + b/S with relative errors. Report the floor a, the ceiling of the speedup, and the break-even sparsity. (b) Take Table 5 of FlashInfer (arXiv 2501.01005; suffix 128; prefix 1,024 / 8,192 / 32,768; batch 16 and 64). Count the tokens each format reads. Compare "reads fall by" with "time falls by". Fit t = a + b × tokens.

**Expected result.** (a) Batch 1: times 594, 302, 131, 74, 34, 18.5, 17.2, 17.1 µs. The last three are flat: a floor near 17 µs. The two-parameter fit is poor at batch 1 (worst error 23%, a = 11.5 µs), so read the floor from the plateau, which gives a ceiling of about 190/17 ≈ 11×. Batch 16: the fit is good (worst error 1%) with a = 12.7 µs; the floor is still far away, and 100× to 500× buys 3.82× there against 8% at batch 1. The floor is about the same at both batch sizes while b grows 12 times: the fixed cost is per call, not per request. The fitted b is 4.5 to 6 times the dense time, so break-even is at S ≈ 4.5 to 6.4. Do not read this as "the sparse kernel is 4.5 to 6 times less efficient per byte". The sparsity is per query head, and 4 query heads share one KV head (32 over 8), so the kernel may read up to 4/S of the KV bytes. A factor of about 4 can come from this alone (my reading; the paper gives no byte count). The script's label "cost per token read against dense" means b/dense, nothing more. (The paper's caption says break-even "at 10×" [checked]; its own 4× column is below 1 and its 10× column above 1.) (b) Batch 16: reads fall 6.0×, 13.0×, 15.1× and time falls 1.03×, 2.56×, 4.35×. The single format at batch 16 fits with a ≈ 15 µs and 1.7 ns per token. The composable format has a larger intercept (29 µs) and a higher cost per token read (5.5 ns), which fits two calls plus a merge, and a prefix that is read once but used by every query (my reading; the paper does not give this split). The batch-64 single fit gives a negative intercept (−6.6 µs): three points with a cost per token that rises with size cannot pin down a floor. Do not over-read it.

**How the tutor checks.** The learner gives the batch-1 floor and the batch-16 floor and says why the fit is trusted at 16 and not at 1. Then: "Your draft reads 512 tokens per request at batch 1. Is it worth cutting that to 128?" (No. Both are deep in the flat part; the time is the fixed cost.) And: "Which number from this exercise would you re-measure on your own 48 GB card before any design decision?" (The floor a: it is hardware and kernel specific; the notes say nobody has measured it for this case.)

```python
# Exercise 2: find the fixed cost per kernel call in two published timing tables.
import numpy as np

def fit(x, t):
    """Least squares for t = a + b*x with relative errors (each row divided by t)."""
    A = np.stack([np.ones_like(x), x], axis=1) / t[:, None]
    (a, b), *_ = np.linalg.lstsq(A, np.ones_like(t), rcond=None)
    return a, b

# ---- (a) arXiv 2605.24168 Table 1: speedup of a sparse decode kernel over dense FlashInfer ----
S = np.array([2, 4, 10, 20, 50, 100, 200, 500], dtype=float)   # S x sparsity = read 1/S of the tokens
dense_us = {1: 190.0, 16: 3080.0}                               # dense FlashInfer time, microseconds
speedup = {1:  np.array([0.32, 0.63, 1.45, 2.58, 5.57, 10.25, 11.05, 11.14]),
           16: np.array([0.45, 0.89, 2.21, 4.35, 10.54, 20.09, 37.32, 76.77])}
print("(a) sparse kernel time in microseconds = dense time / speedup")
for B in (1, 16):
    t = dense_us[B] / speedup[B]
    print(f"  B={B:2d} time:", " ".join(f"{v:7.1f}" for v in t))
    print(f"  B={B:2d} time x S:", " ".join(f"{v:7.0f}" for v in t * S))
    a, b = fit(1 / S, t)
    pred = a + b / S
    print(f"  B={B:2d} fit t = a + b/S: a = {a:.1f} us, b = {b:.0f} us,"
          f" worst relative error = {np.max(np.abs(pred - t) / t):.2f}")
    print(f"       ceiling of the speedup as S grows: dense/a = {dense_us[B] / a:.1f}x;"
          f" break-even S = b/(dense - a) = {b / (dense_us[B] - a):.1f};"
          f" cost per token read against dense: b/dense = {b / dense_us[B]:.1f}x")
t1 = dense_us[1] / speedup[1]
print(f"  B=1: going from 100x to 500x reads 5 times fewer tokens and saves"
      f" {100 * (1 - t1[-1] / t1[-3]):.0f}% of the time")
t16 = dense_us[16] / speedup[16]
print(f"  B=16: the same step saves {100 * (1 - t16[-1] / t16[-3]):.0f}% ({t16[-3] / t16[-1]:.2f}x)")

# ---- (b) arXiv 2501.01005 (FlashInfer) Table 5: shared-prefix kernels, suffix 128, microseconds ----
prefix = np.array([1024, 8192, 32768], dtype=float)
suffix = 128.0
table = {16: (np.array([45.17, 88.67, 217.42]), np.array([46.52, 226.57, 945.67])),
         64: (np.array([87.86, 125.76, 254.54]), np.array([130.49, 931.75, 4090.0]))}
print("(b) tokens read: single = B*(prefix+suffix), composable = prefix + B*suffix")
for B, (comp, single) in table.items():
    n_single = B * (prefix + suffix)
    n_comp = prefix + B * suffix
    print(f"  B={B}: reads fall by", " ".join(f"{v:5.1f}x" for v in n_single / n_comp),
          "| time falls by", " ".join(f"{v:5.2f}x" for v in single / comp))
    a, b = fit(n_single, single)
    print(f"       single format fit t = a + b*tokens: a = {a:.1f} us, b = {1000 * b:.2f} ns per token")
    a, b = fit(n_comp, comp)
    print(f"       composable fit    t = a + b*tokens: a = {a:.1f} us, b = {1000 * b:.2f} ns per token")
```

Output:

```
(a) sparse kernel time in microseconds = dense time / speedup
  B= 1 time:   593.8   301.6   131.0    73.6    34.1    18.5    17.2    17.1
  B= 1 time x S:    1188    1206    1310    1473    1706    1854    3439    8528
  B= 1 fit t = a + b/S: a = 11.5 us, b = 1142 us, worst relative error = 0.23
       ceiling of the speedup as S grows: dense/a = 16.6x; break-even S = b/(dense - a) = 6.4; cost per token read against dense: b/dense = 6.0x
  B=16 time:  6844.4  3460.7  1393.7   708.0   292.2   153.3    82.5    40.1
  B=16 time x S:   13689   13843   13937   14161   14611   15331   16506   20060
  B=16 fit t = a + b/S: a = 12.7 us, b = 13859 us, worst relative error = 0.01
       ceiling of the speedup as S grows: dense/a = 242.4x; break-even S = b/(dense - a) = 4.5; cost per token read against dense: b/dense = 4.5x
  B=1: going from 100x to 500x reads 5 times fewer tokens and saves 8% of the time
  B=16: the same step saves 74% (3.82x)
(b) tokens read: single = B*(prefix+suffix), composable = prefix + B*suffix
  B=16: reads fall by   6.0x  13.0x  15.1x | time falls by  1.03x  2.56x  4.35x
       single format fit t = a + b*tokens: a = 15.2 us, b = 1.68 ns per token
       composable fit    t = a + b*tokens: a = 28.8 us, b = 5.54 ns per token
  B=64: reads fall by   8.0x  32.5x  51.4x | time falls by  1.49x  7.41x 16.07x
       single format fit t = a + b*tokens: a = -6.6 us, b = 1.85 ns per token
       composable fit    t = a + b*tokens: a = 39.5 us, b = 5.25 ns per token
```

### Exercise 3. From a time breakdown to a ceiling and a break-even length ratio (ties to Parts B and C)

**Task.** Part 1 uses SparseSpec's Table 2 (fact 11). (a) Compute the attention share and the Amdahl ceiling. (b) Compare the paper's byte model with its table for attention and for GEMM, and check the paper's k = 16 example. (c) Express the system in Leviathan's terms: compute c and S. (d) Compute ceilings for three designs that start from SparseSpec: more attention work; the same passes but keeping every drafted token (KV-only repair, Part C); no verify pass at all (attention by bytes, and attention free). Divide by λ = 1.20 and λ = 1.61. Part 2 is a toy. (e) Simulate 200,000 requests with a cap of 16,384 tokens, a gain of 1.30 and wrong tokens at a rate of 5% of emitted tokens. Model 1: each wrong token adds 3 tokens. Model 2: each wrong token derails the run to the cap with a small probability h; tune h so that the mean λ equals Model 1. Compare mean, median, share of losing requests, cap rate and how much of the extra length sits in the top tenth.

**Expected result.** (a) f = 0.596, ceiling 2.47×, measured 1.79×. (b) Attention: 5.11× predicted, 3.29× measured. GEMM per token: 1.257× predicted, 1.236× measured, so GEMM time per pass is nearly flat in the number of tokens here. The k = 16 example gives 7.22× with α = 0.75 and 6.78× with α = 0.70. (c) c = 0.434 (floor 0.404), model S = 1.60×, ceiling 1/c = 2.30×. The model is below the measured 1.79× because the table's CPU row also fell from 3.2 to 0.5 ms, which the model does not contain. (With the CPU row left at 3.2 ms the table would give 28.7/18.7 = 1.53×.) (d) More attention work: at most 1.48×. KV-only repair: 1.257×. No verify pass: 1.64× by bytes, 1.80× with free attention. So no design that keeps SparseSpec's draft can afford λ above about 1.8, and KV-only repair at k = 8 cannot afford λ above 1.257. (Q17 gives 1.44 for the no-verify case, not 1.64. The idea is the same. Q17 uses c = 0.43, which keeps the baseline CPU time of 3.2 ms in every draft step; here the CPU row is SparseSpec's 0.5 ms.) With λ = 1.61 the no-verify design is at 1.02× by bytes: break-even at best. (e) Both models have mean λ = 1.176. Per-token cost: median 1.176, no request loses, top tenth holds 21% of the extra tokens. Lasting damage: median 1.000, 6.4% of requests lose badly, top tenth holds 100% of the extra tokens, and one wrong token in about 3,300 is enough. The median request is 1.30× faster while the workload is 1.13× faster.

**How the tutor checks.** Three questions without code. "In (d), why is KV-only repair capped at 1.257 while no-verify reaches 1.64?" (KV-only keeps the full pass; it only stops throwing tokens away. No-verify removes the full pass and its GEMM.) "In (e), the two models have the same mean λ. Which statistic separates them, and which repair fits each?" (Median against mean, or the top-tenth share. Per-token cost: let fewer wrong tokens through. Lasting damage: detect the derailed run and roll back; lowering ρ helps only in proportion.) "Why do the two workload speeds in (e) differ (1.107× and 1.126×) when the mean λ is equal?" (Mean of ratios against ratio of sums: in Model 2 long requests derail more often but have a smaller λ when they do, because the cap is closer.)

The 1.20 and 1.61 in (d) are the two CVSA length increases (+20% with a new selection at every step, +61% with reuse) [from notes: 2611.10634 App. P]. They come from another model and task; they are used here only as sizes. Part 2 is a toy and says nothing about real data. The real pattern (median λ 1.08, mean 1.54 [checked: 2610.23818]; "the tenth of cases with the largest increase holds 59% of the net extra tokens at the median" [checked: 2610.23818 Sec 3.7, primary pairs]) lies between the two toy models.

```python
# Exercise 3: from a published time breakdown to a ceiling and a break-even length ratio.
# Part 1 uses SparseSpec (arXiv 2512.01278) Table 2: Qwen3-8B, AIME, H100, milliseconds.
import numpy as np

base = {"cpu": 3.2, "attn": 17.1, "gemm": 7.2, "other": 1.2}   # vLLM
spec = {"cpu": 0.5, "attn": 5.2, "gemm": 8.9, "other": 1.4}    # SparseSpec
Tb, Ts = sum(base.values()), sum(spec.values())
k, acc_len, s = 8, 6.16, 0.05           # draft length, accepted tokens per round, draft KV budget
alpha = acc_len / k                     # SparseSpec's alpha = accepted / drafted

print("(a) shares and Amdahl")
f = base["attn"] / Tb
print(f"  totals: vLLM {Tb:.1f} ms, SparseSpec {Ts:.1f} ms, measured speedup {Tb / Ts:.3f}x")
print(f"  attention share f = {f:.3f}; ceiling if attention were free = {1 / (1 - f):.3f}x")
print(f"  attention 3.29x faster, other rows unchanged: {1 / ((1 - f) + f / 3.29):.3f}x")

print("(b) the paper's byte model against its own table")
print(f"  attention: predicted (k*alpha+1)/(k*s+1) = {(k * alpha + 1) / (k * s + 1):.2f}x,"
      f" measured {base['attn'] / spec['attn']:.2f}x")
print(f"  GEMM per token: predicted (k+1)/(k*alpha+1) = {(k + 1) / (k * alpha + 1):.3f}x,"
      f" measured {spec['gemm'] / base['gemm']:.3f}x")
for a in (0.75, 0.70):
    print(f"  paper example k=16, s=0.05, alpha={a}: {(16 * a + 1) / (16 * s + 1):.2f}x")

print("(c) the same system in Leviathan's terms")
c = (Tb - base["attn"] + s * base["attn"]) / Tb
print(f"  c = (untouched time + s * attention) / total = {c:.3f}; lower bound 1 - f = {1 - f:.3f}")
print(f"  S = (k*alpha+1)/(k*c+1) = {(k * alpha + 1) / (k * c + 1):.3f}x; ceiling 1/c = {1 / c:.2f}x")

print("(d) ceilings for designs that start from SparseSpec")
print(f"  any further attention-only work: {Ts / (Ts - spec['attn']):.3f}x")
keep_all = Ts * (k * alpha + 1) / (k + 1)
print(f"  same passes, keep every drafted token (KV-only repair): {keep_all:.2f} ms,"
      f" gain {Ts / keep_all:.3f}x")
no_check = spec["cpu"] + s * base["attn"] + base["gemm"] + base["other"]
print(f"  no verify pass, attention by bytes: {no_check:.2f} ms, gain {Ts / no_check:.3f}x")
free = spec["cpu"] + base["gemm"] + base["other"]
print(f"  no verify pass, attention free: {free:.2f} ms, gain {Ts / free:.3f}x")
for lam in (1.20, 1.61):
    print(f"  length ratio {lam}: end-to-end against SparseSpec = {Ts / no_check / lam:.3f}x (bytes),"
          f" {Ts / keep_all / lam:.3f}x (KV-only repair)")

# Part 2: toy model. Same mean length ratio, two kinds of damage.
print("(e) toy: per-token cost against lasting damage, same mean length ratio")
rng = np.random.default_rng(0)
n, cap, gain, rho, m = 200_000, 16384, 1.30, 0.05, 3
L_exact = np.minimum(rng.lognormal(np.log(4000), 0.5, n), cap)

def summary(name, L):
    lam = L / L_exact
    extra = np.sort(L - L_exact)[::-1]
    top = extra[: n // 10].sum() / extra.sum()
    print(f"  {name}: mean lambda {lam.mean():.3f}, median {np.median(lam):.3f},"
          f" requests with lambda > gain {np.mean(lam > gain):.3f},"
          f" at cap {np.mean(L >= cap):.3f}, top tenth holds {top:.2f} of extra tokens")
    print(f"      workload speed against exact (sum of tokens): {gain * L_exact.sum() / L.sum():.3f}x,"
          f" median request: {gain / np.median(lam):.3f}x")
    return lam.mean()

L_tok = np.minimum(L_exact / (1 - rho * m), cap)        # every bad token adds m tokens
target = summary("per-token cost ", L_tok)
u = rng.random(n)
lo, hi = 0.0, 1e-2                                       # h = chance that one bad token derails the trace
for _ in range(60):
    h = (lo + hi) / 2
    L_dmg = np.where(u < 1 - np.exp(-rho * h * L_exact), cap, L_exact)
    lo, hi = (h, hi) if (L_dmg / L_exact).mean() < target else (lo, h)
print(f"  calibrated h = {h:.2e} (one bad token in {1 / h:.0f} derails the trace)")
summary("lasting damage ", L_dmg)
```

Output:

```
(a) shares and Amdahl
  totals: vLLM 28.7 ms, SparseSpec 16.0 ms, measured speedup 1.794x
  attention share f = 0.596; ceiling if attention were free = 2.474x
  attention 3.29x faster, other rows unchanged: 1.709x
(b) the paper's byte model against its own table
  attention: predicted (k*alpha+1)/(k*s+1) = 5.11x, measured 3.29x
  GEMM per token: predicted (k+1)/(k*alpha+1) = 1.257x, measured 1.236x
  paper example k=16, s=0.05, alpha=0.75: 7.22x
  paper example k=16, s=0.05, alpha=0.7: 6.78x
(c) the same system in Leviathan's terms
  c = (untouched time + s * attention) / total = 0.434; lower bound 1 - f = 0.404
  S = (k*alpha+1)/(k*c+1) = 1.601x; ceiling 1/c = 2.30x
(d) ceilings for designs that start from SparseSpec
  any further attention-only work: 1.481x
  same passes, keep every drafted token (KV-only repair): 12.73 ms, gain 1.257x
  no verify pass, attention by bytes: 9.75 ms, gain 1.640x
  no verify pass, attention free: 8.90 ms, gain 1.798x
  length ratio 1.2: end-to-end against SparseSpec = 1.367x (bytes), 1.047x (KV-only repair)
  length ratio 1.61: end-to-end against SparseSpec = 1.019x (bytes), 0.781x (KV-only repair)
(e) toy: per-token cost against lasting damage, same mean length ratio
  per-token cost : mean lambda 1.176, median 1.176, requests with lambda > gain 0.000, at cap 0.006, top tenth holds 0.21 of extra tokens
      workload speed against exact (sum of tokens): 1.107x, median request: 1.105x
  calibrated h = 3.00e-04 (one bad token in 3329 derails the trace)
  lasting damage : mean lambda 1.176, median 1.000, requests with lambda > gain 0.064, at cap 0.068, top tenth holds 1.00 of extra tokens
      workload speed against exact (sum of tokens): 1.126x, median request: 1.300x
```

Other arithmetic used in the question bank was also run (`checks.py` in the scratch folder): Q1 to Q5, Q7 to Q14, Q16, Q17, Q19 and the two-setting example in Q6.

---

## 6. Diagnostic picks

- **Level 2-3: speed_accounting-Q4** (3.5 against 4.9 tokens per round, completions 1.25 times longer). One short computation shows whether the learner holds both factors (gain and length) or only one. A level-3 answer gives the round ratio 0.89 and the 1.12× without help and names the break-even at 1.40.
- **Level 4-5: speed_accounting-Q12** (keep the full pass, never reject). A level-4 answer sees that no pass disappears and gives the ceiling 9/7.16 = 1.257. A level-5 answer adds that a longer draft lifts the bound only to about 1.44 (this needs c = 0.43), turns the ceiling into a bound on the length ratio, and says what it means for Part C: it is the whole speed prize of a "KV channel" result at k = 8.

If the learner passes Q12 at once, go to Q17 (selective verification) and Q16 (the cap). If the learner fails Q4, go back to Q2 and Q1.

---

## 7. Sources

Fetched by the writer (through a fetch tool that returns extracted answers from the page; each id was confirmed against the page title). The verification pass downloaded the raw arXiv HTML of the same five ids, confirmed each title in the page, and read the cited passages there:

| id | title (as shown on the page) | what I read |
|---|---|---|
| arXiv 2211.17192 | Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman, Matias) | arXiv HTML: Def 3.7 and the text near it, Theorem 3.8 with proof, Corollary 3.9, Sec 3.4 (Def 3.10, Theorem 3.11, Table 1), Sec 3.5, the compute-assumption sentences in Sec 3.3, 3.4 and 6, Table 2, the c values of Table 4 |
| arXiv 2512.01278 | Accelerating Large-Scale Reasoning Model Inference: Self-Speculative Decoding with Sparse Attention (SparseSpec) | arXiv HTML: Sec 3.1, Sec 3.2 (cost model and the k = 16 example), Table 2 and the sentences on it, Sec 5.1 (setup), the 6.16-of-8 sentence in Sec 5.3 |
| arXiv 2605.24168 | Inference Time Context Sparsity: Illusion or Opportunity? | arXiv HTML: Table 1 and Table 2 with captions, the definition of S× sparsity, the sentences on break-even |
| arXiv 2501.01005 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | arXiv HTML: Table 5 and the three sentences on it (in the last 13,764 characters of the page); from the first 100,000 characters only the title and the parallel-generation sentences. The page does not say which GPU Table 5 was measured on. |
| arXiv 2610.23818 | Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding | arXiv HTML, first 100,000 of 152,166 characters: Sec 2.1-2.2 (Eq. 1-5 and the ceiling sentence), Sec 3.1 (metrics), Sec 3.3, Sec 3.4, Sec 3.7, the lossless ℓ̄ values of Table B1, the budget and cut-off text. Appendices C and F were not read by the writer. The verification pass read App. C, the Limitations section, Table B1 and Table B2. |

Read directly on disk:

| source | what I read |
|---|---|
| `reports/length-tax.html` | full text |
| `research_notes/Length tax preliminaries/specdec_exact.md` | header, core ideas, ladder, misconceptions, Q1-Q6, Q19, sources (for format and to avoid overlap) |
| `research_notes/Inference speed question search/harvest_kernels_gap.md` | the source list and the "Extracted claims" section |
| `research_notes/Inference speed question search/harvest_specdec_core.md`, `harvest_specdec_self.md`, `harvest_own_projects.md`, `questions_ranked.md`, `verification.md` | keyword search only (2610.23818, floor, per call, rounds) |

Cited from those notes only, not re-read in this run [from notes]: S3R 2610.13202 (80.4% to 96.7% acceptance, 1.370× to 1.058×); CVSA 2611.10634 App. P (+61% and +20% length); the SubSpec fused-kernel result and the kv-fetch AIME cap rate (both via `reports/length-tax.html`). The arXiv ids 2610.13202 and 2611.10634 were not opened in either pass, so their titles are not confirmed here. The other facts that the writer took from notes (2610.23818: the 7%-24% share, the 0.94-1.23 range, the "top tenth holds 59%" sentence; 2512.01278: the kernel-bandwidth remark and "consistent with our estimation") were read in the raw pages by the verification pass and are now tagged [checked].

Open points a reader should know:

- The SparseSpec example "k = 16, α = 0.75, s = 0.05 gives 6.78×" does not match the formula (kα + 1)/(ks + 1), which gives 7.22×; 6.78× matches α = 0.70. Resolved by the verification pass: the raw page prints k = 16, α = 0.75, s = 0.05 and 6.78×. The mismatch is in the paper.
- SparseSpec's Table 2 does not say what one time unit is (per step or per emitted token). Only ratios are used here.
- For Leviathan, the text says "c was always less than 0.05" and Table 4 (App. A.3) lists c = 0.11 for T5-large. Resolved by the verification pass: both are in the paper; the paper does not agree with itself here.
- SparseSpec's 6.16 of 8 is an average over models and datasets (Sec 5.3), not the acceptance of the Table 2 run. Every number in this file that mixes 6.16 with Table 2 (5.11×, 1.257×, 7.16 tokens per round) is a rough size, not a measured value.
- In 2605.24168, "S× sparsity" is per query head, with 32 query heads over 8 KV heads. How many KV bytes the sparse kernel really reads is not given in the paper.
- The "17 µs floor" and the "13 µs" fit are my arithmetic on a published table for an H100. No source measures the fixed cost directly for sparse decode on a 48 GB card.
- The model shape in Q11 (36 layers, 8 KV heads, dimension 128, 16 GB) and the "36 layers, 17 µs" in the Q19 follow-up are given numbers for the exercise, not checked facts about a named model.

---

## Verification log

Second pass by a separate checker, finished 2026-11-13 15:34 CST (time from `date`). No GPU job was run. Nothing was posted, committed or pushed.

**What was done.**

- Every core idea (20 plus the screening recipe), the level ladder, all 8 misconceptions and all 19 questions were solved or derived again before reading the key. 54 items in total, counting the 3 exercises, the diagnostic picks and the source notes.
- All three exercise scripts were copied out of this file and run again on CPU (Python 3.11.5, numpy 1.24.3). The outputs are the same as the pasted outputs, character for character.
- The raw arXiv HTML of five papers was downloaded (one request each): 2211.17192, 2512.01278, 2605.24168, 2501.01005, 2610.23818. Each title was confirmed in the page. Every [checked] fact was read there.
- New arithmetic is in `verify_checks.py` in the scratch folder.

**Wrong, now corrected.**

1. Core idea 2. The file said the paper "notes its model leaves out the per-round cost o". The paper has no such sentence. Replaced by what the paper does say: App. C fits time per round as c0 + c1 × tokens committed, with the values for both servers.
2. Level ladder, level 4. The Amdahl ceiling was written "1 − f". It is 1/(1 − f).
3. Q12 key and the diagnostic pick. The key said a longer draft cannot lift the gain above 1/0.77 = 1.30. That bound holds only if the exact system is also moved to the longer draft with the same acceptance share. Against SparseSpec at k = 8 the right bound is 0.620/0.43 = 1.44 (1.39 at k = 32), and it needs c. The level-4 answer (1.257 at k = 8) did not change. The question now says "the same draft length of 8".
4. Q18 key. Two parts were wrong. "A fixed overhead per round makes fewer rounds save more than the formula says" is false: a fixed cost per round gives exactly the round ratio. "v > 1 at batch 64" does not change the relaxed/exact ratio, because both arms pay the same pass. The key now gives: time per round can grow with tokens committed (true on one of the paper's two servers); the mean and the tail matter on a batch server; and the 7% is a median over rules and drafters, so one's own setting can be better. The two false answers are listed as traps.
5. Q19 key, reason 4. "α is a mean over rounds while the cost is per round" explains nothing (cost per round is fixed, tokens add linearly). Replaced by a real reason: 6.16 of 8 is an average over models and datasets (Sec 5.3), not the Table 2 run. The key also says acceptance alone would need 3.6 of 8, so it is not the whole gap.

**Ambiguous, now rewritten.**

6. Q7. "Reads 1/100 of the tokens" became "each query head attends to 1/100 of the tokens" (the paper's definition). The follow-up called 13 µs "left unexplained by bytes"; that number is a gap, not the fixed cost. A two-point fit gives a ≈ 16.7 µs. A tutor note says so.
7. Q16. The key used a fact that the question did not give ("much of that is answers cut off at the cap"). The question now says some runs stop at the cap, and the key says the loss "may be" cut-off answers.
8. Core idea 19, misconception 3, Exercise 2. "The sparse kernel costs 4.5 to 6 times more per token read" was too strong. Sparsity is per query head and 4 query heads share one KV head, so up to 4/S of the KV bytes may be read. The text now says this, marked as my reading.
9. Core idea 18, misconception 7. "Time is worse than rounds" was given as a general result. The paper finds it on one server and not on the second. Both are now stated.
10. Q19 question. It asks for three reasons and the key has four. The question now says any three count.

**Minor.**

11. Core idea 9: the factor (k + 1)/(kā + 1) is in the paper's own formula, not only a derivation here. The paper's full approximation was added.
12. Core ideas 11 and 14, Exercise 3: a caution that 6.16 is an all-models average without the bonus token; the paper's "more than 77%" attention share (Sec 3.1) against 59.6% in Table 2; a note on why Q17 (1.44) and Exercise 3d (1.64) differ; the 1.53× figure without the CPU gain.
13. Q17: "only worth building if λ is very close to 1" was looser than the numbers. Now: the gain is at most 1.44×.
14. Misconception 6: the title said "13% slower"; Q6 says 15%. Now both say 15% more rounds.
15. Q11: the "large-batch operating point" wording was dropped (the caption does not give the batch). A tutor note says case 2 (93 GB) does not fit on an 80 GB card.
16. Follow-up answers were missing for Q9, Q10, Q11, Q14, Q15, Q16 and Q17. They were added.
17. Tags. Now [checked] with a section: the 7%-24% share (Sec 3.7, Table B2), the 0.94-1.23 range (Sec 3.3), the top-tenth 59% (Sec 3.7), the budget quote (Limitations), the 1.27× temperature control (Sec 3.7, App. C), the kernel-bandwidth remark (2512.01278 Sec 4.2), "consistent with our estimation" (2512.01278 Sec 5.2), Leviathan Table 4 (App. A.3).
18. Two open points were closed. The 6.78× example and the "c < 0.05" against 0.11 conflict are both printed in the papers themselves. They are not extraction errors.

**Nothing was deleted.** No question had to be removed.

**Still not verified.**

- S3R, arXiv 2610.13202 (acceptance 80.4% to 96.7%, speedup 1.370× to 1.058×): [from notes]. The page was not opened, so the id and title are not confirmed here.
- CVSA, arXiv 2611.10634 App. P (+20% and +61% length): [from notes]. The page was not opened, so the id and title are not confirmed here.
- The SubSpec fused-kernel numbers (1.15-1.37× alone, about 2% slower end to end): [from notes], from `reports/length-tax.html` only.
- The kv-fetch AIME cap rate of 41%: read in `reports/length-tax.html`, not in the raw log.
- SparseSpec Table 2: the caption and the sentences on the table do not state the time unit or the batch size, and the acceptance of that run is not given. All numbers that put 6.16 into Table 2 are sizes, not measurements.
- 2605.24168 Table 1: the number of KV bytes the sparse kernel reads, and the cause of the batch-1 plateau, are not in the paper. The "17 µs floor" is arithmetic on a published H100 table. Nobody measured it on a 48 GB card.
- The model shape in Q11 and the "36 layers, 17 µs" in the Q19 follow-up are made-up exercise inputs.
- Exercise 3e is a toy. It shows the two damage models can share a mean; it says nothing about real traces.
