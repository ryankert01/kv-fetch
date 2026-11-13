# Tutor pack: relaxed (lossy) verification and the dose of an approximation

Subskill id: `lossy_verify`. Written 2026-11-13 15:17 CST. Read-only work; no GPU job was run.

Scope: relaxed acceptance rules (budget rules, switch rules, the cascade token rule, typical acceptance) and what each one emits; ways to measure how much a lossy method perturbs generation (the "dose"); the difference between acceptance rate, tokens per round, number of rounds, completion length and wall-clock.

This pack builds on `specdec_exact.md` (same folder). It does not repeat the exact rule or the tree and self-speculative variants.

Notation: **p** is the target (full model) next-token distribution, **q** is the drafter's. TV(a, b) = ½ Σ_v |a(v) − b(v)|. H(a) = −Σ_v a(v) ln a(v), in nats. KL(a‖b) = Σ_v a(v) ln(a(v)/b(v)). norm(f) = f / Σ_v f(v). **r** is the distribution a rule really emits at one position.

How to read the tags:

- **[checked]**: I read the fact in this run, in a copy of the arXiv HTML page saved in the scratch folder on 2026-11-11. Each copy shows the paper title, and I matched it to the id. I did not fetch any page again today. Formulas were re-derived or run in the exercises.
- **[from notes]**: taken from `research_notes/Inference speed question search/verification.md` (card F07) or `reports/length-tax.html`; not re-read in the paper.
- **[unverified]**: from memory only.
- No tag: textbook fact or a derivation written out here.
- **Examiner pass (2026-11-13 15:35 CST):** a second reader solved every item again, read the cited passages again in the saved copies, and ran the exercises again. A [checked] tag now means: read in the source copy by the author and again by the examiner. All changes are listed in the Verification log at the end.

---

## 1. Core ideas

1. **Exact rule, in one line.** Draft x ~ q. Accept with probability min(1, p(x)/q(x)). On rejection, sample from norm(max(0, p − q)). The emitted token follows p, and the acceptance probability is Σ_v min(p(v), q(v)) = 1 − TV(p, q) [checked: 2610.23818 Eq. 2].

2. **The relaxed template.** A relaxed rule replaces p by three "relaxed targets": π_rej in the accept test, π_res in the residual, π_bonus for the bonus token. Accept with probability min(1, π_rej(x)/q(x)); on rejection sample from norm(max(0, π_res − q)); after a fully accepted draft sample one more token from π_bonus [checked: 2610.23818 Sec 2.2 and Algorithm 1 in App. A].

3. **What is emitted.** Let a = Σ_v min(q(v), π_rej(v)) be the acceptance probability. The emitted law is

   r(v) = min(q(v), π_rej(v)) + (1 − a) · norm(max(0, π_res − q))(v).

   If π_rej = π_res = π and π is a distribution, then Σ_v max(0, π − q) = 1 − a, so r(v) = min(q, π) + max(0, π − q) = π(v). The rule then samples exactly from π, and a = 1 − TV(π, q). This procedure is GenSpecSample [checked: 2405.19261 Algorithm 4 gives the procedure]. The proof that it emits π is the short derivation in Q8.

4. **Acceptance gain costs at least the same TV dose.** The accepted mass at v is at most r(v) and at most q(v), so a ≤ Σ_v min(q, r) = 1 − TV(r, q). The exact rule has a* = 1 − TV(p, q). By the triangle inequality, a − a* ≤ TV(p, q) − TV(r, q) ≤ TV(r, p). A rule cannot gain 10 points of per-token acceptance while moving the output less than 0.10 in TV. For a mixture π = (1 − w)·p + w·q the two are equal. Derivation: π − q = (1 − w)(p − q), so the rejection probability is TV(π, q) = (1 − w)·TV(p, q). And π − p = w(q − p), so the dose is TV(π, p) = w·TV(p, q). The gain is TV(p, q) − (1 − w)·TV(p, q) = w·TV(p, q), the same number. (2405.19261 Lemma 3 states the rejection probability only for a yes-or-no choice between q and p, which is w = 1 or w = 0 [checked]. The mixture case is derived here.)

5. **Only overshoot tokens can be added.** The exact rule accepts every draft token with q(x) ≤ p(x) with probability one. So a relaxed rule can only add tokens with q(x) > p(x): tokens the drafter likes more than the target does [checked: 2610.23818 Sec 3.7]. 2607.26627 calls q(x) ≤ p(x) underestimation and q(x) > p(x) overestimation. It keeps the words "severe overshoot" for q(x) ≥ p(x)/ℓ [checked: 2607.26627 Sec 3.1]. This pack says "overshoot" for all of q(x) > p(x), as 2610.23818 Sec 3.7 does.

6. **Two families of rule** [checked: 2610.23818 Sec 2.3]. Budget rules keep the relaxed target within a divergence budget from p (CACTUS, mentored decoding). Switch rules trust the drafter outright when a test at that position passes (fuzzy SD, the two cascade rules).

7. **Mentored decoding (lenience form).** π_rej(v) = min{p(v)/(1 − α), 1}, π_res = p, π_bonus = p [checked: 2610.23818 Table A1]. The accept test is min(1, p(x)/((1 − α)·q(x))), which is Leviathan's lenience rule with ℓ = 1 − α [checked: 2607.26627 Eq. 9]. π_rej is not a distribution, so idea 3 gives the three-region law [checked: 2607.26627 Eq. 10; confirmed numerically in Exercise 1]:

   r(v) = q(v) + Δ·(p(v) − q(v)) if q(v) ≤ p(v); r(v) = q(v) if p(v) < q(v) ≤ p(v)/ℓ; r(v) = p(v)/ℓ if q(v) > p(v)/ℓ,

   with Δ = Σ_v max(0, q(v) − p(v)/ℓ) / Σ_v max(0, p(v) − q(v)) = (rejection probability of the lenient rule) / (rejection probability of the exact rule). Overshoot is kept up to the cap p/ℓ. Leviathan et al. define lenience as multiplying q(x) by ℓ before the comparison with p(x), and state the same cap: no token is sampled with probability above p(x)/ℓ [checked: 2211.17192 App. A.5].

8. **CACTUS.** For the drafted token x: π_rej(x) = γ_x = min{p(x) + sqrt(2α·p(x)·(1 − p(x))), 1}, and every other token v gets π_rej(v) = ((1 − γ_x)/(1 − p(x)))·p(v). π_res = π_rej. α is a budget on KL(π‖p) [checked: 2610.23818 Table A1]. The boost is additive and grows like sqrt(p(x)), so the relative boost γ_x/p(x) is largest for unlikely tokens: at α = 0.35, p(x) = 0.01 gives γ_x = 0.093 (9.3 times), p(x) = 0.5 gives 0.918 (1.8 times).

9. **Cascade OPT rule (position switch).** π = q if max_u q(u) ≥ max_u p(u) − α·TV(p, q), else π = p [checked: 2405.19261 Eq. 10; 2610.23818 Table A1]. When it trusts q, every draft token at that position is accepted, tail tokens included. The cascades paper names the weakness: a poor token can pass "because q_t happens to be more peaked than p_t" [checked: 2405.19261 Sec 4.4].

10. **Cascade token rule (TokenV3, `spec_casc_tok`).** Trusted set A_α = {u : p(u) ≥ (1 − α)·max_w p(w)}. η_α = 1 − Σ_{u∈A_α} q(u). π(v) = q(v)·1[v ∈ A_α] + η_α·p(v) [checked: 2405.19261 Eq. 15; 2610.23818 Table A1]. A trusted token is accepted with probability one. Any other token is tested against η_α·p(v) ≤ p(v), which is stricter than the exact rule. It is the only rule in the sweep that can reject what the exact rule accepts [checked: 2610.23818 Sec 2.3]. Implementation note: on some pairs of that paper (the EAGLE-3 head of Qwen3-8B, the EAGLE heads of the two Llama targets, DSpark), cactus and spec_casc_tok change only the accept test and keep π_res = p [checked: 2610.23818 Sec 3.1 footnote, Table A1 caption]. The emitted law r = π above holds for the full rule only.

11. **Typical acceptance (Medusa).** Accept candidate x if p(x) > min(ε, δ·exp(−H(p))), with ε a hard threshold and δ an entropy-scaled one [checked: 2401.10774 Sec 2.3.1]. The test uses p only and has no coin. At temperature 0 it becomes greedy decoding [checked: 2401.10774 Sec 2.3.1]. Seen as "accept iff x is in an allowed set A", the accepted tokens follow q restricted to A, so the output "mirrors the draft model", not the truncated target [checked: 2607.26627 Definition 1, Eq. 13, Sec 5.1].

12. **The knob α is not comparable across rules.** It is a KL budget in CACTUS and mentored decoding, a JS threshold in fuzzy SD, a deferral cost in OPT, a relative probability cut in the token rule. Every rule except the two cascade rules is exact at α = 0; OPT is exact only for α < −1 and the token rule only for α < 0. So compare rules at matched mean accepted length ℓ̄, not at matched α [checked: 2610.23818 Sec 2.2, Table A1].

13. **Dose: what to measure per token.** (a) Substitution share: the share of committed draft tokens that only the relaxed rule accepts. (b) How unlikely they are: target probability p(x) and target rank of those tokens. (c) Target entropy at those positions. (d) Distance of the emitted law: TV(r, p), KL(r‖p). (e) Excess surprisal: E_r[−ln p(x)] − H(p) = KL(r‖p) + H(r) − H(p). Measure (e) is zero in expectation under exact sampling. It is positive when emitted tokens are less likely than the target's own samples, and negative when the rule sharpens the output.

14. **Dose measures disagree, and the sign matters.** In the toy of Exercise 1, the token rule has TV 0.25 and excess surprisal −0.38 nats; CACTUS has TV 0.18 and excess surprisal +0.10. TV does not say in which direction the output moved.

15. **Measured dose (GPT-OSS-20B, traced runs, loosest settings)** [checked: 2610.23818 Table B2]:

    | rule | share only relaxed accepts | median p, "only" | median p, "both" | mean rank, "only" (0 = top) | mean H, "only" / "both" (nats) |
    |---|---|---|---|---|---|
    | mentored_dec | 16% | 0.09 | 0.93 | 37 | 1.8 / 0.8 |
    | cactus | 24% | 0.01 | 0.78 | 234 | 2.1 / 1.1 |
    | spec_casc_opt | 16% | 0.07 | 0.92 | 88 | 1.8 / 0.8 |
    | r_fuzzy | 21% | 0.01 | 0.83 | 134 | 1.8 / 1.0 |
    | spec_casc_tok | 7% | 0.29 | 0.99 | 0.9 | 1.5 / 0.5 |

    The extra tokens are unlikely under the target and sit at high-entropy positions. The one rule that does not inflate length adds few tokens, and they are near the top of the target's ranking.

16. **The accounting chain.** Per-token acceptance a → mean accepted draft tokens per round ℓ̄ → tokens per round ℓ̄ + 1 → rounds R ≈ L/(ℓ̄ + 1) → time. With i.i.d. acceptance a and N_draft draft tokens, ℓ̄ = Σ_{k=1..N} a^k = a(1 − a^N)/(1 − a), so tokens per round = (1 − a^(N+1))/(1 − a). L is the completion length in tokens. Each arrow can break: a relaxed rule changes L.

17. **Length-corrected speed.** Exact: S ≈ (ℓ̄ + 1)/(1 + N_draft·c_rel), with c_rel = (drafter step cost)/(target step cost) [checked: 2610.23818 Eq. 1]. Relaxed: S_relax ≈ (L̄_AR/L̄_relax)·S, with L̄ the mean completion length [checked: 2610.23818 Eq. 3; the same correction is Eq. 4 of 2607.08690]. Rounds: R_relax/R_strict ≈ λ·(ℓ̄_strict + 1)/(ℓ̄_relax + 1), with λ = L̄_relax/L̄_strict [checked: 2610.23818 Eq. 5]. A relaxed rule saves rounds only if the gain g = (ℓ̄_relax + 1)/(ℓ̄_strict + 1) is larger than λ. Since ℓ̄_relax ≤ N_draft, no relaxation of a drafter can absorb more than λ = (N_draft + 1)/(ℓ̄_strict + 1) [checked: 2610.23818 Sec 2.2].

18. **Headline numbers** [checked: 2610.23818 abstract, Sec 3.3, 3.4]. At the loosest settings, over 360 cells: 1.34 times more tokens per round, but 1.54 times longer completions (means). λ has median 1.08; completions lengthen in 248 cells. R/R* has median 0.93 and mean 1.10; 114 cells need more rounds than exact. The round ratio follows length (Spearman ρ = 0.78 with λ), not the acceptance gain (ρ = −0.12).

19. **Time is not rounds.** Total time ≈ c0·R + c1·L, where c0 is the fixed cost of a round and c1 the cost per committed token. On the paper's first server c0 is 12 to 14 ms and c1 is 2.5 to 3.2 ms per token, so a relaxed round costs 1.12 times an exact round on average; on a second server c1 ≤ 0.08 ms and the ratio is 1.00 [checked: 2610.23818 App. C]. Eleven cells save rounds but not time [checked: 2610.23818 Sec 3.4].

20. **Dose alone does not explain everything yet.** (a) Exact decoding at a higher temperature also inflates: GPT-OSS-20B on GSM8K at T = 1.2 has λ = 1.27 [1.10, 1.47]; Qwen3-8B barely moves (λ 1.02 to 1.04) [checked: 2610.23818 App. C]. (b) Changing the exact draft length N_draft from 2 to 10 does not change length (λ 0.96 to 1.06) [checked: App. C]. (c) Standalone Qwen3 drafters lengthen significantly in none of 60 cells; "the inflation is a property of the drafter" [checked: Sec 3.6]. (d) A per-token guard (for spec_casc_tok at α = 0.3 on GPT-OSS-20B) tuned on 8 AIME24 cases cut their mean length from 13,364 to 10,534 tokens. On all 30 cases the mean length rose from 9,270 to 10,363 tokens, while correct answers went from 22 to 23 [checked: App. F]. (e) The paper says it does "not explain why an unlikely token makes a reasoning model think longer" [checked: Limitations]. This gap is the research question.

---

## 2. Level ladder

- **Level 2.** Can say that a relaxed rule accepts more draft tokens and no longer samples from p. Can name two rules and state that acceptance rate, tokens per round and wall-clock are different numbers.
- **Level 3.** With the formula sheet, can compute the acceptance probability and the emitted distribution of a given rule on a small example. Can apply R_relax/R_strict ≈ λ/g to given numbers and say whether rounds were saved.
- **Level 4.** Without help, can write π_rej and π_res for each rule, derive what is emitted (including the mentored three-region law), explain why only overshoot tokens are added, choose a dose measure and defend it, and separate acceptance, rounds, length and time in a result table.
- **Level 5.** Can teach it and handle edge cases: greedy targets, budget-censored lengths, heavy-tailed length ratios, truncation confounds in a "which tokens to verify" gate. Can design the dose-matched controls for Parts A, B and C and say what each control cannot show.

---

## 3. Misconceptions

1. **"A higher acceptance rate means a faster system."** Wrong. Rounds scale with λ/g, and λ can be larger than g. Over 360 cells the mean gain is 1.34 and the mean λ is 1.54 [checked: 2610.23818 abstract].

2. **"α = 0 is exact for every rule, and the same α means the same looseness."** Wrong. The token rule at α = 0 still trusts the argmax set, and OPT at α = 0 still trusts a drafter that is more peaked than the target. α has a different unit in every rule [checked: 2610.23818 Table A1].

3. **"A relaxed rule accepts everything the exact rule accepts, plus more."** Wrong for the token rule. A token outside the trusted set is tested against η_α·p(v) ≤ p(v), so it can be rejected where the exact rule accepts it [checked: 2610.23818 Sec 2.3].

4. **"Typical acceptance samples from a truncated target, like η-sampling."** Wrong. It accepts draft tokens that fall in the allowed set, so inside the set the output follows the drafter. Against the matched baseline (exact verification with η-sampling on the target) it loses: 69.84 against 76.12 accuracy on MATH with Llama-3.1-8B and EAGLE-3, averaged over the paper's parameter grid. It accepts more tokens per step (4.61 against 3.93). The EAGLE-3 baseline with no truncation has 73.20; the paper marks that row as not directly comparable [checked: 2607.26627 Table 3].

5. **"The mentored rule emits min(p/(1 − α), 1), normalized."** Wrong. π_rej is only the accept test; the residual still uses p. The emitted law is the three-region form of idea 7, and π_rej does not even sum to one.

6. **"The dose is one number, for example TV or the substitution rate."** Wrong. A rule can have a large TV and make the output more likely under the target (the token rule). Table B2 shows a 7% share with rank 0.9 next to a 24% share with rank 234 [checked: 2610.23818 Table B2].

7. **"Fewer rounds means less time."** Wrong on a server where a round costs more when it commits more tokens. Time ≈ c0·R + c1·L, so the c1 part follows length, not rounds [checked: 2610.23818 App. C].

8. **"Find the risky tokens and verify only those; the tax will go away."** Not shown. The per-token guard in the paper shortened its 8 tuning cases (mean 13,364 to 10,534 tokens). It did not shorten the full set: on all 30 cases the mean length rose from 9,270 to 10,363 tokens. Accuracy did not fall (22 to 23 correct of 30) [checked: 2610.23818 App. F]. The length distribution is heavy-tailed, and a guard "judges one draft token at a time and cannot see whether a run will ever close its trace".

---

## 4. Question bank

Each question is asked alone. The tutor waits for the answer before using the follow-up. "Toy pair" below always means p = [0.50, 0.25, 0.12, 0.07, 0.05, 0.01] and q = [0.60, 0.10, 0.05, 0.03, 0.02, 0.20] (TV(p, q) = 0.29); the question text repeats the numbers it needs.

### lossy_verify-Q1
- **Level:** 2. **Type:** explain why.
- **Question:** "A relaxed rule and the exact rule look at the same draft token x and use the same random coin. The relaxed rule accepts x and the exact rule rejects it. What must be true about q(x) and p(x), and why?"
- **Answer key:** q(x) > p(x). The exact rule accepts with probability min(1, p(x)/q(x)), which is 1 whenever q(x) ≤ p(x). So it can only reject tokens that the drafter likes more than the target does. Every token a relaxed rule adds is such an "overshoot" token.
- **Likely wrong answer:** "It must be a token with low q(x), a token the drafter was unsure about."
- **Follow-up:** "Take q(x) = 0.02 and p(x) = 0.30. What is the exact rule's acceptance probability for this token?" (It is 1. A low q does not cause rejection; q above p does.)

### lossy_verify-Q2
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "Each draft token is accepted independently with probability a, and a round drafts N = 6 tokens. A relaxed rule raises a from 0.60 to 0.70. By what factor do the tokens committed per round grow, counting the one extra token from the target? If completions also become 20% longer, are rounds saved?"
- **Answer key:** Tokens per round = (1 − a^7)/(1 − a). For a = 0.60: (1 − 0.0280)/0.40 = 2.430. For a = 0.70: (1 − 0.0824)/0.30 = 3.059. Ratio 1.259. Rounds ratio = λ/g = 1.20/1.259 = 0.953, so about 5% of rounds are saved. The break-even λ is 1.259.
- **Likely wrong answer:** "Acceptance rose by 0.70/0.60 = 1.167, so tokens per round grow 1.167 times, and with 20% longer output rounds go up."
- **Follow-up:** "With a = 0.60, what is the probability that the first three draft tokens are all accepted? With a = 0.70?" (0.216 against 0.343, ratio 1.59. Long accepted runs grow faster than a, so tokens per round are not linear in a.)

### lossy_verify-Q3
- **Level:** 3. **Type:** apply to a new case (numeric).
- **Question:** "A target with an EAGLE-3 drafter accepts ℓ̄* = 2.28 draft tokens per round under exact verification. Relaxed rules raise tokens per round by a factor 1.38, and completions become 1.49 times longer. For an exact completion of 1,000 tokens, how many rounds does each arm need, and what is the ratio?"
- **Answer key:** Exact: 1,000/(2.28 + 1) = 304.9 rounds. Relaxed: length 1,490, tokens per round 3.28 × 1.38 = 4.526, so 1,490/4.526 = 329.2 rounds. Ratio 1.49/1.38 = 1.080: 8% more rounds, although every round accepts more. (The numbers are the GPT-OSS-20B + EAGLE-3 row of Table 3; the measured mean R/R* there is 1.04, because a mean of ratios is not a ratio of means [checked: 2610.23818 Table 3].)
- **Likely wrong answer:** "Tokens per round are up 38%, so about 28% fewer rounds (1/1.38 = 0.72)."
- **Follow-up:** "How many tokens does the relaxed arm have to produce in total, and how many does one of its rounds commit?" (1,490 and 4.53. The learner divides and sees 329.)

### lossy_verify-Q4
- **Level:** 3. **Type:** explain why.
- **Question:** "A table compares CACTUS, fuzzy SD and the cascade token rule, all at α = 0.15. Why is this comparison not fair, and what would you match instead?"
- **Answer key:** α has a different meaning in each rule: a KL budget in nats for CACTUS, a JS-divergence threshold for fuzzy SD, a relative probability cut (1 − α)·max p for the token rule. The same number gives very different looseness. Match the rules on an outcome: mean accepted length ℓ̄ (what the paper does [checked: 2610.23818 Sec 2.2]) or a dose measure such as substitution share or excess surprisal. A full answer adds that the cascade rules are not exact at α = 0.
- **Likely wrong answer:** "It is fair if each α is rescaled to the range 0 to 1."
- **Follow-up:** "At α = 0, what does the token rule do with a draft token that is the target's argmax and has q(x) = 0.9, p(x) = 0.5?" (It accepts with probability 1; the exact rule accepts with 0.5/0.9 = 0.56. So α = 0 is not 'no relaxation' for this rule, and rescaling cannot fix that.)

### lossy_verify-Q5
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "One position has target p = [0.5, 0.3, 0.2] and drafter q = [0.7, 0.1, 0.2]. The cascade OPT rule trusts the drafter when max q ≥ max p − α·TV(p, q), with α = −0.5. Does it trust the drafter here? What is the acceptance probability, and what distribution is emitted?"
- **Answer key:** TV(p, q) = ½(0.2 + 0.2 + 0) = 0.2. Threshold = 0.5 − (−0.5)(0.2) = 0.6. max q = 0.7 ≥ 0.6, so it trusts q. Then π = q: the accept test is min(1, q/q) = 1 for every token, acceptance is 1.0 (exact: Σ min(p, q) = 0.8), and the emitted distribution is q = [0.7, 0.1, 0.2]. The dose is TV(r, p) = 0.2, equal to the acceptance gain 0.2.
- **Likely wrong answer:** "It trusts the drafter, so acceptance is higher, but the residual still corrects the output back to p."
- **Follow-up:** "If every draft token is accepted, how often is the residual sampled?" (Never. So nothing can correct anything; token 0 is emitted 70% of the time, not 50%.)

### lossy_verify-Q6
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "Target p = [0.50, 0.25, 0.12, 0.07, 0.05, 0.01], drafter q = [0.60, 0.10, 0.05, 0.03, 0.02, 0.20]. The cascade token rule with α = 0.55 trusts the set A = {v : p(v) ≥ (1 − α)·max p} and uses π(v) = q(v)·1[v ∈ A] + η·p(v), where η = 1 − Σ_{v∈A} q(v). The drafter proposes the third token (p = 0.12, q = 0.05). What is its acceptance probability under this rule and under the exact rule?"
- **Answer key:** Cut = 0.45 × 0.50 = 0.225, so A = {first, second}. η = 1 − (0.60 + 0.10) = 0.30. For the third token π = 0.30 × 0.12 = 0.036, so the acceptance probability is 0.036/0.05 = 0.72. The exact rule accepts it with probability min(1, 0.12/0.05) = 1. The "relaxed" rule is stricter here. Over the whole position its acceptance is 0.775 against 0.710 exact.
- **Likely wrong answer:** "A relaxed rule accepts at least what the exact rule accepts, so 1 in both cases."
- **Follow-up:** "How much probability does this rule give the first token in its output, and where does that mass come from?" (π(first) = 0.60 + 0.30 × 0.50 = 0.75, above both q and p. The mass is taken from the untrusted tokens, which are scaled by η = 0.3. That is why they are harder to accept.)

### lossy_verify-Q7
- **Level:** 4. **Type:** apply to a new case (numeric).
- **Question:** "The drafter proposes a tail token with p(x) = 0.01 and q(x) = 0.20. Compute the acceptance probability under (i) the exact rule, (ii) mentored decoding with α = 0.75, which accepts with min(1, p(x)/((1 − α)·q(x))), and (iii) CACTUS with α = 0.35, which accepts with min(1, γ_x/q(x)) where γ_x = min{p(x) + sqrt(2α·p(x)(1 − p(x))), 1}. Which rule accepts this tail token most often, and why?"
- **Answer key:** (i) 0.01/0.20 = 0.05. (ii) 0.01/(0.25 × 0.20) = 0.20. (iii) γ_x = 0.01 + sqrt(2 × 0.35 × 0.01 × 0.99) = 0.01 + 0.0832 = 0.0932, so 0.0932/0.20 = 0.466. CACTUS accepts it 9.3 times as often as exact. Mentored multiplies every token's chance by the same factor 1/(1 − α) = 4. CACTUS adds a term of order sqrt(p), which is huge relative to a small p. This matches Table B2: CACTUS-only tokens have median target probability 0.01 and mean rank 234 [checked: 2610.23818 Table B2].
- **Likely wrong answer:** "CACTUS has a KL budget of only 0.35 nats, so it must be the safest for unlikely tokens; mentored with α = 0.75 is the loosest."
- **Follow-up:** "For the top token with p(x) = 0.5, what is γ_x at α = 0.35, and what is γ_x/p(x)? Compare with the tail token." (0.918, ratio 1.84, against ratio 9.3 for the tail token. A KL budget is cheap to spend on rare tokens.)

### lossy_verify-Q8
- **Level:** 4. **Type:** explain why.
- **Question:** "A rule accepts the draft token with probability min(1, π(x)/q(x)) and on rejection samples from norm(max(0, π − q)), where π is a distribution. Show that the emitted token follows π. Then say which step fails for mentored decoding, where the accept test uses min(p/(1 − α), 1) but the residual uses p."
- **Answer key:** P(emit v) = q(v)·min(1, π(v)/q(v)) + P(reject)·max(0, π(v) − q(v))/Z, with Z = Σ max(0, π − q). The first term is min(q(v), π(v)). P(reject) = 1 − Σ min(q, π) = Σ (π − min(q, π)) = Z, because π sums to one. So P(emit v) = min(q, π) + max(0, π − q) = π(v). For mentored decoding, π_rej does not sum to one and the residual uses a different function, so P(reject) ≠ Z and the terms do not combine. The emitted law has three regions: mixed where q ≤ p, equal to q where p < q ≤ p/(1 − α), capped at p/(1 − α) above that.
- **Likely wrong answer:** "Mentored decoding emits min(p/(1 − α), 1) after normalizing."
- **Follow-up:** "With the toy pair and α = 0.75, the last token has p = 0.01, q = 0.20. How much mass does the mentored rule emit on it?" (Accepted mass = q × 0.2 = 0.04 = p/(1 − α); the residual max(0, p − q) is zero there. So 0.04, four times p. Normalizing min(p/0.25, 1) would give 0.04/3.0 = 0.013.)

### lossy_verify-Q9
- **Level:** 4. **Type:** connect two ideas (numeric).
- **Question:** "A position has TV(p, q) = 0.29. Rule E verifies against the mixture π = 0.6·p + 0.4·q. Compute its acceptance probability, its gain over exact, and its dose TV(π, p). Then: the cascade token rule on the same position gains 0.065 in acceptance with a dose of 0.25. Is that possible, and what does it tell you about the token rule?"
- **Answer key:** Rejection probability of a mixture = (weight on p) × TV(p, q) = 0.6 × 0.29 = 0.174, so acceptance = 0.826. Exact acceptance = 0.71, gain = 0.116. Dose TV(π, p) = 0.4 × 0.29 = 0.116. Gain equals dose: the mixture moves straight toward q. In general gain ≤ dose (idea 4), so 0.065 ≤ 0.25 is allowed. The token rule spends most of its TV on something else: it moves mass from the tail to the trusted top tokens, beyond q. That does not raise acceptance much, but it sharpens the output.
- **Likely wrong answer:** "A rule with dose 0.25 must gain more acceptance than a rule with dose 0.116, so the token-rule numbers must be wrong."
- **Follow-up:** "In the toy pair the token rule emits the top token with probability 0.75, while p gives 0.50 and q gives 0.60. Is the output between p and q?" (No. It is past q. TV measures distance, not direction.)

### lossy_verify-Q10
- **Level:** 4. **Type:** find the flaw.
- **Question:** "A paper runs a sampled-draft speculative decoder at temperature 0.7 with no truncation. It reports that typical acceptance gives more tokens per step than exact verification and slightly better accuracy. It concludes that typical acceptance is a free improvement. What is wrong with the comparison?"
- **Answer key:** Typical acceptance only lets through tokens above the threshold min(ε, δ·exp(−H(p))), so it also truncates the tail. Truncation alone can raise accuracy, so the reported gain mixes two effects: the acceptance rule and the truncation [checked: 2607.26627 Sec 3.2]. The fair baseline is exact verification with the same truncation (η-sampling) applied to the target. Against that baseline typical acceptance loses, because inside the allowed set it follows the drafter: 69.84 against 76.12 on MATH, with Llama-3.1-8B and EAGLE-3, averaged over the parameter grid [checked: 2607.26627 Table 3, Sec 5.1]. (The paper in the question is invented for this item. In the real Table 3, typical acceptance is also below the EAGLE-3 baseline with no truncation, 73.20.) A full answer also asks for completion length, which that comparison does not report.
- **Likely wrong answer:** "Nothing is wrong if the accuracy difference is statistically significant."
- **Follow-up:** "If the drafter puts 0.8 on a token where the target puts 0.5, and that token is in the allowed set, how often is it emitted when it is proposed?" (Always. So it is emitted with the drafter's frequency. A truncated target would give it 0.5/Z.)

### lossy_verify-Q11
- **Level:** 4. **Type:** apply to a new case (numeric).
- **Question:** "On one server a verification round costs 13 ms plus 2.85 ms for every token it commits. Exact verification commits 3.28 tokens per round. A relaxed rule commits 1.38 times as many tokens per round and needs 0.96 times the rounds. What is the ratio of total decoding time, relaxed over exact?"
- **Answer key:** Exact round: 13 + 2.85 × 3.28 = 22.35 ms. Relaxed round: 13 + 2.85 × 4.526 = 25.90 ms, ratio 1.159. Total time ratio = 0.96 × 1.159 = 1.113. The rule saves 4% of the rounds and loses 11% of the time. Reason: total time = c0·R + c1·L, and here L grew by λ = 0.96 × 1.38 = 1.325. (The paper's fit is c0 = 12 to 14 ms and c1 = 2.5 to 3.2 ms per token on its first server; 13 and 2.85 are mid-range values [checked: 2610.23818 App. C].)
- **Likely wrong answer:** "0.96. Time is proportional to the number of rounds."
- **Follow-up:** "On a second server c1 is 0.08 ms and c0 is 7.35 ms. Redo the round-time ratio." ((7.35 + 0.08 × 4.526)/(7.35 + 0.08 × 3.28) = 1.013, so time ratio 0.97. Whether rounds predict time depends on c1, which is a property of the server. The paper gives c0 = 7.1 to 7.6 ms and c1 at most 0.08 ms for that server; 7.35 is the mid-range and 0.08 is the upper bound [checked: 2610.23818 App. C].)

### lossy_verify-Q12
- **Level:** 4. **Type:** connect two ideas. **Ties to Part A.**
- **Question:** "For Part A you must tune three disturbances to 'the same dose': relaxed verification, sparse attention and quantization. Name one dose measure that you can compute for all three, say how you would compute it, and say what you lose if you match only the rate of changed tokens."
- **Answer key:** A common reference is needed: the full (dense, full-precision) model scoring the text that was really emitted. Per token, log p_full(x_t | emitted prefix), its rank, and the excess surprisal −ln p_full(x_t) − H(p_full). Under exact sampling the excess surprisal has mean zero, so its mean or running sum is a dose with a natural zero. This needs p_full to be the same distribution that the reference run samples from (same temperature, same truncation) [checked: 2607.11317 abstract and Sec 1 state it for log p(w_t) + H_t; it is also a one-line textbook fact]. For relaxed verification p_full is already computed in the verify pass. For sparse attention and quantization it needs a second, teacher-forced pass of the full model over the emitted text. Matching only the substitution rate loses how unlikely the changed tokens are: 7% of tokens at rank 0.9 and 24% at rank 234 are different doses [checked: 2610.23818 Table B2]. The rate also needs a coupling (the same random numbers in both arms) to be defined at all. A level-5 answer adds: the rescoring pass recomputes the KV entries exactly, so this dose sees only the token channel. That is the point of Part A: after matching it, any remaining difference in length is due to something the dose does not see.
- **Likely wrong answer:** "Match the acceptance rate, or the fraction of tokens that differ from the exact run."
- **Follow-up:** "Rule X changes 7% of tokens and they are the target's second choice on average. Rule Y changes 7% and they are at rank 200. Are X and Y at the same dose under your measure? Would you expect the same λ?" (Same rate, very different surprisal. The paper's data suggest no: the token rule is the X-like case and has mean λ = 0.96 [checked: 2610.23818 Sec 3.6]. Caution: 0.96 is a mean over cells. On GPT-OSS-20B with AIME24 the same rule has λ = 1.48 [checked: 2610.23818 Table 1]. So X-like tokens are not always harmless.)

### lossy_verify-Q13
- **Level:** 4. **Type:** predict. **Ties to Part A.**
- **Question:** "Exact speculative decoding at temperature 1.2 instead of 1.0 makes GPT-OSS-20B's GSM8K completions 1.27 times longer, with a 95% interval of 1.10 to 1.47. Nothing is approximated: the model sees all its context and the verification is exact. What does this result tell you about Part A, and what does it not tell you?"
- **Answer key:** Temperature is a pure dose disturbance: it changes which tokens are emitted (more unlikely ones) and nothing else. So a token-level dose is sufficient to cause inflation in this model [checked: 2610.23818 App. C]. It does not show that dose is the only cause. To test that, compare disturbances with different mechanisms at a matched dose and ask whether λ is equal within noise; as a noise scale, the seed-to-seed standard deviation of λ in the relaxed cells of this model has median 0.08 [checked: App. C; measured on relaxed cells, not on the temperature control]. Two cautions a strong answer gives: Qwen3-8B barely lengthens under the same temperature change (λ 1.02 to 1.04), so the dose-response curve is model-specific; and temperature flattens every position, while a relaxed rule adds only drafter-overshoot tokens, so "matched dose" must say which measure was matched.
- **Likely wrong answer:** "It proves that only the dose matters, so Part A is already answered."
- **Follow-up:** "Suppose sparse attention at the same mean excess surprisal gave λ = 1.60 on the same data. Would the temperature result predict that?" (No. It predicts about 1.27 if dose alone mattered. A clear gap is evidence for a second cause, which is what Part A looks for.)

### lossy_verify-Q14
- **Level:** 4. **Type:** apply to a new case. **Ties to Part B.**
- **Question:** "You run a relaxed rule for the first K tokens of a reasoning trace and then switch to exact verification. Which measurements after the switch separate 'each changed token has a one-time cost' from 'changed tokens do lasting damage', and which common metric cannot separate them?"
- **Answer key:** Measure what is left after the switch: the remaining length, the chance per token that the trace closes, and whether the run ends at the budget. Compare with exact runs at the same stage. One-time cost predicts that the remaining length returns to the exact baseline quickly, and that total extra length keeps growing with K, because each lossy token adds its own cost. Lasting damage predicts that the remaining length stays above baseline, and that the extra length shows up after the switch (Exercise 3 shows both shapes). The metric that cannot separate them is acceptance: after the switch the rule is exact, so acceptance and tokens per round go back to the exact values at once in both stories. Also accept a second answer: one total λ at a single K. Both stories can give the same value (Exercise 3: λ = 1.429 for both at K = 200). A good answer cites the hint in the data: on GSM8K 50% (GPT-OSS-20B) and 82% (Qwen3-8B) of the extra tokens come after the first correct answer is written [checked: 2610.23818 Sec 4, Table F2], which looks like "cannot close", not like many small detours.
- **Likely wrong answer:** "Watch the acceptance rate after the switch. If it recovers, there is no lasting damage."
- **Follow-up:** "After the switch, which rule decides acceptance, and does that rule depend on how many unlikely tokens are already in the prefix?" (The exact rule, min(1, p/q) at each position. It will read normal even in a derailed run, so it cannot be the test.)

### lossy_verify-Q15
- **Level:** 4. **Type:** connect two ideas. **Ties to Part C.**
- **Question:** "In relaxed speculative verification, a draft token that the exact rule would reject is accepted and committed. After that, are the target model's KV-cache entries for the prefix exact or approximate? What does your answer mean for which channel of harm this method can test?"
- **Answer key:** Exact, given the emitted tokens. The target ran a full forward pass over the draft tokens to verify them, so the KV entries of every committed token come from the full model on the true prefix. Only the choice of token was lossy. So relaxed verification is a pure token-channel disturbance: any harm must travel through the emitted tokens. It is the "lossy token, exact KV" arm of a 2×2 design over token source and KV source [from notes: card F07 design note]. Unverified sparse attention changes both: the token and the stored states. To isolate the KV channel one needs the opposite arm: exact token choice with approximate stored states.
- **Likely wrong answer:** "Approximate, because the drafter produced those tokens, so the cache is contaminated by the drafter."
- **Follow-up:** "Which model's forward pass writes the cache entry of an accepted token: the drafter's or the target's verification pass?" (The target's. The drafter's own states are not used by the target.)

### lossy_verify-Q16
- **Level:** 5. **Type:** find the flaw (numeric, adversarial). **Ties to "which tokens need verifying".**
- **Question:** "A proposed gate, at temperature 1.0: if the draft token is inside the target's top-p 0.95 nucleus, accept it with no coin; otherwise reject it and sample the replacement from p. Assume the target puts exactly 5% of its mass outside that nucleus, and the drafter also puts 5% of its mass outside it. The result is λ = 0.95 and the claim is 'only tail tokens need verifying'. How often does this decoder emit a token from outside the nucleus, how often does exact sampling, and which control is missing?"
- **Answer key:** A tail token is emitted only if the draft falls in the tail (0.05) and the replacement from p also falls in the tail (0.05): 0.05 × 0.05 = 0.0025, against 0.05 for exact sampling. That is 20 times fewer tail tokens. For tail tokens the gate is almost top-p 0.95 sampling. Truncation alone changes the output and may change its length, so λ = 0.95 cannot be credited to the gate yet. Rules that accept by an allowed set mix the effect of the rule with the effect of truncation [checked: 2607.26627 Sec 3.2]. The emitted law is r(v) = q(v)·1[v in nucleus] + 0.05·p(v). The cascade token rule would emit the same law with the nucleus as trusted set (the two procedures differ, the law is equal). Missing control: exact verification with top-p 0.95 on the target at the same temperature. Only a difference from that control can be credited to the gate. Second point: inside the nucleus the output follows q, so there is still a dose there, and it should be reported [from notes: card F07 raises this confound].
- **Likely wrong answer:** "5% in both, since tail tokens are still verified by the model. The control is exact decoding at temperature 1.0, which was run."
- **Follow-up:** "Under the gate, write the two events that must both happen for a tail token to be emitted, and multiply their probabilities." (Draft in tail, then replacement in tail: 0.0025.)

### lossy_verify-Q17
- **Level:** 5. **Type:** find the flaw (edge case, numeric).
- **Question:** "A relaxed rule commits 2.0 times as many tokens per round as exact verification. Its measured length ratio is λ = 1.48. But 99% of its runs stop at the token budget, against 33% of the exact runs. The reported round ratio is 0.75. A colleague wants to ship it: '25% fewer rounds'. What is wrong, and what can you say about the true round ratio?"
- **Answer key:** The lengths are censored. Almost every relaxed run was cut at the budget B, so the relaxed mean length is about B. Then λ = 1.48 is about B divided by the exact mean length. It measures the budget, not the rule. The round ratio 0.75 ≈ 1.48/2.0 = 0.74 has the same problem. The true round ratio is λ_true/2.0, and λ_true cannot be read from these runs. It is very likely above 1.48, because the relaxed arm is cut much more often than the exact arm (99% against 33%). The paper also says that cut-off runs make λ a lower bound [checked: 2610.23818 Sec 3.3]. Rounds are lost as soon as λ_true > 2. Grading note: "lower bound" is proved only when the exact arm is never cut. Here it is cut too, so "unknown, very likely higher" is the careful answer; accept "lower bound" as well. Also, a run cut at the budget may never give its answer: in this cell the relaxed accuracy was 0.00. The paper says of this exact case (spec_casc_opt, Qwen3-8B, LiveCodeBench, temperature 0.6): "an acceptance gain of 2.0× against an inflation that the budget cuts short, not a usable setting" [checked: 2610.23818 App. C]. Correct reporting: the share of runs at the budget in both arms, accuracy, and length statistics that treat cut runs as censored.
- **Likely wrong answer:** "Nothing is wrong; rounds were measured directly, so 0.75 is real."
- **Follow-up:** "If the budget were doubled, which of the three numbers (gain 2.0, λ 1.48, round ratio 0.75) would you expect to change?" (λ and the round ratio should rise; the gain should stay. So 0.75 is mostly a property of the budget, not of the rule.)

### lossy_verify-Q18
- **Level:** 5. **Type:** find the flaw (numeric).
- **Question:** "A per-token guard forces exact verification at chosen positions. It was tuned on 8 cases and cut their mean length from 13,364 to 10,534 tokens. On all 30 cases the mean length rose from 9,270 to 10,363 tokens. Without the guard, two of the 8 tuning runs had reached the budget of 32,768 tokens. What share of the tuning tokens did those two runs hold, and what does this say about tuning any 'which tokens to verify' rule?"
- **Answer key:** Tuning total = 8 × 13,364 = 106,912 tokens. The two capped runs hold 2 × 32,768 = 65,536, which is 61%. The paper reports the same 61%. (It does not say for which arm; the arithmetic matches the runs without the guard.) So the tuning signal is mostly "did these two runs close their trace or not". That is one yes-or-no event per run, not a smooth function of the guard. The −21% on 8 cases (10,534/13,364 = 0.79) did not transfer: +12% on 30 cases (10,363/9,270 = 1.12), with 22 then 23 correct answers of 30 [checked: 2610.23818 App. F; the guard was for spec_casc_tok at α = 0.3 on GPT-OSS-20B, AIME24]. Lessons: length ratios are heavy-tailed (the top tenth of cases holds 59% of the net extra tokens [checked: Sec 3.7]); tune and test on separate cases; report paired intervals over cases and the share of capped runs; expect per-run events (closing the trace) to dominate per-token effects.
- **Likely wrong answer:** "The guard works. The 30-case result is just noise; average over more seeds on the same 8 cases."
- **Follow-up:** "If one of the two capped runs had closed at 10,000 tokens by chance, what would the 8-case mean have been without any guard?" ((106,912 − 32,768 + 10,000)/8 = 10,518, about the same as the 'guard effect'. One run's luck is as large as the claimed gain.)

### lossy_verify-Q19
- **Level:** 5. **Type:** apply to a new case (edge case).
- **Question:** "The target decodes greedily, so p is one-hot on its top token m. The drafter still samples, with q = [0.9, 0.1] and m the first token. For mentored decoding, CACTUS, the cascade token rule and fuzzy SD (which uses π = q when the JS divergence of p and q, in nats, is below α; take α = 0.08), say which ones can still emit the second token."
- **Answer key:** Mentored: π_rej = min(p/(1 − α), 1) is still 0 on the second token, so it is never accepted, and the residual is p. No change. CACTUS: γ_x = p(x) + sqrt(2α·p(x)(1 − p(x))) = 0 when p(x) = 0. No change. (For the top token, p(x) = 1 gives γ_x = 1 and it is accepted, so the 0/0 in the other entries of π_rej is never used.) Token rule: the trusted set is {m}, η = 0.1, π = [0.9 + 0.1, 0] = p. No change. Fuzzy SD: JS(p, q) = 0.036 nats here. This is below α = 0.08, so π = q and the second token is emitted 10% of the time. (The paper sweeps α from 0.03 to 0.25 [checked: 2610.23818 Table A1]; at 0.03 the rule would stay exact here. I did not check which log base the implementation uses. In bits JS = 0.052, still below 0.08.) So rules that scale or bound p stay greedy, and a rule that swaps p for q does not. Typical acceptance also stays greedy at temperature 0 [checked: 2401.10774 Sec 2.3.1]. Leviathan et al. make the same point for lenience: at temperature 0 the plain lenience rule can no longer be used [checked: 2211.17192 App. A.5]. The general lesson: "dose" depends on the sampler, and a multiplicative rule has no effect where p is zero. (This is derived from the formulas; I did not check how each implementation handles temperature 0.)
- **Likely wrong answer:** "At temperature 0 every relaxed rule reduces to greedy, because there is no randomness left."
- **Follow-up:** "Under fuzzy SD with π = q, what is the accept probability min(1, π(x)/q(x)) for the second token?" (It is 1. The randomness comes from the drafter's own sampling.)

---

## 5. Exercises

All three were run in this session with CPU-only Python and numpy. The scripts are in `/tmp/claude-1025/-home-ryankert-l-kv-fetch-kv-fetch/b232f66b-c1cf-477e-b08d-10c5771f3def/scratchpad/lossy_verify_work/v2/`. The output blocks are the real output of those scripts.

Examiner re-run (2026-11-13 15:35 CST): the three code blocks were copied out of this file and run again with CPU-only Python. Each output is identical to the block pasted below, character for character.

### Exercise 1. Six rules on one position: what is emitted, and what is the dose

**Task.** Use the toy pair p = [0.50, 0.25, 0.12, 0.07, 0.05, 0.01], q = [0.60, 0.10, 0.05, 0.03, 0.02, 0.20]. Write one function that takes π_rej and π_res and returns the emitted law r and the acceptance probability of each draft token (idea 3). Run it for the exact rule, mentored (α = 0.75), CACTUS (α = 0.35), the cascade token rule (α = 0.55) and the cascade OPT rule (α = −0.5 and −0.1). Add typical acceptance (ε = 0.09, δ = 0.3) with the emitted law "q restricted to the allowed set". For each rule print: acceptance, gain over exact, TV(r, p), KL(r‖p), excess surprisal E_r[−ln p] − H(p). Then, with a shared coin, print the share of accepted tokens that only the relaxed rule accepts, their mean target probability and mean target rank, and the mass that only the exact rule accepts. Before running, the learner predicts: which rule has the largest TV, which has the most positive excess surprisal, and which rule rejects something the exact rule accepts. Pen-and-paper version: do only the mentored rule and the token rule by hand.

**Expected result.** (1) Gain ≤ TV(r, p) for every rule, with equality for mentored, CACTUS and OPT here. (2) The ranking by TV and the ranking by excess surprisal disagree: typical acceptance and the token rule have large TV (0.30, 0.25) but negative excess surprisal (they sharpen); OPT at α = −0.1 and CACTUS have positive excess surprisal. (3) Only the token rule and the typical gate have mass that the exact rule alone accepts. (4) CACTUS accepts the tail token (p = 0.01) with probability 0.466, against 0.05 exact and 0.20 mentored. (5) The mentored output matches the three-region closed form exactly, and its π_rej sums to 3.0. (6) OPT flips from "exact" to "emit q" between α = −0.5 and α = −0.1.

**How the tutor checks.** The learner's table must match the output below to three decimals. Then ask: "TV says typical acceptance is the largest dose. Excess surprisal says it is the most negative. Which one would you use to predict longer thinking, and why?" A level-4 answer says that TV has no direction, and that Table B2 points at unlikely tokens (low p, high rank), which excess surprisal sees and TV does not. A level-5 answer adds that a sharpening rule can still harm in another way (it follows the drafter inside the allowed set), so no single number is safe.

Note on one modelling choice: Medusa itself has no sampled draft distribution; the "typical" row uses the allowed-set view of 2607.26627 (Eq. 13). The mean rank and probability here are mass-weighted means on a 6-token toy; do not compare them with the medians in Table B2.

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)
p = np.array([0.50, 0.25, 0.12, 0.07, 0.05, 0.01])   # target (full model), sorted by rank
q = np.array([0.60, 0.10, 0.05, 0.03, 0.02, 0.20])   # drafter: overshoots on token 0 (top) and token 5 (tail)
tv = lambda a, b: 0.5 * np.abs(a - b).sum()
H = lambda a: -(a[a > 0] * np.log(a[a > 0])).sum()
kl = lambda a, b: (a[a > 0] * np.log(a[a > 0] / b[a > 0])).sum()

def emit(pi_rej, pi_res):
    """pi_rej(x), pi_res(x): relaxed targets when the drafted token is x.
    Returns the emitted law r and the acceptance probability of each draft token."""
    r = np.zeros_like(p); acc = np.zeros_like(p)
    for x in range(len(p)):
        acc[x] = min(1.0, pi_rej(x)[x] / q[x])
        r[x] += q[x] * acc[x]
        if acc[x] < 1:
            res = np.maximum(0.0, pi_res(x) - q)
            r += q[x] * (1 - acc[x]) * res / res.sum()
    return r, acc

def strict():
    return emit(lambda x: p, lambda x: p)
def mentored(al):                       # pi_rej = min(p/(1-al), 1), pi_res = p
    pr = np.minimum(p / (1 - al), 1.0)
    return emit(lambda x: pr, lambda x: p)
def cactus_pi(x, al):                   # boost only the drafted token x
    g = min(p[x] + np.sqrt(2 * al * p[x] * (1 - p[x])), 1.0)
    pi = (1 - g) / (1 - p[x]) * p; pi[x] = g
    return pi
def cactus(al):
    return emit(lambda x: cactus_pi(x, al), lambda x: cactus_pi(x, al))
def casc_tok(al):                       # trusted set A = {v: p(v) >= (1-al) max p}
    A = p >= (1 - al) * p.max()
    pi = np.where(A, q, 0.0) + (1 - q[A].sum()) * p
    return emit(lambda x: pi, lambda x: pi)
def casc_opt(al):                       # trust the drafter for the whole position, or not at all
    pi = q if q.max() >= p.max() - al * tv(p, q) else p
    return emit(lambda x: pi, lambda x: pi)
def typical(eps, delta):                # gate on p only; emitted law of 2607.26627 Eq. 13
    A = p > min(eps, delta * np.exp(-H(p)))
    return np.where(A, q, 0.0) / q[A].sum(), A.astype(float)

rules = [("strict", strict()), ("mentored a=0.75", mentored(0.75)), ("cactus a=0.35", cactus(0.35)),
         ("casc_tok a=0.55", casc_tok(0.55)), ("casc_opt a=-0.5", casc_opt(-0.5)),
         ("casc_opt a=-0.1", casc_opt(-0.1)), ("typical .09/.3", typical(0.09, 0.3))]
r0, acc0 = rules[0][1]; a0 = (q * acc0).sum()
print("TV(p,q)=%.3f  1-TV=%.3f  H(p)=%.4f nats  typical threshold=%.4f" % (tv(p, q), 1 - tv(p, q), H(p), min(0.09, 0.3 * np.exp(-H(p)))))
print("\n%-16s %5s %6s %7s %8s %8s  %s" % ("rule", "acc", "gain", "TV(r,p)", "KL(r||p)", "excess", "emitted law r"))
for name, (r, acc) in rules:
    a = (q * acc).sum(); xs = -(r * np.log(p)).sum() - H(p)
    assert abs(r.sum() - 1) < 1e-12 and a - a0 <= tv(r, p) + 1e-12
    print("%-16s %5.3f %6.3f %7.3f %8.3f %+8.3f  %s" % (name, a, a - a0, tv(r, p), kl(r, p), xs, r))

rank = np.arange(len(p))
print("\n%-16s %6s %7s %7s %7s %7s  %s" % ("rule", "share", "p_only", "p_both", "rk_only", "s_only", "accept prob per draft token"))
for name, (r, acc) in rules[1:]:
    a = (q * acc).sum()
    only = q * np.maximum(acc - acc0, 0)     # accepted by the relaxed rule only (same coin)
    both = q * np.minimum(acc, acc0)         # accepted by both
    s_only = (q * np.maximum(acc0 - acc, 0)).sum()   # accepted by the strict rule only
    po = (only * p).sum() / only.sum() if only.sum() > 0 else np.nan
    ro = (only * rank).sum() / only.sum() if only.sum() > 0 else np.nan
    print("%-16s %6.3f %7.3f %7.3f %7.2f %7.3f  %s" % (name, only.sum() / a, po, (both * p).sum() / both.sum(), ro, s_only, acc))

# closed form for the mentored / lenience rule (2607.26627 Eq. 10), l = 1 - alpha
l = 0.25
D = np.maximum(0, q - p / l).sum() / np.maximum(0, p - q).sum()
closed = np.where(q <= p, q + D * (p - q), np.where(q <= p / l, q, p / l))
print("\nmentored closed form: Delta=%.4f  max |closed - emit| = %.1e" % (D, np.abs(closed - rules[1][1][0]).max()))
print("pi_rej of mentored sums to %.2f (it is not a distribution)" % np.minimum(p / l, 1).sum())
x = 5; g = cactus_pi(x, 0.35)[x]
print("tail token p=%.2f q=%.2f: accept strict %.3f, mentored %.3f, cactus %.3f (gamma=%.4f)" % (p[x], q[x], p[x] / q[x], min(1, p[x] / l / q[x]), g / q[x], g))
```

Output:

```
TV(p,q)=0.290  1-TV=0.710  H(p)=1.3296 nats  typical threshold=0.0794

rule               acc   gain TV(r,p) KL(r||p)   excess  emitted law r
strict           0.710  0.000   0.000   -0.000   -0.000  [0.5  0.25 0.12 0.07 0.05 0.01]
mentored a=0.75  0.840  0.130   0.130    0.054   -0.040  [0.6   0.183 0.089 0.052 0.037 0.04 ]
cactus a=0.35    0.893  0.183   0.183    0.174   +0.103  [0.6   0.155 0.076 0.045 0.031 0.093]
casc_tok a=0.55  0.775  0.065   0.250    0.151   -0.376  [0.75  0.175 0.036 0.021 0.015 0.003]
casc_opt a=-0.5  0.710  0.000   0.000   -0.000   -0.000  [0.5  0.25 0.12 0.07 0.05 0.01]
casc_opt a=-0.1  1.000  0.290   0.290    0.529   +0.392  [0.6  0.1  0.05 0.03 0.02 0.2 ]
typical .09/.3   0.750  0.040   0.300    0.253   -0.449  [0.8   0.133 0.067 0.    0.    0.   ]

rule              share  p_only  p_both rk_only  s_only  accept prob per draft token
mentored a=0.75   0.155   0.387   0.400    1.15   0.000  [1.  1.  1.  1.  1.  0.2]
cactus a=0.35     0.205   0.277   0.400    2.27   0.000  [1.    1.    1.    1.    1.    0.466]
casc_tok a=0.55   0.129   0.500   0.417    0.00   0.035  [1.    1.    0.72  0.7   0.75  0.015]
casc_opt a=-0.5   0.000     nan   0.400     nan   0.000  [0.833 1.    1.    1.    1.    0.05 ]
casc_opt a=-0.1   0.290   0.179   0.400    3.28   0.000  [1. 1. 1. 1. 1. 1.]
typical .09/.3    0.133   0.500   0.432    0.00   0.060  [1. 1. 1. 0. 0. 0.]

mentored closed form: Delta=0.5517  max |closed - emit| = 0.0e+00
pi_rej of mentored sums to 3.00 (it is not a distribution)
tail token p=0.01 q=0.20: accept strict 0.050, mentored 0.200, cactus 0.466 (gamma=0.0932)
```

### Exercise 2. From acceptance to rounds to time

**Task.** (a) With i.i.d. per-token acceptance a and N_draft = 6, compute tokens per round for a = 0.60 and a = 0.70 and compare the ratio with 0.70/0.60. (b) For four pairs from Table 3 of 2610.23818 (exact ℓ̄*, gain g, mean λ), compute: the per-token acceptance that the i.i.d. model needs for the exact and the relaxed arm, the predicted round ratio λ/g, the ceiling (N_draft + 1)/(ℓ̄* + 1), and the time ratio on two servers with time per round = c0 + c1 × tokens committed (server 1: 13 ms and 2.85 ms, mid-range values of the paper's fits; server 2: 7.35 ms, mid-range, and 0.08 ms, the paper's upper bound). Compare λ/g with the measured mean R/R*. (c) A cell has R/R* = 0.96 and g = 1.38 on server 1 with ℓ̄* = 2.28. Compute λ and the time ratio. Pen-and-paper version: do (a), the first row of (b) and (c).

**Expected result.** (a) 2.430 and 3.059 tokens per round: ratio 1.259, not 1.167. (b) λ/g reproduces the measured mean round ratios to within 0.08 on the four pairs (1.08 against 1.04; 0.93 against 0.94; 0.72 against 0.71; 2.10 against 2.03). The Llama pair has λ = 3.30 above its ceiling 2.62, so no relaxation of that drafter could pay at that inflation. On server 1 a relaxed round costs 9 to 21% more than an exact one; on server 2 about 1% more. For Qwen3-8B + EAGLE-3 this turns a 7% saving in rounds into no saving in time on server 1. (c) λ = 1.325 and time ratio 1.113.

**How the tutor checks.** Numbers must match. Then ask: "The all-pairs row gives λ/g = 1.149 but the measured mean R/R* is 1.10. Is Eq. 5 wrong?" Correct answer: no. Eq. 5 holds per cell (within 2% at the median [checked: 2610.23818 Sec 3.4]). The table gives means over cells, and a mean of ratios is not the ratio of means. Also ask what the i.i.d. columns a* and a_rel are good for. Answer: only a rough reading. For the first pair they imply a relaxed-only share of (0.850 − 0.728)/0.850 = 14%, the same order as the 16 to 24% in Table B2, but real acceptance is not i.i.d. across positions.

Data used [checked: 2610.23818 Table 3 and App. C]: ℓ̄* / gain / mean λ / mean R/R* = 2.28 / 1.38 / 1.49 / 1.04 (GPT-OSS-20B + EAGLE-3 nebius); 1.20 / 1.27 / 1.18 / 0.94 (Qwen3-8B + EAGLE-3); 3.60 / 1.25 / 0.90 / 0.71 (Qwen3-8B + Qwen3-1.7B); 1.67 / 1.57 / 3.30 / 2.03 (Llama-3.1-8B-Instruct + EAGLE-3 yuhuili); all pairs: gain 1.34, mean λ 1.54, mean R/R* 1.10. The server-1 constants were fitted on the GPT-OSS-20B pair; using them for the other rows is an illustration, not a measurement.

```python
import numpy as np
N = 6                                             # draft tokens per round (2610.23818 Sec 3.1)
tpr = lambda a: (1 - a ** (N + 1)) / (1 - a)      # tokens per round = lbar + 1, i.i.d. acceptance a

# (a) acceptance rate is not tokens per round
for a in (0.60, 0.70):
    print("a=%.2f  lbar=%.3f  tokens per round=%.3f" % (a, tpr(a) - 1, tpr(a)))
print("acceptance ratio %.3f, tokens-per-round ratio %.3f" % (0.70 / 0.60, tpr(0.70) / tpr(0.60)))

def solve_a(t):                                   # per-token acceptance that gives t tokens per round
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tpr(mid) < t else (lo, mid)
    return lo

# (b) rounds, break-even, ceiling, time.  Rows: lbar*, gain g = (lbar+1)/(lbar*+1), mean lambda, measured mean R/R*
rows = [("GPT-OSS-20B + EAGLE-3 (nebius)", 2.28, 1.38, 1.49, 1.04),
        ("Qwen3-8B + EAGLE-3",             1.20, 1.27, 1.18, 0.94),
        ("Qwen3-8B + Qwen3-1.7B",          3.60, 1.25, 0.90, 0.71),
        ("Llama-3.1-8B + EAGLE-3",         1.67, 1.57, 3.30, 2.03),
        ("all twelve pairs",               None, 1.34, 1.54, 1.10)]
srv = {"server 1": (13.0, 2.85), "server 2": (7.35, 0.08)}   # c0 ms per round, c1 ms per committed token
print("\n%-31s %6s %6s %8s %8s %8s %7s %7s %7s %7s" % ("pair", "a*", "a_rel", "R pred", "R meas", "ceiling", "rt s1", "T s1", "rt s2", "T s2"))
for name, ls, g, lam, rmeas in rows:
    rpred = lam / g                               # Eq. 5: R_relax / R_strict = lambda / gain
    if ls is None:
        print("%-31s %6s %6s %8.3f %8.2f" % (name, "-", "-", rpred, rmeas)); continue
    t = ls + 1
    out = []
    for c0, c1 in srv.values():
        rt = (c0 + c1 * g * t) / (c0 + c1 * t)    # time per round, relaxed / strict
        out += [rt, rpred * rt]                   # total time ratio = rounds ratio x time-per-round ratio
    print("%-31s %6.3f %6.3f %8.3f %8.2f %8.2f %7.3f %7.3f %7.3f %7.3f" % (name, solve_a(t), solve_a(g * t), rpred, rmeas, (N + 1) / t, *out))

# (c) rounds saved, time lost: R/R* = 0.96 with gain 1.38 on server 1 (lbar* = 2.28)
c0, c1 = srv["server 1"]; t = 3.28; g = 1.38; rr = 0.96
lam = rr * g
print("\nR/R*=%.2f, gain %.2f -> lambda=%.3f; time ratio = (c0*R + c1*L)/(c0*R* + c1*L*) = %.3f"
      % (rr, g, lam, (c0 * rr + c1 * lam * t) / (c0 + c1 * t)))
```

Output:

```
a=0.60  lbar=1.430  tokens per round=2.430
a=0.70  lbar=2.059  tokens per round=3.059
acceptance ratio 1.167, tokens-per-round ratio 1.259

pair                                a*  a_rel   R pred   R meas  ceiling   rt s1    T s1   rt s2    T s2
GPT-OSS-20B + EAGLE-3 (nebius)   0.728  0.850    1.080     1.04     2.13   1.159   1.251   1.013   1.094
Qwen3-8B + EAGLE-3               0.553  0.662    0.929     0.94     3.18   1.088   1.011   1.006   0.935
Qwen3-8B + Qwen3-1.7B            0.856  0.934    0.720     0.71     1.52   1.126   0.810   1.012   0.729
Llama-3.1-8B + EAGLE-3           0.642  0.822    2.102     2.03     2.62   1.210   2.544   1.016   2.136
all twelve pairs                     -      -    1.149     1.10

R/R*=0.96, gain 1.38 -> lambda=1.325; time ratio = (c0*R + c1*L)/(c0*R* + c1*L*) = 1.113
```

### Exercise 3. Switch back to exact: one-time cost or lasting damage (ties to Part B)

**Task.** This is a toy, not evidence. A run is "on track" or "derailed". Each token ends the run with probability 1/200 when on track and 1/600 when derailed. During the first K tokens the method is lossy: each token is substituted with probability d = 0.2 (the dose), and a substituted token derails an on-track run with probability h. A derailed run recovers with probability ρ per token. Model "lasting": ρ = 0, h = 0.01. Model "transient": ρ = 0.05, and h is tuned so that both models give the same λ at K = 200. After token K the method is exact (no more substitutions). Compute, without sampling, for K = 25 to 800: λ = mean length / 200, the share of live runs that are derailed at the switch, and the mean remaining length after the switch. Before running, the learner predicts which column separates the two models.

**Expected result.** Both models give λ = 1.429 at K = 200, so one λ at one setting cannot tell them apart. The remaining length after the switch can: 367 tokens in the lasting model against 210 in the transient model (exact baseline: 200). The shape over K also differs. The lasting model saturates (1.535 at K = 400, 1.567 at K = 800) because a derailed run stays derailed. The transient model keeps growing (1.671, 1.914) because it pays while the lossy phase lasts. In the transient model the remaining length does not depend on K.

**How the tutor checks.** Numbers must match. Then ask: "In a real experiment you cannot see the hidden state. Which two measurements stand in for the 'remaining' column and the 'shape over K'?" Good answer: the length (or the chance of closing the trace) after the switch, compared with exact runs at the same stage, and a sweep over K at a fixed dose. A level-5 answer adds two problems the toy hides. Real lengths are heavy-tailed and censored at the budget, so use paired cases and report capped runs. And runs that are still alive at token K are a selected group, so the comparison group must be exact runs that are also still alive at K.

```python
import numpy as np
# Toy run: every token ends the run with probability s_good (on track) or s_bad (derailed).
# Lossy phase = first K tokens: each token is substituted with probability d (the dose);
# a substituted token derails an on-track run with probability h.
# A derailed run recovers with probability rho per token (rho = 0: damage is lasting).
s = np.array([1 / 200, 1 / 600]); d = 0.2; G = 1 / s[0]          # exact baseline: mean length G = 200

def B(rho):                                   # mean remaining length from "derailed", exact phase
    return (1 + (1 - s[1]) * rho * G) / (1 - (1 - s[1]) * (1 - rho))

def run(K, h, rho, T=40000):
    m = np.array([1.0, 0.0]); L = 0.0; after = None       # m = alive mass [on track, derailed]
    for t in range(T):
        if t == K: after = (m @ np.array([G, B(rho)])) / m.sum(); bad = m[1] / m.sum()
        L += m.sum()                                       # one more token is emitted
        m = m * (1 - s)                                    # the run may end
        go = d * h if t < K else 0.0
        m = np.array([m[0] * (1 - go) + m[1] * rho, m[0] * go + m[1] * (1 - rho)])
    return L, after, bad

def fit_h(rho, target, K=200):                # tune h so that both models have the same lambda at K
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if run(K, mid, rho)[0] < target else (lo, mid)
    return 0.5 * (lo + hi)

h_last = 0.01
target = run(200, h_last, 0.0)[0]
h_tran = fit_h(0.05, target)
print("lasting   model: rho=0.00 h=%.4f  mean length when derailed (exact phase) B=%.1f" % (h_last, B(0.0)))
print("transient model: rho=0.05 h=%.4f  mean length when derailed (exact phase) B=%.1f" % (h_tran, B(0.05)))
print("\n%5s | %8s %10s %10s | %8s %10s %10s" % ("K", "lam last", "derailed", "remaining", "lam tran", "derailed", "remaining"))
for K in (25, 50, 100, 200, 400, 800):
    a = run(K, h_last, 0.0); b = run(K, h_tran, 0.05)
    print("%5d | %8.3f %10.3f %10.1f | %8.3f %10.3f %10.1f" % (K, a[0] / G, a[2], a[1], b[0] / G, b[2], b[1]))
print("\nexact baseline: lambda 1.000, remaining length after any token = %.0f" % G)
```

Output:

```
lasting   model: rho=0.00 h=0.0100  mean length when derailed (exact phase) B=600.0
transient model: rho=0.05 h=0.8844  mean length when derailed (exact phase) B=212.9

    K | lam last   derailed  remaining | lam tran   derailed  remaining
   25 |    1.092      0.051      220.3 |    1.098      0.780      210.1
   50 |    1.168      0.103      241.1 |    1.155      0.782      210.1
  100 |    1.287      0.209      283.6 |    1.257      0.782      210.1
  200 |    1.429      0.417      366.8 |    1.429      0.782      210.1
  400 |    1.535      0.737      494.7 |    1.671      0.782      210.1
  800 |    1.567      0.964      585.5 |    1.914      0.782      210.1

exact baseline: lambda 1.000, remaining length after any token = 200
```

---

## 6. Diagnostic picks

- **Level 2-3: lossy_verify-Q6** (the token rule on the toy pair). It needs the template π, the trusted set and η, and it has a sharp wrong answer ("relaxed means at least as accepting"). A learner who gets 0.72 and can say why is at level 3 or above. A learner who cannot start is at level 2 or below; go to Q1 and Q5.
- **Level 4-5: lossy_verify-Q16** (the nucleus gate). It needs the emitted law of a gate (level 4), the number 0.0025 against 0.05, and the missing truncated-target control (level 5 design sense). A learner who finds the number but not the control is at level 4. If time allows, Q12 is the second check for Part A readiness.

---

## 7. Sources

Read in this run from copies of the arXiv HTML pages saved in the scratch folder on 2026-11-11 (title and id matched in each copy; no page was fetched again today):

| id | title (as shown on the page) | what I read |
|---|---|---|
| arXiv 2610.23818 (v1, 16 Oct 2026) | Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding (Wang, Xu, Gu, Zhang) | abstract, Sec 1-5, Limitations, App. A (Algorithm 1, Table A1), App. B (Tables B1, B2, B5), App. C, part of App. D, App. F. For the CACTUS formula I also checked the square root in the HTML math, because the text extraction dropped it. |
| arXiv 2405.19261 (v2, 21 Oct 2024) | Faster Cascades via Speculative Decoding (Narasimhan et al.) | Sec 4.1-4.4 (Algorithm 4, Lemmas 2-4, Eq. 6, 9, 10, 11, 15) |
| arXiv 2401.10774 (v3, 14 Jun 2024) | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (Cai et al.) | Sec 2.3.1, Sec 3.3.2, App. A, App. B.1 |
| arXiv 2607.26627 (v2, 04 Sep 2026) | Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes | abstract, Sec 2.1-2.3 (Eq. 1-6), Sec 3.1-3.2 (Eq. 7-13, Definition 1), Sec 4.1-4.2, start of Sec 5 (Sec 5.1, Table 3, statement of Theorem 1). Appendices not read. |
| arXiv 2607.08690 (v1, 09 Jul 2026) | A Practical Investigation of Training-free Relaxed Speculative Decoding | only Eq. 4 with its paragraph (length-adjusted speed-up, Takeaway 4), the paragraph around Finding 5, and the caption of Fig. 4 |
| arXiv 2607.11317 (v1, 13 Jul 2026) | Calibrated e-CUSUM Decoding for Quantized Reasoning Models: Why Token Log-Probability Is the Wrong Observable for Decoding Monitors | examiner only: abstract and Sec 1 (the mean-zero property of log p(w_t) + H_t) |
| arXiv 2211.17192 | Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman, Matias) | examiner only: App. A.5 (lenience) |

Read directly on disk:

| source | what I read |
|---|---|
| `research_notes/Length tax preliminaries/specdec_exact.md` | full (for the format, and to avoid overlap) |
| `research_notes/Inference speed question search/verification.md` | card F07 only (about 115 lines) |
| `reports/length-tax.html` | Question 1 in full, Question 2, start of Question 3 (about 16,000 of 22,951 characters of text) |

Cited from the notes only [from notes]: the card F07 design notes (the 2×2 over token source and state source, used in Q15; the nucleus-gate confound, used in Q16). The examiner read these lines on card F07. They are design notes, not paper results. 2607.11317 (used in Q12) was [from notes] in the first version; the examiner then read it in the saved copy, so it is now [checked]. The arithmetic in Q16 is derived here for the gate as the question states it.

Not used as evidence: `questions_ranked.md` and the `harvest_*.md` files (not opened in this run). Other papers named on card F07 (2610.07822 Nucleus SD, 2611.10634, 2511.00351, 2606.00206, 2611.12127, 2610.19026, 2506.09501) are not cited in this pack.

Could not verify:

- Medusa: the values ε = 0.09, δ = 0.3 in Exercise 1 are a choice for the toy, not checked as Medusa defaults [unverified]. Corrected by the examiner: Sec 3.3.2 of the page says "setting the α = sqrt(ε)" (the first text extraction lost the square root), with ε swept from 0.01 to 0.25. If that α is the δ of Sec 2.3.1, then δ = sqrt(0.09) = 0.3 fits the toy values. The page does not say that α is δ, so this reading is [unverified].
- How each rule's implementation behaves at temperature 0. Q19 is derived from the formulas only.
- 2607.11317: now read by the examiner (abstract and Sec 1). The property used (E_p[−ln p] = H(p), so excess surprisal has mean zero under exact sampling) is also a one-line textbook fact. What stays a proposal, not a tested fact: that mean excess surprisal is the right common dose for relaxed verification, sparse attention and quantization (Q12).
- Leviathan's own statement of lenience (2211.17192 App. A.5) was not read by the author. The examiner read it in a saved copy; it agrees with 2607.26627 Eq. 9 and with the cap p/ℓ.
- The Table B2 numbers were read from the text extracted from the HTML table, so they passed through one HTML-to-text step. Column meanings were taken from the table caption.
- The claim in Q15 that the target's KV entries are exact after a relaxed acceptance is stated from how speculative verification works in general (see `specdec_exact.md`); I did not read the vLLM code path of the five rules in 2610.23818.

---

## Verification log

Examiner pass finished 2026-11-13 15:36 CST (time from the `date` command). No GPU job was run. No page was fetched from the network: every cited paper has a saved HTML copy in the scratch folder, and each copy shows the title that matches its arXiv id. Only this file was changed.

**What was done.**

- Items examined: 52 (20 core ideas, 8 misconceptions, 19 questions, 3 exercises, 2 diagnostic picks).
- Every numeric answer key (Q2, Q3, Q5, Q6, Q7, Q8, Q9, Q11, Q16, Q18, Q19) was computed again with a separate CPU-only script. All numbers agree with the keys.
- The emitted laws of the mentored rule, CACTUS and the token rule on the toy pair were also checked by simulation (2 million draws each). They agree with Exercise 1 to three decimals.
- The three exercise code blocks were copied out of this file and run again. Each output is identical to the pasted block.
- Every [checked] paper fact was read again in the saved copy: 2610.23818 (Eq. 1 to 5, Sec 2.2, 2.3, 3.1, 3.3, 3.4, 3.6, 3.7, 4, Limitations, Tables 1, 3, A1, B2, App. C, App. F), 2405.19261 (Algorithm 4, Lemma 3, Eq. 10, 11, 15, Sec 4.4), 2401.10774 (Sec 2.3.1, 3.3.2), 2607.26627 (Eq. 9, 10, 13, Definition 1, Sec 3.1, 3.2, 5.1, Table 3), 2607.08690 (Eq. 4, Takeaway 4). Two more sources were opened for the first time: 2607.11317 (abstract, Sec 1) and 2211.17192 (App. A.5). Card F07 in `verification.md` was read for the two [from notes] items.

**No answer key had a wrong number.** The changes below fix wording that claimed too much, questions with more than one defensible answer, and tags.

**Changes.**

1. Idea 3: Algorithm 4 of 2405.19261 gives the procedure, not a theorem. The text no longer calls it a "result"; the proof is the derivation in Q8.
2. Idea 4: the tag [checked: Lemma 3] on the mixture formula was too strong. Lemma 3 covers only a yes-or-no choice between q and p. The mixture case is now derived in the text.
3. Idea 5: 2607.26627 Sec 3.1 says "overestimation" for q > p and keeps "severe overshoot" for q ≥ p/ℓ. The wording was fixed.
4. Idea 7: added Leviathan's own statement of lenience and of the cap p/ℓ [checked: 2211.17192 App. A.5].
5. Idea 10: added the implementation note that on some pairs cactus and spec_casc_tok keep π_res = p.
6. Idea 20(d) and Misconception 8: "hurt on all 30" was too strong. The mean length rose (9,270 to 10,363) but accuracy went from 22 to 23 of 30. The rule, α and model of the guard are now stated.
7. Misconception 4 and Q10: added that the Table 3 numbers are averages over a parameter grid, the tokens per step (4.61 against 3.93), and the untruncated baseline (73.20, marked "not directly comparable" by the paper). Q10 now cites Sec 3.2 for the truncation confound and says that the paper in the question is invented.
8. Q7: "most dangerous for tail tokens" was a matter of opinion. The question now asks which rule accepts the given token most often.
9. Q11 follow-up: 7.35 ms is a mid-range value and 0.08 ms is the paper's upper bound. This is now stated. The same fix was made in the task text of Exercise 2.
10. Q12: the mean-zero property is now [checked: 2607.11317], with the condition that p_full is the sampling distribution of the reference run. The follow-up now warns that the token rule's mean λ = 0.96 hides one cell with λ = 1.48 (GPT-OSS-20B, AIME24, Table 1).
11. Q13: the 0.08 seed standard deviation is the median over relaxed cells of GPT-OSS-20B, not a measurement on the temperature control. This is now stated.
12. Q14: the question had two defensible answers. The key now also accepts "one total λ at a single K". "Grows about linearly with K" was changed to "keeps growing with K".
13. Q16: the question now states that the target puts exactly 5% outside the nucleus. "Truncation alone can shorten a completion" was an unsupported claim; the key now says only that truncation changes the output and may change its length, with a source for the confound. The number 0.0025 did not change.
14. Q17: rewritten. "Doubles the tokens accepted per round" was ambiguous; the paper's 2.0 is tokens committed per round. "1.48 is a lower bound" is not proved when the exact arm is also cut (33% in the real cell), so the key now says "unknown, very likely higher" and accepts "lower bound". "A cut run usually gives no answer" was removed, because on GSM8K most cut runs already contain the answer [checked: 2610.23818 Sec 4].
15. Q18: removed "hard" (the paper says only "eight AIME24 cases") and "close to a coin flip" (not supported). Added the accuracy (22 to 23) and a note on which arm the 61% refers to. The wrong answer was reworded to match.
16. Q19: the question gave no α for fuzzy SD, so the answer depended on a missing number. It now gives α = 0.08 and says nats. Added the CACTUS edge case for the top token and Leviathan's remark on temperature 0.
17. Exercise 2, expected result: "within 0.07" changed to "within 0.08" (the largest gap is 2.102 − 2.03 = 0.072).
18. Section 7: added the two new sources. The Medusa note was corrected: the page says α = sqrt(ε), not α = ε.

Nothing was deleted.

**What stays unverified.**

- Medusa: ε = 0.09 and δ = 0.3 are toy values, not checked as defaults. That the α of Sec 3.3.2 is the δ of Sec 2.3.1 is a reading, not a statement on the page.
- Q19: how each implementation behaves at temperature 0, and the log base of the JS test in r_fuzzy. The key is derived from the formulas only.
- Q15: that the target's KV entries are exact after a relaxed acceptance. This follows from how verification works in general. The vLLM code of the five rules was not read.
- Q15 and Q16: the 2×2 design and the nucleus-gate confound are [from notes] (card F07). They are design arguments, not paper results.
- Q12: that mean excess surprisal is the right common dose for the three disturbances is a proposal. No source tests it.
- Q17: the true length ratio of the censored cell is not known.
- Q18: the paper does not say in which arm the two capped runs held 61% of the tokens. The arithmetic fits the runs without the guard.
- 2607.26627: the appendices were not read. The Table 3 values are averages over a parameter grid; single settings may differ.
- Exercise 3 is a toy model. Its numbers are exact for the toy and are not evidence about real models.
- Table B2 was read from the HTML table through a text extraction (twice, with two extractors, same values).
