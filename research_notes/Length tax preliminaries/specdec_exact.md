# Tutor pack: exact speculative decoding

Subskill id: `specdec_exact`. Written 2026-11-13 15:13 CST. This version replaces the one from 2026-11-11. No GPU job was run. All arithmetic and all exercise outputs below were produced again in this run with CPU-only numpy. A second, independent examiner pass was made on 2026-11-13; its changes are listed in the Verification log at the end.

Scope: the accept/reject rule and why it is lossless; multi-token drafts and the expected number of accepted tokens; tree and self-speculative variants; what the verification pass computes and writes.

Notation in this file (same as Leviathan et al.): **p** is the target model's next-token distribution, **q** is the draft's. Chen et al. use the letters the other way round (q = target, p = draft) [checked: 2302.01318 Sec 4.2]. Always ask "which one is the target?" before reading a formula. ReLU(z) = max(0, z). norm(w) = w / Σ_x w(x). TV(p, q) = ½ Σ_x |p(x) − q(x)| is the total variation distance.

How to read the tags:

- **[checked]**: I downloaded the raw arXiv HTML page in this run and read the passage myself.
- **[checked, saved copy]**: read in this run in raw arXiv page text that an earlier search saved on disk on 2026-11-11. The title on the saved page was confirmed. The page was not fetched again.
- **[checked, local code]**: I read it in this run in `/home/ryankert_l/SparseSpec/serve/`.
- **[from notes]**: taken from `research_notes/Inference speed question search/` or `reports/length-tax.html`; the paper was not opened in this run.
- **[unverified]**: from memory only, or an inference that no source confirms.
- No tag: textbook fact, or a derivation written out here and checked with numpy.

**Learner hook.** In the session the learner described the residual in their own words: "It is like relu layer, but the values of relu layer is like 0." Build on this. The exact statement is: residual = ReLU(p − q), then normalize. The ReLU output is 0 on every token where q(x) ≥ p(x) (fact 2, Q21).

---

## 1. Core ideas

1. **Setup.** At one position, the target gives p(x) and the draft gives q(x) over the vocabulary. Temperature, top-k and top-p are applied first, so p and q are the distributions that are really sampled from ("standard sampling from an adjusted probability distribution") [checked: 2211.17192 Sec 2.2; 2302.01318 Sec 4.2].

2. **The rule.** Draw x ~ q and u ~ Uniform(0,1), independent of x. Accept x if u ≤ p(x)/q(x), that is, with probability min(1, p(x)/q(x)). If x is rejected, draw the output from the **residual distribution** p'(y) = ReLU(p(y) − q(y)) / Σ_z ReLU(p(z) − q(z)) [checked: 2211.17192 Sec 2.3 and Algorithm 1; 2302.01318 Sec 4.2, written there as (q − p)_+]. The residual is zero on every token the draft proposes at least as often as the target wants.

3. **Why it is lossless.** Let β = Σ_x min(p(x), q(x)). Since ReLU(p − q) = p − min(p, q), the normalizer is Σ_z ReLU(p(z) − q(z)) = 1 − β. Then P(output = x) = q(x)·min(1, p(x)/q(x)) + (1 − β)·p'(x) = min(p(x), q(x)) + ReLU(p(x) − q(x)) = p(x) [checked: 2211.17192 Appendix A.1; 2302.01318 Theorem 1].

4. **Acceptance probability.** P(accept) = Σ_x q(x)·min(1, p(x)/q(x)) = Σ_x min(p(x), q(x)) = β = 1 − TV(p, q). Leviathan et al. call this distance D_LK and define α = E[β], the mean over prefixes [checked: 2211.17192 Def 3.1, Def 3.2, Lemma 3.3, Theorem 3.5, Corollary 3.6]. Measured α for T5-XXL (EnDe) with a T5-small draft: 0.75 at T = 0 and 0.62 at T = 1 [checked: 2211.17192 Tables 2 and 3].

5. **The rule is optimal for one draft token.** For any joint way to draw X ~ q and Y ~ p, P(X = Y = x) ≤ min(p(x), q(x)), so P(X = Y) ≤ Σ_x min(p(x), q(x)) = β. No exact rule can keep a single draft token more often than this one.

6. **Greedy special cases.** (a) Both greedy (T = 0): p and q are one-hot. The draft token is accepted only when it equals argmax p, and the replacement is argmax p. (b) Greedy draft, sampled target: q is one-hot on the draft token x_d. So x_d is accepted with probability p(x_d), and the residual is p with x_d set to zero, renormalized. Case (b) is the `NO_DRAFT_PROBS` path in the SparseSpec sampler: the draft probability is set to 1 and the draft token is masked out of the target probabilities before the recovered token is drawn [checked, local code: `sampling/rejection_sampler.py`].

7. **One round with γ draft tokens.** The draft writes x_1 … x_γ one after another. One target pass over the prefix plus the draft gives γ+1 distributions p_1 … p_{γ+1}. Let n be the number of draft tokens accepted before the first rejection. The round emits x_1 … x_n and one more token t. If n < γ, t comes from norm(ReLU(p_{n+1} − q_{n+1})). If n = γ, t comes from p_{γ+1} (the "bonus" token). So each round emits between 1 and γ+1 tokens [checked: 2211.17192 Algorithm 1; 2302.01318 Algorithm 2].

8. **Everything after the first rejection is thrown away.** Draft tokens x_{n+2} … x_γ and the target distributions p_{n+2} … were all computed with the rejected token x_{n+1} in their context. They say nothing about the text that now continues from the replacement token.

9. **Expected tokens per round (i.i.d. model).** If each draft token is accepted independently with probability α, then P(n ≥ k) = α^k for k ≤ γ. So E[n] = Σ_{k=1..γ} α^k = α(1 − α^γ)/(1 − α), and E[tokens per round] = E[n] + 1 = (1 − α^{γ+1})/(1 − α) [checked: 2211.17192 Eq. 1; "a capped geometric variable"]. The paper calls the i.i.d. assumption "only an approximation" [checked: 2211.17192 Appendix A.3].

10. **Without the i.i.d. model.** Let a_i be the probability that draft token i is accepted, given that tokens 1 … i−1 were accepted. Then E[tokens per round] = 1 + Σ_{k=1..γ} Π_{i=1..k} a_i. Early positions count more than late ones. "Acceptance at position k" can mean two different numbers. The *conditional rate* is a_k. The *reach rate* is Π_{i≤k} a_i: the share of all drafted tokens at position k that end up accepted. The reach rate falls with k even when every a_i is the same. Example: S3R (sparse self-speculation, draft length 6, 5% KV budget, T = 1.0, Qwen3-8B) says "acceptance falls from 95.8% for the first proposed token to 67.9% for the sixth", and gives 81.9% for the whole checkpoint [checked, saved copy: 2610.13202 Appendix E.2; setup in Sec 6.1]. The paper does not say which of the two numbers this is. The reach reading fits its own total: the mean of 95.8 and 67.9 is 81.85. That these are reach rates is my inference [unverified]. Under the conditional reading, accepted/drafted would be about 0.64 (my arithmetic, Exercise 2 tutor note). Under the reach reading, the conditional rate after position 1 is about (0.679/0.958)^(1/5) = 0.93 per step, only a little below 0.958. So do not quote 67.9% as a conditional rate. Chen et al. see the share of accepted tokens fall with a longer draft, "since later tokens depend on the acceptance of previous tokens" [checked: 2302.01318 Sec 6.3]. That is the same reach effect. It happens with constant a_i too, so it is not evidence about a_i. Direct evidence that the per-position rate falls with depth comes from EAGLE: its acceptance rate after n predicted features is 0.82, 0.77, 0.72, 0.69, 0.71 for n = 0 … 4 (Vicuna 7B, HumanEval, T = 0) [checked: 2401.15077 Table 8]. These cannot be reach rates, because a reach rate can never go up (0.69 to 0.71).

11. **Speedup.** Let c = (time of one draft step)/(time of one target step). If the verification pass costs one target step, speedup = (1 − α^{γ+1}) / ((1 − α)(γc + 1)) [checked: 2211.17192 Def 3.7, Theorem 3.8]. If α > c, some γ gives a gain, of at least (1 + α)/(1 + c) [checked: 2211.17192 Corollary 3.9]. If the verification pass costs v(γ) target steps instead of 1, replace (γc + 1) by (γc + v(γ)) (my generalisation). The total number of operations goes up, by the factor (1 − α)(γĉ + γ + 1)/(1 − α^{γ+1}), where ĉ is the draft/target ratio of arithmetic operations [checked: 2211.17192 Theorem 3.11].

12. **"Acceptance rate" means different things.** (i) The per-token conditional rate α of facts 4 and 9. (ii) Accepted/drafted = E[n]/γ. (iii) Mean accepted length E[n]. SparseSpec uses (ii): its cost model counts α·k + 1 tokens per round, "with an acceptance rate α and a bonus token" [checked: 2512.01278 Sec 2.3, 3.2]. S3R uses (ii) too: "the fraction of proposed tokens accepted", with τ = 1 + kα tokens per round [checked, saved copy: 2610.13202 Appendix D and Appendix F.3, Eq. 24]. EAGLE defines α in words that read like (ii), "the ratio of accepted to generated tokens during drafting", measured on chain drafts [checked: 2401.15077 Sec 4]. But the numbers it reports are split by depth (n-α, "considering n features predicted by the draft model"), and they are not reach rates (fact 10). Read them as per-position rates, closer to (i). One more paper uses "TAR" for "tokens accepted per iteration", which is (iii) [checked, saved copy: 2402.01528 Sec 3.3]. SparseSpec reports "an average acceptance token length of 6.16 out of 8 tokens", bonus token not counted [checked: 2512.01278 Sec 5.3]. That is 0.77 in sense (ii), but α ≈ 0.94 in sense (i) under the i.i.d. model (my arithmetic, Exercise 2c).

13. **Trees with sampled children (multi-round rule).** A tree node has several candidate children. Each candidate x_i was sampled from a draft distribution q_i. Test them one at a time. Accept x_i with probability min(1, p_cur(x_i)/q_i(x_i)). On rejection set p_cur ← norm(ReLU(p_cur − q_i)) and try the next one. If all are rejected, sample from p_cur. Each step is a valid speculative step for the current residual target, so the output is still exactly p [checked: 2305.09781 Sec 4.3, Algorithm 2 and the statement of Theorem 4.2; 2401.15077 Sec 3.3 and Appendix A.2 Algorithm 1]. SpecInfer compares this with "naive sampling": sample from the target and check if the token is a child. Both are exact; the multi-step rule rejects less often [checked: 2305.09781 Sec 4.3, statement of Theorem 4.3] and verifies 1.2-1.3× more tokens in their test [checked: 2305.09781 Table 3].

14. **Trees with deterministic children.** The q in the ratio must be the distribution the candidate was *really drawn from*. If the children are fixed tokens c_1 … c_k (for example the draft's top-k), that distribution is one-hot. Then c_1 is accepted with probability p(c_1), the residual is p without c_1, renormalized, and c_2 is accepted with probability p(c_2)/(1 − p(c_1)). In total P(some child accepted) = Σ_i p(c_i): the target mass the children cover. (My derivation; checked by exact enumeration in `checks.py`.) If the code picks top-k children but puts the draft's soft probability into the ratio, the output is biased (Q20). EAGLE says it selects the top k tokens in the greedy setting and samples k tokens in the non-greedy setting [checked: 2401.15077 Appendix A.1]. For fixed children, "naive sampling" (fact 13) also accepts with probability Σ_i p(c_i). So the gain of the multi-round rule exists only for sampled children (my derivation). A 2026 paper states the coverage fact as a bound. It proves that "one plus the resulting target coverage is a sharp upper bound on the expected output-block length, including the bonus token, of any exact path verifier" [checked, saved copy: 2610.11750 abstract; I did not read its definition of target coverage].

15. **How a tree is verified in one pass.** All tree tokens go into one target forward pass. A topology-aware causal mask lets each token attend only to the prefix and to its own ancestors. This "computes the exact same attention output as incremental decoding" [checked: 2305.09781 Sec 4.2]. SpecInfer orders the tokens depth-first so that the branches can share one KV cache [checked: 2305.09781 Sec 4.2]. After the rule picks one path, the KV entries of all other branches are useless and must be dropped (my statement; the text I read does not describe this clean-up step).

16. **EAGLE.** The draft is a small "Autoregression Head" (one FC layer and one decoder layer). It predicts the target's next second-to-top-layer feature from two inputs: the feature sequence, and the token sequence advanced by one step. The target's own LM head turns each predicted feature into a token distribution [checked: 2401.15077 Sec 3.1]. The draft is a tree, verified with tree attention in one pass and with the multi-round rule of fact 13 [checked: 2401.15077 Sec 3.3, Appendix A.2]. The paper also says: "we document accepted tokens and their features for use in the next drafting phase" [checked: 2401.15077 Sec 3.3]. So the next draft starts from the target's true features.

17. **Self-speculation.** The draft is the target's own weights, run more cheaply. SparseSpec: "After every k draft steps with sparse attention, a verification step with full attention is performed." The attention scores are dumped during verification, and the draft attends to the top-K tokens by these scores [checked: 2512.01278 Sec 4.1]. The main tests use k = 8 and sparsity ratio s = 0.05 [checked: 2512.01278 Sec 5.4]. Other cheap versions of the same model: skipped layers (Draft & Verify, "selectively skipping certain intermediate layers during drafting" [checked: 2309.08168 abstract]; LayerSkip, early exit at earlier layers [checked, saved copy: 2404.16710 abstract]) and a low-bit copy (QuantSpec, "a hierarchical 4-bit quantized KV cache and 4-bit quantized weights" [checked, saved copy: 2502.10424 abstract]).

18. **What the verification pass computes and writes.** Input: the last committed token plus the γ draft tokens (γ+1 tokens). Output: γ+1 target distributions. A forward pass also writes K and V for every input token at every layer, so the pass leaves target-computed KV entries for those γ+1 positions. After the rule, the entries of draft tokens n+1 … γ must be removed. The replacement or bonus token has no KV entry yet; it is the first input of the next round. One detail for Part C: K and V at layer L depend on the hidden state entering layer L. So a draft step with approximate attention writes exact KV at the first layer and approximate KV at every later layer.

19. **SparseSpec's version of fact 18.** A verify request allocates one new cache slot and feeds k+1 tokens, the last k+1 positions of the request's cache. Afterwards it retracts (k + 1 − number of output tokens) entries, where the output tokens include the recovered or bonus token [checked, local code: `scheduler/spec_scheduler.py`, linear VERIFY branch and `verify_outputs`]. The attention layer appends K and V on every forward pass [checked, local code: `model/model.py` line 713]. So with n accepted tokens, n+1 entries stay (last committed token and the n accepted drafts), all written by the full-attention pass. That the append overwrites the draft-written entries is an inference, not a traced fact [unverified]. Two things support it. First, the position arithmetic above. Second, the append positions come from `flashinfer.get_batch_indices_positions` with the full cache length as `seq_lens`, so an append of k+1 tokens addresses the last k+1 slots [checked, local code: `request/kv_cache_ptr/base.py`, `pillar.py`]. The kernel write itself was not traced. S3R, a similar system, says it in words: "Verification replaces provisional draft states with recomputed dense KV entries, and removes the unused candidate suffix" [checked, saved copy: 2610.13202 Sec 4]. The sampler: the accept test is `target_prob / draft_prob >= uniform_prob`; the recovered token is argmax of ReLU(target − draft) divided by Exp(1) noise, which samples in proportion to the weights without normalizing [checked, local code: `sampling/rejection_sampler.py`; the sampling claim is checked by simulation in `checks.py`].

20. **What "lossless" promises, and the first step away from it.** The output has the same *distribution* as the target, "within hardware numerics" [checked: 2302.01318 abstract and Sec 4.2]. It is not the same token sequence as a plain run with the same seed. A relaxed rule breaks the promise. With a lenience factor l in (0, 1], the draft token is accepted with probability min(1, p(x)/(l·q(x))). The acceptance probability becomes Σ_x min(p(x)/l, q(x)), and the only guarantee left is that no token is emitted with probability above p(x)/l; "there is no guarantee on the minimum probability" [checked: 2211.17192 Appendix A.5]. Measured α for T5-small on EnDe at l = 1, 0.5, 0.3, 0.1: 0.62, 0.71, 0.76, 0.84 [checked: 2211.17192 Table 5]. In the limit "accept everything", the emitted token follows q. It must differ from an exact sample with probability at least TV(p, q) = 1 − β (fact 5). So the rejection rate of the exact check is a *lower bound* on the per-position change rate of the unchecked method. The bound is reached when the two runs are coupled in the best way, for example when the exact run is the speculative run that uses the same draft token. It is also reached at T = 0, where both distributions are one-hot. If the two tokens are sampled independently, they differ more often: with probability 1 − Σ_x p(x)·q(x).

---

## 2. Level ladder

- **Level 2.** Can state the accept rule and say that a rejected token is replaced from "ReLU(p − q), normalized". Can follow the losslessness proof when it is shown. Knows that one round gives 1 to γ+1 tokens.
- **Level 3.** Can compute β, the residual, the tokens per round and the speedup from given numbers. Can say why the ReLU is zero where q ≥ p. With a hint, can explain the greedy cases and why tokens after the first rejection are dropped.
- **Level 4.** Derives the losslessness proof and the tree rule without help. Converts between the different "acceptance" numbers. Says exactly which tokens and which KV entries a given system leaves behind after a round. Computes the emitted distribution of a relaxed rule.
- **Level 5.** Finds subtle exactness bugs: shared random numbers, a sampler filter applied in the wrong place, the wrong q in a tree, numerics at near-ties. Designs the token-only and KV-only repairs and the noise control for the length-tax experiment, and can explain each choice to someone else.

---

## 3. Misconceptions

1. **"Lossless means the same text as the plain run with the same seed."** Only the distribution is the same. With T > 0 the speculative run uses different random draws (draft sample, uniform u, residual sample), so the text differs. Only at T = 0 should the tokens match, and even then near-ties can flip.

2. **"After a rejection, sample the replacement from the target p."** That counts the draft-favoured tokens twice. The output becomes min(p, q) + (1 − β)·p, which is not p (Exercise 1: TV 0.0875 from p in the toy case).

3. **"Acceptance rate = accepted tokens / drafted tokens = per-token α."** Tokens after the first rejection are never tested, but they count as "not accepted". With α = 0.8 and γ = 5 the ratio is 0.54 (Exercise 2a). Papers use both meanings (fact 12).

4. **"A stronger draft model is accepted more often."** Acceptance is 1 − TV(p, q). It measures how close q is to p, not how good q is. One paper reports that draft accuracy on language modeling "does not correlate strongly with its performance in speculative decoding": "a draft model with higher accuracy on language modeling task can have similar TAR to a model with lower accuracy" [checked, saved copy: 2402.01528 abstract and Sec 1; TAR there is tokens accepted per iteration]. Another reports that "even powerful draft models such as GPT-4o, as well as human text cannot achieve high acceptance rates under the standard verification scheme" [checked, saved copy: 2501.19309 abstract]. In practice a larger draft from the same family is often closer to the target, so its acceptance is often higher. The point is that closeness to p decides, not quality.

5. **"Lower temperature always raises acceptance."** As T → 0, β goes to 1 where the two argmax tokens agree and to 0 where they differ. The average usually rises (T5-small on EnDe: 0.62 at T = 1, 0.75 at T = 0 [checked: 2211.17192 Table 3]), but a single position can fall from 0.85 to 0.24 (Q9). Even the average does not always rise: the GPT-like pair in the same table has 0.88 at T = 0 and 0.89 at T = 1 [checked: 2211.17192 Table 3].

6. **"In self-speculation the accepted tokens still sit on the draft's approximate KV entries."** Not in SparseSpec: the verification pass processes the draft tokens again with full attention at the same cache positions (fact 19). Whether approximate KV survives is a design choice. Part C of the research question varies exactly this.

7. **"A relaxed check only lets a few draft tokens through; the rest of the text is still normal target text."** Every later token is sampled from the target *given a prefix the target would rarely have written*. Exact sampling after the switch is exact only relative to that prefix.

8. **"In a tree, always put the draft's probability q(x) in the ratio."** Only if x was really sampled from q. For a child picked by argmax or top-k the true proposal is one-hot, and the correct acceptance probability is p(x), not min(1, p(x)/q(x)) (fact 14, Q20).

---

## 4. Question bank

Each entry: level, type, question, answer key, likely wrong answer, follow-up. Numeric answers were re-computed in this run (`checks.py` in the scratch folder).

### specdec_exact-Q1
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "Three tokens A, B, C. The target gives p = [0.6, 0.3, 0.1] and the draft gives q = [0.3, 0.3, 0.4]. The draft samples one token from q. What is the probability that it is accepted? When it is rejected, from which distribution is the replacement drawn?"
- **Answer key:** β = Σ min(p, q) = 0.3 + 0.3 + 0.1 = 0.7. Residual: ReLU(p − q) = [0.3, 0, 0], normalized = [1, 0, 0]. The replacement is always A.
- **Likely wrong answer:** "0.7, and the replacement is sampled from p."
- **Follow-up:** "If the replacement came from p, what would be the total probability of emitting C? Compare it with p(C) = 0.1." (0.1 + 0.3 × 0.1 = 0.13.)

### specdec_exact-Q2
- **Level:** 2. **Type:** explain why.
- **Question:** "The draft proposes γ = 4 tokens and the target runs one forward pass. Why can this round emit 5 tokens and not only 4?"
- **Answer key:** The pass reads the last committed token and the 4 draft tokens. Each input position outputs the distribution of the *next* token, so there are 5 distributions p_1 … p_5. If all 4 draft tokens pass, p_5 is the target's distribution after x_4, and one token is sampled from it at no extra cost.
- **Likely wrong answer:** "It cannot; the fifth token needs a second target pass."
- **Follow-up:** "In the verification pass the target reads x_4 as an input. What does its output at that position predict?"

### specdec_exact-Q3
- **Level:** 3. **Type:** apply to a new case (numeric).
- **Question:** "Each draft token is accepted independently with probability 0.8. The draft length is 4. One draft step costs 0.1 of a target step, and the verification pass costs one target step. How many tokens does a round emit on average, and what is the speedup?"
- **Answer key:** Tokens = (1 − 0.8^5)/(1 − 0.8) = (1 − 0.32768)/0.2 = 3.3616. Cost = 4 × 0.1 + 1 = 1.4 target steps. Speedup = 3.3616/1.4 = 2.40.
- **Likely wrong answer:** "0.8 × 4 + 1 = 4.2 tokens, so 4.2/1.4 = 3.0×."
- **Follow-up:** "What is the probability that draft token 3 ends up in the output? Is it 0.8?" (0.8^3 = 0.512.)

### specdec_exact-Q4
- **Level:** 3. **Type:** explain why.
- **Question:** "Greedy decoding, 6 draft tokens. The target rejects draft token 3. In the same verification pass, the target's argmax at position 5 happens to equal draft token 5. Why may you not keep draft token 5?"
- **Answer key:** The target's output at position 5 was computed with the rejected token 3 (and token 4 that followed it) in the context. The real text now continues from the replacement at position 3, so that output is for a different prefix. Only tokens before the first rejection have a valid context.
- **Likely wrong answer:** "You can keep it, because it matches the target's choice at that position."
- **Follow-up:** "Which token was at position 3 of the input when the target computed its output for position 5?"

### specdec_exact-Q5
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "The draft is greedy: it always proposes its top token x_d, with no randomness. The target samples at temperature 1. It gives p(x_d) = 0.55, and 0.25 and 0.20 to two other tokens. With what probability is x_d accepted, and what is the distribution of the replacement?"
- **Answer key:** q is one-hot on x_d, so the acceptance probability is min(1, p(x_d)/1) = 0.55. Residual: ReLU(p − q) = [0, 0.25, 0.20], normalized by 0.45 = [0, 0.556, 0.444]. Check: P(output = x_d) = 0.55; 0.45 × 0.556 = 0.25; 0.45 × 0.444 = 0.20.
- **Likely wrong answer:** "x_d is the target's top token, so it is accepted with probability 1."
- **Follow-up:** "If x_d were always accepted, how often would the output be x_d? How often does the target want it?"

### specdec_exact-Q6
- **Level:** 3. **Type:** find the flaw (numeric).
- **Question:** "A paper reports 6.16 accepted tokens out of 8 drafted per round. A colleague says: 'So the per-token acceptance is 77%, and without the check 23% of the tokens would be wrong.' What is wrong, and what is the better estimate?"
- **Answer key:** Accepted/drafted counts every token after the first rejection as not accepted, although those tokens were never tested. Under the i.i.d. model solve α(1 − α^8)/(1 − α) = 6.16, which gives α ≈ 0.94. So about 6% of positions, not 23%. This is still a model: the real rate changes with the position in the draft. Note that 0.77 is correct in the paper's own linear model, which counts α·k + 1 = 0.77 × 8 + 1 = 7.16 tokens per round (fact 12). The mistake is to read it as a per-token rate.
- **Likely wrong answer:** Agrees with 77% / 23%.
- **Follow-up:** "With a per-token acceptance of 0.77 and 8 drafted tokens, how many accepted tokens per round do you expect?" (0.77 × (1 − 0.77^8)/0.23 = 2.93, far below 6.16.)

### specdec_exact-Q7
- **Level:** 4. **Type:** connect two ideas (Part A).
- **Question:** "At one position the exact check accepts a sparse draft with probability β = 0.94. Now you remove the check and emit the draft token directly. On the same prefix, what is the smallest possible probability that the emitted token differs from the token an exact sampler would emit? What does this tell you about measuring the 'dose' of an unchecked method?"
- **Answer key:** The emitted token follows q, the exact one follows p. The best coupling makes them equal with probability Σ min(p, q) = β. So they differ with probability at least 1 − β = TV(p, q) = 0.06. So the exact check's rejection rate, measured offline on a fixed text, is a lower bound on the change rate of the unchecked method. The bound is reached if the comparison run shares its randomness in the best way (for example the exact speculative run with the same draft token), and at T = 0. With two independent samples the tokens differ more often, with probability 1 − Σ_x p(x)·q(x). The rate gives only *how often*, not *how unlikely* the changed token is. And it holds per position on a shared prefix; after the first change the two texts diverge.
- **Likely wrong answer:** "Zero, if the draft token is plausible" or "it cannot be known without generating".
- **Follow-up:** "Without the check, the emitted token is a sample from q. What is the largest probability with which one sample from q and one sample from p can be made equal?"

### specdec_exact-Q8
- **Level:** 4. **Type:** apply to a new case (numeric, Part A).
- **Question:** "Three tokens A, B, C with p = [0.6, 0.3, 0.1] (target) and q = [0.3, 0.3, 0.4] (draft). A lenient rule accepts the draft token with probability min(1, p(x)/(0.5·q(x))). On rejection it samples from ReLU(p − 0.5·q), normalized. What is the acceptance probability, what distribution is emitted, and how far is it from p in total variation?"
- **Answer key:** Accepted mass per token = min(q, p/0.5) = [0.3, 0.3, 0.2], sum 0.8 (exact rule: 0.7). Residual: ReLU(p − 0.5q) = [0.45, 0.15, 0], normalized [0.75, 0.25, 0]. Emitted = [0.3, 0.3, 0.2] + 0.2 × [0.75, 0.25, 0] = [0.45, 0.35, 0.20]. TV = ½(0.15 + 0.05 + 0.10) = 0.15. Token C is emitted twice as often as the target wants, which is the bound p(C)/l. Tutor note: Leviathan Appendix A.5 defines only the accept test. The residual used here is a toy choice of this pack.
- **Likely wrong answer:** "Acceptance 0.8 and the output is still p, because rejected tokens are still replaced from a residual."
- **Follow-up:** "Under this rule, what is the probability that C is proposed and accepted? Compare it with p(C)." (0.2 > 0.1, so no residual can repair it.)

### specdec_exact-Q9
- **Level:** 4. **Type:** predict (numeric).
- **Question:** "Three tokens. At T = 1 the target gives p = [0.5, 0.3, 0.2] and the draft gives q = [0.4, 0.45, 0.15]. Both models are sharpened with the same temperature T: each probability is raised to the power 1/T and the result is renormalized. What does the acceptance probability converge to as T goes to 0? Does lowering T always raise it?"
- **Answer key:** Both distributions become one-hot. β → 1 if the two argmax tokens are the same, 0 if they differ. Here the target's top token is the first and the draft's is the second, so β → 0. Computed: β = 0.85 at T = 1, 0.71 at T = 0.5, 0.50 at T = 0.25, 0.24 at T = 0.1. Worked step at T = 0.5: p² = [0.25, 0.09, 0.04]/0.38 = [0.658, 0.237, 0.105]; q² = [0.16, 0.2025, 0.0225]/0.385 = [0.416, 0.526, 0.058]; Σ min = 0.416 + 0.237 + 0.058 = 0.711. Over many positions the mean tends to the argmax agreement rate.
- **Likely wrong answer:** "Yes, sharper distributions overlap more, so acceptance always rises."
- **Follow-up:** "At T = 0, which token does each model pick here? What is min(p, q) summed over tokens then?"

### specdec_exact-Q10
- **Level:** 4. **Type:** predict (numeric).
- **Question:** "A tree node has two children. They are the draft's two fixed top tokens, not samples. The target samples at temperature 1 and gives them probabilities 0.2 (tested first) and 0.5 (tested second). The exact accept/reject rule is applied to one child after the other. For each child, the proposal distribution in the rule is the one the child was really drawn from: one-hot on that token. What is the probability that one of the two children is accepted? Does the test order change it?"
- **Answer key:** First child: accepted with probability 0.2. If rejected, the residual is p without that token, renormalized, so the second child has probability 0.5/0.8 = 0.625. Total = 0.2 + 0.8 × 0.625 = 0.7 = 0.2 + 0.5. The order does not change the total; it is the target mass the children cover.
- **Likely wrong answer:** "1 − (1 − 0.2)(1 − 0.5) = 0.6" (treats the two tests as independent).
- **Follow-up:** "After the first child is rejected, you know the target's token is not that one. Given this, what is the probability that the target's token is the second child?"

### specdec_exact-Q11
- **Level:** 4. **Type:** connect two ideas (Part C).
- **Question:** "SparseSpec drafts 8 tokens with sparse attention, then runs one full-attention verification pass. 5 draft tokens are accepted. After this round, which KV entries in the cache were computed with sparse attention? How many KV entries does this round leave in the cache?"
- **Answer key:** None. The verification pass takes 9 input tokens (the last committed token and the 8 draft tokens) at the same cache positions and writes full-attention KV for them. (The overwrite is inferred from the SparseSpec code, see fact 19. S3R states the same design in words.) Then 8 + 1 − 6 = 3 entries are retracted (draft tokens 6, 7, 8; the 6 output tokens are the 5 accepted ones and the replacement). So 6 entries stay: the last committed token and the 5 accepted draft tokens. The replacement token has no entry yet. So both the tokens and the cache are exact. That is why exact self-speculation cannot show a "KV channel" effect.
- **Likely wrong answer:** "The 5 accepted tokens keep their sparse-attention KV; only the tokens were checked."
- **Follow-up:** "To get the target's distribution after each draft token, which tokens must the verification pass process as inputs? What does a forward pass write for every input token?"

### specdec_exact-Q12
- **Level:** 4. **Type:** apply to a new case (Part C).
- **Question:** "You start from an exact sparse self-speculative system. Describe two one-sided variants: one that repairs only the tokens and one that repairs only the KV cache. For each, say what stays approximate."
- **Answer key:** *KV-only repair:* run the full pass every k tokens and let it overwrite the KV entries, but keep all draft tokens (no accept/reject). Tokens follow the sparse distribution; the stored KV is exact for the text as written. This is close to ReSA: it decodes with sparse attention, and after every f tokens (default f = 32) it re-encodes the recent tokens with dense attention to rebuild their KV. It "avoids per-token accept/reject decisions" [checked, saved copy: 2506.04108 "Rectification Algorithm" paragraph, Sec 3.1, related work]. *Token-only repair:* apply the exact accept/reject rule, but keep the draft-written KV entries for accepted tokens (the verify pass must not overwrite them). A strong answer adds the catch: later verification passes then attend over approximate entries, so their "target" distribution is no longer exactly p. The tokens are exact only in the first round. Tutor note: this is a design question. Other designs can be right. Grade whether each variant says clearly what is exact and what stays approximate.
- **Likely wrong answer:** "It cannot be separated, the same pass does both" or "relaxed verification is the token-only repair".
- **Follow-up:** "In your token-only variant, the next verification pass attends over the cache. Which entries there are approximate now, and what does that do to the distribution you call 'target'?"

### specdec_exact-Q13
- **Level:** 4. **Type:** connect two ideas (Part B).
- **Question:** "You run a relaxed check with a separate draft model up to token 500, then switch to the exact check. From token 501 on, what distribution do the tokens follow? If the trace is still longer than normal after the switch, what can and what cannot be the cause?"
- **Answer key:** From 501 on, each token is an exact sample from the target *given the actual prefix*, relaxed tokens included. The target's KV for the prefix was written by verification passes, so it is exact for that text. So extra length after the switch cannot come from new errors or from approximate KV. It must come from the first 500 tokens as text (lasting damage through the token channel). With unverified sparse attention this is not true: old KV entries stay approximate after the switch unless they are recomputed.
- **Likely wrong answer:** "After the switch the output is the target's distribution again, so the length returns to normal by construction."
- **Follow-up:** "The exact rule guarantees p(x_t | prefix). Which prefix is that after the switch?"

### specdec_exact-Q14
- **Level:** 4. **Type:** find the flaw.
- **Question:** "As a noise control, a student runs the plain target with seed 0 and exact speculative decoding with seed 0, both at temperature 0.6. The two texts differ from token 37 on. The student concludes that the speculative implementation is lossy. What is wrong with this conclusion, and what is the right control?"
- **Answer key:** Exactness is equality of distributions, not of sample paths. The plain run turns one random number into a token from p. The speculative run uses a draft sample, a uniform u and sometimes a residual sample. The same seed gives different text. The right control compares distributions: many samples per prompt in both arms, with confidence intervals on length and accuracy. At T = 0 the tokens should match, apart from numeric near-ties.
- **Likely wrong answer:** "There is a bug, or hardware numerics changed token 37."
- **Follow-up:** "In the plain run, token 37 comes from one random number applied to p. In the speculative run, how many random draws produce token 37, and from which distributions?"

### specdec_exact-Q15
- **Level:** 5. **Type:** find the flaw (numeric).
- **Question:** "Two tokens. The draft gives q = [0.5, 0.5] and the target gives p = [0.8, 0.2]. An implementation draws one uniform number u. It picks the draft token by inverse CDF (first token if u < 0.5, else the second), and then reuses the same u in the accept test u ≤ p(x)/q(x). What distribution does it output?"
- **Answer key:** If u < 0.5 the draft is token 1, ratio 1.6, accepted. If u ≥ 0.5 the draft is token 2, ratio 0.4, and u ≤ 0.4 is impossible, so it is always rejected. The residual is ReLU(p − q) = [0.3, 0] → token 1. The output is [1, 0], not [0.8, 0.2]. The proof needs u to be independent of the draft sample. (Simulation in this run: output [1.0, 0.0], acceptance 0.50; with an independent u: [0.802, 0.198], acceptance 0.70.)
- **Likely wrong answer:** "[0.8, 0.2]; the rule is lossless for any p and q."
- **Follow-up:** "When the draft token is the second one, which values can u have? Can u ≤ 0.4 be true then?"

### specdec_exact-Q16
- **Level:** 5. **Type:** find the flaw (numeric).
- **Question:** "Four tokens. The raw target is p = [0.5, 0.3, 0.15, 0.05]; with top-p = 0.95 the last token is cut. The draft is uniform, q = 0.25 each. An implementation runs the accept test on the raw p. After a rejection it builds the residual from the top-p filtered target, ReLU(p_filtered − q), and it samples bonus tokens from the filtered target. With what probability does it emit the fourth token at this position, and what should that probability be?"
- **Answer key:** The fourth token is proposed with probability 0.25 and accepted with probability min(1, 0.05/0.25) = 0.2, so it is emitted with probability 0.05. The filtered target gives it 0, so it should never appear. The full output is [0.492, 0.308, 0.150, 0.050] against the filtered target [0.526, 0.316, 0.158, 0] (TV 0.05). The full vector holds for the residual named in the question. Other variants of the same bug give other vectors, but the fourth token gets 0.05 in all of them, so grade only the 0.05 against 0. Fix: apply the same filter to p before the rule. Then a draft token outside the target's support has p(x) = 0 and is always rejected. One 2026 study writes this about one relaxed rule that it implements (CACTUS): "A draft token outside the target's top-p/top-k support is rejected regardless of α" [checked, saved copy: 2610.23818 Appendix A]. The sentence is about that one rule, not about all rules. Why this is a real risk: the sampler in the SparseSpec repo says the bonus token is sampled outside the rejection sampler so that it can use top_p and top_k, "while spec decode does not support these sampling strategies" [checked, local code: `sampling/rejection_sampler.py` docstring].
- **Likely wrong answer:** "It is fine; top-p only changes the tokens that are resampled."
- **Follow-up:** "The filtered target gives the fourth token probability 0. When the draft proposes it, with what probability does the accept test (on the raw p) keep it?"

### specdec_exact-Q17
- **Level:** 5. **Type:** apply to a new case (Parts A and B, noise control).
- **Question:** "You run exact speculative decoding at temperature 0 on a reasoning model. The verification pass processes 9 tokens per request inside a mixed batch; plain decoding processes 1 token per step. The two code paths give logits that differ by about 1e-3. Is the speculative output guaranteed to be token-identical to plain greedy decoding? What does your answer mean for the length-tax experiment?"
- **Answer key:** No. Where the top two logits are closer than the numeric difference, the argmax can flip. After one flip the traces diverge. "Lossless" holds only "within hardware numerics" [checked: 2302.01318 abstract]. So an exact arm can differ in length from the baseline by noise alone. Chen et al. say it directly: "Even with greedy sampling, a single token deviating due to numerics could result in two sequences diverging wildly" [checked: 2302.01318 Sec 6.1]. One paper reports, for DeepSeek-R1-Distill-Qwen-7B under bfloat16 and greedy decoding, "up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size" [checked, saved copy: 2506.09501 abstract]. Every lossy arm therefore needs a noise-only control (same method class, exact rule, different batch shape or seed), and effects must be larger than that spread.
- **Likely wrong answer:** "Yes, at T = 0 exact speculative decoding is bit-identical to the baseline, so the baseline run is the control."
- **Follow-up:** "At a position where the top two logits differ by 1e-4 and the two code paths differ by 1e-3, which token does each path pick?"

### specdec_exact-Q18
- **Level:** 5. **Type:** connect two ideas (numeric, Part A).
- **Question:** "A 5-token toy. The target is p = [0.70, 0.15, 0.08, 0.05, 0.02]. Arm 1 is a lenient check (factor 0.3) with a draft that likes the last token. It emits r1 = [0.6212, 0.1818, 0.0697, 0.0606, 0.0667]. Arm 2 is the target sampled at temperature 1.24. It emits r2 = [0.6109, 0.1761, 0.1060, 0.0725, 0.0346]. Both are 0.089 away from p in total variation. Are they the same 'dose'? For each arm compute the mean surprisal under the target, Σ_x r(x)·(−ln p(x)), and the ratio r/p on the last token. Which number is about equal, which clearly differs, and why does it matter for Part A?"
- **Answer key:** TV is equal by construction. The mean surprisal under the target is almost equal: 1.185 nats for arm 1 against 1.172 for arm 2 (target entropy 0.964). The mass on the target's rarest token differs: 0.0667/0.02 = 3.33 times the target's value under the lenient check, 0.0346/0.02 = 1.73 times under temperature (same numbers as Exercise 3). The lenient check moves mass to tokens the *draft* likes; temperature moves it by the target's own ranking. So "dose" is not one number. Part A must fix which measure it matches (change rate, mean surprisal, tail mass). Sparse attention adds a further difference: its changed tokens depend on what context was missing.
- **Likely wrong answer:** "Same TV means the same dose, so the two arms should give the same extra length."
- **Follow-up:** "Under the lenient check, which tokens gain probability: the ones the target ranks next, or the ones the draft prefers?"

### specdec_exact-Q19
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "A sparse self-speculative system drafts k = 6 tokens per round. Its 'acceptance' means accepted/drafted, so a round emits about 1 + k·α tokens. A draft step reads only the selected share of the KV cache, so a draft step gets slower when that share grows. Giving the draft 50% of the cache instead of 5% raises acceptance from 80.4% to 96.7%. (a) How many tokens does a round emit at each budget? (b) Predict: does the speedup rise or fall? What would have to be true about the cost of a round for it to rise? (c) The measured speedup is 1.370× at 5% and 1.058× at 50%. How many target steps does one round cost at each budget?"
- **Answer key:** (a) 1 + 6 × 0.804 = 5.82 and 1 + 6 × 0.967 = 6.80 tokens. That is 17% more. (b) Speedup = tokens per round / cost of one round in target steps. The tokens grow by only 17% and can never pass k + 1 = 7. The cost grows too, because the six draft steps now read ten times more KV. The speedup rises only if the round cost grows by less than 17%. So expect it to fall. In the limit, every draft token is accepted and a draft step costs one target step: γ + 1 tokens for γ + 1 steps, speedup 1. (c) Cost = tokens / speedup: 5.82/1.370 = 4.25 target steps at 5%, and 6.80/1.058 = 6.43 at 50%. The cost grew by 51%. If the verification pass costs one target step, one draft step costs (4.25 − 1)/6 = 0.54 of a target step at 5% and (6.43 − 1)/6 = 0.90 at 50%. The draft is not cheap even at 5%: sparse attention cuts KV reads, but the projections and feed-forward layers stay. The acceptance and speedup numbers are from S3R (Qwen3-8B, frozen step-60 checkpoint, batch 64, 8K starting context, k = 6) [checked, saved copy: 2610.13202 Sec 6.4 and Figure 6 caption; Sec 5 for the cost remark]. The tokens and costs per round are my arithmetic with the model τ = 1 + kα, which the paper also uses in its estimate (1 + kα)/(c0 + c1·s) [same source, Appendix F.3, Eq. 24]. They are not numbers printed in the paper. The paper reports about 5.5 tokens per round at 5% ("τ≈5.5"), a little below 5.82, so treat the costs in (c) as rough. Grade the method, not the last digit.
- **Likely wrong answer:** "Higher acceptance, so higher speedup."
- **Follow-up:** "Suppose every draft token is accepted, and a draft step costs as much as a target step. How many tokens does a round of γ draft tokens emit, and how many target steps does it cost?" (γ + 1 and γ + 1: speedup 1. Do not plug α = 1 into the closed form; it gives 0/0. Use the limit.)

### specdec_exact-Q20
- **Level:** 5. **Type:** find the flaw (numeric).
- **Question:** "Three tokens A, B, C. The target samples from p = [0.6, 0.3, 0.1]. The draft's distribution is q = [0.3, 0.3, 0.4]. A tree implementation does not sample from q. It always takes the draft's top-1 token as the single child. It then accepts the child with probability min(1, p(child)/q(child)), and on rejection it samples from ReLU(p − q), normalized. What distribution does it output? What is the correct rule for this child?"
- **Answer key:** The child is always C. Buggy rule: accept with min(1, 0.1/0.4) = 0.25. On rejection the residual is ReLU(p − q) = [0.3, 0, 0] → A. Output = [0.75, 0, 0.25]. The target is [0.6, 0.3, 0.1], so TV = ½(0.15 + 0.30 + 0.15) = 0.30, and B can never appear. Correct rule: the child was drawn from a one-hot distribution on C, so use q = [0, 0, 1]. Accept with probability p(C) = 0.1. Residual = p without C = [0.6, 0.3, 0]/0.9 = [0.667, 0.333, 0]. Output = [0.9 × 0.667, 0.9 × 0.333, 0.1] = [0.6, 0.3, 0.1]. The losslessness proof needs x ~ q for the same q that appears in the ratio.
- **Likely wrong answer:** "It is exact; the ratio p/q with the residual is the standard lossless rule."
- **Follow-up:** "In the proof, the term q(x)·min(1, p(x)/q(x)) is the probability that x is proposed and accepted. In this implementation, what is the probability that C is proposed?"

### specdec_exact-Q21
- **Level:** 3. **Type:** explain why.
- **Question:** "You said the residual is like a ReLU layer whose output is often 0. Make this exact. The residual is ReLU(p(x) − q(x)), normalized. For which tokens is the ReLU output zero? Why is it correct that these tokens never come out of the resample branch? And what does the sum of the ReLU outputs over the vocabulary equal?"
- **Answer key:** The ReLU is zero where q(x) ≥ p(x): the tokens the draft proposes at least as often as the target wants. Such a token is proposed with probability q(x) and accepted with probability p(x)/q(x). So the accept branch alone already emits it with probability exactly p(x); any more would be too much. A token with p(x) > q(x) is always accepted when proposed, which gives only q(x). The missing mass is p(x) − q(x) = ReLU(p(x) − q(x)). The sum is Σ_x ReLU(p(x) − q(x)) = TV(p, q) = 1 − β, which is exactly the rejection probability. So the mass that rejections free equals the mass that is missing.
- **Likely wrong answer:** "The ReLU is zero for tokens the target does not like (low p)." (It depends on p − q, not on p. In Q1, token B has p = 0.3 and still gets zero.)
- **Follow-up:** "In Q1, token B has p = q = 0.3. Is B in the residual? With what total probability is B emitted, and through which branch?"

---

## 5. Exercises

All three are CPU-only numpy, each under 30 minutes. The code and outputs below were run in this session in the scratch folder `/tmp/claude-1025/-home-ryankert-l-kv-fetch-kv-fetch/b232f66b-c1cf-477e-b08d-10c5771f3def/scratchpad/specdec/code/` (Python 3 with numpy, no GPU).

### Exercise 1. Check that one step is lossless, and break it in two ways

**Task.** Use p = [0.50, 0.20, 0.15, 0.10, 0.05] (target) and q = [0.30, 0.40, 0.10, 0.10, 0.10] (draft). (a) Compute β, ReLU(p − q) and the residual. (b) Compute the exact output distribution of the correct rule and of the wrong rule "resample from p". (c) Check the correct rule by simulation. (d) Do the greedy-draft case (q one-hot on the draft's argmax). (e) Break it again: pick the draft token by argmax, but keep the soft q in the ratio and in the residual.

**Expected result.** β = 0.75 = 1 − TV. ReLU(p − q) = [0.2, 0, 0.05, 0, 0], sum 0.25. Residual = [0.8, 0, 0.2, 0, 0]. Correct output = p. Naive output = [0.425, 0.25, 0.1375, 0.125, 0.0625], TV 0.0875 from p. Greedy draft: accepted with probability 0.2, output = p. Argmax draft with soft q: output [0.4, 0.5, 0.1, 0, 0], TV 0.3 from p.

**How the tutor checks.** The learner must give the two analytic output vectors, not only a simulation. Ask: "Which tokens gain mass under the naive rule, and why those?" (The ones with q > p: they got their full share through acceptance already.) For (e) ask: "Which distribution was the draft token really drawn from?"

```python
# Exercise 1: one speculative step is lossless only with the residual distribution.
# p = target distribution, q = draft distribution (Leviathan notation).
import numpy as np

p = np.array([0.50, 0.20, 0.15, 0.10, 0.05])   # target
q = np.array([0.30, 0.40, 0.10, 0.10, 0.10])   # draft

relu = lambda z: np.maximum(z, 0.0)
tv = lambda a, b: 0.5 * np.abs(a - b).sum()

# ---- exact (analytic) output distributions ----
beta = np.minimum(p, q).sum()                  # acceptance probability
gap = relu(p - q)                              # un-normalised residual = ReLU(p - q)
resid = gap / gap.sum()                        # residual distribution
out_correct = np.minimum(p, q) + (1 - beta) * resid   # resample from the residual
out_naive   = np.minimum(p, q) + (1 - beta) * p       # WRONG: resample from p itself

print("beta = sum min(p,q)        :", round(beta, 4))
print("1 - TV(p,q)                :", round(1 - tv(p, q), 4))
print("ReLU(p - q)                :", np.round(gap, 4), " sum =", round(gap.sum(), 4))
print("residual distribution      :", np.round(resid, 4))
print("output, residual resample  :", np.round(out_correct, 4))
print("output, naive resample     :", np.round(out_naive, 4))
print("TV(output_correct, p)      :", round(tv(out_correct, p), 4))
print("TV(output_naive, p)        :", round(tv(out_naive, p), 4))

# ---- Monte Carlo check of the correct rule ----
rng = np.random.default_rng(0)
N = 400_000
x = rng.choice(5, size=N, p=q)                 # draft samples
u = rng.random(N)
accept = u < np.minimum(1.0, p[x] / q[x])
y = np.where(accept, x, rng.choice(5, size=N, p=resid))
emp = np.bincount(y, minlength=5) / N
print("empirical acceptance rate  :", round(accept.mean(), 4))
print("empirical output           :", np.round(emp, 4))
print("max |empirical - p|        :", round(np.abs(emp - p).max(), 4))

# ---- greedy-draft special case: q is one-hot on the draft's argmax ----
xd = int(np.argmax(q))                         # draft proposes token 1
q1 = np.eye(5)[xd]
beta1 = np.minimum(p, q1).sum()
resid1 = relu(p - q1); resid1 /= resid1.sum()
out1 = np.minimum(p, q1) + (1 - beta1) * resid1
print("greedy draft token         :", xd, " accept prob = p[xd] =", round(beta1, 4))
print("greedy-draft residual      :", np.round(resid1, 4))
print("greedy-draft output        :", np.round(out1, 4))

# ---- the tempting bug: draft token chosen by argmax, but the ratio uses the soft q ----
a_bug = min(1.0, p[xd] / q[xd])
out_bug = np.zeros(5); out_bug[xd] = a_bug
out_bug += (1 - a_bug) * resid                 # residual built from the soft q
print("argmax draft + soft-q ratio:", np.round(out_bug, 4), " TV to p =", round(tv(out_bug, p), 4))
```

Output of this run:

```
beta = sum min(p,q)        : 0.75
1 - TV(p,q)                : 0.75
ReLU(p - q)                : [0.2  0.   0.05 0.   0.  ]  sum = 0.25
residual distribution      : [0.8 0.  0.2 0.  0. ]
output, residual resample  : [0.5  0.2  0.15 0.1  0.05]
output, naive resample     : [0.425  0.25   0.1375 0.125  0.0625]
TV(output_correct, p)      : 0.0
TV(output_naive, p)        : 0.0875
empirical acceptance rate  : 0.7502
empirical output           : [0.4995 0.2001 0.151  0.0999 0.0496]
max |empirical - p|        : 0.001
greedy draft token         : 1  accept prob = p[xd] = 0.2
greedy-draft residual      : [0.625  0.     0.1875 0.125  0.0625]
greedy-draft output        : [0.5  0.2  0.15 0.1  0.05]
argmax draft + soft-q ratio: [0.4 0.5 0.1 0.  0. ]  TV to p = 0.3
```

### Exercise 2. Tokens per round, and where the i.i.d. formula misleads

**Task.** (a) For α = 0.8, γ = 5, compare the closed form (1 − α^{γ+1})/(1 − α) with a simulation, and report accepted/drafted. (b) For α = 0.8, find the best γ at c = 0.05 and at c = 0.30, and the speedup at γ = 20. (c) Find the i.i.d. α that gives 6.16 accepted tokens of 8. (d) Compare a constant acceptance profile with a decaying one of the same mean (0.96 down to 0.68 in steps of 0.04). (e) Compare α = 0.8 everywhere with a 50/50 mix of α = 0.99 and α = 0.61.

**Expected result.** (a) 3.69 tokens per round; accepted/drafted ≈ 0.54, not 0.8. (b) Best γ = 8 (3.09×) at c = 0.05; best γ = 3 (1.55×) at c = 0.30, and γ = 20 gives 0.71× (a slowdown). (c) α ≈ 0.9416, not 0.77. (d) Constant 4.62, decaying 5.67. (e) 3.69 against 4.14.

**How the tutor checks.** Ask the learner to explain (d) and (e) without code. (d): the product Π a_i puts the most weight on early positions. (e): tokens per round is convex in α, so a mix gives more than the mean α (Jensen). Then ask what this means for a single reported "acceptance rate".

**Tutor note on (d).** The decaying profile is a toy set of *conditional* rates a_i. Do not present it as measured data. The per-position numbers in S3R (95.8% down to 67.9%) look like reach rates, not conditional rates (fact 10). A quick check the learner can do: take conditional rates that go linearly from 0.958 to 0.679 over 6 positions. Then 1 + Σ Π a_i = 4.85 tokens per round and accepted/drafted = 0.64. Now take reach rates that go linearly from 0.958 to 0.679. Then the round gives 1 + 6 × 0.8185 = 5.91 tokens and accepted/drafted = 0.82. The paper reports 81.9% for that checkpoint. (Examiner's arithmetic, CPU-only; the linear shape between the two end points is an assumption.)

```python
# Exercise 2: tokens per round, the i.i.d. formula, and where it breaks.
# alpha = per-token acceptance probability, gamma = draft length,
# c = cost of one draft step / cost of one target pass.
import numpy as np

def tokens_per_round(alpha, gamma):
    # accepted draft tokens + 1 (the corrected token or the bonus token)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)

def speedup(alpha, gamma, c):
    return tokens_per_round(alpha, gamma) / (gamma * c + 1)

# (a) closed form versus simulation, alpha = 0.8, gamma = 5
rng = np.random.default_rng(0)
alpha, gamma, N = 0.8, 5, 200_000
acc = rng.random((N, gamma)) < alpha                 # accept flags per position
n_acc = np.where(acc.all(1), gamma, np.argmin(acc, 1))   # length of accepted prefix
print("(a) closed form tokens/round :", round(tokens_per_round(alpha, gamma), 4))
print("    simulated tokens/round   :", round((n_acc + 1).mean(), 4))
print("    accepted / drafted       :", round(n_acc.mean() / gamma, 4),
      "(not equal to alpha =", alpha, ")")

# (b) best draft length for alpha = 0.8 at two cost ratios
for c in (0.05, 0.30):
    s = [speedup(0.8, g, c) for g in range(1, 41)]
    g_best = int(np.argmax(s)) + 1
    print(f"(b) c={c:4.2f}: best gamma = {g_best:2d}, speedup = {max(s):.3f}, "
          f"speedup at gamma=20 = {speedup(0.8, 20, c):.3f}")

# (c) invert a reported accepted length: 6.16 accepted of gamma = 8 drafted
def accepted(alpha, gamma):
    return alpha * (1 - alpha ** gamma) / (1 - alpha)
lo, hi = 0.5, 0.999999
for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if accepted(mid, 8) < 6.16 else (lo, mid)
a_star = 0.5 * (lo + hi)
print("(c) i.i.d. alpha that gives 6.16 of 8 :", round(a_star, 4))
print("    naive ratio 6.16 / 8              :", round(6.16 / 8, 4))

# (d) same mean acceptance, different shape: constant versus decaying inside the draft
decay = np.array([0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.68])  # mean 0.82
const = np.full(8, decay.mean())
def exact_tokens(a):                                   # 1 + sum_k prod_{i<=k} a_i
    return 1 + np.cumprod(a).sum()
print("(d) mean per-position acceptance      :", round(decay.mean(), 4))
print("    tokens/round, constant profile    :", round(exact_tokens(const), 4))
print("    tokens/round, decaying profile    :", round(exact_tokens(decay), 4))

# (e) heterogeneity across contexts: half the rounds alpha=0.99, half alpha=0.61 (mean 0.80)
mix = 0.5 * tokens_per_round(0.99, 5) + 0.5 * tokens_per_round(0.61, 5)
print("(e) tokens/round, alpha=0.8 everywhere:", round(tokens_per_round(0.8, 5), 4))
print("    tokens/round, 50/50 mix 0.99/0.61 :", round(mix, 4))
```

Output of this run:

```
(a) closed form tokens/round : 3.6893
    simulated tokens/round   : 3.6909
    accepted / drafted       : 0.5382 (not equal to alpha = 0.8 )
(b) c=0.05: best gamma =  8, speedup = 3.092, speedup at gamma=20 = 2.477
(b) c=0.30: best gamma =  3, speedup = 1.554, speedup at gamma=20 = 0.708
(c) i.i.d. alpha that gives 6.16 of 8 : 0.9416
    naive ratio 6.16 / 8              : 0.77
(d) mean per-position acceptance      : 0.82
    tokens/round, constant profile    : 4.6243
    tokens/round, decaying profile    : 5.6726
(e) tokens/round, alpha=0.8 everywhere: 3.6893
    tokens/round, 50/50 mix 0.99/0.61 : 4.142
```

### Exercise 3. The dose of a relaxed check (ties to Part A)

**Task.** Use p = [0.70, 0.15, 0.08, 0.05, 0.02] (target) and q = [0.45, 0.15, 0.05, 0.05, 0.30] (draft; it likes the last token, the target does not). Toy lenient rule: accept x ~ q with probability min(1, p(x)/(l·q(x))); on rejection sample from ReLU(p − l·q), normalized. (The accept test is Leviathan Appendix A.5; the residual is a toy choice of this pack.) For l = 1, 0.5, 0.3, 0.1 compute the acceptance probability, the emitted distribution r, TV(r, p), the mean surprisal E_r[−log p], and r/p on the last token. Then find the temperature T > 1 whose p_T ∝ p^{1/T} has the same TV from p as l = 0.3, and compare the same numbers.

**Expected result.** Acceptance rises 0.72 → 0.74 → 0.77 → 0.90. TV rises 0 → 0.061 → 0.089 → 0.200. The last token is emitted 1×, 2×, 3.33×, 10× as often as the target wants (the bound 1/l is reached). The matching temperature is T ≈ 1.238. At equal TV the mean surprisal is almost equal (1.185 against 1.172 nats), but the tail ratio is 3.33 against 1.73.

**How the tutor checks.** The learner must say in words *where* each method puts its extra mass. Then ask: "Which of these three numbers (TV, mean surprisal, tail ratio) would you match between arms in Part A, and what would the other two then confound?" There is no single right choice; the reasoning must name the trade.

```python
# Exercise 3: a relaxed (lenient) check changes the emitted distribution.
# Measure the "dose" two ways and compare with a temperature change of equal TV.
# Rule (toy): accept draft token x ~ q with probability min(1, p(x) / (l * q(x))), 0 < l <= 1.
# On rejection, sample from norm(max(0, p - l*q)).  l = 1 is the exact rule.
import numpy as np

p = np.array([0.70, 0.15, 0.08, 0.05, 0.02])   # target
q = np.array([0.45, 0.15, 0.05, 0.05, 0.30])   # draft: likes token 4, target does not

tv = lambda a, b: 0.5 * np.abs(a - b).sum()
H = -(p * np.log(p)).sum()                     # target entropy (nats)

def emitted(l):
    acc = np.minimum(q, p / l)                 # P(draft token x is proposed and accepted)
    alpha = acc.sum()
    res = np.maximum(p - l * q, 0.0); res /= res.sum()
    return alpha, acc + (1 - alpha) * res

print(f"target entropy H(p) = {H:.4f} nats")
print(" l    alpha   emitted distribution r                 TV(r,p)  E_r[-log p]  r[4]/p[4]")
for l in (1.0, 0.5, 0.3, 0.1):
    a, r = emitted(l)
    print(f"{l:4.1f}  {a:.4f}  {np.round(r, 4)}  {tv(r, p):.4f}   {-(r * np.log(p)).sum():.4f}      {r[4] / p[4]:.2f}")

# temperature arm with the same TV dose as l = 0.3
_, r03 = emitted(0.3)
target_tv = tv(r03, p)
def p_T(T):
    w = p ** (1.0 / T); return w / w.sum()
lo, hi = 1.0, 5.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if tv(p_T(mid), p) < target_tv else (lo, mid)
T = 0.5 * (lo + hi); pt = p_T(T)
print(f"\ntemperature with the same TV as l=0.3: T = {T:.4f}")
print("p_T                      :", np.round(pt, 4))
print(f"TV(p_T, p)               : {tv(pt, p):.4f}")
print(f"E_pT[-log p]             : {-(pt * np.log(p)).sum():.4f}")
print(f"p_T[4] / p[4]            : {pt[4] / p[4]:.2f}")
print(f"l=0.3: E_r[-log p]       : {-(r03 * np.log(p)).sum():.4f},  r[4]/p[4] = {r03[4] / p[4]:.2f}")
```

Output of this run:

```
target entropy H(p) = 0.9643 nats
 l    alpha   emitted distribution r                 TV(r,p)  E_r[-log p]  r[4]/p[4]
 1.0  0.7200  [0.7  0.15 0.08 0.05 0.02]  0.0000   0.9643      1.00
 0.5  0.7400  [0.646  0.181  0.0727 0.0603 0.04  ]  0.0613   1.0945      2.00
 0.3  0.7667  [0.6212 0.1818 0.0697 0.0606 0.0667]  0.0891   1.1849      3.33
 0.1  0.9000  [0.522  0.1648 0.0582 0.0549 0.2   ]  0.1998   1.5930      10.00

temperature with the same TV as l=0.3: T = 1.2381
p_T                      : [0.6109 0.1761 0.106  0.0725 0.0346]
TV(p_T, p)               : 0.0891
E_pT[-log p]             : 1.1720
p_T[4] / p[4]            : 1.73
l=0.3: E_r[-log p]       : 1.1849,  r[4]/p[4] = 3.33
```

**Extra checks run for the question bank** (`checks.py`, same folder; selected lines of the real output):

```
--- Q3: tokens per round and speedup
tokens 3.3616 speedup 2.4011428571428572 speedup with verify cost 1.3: 1.9774117647058822
--- Q6: three acceptance numbers
iid alpha for 6.16 of 8: 0.9415853644089123  accepted at alpha .77: 2.934123067612298  ratio 0.77
--- Q9: temperature and beta
argmax differ, T 1.0 beta 0.85
argmax differ, T 0.5 beta 0.7109
argmax differ, T 0.25 beta 0.5012
argmax differ, T 0.1 beta 0.2415
--- Q20: top-1 child chosen by argmax, ratio uses the draft's soft q (biased)
child 2 buggy accept 0.25 buggy output [0.75 0.   0.25] TV 0.3
correct accept 0.1 correct output [0.6 0.3 0.1] TV 0.0
--- tree, 5-token toy: (a) one-hot top-2, (b) soft-q top-2, (c) iid samples MSS, (d) no-replacement
(a) children [1 0] accept 0.7 output [0.5  0.2  0.15 0.1  0.05] TV 0.0
(b) soft q in ratio: accept 1.0 output [0.5 0.5 0.  0.  0. ] TV 0.3
(c) after 1 iid candidates from q: accept = 0.7500
(c) after 2 iid candidates from q: accept = 0.8500
(c) after 3 iid candidates from q: accept = 0.9100
(c) output [0.5  0.2  0.15 0.1  0.05] TV 0.0
(d) 2 children without replacement, renormalised q2: [0.5  0.2  0.15 0.1  0.05] accept 0.9056 TV 0.0
(d) 2 children without replacement, raw q (wrong):   [0.4954 0.2    0.1546 0.1    0.05  ] accept 0.9056 TV 0.0046
--- Q15: shared uniform bug (MC)
same u: output [1. 0.] accept 0.5007
independent u: output [0.80187 0.19813] accept 0.69883
--- Q16: top-p applied only to the resample/bonus branch
filtered target [0.5263 0.3158 0.1579 0.    ] mixed-rule output [0.4923 0.3077 0.15   0.05  ] TV to filtered 0.05
consistent rule output [0.5263 0.3158 0.1579 0.    ] beta 0.6579
--- exponential race: argmax(w / E) samples proportional to w (MC)
weights [0.2  0.   0.05] normalised [0.8 0.  0.2] empirical [0.801015 0.       0.198985]
```

Tutor note on line (d): if two children are sampled *without replacement*, the second child's true proposal is q with the first child removed and renormalized. Using the raw q there gives a small bias (TV 0.0046 in this toy). This is a good extension of Q20 for a level-5 learner.

---

## 6. Diagnostic picks

- **Level 2-3: specdec_exact-Q5** (greedy draft, sampled target). One numeric answer shows whether the learner holds the general rule and the residual, or only the "accept if it matches the argmax" shortcut. If the learner stalls, step down to Q21, which starts from their own ReLU picture.
- **Level 4-5: specdec_exact-Q12** (token-only and KV-only repair). A level-4 answer gives the two variants. A level-5 answer also sees that token-only repair stops being exact once later verification passes read approximate KV. It maps directly onto Part C.

---

## 7. Sources

Fetched in this run (raw page from `https://arxiv.org/html/<id>`, one request per paper, converted to text locally). Each id was confirmed against the title printed on the page.

| id | title as shown on the page | what I read |
|---|---|---|
| arXiv 2211.17192 (v2) | Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman, Matias) | Sections of the full text: abstract, Sec 2.1-2.3, Algorithm 1, Sec 3.1-3.6 (Def 3.1-3.2, Lemma 3.3, Theorem 3.5, Cor 3.6, Eq. 1, Def 3.7, Theorem 3.8, Cor 3.9, Theorem 3.11), Tables 1-3, Sec 4.1-4.2, Appendix A.1-A.5, Table 5. Figures not read. |
| arXiv 2302.01318 (v1) | Accelerating Large Language Model Decoding with Speculative Sampling (Chen et al.) | Sections of the full text: abstract, Sec 3, 4.1, 4.2, 5, 6.1-6.3, Algorithms 1-2, Table 1, Theorem 1, Figure 1 caption. Sec 1, 2, 7 not read. |
| arXiv 2305.09781 (v4) | SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification (Miao et al.) | Sections: Algorithm 2, Sec 4.2, Sec 4.3 (statements of Theorems 4.2 and 4.3; the proofs are not in the page text I read), parts of Sec 6 including Table 3. |
| arXiv 2401.15077 (v3) | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Li, Wei, Zhang, Zhang) | Sections: abstract, Sec 2, 3.1, 3.3, metric definitions in Sec 4, Appendix A.1, Appendix A.2 Algorithm 1. |
| arXiv 2512.01278 (v1) | Accelerating Large-Scale Reasoning Model Inference: Self-Speculative Decoding with Sparse Attention (SparseSpec; Zhao, Tang et al.) | Sections: abstract, Sec 2.3, 3.2, 4.1, 4.4, 5.1, 5.3, 5.4. The HTML `<title>` tag of this page is broken ("1Introduction"); the title was confirmed in the page body. Figures not read. |

Read directly on disk in this run:

| source | what I read |
|---|---|
| `/home/ryankert_l/SparseSpec/serve/sampling/rejection_sampler.py` | full file (accept test, recovered-token sampling, bonus token, `NO_DRAFT_PROBS` path, docstrings) |
| `/home/ryankert_l/SparseSpec/serve/scheduler/spec_scheduler.py` | the linear VERIFY input-building branch and `verify_outputs` only; the tree-verify path was not read |
| `/home/ryankert_l/SparseSpec/serve/request/kv_cache_ptr/base.py` | `allocate_tokens`, `retract_tokens`, `dispatch_append_kv_cache` only |
| `/home/ryankert_l/SparseSpec/serve/model/model.py` | `append_batch_kv_cache` and its call site in the attention forward only |
| `/home/ryankert_l/SparseSpec/serve/sampling/sampler.py` | `forward` and `sample` only |
| `research_notes/Inference speed question search/harvest_specdec_core.md` | full (notes) |
| `research_notes/Inference speed question search/harvest_specdec_self.md` | full (notes) |
| `reports/length-tax.html` | text of the four questions and the source list (notes) |
| the 2026-11-11 version of this file | full; used as a draft, every tag and number re-checked |

Read by the examiner pass in raw arXiv page text saved on disk on 2026-11-11 by the earlier question search (session scratch folders `dl/`, `ax/`, `q12/`, `sd/papers/`, `critic2/abs/`). Each title was confirmed on the saved page. These carry the tag [checked, saved copy]. They were not fetched again.

| id | title on the saved page | what was read |
|---|---|---|
| arXiv 2610.13202 | S3R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding | Sec 4 (verification paragraph), first lines of Sec 5, Sec 6.1 setup, budget-sweep paragraph of Sec 6.4 with the Figure 6 caption, Appendix D (acceptance definition), Appendix E.2, Appendix F.3 around Eq. 24. Figures not read. |
| arXiv 2610.23818 | Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding | abstract; Appendix A paragraph on CACTUS |
| arXiv 2610.11750 | Where Draft Trees Lose Target Mass: Exit-Guided Speculative Decoding | abstract only |
| arXiv 2506.04108 | Rectified Sparse Attention | abstract, "Rectification Algorithm" paragraph, Sec 3.1 default settings, related-work sentence on self-speculation |
| arXiv 2506.09501 | Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference | abstract |
| arXiv 2402.01528 | Decoding Speculative Decoding | abstract, Sec 1, conclusion, the TAR sentence in Sec 3.3 |
| arXiv 2501.19309 | Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment | abstract |
| arXiv 2502.10424 | QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache | abstract |
| arXiv 2404.16710 | LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding | abstract page |

Fetched by the examiner pass in this run (one request to the abstract page): arXiv 2309.08168, Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding (abstract only).

Open points (also in the "could not verify" list returned with this pack):

- SparseSpec Sec 3.2 prints "with k=16, α=0.75, s=0.05, the attention latency is reduced by 6.78×". The ratio printed just before it, (kα+1)/(ks+1), gives 13/1.8 = 7.22× with these numbers. I could not reproduce 6.78, so this pack does not use that number.
- That the SparseSpec verify pass physically overwrites the draft-written KV entries is inferred from the position arithmetic and from the append-position call (`flashinfer.get_batch_indices_positions` with the full cache length), not traced to the kernel write.
- Whether SpecInfer and EAGLE implementations sample children with or without replacement, and which q their code puts in the ratio for top-k children, was not checked. Fact 14 and Q20 are my derivation, checked by enumeration.
- Leviathan Appendix A.5 defines the lenient accept test only. The lenient residual in Q8 and Exercise 3 is a toy choice.
- S3R does not say whether its per-position acceptance (95.8% down to 67.9%) is a conditional rate or a reach rate. Fact 10 reads it as a reach rate. That is an inference from the paper's own total (81.9%).
- For arXiv 2610.11750 only the abstract was read. Its definition of "target coverage" was not checked against the Σ p(c_i) of fact 14.

## Verification log

Examiner pass, finished 2026-11-13 15:32 CST (time from `date`). The examiner solved every question before reading its key, recomputed every number with CPU-only Python, re-ran the three exercises, and re-read the cited passages. No GPU was used. Nothing was posted, committed or pushed.

**Items checked: 56.** 20 core ideas, 8 misconceptions, 21 questions, 3 exercises, the "Extra checks" block, the level ladder, the diagnostic picks, and the sources section.

**What matched with no change.** Facts 1 to 9, 11, 13, 15, 16, 18. Misconceptions 1 to 3 and 6 to 8. Q1 to Q6, Q8, Q9, Q13, Q14, Q15, Q20, Q21. All numbers in all keys. Exercises 1 to 3: the code was extracted from this file and run again, and the output is identical to the pasted output. All 29 lines of the "Extra checks" block were produced again. The level ladder and the diagnostic picks (Q5, Q12).

**What changed.**

1. Fact 10 (wrong). It used S3R's 95.8% to 67.9% as proof that the conditional rate a_i falls with position. The paper does not say that the number is conditional. Its own total (81.9%) fits the reach reading. The fact now defines conditional rate and reach rate, says the Chen Sec 6.3 quote is the reach effect, and uses EAGLE Table 8 as the direct evidence for a falling per-position rate.
2. Fact 12 (ambiguous). "SparseSpec and EAGLE use (ii)" was too simple. Now: SparseSpec and S3R use (ii). EAGLE's words read like (ii) but its n-α numbers are per-position rates. "TAR" in 2402.01528 is meaning (iii).
3. Fact 14 (wrong quote). The quote from 2610.11750 was not the paper's wording. It is now the exact sentence from the abstract. One sentence was added: for fixed children, naive sampling accepts with the same probability.
4. Fact 17 (tags). Draft & Verify, LayerSkip and QuantSpec were [from notes]. Each abstract was read; the tags now say so.
5. Fact 19 (minor). Added the second piece of support for the KV overwrite (the flashinfer position call) and the S3R sentence that states the overwrite. The overwrite in SparseSpec stays an inference.
6. Fact 20 and the Q7 key (wrong). They said the exact check's rejection rate *is* the change rate of the unchecked method. It is a lower bound. It is reached under the best coupling and at T = 0. Independent samples differ with probability 1 − Σ p·q (0.69 against 0.30 for p = [0.6, 0.3, 0.1], q = [0.3, 0.3, 0.4]).
7. Misconception 4 (tags, wording). Both papers were read in saved copies; the quotes are now exact. A sentence was added so the tutor does not teach the opposite mistake ("a stronger draft never helps").
8. Misconception 5 (minor). Added the GPT-like row of Leviathan Table 3 (0.88 at T = 0, 0.89 at T = 1).
9. Q10 (minor). The question used the term "one-hot rule" without explaining it. It now states the rule in the question.
10. Q11 (minor). The question now names SparseSpec and asks for the number of entries left (6). The key marks the overwrite as inferred.
11. Q12 key (tag). The ReSA facts were read in a saved copy (f = 32). A tutor note says other designs can be right.
12. Q16 (ambiguous, wrong attribution). The question did not say which residual the buggy code uses, so the full output vector had more than one defensible value. The question now fixes the residual. The key says to grade only the 0.05 on the fourth token. The quoted sentence from 2610.23818 is about one relaxed rule (CACTUS), not about all rules; the key now says so.
13. Q17 key (tags). The 9,000-token fact now has its full setting and exact quote. The Chen Sec 6.1 quote was added.
14. Q18 (ambiguous). The question could not be answered without running Exercise 3, and "same dose?" had no single answer. It now gives p, r1 and r2 and asks for two numbers per arm. Key: 1.185 against 1.172 nats, 3.33 against 1.73 times.
15. Q19 (ambiguous; broken follow-up). The old follow-up asked the learner to plug α = 1 into the closed form, which gives 0/0. The question is now numeric with one answer per part: 5.82 and 6.80 tokens, 4.25 and 6.43 target steps per round. The per-round numbers are the examiner's arithmetic from the paper's τ = 1 + kα, not numbers printed in the paper.
16. Exercise 2 (minor). A tutor note was added: the decaying profile in (d) is a toy, and S3R's per-position numbers are probably reach rates. Code and output were not changed.
17. Tags and sources. A new tag [checked, saved copy] was defined. The paragraph "cited from notes only" in Sources became a table of the saved copies that were read. No [from notes] tag is left on any fact.

Nothing was deleted.

**What stays unverified.**

- Whether S3R's per-position acceptance is a conditional rate or a reach rate. The paper does not define it. The reach reading is the examiner's inference.
- That the SparseSpec verification pass overwrites the draft-written KV entries. The kernel write was not traced.
- Whether SpecInfer and EAGLE code samples children with or without replacement, and which q the code uses for top-k children. Fact 14 and Q20 are derivations.
- SparseSpec's printed 6.78× (the formula gives 7.22×).
- The definition of "target coverage" in 2610.11750 (abstract only).
- The lenient residual in Q8 and Exercise 3 is a toy choice, not from a paper.
- The per-round costs in the Q19 key (4.25, 6.43, 0.54, 0.90) are arithmetic under τ = 1 + kα and "verification costs one target step". The paper reports τ ≈ 5.5 at s = 0.05, a little below 5.82, so these are rough.
- No figure of any paper was read, only text and tables.

**How the sources were read.** The five main papers (2211.17192, 2302.01318, 2305.09781, 2401.15077, 2512.01278) were re-read in the raw pages that the pack author saved in this session's scratch folder on the same day. The nine other papers were read in raw arXiv pages saved on 2026-11-11 by the earlier question search; each title was confirmed. One new network request was made (the abstract page of 2309.08168).
