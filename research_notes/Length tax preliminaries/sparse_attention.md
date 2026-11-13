# Tutor pack: sparse attention at decode time

Subskill id: `sparse_attention`. Written 2026-11-13 15:16 CST. Read-only work; no GPU job was run. A second reader examined and corrected the pack on 2026-11-13; see the Verification log at the end.

Scope: query-aware top-k and page-level KV selection; the budget; reusing a selection across steps and why reuse hurts; oracle selection; how a selection error becomes a token error; self-drafts with a sparse KV budget and acceptance by position inside a draft; the kv-fetch E3 results as a worked example.

Notation in this file:

- One attention head, one decode step. The cache holds n tokens. Token i has key k_i and value v_i. The current query is q. The head dimension is d.
- **α_i** = dense attention weight of the query on token i. α_i = exp(q·k_i/√d) / Σ_j exp(q·k_j/√d), so Σ_i α_i = 1.
- **S** = the set of cached tokens the sparse read is allowed to see. **B** = the budget = number of tokens in S. A **page** is a block of 16 consecutive tokens.
- **c** = coverage = Σ_{i∈S} α_i. **1 − c** = missing attention.
- **M** = reuse window: the selection is computed once and used for M steps (CVSA).
- **k** = number of draft tokens per round (S3R and SparseSpec use k; the sibling pack `specdec_exact` calls it γ). **s** = budget ratio = B / n.
- Two papers use τ for different things. Here **τ_cov** is CVSA's coverage threshold and **N_round** is S3R's number of tokens emitted per round.

How to read the tags:

- **[checked]**: I downloaded the raw arXiv HTML page in this run, turned it into text on disk, and read the fact there. The examiner pass read each of these facts again in the saved page text.
- **[checked, local log]** / **[checked, local file]** / **[checked, local code]**: I read it in this run in `EXPERIMENTS.md`, in a results JSON, or in source code on this machine.
- **[from notes]**: taken from `research_notes/Inference speed question search/` or `reports/length-tax.html`; the paper was not opened in this run.
- **[unverified]**: could not be confirmed.
- No tag: textbook fact, or a derivation written out here. "My arithmetic" marks a number I computed from checked numbers.

---

## 1. Core ideas

1. **A sparse read.** Dense output: O_dense = Σ_i α_i v_i. A sparse read runs the softmax over S only. The weight of token i ∈ S becomes exp(z_i)/Σ_{j∈S} exp(z_j) = α_i / c, with z_i = q·k_i/√d. So the sparse output is O_S = (1/c)·Σ_{i∈S} α_i v_i. The kept weights are not left as they were: each one grows by the factor 1/c. S3R writes the same thing as SparseAttn(q; I) = softmax(q K_Iᵀ/√d) V_I [checked: 2610.13202 App. B.1 Eq 16]. Coverage is defined as c = Σ_{i∈S} α_i [checked: 2611.10634 Sec 3.1].

2. **The error identity.** Let O_comp = (1/(1−c))·Σ_{i∉S} α_i v_i be the normalized output of the missed tokens. Then O_dense = c·O_S + (1−c)·O_comp. Subtract O_S on both sides: O_S − O_dense = (1−c)·(O_S − O_comp), so ‖O_S − O_dense‖₂ = (1−c)·‖O_S − O_comp‖₂ [checked: 2611.10634 Prop 1 (Eq 1), Prop 2 (Eq 2), App. B]. The error has two factors: how much attention was missed, and how different the missed content is from the kept content. The worst-case bound (1−c)·D_V, with D_V = max_{i,j} ‖v_i − v_j‖₂, is "looser by roughly 14×" in the paper's data [checked: 2611.10634 App. B].

3. **Sparse reading is not eviction.** Eviction deletes a KV entry for good. Sparse reading keeps every entry and picks a new set S at each step. This matters because the tokens a step needs depend on its query [checked: 2406.10774 Sec 3.3]. In kv-fetch E1 (38 Qwen3-4B thinking traces, 1K budget), eviction that knows the true future attention still misses 8.5% of attention, about the same as SnapKV (8.6%), "because evicted pages revive". Re-selecting the true top-B at every step misses 5.0% [checked, local log: EXPERIMENTS.md E1].

4. **Budget, and what it counts.** A budget can be a token count (Quest: "Token Budget" = tokens in the selected pages [checked: 2406.10774 Sec 3.4]) or a ratio with a floor (S3R: K_ℓ = min{|C_ℓ|, max{K_min, ⌈s·|C_ℓ|⌉}} [checked: 2610.13202 App. B.1 Eq 15]; SparseSpec code, with its default settings: `max(128, ceil(len × 0.05))` [checked, local code: `serve/request/kv_cache_ptr/pillar.py`]). In kv-fetch the budget is the total attended tokens per KV head, including a pinned sink page and a recent window [checked, local code: `kvfetch/sparse_decode.py`]. The nominal budget is not what the method reads. CVSA-c4 has a 5% base budget but reads 11.7% of the KV cache (base 5.0, recovery 3.7, dense layer 0 2.9) [checked: 2611.10634 Sec 4.1]. Always ask: per head or shared, pinned tokens inside or outside, and how many bytes of summaries are read on top.

5. **Quest's selector.** For each page, store the channel-wise minimum m and maximum M of its keys. For query Q, the page score is U = Σ_i max(Q_i·m_i, Q_i·M_i), summed over the d channels. For every key k in the page, Q_i·k_i ≤ max(Q_i·m_i, Q_i·M_i) in each channel, so U is an upper bound on the attention logit of every key in the page. Quest reads the top-K pages by U and runs normal attention on them. It is not applied to the first two layers [checked: 2406.10774 Sec 3.4, Algorithm 1]. Quest loads the fraction 1/S + K·S/L of the KV cache, where S is the page size, L the context length and K the number of pages [checked: 2406.10774 Sec 3.5]. U bounds the largest logit in the page. It does not estimate the attention mass of the page, and it grows when the keys of a page are spread out (Exercise 1).

6. **Oracle selection.** An oracle selector looks at the true dense attention of the current step and keeps the part with the most mass. There is a ladder of oracles: top-B tokens per head ≥ top pages per head ≥ one page set per layer shared by all KV heads (each step restricts the one before, so coverage can only fall). The kv-fetch "oracle" is the last one: "the pages with the most true attention, summed over KV heads (the page-set ceiling)" [checked, local code: `kvfetch/sparse_decode.py`]. The shared-set ceiling removes only 27% (1K) and 21% (512) of Quest's missing attention [checked, local log: EXPERIMENTS.md E2b]. An oracle is the ceiling for per-step coverage under its own restrictions. It is not a ceiling for accuracy (fact 20), and it needs the dense attention, so it is not a real method.

7. **Two families of selection signal.** (a) Score now: compare the current query with key summaries (Quest's min/max bound; CVSA's page score max_{j∈page} q·k_j/√d, which reads all cached keys [checked: 2611.10634 Sec 2.2]). (b) Score from the past: reuse attention weights that an earlier dense pass already computed. PillarAttn takes the Top-K of the attention scores dumped during verification, "first averaged over k draft tokens and query heads" [checked: 2512.01278 Sec 4.1]. S3R does the same per layer and KV head, and always adds the recent positions: I_ℓ = R_ℓ ∪ TopK(a_ℓ) with ⌈s·|C_ℓ|⌉ − |R_ℓ| top entries [checked: 2610.13202 Sec 4 Eq 4]. Family (b) costs nothing extra, but its selection is already one round old when the draft starts.

8. **Reuse.** Scoring every step costs time: CVSA with a fresh selection every step (M = 1) runs at 0.93× the dense decode speed at 8K context and batch 32 [checked: 2611.10634 App. E]. So methods compute S once and reuse it for M steps. The price on LongBench (short outputs) is small but grows with M: −0.30 points vs dense at M = 1, −0.43 at M = 2, −1.13 at M = 4, −2.04 at M = 8 [checked: 2611.10634 App. E]. Careful: the share of KV that the method reads also changes with M (16.1% at M = 2, 12.0% at M = 4, 9.9% at M = 8 [checked: 2611.10634 Table 5]). So this sweep mixes reuse with a smaller effective budget. In a self-draft the selection is held fixed for the k draft steps [checked: 2610.13202 Sec 4; 2512.01278 Sec 4.1].

9. **Why reuse hurts.** (i) The query changes, so the tokens that matter change: SparseSpec states that "sparsity patterns shift significantly based on semantic context during generation" [checked: 2512.01278 Sec 3.3]. In kv-fetch, a selection that is one step stale raises missing attention from 31.8% to 35.6% [checked, local log: EXPERIMENTS.md H4]. (ii) The failures move around. In a CVSA trace with fresh selection, cells (layer, head) that fail on at least 80% of steps are 1.6% of cells and give 11.1% of failure events; cells that fail on 5-80% of steps give 86.4%; the set of failing heads has Jaccard 0.33 between consecutive steps [checked: 2611.10634 App. F; two-sample Llama-3.1-8B HotpotQA trace]. (iii) On reuse steps nobody looks: "the page selection and the all-context normalizer are stale and no check runs" [checked: 2611.10634 App. P]. The average missing attention hides this. A few steps with a large miss carry most of the loss (Exercise 2).

10. **Reuse and reasoning length.** On MATH-500 with R1-Distill-Llama-8B (200 problems, seed-paired sampling), CVSA-c4 (M = 4) reaches 64.0% against dense 80.5%. Refreshing the selection every step (M = 1) gives 75.0% (+11.0 points, CI [+5.5, +16.5]). Cap-outs fall from 31 to 13 of 200 problems. Length inflation falls from +61% to +20% above dense. A 5.5-point gap remains [checked: 2611.10634 App. P]. The paper does not give the sampler, the cap, or the dense cap-out count. The paper also says long chain-of-thought is out of its scope [checked: 2611.10634 Sec 1].

11. **From attention error to logits.** The head error of fact 2 enters the residual stream. Later layers, which are nonlinear, turn it into a change of the output logits. There is no tight formula for this chain. CVSA's guarantee covers one scoring step of one head, and it calls the compositional analysis an open question [checked: 2611.10634 App. P]. One clean step can be derived. If every logit moves by at most ε, then for any token set A with dense probability a, the sparse probability is at most a·e^ε / (a·e^ε + (1−a)·e^−ε) = σ(logit(a) + 2ε). So TV(p_dense, p_sparse) ≤ max_x [σ(x + 2ε) − σ(x)] = σ(ε) − σ(−ε) = tanh(ε/2) ≤ ε/2. A greedy choice cannot flip if the gap between the top two logits is larger than 2ε. (Derivation mine; a random search over 200,000 cases never exceeded the bound.)

12. **From logits to a changed token: the dose.** At one step, two samplers that share their random numbers in the best possible way emit different tokens with probability exactly TV(p_dense, p_sparse) = ½·Σ_x |p_dense(x) − p_sparse(x)|. With any other way of sharing random numbers the chance is at least this large. An exact verifier rejects a sparse draft token with the same probability (sibling pack `specdec_exact`, fact 4: acceptance = 1 − TV). So the rejection rate of a sparse self-draft measures the per-step token-change rate of the sparse read, on a history that the verifier keeps dense. An unverified sparse run has no such clean history: its own tokens and KV drift, so its per-step rate can be higher.

13. **Teacher-forced and closed-loop numbers are different things.** Teacher-forced: feed the dense run's tokens, compare next-token distributions step by step. Closed-loop: let the sparse run write its own tokens and KV, then look at accuracy and length. In kv-fetch H1, Qwen3.5-4B moves 7.4-7.5× less in KL per unit of missing attention than Qwen3-4B (for example KL 0.011 at 22.9% missing vs 0.035 at 14.1%). Online the advantage is gone: prompt-protected SnapKV at 1K loses 40.0 points on Qwen3.5 and 38.8 on Qwen3-4B, both by not finishing. "The offline KL advantage does not survive long generation" [checked, local log: EXPERIMENTS.md H1 (H-b offline), H3].

14. **Two things a sparse step leaves behind.** (1) The emitted token, which may differ from what the dense model would emit. (2) The KV entries of the new position. The KV of layer ℓ is computed from the input of layer ℓ, which is the output of layers 0 … ℓ−1. So the first-layer KV depends only on the token embedding and is always exact for a given token. If layers 0 … j−1 read densely from exact KV, the KV of layers 0 … j is exact. KV above the first sparse layer is approximate even when the token is the same. (Derivation; Exercise 3 shows it.) Part C of the research question asks which of the two carries the harm.

15. **Verification can close both channels.** S3R's verification "operates on dense historical KV and recomputes the proposed block through all layers"; it "replaces provisional draft states with recomputed dense KV entries, and removes the unused candidate suffix". The paper warns that "Full attention alone is not sufficient if it reuses approximate historical states" [checked: 2610.13202 Sec 4]. SparseSpec's verify pass also feeds the draft tokens through a full-attention pass. That this pass overwrites the KV written by the draft is an inference from the cache positions, not a traced fact [unverified; from the sibling pack `specdec_exact`, fact 19]. ReSA takes a middle road. After every f sparse steps, "we batch these recent tokens and re-encode them using dense attention to reconstruct an updated KV cache" [checked: 2506.04108 Sec 2.2]. It keeps the tokens: it "does not rely on output logits to make per-token accept / reject decisions" [checked: 2506.04108 App. B].

16. **A self-draft with a sparse KV budget.** One model plays both roles. Draft: k steps with sparse reads over the set from fact 7(b), plus the provisional KV of the draft tokens themselves. Verify: one dense pass over the k tokens. S3R uses the full KV cache for the first block, because no verifier attention exists yet [checked: 2610.13202 Sec 4]. SparseSpec's cold start does the same in its own way: it loads the whole cache as "recent" tokens and selects nothing, so the first draft reads the full cache [checked, local code: `pillar.py`, `from_kv_cache`]. Typical settings: S3R k = 6 and s = 0.05 [checked: 2610.13202 Sec 6.1 / App. C.2]; SparseSpec s = 0.05 [checked: 2512.01278 Sec 5.4], with a floor of 128 tokens as the code default [checked, local code: `pillar.py`]. S3R's budget sweep uses a floor of 192 tokens [checked: 2610.13202 App. F.2.3].

17. **Acceptance by position inside a draft.** "For Qwen3-8B at step 20, acceptance falls from 95.8% for the first proposed token to 67.9% for the sixth" [checked: 2610.13202 App. E.2, Fig 9]. Qwen3-1.7B: 95.0% → 62.9% [same]. The paper does not say whether these are conditional rates or survival rates P(first i tokens all accepted). If acceptance were independent with rate 0.958, the sixth survival rate would be 0.958⁶ = 0.773, not 0.679. If the figures are survival rates, the mean conditional rate of positions 2-6 is (0.679/0.958)^(1/5) = 0.934 (my arithmetic). Either way, later positions do worse than the first. Three causes are mixed in this curve: the selection gets older, the query moves away from the block the selection came from, and draft token j reads the sparse-written KV of draft tokens 1 … j−1.

18. **Budget sweep and speed.** A larger budget buys acceptance and costs draft time. In S3R, "Increasing s from 0.05 to 0.50 raises acceptance from 80.4% to 96.7%, while speedup falls from 1.370× to 1.058×"; s = 0.03 lowers both [checked: 2610.13202 Sec 6.4, Fig 6]. The time model is T_SD = k·T_q + T_V per round and speedup η = T_p / (T_SD / N_round), where T_q is one draft step, T_V the verify pass, T_p one dense step [checked: 2610.13202 Sec 5 Eq 5]. The paper sets the tokens per round to 1 + k·α for acceptance α (it writes τ; this pack writes N_round) [checked: 2610.13202 App. F.1]. One measured point (batch 64, 8K context, k = 6): 7 dense steps take 129.85 ms, the 6 draft steps 42.51 ms, the verify pass 26.68 ms [checked: 2610.13202 App. F.2.2 Table 7]. So T_q = 0.38 T_p and T_V = 1.44 T_p (my arithmetic; Q14). A draft step is much more than 5% of a dense step, and the verify pass is more than one dense step. SparseSpec counts attention KV reads only. Per emitted token, relative to dense decoding, they are (k·s + 1)/(k·α + 1) [checked: 2512.01278 Sec 3.2]. The reduction factor is the inverse. Its printed example (k = 16, α = 0.75, s = 0.05 → 6.78×) does not match this formula, which gives 13/1.8 = 7.22× (my arithmetic; 6.78× would fit α = 0.70).

19. **Without a verifier.** Two kinds of harm are reported. In RL training: "Direct sparse rollouts collapse early at s=0.05 and remain unstable at s=0.7 and s=0.8. Only s=0.9 sustains relatively stable learning" [checked: 2610.13202 Sec 6.3]. This sentence is about training reward, not about length. In inference, other papers report longer reasoning under sparse decoding: at a 1,024-token budget on Qwen3-8B, length vs full attention is 1.29× (AIME25) for Quest and 1.75× for TriAttention [from notes: 2610.19026]; Quest at a 2k budget on Qwen3-8B AIME24 gives 13.3% accuracy with 30.0k tokens against 74.5% with 15.1k for full attention [from notes: 2506.08889].

20. **Worked example: kv-fetch E3 and H3.** Qwen3-4B, 30 AIME problems × 8 samples, 16K cap, one page set per layer. Accuracy / share that hit the cap / accuracy among finished samples: full 48.3% / 41.3% / 79.4%; Quest@512 22.5% / 69.6% / 72.6%; ours@512 33.3% / 59.6% / 75.3%; oracle@512 38.3% / 57.1% / 79.6% [checked, local file: `results/e3_explore.json`]. Most of the loss is unfinished reasoning, not wrong answers: of Quest@512's 25.8 lost points, about 22.5 come from fewer finished samples (my arithmetic, Q12). The oracle still loses 10.0 points to full attention (CI [−15.8, −4.2]) [same file]. On Qwen3.5-4B with a 32K cap, ours@1024 (41.25%) is 2.1 points above oracle@1024 (39.2%), CI [−0.8, +5.0]: "minimum missing attention per step is not maximum accuracy" [checked, local log: EXPERIMENTS.md H3]. Caveats from the log: the cap censors lengths; budgets count attended KV only, and at equal total GPU memory ours@512 should be compared with Quest@1K, which wins (40.4% vs 33.3%); a per-head oracle and a 32K cap for Qwen3-4B were not run [checked, local log: EXPERIMENTS.md E3 caveats].

---

## 2. Level ladder

- **Level 2.** Can say what a sparse read does to the softmax (the kept weights are renormalized). Can define coverage, budget, page and reuse window. Can say why the selection should depend on the query. Can follow the error identity when it is shown.
- **Level 3.** Can compute coverage, the head error, Quest's bound and its memory fraction, and a ratio budget with a floor from given numbers. Can say which KV entries stay exact under a sparse read. With a hint, can explain why a reused selection fails on some steps and why the acceptance of a sparse draft falls with position.
- **Level 4.** Derives the identity and the logit-to-TV bound without help. Separates the oracle ladder (token, page, shared set) and says what each oracle is a ceiling for. Reads an online result correctly: splits an accuracy loss into "did not finish" and "finished but wrong", and names the selection bias and the cap censoring. Converts between acceptance by position, survival and tokens per round.
- **Level 5.** Designs experiments that separate causes: budget error from staleness, the token channel from the KV channel, per-step cost from lasting damage. Predicts when two methods with the same per-step dose will give different length inflation, and says how to test it. Spots what a paper's numbers cannot show (missing cap counts, proxy coverage instead of true coverage, teacher-forced numbers used for online claims).

---

## 3. Misconceptions

1. **"Sparse attention is like a ReLU on the attention weights: the small ones become 0 and the rest stay the same."** The softmax runs again over the kept set, so every kept weight grows by 1/c (fact 1). Even if all missed values were zero vectors, the output would change: the error is (1−c)·‖O_S‖, not 0 (Exercise 1, part (c): 0.4240). What is right in the ReLU picture: the dropped weights are exactly 0, and the dropped tokens are simply not read. What is wrong: the kept weights do not stay the same, and the rule is not a fixed threshold. Top-k keeps the k largest weights of this step, however small or large they are. If a learner says "it is like a ReLU, the dropped values are 0", agree with the first half, then ask Q1's follow-up.

2. **"Coverage tells you the error."** Coverage is one of two factors (fact 2). With the same set and the same coverage, the error goes from 0.45 to 0.00 when only the values of the missed tokens change (Exercise 1, part (c)). A head with 99% coverage can have a larger error than a head with 92% (Q2).

3. **"The oracle is the best possible sparse attention."** It maximizes coverage at each step under its own restrictions (pages, one set per layer). Accuracy is a closed-loop quantity. In H3 a real selector scored 2.1 points above the oracle at 1K (inside the noise: CI −0.8 to +5.0), and in E3 the oracle at 512 still lost 10.0 points (fact 20).

4. **"Quest's score estimates how much attention a page gets."** It is an upper bound on the largest logit inside the page (fact 5). It mixes the best channel values of different keys, so it is loose, and it is looser for pages whose keys are spread out (Exercise 1: mean slack 4.70 for wide pages against 2.47 for normal pages).

5. **"Attention changes slowly, so a selection that was good one step ago is still good."** On most steps this is true. The loss sits in the tail. In Exercise 2, 8.4% of the steps carry 89% of the missing attention at age 3. In CVSA's trace the set of failing heads has Jaccard 0.33 from one step to the next (fact 9).

6. **"A few bad heads cause the failures; give those heads more budget."** Chronic cells give only 11.1% of failure events in CVSA's trace; intermittent cells give 86.4% (fact 9). Extra budget for the chronic heads repairs the first group only. A fixed list cannot follow the second group, because those cells fail on some steps and are fine on others. In the paper's controlled comparison a fixed per-head budget (41.03) did not beat a uniform budget (41.19) [checked: 2611.10634 App. F]. This is one two-sample trace and one small comparison. Treat it as a hint, not as a law.

7. **"A small teacher-forced KL means a small online loss."** Teacher forcing resets the context to the dense tokens at every step. It cannot see changed tokens, sparse-written KV, or longer reasoning (fact 13).

8. **"Sparse decoding hurts reasoning by making the answers wrong."** In E3 and H3 it mostly makes the reasoning longer, so that samples hit the cap. Accuracy among finished samples stays at 73-80% (E3) for every method except SnapKV (fact 20). This number has its own trap: the finished samples are not the same problems in each run (Q13).

---

## 4. Question bank

Each entry: level, type, question, answer key, likely wrong answer, follow-up. Numeric answers were computed by the author and computed again by the examiner with a separate script. Questions marked A, B or C tie to that part of the research question.

### sparse_attention-Q1
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "One attention head and three cached tokens. The dense attention weights are 0.5, 0.3 and 0.2. The values are the numbers 4, 0 and −5. A sparse read keeps only the first two tokens. Which weights does the sparse read use, and what are the dense output, the sparse output and the error?"
- **Answer key:** Coverage c = 0.8. Sparse weights: 0.5/0.8 = 0.625 and 0.3/0.8 = 0.375. Dense output: 0.5·4 + 0.3·0 + 0.2·(−5) = 1.0. Sparse output: 0.625·4 + 0.375·0 = 2.5. Error: 1.5. Check with the identity: O_comp = −5, so (1 − c)·|O_S − O_comp| = 0.2 × 7.5 = 1.5.
- **Likely wrong answer:** "The weights stay 0.5 and 0.3, like a ReLU that sets the small one to 0. Output 2.0, error 1.0."
- **Follow-up:** "A softmax over two tokens gives two weights. What must they sum to? Do 0.5 and 0.3 do that?"

### sparse_attention-Q2
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "Head 1 has coverage 0.92, and the distance between its kept output and its missed output, ‖O_S − O_comp‖, is 5. Head 2 has coverage 0.99, and the same distance is 50. Which head has the larger output error ‖O_S − O_dense‖, and what are the two errors?"
- **Answer key:** Error = (1 − c)·‖O_S − O_comp‖. Head 1: 0.08 × 5 = 0.40. Head 2: 0.01 × 50 = 0.50. Head 2 is worse although its coverage is higher.
- **Likely wrong answer:** "Head 1, because it misses 8% of the attention and head 2 only 1%."
- **Follow-up:** "Suppose every missed token had exactly the value O_S. What is the error at coverage 0.5?" (Zero. Missing mass alone does not make an error.)

### sparse_attention-Q3
- **Level:** 2. **Type:** explain why.
- **Question:** "A 10,000-token text hides a passkey, and the question comes at the end. With a budget of 64 tokens, Quest answers 99% of the cases and H2O answers 1%. H2O keeps the tokens with the highest accumulated past attention. Why does H2O fail here, and what does Quest do differently?"
- **Answer key:** Numbers [checked: 2406.10774 Table 1, LongChat-7b-v1.5-32k]. Before the question arrives, no step needed the passkey, so it collected little attention and H2O evicted it. Importance depends on the query. Quest keeps all KV and scores the pages against the current query, so the passkey page can be chosen when the question asks for it.
- **Likely wrong answer:** "H2O's budget is too small; 64 tokens cannot hold the passkey."
- **Follow-up:** "Quest has the same 64 tokens. Where is the passkey's KV entry at the moment the question is asked, under H2O and under Quest?"

### sparse_attention-Q4
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "Keys have two channels. One page holds two keys: A = (1, 0) and B = (2, 3). The query is Q = (2, −1). Ignore the 1/√d factor. Compute the true logit of each key, and Quest's page score U = Σ_i max(Q_i·m_i, Q_i·M_i), where m and M are the channel-wise minimum and maximum of the page. Why are the two numbers not the same?"
- **Answer key:** Logits: Q·A = 2, Q·B = 4 − 3 = 1. So the largest true logit is 2. m = (1, 0), M = (2, 3). Channel 1: max(2·1, 2·2) = 4. Channel 2: max(−1·0, −1·3) = 0. U = 4. The bound takes channel 1 from B and channel 2 from A. That is the key (2, 0), which is not in the page. U is always ≥ the largest true logit, never below it.
- **Likely wrong answer:** "U = 2, the score of the best key in the page."
- **Follow-up:** "Add a third key C = (−3, −3) to the page. Does any true logit go above 2? What happens to U?" (Q·C = −3. U becomes max(2·(−3), 2·2) + max(−1·(−3), −1·3) = 4 + 3 = 7. A wider page gets a higher score without any better key.)

### sparse_attention-Q5
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "Quest uses pages of 16 tokens. It keeps one minimum key and one maximum key per page, and it reads them for every page at every step. The context is 32,768 tokens and the token budget is 2,048. What fraction of the KV cache does one step load? What is the fraction at 131,072 tokens with the same budget?"
- **Answer key:** Fraction = 1/S + K·S/L [checked: 2406.10774 Sec 3.5], with K·S = 2,048 tokens. At 32,768: 1/16 + 2048/32768 = 0.0625 + 0.0625 = 12.5%. At 131,072: 0.0625 + 0.0156 = 7.8%. The floor is 1/16 = 6.25%: the page summaries.
- **Likely wrong answer:** "2048/32768 = 6.25%, and 1.6% at 131K."
- **Follow-up:** "Two keys of summary for every 16 keys of cache: what share is that if you count keys and values together?" (2 of 32 vectors = 1/16.)

### sparse_attention-Q6
- **Level:** 3. **Type:** predict. **Part C.**
- **Question:** "A model decodes one token with Quest. Quest reads densely in the first two layers and sparsely in all later layers. All KV entries of earlier positions are exact, and the input token is the same as in a dense run. For which layers is the new position's KV entry exactly the same as in the dense run?"
- **Answer key:** Layers 0, 1 and 2 (the first three). The KV of layer ℓ is computed from the input of layer ℓ. Layer 0's input is the embedding. Layers 0 and 1 read densely from exact KV, so their outputs are exact, so the inputs of layers 1 and 2 are exact. Layer 2 reads sparsely, so its output differs, and the KV of layers 3 and above differs.
- **Likely wrong answer:** "Only layers 0 and 1, the dense ones." Or: "All of them, because the token is the same."
- **Follow-up:** "Is the KV of a layer written before or after that layer's attention reads the cache? Which hidden state does it use?"

### sparse_attention-Q7
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "A sparse read moves every output logit by at most ε = 0.05. In the dense run the top token has probability 0.60 and the second has 0.30. Can the greedy choice change? What is the largest possible total variation distance between the dense and the sparse next-token distributions? Then answer the first question again for top probabilities 0.36 and 0.33."
- **Answer key:** The gap between two logits can change by at most 2ε = 0.10. Case 1: log(0.60/0.30) = 0.693 > 0.10, so greedy cannot change. TV ≤ tanh(ε/2) = tanh(0.025) = 0.025 (fact 11). Case 2: log(0.36/0.33) = 0.087 < 0.10, so greedy can flip.
- **Likely wrong answer:** "A change of 0.05 is tiny, so the same token comes out and nothing happens." Or: "TV can be as large as 0.05."
- **Follow-up:** "With sampling at T = 1 and TV = 0.02 at every step, what is the chance that the first 100 tokens are all the same as in the dense run, in the best case?" (0.98¹⁰⁰ = 0.13.)

### sparse_attention-Q8
- **Level:** 3. **Type:** apply to a new case (numeric).
- **Question:** "SparseSpec's draft sets its token budget to max(128, ceil(0.05 × cache length)). The budget has two parts. Part one is a recent window that is always kept: the tokens that the last round emitted, plus one more. Part two is chosen by a Top-K over the verifier's attention scores. The cache holds 2,000 tokens and the last round emitted 6 tokens. How many tokens does the Top-K choose? What share of the cache does the draft read? From which cache length on is the share really 5%?"
- **Answer key:** [checked, local code: `pillar.py` `get_select_metadata`; `spec_scheduler.py` `verify_outputs`] ceil(0.05 × 2000) = 100, so the budget is 128. Recent window: 6 + 1 = 7 tokens. Top-K chooses 128 − 7 = 121. Share: 128/2000 = 6.4%. The ratio takes over when 0.05 × length ≥ 128, that is from 2,560 tokens. Note for the tutor: in the code the emitted count is the length of the verifier's output, so it already includes the bonus (or corrected) token. The 128 and the 0.05 are the code defaults.
- **Likely wrong answer:** "100 tokens, 5%."
- **Follow-up:** "What is the share at 1,000 cached tokens?" (12.8%. Short contexts are drafted with a much denser read than '5%' suggests.)

### sparse_attention-Q9
- **Level:** 3. **Type:** find the flaw.
- **Question:** "A colleague says: 'Sparse attention fails in a few bad heads. Find them offline and give them a larger budget.' In one trace with a fresh selection at every step, (layer, head) cells that fail on at least 80% of steps are 1.6% of all cells and give 11.1% of the failure events. Cells that fail on 5% to 80% of steps give 86.4%. The set of failing heads has Jaccard overlap 0.33 between two consecutive steps. What is wrong with the plan?"
- **Answer key:** Numbers [checked: 2611.10634 App. F; one two-sample trace]. The "few bad heads" picture is wrong. Chronic cells give only 11.1% of the failure events. Most events (86.4%) come from cells that fail on some steps and not on others, and the set of failing heads changes from one step to the next (Jaccard 0.33). A fixed list spends budget on steps where a cell is fine, and it misses cells that fail only sometimes. So the main gain needs a decision at run time, per step. The plan is not useless: a larger fixed budget also helps an intermittent cell on the steps where it fails. But the paper's controlled test found no gain: an offline per-head budget scored 41.03 and a uniform budget with about the same read fraction scored 41.19 (LongBench subset, no coverage gate). This is one small comparison; in an earlier table with unmatched settings the offline budget was 1.36 points above the uniform one [checked: 2611.10634 App. F].
- **Likely wrong answer:** "Nothing is wrong; 1.6% of cells is small, so the fix is cheap."
- **Follow-up:** "If you fix every chronic cell perfectly, what share of the failure events is left?" (About 89%.)

### sparse_attention-Q10
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "A model drafts 6 tokens with sparse attention and then checks them with full attention. The reported acceptance is 95.8% for the first draft token and 67.9% for the sixth. Suppose each draft token were accepted independently with probability 0.958. What share of drafts would then have all six tokens accepted? Compare with 67.9%. What does the comparison tell you?"
- **Answer key:** Numbers [checked: 2610.13202 App. E.2]. 0.958⁶ = 0.773. The reported 67.9% is lower. So even if 67.9% is a survival rate, the later tokens are accepted less often than the first: the mean conditional rate of positions 2-6 would be (0.679/0.958)^(1/5) = 0.934. If 67.9% is a conditional rate, the drop is much larger. The paper does not say which it is. Reasons for the drop: the selection is fixed while the query moves on, and later draft tokens read the sparse-written KV of earlier draft tokens.
- **Likely wrong answer:** "67.9% is the acceptance rate of the sixth token; later text is simply harder." (This skips the question of what the number is conditioned on, and it ignores that the selection ages inside the draft.)
- **Follow-up:** "At the sixth draft step, how old is the attention pattern that the selection was taken from, counted in tokens?" (The scores come from the verify pass over the previous block. The newest query in that block is about 6 tokens back, the oldest about 12. So the pattern is 6 to 12 tokens old.)

### sparse_attention-Q11
- **Level:** 4. **Type:** find the flaw.
- **Question:** "A student writes: 'The oracle keeps the pages with the most true attention, so no real selector can score higher than the oracle, and the oracle's loss is the price of the budget alone.' In one experiment at a 1,024-token budget the oracle scored 39.2% and a real selector 41.25% (difference +2.1 points, 95% interval −0.8 to +5.0). At 512 tokens the oracle lost 13.8 points to full attention. Find two flaws in the student's sentence."
- **Answer key:** Numbers [checked, local log: EXPERIMENTS.md H3]. (1) The oracle maximizes coverage at each step. Accuracy depends on the whole run, and coverage is only one factor of the error (fact 2). The +2.1 is inside the noise, so it does not prove that the real selector is better. But the data show no ceiling either, and nothing in the definition makes the oracle a ceiling for accuracy. (2) This oracle is restricted: 16-token pages and one set per layer for all KV heads. A per-head or per-token oracle has higher coverage (fact 6), so its loss is not "the price of the budget alone". Also acceptable: the oracle runs on its own trajectory, so its attention is "true" only for the text it wrote itself.
- **Likely wrong answer:** "The +2.1 is noise, so the sentence is fine."
- **Follow-up:** "Which quantity does this oracle maximize, and over which choices? Name one choice it is not allowed to make."

### sparse_attention-Q12
- **Level:** 4. **Type:** apply to a new case (numeric).
- **Question:** "Each run has 240 samples and a 16K-token cap. Full attention: accuracy 48.3%, 41.25% of samples hit the cap, accuracy among finished samples 79.4%. Quest at a 512-token budget: accuracy 22.5%, 69.6% hit the cap, accuracy among finished samples 72.6%. Split the 25.8-point loss into two parts. Part 1: fewer finished samples, counted at full attention's accuracy among finished samples. Part 2: lower accuracy among the samples that finish under Quest. What do you conclude?"
- **Answer key:** Numbers [checked, local file: `results/e3_explore.json`]. Finish rates: 0.5875 and 0.3042. Part 1: (0.5875 − 0.3042) × 0.794 = 22.5 points. Part 2: 0.3042 × (0.794 − 0.726) = 2.1 points. The rest, 1.2 points, is capped samples that were still marked correct (4 of 240 for full, 1 for Quest). So about 87% of the loss is reasoning that did not finish. (The split depends on the order. Counting the lost finishers at Quest's 72.6% instead gives 20.6 and 4.0 points, about 80%. The conclusion is the same.) The mean length is only 14% longer (14,149 vs 12,385 tokens), but the cap cuts the long tail, so 14% is a lower bound.
- **Likely wrong answer:** "Sparse attention makes the model compute wrong answers: accuracy is halved."
- **Follow-up:** "If sparse attention mainly corrupted the answers, what would the accuracy among finished samples look like? It is 72.6% against 79.4%."

### sparse_attention-Q13
- **Level:** 4. **Type:** find the flaw.
- **Question:** "Same data as before: with full attention 59% of samples finish and 79.4% of those are correct. With Quest at 512 tokens 30% finish and 72.6% of those are correct. A report says: 'Among samples that finish, sparse attention costs only 7 points.' What is wrong with this comparison?"
- **Answer key:** The two finished sets are different samples. Under Quest only 30% finish. These are more likely the easier problems and the shorter solutions. On that easier subset, full attention would score above 79.4%. So the true cost on comparable samples is larger than 7 points. A fair version pairs by problem: compare only problems (or problem-seed pairs) that finish under both, or model finishing and correctness together. The cap adds a second bias: with a higher cap some capped samples would finish, right or wrong.
- **Likely wrong answer:** "Nothing; conditioning on finished samples removes the truncation effect."
- **Follow-up:** "Take the 73 samples that finish under Quest. Would you expect full attention to score 79% on exactly those problems, or more?"

### sparse_attention-Q14
- **Level:** 4. **Type:** apply to a new case (numeric).
- **Question:** "A self-speculative system drafts k = 6 tokens per round with a 5% KV budget. Then one dense pass verifies them. A measurement at batch 64 and 8K context gives three times: 7 dense decode steps take 129.85 ms; the 6 sparse draft steps take 42.51 ms together; the verify pass takes 26.68 ms. A round emits N_round = 1 + 6α tokens, where α is the acceptance rate. (a) How long are one draft step and the verify pass, in units of one dense step T_p? (b) Which acceptance rate is needed to break even with dense decoding? (c) An end-to-end run at the same batch and context reports α = 0.804 and a speedup of 1.370×. What speedup do the three times predict for α = 0.804? What do (a) and (c) tell you?"
- **Answer key:** Times [checked: 2610.13202 App. F.2.2 Table 7]; acceptance and speedup [checked: 2610.13202 Sec 6.4]. (a) T_p = 129.85/7 = 18.55 ms. T_q = 42.51/6 = 7.09 ms = 0.38 T_p. T_V = 26.68 ms = 1.44 T_p. (b) One round costs 42.51 + 26.68 = 69.19 ms = 3.73 T_p. Break-even: 1 + 6α = 3.73, so α = 0.455 (the table prints 45.5%). (c) N_round = 1 + 6 × 0.804 = 5.824. Predicted speedup = 5.824/3.73 = 1.56×. The measured 1.370× is lower. The table says why: its sum leaves out idle iterations and the mixing of draft and verify requests in one batch. Lessons. A draft step that reads 5% of the KV still costs 38% of a dense step, because only the attention read shrinks; the projections and the FFN stay (the paper's cost model: T_q = T_GEMM + s·T_Attn [checked: 2610.13202 App. F Eq 20]). The verify pass costs more than one dense step: it processes k + 1 = 7 tokens per request, and in this measurement it runs without CUDA Graphs. And a sum of stage times is an optimistic estimate of the real speedup. One open detail: the caption of Table 7 gives k = 6 but no model and no budget. Figure 3 shows the same stage costs and is labelled "Qwen3-8B (s=0.05, k=6)" [checked: 2610.13202 Sec 5, Fig 3 caption]. So 5% and Qwen3-8B are very likely, but this is an inference for the table [unverified].
- **Likely wrong answer:** "T_q = 0.05 T_p, because the draft reads 5% of the KV."
- **Follow-up:** "With T_q = 0.05 T_p, T_V = T_p and the same acceptance, what would the speedup be?" (5.824/1.3 = 4.5×. The paper measures 1.37×.)

### sparse_attention-Q15
- **Level:** 4. **Type:** predict (numeric). **Part B.**
- **Question:** "On 200 math problems a sparse-attention method that reuses its page selection for 4 steps scores 64.0%. The dense model scores 80.5%. With a fresh selection at every step the method scores 75.0%. Problems that hit the length cap fall from 31 to 13, and the extra length over dense falls from +61% to +20%. What share of the accuracy gap does the fresh selection recover? If each problem that no longer hits the cap became one more correct answer, how many of the recovered points would that explain? What can this result not tell you about damage that lasts after an error?"
- **Answer key:** Numbers [checked: 2611.10634 App. P]. Gap: 16.5 points. Recovered: 11.0, which is two-thirds. Left: 5.5. Cap-outs: 18 fewer problems = 9.0 points of 200, so up to 82% of the 11.0 recovered points can be explained by finishing alone. Limits: no arm switches back to dense reads, so the result cannot separate "each stale step costs a little" from "one stale step damages what follows". The fresh arm is still sparse and still gated by a proxy, so the residual 5.5 is not a clean "budget only" number. The dense cap-out count, the sampler and the cap are not given.
- **Likely wrong answer:** "Reuse explains everything: refresh every step and the method is as good as dense."
- **Follow-up:** "After the refresh, 13 problems still hit the cap and the text is still 20% longer. Which parts of the method are still not dense in that arm?"

### sparse_attention-Q16
- **Level:** 4. **Type:** connect two ideas. **Part B.**
- **Question:** "Offline, with the dense run's tokens fed in at every step, model X changes its next-token distribution 7 times less than model Y for the same amount of missing attention. Online, with free generation and KV eviction, both models lose about 40 accuracy points, and both lose them by hitting the length cap. Give two reasons why the offline number did not predict the online result."
- **Answer key:** Numbers [checked, local log: EXPERIMENTS.md H1, H3]. (1) Teacher forcing resets the context at every step. It measures the per-step cost only. Online, a changed token stays in the context and the model continues from it; KV written under the lossy read stays too. A small per-step cost can still add up over 10,000 or more steps, because one changed token is enough to leave the dense path. (2) The mean KL hides the tail: the steps where the needed token is missing are few, and with eviction the token is gone for good. (3) Also acceptable: the offline test used sequences of at most 8K tokens and the comparison is confounded (8 perturbed layers against 36). This is the per-token-cost versus lasting-damage split of Part B: the offline metric sees only the first.
- **Likely wrong answer:** "The offline measurement must be wrong or noisy."
- **Follow-up:** "In the teacher-forced test, what happens at step t+1 to an error made at step t?"

### sparse_attention-Q17
- **Level:** 5. **Type:** predict. **Part A.**
- **Question:** "Two lossy methods have the same per-step total variation distance to the exact model, 0.03 on average. Method 1 is relaxed speculative verification with a separate small draft model. Method 2 is sparse attention with a page selection that is reused for 4 steps. One hypothesis says that the extra reasoning length depends only on this dose. What does the hypothesis predict for the two methods? Give two properties of method 2 that could make the prediction fail, and one experiment that would test it."
- **Answer key:** The dose hypothesis predicts the same length inflation. Reasons it may fail for sparse attention: (a) the errors are not spread evenly; they sit on the steps where the model needs a token outside S, so they hit exactly the steps that look something up (Exercise 2: few steps carry most of the missing mass); (b) they persist: the same token stays invisible for the whole reuse window, so the model may fail to find the same fact several times; (c) the error is "could not see X", not "a random other token": the model may re-derive X, which adds length without any single unlikely token; (d) sparse-written KV stays in the cache (method 1 leaves none if its verifier rewrites KV). Test: build a control with the same per-step TV but no hidden context, for example mix the dense distribution with noise, or relax verification until TV matches, and compare length under paired seeds. Also sweep M at fixed budget and plot length against measured dose: if the points of all methods fall on one curve, dose is enough. A level-5 answer also says how to measure the dose (teacher-forced TV, or the rejection rate of an exact verifier) and names the cap censoring.
- **Likely wrong answer:** "Same dose, same damage" with no test. Or: "Sparse attention is worse because it is less accurate", which ignores that the dose was matched.
- **Follow-up:** "In method 2, is the chance of an error at step t independent of what the model is trying to do at step t?"

### sparse_attention-Q18
- **Level:** 5. **Type:** apply to a new case. **Part C.**
- **Question:** "You have a sparse-decoding engine that keeps the full KV cache on the GPU and can also run a dense read at any step. You want to know whether sparse attention makes reasoning longer through the tokens it emits or through the KV entries it writes. Design the arms of the experiment. Say for each arm which read chooses the token and which read writes the KV."
- **Answer key:** A 2 × 2 design. (i) Dense token, dense KV: the baseline. (ii) Sparse token, sparse KV: plain sparse decoding. (iii) Sparse token, dense KV: sample the token from the sparse read, then recompute that position's KV with a dense read (two passes per step, or a dense re-pass every few steps, as ReSA does [checked: 2506.04108 Sec 2.2]). Only the token channel is open. (iv) Dense token, sparse KV: sample from the dense read but store the KV from a sparse pass. Only the KV channel is open. Note that later sparse reads in arm (iii) still read sparsely; the arm removes stored damage, not the current-step error. Good answers add: first-layer KV is exact in every arm (fact 14); use paired seeds; report cap rate and length with the censoring handled; match the selector and budget across arms; arm (iv) needs the engine to run both reads at each step. Exercise 3 is the toy version: with tokens fixed, stored sparse KV alone gives a small but lasting error, and rewriting the KV removes it completely.
- **Likely wrong answer:** "Compare sparse decoding with and without a verification pass." (That changes both channels at once: S3R-style verification replaces the tokens and the KV.)
- **Follow-up:** "In your 'with verification' arm, list everything that differs from plain sparse decoding. Is it one thing or two?"

### sparse_attention-Q19
- **Level:** 5. **Type:** apply to a new case. **Part B.**
- **Question:** "A reasoning run uses sparse reads from step 2,000 to step 2,400 and dense reads before and after. You want to separate two ideas. Idea 1: each sparse step has a small cost, and the cost ends when dense reads return. Idea 2: the sparse steps leave damage that keeps costing later. What do you measure, what do you compare it with, and what does each idea predict?"
- **Answer key:** Measure the remaining length after step 2,400 (and the cap rate), over many paired samples. Compare three continuations from step 2,400: (a) the run as it is (sparse-written tokens and KV both stay); (b) the same tokens, but the KV of steps 2,000-2,400 recomputed densely; (c) a dense run from the start with the same seed. Idea 1 predicts that (a) and (b) need about as many further tokens as (c) needs after the same amount of progress; all extra length sits inside the window and grows in proportion to the window size. Idea 2 predicts extra length after 2,400. If (a) > (b), stored KV carries part of it; if (b) > (c), the changed tokens carry it. A dose-response check: vary the window from 100 to 800 steps; under Idea 1 the total extra length is linear in the window with no offset. Problems to name: "same progress" is hard to define once the token paths differ, so use many samples and compare distributions; the cap censors; the window should start at the same step in all arms.
- **Likely wrong answer:** "Compare the total length of the windowed run with the dense run." (The total mixes the cost inside the window with the cost after it.)
- **Follow-up:** "Your total-length difference is +300 tokens. How many of those were produced before step 2,400 and how many after? Can your measurement tell?"

---

## 5. Exercises

All three are CPU-only numpy, each under 30 minutes (each script runs in about 2 seconds or less). The examiner wrote these scripts again and ran them on 2026-11-13 (Python 3 with numpy, no GPU). The outputs below are pasted from those runs. The scripts and outputs are saved in `/tmp/claude-1025/-home-ryankert-l-kv-fetch-kv-fetch/b232f66b-c1cf-477e-b08d-10c5771f3def/scratchpad/exam_sparse_attn/` as `ex1.py`, `ex2.py`, `ex3.py` and `ex1.out`, `ex2.out`, `ex3.out`. The models are random toys. They show mechanisms. They are not evidence about any real model or method.

### Exercise 1. Coverage, the error identity, and page selectors

**Task.** One head, 4,096 cached tokens in 256 pages of 16, budget 256 tokens (16 pages). Keys of one page share a topic direction. 40 pages have keys that are spread more widely. 12 single tokens ("needles") get an extra push toward the query. (a) Compute the dense attention. For five selectors (oracle tokens, oracle pages, Quest pages, mean-key pages, the last 256 tokens) compute the coverage and both sides of the identity ‖O_S − O_dense‖ = (1−c)·‖O_S − O_comp‖. (b) Check that Quest's score is an upper bound on the largest logit in every page, and compare its slack on normal and on wide pages. (c) Keep the oracle page set fixed and change only the values of the missed tokens: as drawn, scaled by 3, all zero, all equal to O_S.

**Expected result.** The identity holds to rounding error for every selector (gap about 5e-15). Coverage: oracle tokens 0.887, oracle pages 0.850, mean-key pages 0.823, Quest pages 0.775, last 256 tokens 0.019. Quest's bound holds on every page (smallest slack 0.349, never negative). Its slack is larger on wide pages (4.70 against 2.47), and Quest spends 11 of its 16 pages on wide pages while oracle pages spends 4. In part (c) the coverage stays 0.8501 in all four cases, but the error changes: 0.4485, 0.5416, 0.4240 and exactly 0. With all missed values zero the error is still 0.4240, which is (1−c)·‖O_S‖.

**How the tutor checks.** The learner must predict part (c) before running it. Ask: "With all missed values equal to zero, is the error zero? Why not?" (Renormalization: the sparse read returns O_S, not c·O_S.) Ask: "Why does Quest pick wide pages?" (min and max are further apart, so the bound is higher.) Ask the learner to say which of the toy's results could be an artifact of how the toy was built. (The Quest-versus-mean-key ranking is: the wide pages and the needles were put there by hand.)

```python
import numpy as np
rng = np.random.default_rng(0)
n, ps, B, d, dv = 4096, 16, 256, 32, 16          # tokens, page size, budget (tokens), key dim, value dim
P, KP = n // ps, B // ps                         # 256 pages, 16 pages fit in the budget
page = np.arange(n) // ps

q = rng.normal(size=d)
topic = 3.0 * rng.normal(size=(P, d))                # one topic direction per page
wide = np.zeros(P, bool); wide[rng.choice(P, 40, replace=False)] = True
spread = np.where(wide, 1.0, 0.5)[page]          # wide pages: keys spread 2x more around the topic
keys = topic[page] + spread[:, None] * rng.normal(size=(n, d))
needles = rng.choice(n, 12, replace=False)       # 12 single tokens that match the query well
keys[needles] += 1.6 * q
vals = rng.normal(size=(P, dv))[page] + rng.normal(size=(n, dv))

z = keys @ q / np.sqrt(d)                        # logits
alpha = np.exp(z - z.max()); alpha /= alpha.sum()   # dense attention
O_dense = alpha @ vals

def read(S, v=vals):                             # S: boolean mask of kept tokens
    c = alpha[S].sum()
    O_S = alpha[S] @ v[S] / c                    # softmax over S only = kept weights / c
    O_comp = alpha[~S] @ v[~S] / (1 - c)
    return c, np.linalg.norm(O_S - alpha @ v), (1 - c) * np.linalg.norm(O_S - O_comp), O_S

def pages_to_mask(score):                        # keep the KP pages with the highest score
    return np.isin(page, np.argsort(-score)[:KP])

mass = np.bincount(page, weights=alpha, minlength=P)          # true attention mass per page
kp = keys.reshape(P, ps, d)
m, M = kp.min(1), kp.max(1)                                    # channel-wise min and max per page
U = np.maximum(q * m, q * M).sum(1) / np.sqrt(d)               # Quest's upper-bound score
maxlogit = z.reshape(P, ps).max(1)
tok = np.zeros(n, bool); tok[np.argsort(-alpha)[:B]] = True
sel = {"oracle tokens": tok, "oracle pages": pages_to_mask(mass), "Quest pages": pages_to_mask(U),
       "mean-key pages": pages_to_mask(kp.mean(1) @ q), "last 256 tokens": np.arange(n) >= n - B}

print("(a) selector          coverage  ||O_S-O_dense||  (1-c)||O_S-O_comp||   gap     needles kept  wide pages")
for name, S in sel.items():
    c, lhs, rhs, _ = read(S)
    nw = "-" if name == "oracle tokens" else str(int(wide[np.unique(page[S])].sum()))
    print(f"    {name:16s}  {c:.4f}      {lhs:.4f}           {rhs:.4f}        {abs(lhs-rhs):.1e}   {int(S[needles].sum()):2d} of 12     {nw}")
print(f"    attention mass on the 12 needles: {alpha[needles].sum():.4f}; on the 40 wide pages: {mass[wide].sum():.4f}")

slack = U - maxlogit
print(f"(b) Quest bound: smallest slack over all pages {slack.min():.3f} (a negative number would break the bound)")
print(f"    mean slack: normal pages {slack[~wide].mean():.2f}, wide pages {slack[wide].mean():.2f};"
      f"  mean largest logit: normal {maxlogit[~wide].mean():.2f}, wide {maxlogit[wide].mean():.2f}")
r = lambda a: np.argsort(np.argsort(a))
print(f"    rank correlation with the true page mass: Quest score {np.corrcoef(r(U), r(mass))[0,1]:.2f},"
      f" mean-key score {np.corrcoef(r(kp.mean(1) @ q), r(mass))[0,1]:.2f}")

S = sel["oracle pages"]; O_S = read(S)[3]
print("(c) oracle page set fixed; only the values of the missed tokens change")
for name, v in (("as drawn", vals), ("scaled by 3", np.where(S[:, None], vals, 3 * vals)),
                ("all zero", np.where(S[:, None], vals, 0.0)), ("all equal to O_S", np.where(S[:, None], vals, O_S))):
    c, lhs, rhs, _ = read(S, v)
    print(f"    missed values {name:17s} coverage {c:.4f}   error {lhs:.4f}")
print(f"    (1-c)*||O_S|| = {(1 - alpha[S].sum()) * np.linalg.norm(O_S):.4f}")
```

Output:

```
(a) selector          coverage  ||O_S-O_dense||  (1-c)||O_S-O_comp||   gap     needles kept  wide pages
    oracle tokens     0.8865      0.3136           0.3136        4.6e-15   10 of 12     -
    oracle pages      0.8501      0.4485           0.4485        4.8e-15    4 of 12     4
    Quest pages       0.7748      0.7002           0.7002        4.8e-15    3 of 12     11
    mean-key pages    0.8226      0.5008           0.5008        4.4e-15    1 of 12     4
    last 256 tokens   0.0190      5.3112           5.3112        0.0e+00    1 of 12     3
    attention mass on the 12 needles: 0.5285; on the 40 wide pages: 0.1310
(b) Quest bound: smallest slack over all pages 0.349 (a negative number would break the bound)
    mean slack: normal pages 2.47, wide pages 4.70;  mean largest logit: normal 0.56, wide 2.37
    rank correlation with the true page mass: Quest score 0.93, mean-key score 0.97
(c) oracle page set fixed; only the values of the missed tokens change
    missed values as drawn          coverage 0.8501   error 0.4485
    missed values scaled by 3       coverage 0.8501   error 0.5416
    missed values all zero          coverage 0.8501   error 0.4240
    missed values all equal to O_S  coverage 0.8501   error 0.0000
    (1-c)*||O_S|| = 0.4240
```

Reading the output. The "gap" column is the difference between the two sides of the identity: zero up to float rounding. This toy has no recency, so the last 256 tokens are just an arbitrary set of 16 pages (coverage 0.019). The needles are not all heavy. One needle alone holds 0.47 of the attention; the 12 together hold 0.53. Only 4 of the 12 needle pages are among the 16 heaviest pages, so oracle pages keeps 4 needles. Oracle tokens can pick single tokens and keeps 10. Mean-key pages beats Quest pages here (0.823 against 0.775). Do not read this as a result about Quest: the wide pages were built to raise Quest's bound much more than they raise attention mass. They are 40 of 256 pages and hold 0.13 of the mass, yet they take 11 of Quest's 16 pages.

### Exercise 2. Reusing a selection, and what the mean hides

**Task.** One head, 2,048 cached tokens, budget 128. The query drifts slowly (AR(1) with ρ = 0.99) and jumps to a new random direction with probability 0.03 per step. At step t, read with the oracle top-128 set of step t − age, for age 0 to 7. A random readout matrix turns the head output into a next-token distribution. (a) For each age, report the mean missing attention, its 95th percentile, the mean TV to the dense distribution and the greedy flip rate. (b) At age 3, split the steps into "a jump happened since the selection" and the rest. (c) For a reuse window M, the ages cycle 0 … M−1; report the averages for M = 1, 2, 4, 8. (d) Check how well missing attention ranks the TV.

**Expected result.** The mean missing attention grows slowly with age (0.006, 0.034, 0.061, 0.088 for ages 0-3). The 95th percentile jumps: 0.039 at age 1, 0.83 at age 2, 0.98 at age 3. At age 3, the 8.4% of steps with a jump since the selection have mean missing attention 0.93. They carry 89% of all missing attention and 93% of all TV. The other steps miss 0.011. TV and the flip rate grow with M: flip rate 0.001 at M = 1, 0.014 at M = 2, 0.041 at M = 4, 0.094 at M = 8. In this toy, missing attention ranks TV well (Spearman 0.96).

**How the tutor checks.** Ask before the run: "Will the mean or the tail grow faster with age?" Ask after: "A paper reports mean coverage 0.90 for a reused selection. What do you still not know?" (How the missing mass is spread over steps.) Ask: "Why is the 95th percentile small at age 1 and large at age 2?" (A jump since the selection has probability 3% at age 1 and 1 − 0.97² = 5.9% at age 2. Only the second is above 5%.) Ask what the toy leaves out. (New tokens are not added to the cache; there is one head and no later layers; the values are random, which is why coverage alone predicts TV so well here, unlike part (c) of Exercise 1.)

```python
import numpy as np
rng = np.random.default_rng(0)
n, B, d, dv, V, T = 2048, 128, 32, 16, 50, 6000   # cached tokens, budget, key dim, value dim, vocab, steps
rho, p_jump, A = 0.99, 0.03, 8                    # query drift, jump probability, ages 0..7
keys = rng.normal(size=(n, d)); vals = rng.normal(size=(n, dv))
W_out = 1.5 * rng.normal(size=(V, dv))            # toy read-out: head output -> next-token logits

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

q = rng.normal(size=d); jump = np.zeros(T, bool); top = np.zeros((T, B), int)
miss = np.zeros((T, A)); tv = np.zeros((T, A)); flip = np.zeros((T, A), bool)
for t in range(T):
    jump[t] = rng.uniform() < p_jump              # the query jumps to a new direction (a "lookup")
    q = rng.normal(size=d) if jump[t] else rho * q + np.sqrt(1 - rho**2) * rng.normal(size=d)
    alpha = softmax(0.9 * keys @ q)               # dense attention; logits have std about 5
    top[t] = np.argsort(-alpha)[:B]               # oracle top-128 of this step
    p_dense = softmax(W_out @ (alpha @ vals))
    for a in range(min(A, t + 1)):                # read step t with the set chosen at step t - a
        S = top[t - a]; c = alpha[S].sum()
        p_sparse = softmax(W_out @ (alpha[S] @ vals[S] / c))
        miss[t, a] = 1 - c
        tv[t, a] = 0.5 * np.abs(p_dense - p_sparse).sum()
        flip[t, a] = p_dense.argmax() != p_sparse.argmax()

ok = np.arange(T) >= A                            # skip the first steps (no old selection yet)
print("(a) age of the selection | missing attention: mean, 95th percentile | mean TV | greedy flip rate")
for a in range(A):
    print(f"    age {a}:  {miss[ok, a].mean():.3f}  {np.quantile(miss[ok, a], 0.95):.3f}  |  {tv[ok, a].mean():.4f}  |  {flip[ok, a].mean():.4f}")

a = 3                                             # a jump at step t-2, t-1 or t makes the set stale
hit = np.array([jump[max(0, t - a + 1): t + 1].any() for t in range(T)]) & ok
print(f"(b) age 3: steps with a jump since the selection: {hit[ok].mean():.3f} of steps,"
      f" {miss[hit, a].sum() / miss[ok, a].sum():.3f} of all missing attention, {tv[hit, a].sum() / tv[ok, a].sum():.3f} of all TV")
print(f"    mean missing attention: jump steps {miss[hit, a].mean():.3f}, other steps {miss[ok & ~hit, a].mean():.3f}")

print("(c) reuse window M: select at steps 0, M, 2M, ... and reuse | mean missing | mean TV | flip rate | worst 5% of steps: share of TV")
for M in (1, 2, 4, 8):
    age = np.arange(T) % M
    mm, tt, ff = miss[np.arange(T), age][ok], tv[np.arange(T), age][ok], flip[np.arange(T), age][ok]
    worst = np.sort(tt)[-len(tt) // 20:].sum() / tt.sum()
    print(f"    M={M}:  {mm.mean():.3f}  |  {tt.mean():.4f}  |  {ff.mean():.4f}  |  {worst:.2f}")

r = lambda x: np.argsort(np.argsort(x))
print(f"(d) Spearman(missing attention, TV) over steps at age 3: {np.corrcoef(r(miss[ok, 3]), r(tv[ok, 3]))[0, 1]:.2f}")
```

Output:

```
(a) age of the selection | missing attention: mean, 95th percentile | mean TV | greedy flip rate
    age 0:  0.006  0.023  |  0.0033  |  0.0010
    age 1:  0.034  0.039  |  0.0286  |  0.0284
    age 2:  0.061  0.832  |  0.0531  |  0.0559
    age 3:  0.088  0.981  |  0.0777  |  0.0819
    age 4:  0.114  0.995  |  0.1017  |  0.1081
    age 5:  0.139  0.997  |  0.1248  |  0.1348
    age 6:  0.164  0.998  |  0.1478  |  0.1599
    age 7:  0.189  0.999  |  0.1707  |  0.1827
(b) age 3: steps with a jump since the selection: 0.084 of steps, 0.885 of all missing attention, 0.926 of all TV
    mean missing attention: jump steps 0.929, other steps 0.011
(c) reuse window M: select at steps 0, M, 2M, ... and reuse | mean missing | mean TV | flip rate | worst 5% of steps: share of TV
    M=1:  0.006  |  0.0033  |  0.0010  |  0.29
    M=2:  0.019  |  0.0152  |  0.0140  |  0.82
    M=4:  0.046  |  0.0400  |  0.0407  |  0.90
    M=8:  0.099  |  0.0889  |  0.0936  |  0.55
(d) Spearman(missing attention, TV) over steps at age 3: 0.96
```

Reading the output. Almost all of the harm sits in the few steps where the query jumped after the selection was made. The expected share of such steps at age 3 is 1 − 0.97³ = 8.7%; the run gives 8.4%. In part (c) the worst 5% of steps carry 82% to 90% of the TV at M = 2 and M = 4. At M = 8 the share drops to 55%, because more than 5% of the steps are bad by then. The jumps were put in by hand, so the toy shows the mechanism only. It does not say how often a real model jumps.

### Exercise 3. The KV channel alone (ties to Parts B and C)

**Task.** A random 3-layer attention-only decoder runs token by token with a KV cache. The token sequence is fixed (teacher forcing), so no arm can change a token. Arm D reads densely at every step. Arm S reads sparsely (per-step oracle top-32 in every layer) for steps 300-339 and densely again from step 340; the KV written during the window stays. Arm R is arm S plus a dense pass at step 340 that recomputes the KV of positions 300-339. (a) Compare the stored values of the window positions between the arms, layer by layer. (b) Report TV(dense, arm) of the next-token distribution before, inside and after the window. (c) In arm D, measure how much attention the later steps put on the window positions.

**Expected result.** Layer-0 values are identical in all arms. Layers 1 and 2 differ in arm S (relative error 0.047 and 0.070) and not in arm R. Inside the window, S and R are the same run (TV 0.071). After the window, arm S still differs from dense although every read is dense again and every token is the same: TV 0.0067 in the first 20 steps, then 0.0061 and 0.0060. Arm R is exactly dense again (largest difference 0.0). The stored-KV effect is about 10 times smaller than the effect inside the window. It fades very slowly here: 60 to 100 steps later it is still 0.0060.

**How the tutor checks.** Ask before the run: "After step 340 every read is dense and every token is the same. Is arm S equal to arm D?" Then: "Why is layer 0 identical?" (The KV of layer 0 is made from the token and its position only. No attention read comes before it.) Then: "Arm R is exact again after one dense pass. Which assumption makes this true, and when does it fail in free generation?" (The tokens were fixed. If a sparse step had changed a token, no KV rewrite can bring the run back to the dense path.) Then: "Does the stored damage fade?" (Barely, in this toy. Part (c) shows why: later steps keep 7% to 12% of their attention on the window in every layer, because the toy has no recency. In a real model it depends on how much later steps read the damaged positions.) The learner should say what the toy cannot show: with fixed tokens it says nothing about the token channel, and a random model has no reasoning to get longer.

```python
import numpy as np
rng = np.random.default_rng(0)
L, d, V, T, Bk = 3, 32, 50, 440, 32               # layers, width, vocab, steps, sparse budget (tokens)
W0, W1 = 300, 340                                 # sparse reads on steps W0 .. W1-1
SHARP = 3.0                                       # attention sharpness (logit std)
tokens = rng.integers(V, size=T)                  # fixed token sequence: no arm can change a token
E = rng.normal(size=(V, d)); Pos = 0.5 * rng.normal(size=(T, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(L, d, d)) / np.sqrt(d) for _ in range(4))
U = rng.normal(size=(V, d)) / np.sqrt(d) * 3.0    # unembedding
norm = lambda h: h / np.sqrt((h**2).mean())
def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def step(t, K, Vc, sparse, attn_log=None):
    """Decode position t. K, Vc: per-layer lists of stored keys and values. Appends this position's KV."""
    h = E[tokens[t]] + Pos[t]
    for l in range(L):
        x = norm(h)                               # the KV of layer l is made from the INPUT of layer l
        K[l].append(Wk[l] @ x); Vc[l].append(Wv[l] @ x)
        a = softmax(SHARP * np.array(K[l]) @ (Wq[l] @ x) / np.sqrt(d))   # logits have std about SHARP
        if attn_log is not None: attn_log[l] = a
        if sparse:                                # per-step oracle: keep the top-Bk weights, renormalize
            S = np.argsort(-a)[:Bk]; out = a[S] @ np.array(Vc[l])[S] / a[S].sum()
        else:
            out = a @ np.array(Vc[l])
        h = h + Wo[l] @ out
    return softmax(U @ norm(h))

def run(arm):
    K, Vc = [[] for _ in range(L)], [[] for _ in range(L)]; probs, win_attn = [], []
    for t in range(T):
        if arm == "R" and t == W1:                # dense re-pass: recompute the KV of the window positions
            for l in range(L): del K[l][W0:], Vc[l][W0:]
            for u in range(W0, W1): step(u, K, Vc, sparse=False)
        log = {}
        probs.append(step(t, K, Vc, sparse=(arm != "D" and W0 <= t < W1), attn_log=log))
        if arm == "D" and t >= W1: win_attn.append([log[l][W0:W1].sum() for l in range(L)])
    return np.array(probs), [np.array(v) for v in Vc], np.array(win_attn)

pD, vD, wa = run("D"); pS, vS, _ = run("S"); pR, vR, _ = run("R")
print("(a) stored values of positions 300-339: relative error against arm D, by layer")
for name, v in (("S", vS), ("R", vR)):
    print(f"    arm {name}: " + "  ".join(f"layer {l}: {np.linalg.norm(v[l][W0:W1] - vD[l][W0:W1]) / np.linalg.norm(vD[l][W0:W1]):.4f}" for l in range(L)))
tv = lambda p: 0.5 * np.abs(p - pD).sum(1)
print("(b) mean TV(arm D, arm) of the next-token distribution")
for lo, hi, label in ((0, W0, "before the window"), (W0, W1, "inside the window"), (W1, W1 + 20, "after, steps 340-359"),
                      (W1 + 20, W1 + 60, "after, steps 360-399"), (W1 + 60, T, "after, steps 400-439")):
    print(f"    {label:22s} arm S {tv(pS)[lo:hi].mean():.4f}   arm R {tv(pR)[lo:hi].mean():.4f}")
print(f"    largest |p_R - p_D| over all steps from 340 on: {np.abs(pR - pD)[W1:].max():.1e}")
print("(c) arm D: mean attention that later steps put on positions 300-339, by layer")
for lo, hi in ((0, 20), (20, 60), (60, 100)):
    print(f"    steps {W1 + lo}-{W1 + hi - 1}: " + "  ".join(f"layer {l}: {wa[lo:hi, l].mean():.3f}" for l in range(L)))
```

Output:

```
(a) stored values of positions 300-339: relative error against arm D, by layer
    arm S: layer 0: 0.0000  layer 1: 0.0474  layer 2: 0.0697
    arm R: layer 0: 0.0000  layer 1: 0.0000  layer 2: 0.0000
(b) mean TV(arm D, arm) of the next-token distribution
    before the window      arm S 0.0000   arm R 0.0000
    inside the window      arm S 0.0708   arm R 0.0708
    after, steps 340-359   arm S 0.0067   arm R 0.0000
    after, steps 360-399   arm S 0.0061   arm R 0.0000
    after, steps 400-439   arm S 0.0060   arm R 0.0000
    largest |p_R - p_D| over all steps from 340 on: 0.0e+00
(c) arm D: mean attention that later steps put on positions 300-339, by layer
    steps 340-359: layer 0: 0.124  layer 1: 0.110  layer 2: 0.095
    steps 360-399: layer 0: 0.095  layer 1: 0.074  layer 2: 0.077
    steps 400-439: layer 0: 0.076  layer 1: 0.098  layer 2: 0.088
```

Reading the output. Part (a) shows the KV channel: the sparse reads of the window changed what was stored in layers 1 and 2, and a dense read later cannot undo that. Part (b) separates the two effects. Inside the window the read itself is sparse (TV 0.071). After the window only the stored KV is different (TV about 0.006). The numbers depend on the toy's settings (budget 32, attention sharpness 3.0, three layers); only the pattern matters.

---

## 6. Diagnostic picks

- **Level 2-3: sparse_attention-Q2** (two heads, coverage against error). One numeric answer shows whether the learner holds both factors of the identity or only "more coverage is better". If the learner stalls, step down to Q1, which starts from the "ReLU on the weights" picture and shows the renormalization.
- **Level 4-5: sparse_attention-Q18** (token channel against KV channel). A level-4 answer gives the 2 × 2 arms. A level-5 answer also sees that verification changes both channels at once, that first-layer KV is exact in every arm, that the "sparse token, dense KV" arm removes stored damage but not the current-step error, and that the cap censors the length. It maps directly onto Part C.

---

## 7. Sources

Downloaded in this run (raw page from `https://arxiv.org/html/<id>`, one request per paper, converted to text locally and searched on disk). Each id was confirmed against the page text.

| id | title | what I read |
|---|---|---|
| arXiv 2611.10634 (v1) | Coverage-Verified Sparse Attention: Closed-Loop Quality Control for Long-Context LLM Decoding (CVSA) | Sections of the full text: Sec 1 (scope), 2.2, 3.1-3.4, 4.1, Limitations, App. B (proofs, bound), App. C (proxy calibration, in part), App. E (reuse window), App. F (failure trace), App. P. Figures and most tables not read. Examiner pass: also Table 5 (read fraction by M) and the controlled comparison in App. F. |
| arXiv 2610.13202 | S3R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding | Sections: Sec 4 (Eq 4, verification), Sec 5 (Eq 5), Sec 6.1, 6.3, 6.4, App. B.1 (Eq 15, 16), App. C.2, C.3, App. E.2. Figures read through captions and text only. Examiner pass: also App. F.1 (N_round, Eq 20), App. F.2.2 (Table 7, measured stage times), App. F.2.3 (budget sweep setting). |
| arXiv 2406.10774 | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | Sections: Sec 3.3, 3.4 (Algorithm 1), 3.5, Table 1, Sec 4.3.1. |
| arXiv 2512.01278 (v1) | Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding (SparseSpec) | Sections: Sec 3.2, 3.3, 4.1, 5.3, 5.4. The saved HTML page has no title (it starts at "1Introduction"). The examiner took the title from the arXiv abstract page of this id (one request, 2026-11-13). The first version of this pack printed a different title. |
| arXiv 2506.04108 (v2) | Rectified Sparse Attention (ReSA) | Examiner pass, page text on disk: Sec 2.2 (dense re-encoding of recent tokens every f steps), App. B (no per-token accept or reject). |

Read directly on disk in this run:

| source | what I read |
|---|---|
| `/home/ryankert_l/kv-fetch/kv-fetch/EXPERIMENTS.md` | lines 1-262 (E1, E2, E2b, E3 pilot, ablations and caveats, H1, H3, H4) |
| `/home/ryankert_l/kv-fetch/kv-fetch/results/e3_explore.json` | the summary table and the paired comparisons |
| `/home/ryankert_l/kv-fetch/kv-fetch/kvfetch/sparse_decode.py` | docstring, method list, budget and pinned-page lines, `shadow` flag (searched, not read in full) |
| `/home/ryankert_l/kv-fetch/kv-fetch/configs/reasoning.yaml` | cap and budget lines |
| `/home/ryankert_l/SparseSpec/serve/request/kv_cache_ptr/pillar.py` | first 140 lines (budget, recent tokens, cold start) |
| `/home/ryankert_l/SparseSpec/serve/scheduler/spec_scheduler.py` | examiner pass: `verify_outputs`, lines 547-566 (what the accepted-token count includes) |
| `reports/length-tax.html` | full text (notes) |
| `research_notes/Inference speed question search/harvest_specdec_self.md`, `verification.md` | full (notes) |
| `research_notes/Length tax preliminaries/specdec_exact.md` | full; used for format and to avoid repeating its facts |

Cited from notes only, papers not opened in this run and not opened by the examiner [from notes]: 2610.19026 (JSSA; length ratios of sparse methods), 2506.08889 (SeerAttention-R; Quest at 2k on AIME24). Their ids and titles were not confirmed against the arXiv pages.

Open points (also in the "could not verify" list returned with this pack):

- S3R App. E.2 does not define acceptance by position (conditional or survival). Fact 17 and Q10 give both readings. The mean of the first and sixth figures, (95.8 + 67.9)/2 = 81.85, is close to the reported checkpoint rate 81.9%, which fits the survival reading, but this is my inference.
- Q14 uses the measured stage times of S3R App. F.2.2 Table 7 and the paper's own N_round = 1 + k·α. The caption of Table 7 names no model and no draft budget. Figure 3 (same stage costs) is labelled Qwen3-8B, s = 0.05, k = 6, so the pack assumes these for the table.
- SparseSpec Sec 3.2 prints 6.78× for k = 16, α = 0.75, s = 0.05; its formula gives 7.22× (6.78× would fit α = 0.70).
- CVSA's sum 5.0 + 3.7 + 2.9 is 11.6, the paper prints 11.7% (rounding).
- CVSA App. P gives no sampler, no cap and no dense cap-out count. The paper gives two sets of proxy false-accept numbers. In the production probe, 31%, 33% and 39% of proxy accepts are false accepts on Llama, Mistral and Qwen [checked: 2611.10634 App. C table and App. P]. In the smaller stress grid of App. C, 315 of 413 proxy accepts on Qwen are false accepts (76.3%). This pack uses no false-accept number in a question.
- H1 and H3 numbers come from `EXPERIMENTS.md`; `results/h3_explore.json` has no summary table and I did not re-derive them.
- That the kv-fetch engine stores the KV written by sparse passes without a dense re-pass follows from its design (one pass per step); I did not trace the write in the code.
- The bound TV ≤ tanh(ε/2) in fact 11 is my derivation, checked by random search, not taken from a paper.

---

## 8. Verification log

Examiner pass, finished 2026-11-13 15:41 +0800 (time from `date`). A second reader worked every item before reading its key. Numbers were computed again with CPU-only Python. Paper facts were read again in the page text saved on disk. No GPU job was run. Scripts are in `/tmp/claude-1025/-home-ryankert-l-kv-fetch-kv-fetch/b232f66b-c1cf-477e-b08d-10c5771f3def/scratchpad/exam_sparse_attn/` (`arith.py` for the keys, `ex1.py`, `ex2.py`, `ex3.py` for the exercises).

**Checked: 50 items** (20 core ideas, 8 misconceptions, 19 questions, 3 exercises).

**Changed, wrong before:**

1. Exercises 1-3. The code and the output blocks in the first version belonged to other packs (quantization and sampling toys). They did not match the tasks. All three scripts were written again, run, and pasted with their real output. The "Expected result" and "Reading the output" texts now use the new numbers. One claim was dropped (Exercise 2, "TV varies by a factor of 2 inside a narrow band"): it was not measured. One claim was reversed (Exercise 3): the stored-KV effect does not fade quickly in this toy.
2. Numbers taken from the exercises in Misconceptions 1, 2, 4 and 5 were replaced by the new ones (0.4240; 0.45 → 0.00; slack 4.70 against 2.47; 8.4% of steps carry 89% of the missing attention).
3. Q14. The old key said the paper gives no draft-step time and derived T_q = 0.54 T_p from an assumed T_V = T_p. The paper has measured stage times (S3R App. F.2.2 Table 7). The question and key now use them: T_q = 0.38 T_p, T_V = 1.44 T_p, break-even acceptance 0.455, predicted 1.56× against measured 1.370×.
4. Q8. The old key was 128 − 6 = 122. In the code the recent window is the emitted tokens plus one. The question now states this rule, and the key is 128 − 7 = 121.
5. Fact 16. SparseSpec's cold start does not put the budget into the recent window. It reads the whole cache on the first draft. The 128-token floor is a code default, not a number from the paper.
6. Open point on CVSA false accepts. The 31%, 33% and 39% are in the paper (App. C production probe, App. P). The old text said they were not found.
7. Sources. The SparseSpec title was wrong. It is now the title of the arXiv abstract page.

**Changed, ambiguous before:**

8. Fact 19. The heading said "the harm shows up as length", but the S3R quote under it is about RL training reward. The two kinds of evidence are now separated.
9. Q9 and Misconception 6. "A fixed list can only reach about 11%" was too strong. The key now says what the trace shows, that the comparison 41.03 against 41.19 is one small controlled test, and that an unmatched table in the same paper pointed the other way.
10. Q12. The split of the 25.8 points depends on the order of the two parts. The question now names the weighting. The key gives the other order too (20.6 and 4.0 points; same conclusion).

**Changed, minor:**

11. Fact 8: the reuse sweep also changes the read fraction (16.1%, 12.0%, 9.9%). Fact 12: acceptance is measured on a history kept dense; an unverified run can be worse. Fact 15: the KV overwrite in SparseSpec's verify pass is an inference; ReSA facts are now [checked] with quotes. Fact 18: SparseSpec's printed ratio is (k·s + 1)/(k·α + 1), the inverse of what the pack wrote; the Table 7 times were added. Fact 4: the budget formula is marked as code defaults.
12. Q10 follow-up: the age of the pattern is 6 to 12 tokens, not "at least 6 plus the block". Q11 and Misconception 3: the +2.1 points are inside the noise and prove nothing alone. Q18: ReSA tag is now [checked].
13. Misconception 1: two sentences were added on what the ReLU picture gets right (dropped weights are exactly 0) and wrong (kept weights grow by 1/c; top-k is a rank rule, not a threshold).

**Confirmed without change:** facts 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 17, 20; misconceptions 7 and 8; Q1 to Q7, Q13, Q15, Q16, Q17, Q19; the level ladder; the diagnostic picks. The bound TV ≤ tanh(ε/2) was derived again and tested on 200,000 random cases; no case exceeded it.

**Still unverified:**

- 2610.19026 (Quest 1.29×, TriAttention 1.75× length on AIME25) and 2506.08889 (Quest at 2k on AIME24: 13.3%, 30.0k tokens). Both are [from notes]. Neither paper was opened, and neither id was confirmed against its arXiv page.
- Whether S3R's acceptance by draft position is a conditional rate or a survival rate (fact 17, Q10). Both readings are given.
- That S3R Table 7 was measured on Qwen3-8B with s = 0.05. This follows from Figure 3, not from the table's caption.
- That SparseSpec's verify pass overwrites the KV written by the draft (fact 15).
- CVSA gives no sampler, no cap and no dense cap-out count (App. P), so its length numbers cannot be compared with kv-fetch.
- The H1 and H3 numbers come from `EXPERIMENTS.md`. They were not derived again from the raw result files.
- That the kv-fetch engine keeps sparse-written KV without a dense re-pass. The code path was not traced.
- All three exercises are random toys with hand-made structure (needles, wide pages, jumps, fixed tokens). They show mechanisms only.
