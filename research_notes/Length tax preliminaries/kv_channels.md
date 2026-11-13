# Tutor pack: KV cache mechanics and the two error channels

Subskill id: `kv_channels`. Written 2026-11-13 15:11 CST. Read-only work; no GPU job was run. Examined and corrected by a second pass on 2026-11-13; see the Verification log at the end.

Scope: what the KV cache stores per layer and why decode attention is memory-bound; what an approximate step leaves behind (the emitted token, and the KV entries it wrote) and how later steps read both; the two repairs (re-check the token, or rewrite the KV with a dense pass); why SparseSpec's exact verification repairs both at once; what Sirius found about rollback versus KV rewrite.

This file follows `specdec_exact.md` in the same folder. That file covers the accept/reject rule. This one covers the state the rule leaves behind.

Notation in this file:

- Step i takes token x_i as input, **writes** the KV entries of position i, and **emits** token x_{i+1}. So the token an approximate step emits and the KV it writes belong to two different positions.
- Layers are l = 1 … L. h_i^(l−1) is the residual stream that enters layer l at position i. h_i^(0) is the embedding of x_i.
- "Dense" means the full model with full attention. p_i is its next-token distribution at step i on a clean (dense-written) cache. p̃_i is the distribution the approximate step actually samples from.
- KV\*(x_1 … x_i) is the cache the dense model writes for the token sequence x_1 … x_i.

How to read the tags:

- **[checked]**: read in the arXiv HTML page of the paper. The first pass read through a fetch tool that returns extracted answers. The examiner pass (see Verification log) downloaded the raw HTML and read every [checked] fact there again, tables included. All formulas were also re-derived.
- **[checked, local]**: I read it directly on this machine (SparseSpec code, model config files).
- **[from notes]**: taken from the notes in `research_notes/Inference speed question search/` (`verification.md`, `harvest_specdec_core.md`, `harvest_specdec_self.md`) or `reports/length-tax.html`. The paper itself was not re-read. The examiner pass compared each of these with the notes only.
- **[unverified]**: from memory only.
- No tag: textbook fact or a derivation written out here.

---

## 1. Core ideas

1. **What is stored.** At layer l and position i the model computes k_i^(l) = W_K^(l) · norm(h_i^(l−1)) and v_i^(l) = W_V^(l) · norm(h_i^(l−1)), one vector of size d_h per KV head. The cache keeps exactly these: for every layer, every KV head and every past position, one key and one value. It does not keep queries, attention weights, attention outputs, MLP outputs, the residual stream or logits. (With rotary positions, as in Qwen3, the usual engines store the key after it is rotated by its position. Nothing below depends on this.)

2. **How a later step reads the cache.** At step j, layer l, each query head computes a_ji = softmax over i ≤ j of (q_j · k_i / √d_h) and the output o_j = Σ_{i≤j} a_ji · v_i. Here q_j is the query of position j, a_ji is the attention weight on position i, and d_h is the head dimension. An error in k_i changes **where** later steps look. An error in v_i changes **what** they get.

3. **The cache is exact.** The causal mask means k_i and v_i never depend on tokens after i. So decoding with the cache gives the same result as recomputing the whole sequence (Exercise 2a: difference 4·10⁻¹⁶). The cache is a memo, not an approximation.

4. **What each entry depends on.** k_i^(l) and v_i^(l) are functions of the token x_i and of everything position i read from the cache in layers 1 … l−1. So the layer-1 entry depends only on the token (and its position). Entries in layers 2 … L also depend on the attention reads below them. A past token x_i reaches the future in two ways only. First, step i's own forward pass emits x_{i+1}. Second, every later step reads the KV entries of position i. Later positions then carry part of what they read into their own KV entries and into the tokens they emit.

5. **Size.** Bytes per cached token = 2 · L · H_kv · d_h · b, where the 2 is for K and V, H_kv is the number of KV heads and b is bytes per number. For Qwen3-8B (L = 36, 32 query heads, H_kv = 8, d_h = 128, bf16 so b = 2) [checked, local: `config.json`] this is 2·36·8·128·2 = 147,456 bytes = 144 KiB per token. Qwen3-4B has the same four numbers, so the same size [checked, local].

6. **Why decode attention is memory-bound (derivation).** For one cached token in one layer, a decode step does q·k and a·v for each of H_q query heads: 2 · (2·d_h) · H_q = 4·d_h·H_q FLOPs (a multiply-add counts as 2). It reads 2 · H_kv · d_h · b bytes. FLOPs per byte = 2·H_q / (H_kv·b) = H_q/H_kv for bf16 = 4 for Qwen3. A GPU with about 1000 TFLOP/s and about 3 TB/s [unverified, nominal H100-class numbers from memory] can do about 300 FLOPs for each byte it reads. Attention asks for 4, so the memory read sets the time. FlashInfer states the same thing as "operational intensity" O(l_qo) for query length l_qo, and O(g·l_qo) with GQA group size g [checked: 2501.01005 Sec 2.1].

7. **KV traffic scales with the batch; weight traffic does not.** One decode step reads the weights once for the whole batch, but reads each request's KV separately. KV bytes per step = (bytes per token) × Σ over requests of cached length. For a weight matrix in bf16 used by B requests, FLOPs per byte = B. So GEMM gets more compute-heavy as the batch grows and attention never does. SparseSpec says: "MLP is typically compute-bound, while attention is memory-bound" [checked: 2512.01278 Sec 2.1], and reports that KV loading is about 21 ms per step and over 70% of latency for Qwen3-8B on an H100 at batch 128 and output 8192 [checked: 2512.01278 Sec 1]. The same paper adds a limit: "Inference at such small batch sizes is memory-bound for both MLP and attention" [checked: 2512.01278 Sec 3.1]. This fits the formula: at small B the weights also give only B FLOPs per byte.

8. **Why the dense repair pass is cheap per token.** A pass over f tokens at once is prefill-shaped: f queries share one read of the cache. Its KV traffic per token is (one full read)/f. This is the reason both ReSA and SparseSpec can afford a dense pass every few tokens.

9. **Kernel-level picture (FlashInfer).** The KV cache is stored in pages and described to the kernel as a block-sparse row matrix: "Non-zero blocks represent KV-Cache pages accessed by queries" [checked: 2501.01005 Sec 3.1.1, Fig 2]. The kernel gathers tiles "from scattered global memory to contiguous shared memory" and then runs dense tensor-core code [checked: Sec 3.2.1]. With GQA, "a single shared-memory load of the KV-Cache suffices for all query heads in the group" [checked: Appendix A]. For decode kernels the gap between paged and contiguous KV is "negligible (within 1%)"; for prefill kernels it is about 10% [checked: Appendix B]. So a sparse read is just a shorter page list. Bytes fall in proportion; time does not have to (fact 14).

10. **What an approximate step leaves behind.** Two things persist after step i: (a) the token x_{i+1}, sampled from p̃_i instead of p_i; (b) the KV entries of position i, written from an approximate residual stream. For sparse attention, (b) is wrong only in layers 2 … L, because layer 1 depends only on the token (Exercise 2b: layer-1 error exactly 0). For low-bit weights, every layer is affected, because W_K and W_V themselves changed. For a lossy KV store (quantized or low-rank), the stored entry is wrong by construction.

11. **Precise definitions of the two channels.** The **KV channel** error is: stored KV minus KV\*(the tokens actually in the text). The **token channel** error is: the tokens in the text were not drawn from p. There is a third term, the **current-step read error** (p̃_i ≠ p_i even on a clean cache), but it is not stored. It matters while the approximation runs, and it is what creates both stored errors. After a switch back to the exact method, the future is a function of (tokens, stored KV) only.

12. **Repair 1: re-check the token.** A dense pass gives the dense distribution at each position. A rule accepts or rejects each emitted token. On the first rejection the system **rolls back**: it drops every later token and continues from a replacement. The rule can be exact (speculative sampling, lossless) or a threshold. Sirius uses a threshold: the full model compares "the likelihood against a preset threshold", and "the threshold of 0.1 and the temperature of 0.6 works well" [checked: 2409.03856 Sec 4.1; the HTML page prints this text after the 4.1 and 4.2 headings, and by content it belongs to 4.1, "How to Correct the Sparse Output Tokens"].

13. **Repair 2: rewrite the KV (rectification).** ReSA: "we batch these recent tokens and re-encode them using dense attention to reconstruct an updated KV cache" [checked: 2506.04108 Sec 2.2]. Default period f = 32 [checked: Sec 3.1]. Tokens are kept: ReSA "does not rely on output logits to make per-token accept / reject decisions" [checked: Appendix B]. After the pass, stored KV = KV\*(tokens) on the rewritten span. The state is again a pure function of the text, but the text is not what the dense model would have written. This is the same kind of persistent state as relaxed speculative decoding: exact KV, off-distribution tokens. Relaxed verification lengthens completions on average: mean 1.54×, median 1.08×, longer in 248 of 360 cells. But the median is 0.94 for standalone drafters and 1.23 for EAGLE heads [from notes: 2610.23818, Tables 1 and 3]. So the effect is real on average and not uniform. Do not say that every separate drafter lengthens completions.

14. **Cost of the two systems.** Let "1" be one full read of the cache. ReSA: Avg(mem) = mem(KV cache) × (1/b + p + 1/f), where b is ReSA's block size (not the bytes per number of fact 5; the block keys are read every step), p is the share of the cache a sparse step reads, and f is the rectification period [checked: 2506.04108 Sec 2.3, Eq. 6]. With b = 16, share 0.1, f = 32: 0.194, a 5.16× reduction. The paper is not consistent about p: Eq. 6 uses p as the share that is read, Sec 3.1 gives the default p = 0.9, and Sec 3 speaks of the case "when the attention computation ratio is reduced to 0.1" [checked]. Read the default as: a sparse step reads 10% of the cache. SparseSpec: KV traffic per committed token = (k·s + 1)/(k·α + 1), where k is the number of draft steps, s the sparsity ratio, and k·α the mean number of accepted draft tokens [checked: 2512.01278 Sec 3.2]. With k = 8, s = 0.05, 6.16 accepted: 0.196, or 5.1× less traffic. The measured attention time fell 3.29×, from 17.1 ms to 5.2 ms [checked: 2512.01278 Table 2]. The 6.16 of 8 is an average over models and datasets [checked: Sec 5.3], and Table 2 is Qwen3-8B on AIME, so the two numbers do not come from the same run. Bytes are an upper bound on the gain, not a prediction.

15. **One dense pass gives the raw material for both repairs.** A dense pass over the recent tokens produces dense KV for them and dense logits at each of them. ReSA keeps the KV and does not use the logits. SparseSpec uses both. So in what they do with the dense pass, the two systems differ only in the check and the rollback. (They also differ in how the sparse set is chosen.) With the same s and the same period k = f, traffic(verify + rollback) / traffic(rewrite only) = k / (k·α + 1): 1.12 at 6.16 accepted of 8, and 1.78 at 8 accepted of 16 (Exercise 1g). ReSA compared the two at draft length 16 = period 16. It measured throughput, not traffic: "only about 8 tokens are typically accepted" and "ReSA achieves nearly 2× speedup over self-speculation" (average 1.92×) [checked: 2506.04108 Appendix B, Table 3].

16. **Why SparseSpec's exact verification repairs both at once.** Invariant after every round: (i) every committed token was drawn from the dense distribution given a clean cache; (ii) every stored KV entry at a committed position equals KV\* for the committed text. Proof by induction: the prefill is dense. In a round, the verify pass runs dense attention over the last committed token and the k draft tokens; it reads only committed entries (dense by induction) and the entries it computes itself in the same pass. So its distribution at each draft position is the true dense one given the drafts before it, and the exact rule makes the accepted prefix plus the replacement token exact. The pass writes its KV at the same cache positions the draft steps used, and the entries after the first rejection are retracted [checked, local: `SparseSpec/serve/scheduler/spec_scheduler.py`, linear VERIFY branch and `verify_outputs`; `serve/request/kv_cache_ptr/base.py`, `dispatch_append_kv_cache`; `3rdparty/flashinfer/flashinfer/page.py`, `get_batch_indices_positions`. The verify pass allocates one new slot and writes KV at the last k+1 slots of the sequence. These are the k slots the draft steps wrote plus the new one. So the overwrite is confirmed from the code]. The draft's sparse reads only ever produce proposals, so nothing approximate survives. The SparseSpec paper page does not state the overwrite in words [checked: 2512.01278, not found]; S3R does: "Verification replaces provisional draft states with recomputed dense KV entries" [from notes: 2610.13202].

17. **The 2×2 is partly degenerate.** "Repair tokens only, with a clean dense verifier" is lossless speculative decoding and recovers 100% by construction. Plain sparse decoding and ReSA at f > 1 are useful arms, but each mixes both faults. The two arms that isolate one fault each are: **arm B**, sparse logits on dense-written KV (ReSA with f = 1; KV clean, tokens from sparse reads), and **arm C**, dense reads over sparse-written KV (tokens from dense reads, KV left as the sparse pass wrote it). Arm C needs a dense read every step, so it is a diagnostic, not a system [from notes: verification.md F38].

18. **What ReSA shows (KV rewrite only, sparse attention).** On R1-Distill-Qwen math reasoning, 7B averages are Dense 60.72, Sparse 57.72, ReSA 60.52; 1.5B averages are 46.82, 43.50, 45.56 [checked: 2506.04108 Table 1]. So the KV rewrite alone recovers 93% and 62% of a 3-point gap (my arithmetic, Exercise 3). An open-loop arm called "Decode Only" has "all KV cache entries … generated using dense attention, and sparse attention is only used for decoding. This serves as the upper bound for ReSA", and at period 32 "the model's performance almost approaches the upper bound" [checked: Sec 3.3]. Table 1 has one "Avg Length" row per model, not one per method [checked], so it does not show whether sparse decoding changed the length. Not reported: output length per method, a token-only arm, or a control for the block-key refresh ("we also refresh the associated block keys during rectification" [checked: Sec 2.2]).

19. **What Sirius found (rollback versus KV rewrite, sparse MLP).** Sirius corrects *contextual sparsity*: "given the context, only a limited number of the most relevant neurons are selected based on the input activation. The rest contributed to the output far less is discarded"; "CS exists naturally in MLP layers" [checked: 2409.03856 Sec 2.1]. A picture (mine, not the paper's words): a ReLU neuron that outputs exactly 0 can be skipped with no error. Contextual sparsity also skips neurons whose output is small but not 0. So it is lossy. Two kinds: coarse (CSparse) fixes the set of neurons once per prompt; fine (FSparse) picks the set again for every token [checked: Sec 2.1]. The default is 50% sparsity of the MLP [checked: Sec 3]. The sparse and full model share one cache: "The KV cache is mostly populated by the sparse model"; once per period (the "kernel size") the full model "takes in the last kernel size of tokens and generates its KVs for the past kernel size tokens in parallel", and these are "directly written to their corresponding positions in the shared KV Cache" [checked: Sec 4.1]. Ablation on Llama-3-8B-Instruct, table header "GSM8K 20%", accuracy [checked: Table 4, read from the raw HTML]:

    | row | CSparse | FSparse |
    |---|---|---|
    | full model | 0.7538 | 0.7538 |
    | sparse | 0.3674 | 0.5644 |
    | + interleave | 0.3826 | 0.6288 |
    | + KV rewrite | 0.4735 | 0.6629 |
    | + KV rewrite + interleave | 0.4886 | 0.6780 |
    | + rollback + interleave | 0.6591 | 0.7273 |
    | + KV rewrite + interleave + rollback | 0.6667 | 0.7273 |

    "Interleave" means that the full model replaces only the one token it is evaluating: "We started by only letting the LLM correct the token it is evaluating (interleaving only). Then, we add on top of it the KV cache correction, and then the rollback" [checked: Sec 5.3]. There is no "rollback alone" row; rollback always comes with interleave. Each cell is printed as two numbers "A/B". They are equal except in two FSparse cells (0.6780/0.6818 and 0.7273/0.7311). The table does not say what the two numbers are [unverified]. The text: "Rollback seems to be the most effective technique"; "KV Cache is also effective alone, bringing a 12% increment for CSparse and an 11% accuracy increase for FSparse" (this matches the row + KV rewrite + interleave minus plain sparse: 0.121 and 0.114); "Relatively, interleaving is the weakest"; "adding both KV rewriting and rollback is only marginally better than rollback alone"; but for CSparse "adding KV rewrite increases AAL from 12.77 to 13.80", where AAL is the mean number of tokens between two corrections [checked: Sec 5.3, Sec 2.2]. Also useful for "which tokens need verifying": "With merely 10% of tokens needing to be corrected, the sparse model can completely match the full model's performance" [checked: Sec 3.4].

20. **How to read Sirius for our question.** Use the interleave rows as the base. Then the two repairs form a clean 2×2 (CSparse): neither 0.3826, KV rewrite 0.4886, rollback 0.6591, both 0.6667. In shares of the 0.3712 gap to the full model: KV rewrite 28.6%, rollback 74.5%, both 76.5%. The single shares sum to 103.0%, far above the joint 76.5% (FSparse: 118.2% against 78.8%). So the two repairs overlap: a rollback check also catches harm that entered through stale KV, as long as it shows up as an unlikely token. Sirius does not transfer directly: its sparsity is in the MLP and attention stays full, the task is short GSM8K answers scored by accuracy with no length measure, and its rollback is a threshold rule, not the exact rule. One more open point: the paper does not say whether, in the row with rollback and no KV rewrite, the full model's check reads KV that the sparse model wrote [unverified; likely, because the cache is shared]. Also, "rejected" is not "harmful": at one position the exact rule rejects with probability TV(p, q) = ½ Σ_x |p(x) − q(x)| even when the drafted token is fine. ReSA keeps accuracy in a setting where an exact verifier accepts only about 8 of 16 drafted tokens per round (fact 15). Careful: this does not mean 8 tokens are rejected. A round stops at the first rejection, and the drafts after it are thrown away without being judged. If the rejection rate is the same at every position, 8 accepted of 16 means about one rejection in 11 tokens (per-token acceptance about 0.91; my arithmetic from the formula in Q17).

---

## 2. Level ladder

- **Level 2.** Can say what the cache stores and that later steps read K and V of every past position. Knows that sparse decoding and rectification exist, and can follow the two-channel picture when it is drawn.
- **Level 3.** Can compute cache size and KV traffic from a model config, and the traffic ratios of ReSA and SparseSpec from given numbers. With a hint, can say which layers' KV a sparse step corrupts and what each repair leaves untouched.
- **Level 4.** Derives the memory-bound argument and the SparseSpec invariant without help. Says for any system (ReSA, SparseSpec, Sirius, relaxed verification, low-bit weights, lossy KV) which of token error, stored-KV error and current-step read error are present after each step. Sees why "token-only repair" is degenerate and names arms B and C.
- **Level 5.** Designs the Part B and Part C arms with their controls (noise replicate, oracle selection, block-key refresh, dose measured on clean KV). Explains why single-repair shares need not add up, why Sirius does not transfer directly, and finds the hidden KV dose in a Part A comparison.

---

## 3. Misconceptions

1. **"Sparse attention only changes what a step reads. The cache it leaves is the same as dense."** Wrong. The step also writes KV for its own position, and in layers 2 … L those entries are computed from a residual stream that contains the sparse attention output. Only layer 1 is unchanged.

2. **"After a dense KV rewrite the system is back on the dense model's path."** Wrong. The rewrite makes the cache correct *for the tokens that are there*. The tokens stay, and some of them are tokens the dense model would rarely write. The token channel is untouched.

3. **"A token that passes exact verification can still sit on bad KV."** Not in SparseSpec. The verify pass that produces the logits also writes the KV at those positions. The invariant in fact 16 covers both. It would be true only in an engine that skipped the overwrite.

4. **"The tokens a verifier would reject are the harmful tokens, so the rejection count measures the damage."** Wrong. With sampling, the exact rule rejects with probability TV(p, q), which measures how different two distributions are, not whether the token hurts. ReSA keeps accuracy in a setting where an exact verifier accepts only about 8 of each 16 drafted tokens [checked: 2506.04108 Appendix B]. (That is not 8 rejections: the round stops at the first rejection, and the rest are thrown away unjudged. At a constant rate it is about one rejection in 11 tokens; my arithmetic.) Vegas reports that many rejected tokens come from a draft distribution close to the target [from notes: 2602.07223].

5. **"Sirius showed rollback beats KV rewrite, so the token channel carries the harm in sparse attention too."** Does not follow. Sirius sparsifies the MLP, not attention; it measures GSM8K accuracy, not length; and its rollback check sees any harm that shows up as an unlikely token, including harm from stale KV. For sparse attention, ReSA's KV rewrite alone recovers 62-93% of the accuracy gap in its setting.

6. **"Token-only repair is just the third arm of a 2×2; run the verifier and skip the overwrite."** Wrong. If the verifier reads a clean cache, the arm is lossless speculative decoding and tells you nothing. If the overwrite is skipped, the next verification reads sparse-written KV and is no longer the dense model. The arm must be defined as dense reads over sparse-written KV (arm C).

7. **"Reading 5% of the cache makes attention 20 times faster."** Wrong twice. The periodic dense pass adds 1/f (or 1 per round) to the traffic, so 5% becomes about 20%. And time does not follow bytes: SparseSpec's traffic formula gives 5.1× where the measured attention time fell 3.29× [checked: 2512.01278 Table 2; 5.1× is arithmetic].

8. **"Exact verification costs an extra dense pass compared with rectification."** Wrong. Both run the same dense pass at the same period. Verification costs the *draft tokens thrown away after a rejection*. The traffic ratio is k/(k·α + 1): 1.12 (12% more) at SparseSpec's acceptance, 1.78 (78% more) at 8 accepted of 16.

---

## 4. Question bank

Each entry: level, type, question, answer key, likely wrong answer, follow-up.

### kv_channels-Q1
- **Level:** 2. **Type:** explain why.
- **Question:** "At one decode step a layer computes a query, a key, a value, attention weights, an attention output and an MLP output. Which of these go into the KV cache, and why only those?"
- **Answer key:** The key and the value of the current position, per layer and per KV head. A later position j needs from position i only k_i (to compute its weight on i) and v_i (to take its content). The query, weights and outputs of position i are used only at position i.
- **Likely wrong answer:** "The hidden state of each layer is cached."
- **Follow-up:** "Write the attention output of a later position j. Which quantities from position i appear in it?"

### kv_channels-Q2
- **Level:** 2. **Type:** predict (numeric).
- **Question:** "Qwen3-8B has 36 layers, 32 query heads, 8 KV heads and head dimension 128, stored in bf16. A batch holds 32 requests with 12,000 cached tokens each. How many gigabytes of KV does one decode step read with full attention, and how long does that take at 3 TB/s?"
- **Answer key:** Per token: 2 (K and V) × 36 × 8 × 128 × 2 bytes = 147,456 bytes. Batch: 147,456 × 32 × 12,000 = 56.6 GB. Time: 56.6 / 3000 s = 18.9 ms.
- **Likely wrong answer:** Uses 32 heads instead of 8: 589,824 bytes per token, 226 GB, 75.5 ms. Or forgets the factor 2 for K and V (28.3 GB).
- **Follow-up:** "With grouped-query attention, how many different key vectors does one layer hold for one token: 32 or 8?"

### kv_channels-Q3
- **Level:** 3. **Type:** explain why (derivation).
- **Question:** "For one decode step of Qwen3-8B in bf16, derive how many FLOPs attention does per byte of KV it reads, and how many FLOPs a weight matrix does per byte when the batch holds B requests. What does the comparison say about each one as B grows?"
- **Answer key:** Attention, per cached token and layer: 4·d_h·H_q FLOPs over 2·H_kv·d_h·2 bytes = H_q/H_kv = 4 FLOPs per byte, for any batch size and any context length. A weight matrix with P parameters: 2·P·B FLOPs over 2·P bytes = B FLOPs per byte. A GPU can do a few hundred FLOPs per byte read. So attention is always limited by memory reads; GEMM moves toward compute-bound as B grows. Also: weight bytes per step are fixed, KV bytes grow with B and with length.
- **Likely wrong answer:** "Attention is memory-bound because the KV cache is big" (no ratio), or "because softmax is slow".
- **Follow-up:** "If you double the batch size, what happens to the bytes read per step for the weights, and for the KV cache?"

### kv_channels-Q4
- **Level:** 3. **Type:** predict.
- **Question:** "A model runs one decode step with sparse attention: in every layer it reads only the top 5% of the cached entries. Compare the KV entry this step writes for its own position with the entry a dense step would write for the same input token, starting from the same dense-written cache. In which layers do the two differ?"
- **Answer key:** Layer 1 is identical: its key and value are computed from the token embedding only. Layers 2 … L differ: their input is the residual stream, which contains the sparse attention outputs of the layers below. (In the toy of Exercise 2 the error grows with depth. That is a toy result, not a law; do not require it.)
- **Likely wrong answer:** "They are identical. Sparse attention changes reads, not writes."
- **Follow-up:** "Write the layer-2 key of position i as a function of its input. Where did that input get its attention part from?"

### kv_channels-Q5
- **Level:** 3. **Type:** predict.
- **Question:** "ReSA's language-modelling test has an arm called 'Decode Only': the text is fixed, every KV entry is written with dense attention, and only the read at the scored step is sparse. Which of the three errors (wrong tokens, wrong stored KV, current-step read error) does this arm contain, and why does the paper call it the upper bound for ReSA?"
- **Answer key:** Only the current-step read error. The text is fixed, so no token error; the KV is dense-written, so no stored-KV error. Rectification can only clean stored KV; it cannot remove the sparse read of the current step. So ReSA at any period f can at best match this arm. [checked: 2506.04108 Sec 3.3]
- **Likely wrong answer:** "It is the dense baseline", or "it contains the KV error".
- **Follow-up:** "If the rectification period were 1, which of the three errors would ReSA still have on fixed text?"

### kv_channels-Q6
- **Level:** 3. **Type:** predict (numeric).
- **Question:** "ReSA gives the average memory access per token as the full cache times (1/b + p + 1/f): b is the block size, p the share of the cache a sparse step reads, f the rectification period. With b = 16, p = 0.1 and f = 32, what is the reduction factor compared with dense decoding, and at which f would rectification cost as much as the sparse reads?"
- **Answer key:** 1/16 + 0.1 + 1/32 = 0.0625 + 0.1 + 0.03125 = 0.19375, a 5.16× reduction. Rectification equals the sparse reads when 1/f = 0.1, so f = 10.
- **Likely wrong answer:** "10×, because it reads 10% of the cache." (Drops the block-key and rectification terms.)
- **Follow-up:** "During one rectification of 32 tokens, how many times is the whole cache read, and how many tokens share that read?"

### kv_channels-Q7
- **Level:** 4. **Type:** explain why. **Ties to Part C.**
- **Question:** "SparseSpec drafts k tokens with sparse attention and then runs one full-attention verification. State an invariant about the committed tokens and the stored KV that holds after every round, and prove it by induction in a few sentences."
- **Answer key:** Invariant: committed tokens follow the dense model's distribution, and every stored KV entry at a committed position is the dense KV for the committed text. Base: dense prefill. Step: the verify pass is a dense forward over the last committed token plus the k drafts. It reads only committed entries (clean by induction) and its own in-pass entries, so its distributions are the true dense ones given the earlier drafts. The exact rule makes the accepted prefix and the replacement token exact. The pass overwrites the draft-written KV at the same positions, and entries after the first rejection are retracted. The replacement token has no KV yet. Draft-step reads only produce proposals.
- **Likely wrong answer:** "Verification checks the tokens, and if the tokens are right the KV is right."
- **Follow-up:** "Suppose all k drafts were accepted but the engine kept the draft-written KV. Are the tokens committed so far still exact? Is the next round's verification still exact?" (Yes; no, it now reads sparse-written entries.)

### kv_channels-Q8
- **Level:** 4. **Type:** find the flaw. **Ties to Part C.**
- **Question:** "A colleague plans a 2×2 for sparse decoding: repair neither, repair KV only, repair tokens only, repair both. For 'tokens only' they write: run the dense verifier, replace rejected tokens, but keep the draft-written KV. What is wrong with this arm, and which two arms, each with exactly one fault, should replace it?"
- **Answer key:** Two cases. If the verifier reads a clean dense cache, the arm is lossless speculative decoding: it reproduces the dense output whatever the draft KV was, so it recovers 100% by construction. If the draft-written KV is really kept, the next verification reads sparse-written entries, so the "verifier" is no longer the dense model and the arm is not a token repair. The two single-fault arms are: B, sparse logits on dense-written KV (ReSA at f = 1; tokens come from sparse reads, KV is clean); and C, dense reads over sparse-written KV, with the KV still written by the sparse pass. C is diagnostic only, because it needs a dense read every step. (Plain sparse and KV-rewrite-only stay in the design. They are informative, but each mixes both faults.)
- **Likely wrong answer:** "Nothing is wrong. Skipping the overwrite isolates the token repair."
- **Follow-up:** "In round 2 of that arm, the verifier attends to the positions accepted in round 1. Who wrote the K and V it reads there?"

### kv_channels-Q9
- **Level:** 4. **Type:** predict (numeric).
- **Question:** "Two systems use the same sparse step (reads 5% of the cache) and one dense pass every 16 tokens. System R rewrites the KV and keeps all 16 tokens. System V verifies exactly and on average accepts 8 of the 16 drafts, plus one replacement token. In units of one full cache read, what is the KV traffic per committed token in each system, and what is the ratio?"
- **Answer key:** R: (16 × 0.05 + 1)/16 = 1.8/16 = 0.1125. V: (16 × 0.05 + 1)/(8 + 1) = 1.8/9 = 0.2. Ratio 0.2/0.1125 = 1.78 = k/(accepted + 1) = 16/9. ReSA measured an average 1.92× in throughput for this comparison [checked: 2506.04108 Appendix B, Table 3]. Throughput is not traffic, so the two numbers need not agree. With 6.16 accepted of 8 the same ratio is 8/7.16 = 1.12.
- **Likely wrong answer:** "2×, because verification needs an extra dense pass." (Both run one dense pass per 16 tokens. The cost is the discarded drafts.) A second way to reach 2: 16/8, which forgets the replacement token.
- **Follow-up:** "Count the dense passes per 16 drafted tokens in each system. Then count the tokens each system commits from those 16."

### kv_channels-Q10
- **Level:** 4. **Type:** predict (numeric). **Ties to Part C.**
- **Question:** "Sirius reports GSM8K accuracy for Llama-3-8B-Instruct with a sparse MLP. In all four sparse arms the full model also replaces the one token it checks, once per period. Full model 0.7538; sparse with no repair 0.3826; with KV rewrite 0.4886; with rollback 0.6591; with both 0.6667. If the two repairs fixed two independent, additive channels, what accuracy would 'both' reach, and what does the measured value tell you?"
- **Answer key:** Gains: KV rewrite +0.1060, rollback +0.2765. Additive prediction: 0.3826 + 0.1060 + 0.2765 = 0.7651, above the full model (0.7538). Measured: 0.6667. As shares of the 0.3712 gap: 28.6%, 74.5%, sum 103.0%, joint 76.5%. The repairs overlap: rollback already catches most of what the KV rewrite fixes. The numbers cannot be read as "29% of the harm is in the KV, 75% in the tokens". [checked: 2409.03856 Table 4, CSparse column, rows + interleave, + KV rewrite + interleave, + rollback + interleave, all three]
- **Likely wrong answer:** "About a quarter of the harm travels through the KV and three quarters through the tokens."
- **Follow-up:** "Add your two shares. Then compare the sum with the share that both repairs recover together."

### kv_channels-Q11
- **Level:** 4. **Type:** find the flaw. **Ties to Part C.**
- **Question:** "Sirius found that rollback recovers much more accuracy than KV rewrite. A colleague concludes: 'so in sparse-attention decoding of long reasoning traces, the length tax travels through the tokens, not the cache.' Give three reasons why this does not follow."
- **Answer key:** Any three of: (1) Sirius sparsifies MLP neurons and keeps attention full, so there is no read-side selection error and the KV error has a different origin. (2) The task is short GSM8K answers with Llama-3-8B-Instruct, scored by accuracy; no length or cap-hit measure, no 10K-token trace. (3) Sirius's rollback is a likelihood-threshold check by the full model; it catches any harm that shows up as an unlikely token, including harm caused by stale KV, so its effect is not "token channel only". (4) For sparse attention, ReSA's KV rewrite alone recovers 62-93% of the accuracy gap in its setting. (5) The shares do not add up (Q10), so they are not a split.
- **Likely wrong answer:** "It follows: both are sparsity with a shared KV cache, and the ablation is clean."
- **Follow-up:** "When Sirius's full model checks a token, can it tell whether the token is bad because of the sparse MLP at that step or because of a stale KV entry read at that step?"

### kv_channels-Q12
- **Level:** 4. **Type:** apply to a new case. **Ties to Part B.**
- **Question:** "In the Part B experiment you run sparse decoding up to token T and then continue with the dense model. Through what can the sparse phase still influence the dense phase, and what one extra arm tells you which of them carries the lasting damage?"
- **Answer key:** Only through two things: the text x_1 … x_T and the stored KV of those positions (layers 2 … L were written from sparse reads). Extra arm: at the switch, re-encode the prefix with a dense pass (one prefill over T tokens), then continue dense. Now only the text remains. If this arm's remaining trace has normal length and the no-rewrite arm's is long, the lasting damage is in the KV and a periodic rewrite is the fix. If both are long, the text carries it and only rollback or verification helps. Both arms need a noise control (shared random numbers, plus a dense-versus-dense seed replicate).
- **Likely wrong answer:** "After the switch the model is exact, so whatever inflation remains is caused by the tokens."
- **Follow-up:** "After the switch, a dense step attends to position T − 5. Which K and V does it read there, and which pass wrote them?"

### kv_channels-Q13
- **Level:** 5. **Type:** find the flaw (adversarial). **Ties to Part A.**
- **Question:** "In Part A you tune sparse attention, random token swaps and a higher temperature to the same dose, where dose counts changed tokens and how unlikely they are. Sparse attention shows the largest length tax. A colleague concludes that the harm comes from context the model could not see at the moment of choosing a token. Which other explanation does the two-channel picture give, and which control arm removes it?"
- **Answer key:** Dose counts tokens only. The swap and temperature arms have exact KV. The sparse arm also leaves perturbed KV at every position in layers 2 … L, including the positions where the emitted token did not change. So the sparse arm carries an unmeasured dose in the KV channel. Control: sparse reads with a dense KV rewrite every step (arm B, ReSA at f = 1). Then all three arms have state = function of the text. If sparse is still worse at equal dose, the cause is which tokens were changed, and the "could not see" reading stands. The dense rewrite also gives the dense logits needed to measure dose on a clean cache.
- **Likely wrong answer:** "There is no other explanation. At equal dose the three token streams are statistically the same."
- **Follow-up:** "Take a position where sparse and dense emit the same token. Is the KV entry written at that position the same in the sparse arm and in the temperature arm?"

### kv_channels-Q14
- **Level:** 5. **Type:** apply to a new case (edge case).
- **Question:** "Apply the two-channel picture to a lossy KV store, for example quantized or low-rank KV as in Palu, with exact weights and full attention. Which errors exist, and what would 'rectification' do there?"
- **Answer key:** Must contain: (1) The error starts in the store, not in the read selection and not in the weights. (2) Each stored entry has two error parts: its own compression error, and an inherited part because its residual stream was computed from reads of compressed entries. (3) Tokens are sampled from logits computed over the lossy store, so a token channel exists as a consequence. (4) Rectification as defined (recompute with the exact method and overwrite) cannot end in a clean cache: the exact entry cannot be stored without giving up the saving. A batched dense re-encode can remove a part of the inherited error, because inside the pass the recent tokens read each other's exact K and V. But each entry is compressed again when it is stored, so its own compression error always stays. (5) So lossy KV is a pure stored-state approximation and a natural positive control for the KV channel [from notes: verification.md F38].
- **Likely wrong answer:** "There is no token channel, because the weights and the attention are exact."
- **Follow-up:** "The logits at step j are computed from which inputs? Are all of those inputs exact?"

### kv_channels-Q15
- **Level:** 5. **Type:** explain why (edge case).
- **Question:** "In a sparse self-draft with exact verification, the first drafted token of a round is accepted about 95% of the time. The sixth is accepted clearly less often, even when you count only the rounds in which drafts 1 to 5 were all accepted. Name two mechanisms from this topic that predict the decay, and one intervention for each that would isolate it."
- **Answer key:** (1) Stored-KV mechanism: draft step j reads the entries of draft positions 1 … j−1 of the same round, which were written by sparse steps. Intervention: after each draft step, rewrite that position's KV densely, keep the selection pattern fixed; if the decay goes away, it was the KV. (2) Read mechanism: the selection pattern is fixed for the whole stride (in SparseSpec it comes from the last verification's attention scores [checked: 2512.01278 Sec 4.1 via specdec_exact.md and this run]), so it goes stale. Intervention: refresh the pattern every step (for example oracle top-K from dense scores) and keep the draft-written KV. It is not "token errors compounding": by the condition in the question, drafts 1 … j−1 passed the exact check, so the text before position j is a valid sample of the dense model. Note for the tutor: S3R reports 95.8% for the first proposed token and 67.9% for the sixth (Qwen3-8B RL policy, s = 0.05, T = 1.0) [from notes: 2610.13202 Appendix E.2]. The notes do not say whether these numbers are conditional on the earlier drafts being accepted [unverified]. Even if they are not, a constant rate does not explain them: 0.958⁶ = 0.773, which is above 0.679.
- **Likely wrong answer:** "Errors in the earlier draft tokens compound."
- **Follow-up:** "In the rounds you count, drafts 1 to 5 passed the exact check. In the greedy case, how do those five tokens differ from the dense model's own choice? Then what else did those five draft steps leave behind that draft step 6 reads or uses?"

### kv_channels-Q16
- **Level:** 4. **Type:** connect two ideas.
- **Question:** "Take ReSA right after a rectification, and relaxed speculative decoding with a separate small drafter. In terms of what the dense model's later steps will read, what do the two have in common?"
- **Answer key:** In both, the stored KV is dense-written for the text that is there, and the text contains tokens that were not drawn from the dense distribution. So both carry a pure token-channel error in their persistent state. Relaxed verification with exact KV lengthens completions on average (mean 1.54×, median 1.08×, 248 of 360 cells) [from notes: 2610.23818]. So the token channel alone can cause a length tax. The effect is not uniform: the median is 0.94 for standalone drafters and 1.23 for EAGLE heads [from notes]. The difference while running: ReSA's next sparse steps have a read error and write up to f new sparse entries before the next rewrite.
- **Likely wrong answer:** "ReSA has no token error, because the KV was corrected."
- **Follow-up:** "After the rectification, were the tokens in the rectified span sampled from the dense distribution?"

### kv_channels-Q17
- **Level:** 4. **Type:** predict (numeric).
- **Question:** "A sparse self-draft reads 5% of the cache per draft step, drafts k = 8 tokens per round and has a per-token acceptance of 0.95, independent across positions. How many tokens does one round commit on average, and what is the KV traffic per committed token in units of one full read?"
- **Answer key:** Expected accepted drafts = α(1 − α^k)/(1 − α) = 0.95 × (1 − 0.95⁸)/0.05 = 0.95 × 0.33658/0.05 = 6.395. Committed per round = 7.395 (with the replacement or bonus token). Traffic per round = 8 × 0.05 + 1 = 1.4 full reads. Per committed token: 1.4/7.395 = 0.189, about 5.3× less than dense.
- **Likely wrong answer:** "0.05, since each step reads 5%", or uses 8 × 0.95 = 7.6 accepted (forgets that a rejection ends the round).
- **Follow-up:** "If the second draft is rejected, how many of the remaining six drafts can still be committed?"

### kv_channels-Q18
- **Level:** 5. **Type:** find the flaw (adversarial).
- **Question:** "ReSA's rectification rewrites the KV of the recent tokens and also refreshes the block keys that its selector uses. Someone reads ReSA's accuracy recovery as proof that the stored-KV channel carried the loss. What is the confound, and what control separates it?"
- **Answer key:** The block keys (per-block min and max of the keys) decide which blocks later sparse steps read. Refreshing them from dense keys improves the selection, which is a read-side effect, at the same moment the stored K and V are cleaned. So the gain could come from better selection, not from cleaner values. Controls: rewrite K and V but keep the old block keys (or the reverse); or use oracle selection in both arms so that selection quality is equal [from notes: verification.md F38]. A level-5 answer also notes that the paper reports no length per method, so the closed-loop length question is open either way.
- **Likely wrong answer:** "No confound. The block keys are part of the KV cache, so refreshing them is the KV repair."
- **Follow-up:** "Which later computation uses the block keys: the attention sum over values, or the choice of which blocks to load?"

### kv_channels-Q19
- **Level:** 4. **Type:** apply to a new case.
- **Question:** "A model runs with 4-bit weights and stores its KV in bf16. For the same input tokens, in which layers do its KV entries differ from the bf16 model's, and how does that differ from the sparse-attention case?"
- **Answer key:** In every layer, including layer 1, because W_K and W_V themselves are quantized (and the residual stream below is computed with quantized weights). Under sparse attention, layer 1 is exact and only layers 2 … L differ. A rectification here means the bf16 model re-encodes recent tokens and overwrites the entries, so both models must be loaded. Published neighbours: QSpec overwrites low-precision KV with higher-precision KV for accepted tokens [from notes: 2410.11305]; one agent study found 68.2% versus 61.6% task success for recomputed versus reused low-tier KV [from notes: 2602.02711, Appendix F].
- **Likely wrong answer:** "Same as sparse attention: layer 1 is exact because it only depends on the token."
- **Follow-up:** "Which matrix turns the layer-1 input into the layer-1 key, and is that matrix the same in the 4-bit model?"

### kv_channels-Q20
- **Level:** 3. **Type:** connect two ideas.
- **Question:** "A dense pass over the last 32 tokens costs far less KV traffic than 32 dense decode steps. Why, and why does this make both ReSA and SparseSpec practical?"
- **Answer key:** In one pass the 32 queries share a single read of the cache, as in prefill. Traffic is one full read instead of 32, so 1/32 of a full read per token. The FLOPs per byte rise by the same factor of 32, so the pass moves toward compute-bound. Both systems run sparse steps (cheap reads) and pay one shared dense read per period; that pass yields dense KV (used by both) and dense logits (used only by SparseSpec).
- **Likely wrong answer:** "Because the dense pass only reads the last 32 tokens."
- **Follow-up:** "In the dense pass, how much of the old cache does the query of the first of the 32 tokens attend to?"

---

## 5. Exercises

All code was run in this session (CPU, numpy only) in the scratch folder. The outputs below are pasted from those runs.

### Exercise 1. Cache size, traffic, and the price of rollback (pen and paper, 15-20 minutes)

**Task.** For Qwen3-8B (36 layers, 32 query heads, 8 KV heads, head dimension 128, bf16), compute by hand: (a) KV bytes per token; (b) KV for 8192 tokens; (c) attention FLOPs per byte, and compare with a GPU that does 1000 TFLOP/s and reads 3 TB/s; (d) KV traffic and read time per step for 128 requests with mean cached length 4096; (e) the share of KV in total bytes read if the weights are about 16.4 GB; (f) the traffic reduction of ReSA (b = 16, share 0.1, f = 32) and of SparseSpec (s = 0.05, k = 8, 6.16 accepted); (g) the traffic ratio verify-and-rollback over rewrite-only at equal s and period.

**Solution code.**

```python
# Exercise 1: KV-cache size, memory traffic per decode step, and why decode attention is memory-bound.
# Model shape: Qwen3-8B (36 layers, 32 query heads, 8 KV heads, head_dim 128, bf16 = 2 bytes).
import numpy as np

L, Hq, Hkv, d, nbytes = 36, 32, 8, 128, 2

# (a) bytes of KV per cached token
kv_per_tok = 2 * L * Hkv * d * nbytes            # K and V
print(f"(a) KV bytes per token       = {kv_per_tok} B = {kv_per_tok/1024:.0f} KiB")

# (b) one request with n = 8192 cached tokens
n = 8192
print(f"(b) KV for n={n}            = {kv_per_tok*n/1e9:.3f} GB")

# (c) operational intensity of decode attention (one query row per head)
flops_per_tok_layer = 2 * (2 * d) * Hq           # QK^T and AV, 2 FLOPs per multiply-add
bytes_per_tok_layer = 2 * Hkv * d * nbytes       # K and V of one cached token, one layer
I_attn = flops_per_tok_layer / bytes_per_tok_layer
peak_flops, bandwidth = 1.0e15, 3.0e12           # assumed GPU: 1000 TFLOP/s, 3 TB/s
balance = peak_flops / bandwidth
print(f"(c) attention intensity      = {I_attn:.1f} FLOP/byte (= Hq/Hkv = {Hq//Hkv})")
print(f"    machine balance          = {balance:.0f} FLOP/byte -> attention is {balance/I_attn:.0f}x below it")

# (d) batch of B requests, mean cached length n_mean: time to read all KV once per step
B, n_mean = 128, 4096
traffic = B * n_mean * kv_per_tok
print(f"(d) KV traffic per step      = {traffic/1e9:.1f} GB -> {traffic/bandwidth*1e3:.1f} ms at 3 TB/s")

# (e) weights are read once per step for the whole batch; KV is read once per request
weight_bytes = 8.2e9 * nbytes
print(f"(e) weight bytes per step    = {weight_bytes/1e9:.1f} GB; KV share of traffic = {traffic/(traffic+weight_bytes):.2f}")
print(f"    KV traffic = weight traffic when B*n = {weight_bytes/kv_per_tok:,.0f} cached tokens in the batch")
print(f"    MLP/GEMM intensity = B FLOP/byte = {B} at B={B} (weights shared by the batch)")

# (f) sparse decoding with periodic dense pass: KV traffic per committed token, in units of one full read
def resa(s, f, b=None):          # s = share of KV read by a sparse step, f = rectification period, b = block size
    return (0 if b is None else 1/b) + s + 1/f
def sparsespec(s, k, acc):       # k = draft steps per round, acc = mean accepted draft tokens per round
    return (k*s + 1) / (acc + 1)
print(f"(f) ReSA  b=16 s=0.10 f=32   : {resa(0.10,32,16):.4f}  -> {1/resa(0.10,32,16):.2f}x less traffic")
print(f"    SparseSpec s=0.05 k=8 acc=6.16: {sparsespec(0.05,8,6.16):.4f} -> {1/sparsespec(0.05,8,6.16):.2f}x less traffic")
print(f"    measured attention time (SparseSpec Table 2): 17.1 ms -> 5.2 ms = {17.1/5.2:.2f}x")

# (g) price of rollback: same s, same period k=f, no block keys
for k, acc in [(8, 6.16), (16, 8.0), (16, 15.0)]:
    r = sparsespec(0.05, k, acc) / resa(0.05, k)
    print(f"(g) k=f={k:2d}, accepted {acc:5.2f}: verify+rollback / rewrite-only traffic = {r:.3f}  (= k/(acc+1) = {k/(acc+1):.3f})")
```

**Real output.**

```
(a) KV bytes per token       = 147456 B = 144 KiB
(b) KV for n=8192            = 1.208 GB
(c) attention intensity      = 4.0 FLOP/byte (= Hq/Hkv = 4)
    machine balance          = 333 FLOP/byte -> attention is 83x below it
(d) KV traffic per step      = 77.3 GB -> 25.8 ms at 3 TB/s
(e) weight bytes per step    = 16.4 GB; KV share of traffic = 0.82
    KV traffic = weight traffic when B*n = 111,220 cached tokens in the batch
    MLP/GEMM intensity = B FLOP/byte = 128 at B=128 (weights shared by the batch)
(f) ReSA  b=16 s=0.10 f=32   : 0.1938  -> 5.16x less traffic
    SparseSpec s=0.05 k=8 acc=6.16: 0.1955 -> 5.11x less traffic
    measured attention time (SparseSpec Table 2): 17.1 ms -> 5.2 ms = 3.29x
(g) k=f= 8, accepted  6.16: verify+rollback / rewrite-only traffic = 1.117  (= k/(acc+1) = 1.117)
(g) k=f=16, accepted  8.00: verify+rollback / rewrite-only traffic = 1.778  (= k/(acc+1) = 1.778)
(g) k=f=16, accepted 15.00: verify+rollback / rewrite-only traffic = 1.000  (= k/(acc+1) = 1.000)
```

**How the tutor checks it.** (a) must be 147,456 bytes; an answer of 589,824 means the learner used query heads. (c) must come out as H_q/H_kv, with no dependence on length or batch. (d) is the same order as SparseSpec's reported 21 ms; it is only an order-of-magnitude check, because the GPU numbers are assumed, and 77 GB of KV plus the weights would not fit on one 80 GB card. The 8.2 billion count is right (8.19 billion from the config; examiner's arithmetic). But a decode step only looks up a few rows of the input embedding table (0.62 billion numbers), so about 15.1 GB of weights is nearer than 16.4 GB, and the KV share in (e) moves from 0.82 to 0.84. Accept both. Ask what (e) shows: at B = 128 the crude GEMM ratio (128) is still below the assumed balance (333), so the safe statement is about scaling, not about a threshold. For (g) the learner should state the closed form k/(accepted + 1) and say in words that the price of exactness is the discarded drafts.

### Exercise 2. What a sparse step leaves behind, in a toy transformer (numpy, about 30 minutes)

**Task.** The tutor hands over `make_model`, `step` and `dense_reference` from the code below (a 4-layer toy transformer with a KV cache and an optional per-head top-k read). The learner writes the experiment: (a) check that cached decoding equals a recompute without a cache; (b) prefill 96 tokens densely, run 32 sparse steps (each reads 8 entries per head), then compare the stored keys of those 32 positions, layer by layer, with the dense keys for the same tokens; (c) run the dense pass over the same tokens, count how many of the 32 emitted tokens a greedy dense check would flag, overwrite the 32 positions and compare the cache with the dense cache; (d) at the next step, with the same input token, compute the KL divergence from the clean dense distribution for: sparse read on clean KV, dense read on sparse-written KV, both, and dense read after the rewrite.

**Before running, ask the learner to predict:** the layer-1 number in (b), the last number in (d), and which of "read only" and "KV only" is larger.

**Solution code.**

```python
# Exercise 2: the two things an approximate step leaves behind, in a toy transformer (numpy only).
# Toy: 4 layers, 4 heads, pre-norm, ReLU MLP, absolute positions, random weights. Sparse attention = per-head top-k.
import numpy as np

V, D, L, H, DH, T = 64, 64, 4, 4, 16, 200      # vocab, width, layers, heads, head dim, max length
N0, M, TOPK = 96, 32, 8                        # dense prefix, approximate segment, entries a sparse step reads

def make_model(seed):
    r = np.random.default_rng(seed)
    g = lambda *s, sc=1.0: r.normal(0, sc, s)
    return dict(E=g(V, D), P=g(T, D),
                Wq=g(L, D, D, sc=3.0 / np.sqrt(D)), Wk=g(L, D, D, sc=1.0 / np.sqrt(D)),
                Wv=g(L, D, D, sc=1.0 / np.sqrt(D)), Wo=g(L, D, D, sc=1.0 / np.sqrt(D)),
                W1=g(L, D, 2 * D, sc=1.0 / np.sqrt(D)), W2=g(L, 2 * D, D, sc=1.0 / np.sqrt(2 * D)),
                U=g(D, V, sc=1.0 / np.sqrt(D)))

def norm(x): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def kl(p, q): return float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())

def step(m, tok, i, K, Vc, read="dense", write=True):
    """One decode step at position i. Reads K/Vc[:, :i], writes (or only uses) the entry at i. Returns probs."""
    h = m["E"][tok] + m["P"][i]
    for l in range(L):
        x = norm(h)
        k = (x @ m["Wk"][l]).reshape(H, DH); v = (x @ m["Wv"][l]).reshape(H, DH)
        q = (x @ m["Wq"][l]).reshape(H, DH)
        if write: K[l, i], Vc[l, i] = k, v
        Kall = np.concatenate([K[l, :i], k[None]], 0); Vall = np.concatenate([Vc[l, :i], v[None]], 0)
        s = np.einsum("hd,nhd->hn", q, Kall) / np.sqrt(DH)             # (H, i+1)
        if read == "sparse" and i + 1 > TOPK:                           # keep top-k scores per head
            thr = np.sort(s, -1)[:, -TOPK][:, None]
            s = np.where(s >= thr, s, -np.inf)
        a = softmax(s)
        h = h + np.einsum("hn,nhd->hd", a, Vall).reshape(D) @ m["Wo"][l]
        h = h + np.maximum(norm(h) @ m["W1"][l], 0.0) @ m["W2"][l]      # ReLU MLP
    return softmax(norm(h) @ m["U"])

def dense_reference(m, toks):
    """Dense KV and dense next-token probs for a fixed token list (what a full recompute gives)."""
    K = np.zeros((L, T, H, DH)); Vc = np.zeros((L, T, H, DH)); P = []
    for i, t in enumerate(toks): P.append(step(m, t, i, K, Vc))
    return K, Vc, np.array(P)

def run(seed, verbose=False):
    m = make_model(seed); r = np.random.default_rng(1000 + seed)
    toks = list(r.integers(0, V, N0))
    K = np.zeros((L, T, H, DH)); Vc = np.zeros((L, T, H, DH))
    for i in range(N0 - 1): step(m, toks[i], i, K, Vc)                  # dense prefill
    # approximate segment: M sparse steps, each writes KV at position i and emits token i+1 (greedy)
    for i in range(N0 - 1, N0 - 1 + M):
        p = step(m, toks[i], i, K, Vc, read="sparse"); toks.append(int(p.argmax()))
    seg = slice(N0 - 1, N0 - 1 + M)                                     # positions written by sparse steps
    Kd, Vd, Pd = dense_reference(m, toks[:-1])                          # dense pass over the SAME tokens
    # (b) error of the stored KV, per layer
    rel = [np.linalg.norm(K[l, seg] - Kd[l, seg]) / np.linalg.norm(Kd[l, seg]) for l in range(L)]
    # (c) the same dense pass also gives dense logits: which emitted tokens would a greedy check flag?
    flagged = sum(int(Pd[i].argmax()) != toks[i + 1] for i in range(seg.start, seg.stop))
    Kr, Vr = K.copy(), Vc.copy()                                        # rectification = overwrite with dense KV
    for i in range(seg.start, seg.stop): step(m, toks[i], i, Kr, Vr)
    rewrite_err = max(np.abs(Kr[:, :seg.stop] - Kd[:, :seg.stop]).max(), np.abs(Vr[:, :seg.stop] - Vd[:, :seg.stop]).max())
    # (d) next step after the segment, same input token, four conditions
    i = seg.stop; tok = toks[i]
    ref = step(m, tok, i, Kd.copy(), Vd.copy(), "dense")
    out = dict(read_only=kl(ref, step(m, tok, i, Kd.copy(), Vd.copy(), "sparse")),
               kv_only=kl(ref, step(m, tok, i, K.copy(), Vc.copy(), "dense")),
               both=kl(ref, step(m, tok, i, K.copy(), Vc.copy(), "sparse")),
               after_rewrite=kl(ref, step(m, tok, i, Kr.copy(), Vr.copy(), "dense")))
    return rel, flagged, rewrite_err, out

# (a) the cache is exact: cached decode equals a recompute from scratch without any cache
m = make_model(0); toks = list(np.random.default_rng(5).integers(0, V, 40))
_, _, Pc = dense_reference(m, toks)
def no_cache_last(m, toks):                                             # batched forward, causal mask, no cache
    n = len(toks); h = m["E"][toks] + m["P"][:n]
    for l in range(L):
        x = norm(h); q = (x @ m["Wq"][l]).reshape(n, H, DH); k = (x @ m["Wk"][l]).reshape(n, H, DH); v = (x @ m["Wv"][l]).reshape(n, H, DH)
        s = np.einsum("ihd,jhd->hij", q, k) / np.sqrt(DH); s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
        h = h + np.einsum("hij,jhd->ihd", softmax(s), v).reshape(n, D) @ m["Wo"][l]
        h = h + np.maximum(norm(h) @ m["W1"][l], 0.0) @ m["W2"][l]
    return softmax(norm(h) @ m["U"])
print("(a) max |cached - recomputed| over 40 positions:", f"{np.abs(Pc - no_cache_last(m, toks)).max():.1e}")

res = [run(s) for s in range(20)]
rel = np.array([r[0] for r in res]); print("(b) relative error of sparse-written K per layer (mean of 20 models):", np.round(rel.mean(0), 4))
print("    layer-1 error is exactly zero in every run:", bool((rel[:, 0] == 0).all()))
print("(c) tokens of the 32 that a greedy dense check flags (mean, min, max):", np.mean([r[1] for r in res]), min(r[1] for r in res), max(r[1] for r in res))
print("    max |rewritten KV - dense KV| over all runs:", f"{max(r[2] for r in res):.1e}")
for key in ["read_only", "kv_only", "both", "after_rewrite"]:
    vals = np.array([r[3][key] for r in res]); print(f"(d) KL(dense on clean KV || {key:13s}) mean {vals.mean():.4f}  median {np.median(vals):.4f}")
```

**Real output.**

```
(a) max |cached - recomputed| over 40 positions: 4.4e-16
(b) relative error of sparse-written K per layer (mean of 20 models): [0.     0.0747 0.1024 0.1218]
    layer-1 error is exactly zero in every run: True
(c) tokens of the 32 that a greedy dense check flags (mean, min, max): 5.25 2 9
    max |rewritten KV - dense KV| over all runs: 0.0e+00
(d) KL(dense on clean KV || read_only    ) mean 0.0112  median 0.0097
(d) KL(dense on clean KV || kv_only      ) mean 0.0006  median 0.0004
(d) KL(dense on clean KV || both         ) mean 0.0116  median 0.0102
(d) KL(dense on clean KV || after_rewrite) mean 0.0000  median 0.0000
```

**How the tutor checks it.** Four structural results must hold exactly, whatever the seed: (a) is at rounding level; the layer-1 error in (b) is exactly 0 and the deeper layers are not; the rewrite in (c) restores the dense cache exactly; "after_rewrite" in (d) is 0. Then ask three questions. First: "The rewrite made the cache exact, yet (c) says about 5 of the 32 tokens are not the dense choice. What did the rewrite not repair?" (the tokens). Second: "Where did you get the list of flagged tokens?" (from the same dense pass that produced the rewritten KV; this is fact 15). Third: "In this toy, the read error is about 20 times larger than the stored-KV error at the next step. May you conclude that for Qwen3?" The correct answer is no: the weights are random, only 32 of 128 positions are sparse-written, and the measure is one step, not trace length. The toy shows that the three terms can be separated, not how large they are in a real model. (In (c) the rewrite is written as a loop of dense steps; a real engine does it as one prefill-shaped pass.)

### Exercise 3. Reading a two-repair ablation (pen and paper, 15 minutes)

**Task.** (1) From the Sirius table in fact 19, take the four rows that all contain interleave as a 2×2: neither = + interleave; KV = + KV rewrite + interleave; rollback = + rollback + interleave; both = all three. Compute for CSparse and FSparse the share of the gap between the full model and "neither" that is recovered by KV rewrite, by rollback, and by both; also the sum of the two single shares. Then express every row as a share of the gap between the full model and plain sparse. (2) Do the same for ReSA's single arm (fact 18). (3) Let t = 0.30 be the failure probability that the token channel would cause alone and v = 0.15 the one the KV channel would cause alone. Compute the same four numbers in three toy worlds: failure if either cause fires independently, 1 − (1 − t)(1 − v); additive failure t + v; failure only when both act, 4·t·v. (4) Add a fourth world: independent causes, but the token repair also catches 80% of the failures caused by the KV. Which world has the pattern "sum of single shares is larger than the joint share", as in Sirius?

**Solution code.**

```python
# Exercise 3: reading a two-repair ablation. Share of the gap each repair recovers, and whether shares add up.
import numpy as np

def shares(full, none, kv_only, tok_only, both, name):
    gap = full - none
    a, b, ab = (kv_only - none) / gap, (tok_only - none) / gap, (both - none) / gap
    print(f"{name:34s} gap {gap:.4f} | KV rewrite {a:6.1%} | token repair {b:6.1%} | sum {a+b:6.1%} | both {ab:6.1%}")

print("-- Sirius Table 4 (Llama-3-8B-Instruct, 'GSM8K 20%'; accuracy). Base = sparse + interleave. Token repair = rollback.")
shares(0.7538, 0.3826, 0.4886, 0.6591, 0.6667, "CSparse (interleave base)")
shares(0.7538, 0.6288, 0.6780, 0.7273, 0.7273, "FSparse (interleave base)")

print("-- Same table, every row as a share of the gap between full and plain sparse")
rows = ["+ interleave", "+ KV rewrite", "+ KV rewrite + interleave", "+ rollback + interleave", "+ all three"]
table = {"CSparse": (0.7538, 0.3674, [0.3826, 0.4735, 0.4886, 0.6591, 0.6667]),
         "FSparse": (0.7538, 0.5644, [0.6288, 0.6629, 0.6780, 0.7273, 0.7273])}
for name, (full, sparse, vals) in table.items():
    print(f"{name}: gap {full-sparse:.4f} | " + " | ".join(f"{r} {(x-sparse)/(full-sparse):.1%}" for r, x in zip(rows, vals)))

print("-- ReSA Table 1 (accuracy, average of 5 math sets). Only the KV-rewrite arm exists.")
for name, dense, sparse, resa in [("R1-Distill-Qwen 1.5B", 46.82, 43.50, 45.56), ("R1-Distill-Qwen 7B", 60.72, 57.72, 60.52)]:
    print(f"{name:34s} gap {dense-sparse:.2f} | KV rewrite {(resa-sparse)/(dense-sparse):6.1%}")

print("-- Three toy worlds with the same single-repair inputs: t = harm via tokens, v = harm via stored KV")
t, v = 0.30, 0.15            # failure probability each channel would cause alone
worlds = {
    "independent causes (either one fails)": lambda t, v: 1 - (1 - t) * (1 - v),
    "additive causes":                       lambda t, v: t + v,
    "needs both (interaction)":              lambda t, v: 4 * t * v,
}
for name, f in worlds.items():
    none, kv_fixed, tok_fixed, both = f(t, v), f(t, 0), f(0, v), f(0, 0)
    shares(1 - both, 1 - none, 1 - kv_fixed, 1 - tok_fixed, 1 - both, name)

print("-- A fourth world: the token check also catches harm that came through the KV (as in rollback)")
catch = 0.8                  # share of KV-caused failures that show up as a token the check rejects
none = 1 - (1 - t) * (1 - v)
kv_fixed = t
tok_fixed = v * (1 - catch)
shares(1.0, 1 - none, 1 - kv_fixed, 1 - tok_fixed, 1.0, "token check covers 80% of KV harm")
```

**Real output.**

```
-- Sirius Table 4 (Llama-3-8B-Instruct, 'GSM8K 20%'; accuracy). Base = sparse + interleave. Token repair = rollback.
CSparse (interleave base)          gap 0.3712 | KV rewrite  28.6% | token repair  74.5% | sum 103.0% | both  76.5%
FSparse (interleave base)          gap 0.1250 | KV rewrite  39.4% | token repair  78.8% | sum 118.2% | both  78.8%
-- Same table, every row as a share of the gap between full and plain sparse
CSparse: gap 0.3864 | + interleave 3.9% | + KV rewrite 27.5% | + KV rewrite + interleave 31.4% | + rollback + interleave 75.5% | + all three 77.5%
FSparse: gap 0.1894 | + interleave 34.0% | + KV rewrite 52.0% | + KV rewrite + interleave 60.0% | + rollback + interleave 86.0% | + all three 86.0%
-- ReSA Table 1 (accuracy, average of 5 math sets). Only the KV-rewrite arm exists.
R1-Distill-Qwen 1.5B               gap 3.32 | KV rewrite  62.0%
R1-Distill-Qwen 7B                 gap 3.00 | KV rewrite  93.3%
-- Three toy worlds with the same single-repair inputs: t = harm via tokens, v = harm via stored KV
independent causes (either one fails) gap 0.4050 | KV rewrite  25.9% | token repair  63.0% | sum  88.9% | both 100.0%
additive causes                    gap 0.4500 | KV rewrite  33.3% | token repair  66.7% | sum 100.0% | both 100.0%
needs both (interaction)           gap 0.1800 | KV rewrite 100.0% | token repair 100.0% | sum 200.0% | both 100.0%
-- A fourth world: the token check also catches harm that came through the KV (as in rollback)
token check covers 80% of KV harm  gap 0.4050 | KV rewrite  25.9% | token repair  92.6% | sum 118.5% | both 100.0%
```

**How the tutor checks it.** The Sirius shares must match the first two lines. The learner should say: with independent causes the single shares sum to *less* than the joint share; with additive causes they sum to exactly the joint share; they sum to *more* only when the causes interact or when one repair also covers the other's harm. Sirius shows the third pattern (103.0% against 76.5%, 118.2% against 78.8%), so its table is not a split of the harm into channels. A common slip: taking "+ KV rewrite" (no interleave) and "+ rollback + interleave" as the two single repairs. Those two rows differ in two things, so they are not a clean pair. The learner should also notice two limits: the Sirius "both" arm stays below 100% (a likely reason, not stated in the paper: the rollback rule is a threshold, not the exact rule); and ReSA's 93% rests on a 3-point gap where one benchmark (AIME24, 30 problems) has ReSA above Dense (51.1 against 48.1 [checked: 2506.04108 Table 1]), so it is inside noise. Last question: "For the planned experiment, which single number would you report instead of 'share per channel'?" A good answer: the effect of each repair with and without the other (all four cells), with paired confidence intervals.

---

## 6. Diagnostic picks

- **Level 2-3: kv_channels-Q4** (which layers' KV does a sparse step corrupt). One short answer shows whether the learner sees that a step writes as well as reads, and that the write depends on the reads below it. The common wrong answer ("sparse attention only changes reads") blocks everything about the KV channel.
- **Level 4-5: kv_channels-Q8** (the flawed "tokens only" arm). A level-4 answer sees that a clean verifier makes the arm lossless and names arms B and C. A level-5 answer adds that skipping the overwrite silently changes the verifier, and that arm C is a diagnostic with no cheap system behind it. It maps directly onto the design of Part C.

---

## 7. Sources

Fetched in this run (through a fetch tool that returns extracted answers from the page, first 100,000 characters of each page unless noted):

| id | title (as shown on the page) | what I read |
|---|---|---|
| arXiv 2506.04108 | Rectified Sparse Attention (Sun, Ye, Dong, Xia, Chen, Gao, Cao, Wang, Wei) | arXiv HTML v2: abstract, Sec 1, Sec 2.1-2.3 (Eq. 3-6), Sec 3.1, Sec 3.3, Table 1, Sec 3.5, Appendix B and Table 3. The figures of Sec 3.3 were not readable as numbers. |
| arXiv 2512.01278 | Accelerating Large-Scale Reasoning Model Inference: Self-Speculative Decoding with Sparse Attention (SparseSpec; Zhao et al.) | arXiv HTML v1: abstract, Sec 1, 2.1, 3.1, 3.2 (latency formulas), 4.1, 4.3, 4.4, 5.3, 5.4, 5.5, Table 2. No sentence on KV overwrite was found. |
| arXiv 2409.03856 | Sirius: Contextual Sparsity with Correction for Efficient LLMs (Zhou, Chen, Xu, Lin, Chen) | arXiv HTML v1: Sec 1, 2.1, 2.2, 3.1, 3.2, 3.4, 4, 4.1, 5.2, 5.3, Table 4 (the first pass read a garbled copy and guessed the row labels; the examiner pass read the table in the raw HTML and the rows in fact 19 are the real ones); the last 14,000 characters (Tables 17-20) were also read and contain no algorithm or ablation. |
| arXiv 2501.01005 | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving (Ye et al.) | arXiv HTML v2: abstract, Sec 2.1, 2.3, 3.1.1, 3.1.2, 3.2.1-3.2.3, 3.3.1, 4.3, 4.4, Appendix A, B. Table 5 was not in the part read. |

Read directly on disk:

| source | what I read |
|---|---|
| `/home/ryankert_l/SparseSpec/serve/scheduler/spec_scheduler.py` | the linear VERIFY input-building branch (lines 360-420) and `verify_outputs` (lines 540-575) |
| `/home/ryankert_l/SparseSpec/serve/request/kv_cache_ptr/base.py`, `/home/ryankert_l/SparseSpec/3rdparty/flashinfer/flashinfer/page.py` | examiner pass: `dispatch_append_kv_cache` and `get_batch_indices_positions` (where the verify pass writes its KV) |
| `/share2/huggingface/hub/models--Qwen--Qwen3-8B/…/config.json`, `…Qwen3-4B/…/config.json` | layer, head and dimension fields |
| `research_notes/Inference speed question search/verification.md` | section F38 in full (lines 445-654); keyword search elsewhere |
| `reports/length-tax.html` | full text |
| `research_notes/Length tax preliminaries/specdec_exact.md` | Core ideas, Level ladder, Sources (for format and to avoid overlap) |

Cited from the notes only, not re-read in this run [from notes]: 2610.23818 (Acceptance Is Not Speed), 2610.13202 (S3R), 2602.07223 (Vegas), 2410.11305 (QSpec), 2602.02711 (Dynamic Mixed-Precision Routing), and the F38 design (arms B and C, oracle-selection control). Palu is named only as an example of a lossy KV store; no Palu paper was read in this run.

Not verified: the GPU numbers used for the machine balance (about 1000 TFLOP/s, about 3 TB/s) are from memory. In Sirius Table 4, three things were not found: the meaning of "GSM8K 20%" in the header (probably a 20% subsample; another table of the paper says "30% subsampled GSM8K"), the meaning of the two numbers in each cell, and the threshold and kernel size used. The full list is in the Verification log.

---

## 8. Verification log

Examiner pass finished: Fri Nov 13 03:33:49 PM CST 2026. CPU only; no GPU job was run. Nothing was posted, committed or pushed.

**What was done.** 53 items were checked: 20 core ideas, 8 misconceptions, 20 questions, 3 exercises, 2 diagnostic picks. Each question was solved before its key was read. All numeric keys were recomputed. The raw arXiv HTML of the four papers (2506.04108, 2512.01278, 2409.03856, 2501.01005) was downloaded, each id was checked against the page title, and every [checked] fact was read there again. The [from notes] facts were compared with the notes on disk, not with the papers. The code of all three exercises was extracted from this file and re-run; all three outputs are identical to the pasted ones (Exercise 3 after its correction).

**Wrong, now corrected.**

1. Sirius Table 4 row labels (fact 19, fact 20, Q10, Exercise 3, misconception text that used the numbers). The first pass guessed the rows from a garbled copy. It dropped the three "interleave" rows and called 0.6591 "rollback". The real row is "rollback + interleave"; there is no "rollback alone" row. The full table is now in fact 19. The clean 2×2 uses the interleave rows: CSparse 0.3826 / 0.4886 / 0.6591 / 0.6667; shares 28.6%, 74.5%, sum 103.0%, both 76.5% (was 27.5%, 75.5%, 103%, 77.5% on a mixed base). FSparse: 39.4%, 78.8%, sum 118.2%, both 78.8% (was 52%, 86%, 138%, 86%). The conclusion did not change: single shares sum to more than the joint share. Exercise 3 code and output were replaced and re-run.
2. "About 8 of 16 tokens would be rejected" (misconception 4) and "about half of each 16 tokens would be rejected" (fact 20). ReSA says about 8 of 16 are *accepted* per round. The round stops at the first rejection; the rest are discarded, not rejected. At a constant rate this is about one rejection in 11 tokens.
3. "Relaxed speculative decoding with a separate drafter already lengthens completions" (fact 13, Q16). The notes say: mean 1.54×, median 1.08×, 248 of 360 cells; median 0.94 for standalone drafters, 1.23 for EAGLE heads. The text now gives these numbers and warns against the general claim.
4. Q14 key, item 4: "A re-encode over compressed entries changes nothing." Not right. A batched dense re-encode removes part of the inherited error; it never removes an entry's own compression error. Rewritten.
5. Fact 4, last sentence: "only through the KV entries at its own position". A token also acts through its own step's forward pass, which emits the next token. Rewritten.

**Ambiguous, now rewritten.**

6. Q8: "which two arms carry real information" had more than one defensible answer (plain sparse and ReSA at f > 1 are informative too). It now asks which two single-fault arms should replace the flawed arm. Fact 17 was changed in the same way.
7. Q15: the key ("not token errors compounding") needs acceptance to be counted only in rounds where the earlier drafts passed. The question now states this condition. The follow-up was rewritten to match.
8. Q4: added "starting from the same dense-written cache"; "error grows with depth" is marked as a toy result.
9. Q10: the question now says that all four sparse arms include interleave, so it is self-contained.

**Minor.**

10. Sirius quotes were cited as Sec 4.2; they belong to Sec 4.1 (facts 12, 19).
11. Fact 19: the one-line description of contextual sparsity was "[unverified wording]". It is replaced by the paper's own sentence (Sec 2.1) plus a ReLU picture marked as mine. CSparse and FSparse are now defined.
12. Fact 14: b is named as ReSA's block size (it clashed with bytes per number in fact 5); the p inconsistency is now [checked], with the three places named; 6.16 of 8 (average over models and datasets) and Table 2 (Qwen3-8B, AIME) are not from the same run.
13. Fact 15 and Q9: ReSA's 1.92× is throughput, not traffic. "The two systems differ only in rollback" is limited to the use of the dense pass.
14. Fact 16: the KV overwrite in SparseSpec is now confirmed from code (three files named), not inferred.
15. Fact 7: added SparseSpec's own limit (small batch: MLP is memory-bound too). Fact 9: added the 10% prefill gap. Fact 18: Table 1 has one "Avg Length" row per model, not per method. Fact 1: note on rotary positions.
16. Q2: wrong-answer time is 75.5 ms, not 75 ms. Q9: second route to the wrong answer 2 (16/8). Misconception 8: ratios written as 1.12 and 1.78.
17. Exercise 1 check text: the parameter count 8.2 billion is confirmed from the config (8.19 billion); a step does not read the whole input embedding table, so 15.1 GB is nearer than 16.4 GB and the KV share is 0.84, not 0.82. The code was left as it was.
18. Exercise 3 check text: "both stays below 100% because the rollback rule is a threshold" is now marked as a likely reason, not a fact.

**Checked and left unchanged.** Facts 2, 3, 5, 6, 8, 10, 11; misconceptions 1, 2, 3, 5, 6, 7; Q1, Q3, Q5, Q6, Q7, Q11, Q12, Q13, Q17, Q18, Q19, Q20; Exercises 1 and 2 (code and output); both diagnostic picks. No item was deleted.

**Still unverified.**

- The GPU numbers for the machine balance (about 1000 TFLOP/s, about 3 TB/s): from memory. Every "about 300 FLOPs per byte" statement rests on them.
- Sirius Table 4: the meaning of "GSM8K 20%" (probably a 20% subsample); the meaning of the two numbers "A/B" in each cell; the threshold and kernel size used for the table.
- Sirius: whether the full model's check in the row with rollback and no KV rewrite reads KV written by the sparse model. Likely, because the cache is shared; the paper does not say it.
- All [from notes] facts were compared with the notes only: 2610.23818 (length numbers), 2610.13202 (S3R quote on overwrite; 95.8% and 67.9%), 2602.07223 (Vegas), 2410.11305 (QSpec), 2602.02711 (68.2% against 61.6%), and the F38 design (arms B and C, controls). These arXiv ids were not opened in either pass.
- Whether S3R's per-position acceptance (95.8% to 67.9%) is conditional on the earlier drafts being accepted.
- Palu was not read; it is named only as an example of a lossy KV store. The claim in Q14 that a lossy store compresses each entry again at write time is the usual design, not a checked fact about Palu.
- "The usual engines store the key after the rotation" (fact 1) is from general knowledge of the common implementations, not checked in vLLM code in this pass.
- The toy numbers of Exercise 2 (read error about 20 times the stored-KV error) say nothing about real models; the pack already warns about this.
