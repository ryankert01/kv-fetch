# Gaps, overlaps and teaching order for the eight tutor packs

Written 2026-11-13 15:47 CST. No GPU job was run. Nothing was posted, committed or pushed.

**What I read.** All eight packs in this folder (core ideas, ladder, misconceptions, the level line of every question; full question text only for `specdec_exact.md` Q7-Q21). `reports/length-tax.html` in full. In `Inference speed question search/verification.md`: the F07 and F38 cards in full, the top and the feasibility part of F01 and F17, and the completeness critic. The outline of `questions_ranked.md`. Parts of `kvfetch/sparse_decode.py`, `kvfetch/tasks/sweep.py`, `scripts/e3_repetition.py`.

**Tags.** [checked] = I read it in the source in this run. [checked, local code] = I read it in the code on this machine in this run. [from notes] = taken from the search notes, not re-read. [unverified] = from memory or inference.

**Short answer.** The packs teach the ideas well. They teach almost no tooling. A learner who finishes all eight can design Parts A, B and C on paper, but cannot yet build the arms. The largest holes are: the dual-pass step, per-token logging, the switch-back mechanics, and one paper (R2R) that no pack names.

---

## 1. Gaps: what no pack covers

Each gap says what is missing, why the first experiments need it, and where it should go.

### G1. Hooking sparse attention and a relaxed rule into a real decoder
- **Missing.** No pack shows how the engine works. Words with zero hits in all eight packs: `forward hook`, `past_key_values`, `position_ids`, `attention_mask`, `dual-pass`. `sparse_decode.py` is named in two packs only, and both say it was "searched, not read in full".
- **What the learner must be able to do.**
  - Read `kvfetch/sparse_decode.py`: it replaces the attention function (`_attention`, `register_sparse_attention`), holds one method per engine (`SparseEngine`, `Method`), and has a `shadow` flag that selects like the method but returns dense attention [checked, local code].
  - Add a token-level per-head top-K method. The engine today is page-based. The F07 card asks for 5% of context, minimum 128, sink plus recent window, refresh every M steps [from notes: verification.md F07, feasibility].
  - Know where a relaxed rule lives in a real sampler: `vllm/v1/sample/rejection_sampler.py` in vLLM 0.32, and `serve/sampling/rejection_sampler.py` in SparseSpec [from notes: verification.md F01 feasibility; F07 feasibility]. Know that sampler filters (top-k, top-p, temperature) must be applied to p before the rule, or the "exact" arm is not exact.
  - Know why the SparseSpec speculative path cannot be used here: it needs the FlashInfer persistent kernel, which does not run on Ampere cards (A6000) [from notes: verification.md F07, blockers].
- **Needed for.** Parts A and C (every sparse arm), Part A relaxed-rule arms.
- **Belongs in.** A new tooling pack. Short pointers from `sparse_attention.md` and `lossy_verify.md`.

### G2. The dual-pass step (one token, two forward passes, one of them owns the KV)
- **Missing.** `kv_channels.md` fact 17 defines arm B (sparse logits, dense-written KV) and arm C (dense read, sparse-written KV). No pack says how to build them.
- **What the learner must know.** In the engine the KV write goes to position `lens`, and `lens` moves only at the end of the step. So two passes in one step write the same slot, and the second pass wins. Sparse pass then dense pass gives arm B. Dense pass then sparse pass gives arm C. Estimated 40-60 changed lines, not written, not run [from notes: verification.md F38, feasibility]. Both passes must sit inside one CUDA graph, and the graphs must be captured again [from notes: verification.md F07, engine gaps].
- **Also missing.** The check that the build is right: with both passes dense, the output must equal the plain dense run token for token under the shared seed.
- **Needed for.** Part C (both isolating arms). Part A (measure the dose of sparse attention on a clean history).
- **Belongs in.** New tooling pack, with a pointer from `kv_channels.md` fact 17.

### G3. Logging per-token probabilities under the full model
- **Missing.** `logprob` has zero hits in all packs. The packs define dose measures (target probability, rank, excess surprisal) but never say how to get them from a run.
- **What the learner must know.**
  - The engine has a `logits_out` hook. It stores the full logit vector of every step on the CPU, and it turns CUDA graphs off (`graphs=graphs and logits_out is None`) [checked, local code: `run_batch`]. That is fine for a check, not for a study.
  - Size arithmetic: a vocabulary of about 152 thousand entries [unverified, from memory] in float32 is about 0.6 MB per token. One 8,192-token run is about 5 GB. So log scalars only: p(x), q(x), rank of x under p, in-support flag, entropy of p, lossless-accept flag [from notes: verification.md F07, engine gap (c)].
  - Two ways to get p: (i) the dual pass of G2, on line; (ii) replay a saved trace teacher-forced under dense with `Job.forced` [checked, local code]. Way (ii) is the cheap first step the F07 card names [from notes].
  - In vLLM 0.32 the sampler computes raw logprobs before logits processors run, so the target's log-probability of an injected token comes back in the normal `logprobs` field [from notes: verification.md F01, feasibility].
- **Needed for.** Parts A and B (dose per token, "which tokens need verifying").
- **Belongs in.** New tooling pack. `lossy_verify.md` should point to it where it defines dose.

### G4. Forcing a prefix and then continuing with a different method (Part B)
- **Missing.** Zero hits for `forced prefix`. `experiment_stats.md` idea 20 gives the design of the switch-back test. `reasoning_length.md` idea 19 says no source reports one. No pack says how to run it.
- **What the learner must know.**
  - `Job.forced` cannot do it today. A forced job ends at `len(forced)`; there is no "then sample freely" mode [checked, local code: `finished()`].
  - A forced job draws u = 0.0 and does not advance its random stream [checked, local code: the `u_t.copy_` line]. So after a forced prefix of t tokens the stream is at position 0, not t. The learner must choose: advance the stream t times (the exact arm then re-joins its own seed path), or start a fresh stream. The choice changes what the control means.
  - What happens to the KV at the switch: keep the lossy-written KV (tests tokens plus state), or re-encode the prefix densely (tests tokens only). This is the Part B / Part C boundary and must be stated per arm.
  - The twin control: switch exact to exact at the same t. It should give zero change. If it does not, the harness is wrong.
  - For quantization: a switch means a second set of weights in memory, and a decision whether the KV of the prefix is recomputed by the full-precision model.
  - How to choose t (fixed token count, fixed share of the exact length, or just before the first answer) and why each choice selects different runs.
- **Needed for.** Part B, entirely.
- **Belongs in.** New tooling pack. `experiment_stats.md` idea 20 should point to it.

### G5. Prompt format, thinking mode, stop tokens, end-of-thinking detection
- **Missing.** Zero hits for `chat template`, `tokenizer`, `enable_thinking`, `harmony`.
- **What the learner must know.** How each model marks thinking and stops: R1-Distill-Llama-8B, Qwen3, GPT-OSS (harmony format, not tested in this engine) [from notes: verification.md F07, engine gap (d) and blockers]. Where the helpers are: `kvfetch/models.py` has `format_prompt` and `stop_token_ids` [checked, local code: function names only]. Which token id closes the thinking block, and what to do when the model opens it again (`reasoning_length.md` idea 10 names this, without a recipe).
- **Needed for.** All parts. A wrong stop id makes every length wrong.
- **Belongs in.** New tooling pack, or a short add-on to `reasoning_length.md`.

### G6. Measurement recipes: answer extraction, first appearance of the answer, loops
- **Missing.** Zero hits for `boxed`, `regex`, `answer extraction`, `n-gram`, `loop detect`. `reasoning_length.md` idea 6 says "first appears" means the reference answer shows up as a standalone number, and that it is a proxy. That is the only sentence on it.
- **What the learner must be able to do.**
  - Score the final answer. The repo takes the last `\boxed{...}` in the text [checked, local code: `kvfetch/tasks/sweep.py`, `score`].
  - Find the first token position where the reference answer appears: normalise numbers (commas, fractions, units), require word boundaries, map the character offset back to a token index. Know the false positives: small answers (for example "2") appear by chance; an intermediate value can equal the answer.
  - Know the second method and when to use it: forced readouts (close the thinking block at position t and read the answer). `reasoning_length.md` idea 7 describes this from a paper. It costs extra generation.
  - Detect a loop. The repo has `repeat_share(text, tail, n)` in `scripts/e3_repetition.py` [checked, local code]. No pack names it.
- **Needed for.** Part B ("does extra length come after the answer formed"), and the cap analysis in every part.
- **Belongs in.** `reasoning_length.md` (one new idea and one exercise with a regex on a toy trace).

### G7. Building the matched-dose control
- **Missing.** `experiment_stats.md` idea 19 says what to match. No pack gives a procedure. `rank profile` appears twice, both as a goal.
- **What the learner must be able to do.**
  - Arm R of the F07 card: random in-support substitutions at the same rate as the gated arm, with the same profile of rank under dense [from notes: verification.md F07, arm table]. This needs an algorithm: at each step decide by a coin whether to substitute, then draw a token whose dense rank follows the logged histogram.
  - Tune a temperature so that mean excess surprisal equals a target. This is a one-dimensional search on teacher-forced data.
  - Check the match offline before any closed-loop run.
  - Know the caveat from the F01 card: random rank swaps are not the same as real drafter overshoot tokens [from notes: verification.md F01, sharpened question].
  - Know that a nucleus gate at T = 1 is itself a truncation, so it needs its own dense top-p control [from notes: verification.md F07, design cautions].
- **Needed for.** Part A. Without this control "dose only" cannot be tested.
- **Belongs in.** `experiment_stats.md` (a new exercise), with the dose definitions from `lossy_verify.md`.

### G8. Prior work that no pack teaches
- **R2R** (arXiv 2505.21600, "R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing") [checked: id and title on the abs page]. Zero hits in all eight packs. It is the nearest published "token-only repair that works": a 1.5B model differs from the 32B model on 11% of tokens, 4.9% are divergent, and routing 11-15% of tokens to the large model keeps 92% of its accuracy [from notes: verification.md F38, closest papers]. It bears on "which tokens need verifying". Its limit: the weak path is a separate small model, not an approximation of the same model.
- **Sirius** (arXiv 2409.03856, "Sirius: Contextual Sparsity with Correction for Efficient LLMs") [checked: id and title on the abs page]. Covered in depth, but only in `kv_channels.md`. The blog page says it must be read first. A learner who starts with another pack will not meet it.
- **Acceptance Is Not Speed** (arXiv 2610.23818) [checked: id and title on the abs page]. Covered in six packs. No gap. Its released code (anonymous.4open.science/r/lossy-token-eff [from notes: verification.md F07]) is not named in any pack.
- **Papers with zero hits in the packs that the cards treat as close** [all from notes: verification.md F07 and F38; ids not re-checked by me]: LessIsMore 2508.07101 (length is set by the selector with no repair); Sparse-RL 2601.10079 (about 7% of unrepaired sparse rollouts hold a token outside dense support); 2610.05685 (forced close at loop onset recovers 15% of capped runs); KV-Rescue 2608.15797; Mix-Quant 2605.20315 (a precision 2x2 over prefill and decode); 2610.24346 (token replacement for low-bit agents nets about zero); 2610.24715 (repetition loops under truncation); SSV 2605.19893 and Dustin 2606.24957. Pivot-Aware SD 2511.00351 and Nucleus SD 2610.07822 have one hit each, in `lossy_verify.md` only.
- **The skill of a prior-work search.** The blog page says "a full search of prior work has to come first". No pack teaches how (citation-graph scan from Sirius, ReSA and 2610.23818; what a null search does and does not show).
- **Belongs in.** A one-page reading list at the front of the course. R2R also in `kv_channels.md` (token-only repair) and `lossy_verify.md` (selective verification).

### G9. Choosing the model, the task and the cap; counting GPU time
- **Missing.** Zero hits for `GPU-hour`, `GPU-day`, `compute budget`.
- **What the learner must know.**
  - Use a model whose length reacts. The F07 card says Qwen3-8B is flat and suggests R1-Distill-Llama-8B or GPT-OSS-20B; GPT-OSS-20B is not on disk [from notes].
  - Pick the cap after a dense pilot: dense cap-hit rate below 10%. AIME at 16K has 41% dense cap hits, so drop it for week 1; MATH-500 at 8,192 is the proposal [from notes: verification.md F07].
  - Budget arithmetic: GPU-hours = runs x mean tokens / (tokens per second x 3600). Example: 400 runs x 3,500 tokens = 1.4M tokens; at 165 tokens per second that is 2.4 GPU-hours for one single-pass arm. Dual-pass arms cost about twice as much. The card's total is about 26 GPU-hours per model-and-sampler setting [from notes].
  - Disk: scalars only, about 0.5 GB [from notes].
  - Shared machine rules: never GPU 0; use a free card among 1-6.
  - The vLLM 0.32 environment named in the cards lives under `/tmp` and may be gone [from notes: verification.md F01 and F17].
- **Needed for.** All parts, before the first run.
- **Belongs in.** New tooling pack (a "pilot checklist").

### G10. Decision rules written before the run
- **Missing.** `pre-regist` appears only in `reasoning_length.md` (3 hits). `experiment_stats.md` has none. Zero hits for `Bonferroni`, `Holm`, `multiple comparison`.
- **What the learner must know.** Write the bands first (F07 card: outcome A if the gated arm is at or below 1.05; outcome B if it is at or above 1.15 and the random arm is at or below 1.05; the middle is "partial") [from notes]. With seven arms and several metrics, some differences look real by chance; fix one primary metric and one primary contrast per part.
- **Belongs in.** `experiment_stats.md`.

### G11. Tools for capped lengths that the packs describe but do not name
- **Missing.** Zero hits for `Kaplan`, `log-rank`, `restricted mean`, `hazard ratio`. `experiment_stats.md` ideas 10-13 teach the survival curve and the area under it up to the cap. With one shared cap that is enough. It is not enough when arms have different caps, or when runs are cut for other reasons (out of memory, a killed job).
- **Belongs in.** `experiment_stats.md`, as a short note, low priority.

### G12. Concepts the packs use but never teach
- **Coupling in practice.** The shared seed is used in four packs. Only `experiment_stats.md` idea 3 explains it. No pack says what breaks it: a different top-k/top-p filter, a batch-size change, a forced step that does not advance the stream (see G4).
- **Reading the 2x2 as a causal split.** `mediation` has zero hits. `experiment_stats.md` gives shares and the interaction term. It does not name the general idea (direct effect, effect through a mediator, why shares need not add). Low priority; the pack's own treatment is workable.
- **The third account.** The F38 card says a selector-oracle arm is needed, or "neither repair recovers a third" will be misread [from notes]. `kv_channels.md` lists the oracle control at Level 5 and in one answer key. `experiment_stats.md` does not put it in the 2x2. EpiKV's position-convention artifact (H2O loops on 100 of 100 problems with re-embedded positions) [from notes: verification.md F38] has zero hits; it is a tooling trap for any arm that drops or reorders KV.
- **Other lossy state.** KV eviction: 4 hits in `sparse_attention.md`, 2 in `reasoning_length.md`, none teach it. Low-rank KV (Palu): named in `kv_channels.md` only as an example. Hybrid recurrent state (Qwen3.5): named in three packs, never taught. These are outside Parts A-C as written, so this is a note, not a must.

---

## 2. Overlaps between packs

Same content, taught more than once. The tutor should teach it in the first pack listed and only recall it later.

1. Accept rule and residual: `specdec_exact` facts 2-4; `lossy_verify` idea 1.
2. Lenience rule: `specdec_exact` fact 20, Q8, Exercise 3; `lossy_verify` idea 7.
3. Tokens per round and the speedup formula: `specdec_exact` 9-12; `lossy_verify` 16-17; `speed_accounting` 2-5; `reasoning_length` 3; `sparse_attention` 18.
4. Length inflation λ, round ratio λ/g, ceiling: `lossy_verify` 17-18; `reasoning_length` 3-4; `speed_accounting` 12-14 and 16; `experiment_stats` 1.
5. The cap and censoring: `reasoning_length` 11-12; `experiment_stats` 10-13; `speed_accounting` 17 and misconception 8; `sparse_attention` 20.
6. Heavy tails, mean against median: `reasoning_length` 5; `experiment_stats` 14-15; `speed_accounting` 16-17.
7. The switch-back toy exercise appears in at least five places: `lossy_verify` Ex 3; `reasoning_length` Ex 2; `experiment_stats` Ex 3; `speed_accounting` Ex 3 part 2; `sparse_attention` Ex 3; also `quantization` Q18.
8. Two channels, layer-1 KV exact, verify pass overwrites KV: `kv_channels` 10-17; `sparse_attention` 14-15; `specdec_exact` 18-19; `experiment_stats` 20; `quantization` 19.
9. SparseSpec traffic formula and Table 2: `kv_channels` 14; `sparse_attention` 18; `speed_accounting` 9 and 11.
10. S3R acceptance by position (95.8 to 67.9): `specdec_exact` 10; `sparse_attention` 17.
11. S3R budget sweep (80.4 to 96.7; 1.370 to 1.058): `specdec_exact` Q19; `sparse_attention` 18; `speed_accounting` 8.
12. Noise floor (2506.09501) and the zero-dose control: `specdec_exact` Q17; `reasoning_length` 18; `experiment_stats` 4-5.
13. Hesitation tokens (2606.00206): `quantization` 16-17; `reasoning_length` 16.
14. Dose and Table B2 of 2610.23818: `lossy_verify` 13-15; `reasoning_length` 17; `experiment_stats` 19; `specdec_exact` Q18.
15. The App. F guard: `lossy_verify` 20; `reasoning_length` 20; `experiment_stats` 17.
16. ReSA: `kv_channels` 13-18; `sparse_attention` 15; `specdec_exact` Q12; `experiment_stats` 20.
17. The T = 1.2 control: `lossy_verify` 20; `reasoning_length` 18; `experiment_stats` 9.
18. Teacher-forced against closed-loop: `sparse_attention` 13 (taught); used in `quantization`, `lossy_verify`, `speed_accounting`.

**Conflicts to fix before teaching.**
- The SparseSpec KV overwrite is "[unverified] inference" in `specdec_exact` fact 19 and `sparse_attention` fact 15, but "confirmed from the code" in `kv_channels` fact 16. One of them must change.
- Arm names differ: B and C in `kv_channels`; D, U, T, S in `experiment_stats`; D, U, T, G, S, R, H in the F07 card. Use the card's names everywhere.
- Symbols clash across packs: α, λ, k / γ / N, s, p, τ, b each mean different things in different packs.
- Three "ReLU" pictures: the residual norm(ReLU(p − q)) in `specdec_exact` (this is the learner's own picture); thresholded attention weights in `sparse_attention` misconception 1; an MLP neuron in `kv_channels` fact 19. Keep the word for the residual only.

---

## 3. Teaching order (prerequisites first)

1. `specdec_exact` — the exact rule and the residual. The learner's last answer ("like a ReLU layer") starts here. Everything else uses p, q, accept, residual.
2. `lossy_verify` — relax the rule just learned. Gives the template (reject, residual, bonus) and the dose measures. Needs only pack 1.
3. `reasoning_length` — the outcome variable: end of thinking, λ, the cap, first answer, per-token cost against lasting damage. Needs λ from pack 2. Every later pack reports lengths.
4. `sparse_attention` — the second lossy method: coverage, read error, teacher-forced against closed-loop, the E3 example. Needs the cap from pack 3.
5. `kv_channels` — splits the sparse error into token, stored KV and current-step read. Needs pack 1 (what the verify pass writes) and pack 4 (read error). Sirius is here.
6. `quantization` — the third method. Uses both channels (pack 5) and hesitation tokens (pack 3).
7. `experiment_stats` — pairs, noise control, censoring, matched dose, switch-back, the 2x2. It pulls together dose (2), the cap (3) and the channels (5), so it must come after them.
8. `speed_accounting` — break-even λ* = g, Amdahl, screening. Needed least for the first experiments, and it is the learner's strongest area.

Then the missing ninth unit: **tooling** (G1-G5, G9), taught on the real engine file. Without it the learner can plan but not run.

**By part.** Part A needs packs 1, 2, 3, 4, 7 and gaps G1, G2, G3, G7. Part B needs packs 2, 3, 7 and gaps G3, G4, G6. Part C needs packs 4, 5, 7 and gaps G1, G2.

**A shorter path, if time is short.** 1, 2, 3, then 7 (ideas 1-13 only), then the tooling unit, then the F07 cheap first step (teacher-forced replay of saved sparse traces under dense). Packs 4, 5, 6, 8 can follow.

---

## 4. Limits of this note

- I did not read the full text of every question, the exercises, or the source lists of seven packs. "Zero hits" comes from case-insensitive keyword searches. A pack may cover an idea under another word.
- Paper facts tagged [from notes] were not re-read by me. I confirmed three ids against their arXiv abs page titles only: 2409.03856, 2505.21600, 2610.23818.
- Line counts, throughput and cost figures for the engine come from the cards, written 2026-11-11. The code may have changed. I read only the parts of `sparse_decode.py` named above.
