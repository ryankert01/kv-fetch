# Harvest: the student's own surprising measurements, connected to published claims

Working notes, 2026-11-11. Lens: start from numbers in this repo that were surprising or left unexplained, find the published claim each one bears on, and phrase the open question.

Records read: `RESULTS.md`, `EXPERIMENTS.md`, `results/profile/NOTEBOOK.md`, `results/prefix/pilot/precision_summary.json`, `configs/prefix_pilot.yaml`, and the RL rollout research map (skimmed).

Dropped directions referred to below:
- D1: KV page selection at decode time for long reasoning
- D2: state checkpoints / prefix caching for hybrids
- D3: vLLM 0.32 single-GPU profiling
- D4: RL rollout acceleration

Three of the five cards sit on D1 data and one each on D2 / D4 data. That follows from the lens (the surprises live where the experiments were run), and each card says so.

Peer-review status matters here: most of the 2026 work on hybrid states and bf16 lag is preprint only. Each source below is marked accepted or preprint.

---

## Card 1. Why do small state errors persist while a wiped state recovers?

**Question.** Do small recurrent-state errors in a Gated DeltaNet hybrid persist because they sit in low-energy directions of the state that later keys rarely overwrite?

**Tension type.** Two preprints disagree, and the student's pilot shows both behaviours in one model.

**Own measurement** (`results/prefix/pilot/precision_summary.json`; Qwen3.5-4B, 81 checkpoints, 2,048 teacher-forced tokens after perturbing the GDN state; KL against the unperturbed run):

| Perturbation | KL, tokens 0-16 | KL, tokens 1,024-2,048 | Fade |
|---|---|---|---|
| State zeroed | 0.556 | 0.0086 | 65x |
| int4, group 128 | 0.159 | 0.0123 | 13x |
| int4, group 32 | 0.0179 | 0.0031 | 5.7x |
| int8 per row | 0.00080 | 0.00058 | 1.4x |
| Rank-32 truncation | 0.00080 | 0.00068 | 1.2x |
| Rank-64 truncation | 0.00029 | 0.00035 | grows 1.2x |
| Re-chunking only (bf16 kernel-path floor) | 0.00025 | 0.00028 | flat |

- Effective rank of the 128x128 state at 99% energy is 11.8 on average (1.1 to 19.4 by layer), yet keeping 32 directions still costs 2.8x the floor.
- Storing the state in bf16 gives KL exactly 0 for the first 64 tokens, then sits at the floor (0.93x overall).
- Chained checkpoints grow the error: fp8 1.02-1.05x floor once, 1.13x chained; int8 2.36x once, 3.75x chained (`EXPERIMENTS.md`, K0 pilot).
- Caveat: the rank-32 and int8 rows are close (1.2x against 1.4x). The strong contrast is by error size, not by error type. The spectral explanation is one candidate, not something the pilot shows.

**External claims.**
- Gated Delta Networks, ICLR 2025 (accepted), https://arxiv.org/abs/2412.06464: "gating enables rapid memory erasure while the delta rule facilitates targeted updates." Opened.
- STEPQuant, arXiv 2609.38169 (preprint), https://arxiv.org/html/2609.38169v1: "When retention is close to one, these errors can persist for many decoding steps." and "Thus, errors in directions rarely aligned with subsequent keys depend mainly on gate decay." Long reasoning tasks on Qwen3.8-27B: FP32 80.60, INT8 71.86, INT4 12.73. Opened.
- "Why Gated DeltaNet Survives 4-Bit Quantization", arXiv 2609.04098 (preprint), https://arxiv.org/html/2609.04098v1: "A one-off 1% state impulse injected at t=1,024 falls to 1/e within 80–1,382 steps." and "the decay-implied horizons 1/(1-α) of the same layers reach 44K–62K tokens." Opened.
- Ring-linear, "Every Attention Matters", arXiv 2510.19338 (technical report), https://arxiv.org/html/2510.19338: "If the KVCache is initialized as BF16 in the inference engine, errors will accumulate progressively during the recurrent process". Opened.
- State Rank Dynamics in Linear Attention LLMs, arXiv 2602.02195 (preprint), https://arxiv.org/html/2602.02195v1: "low-rank heads are indispensable for model reasoning, whereas high-rank heads exhibit significant redundancy." Pruning low-rank heads drops Qwen3-Next-Instruct NIAH from 93.8% to 46.9%. Opened.
- On State Reduction in Linear Attention, arXiv 2602.04852 (preprint), https://arxiv.org/abs/2602.04852: "the recurrent state of pretrained linear attention models can often be halved". Abstract opened.
- LeapQuant, arXiv 2609.38166 (preprint), https://arxiv.org/abs/2609.38166: "near-lossless performance under 8-bit recurrent-state quantization". Abstract opened.

**Why unknown.** STEPQuant states the key-alignment mechanism but reports no measurement of lifetime by state direction. The NVFP4 paper measures impulse lifetime on one model and one format, with no spectral breakdown. The rank papers study rank, not error persistence. None separates error carried in the state from error already written into the attention layers' KV cache.

**If yes.** The two preprints are both right, in different subspaces. Precision should be allocated by subspace (protect the tail, compress the head), and a stale or compressed state has a predictable cost by direction.

**If no.** The persistence comes from elsewhere: the bf16 floor re-seeding itself each step, or the error being copied into the KV cache of the softmax layers, where nothing decays. The same experiment separates these. This outcome is less tidy but still names the mechanism.

**First experiment.** Qwen3.5-4B, the 81 pilot checkpoints, run in fp32 to remove the bf16 floor. Inject perturbations of equal Frobenius norm confined to (a) the top-8 singular directions, (b) directions 9-32, (c) directions 33-128, (d) random. Teacher-force 8K tokens. Report the half-life of KL per subspace. Second arm: recompute the attention KV from clean hidden states to split state-carried from KV-carried error. About one GPU-day.

**Closest existing answer.** STEPQuant (mechanism stated, per-head half-life correlation about 0.80, no spectral test) and arXiv 2609.04098 (impulse lifetimes). No paper found that measures lifetime against singular direction.

**Overlap with dropped directions.** D2 (uses the K0 pilot data; the question is about error dynamics, not caching headroom) and D4 (stale state in RL rollouts). Risk: three groups published on GDN state quantization in September 2026, so this is crowded.

---

## Card 2. Is attention mass the wrong objective for sparse attention on long reasoning?

**Question.** On long reasoning traces, does a selector that tracks thought structure or redundancy beat a per-step attention-mass oracle at the same token budget?

**Tension type.** Published near-lossless claims at 5-16% of the cache against the student's oracle, which loses 10-14 points at about 4%.

**Own measurement** (`EXPERIMENTS.md`, E3 and H3; 30 AIME problems, avg@8):
- Qwen3-4B, 16K cap: full 48.3%, oracle page set 44.2% at 1K and 38.3% at 512.
- Qwen3.5-4B, 32K cap: full 49.6%, oracle at 512 loses 13.8 points, oracle at 1,024 scores 39.2% against 41.25% for the heuristic (difference 2.1, CI95 [-0.8, +5.0], not significant).
- Accuracy among samples that finish is 78-92% for every method; the loss is cap hits (43.8% of full-attention samples, 63.3% of ours at 512, 75.4% of Quest at 512).
- Mean trace length 11,917 tokens, so 512 tokens is about 4% and 1K about 9%.

**External claims.**
- ThinKV, ICLR 2026 Oral (accepted), https://proceedings.iclr.cc/paper_files/paper/2026/hash/b2b502c3629beadda06311386d2c6f73-Abstract-Conference.html: "achieves near-lossless accuracy with less than 5% of the original KV cache." Full text (https://arxiv.org/html/2510.01290v1): "Complete eviction (min R=0) severely degrades accuracy, as the model loses track of explored reasoning trajectories and results in an endless reasoning loop." Opened.
- R-KV, NeurIPS 2025 poster per the repo catalogue (accepted; the arXiv page itself states no venue), https://arxiv.org/abs/2505.24133: "preserves nearly 100% of the full KV cache performance using only 10% of the KV cache" and "Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache." Opened.
- LessIsMore, arXiv 2508.07101 (preprint), https://arxiv.org/html/2508.07101v1: "Quest, SeerAttention-r and TidalDecode each generate 30.0K, 19.8K, and 17.4K tokens." (full attention 14.8K, Qwen3-8B, AIME-24, budget 2K) and "attention recall serves as an indicator of both selection accuracy and computational efficiency". Opened.
- Quest, ICML 2024 (accepted), https://proceedings.mlr.press/v235/tang24l.html: "we observe the criticality of a token highly depends on the query." Opened.

**Why unknown.** ThinKV and R-KV were evaluated on DeepSeek-R1 distillations, GPT-OSS, QwQ and AceReason, not Qwen3 thinking models, and never against an attention-mass oracle. LessIsMore reports attention recall but no oracle row. Nobody has put a hindsight attention oracle and a thought-aware selector on the same traces.

**If yes.** Attention recall is not an upper bound and should stop being used as one. The objective to optimise is trajectory tracking, and the student's oracle rows become the evidence.

**If no.** The 5% and 10% claims do not transfer to Qwen3-class models or to traces of 12K tokens. That is a negative replication of two accepted papers, useful but less generative.

**First experiment.** Run R-KV's released code (and ThinKV if its code is available; not checked) on Qwen3-4B, the same 30 AIME problems, avg@8, 16K cap, budgets 512 and 1K. Compare with the existing oracle rows (38.3% and 44.2%). Report the paired accuracy difference with a bootstrap CI and the cap-hit rate.

**Closest existing answer.** ThinKV (thought types, loops when trajectories are lost), LessIsMore (cross-head selection keeps generation length at 2K+ budgets). CAOTE (ICLR 2026) and VATP argue attention score alone is insufficient, on short-output benchmarks; found by search only, not opened.

**Overlap with dropped directions.** D1, directly. It reopens D1 with a different objective. If the answer is yes, the student's own selector is not the contribution; the oracle comparison is.

---

## Card 3. What are the softmax layers of a hybrid doing?

**Question.** In a trained GDN-plus-attention hybrid, are the few softmax layers more diffuse and less predictable across layers than a dense model's because the recurrent layers have taken over the sharp lookups?

**Tension type.** A premise carried over from dense models (sparse, layer-stable attention) against measurements on a production hybrid.

**Own measurement** (`EXPERIMENTS.md`, H1; 30 problems, 8 attention layers):
- Oracle shared set at 512 misses 22.7% of attention on Qwen3.5-4B against 11.6% on Qwen3-4B (per layer 18-34% against 5-20%).
- Selector ranking flips: on the hybrid, mean-key 25.5% beats Quest 31.8%; on the dense model mean-key is worst at 19.5% and Quest 14.3%.
- Predicting the query across one GDN layer costs +3.4 points of missing attention, against +0.2 across one dense layer.
- KL per unit of missing attention is 7.4-7.5x lower on the hybrid (confounded: 8 perturbed layers against 36).
- Confound: the two models are different generations and were measured on their own traces.

**External claims.**
- A Systematic Analysis of Hybrid Linear Attention, arXiv 2507.06457 (preprint), https://arxiv.org/html/2507.06457v1: "While language modeling remains stable across linear-to-full attention ratios, recall significantly improves with increased full attention layers" and the limitation "Our analysis is confined to models up to 1.3B parameters, a 2,048-token context window, and block-wise mixing ratios." Abstract: "trained and open-sourced 72 models: 36 at 340M parameters (20B tokens) and 36 at 1.3B parameters (100B tokens)." Opened.
- HySparse, arXiv 2602.03560 (preprint), https://arxiv.org/html/2602.03560v1: "token saliency is stable across consecutive layers" and "Sparse attention directly reuses the KV cache and important token indices from the preceding full attention layer." Opened.
- Quest, ICML 2024 (accepted), https://proceedings.mlr.press/v235/tang24l.html: "we observe the criticality of a token highly depends on the query." Opened.

**Why unknown.** The systematic study measures recall against ratio, not attention patterns, and stops at 2,048 tokens. HySparse assumes stability in its own architecture. No paper found measures attention concentration in the softmax layers of Qwen3.5, Qwen3-Next or Kimi Linear.

**If yes.** Page selection and cross-layer index reuse built on dense models are the wrong tools for hybrids. Aggregate summaries (the mean-key result) fit better, and the linear-to-full ratio predicts how compressible the remaining attention is.

**If no.** The 22.7% against 11.6% gap is a property of this model pair (training data, trace length), dense-model methods transfer, and D1's small margin on the hybrid had another cause. This closes a confound; it is the less interesting answer.

**First experiment.** Part 1, controlled: the released 1.3B checkpoints from arXiv 2507.06457 (same data, ratio varied). On the same 2,048-token texts, measure for each softmax layer the attention mass outside the top 64 tokens and the overlap of top-64 sets between consecutive softmax layers; plot against ratio. Part 2: teacher-force Qwen3.5-2B/4B/9B and Qwen3-1.7B/4B/8B on identical token text and report oracle-at-512 missing attention per layer. Two to three days. Not checked: that the 72 checkpoints download and load.

**Closest existing answer.** arXiv 2507.06457 (recall against ratio). arXiv 2510.20787 and HyLRA (arXiv 2602.00777) were found by search and not opened. None found.

**Overlap with dropped directions.** D1, directly: the H1 numbers were measured for D1. A yes answer explains D1's result on the hybrid; it does not revive it.

---

## Card 4. Does lag below the numeric floor still hurt RL because it is directional?

**Question.** Does a policy lag smaller in KL than the rollout engine's numeric noise still cost reward in RL post-training because it is directional, where random numeric mismatch of the same KL does not?

**Tension type.** A published gain from removing a lag that the student's KL measurement cannot distinguish from noise.

**Own measurement** (`results/profile/NOTEBOOK.md`, R2):
- One Adam step at lr 1e-6 changes 3.4% of bf16 weights; the KL between the policy before and after that step equals the kernel-path floor (about 3e-4).
- After 32 GRPO steps the policy has moved 1.73e-3 to 1.88e-3 (about 5.6-6x the floor).
- A stale prefix cache on the hybrid costs 4.82e-4 to 5.70e-4 (1.55-1.74x the floor).
- Side discrepancy, not a card: the student saw 3.4% of weights change per step; the two papers below report about 1% and 0.4-0.5%.

**External claims.**
- AReaL, arXiv 2505.24298 (reported as NeurIPS 2025 in the repo's research map; the arXiv page states no venue), https://arxiv.org/html/2505.24298v3: "even with the decoupled objective, unbounded staleness (maximum staleness →∞) still results in inferior performance compared to the zero-staleness oracle." AIME24 on the 1.5B model: 42.0 at η = 0, 42.2 at η = 4, 36.9 at η = ∞; lr 2e-5. Opened.
- "bf16 Rounding Silently Discards Most RL Policy Updates", arXiv 2610.41073 (preprint), https://arxiv.org/html/2610.41073: "only 0.4–0.5% of served weights change per step"; the served policy lags by "50–150 steps"; Fair Cast is "ahead on 18, tied on 6 and behind on 1" of 25 seed pairs, mean gain 6.1 points; lr 1e-6; models up to 4B. Opened.
- "Defeating the Training-Inference Mismatch via FP16", arXiv 2510.26788 (preprint), https://arxiv.org/html/2510.26788v1: "We identify that the modern standard for mixed-precision training, BFloat16 (BF16), is the primary culprit." Opened.
- Weight update sparsity, arXiv 2602.03839 (preprint), https://arxiv.org/html/2602.03839v1: "we find that the phenomenon arises from the interaction between BF16 precision and learning rate". Opened. This already answers "why do so few weights change".

**Why unknown.** AReaL counts staleness in versions at lr 2e-5. Fair Cast measures per-token |log ρ| and effective sample size, not KL against a numeric floor, and has no matched-size random control. "Stale but Stable" (arXiv 2607.18722) uses the log-ratio as a staleness proxy but does not relate it to precision. No paper found compares directional and random mismatch at equal KL.

**If yes.** Staleness has to be measured with a sign (projection on the update direction), not by KL or version count. A stale cache or a quantised state can then be judged by its direction, not its size.

**If no.** KL in units of the numeric floor is a usable budget for any mismatch. The stale cache at 1.6x the floor has a predictable cost, and Fair Cast's gain needs another explanation.

**First experiment.** Qwen3-1.7B, GRPO at lr 1e-6, one Reasoning Gym task where Fair Cast reports a gain. Arms: (a) synchronous; (b) rollouts from weights k steps old, k chosen so the behaviour-to-current KL is 2x the floor; (c) current weights with a random kernel-path perturbation of matched KL; five seeds each. Report final reward per arm and the per-batch KL in floor units. Caveat: Fair Cast needed 25 seed pairs to show 18/6/1, so five seeds may not resolve it in a week.

**Closest existing answer.** arXiv 2610.41073 (version lag and rounding lag are additive) and arXiv 2607.18722 (log-ratio proxy). ENTR (arXiv 2607.22186) reports stable training "up to 30 policy versions of staleness". Abstracts opened for the last two.

**Overlap with dropped directions.** D4, directly: it is a child of R2. The question is about the learning signal, not throughput. If the answer licenses stale-cache reuse, the throughput gain is the same small one that closed D4.

---

## Card 5. Is there an open-loop metric that predicts closed-loop collapse?

**Question.** Is there a teacher-forced metric that predicts which sparse-attention selectors fail to terminate on long reasoning, given that mean missing attention does not?

**Tension type.** A common evaluation proxy against the student's open-loop and closed-loop numbers. Sibling of card 2.

**Own measurement** (`EXPERIMENTS.md`, E-track and E3, Qwen3-4B at 1K):
- Offline, SnapKV-shared misses 8.6% of attention and Quest 8.0%: a gap of 0.6 points.
- Online, SnapKV-shared scores 8.3% and Quest 40.4%: a gap of 32 points. SnapKV's truncated outputs are verbatim loops.
- Using observed attention history adds nothing online (+0.0, CI95 [-3.8, 3.8]).
- The eviction ceiling with true future attention misses 8.5%, no better than SnapKV's 8.6%, because evicted pages revive.

**External claims.**
- LessIsMore, arXiv 2508.07101 (preprint), https://arxiv.org/html/2508.07101v1: "attention recall serves as an indicator of both selection accuracy and computational efficiency" and "Sparse attention methods exhibit a concerning tendency that extends generation lengths on reasoning tasks". Opened.
- Hold Onto That Thought, arXiv 2512.12008 (preprint), https://arxiv.org/html/2512.12008v1: "lower budgets are capable of triggering longer reasoning traces" and "we discover that H2O and our decoding-enabled variant of SnapKV are dominant strategies for reasoning models". Max 2,048 new tokens. Opened.
- ThinKV, ICLR 2026 Oral (accepted), https://arxiv.org/html/2510.01290v1: "quantization can inflate generation length by up to 5.1×across datasets and techniques." Opened.

**Why unknown.** Papers report longer traces as an outcome, never predicted from an offline number. Hold Onto That Thought has no breakdown of wrong answers against non-termination and caps output at 2,048 tokens, where the student's traces average 12K.

**If yes.** Selectors can be screened without closed-loop runs, and the metric itself says what the model needs to finish.

**If no.** Closed-loop evaluation at full length is mandatory, and results from short caps or attention recall cannot be trusted. Useful as methodology, weaker as a finding.

**First experiment.** Reuse the existing cells (about five selectors, two budgets, Qwen3-4B and Qwen3.5-4B). On full-attention traces compute candidate metrics: mean and 99th-percentile missing attention per step, missing attention on the model's own recent output, next-token KL, top-1 flip rate. Report the rank correlation of each with the online cap-hit rate, then predict a held-out selector (R-KV) before running it. Caveat: about 20 cells is a small sample.

**Closest existing answer.** LessIsMore (attention recall as indicator), ThinKV (loop cause described qualitatively). None found that predicts non-termination offline.

**Overlap with dropped directions.** D1, directly. This is diagnosis of D1's evaluation, and it shares its anomaly with card 2.

---

## Observations dropped

- **Prefix-caching headroom of 2.6-5.5% on Mooncake.** Marconi (MLSys 2025, not ICLR as an earlier note said; https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html) reports "up to 34.4$\times$ higher token hit rates". The student already explained the gap (94-97% of hits extend an earlier prompt). Answered; D2.
- **Hybrid readmission ping-pong and the watermark.** The only external anchor is a vLLM PR. No accepted-paper claim; D3 / D4.
- **MTP halving hybrid concurrency; watermark gaining less with MTP.** No external claim found.
- **Pinned pages hold 76-78% of attention on Llama-3.2-3B against 44-45% on Qwen3-4B.** No external claim attached.
- **"Only 3.4% of bf16 weights change per step".** Already answered by arXiv 2602.03839 and arXiv 2610.41073. Kept only as context for card 4.
- **Hybrid traces about twice as long as dense.** Different model generations; not attributable.

## Not verified at primary source

- AReaL's NeurIPS 2025 venue (the arXiv page states none; the claim comes from the repo's research map).
- R-KV's NeurIPS 2025 listing (from the repo catalogue's OpenReview link; not reopened).
- CAOTE (ICLR 2026) and VATP: search results only.
- arXiv 2510.20787 and HyLRA arXiv 2602.00777: search results only.
- Mukherjee et al., NeurIPS 2025 (arXiv 2505.11711), "RL updates 5-30% of parameters": search results only; not used as evidence.
- MiniMax-M2 blog and Zhang et al. quotes in the RL research map: not reopened; not used.
- Whether ThinKV's code and the 72 hybrid checkpoints are usable.
- Web fetches return model-extracted quotes. Each quote above was returned as verbatim by the fetcher, but none was checked against the PDF.
