# Question harvest: unexplained anomalies reported in papers

Date: 2026-11-11. Lens: results that a paper reports but does not explain, results that two papers report with opposite signs, and anomalies in our own logs that a published claim does not predict.

How I searched:
1. Read the literature catalogue (`research_notes/Reproducible LLM systems paper setups/*.md`) for entries with a surprising number.
2. Opened each primary source (arXiv HTML) and looked for words such as "surprisingly", "contrary to expectations", "remarkably", and for tables whose numbers the text does not explain.
3. Read our own logs (`EXPERIMENTS.md`, `results/profile/NOTEBOOK.md`, `results/profile/findings.md`) for numbers that disagree with a published claim.
4. For each candidate, searched for a paper that already answers it. Three candidates changed shape after this step (see "Closest existing answer" in each card).

"Verified" below means I opened the primary source and saw the quote or number. "Local" means the number is from this repository.

Six cards follow. They are in the order I found them, not ranked.

---

## Card 1. KV eviction in reasoning models: accuracy change or length change?

**Question.** In reasoning models under KV-cache eviction, is the accuracy change (the losses, and R-KV's reported gain above full KV) carried by a change in reasoning length against the token cap, rather than by a change in the correctness of chains that finish?

**Tension type.** Two accepted papers state opposite things about length; one accepted paper reports a gain it does not explain; our own logs show the loss is almost all truncation.

**Evidence.**
- ThinKV (ICLR 2026 Oral, arXiv 2510.01290), verified: "Unlike quantization, eviction does not cause an increase in generation length". The paper gives no per-baseline length numbers for eviction. It caps generation at 32K tokens. https://arxiv.org/html/2510.01290
- Rethinking Key-Value Cache Compression Techniques for LLM Serving (MLSys 2025, arXiv 2503.24000), verified: "compression algorithms tend to increase the response length." On 200 ShareGPT requests that compression lengthened: H2O-512 1.55x, StreamingLLM-512 1.76x (eviction), KIVI-4 1.69x, GEAR-4 1.70x (quantization). The authors' explanation is a hypothesis only: "The lengthy responses may serve as an implicit way for KV cache compression to compensate for accuracy loss." https://arxiv.org/html/2503.24000
- R-KV (NeurIPS 2025 poster, arXiv 2505.24133v4), verified: "Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache." Table 2: R1-Llama on AIME24 52.29 at budget 2,048 vs 49.79 full KV; 64 responses per question; no confidence intervals; no explanation in the paper. https://arxiv.org/html/2505.24133
- Quantized Reasoning Models Think They Need to Think Longer, but They Do Not (arXiv 2606.00206v1), verified: 3-bit AWQ raises mean chain length of DeepSeek-R1-Distill-Qwen-1.5B on MATH-500 from 5.2K to 23.4K tokens; "in up to 52% of the quantized models' failures, models reach the right answer in intermediate reasoning steps." Covers weight quantization and KV quantization only, not eviction. https://arxiv.org/html/2606.00206v1
- Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference (arXiv 2506.09501v2), verified: "can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size". This is the noise floor any length or accuracy comparison has to clear. https://arxiv.org/html/2506.09501
- Local, `EXPERIMENTS.md` E3 and H3: on Qwen3-4B (16K cap) Quest@512 truncates 70% of samples vs 41% for full attention, and accuracy on finished samples is 73-80% for every non-SnapKV run. On Qwen3.5-4B (32K cap) 43.8% of full-attention samples hit the cap vs 63.3% (ours@512) and 75.4% (Quest@512); accuracy among finished samples is 78-92% for every method (full 88%).

**Why unknown.** No eviction paper reports the share of samples that hit the cap, accuracy among finished samples, or tokens until the first correct answer. ThinKV asserts no lengthening without numbers. The MLSys'25 result is on chat models with a 1,024-token cap. arXiv 2606.00206 explains lengthening for quantization only. Our own logs answer "yes" for Quest and SnapKV on two 4B Qwen models but not for the published methods at their published settings, and say nothing about the gain side.

**If yes.** Accuracy at a fixed cap is measuring finish rate. Eviction methods would need to be compared on tokens-to-answer and finish rate, and the memory saving has to be netted against the extra tokens. It also points to a mechanism to test: eviction perturbs high-entropy positions and raises the rate of "wait / but / alternatively" markers, as 2606.00206 found for quantization. R-KV's gain would then be the same effect in reverse (shorter chains, fewer capped samples).

**If no.** The loss sits in the content of finished chains, so selection quality matters, which is the path where we already measured small margins (D1). R-KV's gain would need a different explanation. This answer is less new, but it would show the truncation story is specific to Qwen3-family models.

**First experiment.** R-KV's public code and its two DeepSeek-R1-distill models on AIME24 and MATH-500; full KV vs R-KV at budgets 2,048 and 4,096 vs H2O and SnapKV at the same budgets; 8 or more samples per question; batch size and GPU fixed across arms. Measure per arm: share of samples at the cap, accuracy among finished samples, token index where the correct answer first appears, rate of overthinking markers. The number: the share of the accuracy difference from full KV that the finish-rate term explains in `delta_acc = delta_finish x acc_finished + finish x delta_acc_finished`. Then repeat with the cap doubled.

**Who cares.** Authors of KV compression methods for reasoning models; people who serve reasoning models with a token cap; benchmark designers.

**Closest existing answer.** arXiv 2606.00206 (quantization only). MLSys'25 (chat models, hypothesis only). ThinKV asserts the opposite for eviction. LazyEviction (arXiv 2506.15969) appears from search to claim unchanged output length; I did not open it.

**Overlap with dropped directions.** D1. Same methods and benchmarks, and the anomaly comes from the D1 runs. The question is why accuracy moves, not a better selector.

---

## Card 2. Do agent cost findings survive a change of cost unit?

**Question.** Do agent cost findings measured in API dollars (turning reasoning off costs more per task; compaction cuts cost; input tokens drive cost) keep their sign when cost is measured in GPU-seconds on a self-hosted engine with prefix caching?

**Tension type.** A conclusion drawn under one cost model, and measurements under another cost model that imply a different weighting of the same tokens.

**Evidence.**
- What Effort Level Should I Choose? On the Economics of Reasoning Effort in LLM Agents (arXiv 2610.34491, preprint), verified: on always-solved tasks GPT-5.6-luna averages 21.5 model calls at effort "none" vs 5.4 at "low", and cost at "none" is 4.58x cost at "low", driven mainly by input tokens. "Spending less on reasoning within each call can thus be offset by spending more on the trajectory as a whole." Cost is API dollars only; one run per task per effort level. https://arxiv.org/html/2610.34491
- Agentic AI Workload Characterization (IISWC 2026 per artifact appendix, arXiv 2605.26297), verified: "On ADE, Gemma Thinking averages only 18.0±15.3 turns, while Gemma Instant averages 108.8±178.7 turns." Prefill "remains below 10% of total request time" even under thrashing. No success rates by mode and no per-task cost. https://arxiv.org/html/2605.26297
- Agentic Coding in the Wild: Characterizing GitHub Copilot at Production Scale (arXiv 2608.00101v1, preprint), verified: "the cache hit rate on the first call after compaction results in a median drop of 66.1%."; compacting sessions are 7.8% of sessions and 44.2% of tokens. The paper has no task-quality data. https://arxiv.org/html/2608.00101
- The Complexity Trap (arXiv 2508.21433v3), verified: LLM summarisation gives about 15% longer trajectories; "context summaries act as a reinforcing signal, encouraging the agent to keep going." https://arxiv.org/html/2508.21433
- How Do AI Agents Spend Your Money? (arXiv 2604.22750v2), verified: "runs on the same task can differ by up to 30× in total tokens." The claim that input tokens drive cost even with caching is from a search summary, not verified verbatim. https://arxiv.org/html/2604.22750v2
- CliffCompaction (arXiv 2609.26779), abstract page only, not verified in the body: "reduces cost by up to 50% under a bounded context". https://arxiv.org/abs/2609.26779

**Why unknown.** No paper prices the same trajectories both ways. Sun et al. use a price sheet. The IISWC paper measures time shares but reports neither per-task GPU time by mode nor success. The Copilot paper has cache numbers without quality. On a GPU, a cached input token costs almost nothing and a thinking token is a decode step, which is the reverse of the weighting that makes input tokens dominate in dollars.

**If yes.** The API price sheet is a faithful proxy for hardware cost, and agent-level policies (keep reasoning on, compact) transfer to self-hosted serving. This answer is the less interesting one, but it removes a doubt that the measurement papers leave open.

**If no.** The sign of these findings is set by the price sheet. Reasoning-off could be cheaper in GPU-seconds while being 4x dearer in dollars; compaction could cost more because it invalidates the cache and adds decode. Self-hosted policies should then differ from API-side ones, and the cache discount is mispriced relative to hardware.

**First experiment.** A 27B-class open model on vLLM with prefix caching, mini-SWE-agent on a 50-task SWE-bench Verified subset, thinking on vs off, 3 runs each. Log per request: prompt tokens, cached tokens, output tokens, prefill time, decode time. The number: the cost ratio (thinking off / thinking on) per attempted task and per solved task, computed (a) with a public price sheet including the cache discount and (b) from measured GPU-seconds at fixed concurrency. The finding is whether the two ratios fall on opposite sides of 1. Add a compaction on/off arm if time allows. Limit: open models at this size solve few SWE-bench tasks, so per-solved-task cost will be noisy.

**Who cares.** Agent framework authors; people who self-host agents; authors of agent cost benchmarks.

**Closest existing answer.** Sun et al. (dollars only). IISWC'26 (time shares; no per-mode cost or success). "The Reasoning Tax" (arXiv 2608.26235) was read at abstract level only.

**Overlap with dropped directions.** Light contact with D2: the prefix-cache hit rate is an input to the accounting. The question is not about a cache policy.

---

## Card 3. Is context growth a cause of agent failure or its by-product?

**Question.** On coding agents with open models, does adding the volume of error output that failed runs accumulate to otherwise identical trajectories lower the solve rate?

**Tension type.** A correlation with a stated but untested mechanism; an exception inside the same paper; interventions in other papers with opposite signs.

**Evidence.**
- Agentic AI Workload Characterization (IISWC 2026 per artifact appendix, arXiv 2605.26297), verified: "Failed agents often take more turns and also carry more accumulated context." Terminal-Bench, Qwen Thinking: failed 79.2K vs successful 44.4K tokens. Proposed mechanism: "failed Bash, Edit, or Read actions append error messages, stack traces, failed patches." Exception in the same paper: Gemma Instant on SWE-bench Pro, successful 82.2K vs failed 72.3K. https://arxiv.org/html/2605.26297
- The Complexity Trap (arXiv 2508.21433v3), verified: solve rate raw / masking / LLM-summary on SWE-bench Verified: Gemini 2.5 Flash thinking 40.4 / 36.4 / 31.4 (removing context hurts) but Qwen3-32B thinking 23.0 / 24.6 / 24.8 and Gemini 2.5 Flash 32.8 / 35.6 / 36.0 (removing context helps or is neutral). https://arxiv.org/html/2508.21433
- Causal Decomposition of LLM Agent Failure via Oracle-State Intervention (NeurIPS 2026 workshop poster), verified on the NeurIPS page: across 420 failed WebArena trajectories, replacing the context after failure recovers under 10%; annotators attribute 11.7% of failures to context. https://neurips.cc/virtual/2026/172976
- How Do AI Agents Spend Your Money? (arXiv 2604.22750v2), verified: "higher token usage does not translate into higher accuracy; instead, accuracy often peaks at intermediate cost and saturates at higher costs." https://arxiv.org/html/2604.22750v2

**Why unknown.** The IISWC result is observational. Jia et al. repair context after the failure, on web tasks; that is the reverse direction and a different domain. The Complexity Trap removes context and finds signs that differ by model with no account of why. No one has added context in a controlled dose to a coding agent and measured the solve rate.

**If yes.** Tool-error hygiene has a causal payoff and a dose-response curve; context growth can be used as an early failure signal with a causal reason behind it.

**If no.** Context length is a symptom. Context management then buys cost, not accuracy, which would explain Jia et al.'s under-10% recovery. Growth still predicts failure but trimming does not rescue it.

**First experiment.** One open coding model (27B to 32B), mini-SWE-agent, 100 SWE-bench Verified tasks, three arms: raw; padded (after each tool call, append stderr text taken from other tasks' failed runs until the per-run total matches the failed-minus-successful gap, about 20K to 35K tokens); masked (failed tool outputs truncated to 200 tokens). The number: solve-rate difference between arms with a paired bootstrap interval, and the turn count per arm. Limit: with solve rates near 17-25% for this model size (Complexity Trap), 100 tasks resolve only differences of roughly 8 points or more.

**Who cares.** Agent harness authors; agent-serving schedulers that want a failure signal; context-compaction authors.

**Closest existing answer.** Jia et al. (reverse direction, WebArena). Complexity Trap (removal, mixed signs). Long-context degradation on single-shot QA is a separate literature that I did not search exhaustively.

**Overlap with dropped directions.** None.

---

## Card 4. Are the attention layers that hybrids keep less sparse?

**Question.** Do the few full-attention layers kept in hybrid linear-attention models have systematically more diffuse attention than the layers of a comparable dense model?

**Tension type.** A measured anomaly in our logs that contradicts the working assumption behind sparse-attention and KV-selection methods; no published measurement either way.

**Evidence.**
- Local, `EXPERIMENTS.md` H1: "the oracle shared set at 512 misses 22.7% of attention on Qwen3.5-4B vs 11.6% on Qwen3-4B, and each of Qwen3.5's 8 attention layers misses more than its Qwen3-4B counterpart (18-34% vs 5-20%). Attention in the hybrid's attention layers is less concentrated, not more." The ranking of selectors also flips between the two models.
- Jet-Nemotron (arXiv 2508.15884v3), verified: "In the pre-trained full-attention model, not all attention layers contribute equally." and "Layers that are critical for MMLU accuracy are not necessarily important for retrieval tasks." The paper chooses which layers to keep by task accuracy and does not report what the kept layers attend to. https://arxiv.org/html/2508.15884
- MagicPIG (arXiv 2410.16179v4), verified on the abstract page: "TopK attention itself suffers from quality degradation in certain downstream tasks" because attention is "not always as sparse as expected." This is for dense models. https://arxiv.org/abs/2410.16179

**Why unknown.** Our measurement is one model pair on one kind of text (AIME reasoning traces), and Qwen3 and Qwen3.5 differ in training data as well as architecture. I found no published per-layer sparsity analysis of a hybrid model.

**If yes.** Methods that select a subset of KV entries have a lower ceiling on hybrids than on dense models, because the recurrent layers already absorb the concentrated lookups and leave the diffuse ones to attention. This gives a division-of-labour account that can be tested by ablation, and it bears on which layers a hybrid should keep and how much budget each needs.

**If no.** The effect is specific to Qwen3.5 or to reasoning traces, and KV selection on hybrids is not ruled out in general. This answer is the less interesting one.

**First experiment.** Qwen3.5-4B vs Qwen3-4B, plus Jet-Nemotron-2B vs the dense model it was derived from, on three kinds of text (AIME reasoning traces, long-document QA, code) at 4K to 32K context. The number: per layer, attention mass outside the per-step oracle top-512 set, and attention entropy, hybrid vs dense. One causal probe: reset the recurrent state every N tokens and see whether the attention layers become more concentrated.

**Who cares.** Sparse-attention and KV-selection authors; designers of hybrid architectures.

**Closest existing answer.** None found after searching for attention-sparsity analyses of hybrid or linear-attention models. Jet-Nemotron and MagicPIG are adjacent. HySparse appeared in search and was not opened.

**Overlap with dropped directions.** D1. The anomaly comes from the D1 measurements, and a "yes" would explain why the D1 margin was small on the hybrid. It is a measurement question, not a selector.

---

## Card 5. Speculative decoding at batch 16 and above: target or drafter?

**Question.** Is the published result that speculative decoding stops paying at batch size 16 and above on short contexts a property of the target model's verification cost, or of the drafter's acceptance rate and cost?

**Tension type.** A published rule that did not reproduce in our logs for one configuration, with two confounded explanations.

**Evidence.**
- TurboSpec / SmartSpec (arXiv 2406.14066v3, preprint), verified: "batch sizes ≥16 render speculative decoding computationally inefficient"; "even minimal speculation (k=1) degrades performance by 14% (from 13.7s to 15.9s)." (L40S, Vicuna-7B, QPS 16). https://arxiv.org/html/2406.14066
- MagicDec (arXiv 2408.11049v5), verified: Table 3, LLaMA-3.1-8B on A100, prompt length 4,000: speedup 0.90 (batch 32), 0.81 (batch 64), 0.76 (batch 128); at 32,000: 1.22 / 1.28 / 1.35. "when the sequence length exceeds 4000, speculative decoding achieves speedup, which increases with batch size." https://arxiv.org/html/2408.11049
- MoESD (NeurIPS 2025 poster, arXiv 2505.19645v4), verified: "under a moderate batch size, SD can be more effective for MoEs than for dense models". This shows the crossover depends on the target architecture. https://arxiv.org/html/2505.19645v4
- Local, `results/profile/NOTEBOOK.md` lines 327-353 (90 AIME prompts, 1,024 output tokens, greedy, ignore_eos): speedup at batch 1 / 16 / 64 for Qwen3.5-4B with its native MTP head: k=1 1.55 / 1.51 / 1.19; k=2 2.02 / 1.74 / 1.22; k=3 2.31 / 1.86 / 1.18. For Qwen3-4B with a community EAGLE3 drafter: k=1 1.51 / 1.28 / 0.82; k=2 1.82 / 1.37 / 0.76; k=3 1.85 / 1.28 / 0.69. Acceptance: hybrid 0.80-0.97, dense 0.33-0.65.

**Why unknown.** Our comparison changes the target (hybrid vs dense) and the drafter (native MTP vs external EAGLE3) together, and ignore_eos inflates acceptance. The published analyses use external drafters on dense targets. MagicDec's account (speculation pays when reading the KV cache dominates) predicts that a hybrid, which has less KV to read, should gain less, which is the opposite of what we measured unless the drafter explains it.

**If the drafter.** The batch-16 rule is not fundamental. The crossover batch size would be predictable from acceptance and draft cost alone, and jointly trained heads move it.

**If the target.** Hybrids verify more cheaply at batch, which would be a new property of the architecture and would contradict the KV-bottleneck account for short contexts.

**First experiment.** Qwen3.5-4B with its MTP head, Qwen3.5-4B with an external drafter (a smaller model from the same family), and Qwen3-4B with EAGLE3; natural EOS; batch 1, 16, 64; context 1K and 8K. Measure accepted tokens per verify step, draft time share, and verify-step time over plain decode-step time. The number: the batch size at which speedup crosses 1.0 in each cell, and the gap between measured speedup and the speedup predicted from acceptance and the two time ratios. Limit: I know of no dense model of this size with a native MTP head, so the fourth cell of the 2x2 is missing and the answer may stay partial.

**Who cares.** vLLM speculative-decoding maintainers; people serving reasoning models at batch.

**Closest existing answer.** TurboSpec, MagicDec and MoESD model the crossover for external drafters. None covers a native MTP head or a hybrid target. Upstream vLLM has several open PRs on speculative decoding for GDN models, so the engineering side is crowded.

**Overlap with dropped directions.** D3. The anomaly comes from the D3 profiling, which found no bottleneck here. The question is why a published rule did not hold.

---

## Card 6. What causes the agent context-thrashing cliff?

**Question.** Is the prefix-cache cliff reported for concurrent agents (hit rate from about 98% to 17-27%) caused by preemption and readmission of running requests, rather than by eviction of idle sessions' prefixes during tool waits?

**Tension type.** An unexplained step change in a measurement paper; a systems paper that asserts one mechanism without counting it; our logs show a second mechanism that produces the same symptom and is removed by one engine flag.

**Evidence.**
- Agentic AI Workload Characterization (IISWC 2026 per artifact appendix, arXiv 2605.26297), verified: Table II, Qwen3.6-27B. Terminal-Bench 2.0 at 15 / 20 / 25 jobs: hit rate 98.85 / 16.78 / 19.21%; throughput 30.08 / 16.96 / 10.30 tok/s per request; prefill 4.42 / 8.38 / 7.81% of request time. DABStep at 5 / 10 / 15 jobs: hit rate 98.53 / 25.17 / 26.60%. The text says only that "thrashing repeatedly evicts useful cached prefixes from GPU memory". It does not say which blocks are evicted, does not use the words preemption or watermark, and gives no engine flags beyond vLLM v0.20.0 with TP=2. https://arxiv.org/html/2605.26297
- ThunderAgent (arXiv 2602.13692, preprint), verified: thrashing is described as "KV cache to be frequently evicted during tool execution to accommodate newly arriving requests" and is said to raise end-to-end latency "by up to 7.14×". The paper reports no preemption counts, no share of recomputed tokens in the text, and no watermark setting or vLLM version for the baseline. https://arxiv.org/html/2602.13692
- Local (project notes and `results/profile/NOTEBOOK.md`): in saturated closed-loop serving of hybrid models, with no tool waits, the scheduler preempts requests that were just readmitted. Setting `--watermark 0.1` cut preemptions from 3,474 to 128 and raised throughput 35% (4B), and from 1,006 to 46 with +28% (27B). The vLLM default of 0 cost the hybrid 26-37% and the dense model 5-7.6% against a perfect-packing bound.

**Why unknown.** Neither paper counts how many lost cache hits come from each mechanism. The IISWC table itself is hard to read under the idle-eviction account alone: prefill stays under 10% of request time while per-request throughput falls 44-66%. Part of that fall is ordinary batching at higher concurrency, and the paper does not separate the two. Our measurement has no tool waits, so it shows the second mechanism exists but not how large it is for agents. I could not confirm from the paper whether Qwen3.6-27B is a hybrid.

**If yes.** Part of the published cliff, and part of the baseline gap that program-aware schedulers report, comes from an engine default. The first thing to report would be how much of the cliff one flag removes.

**If no.** Idle-prefix eviction dominates, which confirms the premise of ThunderAgent and Continuum and shows the watermark does not matter for agent workloads. This answer is the less new one.

**First experiment.** One hybrid and one dense model of about 27-30B on vLLM; replay agent trajectories with recorded tool-wait times (the IISWC artifact, or mini-SWE-agent runs); sweep concurrency across the cliff with watermark 0 and 0.1; use the existing scheduler instrumentation (`thrash_trace.py`) to tag each recomputed prompt token by cause (blocks freed by preempting a running request vs blocks of an idle session evicted). The number: the share of recomputed tokens from each cause, and the hit rate at the cliff concurrency with watermark 0.1.

**Who cares.** Authors of agent-serving schedulers; vLLM maintainers; people who run agent fleets or RL rollouts on a fixed GPU budget.

**Closest existing answer.** ThunderAgent and Continuum (arXiv 2511.02230) assume idle eviction and build for it. Our R1 result covers batch rollouts without tool waits. No paper found that separates the two causes.

**Overlap with dropped directions.** Heavy. D4: the watermark result is the D4 finding, and D4 was dropped because the watermark alone came within 7% of a perfect schedule. D2: prefix caching. D3: it reuses the profiling tools. Our own evidence already leans towards "yes" in the setting without tool waits.

---

## Candidates considered and not carded

- Numerical noise in bf16 (arXiv 2506.09501): is the batch-size effect on reasoning length biased or zero-mean? The likely answer is zero-mean, and neither outcome points to a design.
- Shorter chains are more accurate (arXiv 2505.17813): is this causal? Already well covered by that paper's finetuning result and by 2606.00206.
- Predicting agent cost early in a run: BAGEN (arXiv 2606.00198), EarlyEval (2609.02783) and StopLoss (2610.25525) appear to cover it (search level only).
- Why "thinking" reduces turns for Gemma but the paper reports no success rates: folded into Card 2.
- Offline KL advantage of the hybrid does not survive long generation (`EXPERIMENTS.md` H-b): folded into Card 1, since both models fail by not finishing.

## Sources opened (primary)

arXiv 2605.26297, 2503.24000, 2504.04823, 2506.09501, 2505.24133, 2510.01290, 2606.00206, 2505.17813, 2608.00101, 2506.04301, 2508.21433, 2604.22750, 2610.34491, 2406.14066, 2408.11049, 2505.19645, 2508.15884, 2602.13692; abstract pages of 2410.16179, 2609.26779, 2609.20804, 2608.26235; neurips.cc/virtual/2026/172976.

## Not verified

- Venues: ThinKV as ICLR 2026 Oral, R-KV as NeurIPS 2025 poster and MoESD as NeurIPS 2025 poster are taken from the catalogue or a search listing; I did not open the venue pages. The IISWC 2026 label comes from the paper's artifact appendix; the paper text names no venue. MagicDec's ICLR 2025 venue was not rechecked. A NeurIPS 2026 oral listing for arXiv 2606.00206 and a NeurIPS 2026 listing for arXiv 2604.22750 appeared in search and were not opened.
- Jia et al.: the ACL Anthology page (aclanthology.org/2026.realm-1.6) refused the connection; numbers are from the NeurIPS virtual page.
- R-KV: other versions of the paper give 38% / 33% of the cache rather than 16% for the above-full-KV result. I read v4.
- MLSys'25 Table 5: the row labels for "longer" and "shorter" look inverted in the paper, so I did not use that table's direction.
- Long pages: only the first 100,000 characters were read for ThinKV (appendices E and F unread), ThunderAgent (appendices C to G unread) and the MLSys'25 paper (last ~4,000 characters unread).
- CliffCompaction, the harness-design study (2609.20804), The Reasoning Tax (2608.26235): abstract only. HySparse, BAGEN, EarlyEval, StopLoss, LazyEviction, Fail-Fast Restart-Smart (2608.03222), "Failure as a Process" (2607.09510): search level only.
- Whether Qwen3.6-27B is a hybrid model: not stated in the IISWC paper; not checked elsewhere.
- Which base models Jet-Nemotron-2B was derived from: not rechecked for Card 4's experiment.
