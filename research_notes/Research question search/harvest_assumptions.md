# Question harvest: shared, untested assumptions in evaluation setups

Date: 2026-11-11. Lens: assumptions that many accepted papers' evaluation setups share and that nobody has tested, or that one study has shown to be false.

How to read "verified": a source marked verified was opened in this session and the quote or number was returned from that page by the fetch tool. The fetch tool summarises pages with a small model, so I did not see the raw text myself. Re-check every quote against the PDF before it goes into a paper. Items marked unverified come from the existing catalogue or from search snippets only.

Six cards follow, in the order I found them (not ranked). After them: candidate questions I checked and dropped because a paper already answers them, then the source list.

---

## Card 1. Open-loop arrivals versus closed-loop agent sessions

**Question.** How much of the latency gain that length-aware and program-aware schedulers (SJF by predicted length, MLFQ, Autellix-style program priority) show under open-loop Poisson arrivals survives when the same engine is loaded by closed-loop agent sessions, where each next request is issued only after the previous response and tool call return?

**Assumption and who makes it.**
- Sarathi-Serve (OSDI'24): "The request arrival times are generated using Poisson distribution." (verified, https://arxiv.org/html/2403.02310)
- DistServe (OSDI'24): arrivals are generated "using Poisson distribution with different request rates" (verified, https://arxiv.org/html/2401.09670)
- Autellix / Agentix (NSDI'26): whole programs arrive by a "Poisson process λ"; programs have 6.66 (ShareGPT), 10.75 (BFCL) and 159.7 (LATS) LLM calls on average; "a noticable gap remains between Autellix and SRPT" (verified, https://arxiv.org/html/2502.13965)
- Learning-to-Rank (NeurIPS'24) and PARS (arXiv 2510.03243) use fixed arrival rates and a burst of 2,000 simultaneous requests (verified, https://arxiv.org/html/2408.15792, https://arxiv.org/html/2510.03243)

**Evidence that doubts it.**
- Schroeder, Wierman, Harchol-Balter (NSDI'06), principle (iv): "While open systems benefit significantly from scheduling with respect to response time, closed systems improve much less." Principle (vii): "A partly-open system behaves similarly to an open system when the expected number of requests per session is small" and "similarly to a closed system when the expected number of requests per session is large" (rule-of-thumb cutoffs about 5 and 10 requests per session). (verified, https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder_html/)
- GitHub Copilot production study (arXiv 2608.00101): "Most LLM calls result in a tool action, and most tool results immediately trigger another LLM call."; "The median session contains just 3 user turns, 15 LLM calls, and lasts 4.2 minutes."; P90 of 100.5 LLM calls per session. (verified, https://arxiv.org/html/2608.00101)
- TraceLab (arXiv 2606.30560): sessions average about 8.9 user-initiated steps plus 73.6 tool-initiated steps; tool executions have median 0.3 s. (verified, https://arxiv.org/html/2606.30560v1)
- By the Schroeder cutoffs, a session of 15 to 100 calls sits on the closed side. No LLM scheduling paper I found reports its gain under both loop types.

**Why unknown.** The Schroeder result is for classic single-server queues. An LLM engine with continuous batching runs all admitted requests at once and queues only when KV memory or the batch limit is hit, so neither the open nor the closed prediction transfers directly. Autellix reports large gains in a partly-open setup with 160 calls per program, which the classic result would not predict.

**If the gain mostly disappears.** Scheduling order is a second-order knob for agent and rollout serving; effort moves to admission and memory control (which is what ThunderAgent and Continuum change). Published speedups measured under Poisson need a loop-type caveat. A benchmark-method paper follows.
**If the gain survives.** Then something specific to batched LLM engines (memory-driven preemption, cache thrashing) makes ordering matter even in closed systems, and that mechanism is worth isolating; it would say where program-aware scheduling earns its gain.

**First experiment (about one week, 1 GPU).** vLLM, Qwen3-8B or Qwen3.5-9B on one 48 GB card. Policies: FCFS, oracle SJF per call, oracle shortest-remaining-program. Loads: (a) open-loop Poisson per call at 70-95% of capacity; (b) closed loop with N sessions and tool times drawn from the Copilot dataset; (c) partly-open: Poisson session arrivals with k calls per session, k in {1, 5, 15, 100}. Session shapes (calls per session, token counts, tool durations) from the public Copilot trace. Number to measure: ratio of mean and P99 session completion time, FCFS over SJF, as a function of loop type and k; report the k at which the gain halves.

**Closest existing answer.** Schroeder et al. NSDI'06 (not LLM engines). "Human-less LLM Serving" (arXiv 2606.20577) uses a closed-loop agent workload but varies chunked-prefill and SLO knobs; it does not test SJF or any ordering policy. None found that compares scheduler gains across loop types for LLM engines.

**Overlap with dropped directions.** D4: RL rollouts are a closed-loop workload. The D4 result (watermark alone within 7% of a perfect schedule) is one data point consistent with the closed-system prediction; this card asks whether that is general.

---

## Card 2. Scheduler rankings under production-like open-loop workloads

**Question.** Does the ranking among published engine-level schedulers (FCFS, chunked-prefill Sarathi, MLFQ, learning-to-rank SJF) stay the same when ShareGPT lengths plus Poisson arrivals are replaced by ServeGen-generated workloads with per-client burstiness and rate-length correlation, and if it changes, which workload property causes the change?

**Assumption and who makes it.** Same Poisson quotes as Card 1 (Sarathi-Serve, DistServe). Sarathi-Serve's two length sets are openchat_sharegpt4 (median prompt 1730, output 415 tokens) and arxiv_summarization (7059 / 208). (verified, https://arxiv.org/html/2403.02310)

**Evidence that doubts it.**
- ServeGen (NSDI'26): on a 10-minute, 30,000-request period, "ServeGen results in provisioning 25 instances (4% over the actual number needed)" while "Naive results in only 12 instances (50% under-provisioning)". For prefill/decode splits: "Results may disagree about the best PD configuration under Naive and ServeGen workloads." "Out of 2,412 clients, the top 29 clients" make 90% of requests. (verified, https://arxiv.org/html/2505.09999)
- ServeGen did not evaluate schedulers (verified by the same read).
- "Beyond Prediction: Tail-Aware Scheduling for LLM Inference" (ICML'26, arXiv 2606.18431) says prediction-driven policies "can be fragile under distribution shifts, bursty arrivals, and GPU memory pressure" (abstract only read; workloads not confirmed).

**Why unknown.** ServeGen showed that a configuration ranking flips (PD splits) but nobody has rerun the scheduler comparisons.

**If rankings change.** Several published scheduler conclusions need to be restated per workload; the property that drives the flip (burst length, client skew, rate-length correlation) becomes a design input.
**If rankings hold.** Reassuring and somewhat boring: Poisson plus ShareGPT is adequate for comparing schedulers even though it is wrong for provisioning. Still worth one paragraph in any later paper.

**First experiment.** vLLM, Llama-3.1-8B or Qwen3-8B on one 48 GB card. Schedulers: FCFS, Sarathi-style token budget (vLLM chunked prefill at two budgets), priority by oracle length, the public vllm-ltr predictor. Workloads: ShareGPT + Poisson versus ServeGen (public generator) at matched mean rate and matched mean lengths; then ablate ServeGen features one at a time (burstiness only, client skew only, rate-length correlation only). Number: capacity at P99 TBT and P99 TTFT SLO for each scheduler, and Kendall's tau between the two scheduler rankings.

**Closest existing answer.** ServeGen (provisioning and PD configuration only). None found for schedulers.

**Overlap.** None directly; it reuses the D3/D4 measurement harness.

---

## Card 3. How much of reasoning-model output length is fixed by the prompt

**Question.** For reasoning models with long thinking, how much of the output-length variation a scheduler faces is fixed by the prompt and how much is sampling or numerical noise; that is, what latency does a perfect prompt-level predictor (the per-prompt median length) achieve relative to FCFS and to the realized-length oracle that papers report as the upper bound?

**Assumption and who makes it.**
- Learning-to-Rank (NeurIPS'24): "predicting the exact generation length of each request is infeasible", but "just knowing the relative ordering between request lengths is sufficient for SJF/SRTF scheduling." Its Table 3 (Llama-3-70B, burst): ranking tau 0.54, 0.78 s/token versus oracle 0.46 s/token on ShareGPT. (verified, https://arxiv.org/html/2408.15792)
- DeepServe (ATC'25): "Accurately predicting decode lengths with minimal overhead remains a challenging research problem." (from the catalogue; not re-opened, unverified)
- All of these compare against an oracle that knows the one realized length, which treats the gap as predictor error.

**Evidence pulling in two directions.**
- Small noise: ProD (arXiv 2604.07931), 16 samples per prompt at temperature 0.8 on Qwen-2.5-7B and Llama-3-8B: median noise ratio "from 11.5% on Qwen/Math to 18.2% on Llama/LongSequence". No reasoning models, no scheduling experiment. (verified, https://arxiv.org/html/2604.07931)
- Small noise: PARS (arXiv 2510.03243), 200 Alpaca prompts, 10 runs, Llama 3.1 and DeepSeek-R1: "relative output length variance across repeated generations for the same prompt typically remained around 20%". (verified, https://arxiv.org/html/2510.03243)
- Large noise: "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference" (arXiv 2506.09501): under bfloat16 with greedy decoding, DeepSeek-R1-Distill-Qwen-7B shows "up to 9% variation in accuracy and 9,000 tokens difference in response length" from GPU count, GPU type and batch size alone. (verified, https://arxiv.org/abs/2506.09501)
- The two small-noise measurements are on short, easy prompts; the large one is on a reasoning model with no sampling at all.

**Why unknown.** Nobody has measured the repeated-sample length distribution of thinking models on hard problems (AIME-level, 10K+ thinking tokens) and turned it into a scheduling ceiling.

**If length is mostly prompt-determined.** The gap to oracle is closable; better predictors are worth building for reasoning traffic.
**If noise dominates.** Prompt-level prediction has a hard ceiling for reasoning workloads; the oracle baselines in scheduling papers overstate what is attainable; design shifts to signals available during generation (progress probes, remaining-length updates, preemption).

**First experiment.** Qwen3-8B thinking, DeepSeek-R1-Distill-Qwen-14B and Qwen3.5-9B on vLLM. Prompts: MATH-500, AIME 2024/2025, a LiveCodeBench subset, 500 LMSYS-Chat-1M prompts. 16 samples per prompt at temperature 0.6, max 32K tokens. Numbers: share of log-length variance that is between prompts (intraclass correlation), per-prompt max/median ratio, and in a replay (burst and Poisson) the fraction of the realized-oracle SJF gain that per-prompt-median SJF captures.

**Closest existing answer.** ProD (arXiv 2604.07931) and PARS (arXiv 2510.03243). Both measure same-prompt spread on non-reasoning or easy prompts; neither computes a scheduling ceiling; ProD lists serving integration as future work.

**Overlap.** None (shares models and benchmarks with D1, not the mechanism).

---

## Card 4. Prefill-decode interference at production cache-hit rates

**Question.** For agent workloads at production-observed cache behaviour (about 90% hits within a turn, about 55% across turns, plus compaction), is prefill-decode interference still a first-order limit on serving capacity, or does it shrink to a few percent, and which events (cross-turn misses, compaction, thrashing) carry whatever remains?

**Assumption and who makes it.**
- Sarathi-Serve: "eagerly scheduling prefills of requests C and D delays the decodes of already running requests A and B"; "a generation stall in vLLM can last over several seconds". Prefix caching is not mentioned. (verified, https://arxiv.org/html/2403.02310)
- DistServe: "colocation leads to strong prefill-decoding interference"; no prefix caching or multi-turn reuse. (verified, https://arxiv.org/html/2401.09670)
- Both evaluate every request as a full, uncached prefill.

**Evidence, and the two regimes that disagree.**
- IISWC'26 agent characterisation (arXiv 2605.26297): "decode accounts for 91.0–98.6% of LLM time, while prefill contributes only 1.4–9.0%" at cache-hit ratios of 84.6-99.5%; under thrashing the hit rate drops "from about 98% to 16–26%". (verified, https://arxiv.org/html/2605.26297)
- "Human-less LLM Serving" (arXiv 2606.20577): on a closed-loop agent workload at 64K context, chunked prefill with C=8192 costs 20.4% throughput (5130 vs 4260 tok/s), up to about 60% at C=512; prefix-cache hit rates were not modeled and the authors leave "how much survives under realistic prefix-cache hit rates" to future work (paraphrase of the fetch summary). (verified, https://arxiv.org/html/2606.20577v1)
- PPD (ICML'26, arXiv 2603.13358): append-prefill causes "an order-of-magnitude smaller decoding slowdown" than full prefill. (abstract read, https://arxiv.org/abs/2603.13358)
- Copilot production (arXiv 2608.00101): "KV cache hit rates averaging 90% within a turn, but falling to 55% across turn boundaries." (verified). The catalogue also records that 7.8% of sessions compact but hold 44.2% of tokens (not re-checked).

**Why unknown.** One paper measures no cache (large cost), one measures a near-perfect cache on a dedicated server (negligible cost). Production sits between, and nobody has measured the stall and capacity cost at that mix. Honest prior: the evidence leans toward "small within a turn"; the open part is whether cross-turn misses and compaction events are frequent and large enough to matter.

**If still first-order.** Chunked prefill and PD multiplexing remain relevant for agents, but the trigger is cache-miss events, so mechanisms should key on those.
**If a few percent.** The chunked-prefill and PD-disaggregation line addresses a chat-era problem; for agent serving the capacity question is on the decode and memory side.

**First experiment.** vLLM with Qwen3.6-27B at TP=2 on 2x 48 GB (or Qwen3.5-9B on one card). Replay sampled sessions from the public Copilot trace (it has prompt, cached-prompt and completion token counts, timestamps and tool durations) with synthetic token content that reproduces the prefix structure. Three cache regimes: perfect within-session reuse, production-like (TTL set to reproduce the cached-token share in the trace), none. Toggle the chunked-prefill token budget. Numbers: throughput at equal load with and without chunking, share of GPU time in prefill, P99 inter-token stall of co-running decodes.

**Closest existing answer.** PPD (ICML'26) and the IISWC'26 characterisation. Neither uses a production cache-miss mix, and neither reports decode stalls on a colocated engine.

**Overlap.** D2 (uses prefix caching) and D3 (profiling harness); it tests a premise, it does not build a cache policy.

---

## Card 5. Trace window length and cache-policy headroom

**Question.** Is the small headroom that cache-policy studies find on one-to-two-hour traces a property of the workload or of the trace window: does the value of a better prefix or state retention policy over LRU grow when the same policy is evaluated on multi-day session traces whose human-turn gaps fall inside the window?

**Assumption and who makes it.**
- Mooncake FAST'25 tool-and-agent trace: 23.6K requests, 1 hour (as listed in the comparison table of arXiv 2609.28870; verified there). LMetric (OSDI'26) evaluates on it (catalogue).
- Marconi (MLSys'25): average response time between requests in a session swept "from 5 s to 10 s"; session arrival 0.5 to 2 per second; no real timestamps. (verified, https://arxiv.org/html/2411.19379)

**Evidence that doubts it.**
- "When Fancy Eviction Fails" (arXiv 2609.28870): Qwen Bailian 2-hour windows have 34%-54% compulsory misses versus 6.0% for the 7-day FreeInference trace; Belady's oracle "still sits far above the best online algorithm"; LRU catches up at 37 GiB on FreeInference but 60-462 GiB on Qwen; the Mooncake 1-hour trace is not evaluated. (verified, https://arxiv.org/html/2609.28870)
- TraceLab (arXiv 2606.30560): raising the eviction timeout from 1 min to 1 h raises the achievable hit rate from 85.4% to 98.6%; human-input waits have median 1.4 min and p90 20.6 min. (verified)
- Copilot: "The median user idle time is 1,512 s (25.2 minutes), with a long tail extending beyond one day." (verified)
- A 1-hour window cannot contain most reuses that follow a 25-minute idle gap, and it starts cold.

**Why unknown.** No study truncates one long trace to several window lengths and reports policy headroom at each. For hybrid-model state checkpoints (Marconi's setting) there is no long-trace result at all.

**If headroom grows with window length.** Conclusions drawn from 1-hour traces, including the D2 result, understate policy value; retention across human idle gaps is where the gain is.
**If headroom stays small.** The D2 conclusion is robust; cache policy for these workloads is close to solved by recency at realistic capacities, as the eviction paper argues for large pools.

**First experiment (no GPU for the first pass).** Take the public 7-day FreeInference trace (Hugging Face harvardMadsys/2026_prefixcache; arrival times, session IDs, block IDs) and the public Copilot trace. Cut each into windows of 1 h, 2 h, 6 h, 1 day, 7 days. Run LRU, Belady and the existing D2 perfect-policy simulator at capacities that correspond to one and four 48 GB cards. Number: gap between LRU and the oracle in compute-saving ratio versus window length. Confirm one point end-to-end with Qwen3.5-9B on vLLM.

**Closest existing answer.** "When Fancy Eviction Fails" (arXiv 2609.28870, September 2026). For attention-only models it already shows headroom is capacity-dependent and that short windows have many compulsory misses. It does not do a window-length ablation on one trace and does not cover hybrid-model state. Part of this card is therefore already indicated by that paper.

**Overlap.** D2, directly. This is a re-test of the premise behind a dropped direction, not a new topic; the D2 number (2.6-5.5% on Mooncake) came from 1-hour traces.

---

## Card 6. Step-level speculative reasoning under concurrency

**Question.** Does step-level speculative reasoning (a small model drafts reasoning steps and the base model verifies them) keep its 1.4-3.0x latency gain when the engine serves many concurrent requests, given that token-level speculative decoding loses its gain as batch size grows, or does it behave differently because it removes base-model compute instead of adding verification compute?

**Assumption and who makes it.**
- SpecReason (NeurIPS'25): "We run our evaluations on two NVIDIA A6000-48GB GPUs."; "Inference is performed sequentially: the small and base models take turns, avoiding kernel-level interference."; KV memory "is statically partitioned between the two models"; no concurrency, batching or throughput analysis. (verified, https://arxiv.org/html/2504.07891)
- Spec-Bench and SWIFT run at batch size 1 (catalogue, unverified here).

**Evidence that doubts it.**
- "Speculative Decoding: Performance or Illusion?" (MLSys'26, arXiv 2601.11580), vLLM v0.10.1.1: "Increasing batch size improves absolute throughput but systematically reduces the relative speedup of speculative decoding."; EAGLE on GSM8K with Llama3.1-8B falls from 1.73x to 1.21x between batch 1 and 128; reasoning benchmarks were run only up to batch 16; SpecReason-style step speculation is not evaluated. (verified, https://arxiv.org/html/2601.11580)
- EAGLE-3 falls below the no-speculation baseline from batch 24 in SGLang; SPEX says larger batches are compute-bound, "reducing SPEX's relative benefits" (both from the catalogue, unverified here).

**Why unknown.** Token-level speculation spends extra base-model compute to save latency, so it must lose when compute-bound. Step-level speculation moves most decode tokens to a model about 20x smaller, which should help when compute-bound, but it splits KV memory between two models and adds a verification prefill per step. Which effect wins under load has not been measured.

**If the gain holds or grows.** Step-level speculation is a throughput technique, not only a latency one; serving it well becomes a two-model memory-partition and scheduling problem.
**If the gain shrinks.** The published speedups are single-request numbers; the mechanism that kills them (KV partition, verification prefill interference, rejection rework) says what a load-aware variant must control.

**First experiment.** Reproduce SpecReason with its public code on its own testbed class (2x A6000, vLLM): QwQ-32B plus a 1.5B reasoning model on MATH-500 and AIME. Closed-loop concurrency 1, 4, 16, 64. Numbers: per-request latency ratio versus base-only at each concurrency, problems per hour, accuracy, and GPU time split between small-model decode, base decode and verification.

**Closest existing answer.** "Speculative Decoding: Performance or Illusion?" (MLSys'26): token-level methods only, fixed batch, reasoning up to batch 16.

**Overlap.** None for the mechanism; reuses the D3 profiling harness.

---

## Checked and dropped (a paper already answers these)

- "Do reasoning KV-compression methods tuned on single math prompts survive multi-turn agent loops where the KV is the next turn's prefix cache?" Taken by IntentKV (arXiv 2606.09916: compaction baselines fall to 0-3% prefix-hit rate; IntentKV holds 20.7%) and CommitKV (arXiv 2608.07855). Both are preprints; neither measures engine throughput. ThinKV (ICLR'26) and R-KV (NeurIPS'25) evaluate no multi-turn or agent task (verified from their arXiv HTML).
- "Are reported accuracy deltas on AIME (30 problems) above the noise floor?" Answered by "A Sober Look at Progress in Language Model Reasoning" (arXiv 2504.07086, search summary only) and arXiv 2506.09501.
- "Does token-level speculative decoding survive batching?" Answered by MLSys'26 arXiv 2601.11580.
- "Do TTFT/TPOT-oriented mechanisms cost throughput for agent traffic that nobody watches?" Largely answered by "Human-less LLM Serving" (arXiv 2606.20577); the residual is Card 4.
- "Does KV compression throughput exist only at batch sizes that violate any per-stream speed target?" ThinKV at iso-batch 256 on A100 reports 5298.4 tok/s (about 21 tok/s per stream), so the gain is present at usable per-stream speed; dropped.

## Could not verify

- OpenReview pages (EGTP / ForeLen, forum 3loQDtveWI) are behind a bot check.
- DeepServe's quote, FastServe's Poisson setup, EAGLE-3 and SPEX batch numbers, and the Copilot compaction figure come from the existing catalogue and were not re-opened.
- "Beyond Prediction" (ICML'26) and PPD (ICML'26): abstract pages only; workloads and hardware not confirmed.
- "Queueing, Predictions, and Large Language Models: Challenges and Open Problems" (Stochastic Systems) was found by search and not opened; it may discuss closed systems.
- The D2 and D4 numbers (2.6-5.5%; within 7%) are the student's own results as stated in the task, not re-derived here.
- XPerf (arXiv 2608.20370): abstract only; whether it supports closed-loop replay was not confirmed.

## Sources opened in this session

- Sarathi-Serve https://arxiv.org/html/2403.02310
- DistServe https://arxiv.org/html/2401.09670
- Learning to Rank https://arxiv.org/html/2408.15792
- Autellix https://arxiv.org/html/2502.13965
- ServeGen https://arxiv.org/html/2505.09999
- Schroeder et al. NSDI'06 https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder_html/
- Copilot production study https://arxiv.org/html/2608.00101 ; dataset https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md
- IISWC'26 agent workloads https://arxiv.org/html/2605.26297
- Human-less LLM Serving https://arxiv.org/html/2606.20577v1
- TraceLab https://arxiv.org/html/2606.30560v1
- When Fancy Eviction Fails https://arxiv.org/html/2609.28870 ; dataset https://huggingface.co/datasets/harvardMadsys/2026_prefixcache
- Marconi https://arxiv.org/html/2411.19379
- ProD https://arxiv.org/html/2604.07931
- PARS https://arxiv.org/html/2510.03243
- Numerical nondeterminism https://arxiv.org/abs/2506.09501
- SpecReason https://arxiv.org/html/2504.07891
- Speculative Decoding: Performance or Illusion? https://arxiv.org/html/2601.11580
- ThinKV https://arxiv.org/html/2510.01290 ; R-KV https://arxiv.org/html/2505.24133
- IntentKV https://arxiv.org/html/2606.09916v1
- PPD https://arxiv.org/abs/2603.13358 ; Beyond Prediction https://arxiv.org/abs/2606.18431 ; XPerf https://arxiv.org/abs/2608.20370 ; Floor-First Triage https://arxiv.org/abs/2607.05876 (abstract pages)
