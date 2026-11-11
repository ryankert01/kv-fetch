# Question harvest: contradictions between accepted papers

Date: 2026-11-11. Lens: pairs or groups of 2024-2026 papers whose claims conflict, plus "rethinking / revisiting" papers.
Each card is a question, not a system. Quotes marked [V] were extracted from the primary source (arXiv HTML or the USENIX PDF) downloaded locally; [U] means I only saw a search summary or a WebFetch paraphrase.

Hardware assumed for first experiments: up to 4x 48 GB cards (never GPU 0), vLLM 0.32, open models up to about 30B.

---

## Card 1. Speculative decoding and batch size: which way does the speedup go?

**Question.** On 48 GB cards in vLLM, does the speedup from speculative decoding rise or fall with batch size once context reaches several thousand tokens, and is the turning point set by context length, by the card's compute-to-bandwidth ratio, or by the kind of drafter?

**Tension.** Direct contradiction, with a proposed explaining variable that the later paper never tested.

**Evidence.**
- [V] SmartSpec (arXiv 2406.14066, 2024; venue not confirmed): "deploying SD in real online LLM serving systems (with continuous batching) does not always yield improvement – under higher request rates or low speculation accuracy, it paradoxically increases latency."
- [V] MagicDec (ICLR 2025, arXiv 2408.11049v5): "existing research ... has discouraged the use of speculative decoding to serve large batches of requests. In our work, we show that this claim only applies to short sequences." and "The speedup decreases with batch size for S<4000, but for S≥4000, the speedup increases with batch size." Also: "For a device with higher FLOPS-to-memory bandwidth ratio, we expect a lower S_inflection." and "GQA model like LLaMA-3.1-8B tends to have higher S_inflection". Table 1 row: Llama-3.1-8B, StreamingLLM drafter, PG-19, 8xL40, prefill 32000, batch 32: 1.21x. Self-built prototype, not vLLM.
- [V] "Speculative Decoding: Performance or Illusion?" (MLSys 2026 per mlsys.org listing, arXiv 2601.11580): "Prior studies only used prototype implementations rather than production-level systems, which greatly undermines the validity of their conclusions." and "The speedup decreases as the batch size increases" and "for the Llama3.1-8B model on the GSM8K dataset, the speedup of EAGLE over the non-SD baseline decreases from 1.73× to 1.21× as the batch size increases from 1 to 128." vLLM v0.10.1.1, H100 80 GB, max output 8K. A text search of the paper for "MagicDec" and "Sadhukhan" finds nothing, so the long-context regime is not tested or discussed.
- [V] MoESD (NeurIPS 2025 per search, arXiv 2505.19645v4): "Across different hardware platforms and MoE models, SD speedup first increases and then decreases" and "Experiments on different GPUs show up to 2.29x speedup for Qwen2-57B-A14B at medium batch sizes". Run in vLLM, prompts 5 to 391 tokens.
- [V, practitioner blog, not a paper] ReplaySSM (tridao.me, 2026): "vLLM's existing speculative decoding falls below standard-decoding throughput at serving batch sizes, while ReplaySSM speculative decoding delivers 1.87–1.96x."

**Why unknown.** The only production-engine study stops at 8K output on H100 and reports a monotone decline; the only long-context study is a prototype on L40/A100/H100 with a compressed-KV drafter that vLLM does not ship. Nobody reports the batch x context grid in vLLM, and nobody reports it on a card with the A6000's compute-to-bandwidth ratio. MagicDec's own theory predicts the turning point moves with that ratio, in a direction that could put it out of reach on 48 GB cards.

**If yes (speedup rises with batch at long context in vLLM).** Reasoning workloads (long outputs, so long self-context) are the regime where SD should be on by default, and the drafter's KV footprint becomes the design variable.
**If no.** MagicDec's result depends on its drafter or its prototype, and the "SD is for small batch" rule holds for production engines; the gap between prototype and engine is itself the finding.

**First experiment (about one week).** vLLM 0.32, one 48 GB card per run. Target Llama-3.1-8B (the model both papers share) and Qwen3-8B; drafters as shipped in vLLM (small draft model, n-gram, EAGLE head where available). Prompts: PG-19 continuations (MagicDec's data) truncated to 512 / 2K / 4K / 8K / 16K tokens, 256 output tokens, batch 1 / 4 / 16 / 32 / 64 where KV fits (Llama-3.1-8B needs about 0.13 MB of KV per token in bf16, so 8K x 32 is about 33 GB; my arithmetic). Measured number: tokens/s with SD divided by tokens/s without, per cell, and the sign of its slope in batch size at each context length.

**Who cares.** vLLM maintainers and anyone serving reasoning models; authors of both papers.
**Closest existing answer.** "Performance or Illusion?" measures vLLM on H100 up to 8K output and batch 128 but does not vary prompt length into MagicDec's regime or cite it.
**Dropped-direction overlap.** D3, mild: it reuses the profiling harness and a memory-bound versus compute-bound argument, but asks about a specific disputed claim rather than hunting for a bottleneck.

---

## Card 2. Long chains versus many short chains, once the server is busy

**Question.** At what server load does "sample k chains and take the first finishers" stop beating one long chain in accuracy per GPU-second, and does that crossover sit at the batch occupancy where the engine turns from memory-bound to compute-bound?

**Tension.** Three papers, three different rules; the third names the regime variable but does not measure it.

**Evidence.**
- [V] "Don't Overthink it" (arXiv 2505.17813v2; venue not confirmed): "shorter reasoning chains within individual questions are significantly more likely to yield correct answers—up to 34.5% more accurate than the longest chain sampled for the same question." and "short-3@k ... consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction)."
- [V] "Let Me Think!" (NeurIPS 2025 poster, arXiv 2505.21825): "sequential scaling can be exponentially more powerful than parallel scaling." and, on s1-32B with AIME 2024, "sequential scaling can not be efficiently replaced by parallel scaling for this mathematical task".
- [V] "Faster and Better LLMs via Latency-Aware Test-Time Scaling" (Findings of EMNLP 2025 per search, arXiv 2505.19634v4): "although sequential scaling has a better token efficiency, achieving a similar performance with 9× fewer tokens compared to parallel scaling, it turns out that parallel scaling achieves 1.6× lower latency." Limitations: "For large-scale servers, which deals with hundreds or thousands of requests at one time, the main bottleneck of inference is computation, and the budget of inference can be measured by #tokens."
- [U] arXiv 2511.02309: sequential beats parallel self-consistency in 95.6% of configurations at matched compute (search summary only).

**Why unknown.** The parallel-wins results are measured with the GPU serving one question at a time, where extra branches ride on idle compute. The latency-aware paper concedes the result flips on a loaded server but only sketches this in an appendix. No paper measures where the flip happens or which accuracy claim holds when the unit is GPU-seconds under load.

**If yes (the crossover tracks the memory-bound to compute-bound boundary).** One measured engine curve (throughput versus batch occupancy) predicts the right thinking strategy at any load, which points to a load-aware choice between branching and continuing.
**If no.** The crossover is set by something else (variance of chain length, early-stop savings, KV memory per branch), and the roofline story used by both the test-time-scaling and the speculative-decoding papers is not sufficient.

**First experiment (about one week).** DeepSeek-R1-Distill-Qwen-14B and Qwen3-14B in thinking mode on one 48 GB card each, vLLM 0.32. Data: AIME 2024 + 2025, HMMT Feb 2025 (the first paper's sets) plus MATH-500 for statistical power. Record 20 sampled chains per question once, with per-token timestamps. Then run each strategy (one chain with a long budget; majority@k; short-1@k; short-3@k; k in 2, 4, 8, 16) while the server holds a fixed number of other active sequences (0, 8, 32, 64, 128). Measured numbers: accuracy at equal wall-clock and at equal GPU-seconds per question, and the occupancy at which the parallel strategy's latency advantage reaches zero.

**Who cares.** Serving teams that price reasoning effort; authors of test-time-scaling papers who report token counts or single-request latency.
**Closest existing answer.** The latency-aware paper (appendix A.4) states the regime change without measuring it. SART (arXiv 2505.13326) and a parallel MCTS paper (arXiv 2604.00510) serve branching under load but do not compare against one long chain at equal GPU-seconds [U, search summaries].
**Dropped-direction overlap.** None. It uses the D3 throughput-versus-batch curves as an input.

---

## Card 3. What makes a reasoning model write more when its KV cache is degraded?

**Question.** Is the lengthening of reasoning traces under KV-cache compression caused by the kind of approximation (eviction versus quantization), or does one measurable quantity, such as the shift in next-token distribution from the full-cache run, predict it for every method?

**Tension.** Four papers disagree on which methods lengthen outputs. The disagreement is direct and three-sided.

**Evidence.**
- [V] ThinKV (ICLR 2026 oral, arXiv 2510.01290v2): "in LRMs we find that aggressive quantization inflates L_gen, eroding memory savings and simultaneously degrading accuracy." and "Unlike quantization, eviction does not cause an increase in generation length". (L_gen is a math symbol in the source. Quantizer: KIVI; eviction: R-KV; GPT-OSS-20B on LiveCodeBench.) Also: quantization-only "yielding only a modest 1.1× improvement in throughput" because of "substantial generation-length inflation".
- [V] "Hold Onto That Thought" (arXiv 2512.12008, preprint; a NeurIPS 2025 virtual page exists, track not confirmed): "eviction strategies at low budgets can produce longer reasoning traces" and "KNorm, arguably the lowest performing strategy, tends to cause the greatest elongation of outputs." Table 5, end-to-end tokens/s on GSM8K: Full 30.71; every compressed setting is lower (H2O 23.41 to 25.81; R-KV 27.72 to 30.04).
- [V] "Quantization Hurts Reasoning?" (arXiv 2504.04823; venue not confirmed): "Contrary to expectations, quantized models do not exhibit increased output lengths." and "for various quantization configurations (e.g., W4G128, W8A8KV8, and KV4), the generated output lengths closely align with those of the BF16 models. However, under more aggressive low-bit quantization (e.g., W3G128 and W4A4KV4), LLMs tend to produce longer outputs." KV quantizer: QuaRot.
- [V] "Rethinking Key-Value Cache Compression Techniques for LLM Serving" (MLSys 2025, arXiv 2503.24000): "compressing KV cache may lead to longer outputs, resulting in increased end-to-end latency" and "more than 20% of the samples show at least a 1.5× increase in response length." Both quantization (KIVI-4, GEAR-4) and eviction (H2O-512, Stream-512) rows show it, on a chat model.
- [V] R-KV (NeurIPS 2025 poster, arXiv 2505.24133): claims "a 90% memory saving and a 6.6× throughput", and states that "The primary throughput improvement from R-KV stems from enabling significantly larger inference batch sizes".

**Why unknown.** Each paper uses different models, tasks, quantizers and budgets, and none puts eviction, quantization and a matched random perturbation on one axis. Candidate explaining variables that nobody has separated: the quantizer (KIVI versus QuaRot), the size of the accuracy loss, the task (code versus maths versus chat), and whether the damage falls on the tokens the model uses to decide it is finished.

**If yes (one quantity predicts lengthening).** Length cost can be forecast offline from a cheap divergence measure, and the papers' disagreement reduces to different operating points. Throughput claims can then be corrected for length without running the benchmark.
**If no (it depends on the method).** There is a method-specific mechanism, for example eviction removing the earlier result a model checks against, or quantization noise moving the end-of-thinking decision. That says which part of the cache has to be protected.

**First experiment (about one week).** DeepSeek-R1-Distill-Qwen-14B and GPT-OSS-20B (ThinKV's model), HF Transformers with the public R-KV code and NVIDIA's kvpress, one 48 GB card each. MATH-500 plus AIME 2024/2025, 8 samples per question. Conditions: full cache; KIVI at 4 and 2 bits; QuaRot KV4; R-KV and H2O at 3 budgets; Gaussian noise added to cached keys and values, scaled to match each method's divergence. Measured numbers: mean generated tokens relative to full cache, accuracy, and KL between compressed and full next-token distributions on teacher-forced full-cache traces. The test is whether length ratio versus KL falls on one curve across methods.

**Who cares.** Authors of KV-compression papers that report tokens/s without output length; serving teams deciding whether FP8 or 4-bit KV is free for reasoning models.
**Closest existing answer.** ThinKV asserts the quantization-versus-eviction split from one figure on one model; "Hold Onto That Thought" reports the opposite for eviction. Neither tests a common variable.
**Dropped-direction overlap.** D1, substantial. Same setting (KV subset or compression during long reasoning outputs). The difference is the dependent variable: output length and its cause, not selection quality. If D1 compared methods at fixed output length or did not log length, this is a gap in D1's own measurement.

---

## Card 4. Does the Poisson + ShareGPT recipe change which scheduler wins, or only how many machines you need?

**Question.** Do the relative gains of serving policies (length-aware ordering, preemption settings) measured under Poisson arrivals with ShareGPT lengths keep their sign and size under ServeGen workloads, and in particular under reasoning workloads, whose arrivals ServeGen itself finds close to Poisson?

**Tension.** A "rethinking" paper says the standard recipe is misleading, while its own data says the recipe's arrival model is roughly right for one important workload class.

**Evidence.**
- [V] ServeGen (NSDI 2026 per the repo catalogue and the author's page, arXiv 2505.09999v3): "The de facto approach (referred to as Naive) adopted by many studies ... generates workloads by simply combining certain arrival traces (e.g., sampled from Poisson or Gamma processes, or scaled from published traces) with datasets (e.g., ShareGPT)." and "ServeGen results in provisioning 25 instances (4% over the actual number needed), while Naive results in only 12 instances (50% under-provisioning)." and "the Naive workloads are misleadingly easier to serve than real workloads."
- [V] Same paper, reasoning workloads: "both workloads are non-bursty" and "the Exponential distribution fits the inter-arrival time distribution quite well (i.e., the arrival is roughly modeled by Poisson processes)"; "reason lengths can be on average 4× longer than answer lengths"; "their per-request ratio exhibits a consistent bimodal distribution".
- [U] BurstGPT (arXiv 2401.17644): search summary says its authors argue that KV-cache, scheduling and disaggregation optimizations generalize poorly under realistic workloads. Not opened.
- Repo catalogue (reports/Reproducible LLM systems paper setups.md): ShareGPT + Poisson is the most common recipe among the 160 papers surveyed.

**Why unknown.** ServeGen's two use cases are capacity provisioning and prefill-decode tuning. It does not re-run published scheduling policies. For reasoning workloads the recipe's error would have to come from lengths (bimodal reason-to-answer ratio, 4x longer reasoning), not from arrivals, and nobody has measured whether that changes policy rankings.

**If yes (rankings and gain sizes survive).** The cheap recipe is adequate for comparing policies even though it is wrong for sizing clusters. That is a useful licence for small labs.
**If no.** The property that flips the ranking (burstiness per client, or bimodal lengths) is the thing a policy should be designed around, and published gains need restating.

**First experiment (about one week).** vLLM 0.32, Qwen2.5-14B (ServeGen's model) or Qwen3-14B on one 48 GB card. Policies: FCFS; priority scheduling with oracle output length (an upper bound on every length-prediction scheduler); two preemption-watermark settings. Workloads at matched mean rate and mean tokens: Poisson + ShareGPT; ServeGen language category; ServeGen reasoning category (ServeGen's generator is public). Measured number: ratio of P99 TTFT and of goodput between each policy and FCFS, per workload, and whether the ordering of policies changes.

**Who cares.** Scheduler authors, reviewers of serving papers, the vime and vLLM default-setting discussions.
**Closest existing answer.** ServeGen (provisioning and PD only). BurstGPT may partly answer for chat workloads [U]; the reasoning-workload part appears open.
**Dropped-direction overlap.** D4, moderate: D4 found the watermark setting within 7% of a perfect schedule on RL rollouts. This asks whether such headroom numbers depend on the workload recipe; if D4's answer carries over, the "no headroom" result is the likely outcome and the card is less interesting.

---

## Card 5. Is "better than the full cache" real?

**Question.** Is R-KV's above-baseline accuracy (105% of full-cache accuracy at 16% of the cache on AIME-24) a real effect of removing redundant reasoning tokens, or is it inside the run-to-run noise of a 30-question benchmark?

**Tension.** A headline claim in one accepted paper sits below the noise level that two critical papers report for the same benchmark.

**Evidence.**
- [V] R-KV (NeurIPS 2025 poster, arXiv 2505.24133): "Remarkably, R-KV even achieves 105% accuracy of the full KV baseline with around 16% of the KV cache using DeepSeek-R1-Distill-Llama-8B on the AIME-24 dataset." Pass@1 from 64 samples per question. The paper also says: "We find that using greedy decoding to evaluate long-output reasoning models results in significant variability across different setups."
- [V] "A Sober Look at Progress in Language Model Reasoning" (arXiv 2504.07086v2; venue not confirmed): "AIME'24 and AMC'23 each contain only 30–40 examples, making performance metrics highly volatile, where even one question can shift Pass@1 by over 3 percentage points."
- [V] "Give Me FP32 or Give Me Death?" (arXiv 2506.09501; venue not confirmed): "under bfloat16 precision with greedy decoding, a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size."
- [V] "Don't Overthink it" gives a reason the effect could be real: within a question, shorter chains are "up to 34.5% more accurate than the longest chain".

**Why unknown.** R-KV reports no confidence interval. With 30 questions x 64 samples, the binomial standard error per arm is at most about 1.1 points (my arithmetic, treating samples as independent, which overstates precision because samples cluster by question). A 5% relative gain is of that order. Nobody has tested the improvement on a set large enough to resolve it.

**If yes (real).** Eviction acts as a filter that helps reasoning, which is a mechanism worth isolating: which evicted tokens were hurting?
**If no (noise).** This is the less interesting answer: a negative replication. Its use is as a noise floor that every "lossless at X% budget" claim, and the small margins in D1, must clear.

**First experiment (about one week).** DeepSeek-R1-Distill-Llama-8B with the public R-KV code, one 48 GB card per arm. Full cache versus R-KV at 10%, 16%, 34% budgets on MATH-500 (16 samples per question) and AIME 2024 + 2025 (64 samples). Run the full-cache arm twice at different batch sizes to measure the engine's own noise. Measured number: paired bootstrap confidence interval over questions for the accuracy difference, compared with the full-versus-full difference.

**Who cares.** Authors and reviewers of KV-compression and quantization papers evaluated on AIME.
**Closest existing answer.** "Hold Onto That Thought" re-benchmarks R-KV on more tasks and does not report above-baseline accuracy as a finding; no paper found that tests the claim with intervals.
**Dropped-direction overlap.** D1: same family of methods and the same problem of small accuracy margins.

---

## Card 6. Cache thrashing with agents: policy problem or capacity problem?

**Question.** When concurrent coding agents overflow GPU memory, how much of the reported prefix-cache collapse (hit rate from about 98% to 16-26%) can an oracle eviction order recover, given that the same oracle bound was only 2.6-5.5% on chat traces?

**Tension.** One accepted paper says a small cache is enough; another reports collapse. The student's own measurement agrees with the first.

**Evidence.**
- [V] "KVCache Cache in the Wild" (USENIX ATC 2025 per search, arXiv 2506.02634): "the P99 lifespan of KV$ in to-B workloads is 97 seconds, so a small cache is typically sufficient for the KV$ cache." and "the overall cache size required for an ideal cache hit ratio is moderate."
- [V] "Agentic AI Workload Characterization" (IISWC 2026, arXiv 2605.26297): "Context thrashing first appears as a sharp prefix-cache hit-rate drop, from about 98% to 16–26%, shifting the workload toward prefill." and "reducing average generation throughput by up to 66%". Also: "Our results assume recomputation under thrashing." Mean context on SWE-bench Pro "between 68.7K and 80.1K tokens". Two H100 NVL, vLLM v0.20.0.
- Measured in this repo (D2): a perfect checkpoint and prefix-cache policy gained 2.6-5.5% on Mooncake traces.

**Why unknown.** The agent paper measures the default policy with recomputation and no oracle. The production paper covers chat and API traffic with short reuse gaps. Nobody reports the oracle-versus-LRU gap for agent sessions, where the reuse gap is a tool call and the context is 50K-80K tokens.

**If yes (an oracle recovers most of it).** The collapse is a policy problem and predictable tool durations are the signal to exploit.
**If no.** It is a capacity problem; only admission control or offloading helps, and eviction-policy papers for agents have little room.

**First experiment (about one week).** Replay the agent paper's public request traces (its artifact appendix lists raw and processed data on figshare; I did not open the data) through the D2 cache simulator with LRU and Belady orders at the KV capacity of one 48 GB card for Qwen3-14B, for 2 to 32 concurrent agents. Then confirm two points live in vLLM 0.32. Measured numbers: hit rate and generation throughput, LRU versus oracle, against the number of concurrent agents.

**Who cares.** Teams serving coding agents on fixed GPU budgets.
**Closest existing answer.** The agent characterization paper itself (no oracle). ThunderAgent (arXiv 2602.13692) and similar systems propose policies; I did not open them, so whether one reports an oracle bound is unchecked.
**Dropped-direction overlap.** D2, substantial: same method (oracle policy bound for prefix caching), different workload (agents, plain transformers). If the D2 Mooncake runs already included the tool-agent trace, part of this is answered by existing data.

---

## Checked and not carded

- Tensor versus pipeline parallelism on PCIe. "Revisiting Pipeline Parallelism for LLM Serving" (OSDI 2026) [V]: "for Qwen2.5 32B and 14B on four NVIDIA A100 40GB GPUs, pipeline parallelism with our mechanisms outperforms tensor parallelism." Four A100 "connected via PCIe 4.0". This contradicts the usual default and matches the student's hardware, but the paper answers it, including a decode-heavy workload. Residue: vLLM instead of SGLang, reasoning-length outputs, MoE and hybrid models.
- Prefill-decode aggregation versus disaggregation. TaiChi (arXiv 2508.01989) [U] already gives the explaining variable (which SLO is tight).
- ReplaySSM blog versus the D3 result. Blog [V]: caching SSM inputs "speeds up standard decoding by up to 1.48x"; D3 measured decode at 89-95% of the hardware limit. Worth one check of whether the D3 limit counted the per-step state writes the blog removes. A blog, and a fix already exists, so not a card.
- Prompt caching for agents. "Don't Break the Cache" (arXiv 2601.06007) [V]: caching "reduces API costs by 41-80%" while naive full-context caching "can paradoxically increase latency". API providers only; nothing to measure locally.
- Tool time share of agent latency (30.2% in the HPCA 2026 paper versus up to 28.7% on GAIA in the IISWC 2026 paper, from the repo catalogue). Different benchmarks; not a real conflict.
- Output-length prediction for scheduling (arXiv 2604.00499, 2603.07917) [U]: no clean opposing pair found.

## Not verified

- Venues: SmartSpec, "Don't Overthink it", "Quantization Hurts Reasoning?", "A Sober Look", "Give Me FP32", "Hold Onto That Thought" (preprint; a NeurIPS 2025 virtual page exists). MoESD as NeurIPS 2025, latency-aware TTS as Findings of EMNLP 2025, and ATC 2025 for the KV-cache trace paper come from search results, not the papers' pages.
- Search-summary only: arXiv 2511.02309 numbers, BurstGPT's generalization claim, SART, arXiv 2604.00510, TaiChi numbers, ThunderAgent.
- MagicDec's headline "up to 2.51x" and its implementation stack came from a WebFetch summary; the quotes used above came from the v5 HTML.
- In "Performance or Illusion?", the statement that tree verification at k=21 falls below 1x by batch 64 came from a WebFetch summary.
- Rethinking-KV Table 4 length ratios (KIVI-4 1.69, GEAR-4 1.70, H2O-512 1.55, Stream-512 1.76) appear to describe the lengthened samples, not all samples; I did not resolve this.
- A6000 and 6000 Ada compute-to-bandwidth ratios relative to L40/H100 were not looked up.
- Throughput and runtime estimates in the first experiments are my arithmetic, not measurements.
