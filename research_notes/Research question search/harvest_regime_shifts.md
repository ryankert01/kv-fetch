# Harvest: regime shifts that make earlier results stale

Working notes, 2026-11-11. Lens: results that were established under an older setting (open-loop chat traffic,
dense Transformers, short outputs, API pricing) and may not hold under a 2025-2026 setting (closed-loop agent
sessions, hybrid linear-attention models, long thinking outputs, self-hosted engines).

Each card is a question, not a system. Cards are not ranked. "Verified" means I opened the primary source and
copied the quote or number from its text. Dropped-direction labels: D1 decode-time KV page selection, D2 state
checkpoints and prefix caching for hybrids, D3 vLLM 0.32 single-GPU profiling, D4 RL rollout acceleration.

## Method used

1. List the regime shifts that are documented in a primary source (table below).
2. For each shift, list published results whose evaluation setting sits on the old side of the shift.
3. Keep a pair only if (a) the direction of the effect under the new setting is arguable both ways, (b) at least
   two primary sources give specific claims or numbers, and (c) a search does not turn up a paper that already
   answers it.
4. Check each kept pair against D1-D4 and say where it overlaps.

## Regime shifts with a primary source

| Shift | Source | Quote or number (verified) |
|---|---|---|
| Single-turn requests to closed-loop multi-turn agents | Agentic AI Workload Characteristics, IISWC'26 (venue as stated by authors), https://arxiv.org/html/2605.26297 | "They shift the LLM serving paradigm from isolated single-turn prompt–generation requests to a closed-loop, multi-turn execution process"; "most configurations average between 12–62 turns per task" |
| Decode dominates agent serving | same | "decode accounts for 91.0–98.6% of LLM time, while prefill contributes only 1.4–9.0%" |
| Thinking is a large share of output | same | "Gemma Thinking spends 45.8–67.6% of its output on thinking across workloads, while Qwen Thinking spends 29.0–40.7%" |
| Cache thrashing cliff under concurrency | same, Table II | Terminal-Bench 2.0, Qwen3.6-27B: concurrency 15/20/25 gives prefix-cache hit 98.85/16.78/19.21% and 30.08/16.96/10.30 tok/s per request |
| Hybrid layout in shipped open models | Qwen3.6-27B model card, https://huggingface.co/Qwen/Qwen3.6-27B | "Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))" |
| Models trained to keep earlier thinking | same | "Qwen3.6 has been additionally trained to preserve and leverage thinking traces from historical messages." |
| Same-prompt output length is itself random | Beyond Prediction: Tail-Aware Scheduling, ICML 2026, https://arxiv.org/html/2606.18431v1 | S1K reasoning data: "Qwen: CV∈[0.00,0.19], DeepSeek: CV∈[0.00,0.12]" |

## Card 1. Open-loop scheduling gains under closed-loop agent load

**Question.** Do the latency and throughput gains of length-aware (shortest-job-first style) LLM request
scheduling, reported under open-loop Poisson arrivals, survive when the load is a fixed population of multi-turn
agent sessions, or does request ordering stop mattering while memory admission sets throughput?

**Tension.** Evaluation-regime mismatch. A classical systems result predicts the published gains shrink in closed
systems; LLM serving adds cache state that could push the other way.

**Evidence.**
- Schroeder, Wierman, Harchol-Balter, "Open Versus Closed: A Cautionary Tale", NSDI 2006,
  https://www.usenix.org/legacy/events/nsdi06/tech/full_papers/schroeder/schroeder_html/ (verified): "favoring short
  jobs is highly effective in improving mean response time in open systems, while this is far less true under a
  closed system model. We find that closed system models only benefit from scheduling under a narrow range of
  parameters, when load is moderate and the MPL is very high."
- Learning to Rank, NeurIPS 2024, https://arxiv.org/html/2408.15792 (verified): "Under the rate of 64
  requests/second, our proposed method improve the mean latency by up to 6.9× compared with FCFS".
- Autellix, NSDI 2026, https://arxiv.org/html/2502.13965 (verified): "we synthesize a trace by randomly sampling
  programs, not LLM calls, from the above workloads and generating programs' arrivals using a Poisson process";
  "Faster call completions prompt programs to issue subsequent calls more quickly, increasing the arrival rate of
  LLM calls."; "Autellix achieves up to 8× throughput of vLLM".
- Continuum, arXiv 2511.02230 (venue not confirmed), https://arxiv.org/html/2511.02230v2 (verified): "We set up our
  agent client environment by adding a job distributor for the SWE-Bench platform that distributes agents in
  poisson distribution."
- IISWC'26 characterization (verified, see table): closed-loop framing; hit rate 98.85% to 16.78% between
  concurrency 15 and 20; "reducing average generation throughput by up to 66%". FCFS vLLM only, no policy comparison.
- Own measurement (project memory, kv-fetch-phase0.md; EXPERIMENTS.md "RL rollouts" R1): on closed-loop RL
  rollouts the hybrid at watermark 0.1 is 6.4-6.8% from the perfect-packing bound, while the default watermark 0
  costs the hybrid 26-37% (dense 5-7.6%).

**Why unknown.** The LLM scheduling papers use open or partly-open arrivals (requests or programs drawn from a
Poisson process). The agent characterization paper ran closed-loop concurrency but compared no scheduling
policies. The 2006 result is for web servers with no per-session memory state. In LLM serving a closed loop also
means each session's cache must stay resident between turns, so ordering decides whose cache gets evicted; that
could make ordering matter more, not less. No paper found that runs the same policy under both arrival models.

**If the gains survive.** The 2006 result does not transfer, and the reason would be cache-state coupling. That
points to ranking by cache residency instead of by remaining length.

**If the gains vanish.** Published scheduler speedups are partly a product of the arrival model, and the number of
admitted sessions (MPL) is the knob that matters. This agrees with what D4 already showed, so the new content is
the size of the gap in published numbers and the conditions (think time, session count) where it flips, not a new
mechanism.

**First experiment.** vLLM 0.32, Qwen3.5-9B and Qwen3.6-27B FP8, one 48 GB card each. Record about 100 agent
task traces (mini-SWE-agent on SWE-bench Verified; the IISWC'26 authors say their infrastructure will be
open-sourced, use that if it is out) and replay the per-turn prompt and output lengths with tool latency as think
time. Two drivers: Poisson program arrivals as in Autellix, and a fixed population of N sessions where a new one
starts when one finishes. Policies: FCFS, oracle SJF on remaining tokens, program-level least attained service.
Measured number: the ratio of mean and P95 program completion time, policy over FCFS, in each driver, across N and
think time.

**Who cares.** Authors and users of LLM schedulers, people who design serving benchmarks, operators of agent and
RL rollout infrastructure.

**Closest existing answer.** Schroeder et al. 2006 (general systems). Autellix states the feedback effect but
evaluates with Poisson program arrivals. A NeurIPS 2026 ML-for-Systems workshop poster, "From Conversational Chats
to Agents: Workload-Dependent Bottleneck Shifts in LLM Serving" (https://openreview.net/forum?id=mTDnSps98P, seen
only in a search summary, not opened) reports that scheduling cannot compensate for exhausted GPU capacity; the
summary does not show an open-versus-closed comparison. No direct answer found.

**Overlap.** D4. The RL rollout workload was closed-loop and its result is one data point for "gains vanish". A
new scheduler here should expect the same small headroom D4 found; what is open is the measurement.

## Card 2. What the remaining attention layers in a hybrid are doing

**Question.** Does the premise of KV eviction and selection, that a small set of tokens carries most of the
attention mass, hold for the few attention layers left in Gated DeltaNet hybrids, or do those layers attend more
diffusely than a dense Transformer's while the output depends on that attention less?

**Tension.** A stated hypothesis in the hybrid literature (attention is the retrieval mechanism, which predicts
concentrated attention) against a measured anomaly (attention in the hybrid is more diffuse).

**Evidence.**
- H2O, NeurIPS 2023, https://arxiv.org/abs/2306.14048 (verified): "a small portion of tokens contributes most of the
  value when computing attention scores".
- Quest, ICML 2024, https://arxiv.org/abs/2406.10774 (verified): "Previous works have shown that a small portion of
  critical tokens will dominate the attention outcomes. However, we observe the criticality of a token highly
  depends on the query."
- Mamba-2 (SSD), ICML 2024, https://arxiv.org/html/2405.21060 (verified): "We hypothesize that the SSM layers
  function well as a general sequence-to-sequence mapping, and attention layers act as a retrieval mechanism to
  quickly refer to previous tokens in the sequence instead of forcing the model to compress all the context to its
  memory (SSM states)."
- Jet-Nemotron, NeurIPS 2025, https://arxiv.org/html/2508.15884v3 (verified): "For MMLU, only two layers exhibit
  critical importance, while for retrieval tasks, just two to three layers are particularly crucial."; "the optimal
  placement of these layers remains unclear."
- Priming, arXiv 2605.08301 (venue not confirmed), https://arxiv.org/html/2605.08301 (verified): SSM hybrids
  outperform a sliding-window-attention hybrid "by at least 5.5% (relative) on long-context benchmarks, even when
  the SWA window is sized to give a 2-4× larger KV cache footprint than the SSM state."
- Own measurement (project memory, H-a and H-b): oracle top-512 misses 22.7% of attention on Qwen3.5-4B against
  11.6% on Qwen3-4B, every layer more diffuse; offline KL per unit of missing attention is 7.5-9.6x lower on the
  hybrid; online SnapKV at 1K loses 40.0 vs 38.8 points (ratio 1.03).

**Why unknown.** Hybrid papers choose attention-layer placement by task accuracy and do not report token-level
attention distributions. Eviction and selection papers used dense models. The own measurement is one model pair on
reasoning prompts, and its offline and online halves disagree, so there is no explanation yet.

**If concentrated on retrieval-type inputs.** The diffuse result is specific to reasoning traces; eviction methods
carry over to hybrids for retrieval work, and budgets should depend on the task.

**If diffuse across tasks.** The retrieval account is incomplete for these models. The attention layers would be
doing something closer to broad aggregation, which points to compressing them by pooling or low-rank keys rather
than by top-k token choice.

**First experiment.** Qwen3.5-4B and 9B against Qwen3-4B and 8B. Three input families: retrieval (passkey and
RULER subsets at 8K-32K), long-document QA (a LongBench subset), reasoning traces (MATH-500, AIME). Per attention
layer: attention mass missed by oracle top-512 and top-1K, attention entropy, and next-token KL when attention is
restricted to the top-k. Then one causal check: online accuracy at budget k on passkey versus MATH-500. Forward
passes only, one GPU.

**Who cares.** KV compression researchers, designers of hybrid models (layer placement), engine developers sizing
attention KV for hybrids.

**Closest existing answer.** Jet-Nemotron (layer importance by task), Priming (SSM versus sliding window), FlashMorph
(arXiv 2606.30562: "Keeping more full-attention layers preserves the retrieval and precise matching ability of the
original Transformer"). An EMNLP Findings 2026 paper on hybrids losing retrieval ability could not be opened
(connection refused). None found that measures token-level attention concentration in hybrid attention layers.

**Overlap.** D1, strongly: same oracle@k harness, and it asks why D1's premise behaved differently on hybrids. If
the answer is "diffuse and unimportant" it weakens D1 on hybrids further. D2 only in that both concern hybrids.

## Card 3. Keeping or dropping earlier-turn thinking in agent loops

**Question.** For open reasoning models running multi-turn tool-using agent tasks, does keeping earlier turns'
thinking in the context raise or lower task success and GPU time per solved task, compared with keeping only the
latest turn's thinking or none?

**Tension.** Sources disagree on direction, and none measures serving cost.

**Evidence.**
- Qwen3.6-27B model card, https://huggingface.co/Qwen/Qwen3.6-27B (verified, raw README): "By default, only the
  thinking blocks generated in handling the latest user message is retained"; "This capability is particularly
  beneficial for agent scenarios, where maintaining full reasoning context can enhance decision consistency and, in
  many cases, reduce overall token consumption by minimizing redundant reasoning. Additionally, it can improve KV
  cache utilization, optimizing inference efficiency in both thinking and non-thinking modes." No numbers given.
- SWE-AGILE, arXiv 2604.11716 (venue not confirmed), https://arxiv.org/html/2604.11716 (verified): "Retaining full
  reasoning history (Interleaved Thinking) significantly degrades performance (12.42%) compared to discarding it
  (Current-Step Thinking)(15.83%), despite the richer context. Additionally, we hypothesize that this degradation
  may also be partially related to the post-training process of Qwen3 models." (Qwen3-8B, SWE-Bench-Verified)
- "Unified Audio Intelligence Without Regressing on Text Intelligence", arXiv 2607.05196 (venue not confirmed),
  https://arxiv.org/html/2607.05196 (verified): no-carry-over "consistently reduces scores by 3–5 points relative to
  latest-turn thought retention"; full retention "gives similar accuracy to latest-turn thought retention while
  incurring substantially longer contexts."
- IISWC'26 characterization (verified): thinking is 29.0-67.6% of output tokens, so the policy changes context
  growth by a large factor.

**Why unknown.** Three sources, three models, three benchmarks, one or two policies each. The vendor claim has no
numbers. SWE-AGILE blames part of its result on Qwen3 post-training, and Qwen3.6 was trained for retention, so
model generation is a confound. Nobody reports context growth, cache reuse or GPU time alongside accuracy.

**If keeping helps on both counts.** The card's claim gets numbers, thinking should be treated as cacheable
context, and longer contexts move the thrashing point to lower concurrency by a measurable amount.

**If keeping hurts or is neutral at higher cost.** The strip-by-default policy is right, but then every user turn
invalidates the cached prefix after the last retained message. For a hybrid that needs a recurrent-state
checkpoint at the strip point. Neutral accuracy at higher cost is the least interesting outcome; it still settles a
default.

**First experiment.** Qwen3.6-27B FP8 (preserve_thinking option in the chat template) and Qwen3.5-9B on vLLM.
100-task subsets of SWE-bench Verified with mini-SWE-agent and of tau2-bench. Three policies: keep all thinking,
latest turn only, none. Fixed concurrency of 8 sessions per GPU. Measured numbers: task success, generated tokens
per task, thinking tokens per turn, mean context length, prefix-cache hit rate, GPU-seconds per solved task.

**Who cares.** Agent framework authors, model vendors, engine developers working on prefix reuse, anyone paying
for agent inference.

**Closest existing answer.** SWE-AGILE (one model, one benchmark, accuracy only) and the audio paper's ablation
(different model, tau2-bench). No controlled study across policies with accuracy and serving cost found.

**Overlap.** D2, partly. If dropping wins, the strip point is where a hybrid needs a state checkpoint, the same
mechanism as D2 on a workload different from the Mooncake traces. Whether the ceiling is larger than D2's 2.6-5.5%
is not known.

## Card 4. Output-length inflation under lossy KV methods

**Question.** When KV-cache quantization, eviction or page selection is applied to a reasoning model, does the
output get longer by enough to cancel the per-token savings, and does that depend on the method, the model, or how
close the problem is to the model's limit?

**Tension.** Published papers disagree on whether and when inflation happens; throughput is usually reported at a
fixed sequence length.

**Evidence.**
- Rethinking Key-Value Cache Compression, MLSys 2025, https://arxiv.org/html/2503.24000 (verified): "KV cache
  compression leans toward generating lengthy responses. Specifically, more than 20% of the samples show at least a
  1.5× increase in response length." (1,000 ShareGPT samples, LLaMA-3.1-8B-instruct)
- ThinKV, ICLR 2026 (Oral), https://arxiv.org/html/2510.01290 (verified): "our R1-Llama-8B results show that pure
  quantization can inflate generation length by up to 5.1×. In contrast, eviction-based approaches do not induce
  such drastic inflation."; "substantial generation-length inflation negates most of the compression gains,
  yielding only a modest 1.1× improvement in throughput".
- "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models", arXiv 2504.04823 (venue not
  confirmed), https://arxiv.org/html/2504.04823 (verified): "Contrary to expectations, quantized models do not
  exhibit increased output lengths."; "under more aggressive low-bit quantization (e.g., W3G128 and W4A4KV4), LLMs
  tend to produce longer outputs."
- R-KV, NeurIPS 2025, https://arxiv.org/html/2505.24133 (verified): "at a sequence length of 16K, R-KV achieves up
  to 9× larger batch sizes and over 6.6× higher throughput under a 10% compression ratio".
- Own measurement (project memory, E3 pilot and H3): under page selection, "Losses are mostly truncation at 16K
  (not verbatim loops, except SnapKV)"; on the hybrid, losses mostly 32K truncation.

**Why unknown.** The three papers use different models, bit widths and tasks. None covers hybrids or selection
methods. None separates the method's effect from ordinary run-to-run length variance, which the ICML 2026
tail-aware scheduling paper measured at CV up to 0.19 for the same reasoning prompt. The own truncation losses fit
inflation but were not separated from problems that would have failed anyway.

**If inflation is systematic.** Throughput at a fixed sequence length overstates the end-to-end gain, the right
metric is GPU-seconds per correct answer, and the mechanism (for example re-deriving facts lost from context)
would say which tokens to protect.

**If inflation is confined to a few settings.** ThinKV's quantization result is narrow, fixed-length throughput
claims stand, and the truncation losses in D1 have another cause. Less striking, but it reconciles two papers.

**First experiment.** DeepSeek-R1-Distill-Llama-8B (ThinKV's model), Qwen3-8B, Qwen3.5-9B on MATH-500 and AIME with
a 32K cap. Methods: full cache, KV quantization at 8 and 4 bits, eviction with the public R-KV code at 10-30%
budget, Quest-style selection at 512 and 1K with the existing harness. At least three seeds per problem. Measured
numbers: paired output-length ratio against full cache (median and share at 1.5x or more), share hitting the cap,
accuracy, generated tokens per correct answer.

**Who cares.** KV compression authors, operators who size by throughput, benchmark designers.

**Closest existing answer.** ThinKV's length analysis (quantization versus eviction, one model) and the quantized
reasoning study (lengths match except at aggressive bit widths). They conflict and cover neither hybrids nor
selection.

**Overlap.** D1. Reuses the D1 harness and explains D1's loss mode. It does not change D1's margin over Quest.

## Card 5. Append-only context versus compaction on a self-hosted engine

**Question.** On a self-hosted engine, where cached context is not billed but still occupies KV memory and is
attended over at every decode step, does the "append-only, never break the cache" rule for agents still minimise
GPU time per solved task, or does periodic context compaction win once concurrency nears the cache-thrashing point?

**Tension.** A rule derived from API prices applied to a setting with a different cost model.

**Evidence.**
- Manus engineering blog (practitioner source, not peer reviewed),
  https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus (verified): "the KV-cache hit
  rate is the single most important metric for a production-stage AI agent."; "cached input tokens cost 0.30
  USD/MTok, while uncached ones cost 3 USD/MTok—a 10x difference."; "Make your context append-only."
- "Don't Break the Cache", arXiv 2601.06007 (venue not confirmed), https://arxiv.org/html/2601.06007 (verified):
  "prompt caching reduces API costs by 41 to 80% and improves TTFT by 13 to 31%"; "Techniques such as summarizing or
  pruning old tool calls (Ji, 2025) break cached representations, making tool call caching counterproductive."
  Provider APIs only.
- IISWC'26 characterization, https://arxiv.org/html/2605.26297 (verified): "decode accounts for 91.0–98.6% of LLM
  time, while prefill contributes only 1.4–9.0%"; hit rate falls from about 98% to 16-26% past a concurrency
  threshold; "dynamic context compaction or summarization ... are outside the scope of this study ... we leave their
  characterization to future work."

**Why unknown.** The rule's argument is a price ratio. On an owned GPU, reused prefill is free either way, while
resident context limits how many sessions fit and adds attention cost to every decode step. The one study of the
thrashing cliff ran append-only agents and left compaction out. On hybrids only a quarter of the layers hold KV,
which changes the trade again. Compaction also changes task success, so cost and accuracy have to be measured
together.

**If append-only still wins.** The rule transfers, and the cliff has to be handled by admission control, not by
context management.

**If compaction wins past some concurrency.** There is a crossover, and its location is the result. It points to
compaction triggered by an engine memory-pressure signal instead of by the context-window limit.

**First experiment.** Qwen3.6-27B FP8 on vLLM, one or two 48 GB cards. 50-100 tasks from Terminal-Bench 2.0 or
SWE-bench Verified. Conditions: append-only, and compaction of older tool outputs (summarised by the same model)
when context passes 25% or 50% of the limit. Sweep concurrency across the thrashing point. Measured numbers:
GPU-seconds per solved task, success rate, prefix-cache hit rate, generation throughput per request; the headline
is the concurrency, if any, where compaction's GPU-seconds per solved task drops below append-only's.

**Who cares.** Agent harness authors, teams that self-host, engine developers.

**Closest existing answer.** "Don't Break the Cache" (API cost), the IISWC'26 paper (cliff without compaction),
Continuum (keeps KV pinned across tool calls, does not compact). No self-hosted measurement found.

**Overlap.** D2 partly (prefix reuse on hybrids, but as an agent-side policy, not a checkpoint mechanism). D4
partly (the same memory-pressure and preemption behaviour sets where the cliff is).

## Candidates screened and not kept

| Candidate | Why not kept | Source |
|---|---|---|
| Is reasoning output length predictable enough for SJF? | Largely answered: same-prompt length variance measured on 2 models x 3 datasets, 20 runs each | Tail-Aware Scheduling, ICML 2026, https://arxiv.org/html/2606.18431v1 (verified) |
| Does the speculative-decoding batch crossover (EAGLE-3: "38% throughput improvement at a batch size of 64"; MagicDec: "for S≥4000, the speedup increases with batch size") hold on hybrids with MTP? | Answered on mechanism: vLLM stores a state snapshot per draft token, "storing the snapshots caps concurrency, so speculative decoding gains little at serving batch sizes". vLLM's Qwen3.5 recipe says the same for MTP ("degrades text throughput under high concurrency because speculative tokens consume KV cache capacity"). Own measurements agree (MTP halves engine concurrency on hybrids) | ReplaySSM, arXiv 2610.13778 (verified, abstract and intro); vLLM recipe https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html (verified); EAGLE-3 https://arxiv.org/html/2503.01840 and MagicDec https://arxiv.org/html/2408.11049 (verified). SpecLA (arXiv 2607.16673) and STree (arXiv 2505.14969) not opened |
| MoE batch scaling at large batch | Mechanism already described (expert activation spreads with batch) | Lynx arXiv 2411.08982, XShare arXiv 2602.07265, MoE-Inference-Bench arXiv 2508.17467; none opened, search summaries only |
| Prefill-decode disaggregation under output-heavy reasoning load | Believed answered by TaiChi-style work; from memory, not searched in this pass | not verified |
| Diffusion LMs under batching | Answer looks predictable: Fast-dLLM concedes at larger batch it "struggles to match LLaMA" (as quoted in the project's synthesis report, primary not opened) | not verified |
| Reasoning-phase TTFT scheduling | Covered by PASCAL (from the project's catalogue, primary not opened) | not verified |

## Not verified

- NeurIPS 2026 ML-for-Systems poster "From Conversational Chats to Agents" (https://openreview.net/forum?id=mTDnSps98P): search summary only.
- EMNLP Findings 2026 paper 1194 on hybrids and retrieval: fetch failed, description from a search summary only.
- SMetric (arXiv 2607.08565), SpecLA, STree, Lynx, XShare, MoE-Inference-Bench, MoE-Gen: search summaries only.
- Venues for Continuum, Priming, FlashMorph, SWE-AGILE, the audio paper, the quantized-reasoning study, "Don't Break
  the Cache" and ReplaySSM were not confirmed; they are cited as arXiv.
- IISWC'26 and NSDI'26 (Autellix) venue labels are as recorded in the project's synthesis report; not rechecked.
- Whether the IISWC'26 tracing infrastructure is public yet (the paper says it "will be open-sourced").
- Card 1 has no pilot on agent traces; the only closed-loop data point is the RL rollout result.
