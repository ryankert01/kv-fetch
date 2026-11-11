# Harvest: unresolved practitioner disputes and unexplained behaviour

Date: 2026-11-11. Lens: places where practitioners (issue threads, RFCs, vendor model cards, forums, provider blogs) disagree, cannot explain something, or cannot reproduce a paper claim. Read-only harvest; nothing was posted anywhere.

Five question cards came out. They are not ranked. Each card says where it overlaps a direction that was already dropped (D1 KV page selection, D2 hybrid state checkpoints / prefix caching, D3 vLLM single-GPU profiling, D4 RL rollout acceleration).

Verification key: **[V]** = I opened the primary source and the quote or number is copied from it. **[U]** = not opened or could not be read; the statement comes from a search summary or a secondary quote.

## How the search was done

1. `gh search issues` / `gh issue view` (read-only) on vllm-project/vllm, ggml-org/llama.cpp, sgl-project/sglang and a few agent harness repos, sorted by comment count, filtered to 2025-2026 and to threads that are still open or closed without a root cause.
2. For each thread cluster, look for the paper claim it touches; fetch the arXiv HTML and grep for the exact sentence.
3. Drop a candidate when a paper already answers it (list at the end).

What the 2026 practitioner record clusters around: numerics of hybrid (GDN / Mamba-2) models, training-inference consistency for RL, handling of reasoning traces in multi-turn history and the cache breaks that follow, and quantization anecdotes for agent use.

Not mined: huggingface/text-generation-inference, NVIDIA Dynamo, TensorRT-LLM (no time). SGLang gave only closed "slower than vLLM" threads with configuration answers.

---

## Card A. Does the decode-vs-scan logprob gap in hybrid models grow with generated length?

**Question.** In hybrid recurrent-state models (GDN / Mamba-2 layers), does the logprob gap between step-by-step decode and chunked-scan rescoring of the same tokens grow with generated length because rounding differences compound through the recurrent state, or does it stay at last-bit scale as it does for attention layers?

**Tension type.** Unexplained behaviour with a missing guarantee: an engine RFC states the mismatch exists at layer level, a paper shows small mismatches can collapse RL training on transformers, nobody has measured the hybrid case end to end.

**Evidence.**
- [V] issue thread, vLLM #42259 "[RFC]: Logprobs/Logits Semantics and Determinism Across the vLLM Ecosystem" (open), comment by NolenLiang, 2026-09-06, https://github.com/vllm-project/vllm/issues/42259 : "Mamba2 layers use different kernels for prefill (chunked scan) and decode (recurrent state update), so returned logprobs depend on how a request was scheduled: prefill vs decode, chunked-prefill split points, recompute after preemption. For RL this is also the layer-level source of rollout/trainer logprob mismatch on Mamba2 and hybrid models, since the trainer scores generated tokens with the chunked scan." Same comment: "the two kernels differ from token 0 at bf16 last-bit scale over 22-45% of output elements".
- [V] PR, vLLM #54993 "[Feature][Mamba] Support batch-invariant Mamba2 prefill and recovery" (open), https://github.com/vllm-project/vllm/pull/54993 : "It does not establish general `prefill(prefix + token) == decode(token)`, RL training/inference alignment, equality between BI-on and BI-off, or equality across hardware, execution modes, or TP sizes." and "Mamba1 and GDN do not acquire support through this change."
- [V] issue thread, vLLM #48613 (open), https://github.com/vllm-project/vllm/issues/48613 : "RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN."
- [V] paper (preprint, venue not confirmed), arXiv 2605.14220 "Diagnosing Training Inference Mismatch in LLM Reinforcement Learning via a Zero-Mismatch Reference", https://arxiv.org/abs/2605.14220 : "although the average difference in token-level probabilities is small per training batch, the maximum difference can even reach 1.0 for some extreme tokens"; GRPO on Qwen3-1.7B: "VeXact keeps the training reward around 0.93, vLLM recomputation first degrades from about 0.87 to roughly 0.40 during the first 650 steps". Transformers only (dense Qwen3, Qwen3-30B-A3B, Llama).
- [V] issue thread, vLLM #62851 (open, partly retracted by the reporter), https://github.com/vllm-project/vllm/issues/62851 , independent check by potoior 2026-11-04: "the hybrid GDN probe twin is bit-exact across 17 serving configurations — 835,584 generated tokens, 0 mismatched". This is decode against decode; it says nothing about decode against scan.
- [V] issue thread, vLLM #48305 "[RFC] Training-Inference Consistency for RL" (open), https://github.com/vllm-project/vllm/issues/48305 : "Batch invariance is one component of training–inference numerical consistency, not a full parity guarantee."
- [U] Yao, Liu et al., "On the Rollout-Training Mismatch in Modern RL Systems" (NeurIPS 2025 workshop listing https://neurips.cc/virtual/2025/124259 , blog https://fengyao.notion.site/off-policy-rl ). The blog is a JS page I could not read; not quoted.

**Why unknown.** The layer-level difference is reported, the end-to-end consequence is not. For attention, a rounding difference at step t does not feed a state that later steps read through a recurrence; for GDN it does. Whether the recurrence damps or amplifies it in deployed checkpoints is not in any thread or paper I found. The VeXact paper has no hybrid model.

**If yes (gap grows).** Rescoring-based RL (GRPO, PPO with TIS) on hybrids has a length-dependent off-policy error that long reasoning rollouts hit hardest; the design question becomes which side to align (trainer steps like decode, or decode re-scans) and what it costs.

**If no (gap stays flat).** Hybrid models are no worse than transformers for training-inference mismatch; the open vLLM RFC item can be closed with a measured bound. Useful, less generative.

**First experiment (about a week, 1 GPU).** Qwen3.5-4B and Qwen3.5-9B (hybrid) against Qwen3-4B (attention control), bf16 weights, SSM cache in fp32 and bf16. Sample 200 MATH-500 prompts with thinking on, generate up to 8k tokens in vLLM recording sampled-token logprobs. Rescore each full sequence in one forward pass (chunked scan). Number: per-position |Δlogprob| and importance ratio as a function of generated position (slope, and fraction of tokens with ratio outside [0.9, 1.1] in the first 256 tokens versus tokens 4k-8k).

**Who cares.** People doing RL post-training on Qwen3.5/3.6-class models; vLLM maintainers on #42259 / #48305 / #54993.

**Closest existing answer.** arXiv 2605.14220 (VeXact): shows mismatch matters and gives a zero-mismatch reference, transformers only, no recurrent state. The general question "does rollout/trainer numerical mismatch matter?" is therefore already answered; only the hybrid, length-dependent part is open.

**Dropped-direction overlap.** D4: same setting (RL rollouts), but about correctness of the logprobs, not rollout speed. Slight D3 (uses the same vLLM hybrid code paths).

---

## Card B. Does GDN state magnitude keep growing with context, and where is the precision floor?

**Question.** Does the recurrent-state magnitude in deployed GDN hybrids (Qwen3.5 / Qwen3.6) keep growing with context length, and at what length, if any, does a bf16 / fp16 state or readout lose enough precision to change the output?

**Tension type.** Unexplained anomaly with competing partial explanations and non-reproductions, plus a disputed default (SSM state dtype).

**Evidence.**
- [V] PR, vLLM #54146 "[Bugfix][Kernel] GDN readout: honor fp32 SSM state precision to avoid fp16 overflow → NaN" (open), https://github.com/vllm-project/vllm/pull/54146 : "The GDN recurrent state is an *unnormalized* accumulator: it grows over long, low-decay context and is normalized only at the readout by `RMSNormGated`. With an fp32 state cache the readout exceeds fp16's 65504 range and overflows to `±inf` at the store; `RMSNormGated` then produces `inf / inf = NaN`, and decode collapses to a single repeated token." Test: "Before: 107 non-finite entries, `max|o| = inf`."
- [V] issue thread, vLLM #55291 "[Bug]: Qwen3.6-27B-FP8 eventually collapses into repeated ! tokens, affecting all subsequent requests" (open), https://github.com/vllm-project/vllm/issues/55291 : "The exact trigger for the first corrupted request is not yet deterministic." Non-reproduction in the same thread (MaCoredroid, 2026-09-18, v0.28.0): "298 requests, 249,544 generated tokens, ~3h16m, zero detected collapses/errors" (peak context 36,496 tokens; filled 128k contexts not tested).
- [V] issue thread, llama.cpp #23577 "Eval bug: MTP with Qwen3.6 27B outputs repeated //// after long session" (open, 34 comments), https://github.com/ggml-org/llama.cpp/issues/23577 : "The server started outputting "///////////////" in a loop, after a very long session (hours)"; another user: "I see this with and without MTP, I also seen it on qwen3.5/3.6 both MOE and dense variants". Reports across CUDA, Vulkan and ROCm; several different causes proposed, none settled.
- [V] paper, Buitrago & Gu, "Understanding and Improving Length Generalization in Recurrent Models", arXiv 2507.02782 (venue believed ICML 2025, not re-checked), https://arxiv.org/abs/2507.02782 : "the official Mamba-2 checkpoint reaches a standard deviation almost twice as large at position t=8192 than the one at position t=2048"; hypothesis: "models fail to length generalize when their recurrence applied to long sequences produces state distributions that have not been explored during training".
- [V] paper, Gated DeltaNet, arXiv 2412.06464 (ICLR 2025 per search result, not re-checked): normalises keys/queries ("we found L2 normalization to be essential for optimal performance"); I found no analysis of state-norm growth with length by grep.

**Why unknown.** The overflow PR gives a mechanism on one backend (AMD MI50), the vLLM collapse thread has a non-reproduction at 36k context, the llama.cpp thread mixes at least three suspected causes. Nobody has published the state-magnitude curve of a deployed Qwen3.5/3.6 checkpoint against context length and content type, nor the length at which a lower-precision state starts to change tokens. Qwen3.6 ships `mamba_ssm_dtype=float32` without a stated reason.

**If yes (grows, with a reachable floor).** There is a context length beyond which bf16 state is unsafe; state dtype becomes a length-dependent policy, and it bears on whether cached state checkpoints can be stored compressed.

**If no (saturates well below any limit).** The collapse reports are ordinary engine bugs, bf16 state is safe at all supported lengths, and the float32 default is costing memory and speed for nothing. This would be a clean negative with a practical consequence, though less of a research direction.

**First experiment (about a week, 1-2 GPUs).** Qwen3.5-4B, Qwen3.5-9B, Qwen3.6-27B-FP8. Feed contexts up to the model limit from four content types (natural text, code, agent tool logs, low-entropy repeated text). Hook every GDN layer; every 1k tokens record state max-abs and Frobenius norm and readout max-abs before `RMSNormGated`. Run the state in fp32 and in bf16. Number: growth curve of state max-abs versus position per content type, and the first position where the bf16-state run's next-token top-1 differs from the fp32-state run in more than 1% of positions (or KL above a fixed threshold).

**Who cares.** Operators of Qwen3.5/3.6 at long context; vLLM / llama.cpp maintainers on the collapse threads; anyone storing or quantizing recurrent state.

**Closest existing answer.** arXiv 2507.02782: state statistics drift with length on Mamba-2 and relate to length generalisation; no GDN hybrids, no numeric-precision interaction, no serving-engine anomaly.

**Dropped-direction overlap.** Mild D2/D3: the student already ran `abl_ssm_bf16` ablations (for speed) and worked on state checkpoints; this asks about numerics, not caching gain. Related to Card A (both depend on whether the state recurrence is contractive in deployed checkpoints).

---

## Card C. Is prior-turn reasoning in the history a net gain or context pollution?

**Question.** Does keeping prior-turn reasoning in the context improve multi-turn task accuracy enough to pay for the longer context, or does it act as context pollution, and does the sign depend on whether the model was trained to see its own reasoning?

**Tension type.** Vendor guidance reversal plus conflicting paper results; harnesses disagree on the default.

**Evidence.**
- [V] vendor model card, Qwen3-8B, https://huggingface.co/Qwen/Qwen3-8B : "**No Thinking Content in History**: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content."
- [V] vendor model card, Qwen3.6-27B, https://huggingface.co/Qwen/Qwen3.6-27B : "Qwen3.6 has been additionally trained to preserve and leverage thinking traces from historical messages." and "This capability is particularly beneficial for agent scenarios, where maintaining full reasoning context can enhance decision consistency and, in many cases, reduce overall token consumption by minimizing redundant reasoning. Additionally, it can improve KV cache utilization, optimizing inference efficiency in both thinking and non-thinking modes." No numbers given.
- [V] paper (preprint), arXiv 2606.00135 "On Effectiveness and Efficiency of Agentic Tool-calling and RL Training", https://arxiv.org/abs/2606.00135 : "retaining thinking history consistently improves Qwen3 models (e.g., ∼3–5% for Qwen3-8B and Qwen3-4B)"; "Takeaway 3.3 Retaining thinking history improves by ∼2–5% on BFCL multi-turn for reasoning models." This is on Qwen3, whose own card says to strip.
- [V] paper (preprint), arXiv 2602.24287 "Do LLMs Benefit From Their Own Words?", https://arxiv.org/abs/2602.24287 : "replacing all prior assistant turns with one-sentence summaries or keeping only the most recent user–assistant exchange often matches storing full context in performance while using roughly 8× less context"; names "context pollution, a phenomenon in which reasoning errors, hallucinations, or stylistic artifacts propagate across turns". Chat conversations, not tool loops.
- [V] PR thread, opencode #28352 (closed), https://github.com/anomalyco/opencode/pull/28352 : "WITH reasoning_content → 97.2% cache hit, WITHOUT → 0% cache hit".
- [V] issue thread, hermes-agent #56004 (closed), https://github.com/NousResearch/hermes-agent/issues/56004 : "the model's chain-of-thought from earlier turns is never fed back to it on subsequent turns, even when the server is configured to preserve it."
- [U] Agent-Omit, arXiv 2602.04284 (ICML 2026 per search summary): selectively omitting thoughts/observations reduces tokens without hurting accuracy. Not opened.

**Why unknown.** The vendor reversed its guidance between Qwen3 and Qwen3.6 without publishing an ablation. One paper finds a gain from retaining reasoning on a model told to strip it; another finds assistant history is mostly dispensable. No source I found measures accuracy and compute together (context growth, cache hits, reasoning tokens saved) with the flag on and off, on a model trained to preserve.

**If yes (net gain).** Reasoning traces are working memory; the follow-up is which parts carry the benefit and whether a compressed form keeps it.

**If no (pollution or wash).** Stripping is right even for models trained to preserve, and the serving problem becomes how to strip without paying a full re-prefill (which on hybrids needs a state checkpoint at the strip point).

**First experiment (about a week, 1-2 GPUs).** 2x2: Qwen3.6-27B-FP8 (trained to preserve) and Qwen3-8B (told to strip), history with and without prior reasoning, same chat template otherwise, on BFCL multi-turn plus 50 multi-step tasks from a public agent benchmark that the student can run locally. Number: task accuracy, generated reasoning tokens per task, prefill tokens actually computed (cache misses) per task, and GPU-seconds per solved task.

**Who cares.** Agent harness authors (opencode, hermes-agent, koboldcpp all changed behaviour here in 2026), serving teams sizing context and cache, the Qwen team.

**Closest existing answer.** arXiv 2606.00135: accuracy on BFCL multi-turn for Qwen3 only; no compute accounting, no trained-to-preserve model, no long-horizon loop. arXiv 2602.24287: chat only, not reasoning traces specifically.

**Dropped-direction overlap.** D2: stripping reasoning rewrites history and breaks the prefix cache; on hybrids the fix is a state checkpoint at the strip point. The question itself is about accuracy versus cost, not cache policy.

---

## Card D. How much agent prefill is recomputation caused by history rewrites?

**Question.** What fraction of prefill work in real agent sessions is recomputation caused by the harness rewriting earlier history (stripped reasoning, re-serialised tool calls, compaction) rather than new tokens, and do public traces such as Mooncake capture it?

**Tension type.** Conflicting measurements: a characterisation paper reports near-ideal cache hits; practitioner threads report full reprocessing every turn.

**Evidence.**
- [V] paper, "Agentic AI Workload Characteristics" (IISWC 2026, author-stated), arXiv 2605.26297, https://arxiv.org/abs/2605.26297 : "Most input tokens are reused across turns, yielding high empirical cache-hit ratios of 84.6–99.5% and making decode dominate 91.0–98.6% of LLM time." Limitations: "Advanced orchestration techniques, including speculative or overlapped tool execution, dynamic context compaction or summarization, and branching or tree-search workflows, are outside the scope of this study. These techniques may change tool latency, context evolution, and KV-cache behavior, and we leave their characterization to future work." One harness (Claude Code) on vLLM.
- [V] issue thread, llama.cpp #22746 "Eval bug: Qwen 3.6 27B forcing full prompt re-processing due to lack of cache data" (closed, 126 comments), https://github.com/ggml-org/llama.cpp/issues/22746 : "multi-turn requests with any rendering drift (tool-call JSON re-serialization, missing reasoning_content, etc.) trip the new Patch 2 fallback every turn — the slot gets wiped and the full prompt re-evaluated at O(N) per turn." Another user in the same thread: "The captured requests were append-only. Consecutive requests had a 100% common prefix relative to the previous request." (and still saw a fallback).
- [V] PR thread, opencode #28352 (closed), https://github.com/anomalyco/opencode/pull/28352 : "An empty string changes the token stream, breaking KV cache prefix matching — observed as 0% cache hits on 196K token prompts in production captures."
- [V] paper (preprint), arXiv 2601.06007 "Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks", https://arxiv.org/abs/2601.06007 : "prompt caching reduces API costs by 41-80% and improves time to first token by 13-31% across providers"; "naive full-context caching, which can paradoxically increase latency". Provider APIs, not self-hosted engines; no rewrite taxonomy.
- [V] issue thread, vLLM #38182 (open), https://github.com/vllm-project/vllm/issues/38182 : prefix-cache hit rate "around 92%" without MTP, "around 71%" with MTP on Qwen3.5 (engine-side cause: last matched block dropped, block size above 1024 tokens).
- Repo fact (student's own result, from the task brief): a perfect checkpoint policy gained only 2.6-5.5% on Mooncake traces.

**Why unknown.** The paper measures one harness without compaction and says so. The threads are single-user captures. The Mooncake trace has block hashes but no text, so it cannot say why a prefix stopped matching. Nobody has split missed tokens by cause across harnesses.

**If yes (rewrites are a large share).** Public traces understate recomputation; the D2 negative result would be a property of the trace, not of agents, and non-append-only history becomes the workload to design for.

**If no (small share).** Agent sessions are effectively append-only once harness bugs are fixed; this confirms D2's negative result on firmer ground. Honest note: this answer is the less interesting one and the IISWC numbers make it plausible for at least one harness.

**First experiment (about a week, 2 GPUs).** Qwen3.6-27B-FP8 on vLLM (TP=2). Run two or three open harnesses (opencode, hermes-agent, one more) on 30-50 tasks from a public coding/terminal benchmark, logging the token ids of every request. Offline, per turn, compute the longest common prefix with earlier requests of the session and label the first divergence (reasoning dropped, tool-call re-serialised, compaction, dynamic system-prompt field, other). Number: share of prefill tokens that are recomputation of already-seen tokens under an ideal exact-prefix cache, by cause and by harness; the same statistic from the Mooncake toolagent trace for comparison.

**Who cares.** Serving-system researchers who evaluate on Mooncake-style traces; harness authors; anyone revisiting hybrid-model checkpointing.

**Closest existing answer.** arXiv 2605.26297 (near-ideal hits for one harness, compaction out of scope) and arXiv 2601.06007 (provider-side cost/TTFT, no cause breakdown).

**Dropped-direction overlap.** D2, directly. It asks whether the workload assumption behind the D2 negative result holds; it would reopen or close that direction rather than start a new one.

---

## Card E. Is decode-phase quantization damage accumulation or per-step error, and does it widen with context?

**Question.** Is the extra damage from quantizing the decode phase caused by trajectory divergence that accumulates over generated steps, or by a per-step error that does not depend on length, and does the gap to full precision widen with context length?

**Tension type.** Paper claims against practitioner anecdotes and one small forum benchmark. This card has the weakest practitioner evidence of the five.

**Evidence.**
- [V] paper (preprint), arXiv 2605.20315 "Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs", https://arxiv.org/abs/2605.20315 : "decoding errors can accumulate over time rather than remaining local"; "prefill quantization errors do not simply grow linearly or explosively with the context prefilling length".
- [V] paper, arXiv 2504.04823 "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models": "8-bit weight-activation quantization preserves accuracy across tasks and model sizes, while 4-bit weight-only quantization or KV cache quantization also achieves near-lossless results (≤1% drop)".
- [V] paper (preprint), arXiv 2604.22577 "QuantClaw": "High-sensitivity tasks (e.g., code, compliance, terminal, safety-critical) suffer notable degradation under NVFP4"; "For smaller models (e.g., Qwen3.5-9B), we observe noticeable performance drops (approximately 3-4%)".
- [V] forum, r/LocalLLaMA, u/OsmanthusBloom, "Qwen3.6-35B-A3B tool calling benchmark: ByteShape vs. Unsloth GGUFs, KV cache quants & long context performance" (read through a snapshot mirror: https://reddit.sentinel-team.org/posts/1u0isbo/snapshots/2026-06-13T02%3A56%3A06.64152Z ): "f16 and q8_0 KV cache quants are practically tied, so q8_0 could be seen as free lunch."; "Long context hurts performance very much, with an average gap of almost 10 points between cp=0.0 and cp=0.5 cases." Long-context scores 140.5 (f16) / 140.1 (q8_0) / 139.3 (q4_0), so the q4_0 gap does not visibly widen. A commenter: "you just don't have enough measurements to say anything meaningful beyond 'longer context = worse'."
- [V] blog (anecdotes only, self-described), https://insiderllm.com/guides/qwen-3-6-q4-quant-coding-agents/ : "broke after roughly 30k context and became unable to handle tools at all"; "there's no public agentic eval that isolates Qwen 3.6 quant levels."
- [V] issue thread, vLLM #54035 "[Bug]: FP8 KV cache causes systematic decode/prefill logprob mismatch on Hopper FA3" (open), https://github.com/vllm-project/vllm/issues/54035 : mismatch at "115 / 224" positions, max |Δlogprob| 0.50, BF16 control 0 mismatches (root cause identified: different block sizes in the two paths).
- [V] blog, Artificial Analysis Endpoint Accuracy Index (2026-08-04), https://artificialanalysis.ai/articles/endpoint-accuracy-index : "some endpoints score 22% on BFCL-500 against 37% for the reference." The article attributes gaps mostly to token limits and reasoning-effort settings, not to quantization.

**Why unknown.** Mix-Quant states that decode errors accumulate; in the text I read I did not find a teacher-forced control that separates accumulation from per-step error (check before relying on this). Practitioners report quantized models breaking at long context; the one forum benchmark that varies both shows no interaction but is too small to conclude.

**If yes (accumulation, widening with length).** Precision should be spent where the trajectory forks, which points to step-selective or position-selective precision rather than phase-level.

**If no (per-step, length-independent).** The "quantized models fall apart at long context" belief is a confound (template, context limit, sampler), and uniform low-bit decode is fine wherever its per-step error is acceptable. Useful mainly as a debunk.

**First experiment (about a week, 1-2 GPUs).** Qwen3.5-9B and Qwen3-8B, bf16 against 4-bit weights (AWQ or GPTQ) and FP8 KV cache. Three arms at context lengths 2k / 8k / 32k / 100k: free-running quantized; quantized model teacher-forced on the bf16 model's own trajectory (per-step KL with no accumulation); bf16 decode on a quantized-prefill cache. Tasks: MATH-500 with distractor context, BFCL multi-turn. Number: teacher-forced per-step KL as a function of decode step and context length, and the accuracy gap between free-running and bf16 at each length.

**Who cares.** Local-model users choosing quant levels for agents; providers serving FP8 / NVFP4 endpoints.

**Closest existing answer.** arXiv 2605.20315 (Mix-Quant): asserts accumulation and exploits it; arXiv 2504.04823: accuracy by bit-width on short reasoning tasks, no context-length axis.

**Dropped-direction overlap.** None (D1 is tangential: both touch which context tokens matter at decode time).

---

## Candidates considered and not turned into cards

- Speculative decoding (MTP) versus prefix caching on Qwen3.5/3.6 (vLLM #38182: hit rate about 92% to about 71%; #43559: accuracy drop about 20%, closed). Mechanism is known (last matched block dropped, block size above 1024 tokens); what is left is engineering and it sits inside D2.
- Greedy decoding loops in reasoning models: reportedly answered by "Wait, Wait, Wait... Why Do Reasoning Models Loop?" (arXiv 2512.12895; [U], not opened).
- "Does rollout/trainer numerical mismatch matter?" for transformers: answered by arXiv 2605.14220. Only the hybrid part is kept (Card A).
- Provider-to-provider accuracy variance: the Artificial Analysis article already names the causes.
- Determinism tax of batch-invariant mode (vLLM #27433: compiled −29.9% throughput on RTX 4090 D, −35.0% on H20; "`matmul_kernel_persistent` alone is 79.4% of BI=1 kernel time"): measured and attributed in the thread; an optimisation task, not a question.
- Hybrid prefix caching "align" mode overhead (vLLM #60008: "costs 11-16% throughput on Nemotron-3.5-Lightning with no cache hits"; #51250): inside D2.
- P/D disaggregation versus chunked prefill; speculative decoding under batching; structured output versus reasoning quality: each appeared to have a published answer (from memory, not re-verified here).

## Things I could not verify

- Yao, Liu et al. rollout-training mismatch paper and blog (JS-only page): not quoted.
- Agent-Omit (arXiv 2602.04284) and arXiv 2512.12895: known only from search summaries.
- Venue labels for arXiv 2506.09501, 2507.02782, 2412.06464 and 2605.14220 were not re-checked.
- The Reddit post was read through a snapshot mirror, not on reddit.com.
- Whether Mix-Quant contains a teacher-forced control (Card E) needs a careful read of the full paper.
- vLLM #55291 and llama.cpp #23577 are still moving; statuses are as of 2026-11-11.
