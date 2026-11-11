# Question harvest: techniques measured alone, and whether gains compose

Date: 2026-11-11. Lens: a technique is reported in isolation (batch 1, one GPU class, one model family, single turn), and a second technique or serving condition is known to change the regime. The question in each card is what decides whether the gain survives, not how to build the combined system.

Six cards. Quotes are verbatim from the pages I opened (arXiv HTML unless stated). Items I could not open are listed at the end and are marked unverified wherever they are used.

Dropped directions referred to below: D1 decode-time KV page selection for long reasoning outputs; D2 state checkpoints / prefix caching for hybrids in vLLM; D3 profiling vLLM 0.32 single-GPU; D4 accelerating RL rollouts (watermark).

---

## Card 1. Speculative decoding x batch size: what sets the sign?

**Question.** Do one measured cost ratio (verifying k+1 tokens versus one plain decode step at the operating point) and acceptance together predict whether speculative decoding helps or hurts at serving batch sizes, across batch size, context length, weight precision, GPU class and dense versus hybrid models, including whether long context restores the gain on hybrids as it is reported to on dense models?

**Tension.** Published reports disagree on the direction for the same model class, and each varies one axis on its own stack.

Evidence (all opened):
- Meta, "Efficient Speculative Decoding for Llama at Scale", arXiv 2508.08192 (preprint, 2025), https://arxiv.org/html/2508.08192 : "the Llama3.1 8B model exhibits greater speculative decoding speedup at large batch sizes"; "attention dominates the computation even for large batch sizes, and therefore the workload stays memory-bound". Internal engine, H100.
- Liu, Yu, Park, Stoica, Cheung, "Speculative Decoding: Performance or Illusion?", MLSys 2026, https://arxiv.org/html/2601.11580v1 : "Increasing batch size improves absolute throughput but systematically reduces the relative speedup of speculative decoding"; "the speedup of EAGLE over the non-SD baseline decreases from 1.73× to 1.21× as the batch size increases from 1 to 128". vLLM, H100. No quantized targets, no hybrid models, no context-length axis.
- EAGLE-3, NeurIPS 2025, https://arxiv.org/html/2503.01840 : "As batch sizes increase, this redundancy decreases, reducing the effectiveness of speculative sampling." vLLM table: EAGLE 1.30x (b=2) to 0.71x (b=56); EAGLE-3 1.75x (b=2) to 1.01x (b=56). SGLang on H100: EAGLE 0.93x at b=24. (Hardware for the vLLM table is inconsistent in the paper: text says RTX 3090, caption says A100.)
- MagicDec, ICLR 2025, https://arxiv.org/html/2408.11049v5 : "speculative decoding consistently outperforms autoregressive decoding except when batch size is large and sequence length is short"; "The self-implement inference backend is built on GPT-Fast."; "Each run generates 96 tokens per sentence in the batch through greedy decoding on 20 batches". Static homogeneous batches.
- SpecMQuant, "Speculative Decoding Meets Quantization", arXiv 2505.22179 (preprint), https://arxiv.org/html/2505.22179v2 : "the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding"; "EAGLE-2 provides limited benefit for 4-bit weight quantized models (W4A16 and W4A8)". Batch 1 only.
- The Mamba in the Llama, NeurIPS 2024, https://arxiv.org/html/2408.15237 : naive implementation "achieves a decent speedup on 3090 GPUs" but "no speedup at all on H100s".
- ReplaySSM blog post (Tri Dao, 2026; paper is arXiv 2610.13778, preprint), https://tridao.me/blog/2026/replayssm/ : "speculative decoding, a reliable speedup for Transformers, barely helps SSMs at serving batch sizes"; storing a state per speculative token "adds T times more memory traffic per decoding step to an already memory-bound path." Models include Qwen3.5-4B on 1x H100. The post does not measure how context length changes the SD speedup.
- Own measurement, /home/ryankert_l/kv-fetch/kv-fetch/results/profile/NOTEBOOK.md (08:48 entry; vLLM 0.32, A6000-class, 90 AIME prompts, 1024 output tokens, ignore_eos, greedy): speedup at b=1 / 16 / 64: hybrid Qwen3.5-4B MTP k=3 2.31 / 1.86 / 1.18; dense Qwen3-4B EAGLE3 k=3 1.85 / 1.28 / 0.69. Note in the entry: "verifying 64 x (k + 1) tokens is compute-bound either way". Acceptance there is inflated by ignore_eos.

**Why unknown.** The ratio form exists (MagicDec; Scaling Laws for SD, arXiv 2505.07858, "Our work primarily concentrates on dense LLM architectures") but has been checked on static batches, dense models, H100/H800 and research backends. Two explanations now exist for the hybrid loss at batch: ReplaySSM says memory traffic of per-token states on a memory-bound path; the notebook measured the same per-draft-token extra for dense and hybrid at b=64 and read it as compute-bound. Nobody has measured the context-length axis for SD on a hybrid, where 24 of 32 layers cost the same at any context length.

**If yes.** Two cheap measurements tell an operator when to switch SD on, and the conflicting papers become different operating points of one curve. This is the less surprising outcome (a consolidation); its value is the cross-axis check and the hybrid context-length result.
**If no.** The residual names a missing term (state rollback traffic, draft-head KV, scheduler or sampling overhead), each a design target. If hybrids do not recover at long context, SD for hybrids needs a different design from dense SD.

**First experiment.** vLLM 0.32, one RTX A6000 and one 6000 Ada. Qwen3-4B + EAGLE-3 head and Qwen3.5-4B + MTP (k=3). Grid: batch {1, 8, 32, 64, 128} x context at decode {1K, 8K, 32K} x weights {bf16, FP8}. Measure plain step time, verification step time with drafts forced (independent of acceptance), acceptance per proposed token with natural EOS, end-to-end speedup. Numbers: prediction error per cell, and the sign of d(speedup)/d(context) at b=64 for dense versus hybrid.

**Closest existing answer.** MagicDec (ratio analysis; dense; GPT-Fast); Scaling Laws for SD (dense, one model, in-house engine); Liu et al. MLSys 2026 (vLLM, H100, batch axis only); TurboSpec (per-deployment fitted model); ReplaySSM (explains and fixes the hybrid loss on H100/B300, no context axis, no 48 GB cards).
**Overlap.** D3 (reuses the profiling harness; the subject is SD composition, not headroom).

---

## Card 2. Lossy techniques x reasoning length

**Question.** Which lossy inference techniques (weight quantization, KV-cache quantization, KV eviction) make a reasoning model think longer, and is the length change predicted by one property of the perturbation, so that a per-token saving can be converted to an end-to-end saving in advance?

**Tension.** Sources disagree on whether length grows, and for which technique family.

Evidence (all opened):
- Liu et al., "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models", arXiv 2504.04823v2 (preprint, 2025), https://arxiv.org/abs/2504.04823 : "Contrary to expectations, quantized models do not exhibit increased output lengths." but also "under more aggressive low-bit quantization (e.g., W3G128 and W4A4KV4), LLMs tend to produce longer outputs."
- Lotfi, Kirichenko, Li, Liu, "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", arXiv 2606.00206 (2026; a search result lists it as a NeurIPS 2026 oral, venue page not opened), https://arxiv.org/html/2606.00206 : "3-bit AWQ reduces accuracy from 85.6% to 47.0%" while "increasing the average CoT from 5.2K to 23.4K tokens"; Spearman rho = -0.73 across 28 model-quantization pairs. Its KL analysis "is also not specific to quantization". Weight quantization only; no throughput, no KV methods.
- ThinKV, ICLR 2026, https://arxiv.org/html/2510.01290 : "pure quantization can inflate generation length by up to 5.1×"; "eviction-based approaches do not induce such drastic inflation." (I read the sentence but did not confirm which method and bit width give 5.1x.)
- "Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving", MLSys 2025, https://arxiv.org/html/2503.24000 : "compression algorithms tend to increase the response length"; Table 4 (LLaMA-7B) 1.69x (KIVI-4), 1.70x (GEAR-4), 1.55x (H2O-512), 1.76x (StreamingLLM-512); "the lengthier responses produced by these methods can outweigh the throughput speedup benefits". Chat models, 1024-token cap.
- R-KV, NeurIPS 2025, https://arxiv.org/html/2505.24133 : reports "up to 9× larger batch sizes and over 6.6× higher throughput" at 16K; no per-response length change reported.

**Why unknown.** Each source covers one family, on different models, with different length caps. ThinKV says eviction does not inflate; Rethinking-KV reports 1.55-1.76x for eviction on chat models. The Meta paper explains weight quantization and says its analysis is not specific to it, but did not test KV methods.

**If yes.** One predictor gives a rule: net gain = per-token speedup / length ratio; throughput tables for reasoning models would need a length column.
**If no.** Family-specific mechanisms (for example eviction removing an earlier intermediate answer versus quantization flattening next-token distributions) say which family to prefer for reasoning models.

**First experiment.** DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B (thinking), MATH-500 and AIME, cap 32K tokens, 3 seeds. Arms: bf16; FP8 weights; AWQ 4-bit and 3-bit; KV cache FP8 (vLLM); R-KV eviction at two budgets (official HF code). Numbers: mean reasoning tokens, accuracy, per-token throughput at fixed batch, time per solved problem; correlation of length ratio with accuracy drop across families.

**Closest existing answer.** arXiv 2606.00206 (weight quantization, mechanism and mitigation; no KV methods or throughput); arXiv 2504.04823 (lengths for weight/activation/KV quantization; no end-to-end time); Rethinking-KV (chat models, capped).
**Overlap.** D1 in part (KV eviction for reasoning output is one arm).

---

## Card 3. Step-level speculation x concurrent load

**Question.** Does step-level speculation for reasoning (a small model writes steps, the large model only judges them) keep its latency gain when the engine is saturated with concurrent requests, where token-level speculative decoding loses its gain?

**Tension.** Measured one request at a time; token-level speculation is known to lose its gain at batch.

Evidence (all opened):
- SpecReason, NeurIPS 2025, https://arxiv.org/html/2504.07891 : "Inference is performed sequentially: the small and base models take turns, avoiding kernel-level interference."; "We run our evaluations on two NVIDIA A6000-48GB GPUs."; GPU memory "is statically partitioned between the two models."; with token-level SD added, "further latency reductions of 19.4%–44.2%, 30.8%–58.0%, 8.8%–52.2%, and 25.1%–51.8%". No concurrency or throughput results.
- SPEX, OSDI 2026, https://arxiv.org/html/2605.10195 : "The speedup is particularly notable at smaller batch sizes."; "As the batch size increases, the system transitions from being memory-bound to compute-bound."; "SPEX synergizes with token-level speculative decoding, achieving cumulative speedups of up to 4.1×" (alone: MTP about 2.0x, SPEX about 3.1x at batch 1; the product would be about 6.2x, my arithmetic).
- EAGLE-3 (see Card 1): token-level gain falls to 1.01x at b=56 in vLLM.
- PARSE, arXiv 2605.04263 (preprint, 2026), https://arxiv.org/html/2605.04263 : "Parse reaches accuracy on par with SpecReason on every benchmark, and at substantially higher throughput." No concurrency, batch or request-rate conditions reported.

**Why unknown.** Two arguments point opposite ways. Step-level speculation removes large-model tokens instead of spending idle compute, so it could survive saturation. It also splits GPU memory between two models (smaller KV pool for the large one), adds judge calls, and its two models stop taking turns once there are many requests.

**If yes.** Speculation that replaces work composes with batching while speculation that uses idle compute does not: a design rule.
**If no.** The reported gain is a low-load property; the measurement says which term removes it (memory split, judge prefill, interference).

**First experiment.** SpecReason's public code on 2x A6000 with vLLM, QwQ-32B + a 1.5B-class small model, AIME and MATH-500, closed-loop concurrency {1, 8, 32, 64}. Baselines at the same GPU memory: large model alone; large model + token-level SD. Numbers: mean and p95 latency per request, tokens/s, accuracy; ratio of SpecReason latency to base latency at each concurrency.

**Closest existing answer.** SpecReason (no load), SPEX (Tree-of-Thought search, reports shrinkage with batch), PARSE (no load conditions). None measures step-level speculation under concurrent requests.
**Overlap.** None.

---

## Card 4. Thinking-KV eviction x multi-turn reuse

**Question.** Is the thinking-token KV that can be evicted within a turn at little accuracy cost also dispensable for later turns when the history keeps the thinking, or do later turns depend on what within-turn importance scores discard?

**Tension.** Eviction methods for reasoning models are scored within one turn; multi-turn benchmarks show within-query importance does not carry over; newer models are trained to keep thinking in history.

Evidence (all opened):
- ThinKV, ICLR 2026, https://arxiv.org/html/2510.01290 : "it is not directly applicable to settings dominated by long input contexts."; no multi-turn or prefix-caching evaluation.
- SCBench, ICLR 2025, https://arxiv.org/html/2412.10319 : "Sparse decoding methods (sub-O(n) memory) perform well on the first query but lose accuracy in subsequent requests."; "This explains why O(k) KV cache compression methods perform well in single-query tests but fail in follow-ups."
- CommitKV, arXiv 2608.07855 (preprint, 2026), https://arxiv.org/html/2608.07855 : "low attention in the current turn does not imply future irrelevance"; at budget 2048 on Qwen3-14B it "outperforms the best R-KV method by 22.24%". SGLang.
- Qwen3.6-35B-A3B model card (2026), https://huggingface.co/Qwen/Qwen3.6-35B-A3B : "By default, only the thinking blocks generated in handling the latest user message is retained"; "Qwen3.6 has been additionally trained to preserve and leverage thinking traces from historical messages."; "Additionally, it can improve KV cache utilization". No measurements.
- vLLM forum thread (2026-09-10), https://discuss.vllm.ai/t/decode-kv-writeback-can-multi-turn-chat-preserve-cache-reuse-with-default-templates/2955 : "Identical final-answer text is insufficient because its KV was computed with a different preceding context."

**Why unknown.** ThinKV and R-KV evaluate one turn. SCBench uses non-reasoning models and long shared inputs. CommitKV treats agent history as a whole and does not separate thinking from other history, nor compare thinking kept versus stripped.

**If yes (dispensable).** Preserved thinking can be held at a small budget: confirmation rather than surprise, but it removes the memory objection to preserve_thinking.
**If no.** Within-turn importance is the wrong signal for retention across turns; points to scoring by later-turn use.
Limit stated honestly: with the default template (thinking stripped) on a dense model the stake is small, since only the last answer is recomputed. The stake is in preserved-thinking agents and hybrids.

**First experiment.** Qwen3.6-35B-A3B FP8 on one 48 GB card, preserve_thinking on. Two-turn tasks built from MATH-500 / AIME (turn 2 asks about or builds on a step of turn 1), plus SCBench multi-turn subsets. Conditions for the turn-1 thinking: full KV; evicted to two budgets with R-KV scoring; stripped. Numbers: turn-2 accuracy, turn-2 reasoning tokens, KV bytes held between turns.

**Closest existing answer.** CommitKV (preprint), SCBench, KVzip (query-agnostic compression, NeurIPS 2025).
**Overlap.** D1 and D2.

---

## Card 5. Speculative decoding x memory pressure

**Question.** When an engine is KV-memory-bound and preempting, is speculative decoding's effect on throughput governed by what it does to per-request memory and residency time rather than by its per-step speedup, and does that make it sub-additive with other fixes for memory pressure?

**Tension.** A measured anomaly on the student's node, against papers that state SD has no memory side or that exclude preemption.

Evidence:
- Own measurement, /home/ryankert_l/kv-fetch/kv-fetch/results/profile/NOTEBOOK.md (19:42 / 22:18 entries; vime rollout step, Qwen3.5-4B, one run each): no MTP, 1 GPU: watermark 0 = 4,755 s with 5,531 preemptions, watermark 0.05 = 3,960 s with 530 (-16.7%); MTP, 1 GPU: 4,327 s with 2,589 vs 4,275 s with 176 (-1.2%); MTP TP2: -6.7%. "the no-MTP engine runs 47 requests, the MTP engine 21.6". "I don't have a checked explanation for why the watermark gains less once MTP is on".
- TurboSpec / SmartSpec, arXiv 2406.14066v3 (preprint), https://arxiv.org/html/2406.14066v3 : proposal length "has no impact on the key-value (KV) cache size"; "the combined memory footprint of the draft model and its KV cache remains consistently below 4% of total memory usage". Preemption not considered.
- Liu et al., MLSys 2026, https://arxiv.org/html/2601.11580v1 : reasoning runs "can trigger request preemption once the KV cache becomes full" and the study avoids that condition; "we do not account for the overhead of maintaining the KV cache for EAGLE's proposing head".
- ReplaySSM blog (2026), https://tridao.me/blog/2026/replayssm/ : "the baseline's preallocated states cut the maximum decode batch by roughly 4× relative to standard serving."

**Why unknown.** For hybrids the concurrency cost of per-token states is now published (ReplaySSM) and matches the notebook. Not explained: why the watermark's gain falls from 16.7% to 1.2% once MTP is on, and what happens on a dense model, where SD costs no per-request KV but shortens residency. SD studies hold memory slack.

**If yes.** Under saturation, SD should be judged by memory-time per request; gains from SD and from admission control overlap and should not be added.
**If no.** Per-step speedup carries over, batch-size studies are enough, and the anomaly is specific to hybrid state blocks (a boring answer, already mostly covered by ReplaySSM).
Confidence is low: one run per cell so far.

**First experiment.** vLLM 0.32, 2x2x2 with 3 repeats: {Qwen3-4B + EAGLE-3, Qwen3.5-4B + MTP} x SD {on, off} x watermark {0, 0.05}, AIME long-output workload sized so KV usage sits at 96-98% with requests waiting. Numbers: wall time, preemptions, mean running requests, recomputed tokens; the interaction term (watermark gain with SD minus without) against the repeat spread.

**Closest existing answer.** TurboSpec (goodput under load, no memory saturation); Liu et al. MLSys 2026 (excludes preemption); ReplaySSM (hybrid concurrency cost).
**Overlap.** D4 (same watermark runs) and D2.

---

## Card 6. Early exit x quantization

**Question.** Does confidence-based early exit recover the accuracy and length that quantization costs a reasoning model, or does quantization damage the confidence signal that early exit relies on?

**Tension.** One paper shows quantized models often pass through the right answer; early-exit methods were evaluated on full-precision models only.

Evidence (both opened):
- arXiv 2606.00206 (see Card 2), https://arxiv.org/html/2606.00206 : "in up to 52% of the quantized models' failures, models reach the right answer in intermediate reasoning steps"; its logit penalty "can reduce CoT length by 12--23%".
- DEER, ICLR 2026, https://arxiv.org/html/2504.15895 : "original DEER reduces the latency by 27.9% to 40.1%" (HuggingFace transformers, full precision); trial answers add cost: "This overhead diminishes the efficiency gains achieved through substantial reduction of generated CoT sequences."

**Why unknown.** The Meta paper's mitigation penalises continued thinking; it does not test exit on answer confidence. DEER does not test quantized models. Quantization may flatten the trial-answer distribution, which would remove the exit signal exactly where it is needed.

**If yes.** The pair is super-additive: a quantized model with early exit may match full precision at lower cost.
**If no.** Quantization harms calibration more than reasoning; points to exit signals that do not rely on one-step confidence, or a full-precision judge.

**First experiment.** DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B at bf16, AWQ 4-bit, AWQ 3-bit; MATH-500 and AIME; DEER official code. Numbers: accuracy, mean CoT tokens, early-stop rate, AUROC of trial-answer confidence against correctness per precision; share of the quantization accuracy loss recovered.

**Closest existing answer.** arXiv 2606.00206 (logit penalty on quantized models); DEER (full precision). Search results only, not opened: TRACE (ACL Findings 2026, reported as saying single-step confidence exits score 0.44 vs 0.55 for full reasoning on hard math), a probe-based stopping paper (arXiv 2504.05419), EAT (arXiv 2509.26522).
**Overlap.** None.

---

## Considered and set aside
- SD for test-time scaling measured at batch 1 (SpecTTS-Bench, ICLR 2026, https://arxiv.org/html/2509.04474 : "apply greedy/stochastic decoding with a batch size of 1", single RTX A6000) although Best-of-N is itself a batch. Folded into Card 1; the ranking of a CPU suffix drafter against EAGLE-3 under batch would be one extra row there.
- KV-compression throughput "from larger batches" inside a paged engine: near D1 and the answer is predictable (R-KV: "When the batch size is 1, R-KV exhibits a slight throughput advantage over FullKV.").
- DEER token savings x batching alone: answer likely boring.
- KV quantization x SD alone: folded into Card 1 (SpecMQuant names "long-context tasks with KV cache quantization" as not assessed).

## Sources opened
arXiv HTML: 2503.01840 (EAGLE-3), 2408.11049v5 (MagicDec), 2406.14066v3 (TurboSpec), 2601.11580v1 (Liu et al. MLSys 2026), 2508.08192 (Meta), 2505.07858v1 (Scaling Laws for SD), 2505.22179v2 (SpecMQuant), 2408.15237 (Mamba in the Llama), 2509.04474 (SpecTTS-Bench), 2503.24000 (Rethinking KV), 2510.01290 (ThinKV), 2505.24133 (R-KV), 2504.04823v2 (Liu et al. quantized reasoning), 2606.00206, 2412.10319 (SCBench), 2608.07855 (CommitKV), 2504.07891 (SpecReason), 2605.04263 (PARSE), 2605.10195 (SPEX), 2504.15895 (DEER). Other: Qwen3.6-35B-A3B model card; vLLM forum thread 2955; Kai Dang's blog post on chat templates and prefix caching; Tri Dao's ReplaySSM post. Local: results/profile/NOTEBOOK.md and the three catalogues under research_notes/Reproducible LLM systems paper setups/.

## Not verified
- ReplaySSM paper (arXiv 2610.13778): abstract page and the blog post only; the full paper was not read.
- ThinKV "up to 5.1×": sentence read, method and bit width behind it not confirmed.
- NeurIPS 2026 venue page for arXiv 2606.00206: not opened.
- TRACE (ACL Findings 2026), arXiv 2504.05419, EAT: search results only (the ACL PDF fetch failed).
- DEER Appendix H (batched throughput, if any): not read.
- Modal "Spec Dec Roofline Model" page and "SpecDecode-Bench" blog figures: not opened, not used.
- EAGLE-3 vLLM table hardware: the paper is inconsistent (RTX 3090 in text, A100 in caption).
- Own MTP / watermark numbers: one run per cell; acceptance in the 08:48 table is inflated by ignore_eos.
- Quotes are as returned by the fetch tool's extraction of each page; I did not diff them against the PDFs.
