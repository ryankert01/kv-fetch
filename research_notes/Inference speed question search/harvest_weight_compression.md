# Harvest: weight quantization, pruning, distillation, early exit as speed techniques

Date: 2026-11-11. Read-only literature pass. No GPU jobs, nothing posted, nothing committed.
Output: 5 question cards (full cards are in the structured output; short forms in section 5).

How sources were read. "Full" = arxiv.org/html read through WebFetch (first 100k characters; a
second window for 2609.11716). "Abs" = abstract read through the Semantic Scholar batch API.
Quotes below are copied from those reads. Anything marked "not verified" was not read this session.

## 1. Reading list

All ids were confirmed to exist (arXiv API id_list or Semantic Scholar) in this session.

### Read in full (first 100k characters)

| id | title (short) | why |
|---|---|---|
| 2609.11716 | Why Does Post-Training Quantization Work? | counteraction mechanism |
| 2609.32060 | Depth Laws for the Precision Floor of Trained Neural Networks | disputes cancellation; depth vs noise tolerance |
| 2608.30076 | Budget-Aware Compression Pipeline for Single-GPU LLM Inference | depth pruning + AWQ, order claim |
| 2603.18426 | Prune-then-Quantize or Quantize-then-Prune? (ICLR 2026) | order of compression, Progressive Intensity Hypothesis |
| 2606.00206 | Quantized Reasoning Models Think They Need to Think Longer, but They Do Not | length inflation, noise-at-uncertain-positions story |
| 2606.25519 | Quantization Inflates Reasoning: Token Inflation as a Hidden Cost | CTIR table by quantizer |
| 2611.12127 | Lost in Rounding: bf16 Recurrent State Makes Hybrid Reasoning Models Think Longer | bias, not resolution |
| 2606.15682 | ReQAT (4-bit FP QAT for reasoning) | entropy routing between BF16 and FP4 |
| 2606.13233 | ReSET (step-aware temperature for NVFP4) | entropy shift sign by token class |
| 2607.04640 | Wrong Before Right: Late Rescue and Interface Failure | structural vs low-bit flips dissociate |
| 2611.11695 | Capabilities fail in a shared, predictable order under compression | shared task order; GPTQ exception |
| 2610.00694 | How Divergence Becomes Decision Flips in Compressed LMs | flip rate = TV; draft acceptance |
| 2602.05970 | Inverse Depth Scaling From Most Layers Being Similar | ensemble-averaging view of depth |
| 2603.23701 | The Diminishing Returns of Early-Exit Decoding in Modern LLMs | early-exit potential by architecture |
| 2603.22473 | Component Ablation for Efficient Hybrid LM Architectures | GDN vs attention layer removal |
| 2504.02010 | (compressing R1-style reasoning models: quantization, distillation, pruning) | final-layer modules |
| 2512.01278 | SparseSpec (sparse self-speculative decoding) | sparse-attention draft acceptance |

### Abstract only

Quantization and reasoning length: 2504.04823 (Quantization Hurts Reasoning?), 2606.02011 (2-bit
failure modes, FP16 planning + loop rescue), 2610.29378 (OnQ), 2610.34886 (ThinkBit), 2601.14888
(Reasoning-QAT), 2607.11317 (e-CUSUM), 2610.24346 (precision upgrades in closed-loop agents),
2602.02711 (Dynamic Mixed-Precision Routing), 2610.28537 (Beyond 4 bits), 2604.19884 (two failure
modes of quantization), 2610.07522 (Activation Denoising), 2609.39277 (compressing looped models),
2605.08755 (LAQuant), 2602.13595 (The Quantization Trap), 2510.10964 (Not All Bits Are Equal),
2611.08713 (ResServe), 2609.04098 (Why GDN survives 4-bit), 2505.22179 (SD meets quantization),
2410.11305 (QSpec).

Pruning, depth, early exit: 2403.03853 (ShortGPT), 2403.17887 (Unreasonable ineffectiveness of
deeper layers), 2402.09025 (SLEB), 2602.01997 (limits of layer pruning for generative reasoning),
2611.01133 (pruning via test-time scaling), 2610.30552 (Repair changes which layers to remove),
2505.24680 (LinearPatch), 2605.07271 (Silent vs Decisive phase), 2605.16234 (No Free Swap),
2611.00877 (depth-pruned small LMs), 2609.37062 (vSkipper), 2609.17008 (FlexEE), 2608.28846
(layer-skipping audit), 2610.34103 (T2D), 2609.31098 (effective depth), 2609.16537 (layer
importance in transformers vs SSMs), 2610.01165 (block-bypass over pretraining), 2609.39259 (CFS),
2610.39068 (Closed Gates Leak), 2510.18871 (How do LLMs use their depth), 2606.21906 (Confident
Layer Decoding), 2510.23766 (BitSkip), 2604.18396 (River-LLM), 2509.12464 (RAC), 2610.34336 (CSD),
2511.18864 (SSGR), 2602.09130 (UniComp), 2610.02462 (capability scaling-down laws), 2608.19670
(asymmetric harms), 2607.00908 (perplexity illusion), 2606.07819 and 2606.04063 (joint pruning +
mixed precision).

Anchors (abstract checked, not re-read): GPTQ 2210.17323, AWQ 2306.00978, SpQR 2306.03078, QuIP#
2402.04396, AQLM 2401.06118, BitNet b1.58 2402.17764, LLM-Pruner 2305.11627, SliceGPT 2401.15024,
Sheared LLaMA 2310.06694, LayerSkip 2404.16710, Palu 2407.21118, Judge Decoding 2501.19309.

Own-project data point used (file, not a paper):
/home/ryankert_l/.claude/projects/-home-ryankert-l-adaptfm-contest/memory/project_distillation_pipeline.md

## 2. Extracted claims (exact quotes and numbers)

### Why 4-bit weights work, and what depth does to weight error

- 2609.11716 (full): "Counteraction offsets part of the newly introduced error, and slows the
  hidden-error norm growth." Qwen3-32B: the cross term cancels 50.2% of summed block-update error
  energy; removing counteraction raises final relative hidden error 2.94x, reversing it 8.41x;
  strongest in blocks 17-48 (of 64); the input-error response accounts for 99.9% of the negative
  interaction, under 0.1% from the direct weight effect. Training: "counteraction is weak near
  initialization and becomes more pronounced at later checkpoints." Setting is NVFP4 W4 RTN; no
  pruning, layer removal or random-noise test. Limits: "but not how these mechanisms extend to
  multi-token generation". About 0.28% of positions excluded: "Counteraction is less effective when
  the W4 norm trajectory departs sharply from the original trajectory."
- 2609.32060 (abstract): "Decision margins, cross-layer error cancellation and heavy tails do not
  set the floor." (full, related work): "We test this mechanism in our networks and do not find it."
  (full): "Deeper GPT-2 models are more robust to perturbation, and full-precision amplification
  predicts by how much." GPT-2 noise floor 0.146 (small) to 0.301 (XL). The cancellation test is in
  a section I could not read (page truncated); it appears to be on the paper's own MLP/CNN/ViT nets.
- 2608.30076 (full): abstract "Controlled ablations show that layer-wise pruning makes weight
  quantization more robust."; body "subsequent depth pruning shortens the path over which
  quantization errors accumulate and propagate."; "Quantizing the full model first yields more
  stable scaling and protection." (order argued, no table for the reverse order). R1-Distill-
  Llama-70B, ShortGPT + AWQ W4A16; Table 5 Short-20 ARC-E 0.7222 vs AWQ-20 0.7205.
- 2403.03853 (abstract): "ShortGPT is orthogonal to quantization-like methods".
- 2603.18426 (full): "introduce the Progressive Intensity Hypothesis, which states that weaker
  perturbations should precede stronger ones." Table 1, order advantage of quantize-then-prune,
  SparseGPT / SLEB at 5% pruning with QuaRot: 4-bit -49.899 / -9.379; 5-bit -0.017 / -0.057.
  Theory assumes layer-wise independence; no per-layer analysis of which layers carry the interaction.
- 2602.05970 (full): "functionally similar layers reducing error through ensemble averaging rather
  than compositional learning"; no deletion or noise experiment.
- 2610.39068 (abstract): adjacent-layer cancellation is generic damping; "the previous layer's
  write is not singled out".
- 2610.07522 (abstract): under parallel quantization "quantization errors then compound through
  the residual stream, as no layer corrects for the errors of the layers before it."
- Own data: source AWQ IFEval 0.855; "All depth-pruned AWQ variants: IFEval 0.05-0.20"; 24 of 32
  layers gave 3.58x average speedup (so outputs were also much shorter); no healing before AWQ.
- 2603.23701 (full): "Our results show a diminishing trend in early-exit effectiveness across newer
  model generations." EAS: Llama2-7B 0.52, Llama3-8B 0.46, Qwen2-7B 0.36, Qwen3-8B 0.51,
  GPT-OSS-20B 0.59 (the numbers do not show a clean decline). "post-trained models generally
  exhibit delayed logit alignment compared to the base model." (one Qwen3-4B pair).
- 2510.23766 BitSkip (abstract): quantization and early exit compose poorly, but on 85M models
  trained from scratch.

### Which items and tokens fail under which compression

- 2607.04640 (full): "items with large dips on the intact model are 3-7x more likely to flip under
  genuine late-layer low-rank compression, block dropping, or structured pruning"; "deaths under
  3-bit RTN quantization are dip-blind (item-level rho~0 in all four models)"; "low-bit damage is
  diffuse across depth, so no layer-localized signature should--or does--predict it"; for
  quantization deaths the intact final margin (rho about -0.30 to -0.35) beats the dip (+0.06 to
  +0.12). Minimal pairs, single-token margins; overlap of the two flip sets not reported.
- 2611.11695 (full): "task effects under Wanda, SparseGPT and GPTQ correlate with the same model's
  random-damage task effects with a median of 0.875 over 18 cells."; "the order under GPTQ agreed
  with each model's own random-damage order but not between models" (0.095 vs 0.924 Wanda, 0.927
  SparseGPT); "longer generative tasks degraded in patterns of their own"; "Generation used at most
  32 new tokens".
- 2610.00694 (full): flip rate "tracks total variation at a ratio with median 1.05, with no fitted
  constant."; "Total variation predicted greedy acceptance with mean errors of 1.1-2.4% and ranked
  the drafts at Spearman 0.86-1.00."; "Every statistic is measured under teacher forcing on greedy
  references."; "Nor do per-position statistics say how changed decisions compound over free
  generation." Nine families incl. layer removal and Gaussian weight noise; no overlap or stacking
  analysis; sparse attention not included.
- 2512.01278 (full): "PillarAttn achieves an average acceptance token length of 6.16 out of 8
  tokens"; "both NGram and EAGLE3 can only draft fewer than 2 accepted tokens."; combination with
  other drafts is discussed, not measured; no layer-skip or quantized draft compared.
- 2504.02010 (full): "Final-layer modules, along with the mlp.gate_proj of R1 distilled Llama and
  Qwen, are overly compressed by popular quantization methods."; protecting final-layer MLP
  modules in 3-bit AWQ: 46.0 -> 52.57 with "only about 2% of all weights remain in 16-bit."

### Reasoning length under quantization

- 2606.25519 (full), Table 2, average CTIR, weight-only, g128. Qwen3-4B: RTN INT4 +42.5%; AWQ INT4
  +8.2%, INT3 +276.1%; GPTQ INT4 +12.0%, INT3 +48.7%; HQQ INT4 +6.0%, INT3 +292.1%; ParoQuant INT4
  +4.7%. Qwen3-30B: RTN INT4 +43.0%; AWQ +14.4%; GPTQ +3.7%; HQQ +3.9%. No configuration has a
  negative average. Qwen3-30B GPTQ INT4: accuracy drop 0.2 points with CTIR +3.7%. Mechanism only
  hypothesised: "small perturbations can accumulate across many decoding steps". Single seed,
  greedy, 100 questions per benchmark. Serving: "A 1.1x CTIR increase raises p90 TTFT from 1050 s
  to 1400 s".
- 2606.00206 (full): "3-bit AWQ increases MATH-500 CoT from 5.2K to 23.4K tokens, a 4.5x
  increase."; "Position-level KL divergence correlates strongly with the BF16 model's next-token
  entropy (Spearman rho=0.92)."; rho = -0.73 across 28 model-quantization pairs (accuracy vs CoT
  length); logit penalty on markers cuts CoT length 12-23%; the margin argument "provides intuition
  for one specific failure mode, not a complete causal explanation". No Gaussian or stochastic-
  rounding control; RTN not tested.
- 2504.04823 (abstract): "Contrary to expectations, quantized models do not exhibit increased
  output lengths."
- 2611.12127 (full): "five of the six models reason longer"; "On Qwen3.5-9B, cap hits rise from
  118 to 223 and accuracy falls from 85.6% to 70.8%."; "Same-width bf16 SR removes the length
  increase in four of the five affected models."; "so the failure is mostly rounding bias, not
  missing resolution."; 85% of unfinished bf16 runs on Qwen3.5-9B are exact repetition loops;
  weight quantization not tested; no Gaussian control on real models.
- 2606.15682 ReQAT (full): threshold "tau=0.6, which corresponds to approximately the top 25%
  highest-entropy tokens."; "routing low-entropy prediction to BF16 recovers a large fraction of
  the accuracy lost under quantization"; "routing only high-entropy prediction to BF16 yields
  relatively small improvement"; "quantization-induced failures are dominated by errors at
  low-entropy token prediction, rather than high-entropy token prediction." Length is not reported
  per routing arm; the low-entropy arm is about 75% of tokens.
- 2606.13233 ReSET (full): low-entropy tokens are "locally constrained symbolic decisions such as
  digits and operators."; "these errors are not driven by top-1 flips."; at very high entropy "the
  entropy shift becomes negative in high-uncertainty steps." No oracle replacement; no length result.
- 2610.24346 (abstract): "replacing an action with that of a second 4-bit checkpoint changed about
  as many outcomes as replacing it with the BF16 action".
- 2602.02711 (abstract): "decoding a step at low precision turns a success into a failure at only
  about 1% of the steps".

### Hybrids and depth

- 2603.23701 (full): "dense transformers generally offer greater early-exit potential than
  Mixture-of-Experts and State Space Models."; "SSM further minimize redundancy by tightly coupling
  sequential state updates across depth." (not tested; no numeric dense-vs-SSM table in the text).
- 2603.22473 (full): Qwen3.5-0.8B, "removing linear attention increases perplexity from 7.6 to
  268,337"; removing attention gives 625; the hybrid has "greater normalized tolerance to matched
  random layer removal than the same-family Transformer control."; "Early-layer ablation causes
  substantially more damage than late-layer ablation for both SSM and attention."; hypothesis only:
  "the surviving component type in neighboring layers partially compensates for structural
  disruption, but this mechanism should be validated in additional model families."; sub-1B models,
  no healing, damage levels are catastrophic (thousands-fold perplexity).
- 2609.16537 (abstract): "Necessity and Plasticity anti-align across depth, whereas in the
  evaluated Mamba-style SSMs they point to overlapping regions."
- 2609.31098 (abstract): effective depth below the orthogonal-update reference in fifteen of
  sixteen models "(Qwen3.5: 32--44%, OLMo-2: 40--41%, Pythia: 23--28%)".

## 3. Tensions considered and dropped

1. Realised GPU speed of 4-bit weights (LAQuant 3.42x on A6000, Marlin 1.2-1.4x at batch 1-16,
   ReSET's small-M kernel, The Quantization Trap). Predictable from memory-bound vs compute-bound
   arithmetic, formalised by 2602.13595, and it is profiling (near dropped direction D3).
2. Compression order (prune first or quantize first). Looked like 2608.30076 vs 2603.18426, but
   both fit "weaker perturbation first"; answered by 2603.18426.
3. Depth pruning damage: lost capability or lost sampling reliability (2602.01997 vs 2611.01133).
   2611.01133 measures pass@k directly. Answered.
4. Which positions cost a quantized model accuracy. Answered for accuracy by ReQAT's routing
   experiment. Only the length half is open (kept as card 4).
5. What reasoning-trace calibration data buys (2606.25519 Math-CoT result, RAC, SSGR, RECAP,
   LAQuant, OnQ). Crowded; an expert would predict the direction.
6. "Early exit is dying in newer models" (2603.23701) vs vSkipper gains. The paper's own numbers do
   not show a clean decline; not a real contradiction. Used only as evidence in card 1.
7. Why the GPTQ task order does not transfer across models (2611.11695, 0.095). Most likely a
   small-effect noise floor at 4 bits; expert confidence high. Dropped.
8. Thinking in 4-bit and answering in full precision. Built by 2606.02011 (loop rescue, FP16
   fallback) and 2609.39277 (8-bit finishing loops).
9. Depth for tokens: can a depth-pruned model buy back accuracy with more thinking tokens.
   2602.01997 ("deficits persist even on simple arithmetic tasks") makes "no" the confident guess.
10. Is depth-pruning damage a temperature-like flattening. Greedy pass@1 loss cannot be fixed by
    sharpening; predictable.
11. Pruned or quantized target as drafter. Covered by T2D 2610.34103, QSpec, 2505.22179, and the
    TV law of 2610.00694.
12. IFEval and function calling collapse first under depth pruning. Documented by 2611.00877 and
    UniComp.
13. Sign of the length change under pruning (RAC and CSD say longer, 2611.01133 says shorter
    responses under severe pruning, own run got 3.58x from a 25% layer cut). Most likely severity:
    mild damage lengthens, broken models stop early. Overlaps earlier Q05. Dropped.
14. Final layers "overly compressed" (2504.02010) vs "low-bit damage is diffuse across depth"
    (2607.04640). Real, but first/last-layer sensitivity is old mixed-precision lore; folded into
    card 3 as a possible extra arm (quantize only the last k blocks).

## 4. Prior-answer searches

Semantic Scholar keyword search returned errors for most of the session and the arXiv API search
returned 429, so searches were: citations of the papers each card rests on, plus 6 WebSearch calls.

- Citations pulled (Semantic Scholar): 2609.11716 (only 2609.32060), 2607.04640 (one unrelated),
  2610.00694 (none), 2603.18426 (six, none on depth pruning + counteraction), 2606.00206
  (seventeen, all in the reading list or unrelated), 2606.25519 (two), 2611.12127 (none),
  2609.16537 (none), 2603.23701 (two), 2403.17887 and 2403.03853 (earlier in the session).
- WebSearch 1: weight quantization reasoning length, stochastic rounding or Gaussian noise control.
  Closest: 2606.00206, 2606.02011. No matched-perturbation control found.
- WebSearch 2: depth pruning then quantization, block influence, counteraction. Closest:
  2603.18426, 2606.07819, 2606.04063. None looks at which layers carry both.
- WebSearch 3: overlap of rejected tokens across draft types, compound or cascade self-drafts.
  Nothing direct.
- WebSearch 4: selective full-precision verification by token entropy. Closest: ReQAT 2606.15682
  (accuracy only), DMR 2602.02711 (step level, agents).
- WebSearch 5: layer pruning or skipping in GDN / Mamba hybrids. Closest: 2603.22473 (sub-1B
  ablation), one unverified community report of removing 1 in 3 GDN layers.
- WebSearch 6: quantization x early exit. Closest: BitSkip 2510.23766, River-LLM 2604.18396.

Sibling harvest notes in this folder overlap two of my cards: "Do a sparse-KV self-draft and a
4-bit-weight self-draft fail at the same positions?" (self-speculation and own-projects harvests)
and "Persistent error or per-step error size" / "low-bit damage from bias or from error size"
(own-projects and KV harvests). My versions are the weight-compression ones: depth removal vs
4-bit weights for the first, weight rounding rather than KV or state rounding for the second.
The composition harvest dropped "Depth pruning x quantization" as "Predictable sub-additivity";
card 1 below disagrees and says why (three published positions that do not agree, plus one run of
the user's that collapsed).

## 5. Kept cards (short form)

1. When layers are removed (depth pruning or layer skip), does 4-bit weight error downstream grow
   because those layers were absorbing it, or shrink because there is less depth to accumulate it?
   Contradiction: 2609.11716 (absorber) vs 2609.32060 and 2608.30076 (accumulator / cancellation
   does not matter) vs ShortGPT ("orthogonal"). Guess: mildly worse than additive. Confidence low.
   1.5 GPU-days.
2. Do depth removal and 4-bit weights get the same tokens wrong? Contradiction: 2607.04640
   (item-level dissociation) vs 2611.11695 (shared order). Guess: partial overlap, mostly explained
   by target entropy. Confidence medium. 1 GPU-day.
3. Does 4-bit weight rounding lengthen reasoning through the size of the weight error or through
   its systematic part? 2611.12127 (bias, for recurrent state) vs 2606.00206 (noise at uncertain
   positions), with the RTN vs HQQ gap in 2606.25519 unexplained. Guess: size. Confidence medium.
   1.5 GPU-days.
4. Do the extra reasoning tokens start at the positions that cost accuracy (confident symbolic
   tokens) or at uncertain discourse positions? ReQAT vs 2606.00206; neither measured the other's
   variable. Guess: different positions. Confidence medium. 2 GPU-days.
5. In a GDN-attention hybrid, what decides which depth is removable: layer type or position?
   2603.23701 vs 2603.22473, both weak. Guess: position. Confidence low. 1 GPU-day. Weakest card.

## 6. Not done

- Keyword search of 2025-2026 preprints was limited to citations and six web searches (see 4).
- Paper tails beyond 100k characters were not read (2609.11716 last 25k, 2609.32060 sections 6-7,
  2606.00206, 2610.00694, 2603.18426, 2504.02010, 2606.13233, 2606.15682 appendices).
- Not read beyond the abstract: 2602.01997, 2611.01133, 2610.30552, 2609.16537, 2609.31098,
  2610.24346, 2510.23766 and the anchors.
- Distillation got no card of its own. The distillation papers read (OnQ, Draft-OPD 2605.29343,
  T2D, 2610.02462) pointed to questions that are crowded or predictable.
