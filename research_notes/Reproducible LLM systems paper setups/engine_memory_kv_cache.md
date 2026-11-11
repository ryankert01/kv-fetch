# Memory management inside LLM inference engines (KV cache, offload, prefix caching, GPU memory, model swapping, hybrid-model state) — top-venue papers 2024-2026

Notes on method: facts below were read from the papers' PDFs (USENIX/arXiv/author copies) and from the venues' own program pages. "Venue verified" means the venue's program/accepted list or the proceedings cover page / running header of the PDF names the venue. Anything else is marked unverified. Reproducibility verdicts are the researcher's inferences and are kept in the Inferences sub-sections, not mixed with cited facts.

## Key question 1: Which top-venue papers (2024-2026) address engine-level memory management, and is the venue confirmed?

### Takeaway
Twenty papers with venue evidence were catalogued in detail (OSDI'26 x4, SOSP'25 x3, NSDI'26 x1, FAST'26 x2, FAST'25 x2, EuroSys'26 x1, EuroSys'25 x3, ASPLOS'25 x1, MLSys'25 x2, ICML'25 x1 unverified), plus eight shorter secondary entries (SIGCOMM'24, OSDI'24 x2, SOSP'25, FAST'26, NSDI'26, ICLR'25, ASPLOS'26). OSDI'26 has a whole session named "KV Cache and Long Context", and FAST'26 has an "AI and LLMs" session with three KV-cache/model-loading papers.

### Cited Findings

#### Venue program evidence
- OSDI'26 technical sessions list a session "KV Cache and Long Context" containing Strata, ECHO, "No Buffer, No Bottleneck" (DirectKV), "Simple Is Better: Multiplication May Be All You Need for LLM Request Scheduling", and Prism — [Source](https://www.usenix.org/conference/osdi26/technical-sessions)
- NSDI'26 technical sessions list DroidSpeak, HydraServe and FastServe — [Source](https://www.usenix.org/conference/nsdi26/technical-sessions)
- FAST'26 technical sessions list, in session "AI and LLMs I", SolidAttention, CacheSlide, Bidaw, and "Accelerating Model Loading in LLM Inference by Programmable Page Cache" — [Source](https://www.usenix.org/conference/fast26/technical-sessions)
- The SOSP'25 accepted-papers list includes DiffKV, Jenga, Aegaeon, PrefillOnly, IC-Cache, Pie, KTransformers, Mercury, PhoenixOS and LithOS — [Source](https://sigops.org/s/conferences/sosp/2025/accepted.html)
- The ASPLOS'26 program lists REPA (reconfigurable PIM for KV cache offloading), SpeContext, ZipServ, PAT and "I/O Analysis is All You Need" — [Source](https://www.asplos-conference.org/asplos2026/program/)

#### Main catalogue (detail in Key question 2)
| # | Paper | Venue, year | Venue evidence | Topic bucket |
|---|---|---|---|---|
| 1 | Strata: Hierarchical Context Caching for Long Context Language Model Serving | OSDI 2026 | program page + proceedings PDF | KV offload hierarchy (GPU/DRAM/SSD) |
| 2 | ECHO (KV offloading for native sparse attention) | OSDI 2026 | program page + proceedings PDF | KV offload to host memory |
| 3 | No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs (DirectKV) | OSDI 2026 | program page + proceedings PDF | KV offload to host memory |
| 4 | Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning | OSDI 2026 | program page + proceedings PDF | memory sharing between models |
| 5 | DiffKV | SOSP 2025 | title on accepted list; facts read from arXiv version | KV compression + GPU memory manager |
| 6 | Jenga | SOSP 2025 | title on accepted list; facts read from arXiv version | GPU memory allocation / fragmentation |
| 7 | Aegaeon | SOSP 2025 | accepted list + PDF header "SOSP '25, October 13-16, 2025, Seoul" | model swapping / GPU pooling |
| 8 | DroidSpeak | NSDI 2026 | program page + proceedings PDF | KV reuse across models |
| 9 | CacheSlide | FAST 2026 | program page + proceedings PDF | prompt caching with shifting positions |
| 10 | Bidaw | FAST 2026 | program page + proceedings PDF | KV offload to host memory + SSD |
| 11 | KunServe | EuroSys 2026 | PDF header only ("EUROSYS '26, April 27-30, 2026, Edinburgh"); program page returned 404 | GPU memory overload under bursts |
| 12 | Marconi: Prefix Caching for the Era of Hybrid LLMs | MLSys 2025 | PDF footer "Proceedings of the 8th MLSys Conference, Santa Clara, CA, USA, 2025"; mlsys.org virtual page exists | state caching for hybrid/SSM models |
| 13 | Mooncake | FAST 2025 | proceedings PDF | KV-centric disaggregated serving |
| 14 | IMPRESS | FAST 2025 | proceedings PDF | prefix KV on SSD |
| 15 | HCache | EuroSys 2025 | PDF header "EuroSys '25, March 30-April 3, 2025, Rotterdam" | state restoration from storage |
| 16 | Pensieve | EuroSys 2025 | PDF header "EuroSys '25, March 30-April 3, 2025, Rotterdam" | multi-turn KV caching GPU/CPU |
| 17 | vAttention | ASPLOS 2025 | PDF header "ASPLOS '25, March 30-April 3, 2025, Rotterdam" | GPU virtual memory for KV |
| 18 | Rethinking Key-Value Cache Compression Techniques for LLM Serving | MLSys 2025 | PDF footer "Proceedings of the 8th MLSys Conference ... 2025" | KV compression (measurement study) |
| 19 | Compute Or Load KV Cache? Why Not Both? (Cake) | ICML 2025 (unverified) | no venue line in the arXiv PDF | KV load vs recompute |
| 20 | CacheBlend | EuroSys 2025 (unverified from PDF/venue page) | no venue line in the arXiv PDF | KV reuse for multi-chunk contexts |

- Sources for the rows above: Strata — [Source](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang); ECHO — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda); DirectKV — [Source](https://www.usenix.org/conference/osdi26/presentation/luo); Prism — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan); DiffKV — [Source](https://arxiv.org/abs/2412.03131); Jenga — [Source](https://arxiv.org/abs/2503.18292); Aegaeon — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf); DroidSpeak — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan); CacheSlide — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang); Bidaw — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng); KunServe — [Source](https://arxiv.org/abs/2412.18169); Marconi — [Source](https://arxiv.org/abs/2411.19379) and [MLSys page](https://mlsys.org/virtual/2025/3360); Mooncake — [Source](https://www.usenix.org/system/files/fast25-qin.pdf); IMPRESS — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf); HCache — [Source](https://arxiv.org/abs/2410.05004); Pensieve — [Source](https://arxiv.org/abs/2312.05516); vAttention — [Source](https://arxiv.org/abs/2405.04437); Rethinking KV compression — [Source](https://arxiv.org/abs/2503.24000); Cake — [Source](https://arxiv.org/abs/2410.03065); CacheBlend — [Source](https://arxiv.org/abs/2405.16444)

#### Secondary entries (shorter records in Key question 2)
- CacheGen, SIGCOMM 2024 (PDF header "SIGCOMM '24, August 4-8, 2024, Sydney") — [Source](https://arxiv.org/abs/2310.07240)
- InfiniGen, OSDI 2024 (venue not confirmed from the venue page in this pass) — [Source](https://arxiv.org/abs/2406.19707)
- ServerlessLLM, OSDI 2024 (venue not confirmed from the venue page in this pass) — [Source](https://arxiv.org/abs/2401.14351)
- PrefillOnly, SOSP 2025 (title on accepted list) — [Source](https://sigops.org/s/conferences/sosp/2025/accepted.html)
- Accelerating Model Loading in LLM Inference by Programmable Page Cache, FAST 2026 — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yubo)
- HydraServe, NSDI 2026 (cold start; mostly cluster-level, so only noted) — [Source](https://www.usenix.org/conference/nsdi26/technical-sessions)
- SCBench, ICLR 2025 (PDF header "Published as a conference paper at ICLR 2025") — [Source](https://arxiv.org/abs/2412.10319)
- HILOS, near-storage KV processing for offline batched inference; ASPLOS 2026 per search result only (unverified against the program page) — [Source](https://arxiv.org/abs/2502.09921)

### Inferences
- The 2026 crop (OSDI'26, FAST'26, NSDI'26, EuroSys'26) is dominated by KV offload/hierarchy and multi-model memory sharing; KV compression inside a serving engine appears in SOSP'25 (DiffKV) and MLSys'25 (measurement study).
- Only one peer-reviewed paper in the catalogue targets hybrid/SSM state caching directly (Marconi, MLSys'25); Jenga (SOSP'25) covers Mamba layers as one of several heterogeneous-memory cases.

### Gaps
- USENIX ATC'26 and EuroSys'26 program pages returned 404 when fetched, and the MLSys'26 papers page loads dynamically and returned no content; papers at those three venues may be missing from this catalogue. KunServe's EuroSys'26 venue rests on its PDF header only.
- ICML'25 (Cake), EuroSys'25 (CacheBlend), OSDI'24 (InfiniGen, ServerlessLLM) and ASPLOS'26 (HILOS) were not confirmed on the venues' own pages in this pass; treat as unverified.
- VLDB, SC, HPCA/MICRO/ISCA, NeurIPS and ICLR 2026 programs were not swept systematically; SolidAttention (FAST'26), SpeContext, ZipServ, PAT and REPA (ASPLOS'26), IC-Cache, Pie and KTransformers (SOSP'25) are listed on program pages but their papers were not read.
- No paper on state caching for linear-attention or Qwen3-Next-style (gated delta network) hybrids at a peer-reviewed venue was found; only engine blog posts and pull requests (see Key question 2, Marconi follow-ups).

## Key question 2: Per-paper record (problem, evaluation setup, stated limitations, follow-ups)

### Takeaway
Most evaluation setups reuse a small public set of workloads (ShareGPT, LongBench/LooGLE/NarrativeQA, BurstGPT, SWE-bench agent traces) with Poisson arrivals, vLLM or SGLang as the baseline, and TTFT / throughput / SLO attainment as headline metrics; the hardware ranges from a single 48 GB or 80 GB GPU (DiffKV, IMPRESS, Bidaw, CacheSlide, Rethinking-KV) to 16-32 H100/H800-class GPUs (Prism, Aegaeon, KunServe, Mooncake). Several papers state no explicit limitations section; where none was found this is recorded rather than filled in.

### Cited Findings

#### 1. Strata — OSDI 2026
- Links: paper page https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang ; arXiv 2508.18572. Implemented "as part of SGLang and deployed in production"; no separate artifact repository was identified — [Source](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
- Problem: hierarchical context caching (GPU HBM / host DRAM / SSD) for long-context serving becomes I/O-bound: fragmented KV layouts cause many small transfers, cache loading stalls prefill, and cache hits are delayed. Affected: providers serving long shared contexts (documents, multi-turn) — [Source](https://arxiv.org/abs/2508.18572)
- Models: Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct-1M (one GPU each), Llama-3.1-70B-Instruct (4 GPUs, tensor parallel) — [Source](https://arxiv.org/abs/2508.18572)
- Workloads (public datasets): LooGLE Wikipedia portion (avg input 21,613 tokens; 105 contexts; 2,410 queries), NarrativeQA (54,797; 50; 1,461), ReviewMT (17,708; 100; 1,092), ShareGPT (avg 680.9; 200,869 queries) — [Source](https://arxiv.org/abs/2508.18572)
- Hardware: node with 8x H200 (NVLink, 1.6 TB DRAM, PCIe 5.0 x16 at 64 GB/s); storage node with 8x H20 plus Intel P5510 NVMe (7 GB/s); GH200 node (1 H100 + Grace CPU, 464 GB LPDDR5X) — [Source](https://arxiv.org/abs/2508.18572)
- Baselines: vLLM v0.8.5, vLLM + LMCache v0.2.1, TensorRT-LLM v0.17.0, TRT-LLM-HiCache, SGLang v0.4.5, SGLang-HiCache (the authors' reimplementation in line with CachedAttention / Pensieve / FlashGen) — [Source](https://arxiv.org/abs/2508.18572)
- Headline: up to 5x throughput over vLLM + LMCache and 3.75x over TensorRT-LLM on long-context benchmarks — [Source](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
- Stated limitations (Discussion): GPU-assisted I/O "trades a small amount of SM contention"; cache-aware batching "can still treat requests unevenly, risking SLO violations for individual requests" (SLO-/fairness-aware batch formation and hiding prefetch latency from slower tiers named as open); "Strata targets hierarchical caching for the dense KV caches produced by standard attention. As hybrid mechanisms such as sparse and linear attention gain adoption, extending the same level of support to these models becomes increasingly important and raises distinct challenges in both memory management and scheduling-policy design." — [Source](https://arxiv.org/abs/2508.18572)
- Follow-ups / crowdedness: ECHO (OSDI'26) lists Strata, ArkVale, InfiniGen, FreeKV, CLO and SparseServe as prior offloading systems in its Table 1 — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda). arXiv-only 2026 items found by search (not peer-review-verified): KVDrive, a GPU/DRAM/SSD multi-tier KV system — [Source](https://arxiv.org/abs/2605.18071); Swarm, multi-SSD KV offload — [Source](https://arxiv.org/abs/2603.17803); a 2026 survey covering 30+ KV-management systems — [Source](https://arxiv.org/abs/2607.02574)

#### 2. ECHO — OSDI 2026
- Links: paper https://www.usenix.org/conference/osdi26/presentation/liu-guangda ; code https://github.com/sjtu-zhao-lab/ECHO ; built atop SGLang and DeepGEMM — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Problem: for LLMs with native sparse attention (DeepSeek Sparse Attention), KV size still grows with context and HBM capacity caps concurrency; offload KV to host memory and prefetch the selected tokens. Affected: operators of very large long-context models — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Model: 4-bit quantized DeepSeek-V3.2-Exp (671B MoE) — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Workloads: 318 requests of 80K-100K tokens from InfiniteBench (about 26M input tokens); ShareGPT for short contexts — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Hardware: single node, 8x H20 (96 GB each), PCIe Gen5, 1.5 TB DRAM; host KV pool of 1.8M tokens, about 1,000 GB host memory — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Baselines: vLLM v0.11.1, SGLang v0.5.4. Headline: up to 2.1x generation throughput (2.15x vs SGLang and 4.1x vs vLLM at infinite request rate) — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Stated limitations: "DSA is currently the only publicly available native sparse attention validated on large-scale models"; prefetching relies on properties of the DSA indexer; co-designing sparse attention with prefetch-friendly selection is left "to future work"; "offloading introduces latency overhead", making ECHO "most attractive for throughput-oriented long-context serving" — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Follow-ups: none found (published July 2026).

#### 3. DirectKV ("No Buffer, No Bottleneck") — OSDI 2026
- Links: paper https://www.usenix.org/conference/osdi26/presentation/luo ; code https://github.com/shutianluo/DirectKV ; standalone prototype (about 5,300 lines of CUDA/C++ on FlashAttention-3 with PyTorch bindings), described as complementary to rather than integrated into vLLM/SGLang — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Problem: offloading KV to CPU memory needs GPU staging buffers and double transfers; attention kernels should read host-resident KV zero-copy. Affected: long-context serving on memory-limited GPUs — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Models: Llama-3.1-8B, OPT-13B, OPT-30B. Workloads: ShareGPT, Alpaca; Poisson arrivals up to 30 req/s; contexts 1K-32K — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Hardware: NVIDIA GH200 (96 GB HBM3, NVLink-C2C CPU-GPU link); an H100 on PCIe Gen5 for comparison — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Baselines: SGLang (HBM-resident KV), Pie, FlexGen, Neo. Headline: transfer volume -50%, GPU memory -43%, end-to-end up to 1.2x; at 30 req/s on Llama-8B, latency 0.75 s vs 1.55-2.95 s for other offloading systems — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Stated limitations: "on PCIe-only platforms, DirectKV may still reduce GPU memory usage, but throughput benefits are often limited by interconnect bandwidth"; "main limitations are context-parallel scale-out and host-memory pressure: coordinating remotely owned KV blocks or integrating with a multi-tier KV hierarchy would require additional runtime support, and zero-copy benefits diminish once accesses fall back to pageable or disk-backed tiers. We leave these extensions to future work" — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Follow-ups: none found (published July 2026). Baseline Pie is itself a SOSP'25 paper — [Source](https://sigops.org/s/conferences/sosp/2025/accepted.html)

#### 4. Prism — OSDI 2026
- Links: paper https://www.usenix.org/conference/osdi26/presentation/yu-shan ; arXiv 2505.04021 ; memory component kvcached https://github.com/ovg-project/kvcached ; built on SGLang (about 10,400 lines Python + 774 C++, a 22-line SGLang change, CUDA virtual memory management APIs) — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
- Problem: inference providers host many low-volume models; time-sharing thrashes and static space-sharing starves KV memory. Prism balloons GPU memory between co-located models on demand — [Source](https://arxiv.org/abs/2505.04021)
- Models: 58 LLMs (43 of 1B-3B, 8 of 4B-8B, 3 of 9B-30B, 4 of 31B-70B) — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
- Traces: Hyperbolic (24 models, 4 months) and Arena-Chat (Chatbot Arena, 84 models, 11 days); trace analysis also uses Novita (16 models) and Arena-Battle (129 models) — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
- Hardware: 4 nodes x 8 H100-80G (32 GPUs), 100 Gbps Ethernet, 1.7 TB DRAM — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
- Baselines: static partitioning, MuxServe++ (the authors' SGLang port), QLM, ServerlessLLM. Headline: up to 3.3x higher TTFT SLO attainment; more than 2x cost reduction or 3.5x more requests at the same SLO attainment; kvcached reported deployed on 10K+ GPUs — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
- Stated limitations: no dedicated limitations section was found by text search of the PDF.
- Follow-ups: none found (published July 2026). Same problem space as Aegaeon (SOSP'25) and MuxServe/ServerlessLLM baselines — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)

#### 5. DiffKV — SOSP 2025
- Links: arXiv https://arxiv.org/abs/2412.03131 ; code https://github.com/zyqCSL/DiffKV ; implemented on vLLM (about 4.5K lines CUDA/C++ and 9K Python) — [Source](https://arxiv.org/abs/2412.03131)
- Problem: KV compression treats keys, values, tokens and heads uniformly; differentiated compression yields irregular per-request memory that needs an on-GPU memory manager with parallel compaction. Affected: serving under KV-memory-bound batch sizes — [Source](https://arxiv.org/abs/2412.03131)
- Models: Llama3-8B/70B, Qwen2.5-7B/32B, QwQ-32B, R1-Distill-Qwen-14B, R1-Distill-Llama-8B — [Source](https://arxiv.org/abs/2412.03131)
- Benchmarks (all public): MMLU, MMLU-Pro, GSM8K, MATH, HumanEval+, MBPP+, LongBench, AIME24, GPQA — [Source](https://arxiv.org/abs/2412.03131)
- Hardware: NVIDIA L40 48 GB GPUs (1 GPU for 7B/8B models, 2 for 32B, 4 for Llama3-70B) — [Source](https://arxiv.org/abs/2412.03131)
- Baselines: vLLM, Quest, SnapKV. Headline: KV compressed 2.7x-5.7x with near-lossless accuracy; throughput 1.9x-5.4x — [Source](https://arxiv.org/abs/2412.03131)
- Stated future work: more than two precision levels ("we leave the exploration of such flexible policies for future work"); "future models employing more aggressive training-time KV compression (e.g., architectures with higher queries-per-KV ratios in GQA) may benefit from an additional high-precision level ... which we leave to future investigation" — [Source](https://arxiv.org/abs/2412.03131)
- Follow-ups: the MLSys'25 measurement study (entry 18) evaluates the earlier KV-compression algorithms (KIVI, GEAR, StreamingLLM, H2O) in serving settings — [Source](https://arxiv.org/abs/2503.24000). No 2026 paper using DiffKV as a baseline was found in this pass.

#### 6. Jenga — SOSP 2025
- Links: arXiv https://arxiv.org/abs/2503.18292 ; no code link found in the PDF; implemented on vLLM (about 4,000 lines of Python, no CUDA kernel changes, stated "compatible with all the 90 models in vLLM v0.6.4") — [Source](https://arxiv.org/abs/2503.18292)
- Problem: heterogeneous embedding sizes and attention types (vision encoders, sliding window, Mamba layers, cross-attention) fragment a fixed-page KV allocator and need layer-specific caching and eviction. Affected: engines serving multimodal and non-uniform architectures — [Source](https://arxiv.org/abs/2503.18292)
- Models: Llama vision, Gemma-2, Ministral, Jamba (smallest version 52B), a reimplemented Character.ai-style model, PyramidKV, LLaVA-OneVision 7B, InternVL2 8B, Phi-3 Vision 4B, Paligemma2 10B, plain Llama. Datasets: MMLU-pro, MMMU-pro, arXiv-QA — [Source](https://arxiv.org/abs/2503.18292)
- Hardware: H100 80 GB (default) and L4 24 GB — [Source](https://arxiv.org/abs/2503.18292)
- Baselines: vLLM v0.6.3 end to end; memory-manager breakdown against vLLM, SGLang and TGI. Headline: GPU memory utilization improved by up to 79.6%; throughput up to 4.92x (1.80x average) on H100 and up to 3.29x (1.69x average) on L4 — [Source](https://arxiv.org/abs/2503.18292)
- Stated limitations: "We leave the full support of serving multiple models as a future work"; Hamba not evaluated because it lacks GPU kernels and is unsupported by vLLM — [Source](https://arxiv.org/abs/2503.18292)
- Follow-ups: Prism (OSDI'26) addresses memory sharing across multiple models with CUDA VMM ballooning — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)

#### 7. Aegaeon — SOSP 2025
- Links: PDF https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf (DOI 10.1145/3731569.3764815); no public code link found — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Problem: model marketplaces (Alibaba Cloud Model Studio) host many sporadically used models; request-level auto-scaling packs only 2-3 models per GPU. Aegaeon pre-empts and swaps models at token granularity — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Models: Qwen, Llama, InternLM and Yi families, mostly 6B-14B; a 72B model with TP=4 on 8x H800 — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Workloads: ShareGPT, ShareGPT-ix2 (inputs doubled), ShareGPT-ox2 (outputs doubled), Poisson-synthesized arrivals; SLO of TTFT 10 s and TBT 100 ms, with stricter variants down to 2 s / 20 ms — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Hardware: 2 nodes x 8 H800 80 GB (16 GPUs), 2 TB DDR5; 6 GPUs for prefill and 10 for decode — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Baselines: ServerlessLLM, ServerlessLLM+ (oracle shortest-job-first), MuxServe (which refuses more than 2 models per GPU). Headline: sustains 2-2.5x higher arrival rates or 1.5-9x goodput; auto-scaling overhead reduced 97%; production beta reduced GPUs from 1,192 to 213 (82% saving) — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Stated limitations: no explicit limitations section was found by text search of the PDF.
- Follow-ups: Prism (OSDI'26) evaluates the same multi-LLM sharing problem against ServerlessLLM and a MuxServe port — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)

#### 8. DroidSpeak — NSDI 2026
- Links: paper https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan ; arXiv 2411.02820 ; about 3K lines of Python on PyTorch; no dedicated repository link found — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)
- Problem: compound/agentic systems call several fine-tuned variants of one base model on the same context prefix; KV caches cannot be shared across models, so each model re-prefills. DroidSpeak reuses KV of selected layers across models — [Source](https://arxiv.org/abs/2411.02820)
- Models: eight pairs of fine-tunes of Mistral-7B, Mistral-24B, Llama-3.1-8B, Llama-3-8B, Phi-3.5-mini, Llama-3-70B / 3.1-70B (AWQ 4-bit) — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)
- Datasets: six, including HotpotQA, 2wikimQA, multifieldQA_en, multi_news, lcc; 50 HotpotQA contexts used for offline profiling — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)
- Hardware: two Azure Standard_ND96amsr_A100_v4 VMs (8x 80 GB A100 each), 200 Gbps InfiniBand — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)
- Baselines: full prefill, full KV reuse, CacheBlend. Headline: up to 4x throughput; prefill 1.7-3.1x faster (2.1x average) with negligible quality loss — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)
- Stated limitations (§6): works only between models fine-tuned from the same foundation model (cross-foundation sharing is a "challenging problem that we leave for future work"); adapts to load but not to bandwidth; offline profiling is subject to data drift; "may not work for all model pairs"; only two-model reuse evaluated — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)
- Follow-ups: none found after the NSDI'26 publication in this pass.

#### 9. CacheSlide — FAST 2026
- Links: paper https://www.usenix.org/conference/fast26/presentation/liu-yang ; code https://github.com/SJTU-Storage-Lab/CacheSlide ; implemented on vLLM 0.8.5 — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- Problem: agent prompts mix fixed and dynamic segments, so reusable segments keep their relative order but shift in absolute position, which defeats prefix caching (the paper's "Relative-Position-Dependent Caching"). Affected: agent frameworks with memory/reflection — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- Models: Mistral-7B, MPT-30B, Llama-3 70B, with CoPE positional-encoding support added "via adapter-based continued pretraining" (LoRA) — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- Workloads: HotPotQA with Reflexion, Multi-Session Chat with MemGPT, SWE-Agent-Bench with SWE-Agent — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- Hardware: single A100 80 GB (two for the 70B model), 500 GB DRAM, 2 TB NVMe, PCIe Gen4 — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- Baselines: ContextCache (Kimi), PromptCache, CacheBlend (via LMCache, 18% recompute), EPIC. Headline: 3.11-4.3x latency reduction and 3.5-5.8x throughput; 1.21-2.11x TTFT reduction vs CacheBlend — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- Stated limitations: no explicit limitations section was found by text search of the PDF. The method requires models adapted to CoPE through LoRA continued pretraining — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- Follow-ups: none found (published early 2026).

#### 10. Bidaw — FAST 2026
- Links: paper https://www.usenix.org/conference/fast26/presentation/hu-shipeng ; released workload trace https://github.com/ShipengHu-777/Interactive-conversation-workload ; no system code link identified — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Problem: two-tier (host memory + SSD) KV caching for multi-round interactive serving; KV loading from SSD raises latency by up to 3.8x and cuts throughput by up to 2.0x relative to an ideal all-in-memory cache — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Models: OPT-6.7B, Qwen-7B, OPT-13B, Qwen-14B, OPT-30B — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Workloads: the authors' interactive conversation workload (public trace) and ShareGPT with Poisson timestamps — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Hardware: one server with one 80 GB A800, 200 GB host memory, SSD tier at 1.5 GB/s (RAID-5 of 4 SATA SSDs), PCIe Gen4 — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Baselines: vLLM, CachedAttention and FlashGen (both reimplemented on vLLM because the originals are closed-source), ideal all-in-memory. Headline: latency reduced by up to 3.58x and throughput raised by up to 1.83x — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Stated caveat: the storage-efficient history-tensor caching targets MHA models; "for GQA-based LLMs, KV is smaller ... KV should be cached instead". No explicit limitations section found — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Follow-ups: none found (published early 2026). Shares the DRAM+SSD KV tiering problem with IMPRESS (FAST'25), HCache (EuroSys'25) and Strata (OSDI'26) — [Source](https://arxiv.org/abs/2508.18572)

#### 11. KunServe — EuroSys 2026 (venue from PDF header only)
- Links: arXiv https://arxiv.org/abs/2412.18169 ; code https://github.com/SJTU-IPADS/kunserve — [Source](https://arxiv.org/abs/2412.18169)
- Problem: request bursts overload GPU memory (KV cache), causing queueing and tail-latency spikes; KunServe frees memory by dropping replicated model parameters across instances and serving through pipeline parallelism — [Source](https://arxiv.org/abs/2412.18169)
- Models: Qwen-2.5-14B, Qwen-2.5-72B — [Source](https://arxiv.org/abs/2412.18169)
- Workloads: BurstGPT arrival trace (scaled with TraceUpscaler); request content from BurstGPT (avg 642 in / 262 out tokens), ShareGPT (1,660 / 373), LongBench (5.9K / 499) — [Source](https://arxiv.org/abs/2412.18169)
- Hardware: Cluster A, 8 servers x 1 A800 80 GB with 200 Gbps RDMA; Cluster B, 2 servers x 8 H800 80 GB with NVLink and 400 Gbps RDMA — [Source](https://arxiv.org/abs/2412.18169)
- Baselines: vLLM (data parallel), vLLM (pipeline parallel), InferCept, Llumnix. Headline: P99 latency reduced 8.4-11.9x vs baselines on the 72B model, at the cost of an 18.3-22.7% increase in P50/P99 TPOT — [Source](https://arxiv.org/abs/2412.18169)
- Stated limitations: current implementation focuses on dense models (MoE only discussed); the lookahead scheduler "assumes a sufficient number of requests queued ... which we leave possible solutions as a future work" — [Source](https://arxiv.org/abs/2412.18169)
- Follow-ups: none found in this pass.

#### 12. Marconi — MLSys 2025
- Links: arXiv https://arxiv.org/abs/2411.19379 ; code https://github.com/ruipeterpan/marconi ; Zenodo artifact https://zenodo.org/records/14970139 ; artifact appendix states Python, about 20 GB disk, about 12 hours, tested on CloudLab nodes — [Source](https://arxiv.org/abs/2411.19379)
- Problem: prefix caching for Attention+SSM hybrid models: SSM states are updated in place and cannot be rolled back to a shorter prefix, so exact-match checkpoints are needed, and fine-grained checkpointing fills the cache with large, rarely hit entries — [Source](https://arxiv.org/abs/2411.19379)
- Models: a 7B hybrid with {4, 24, 28} {Attention, SSM, MLP} layers for cache-policy experiments; Jamba-1.5-Mini (12B active / 52B total) on 4x A100-40GB for TTFT — [Source](https://arxiv.org/abs/2411.19379)
- Workloads: LMSys, ShareGPT, and SWE-Agent on SWE-Bench, as request traces with token IDs (about 7 GB), with varied inter-session and inter-request arrival rates — [Source](https://arxiv.org/abs/2411.19379)
- Hardware: AWS p4d.24xlarge (8x A100-40GB) — [Source](https://arxiv.org/abs/2411.19379)
- Baselines: vanilla (no caching), vLLM+ (block size 32) and SGLang+ (LRU); "neither vLLM nor SGLang natively supports prefix caching for Hybrid/SSM models, so we have extended both" — [Source](https://arxiv.org/abs/2411.19379)
- Metric and headline: token hit rate, plus P5/P50/P95 TTFT; average token-hit-rate gains of 4.5x / 7.3x / 34.4x (LMSys / ShareGPT / SWEBench) over vLLM+ — [Source](https://arxiv.org/abs/2411.19379)
- Stated limitations: no limitations section found; the paper states "We do not evaluate downstream metrics" — [Source](https://arxiv.org/abs/2411.19379)
- Follow-ups (engineering work and blog posts, not peer-reviewed papers):
  - SGLang added a MambaRadixCache for hybrid models — [Source](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
  - LMSYS "Unified Radix Cache" blog post (August 2026) — [Source](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache)
  - SegLen (Cornell, blog post dated 2026, no venue or arXiv id): eviction heuristic scoring entries by replay distance to the nearest ancestor with a valid recurrent state; evaluated with Qwen3.5-9B on one H100 in SGLang against LRU and Marconi on SWE-bench traces and ShareGPT; mean TTFT on SWE-bench (about 5x reuse) 2107.68 ms (LRU), 1178.64 ms (Marconi), 1027.35 ms (SegLen); on ShareGPT 72.48 / 75.29 / 72.39 ms. The post says "Marconi's FLOP-efficiency assumes parent nodes are valid" (tombstone nodes in SGLang break this) and that FLOP estimation makes the policy "tightly coupled to model details and harder to maintain" — [Source](https://abdelfattah-lab.github.io/blogs/seglen/)
  - vLLM pull request for partial prefix-cache hits on attention+Mamba hybrids — [Source](https://github.com/vllm-project/vllm/pull/46384); vLLM issue reporting prefix caching ineffective for Qwen3.5 hybrids below a 528-token block — [Source](https://github.com/vllm-project/vllm/issues/40696)
  - A targeted search for 2026 papers using Marconi as a baseline returned only the original paper and the SegLen post — [Source](https://abdelfattah-lab.github.io/blogs/seglen/)

#### 13. Mooncake — FAST 2025
- Links: paper https://www.usenix.org/system/files/fast25-qin.pdf ; code and traces https://github.com/kvcache-ai/Mooncake — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Problem: serving Kimi (Moonshot AI), a long-context chatbot, under TTFT/TBT SLOs; a KVCache-centric disaggregated architecture separates prefill and decode pools and keeps a global KV cache over DRAM/SSD, moved by RDMA — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Model: a dummy LLaMA3-70B (same architecture); cache block size 256 tokens — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Workloads (open-sourced traces): Conversation (avg input 12,035 / output 343 tokens, cache ratio 40%, 12,031 requests), Tool&Agent (8,596 / 182, 59%, 23,608), Synthetic (15,325 / 149, 66%, 3,993, Poisson arrivals) — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Hardware: cluster nodes each with 8x A800-SXM4-80GB and 4x 200 Gbps RDMA NICs — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Baseline: vLLM v0.5.1 (prefix caching and chunked prefill tested separately). Metric: effective request capacity under TTFT 30 s and TBT 100/200/300 ms thresholds. Headline: +59% to +498% effective request capacity; in production +115% / +107% requests on A800 / H800 clusters — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Stated limitations: not extracted in this pass (see Gaps).
- Follow-ups: Strata (OSDI'26) reuses the Mooncake Tool&Agent trace for its delay-hit analysis (38% of requests share a prefix of at least 6k tokens with another request arriving within one second) and cites Mooncake as a hashing-based KV-reuse design — [Source](https://www.usenix.org/system/files/osdi26-xie-zhiqiang.pdf); a 2026 survey groups Mooncake and LMCache as "shared-store" designs — [Source](https://arxiv.org/abs/2607.02574)

#### 14. IMPRESS — FAST 2025
- Links: paper https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf ; no code link identified in the PDF — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Problem: when prefix KVs are stored on SSD, I/O accounts for 51-98% of TTFT; IMPRESS loads only the important tokens' KVs and reorganizes storage and caching around importance — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Models: OPT-6.7B, OPT-13B, OPT-30B — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Workloads: PIQA, RTE, COPA, OpenBookQA with 2-10 few-shot examples used as shared prefixes, chosen "Due to the lack of open-source, real-world datasets for prefix reuse"; prefixes up to 4K tokens (OPT-30B) or 10K — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Hardware: 2x EPYC 7763, 128 GB DRAM, one A100 80 GB, one 2 TB Intel SSD (about 5 GB/s), PCIe 4.0 x16; 10 GB GPU cache and 32 GB CPU cache budgets — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Baselines: ReComp (recompute), AS-like (AttentionStore reimplementation), AS+H2O+LRU, AS+H2O+LFU. Headline: TTFT up to 2.8x lower with comparable accuracy — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Stated limitations: not extracted in this pass (see Gaps).
- Follow-ups: Bidaw (FAST'26) and Strata (OSDI'26) address KV loading from DRAM/SSD tiers — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng); arXiv-only 2026 SSD-offload work found by search: Swarm — [Source](https://arxiv.org/abs/2603.17803), KVDrive — [Source](https://arxiv.org/abs/2605.18071)

#### 15. HCache — EuroSys 2025
- Links: arXiv https://arxiv.org/abs/2410.05004 ; built on DeepSpeed-MII; no public code link found in the PDF — [Source](https://arxiv.org/abs/2410.05004)
- Problem: restoring evicted LLM state for multi-turn/RAG contexts is slow either way (token recomputation is compute-bound, KV offload is I/O-bound); HCache restores from saved intermediate hidden states — [Source](https://arxiv.org/abs/2410.05004)
- Models: Llama2-7B and Llama2-13B (one GPU), OPT-30B (4 GPUs, tensor parallel); context extended to 16K. Traces: ShareGPT4, L-Eval — [Source](https://arxiv.org/abs/2410.05004)
- Hardware: 4x A100-40G (NVLink), 256 GB DDR4, 4x Samsung PM9A3 4 TB SSDs; sensitivity runs on A30, RTX 4090, L20 (48 GB), H800 — [Source](https://arxiv.org/abs/2410.05004)
- Baselines: recomputation (DeepSpeed-MII), KV offload (AttentionStore reimplemented), ideal. Headline: TTFT up to 1.93x lower than KV offload with 1.92-2.40x less storage; up to 5.73x lower than recomputation; under 4% TBT overhead — [Source](https://arxiv.org/abs/2410.05004)
- Stated limitations: not extracted in this pass (see Gaps).
- Follow-ups: Bidaw (FAST'26) caches history tensors rather than KV for MHA models on a host-memory + SSD hierarchy — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)

#### 16. Pensieve — EuroSys 2025
- Links: arXiv https://arxiv.org/abs/2312.05516 ; standalone engine of about 7K lines of C++/CUDA; no code link found — [Source](https://arxiv.org/abs/2312.05516)
- Problem: stateless serving reprocesses the whole conversation history every turn; Pensieve keeps per-conversation KV in a GPU/CPU multi-tier cache with a multi-token attention kernel over non-contiguous cache — [Source](https://arxiv.org/abs/2312.05516)
- Models: OPT and Llama 2 families, small models on 1 GPU and large on 4 GPUs (Llama 2-13B modified to GQA) — [Source](https://arxiv.org/abs/2312.05516)
- Workloads: ShareGPT (48,159 conversations, mean 5.56 turns), UltraChat (1,468,352 conversations); Poisson arrivals respecting turn order; max context 16,384 — [Source](https://arxiv.org/abs/2312.05516)
- Hardware: Azure NC A100 v4 (up to 4x A100-80GB, 220 GB CPU memory per GPU), 40 GB per GPU reserved for KV — [Source](https://arxiv.org/abs/2312.05516)
- Baselines: vLLM v0.2.0, TensorRT-LLM v0.12.0. Headline: 1.14-3.0x throughput — [Source](https://arxiv.org/abs/2312.05516)
- Stated limitations: not extracted in this pass (see Gaps).
- Follow-ups: Strata's SGLang-HiCache baseline is a reimplementation "in line with" CachedAttention / Pensieve / FlashGen — [Source](https://arxiv.org/abs/2508.18572)

#### 17. vAttention — ASPLOS 2025
- Links: arXiv https://arxiv.org/abs/2405.04437 ; code https://github.com/microsoft/vattention ; artifact DOI 10.5281/zenodo.14048692 ; inside vLLM v0.2.7 with FlashAttention-2 v2.5.9 and FlashInfer v0.4.0 back-ends — [Source](https://arxiv.org/abs/2405.04437)
- Problem: PagedAttention makes the KV cache non-contiguous in virtual memory, which forces rewritten attention kernels and adds overhead; vAttention keeps virtual contiguity and maps physical GPU memory on demand through CUDA virtual memory APIs — [Source](https://arxiv.org/abs/2405.04437)
- Models and hardware: Yi-6B on one A100 80 GB; Llama-3-8B and Yi-34B on two A100 (TP-2); 1-2 H100 for portability — [Source](https://arxiv.org/abs/2405.04437)
- Baseline and headline: paged kernels in vLLM; up to 1.23x throughput — [Source](https://arxiv.org/abs/2405.04437)
- Stated discussion points: cudaMallocManaged judged unsuitable (no partial freeing, no aliasing, 2 MB pages); small page sizes need modified NVIDIA drivers, with a tensor-slicing alternative otherwise — [Source](https://arxiv.org/abs/2405.04437)
- Follow-ups: Prism (OSDI'26) also builds on CUDA virtual memory management APIs, for cross-model memory ballooning — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan); Jenga (SOSP'25) addresses allocation for heterogeneous KV sizes inside vLLM — [Source](https://arxiv.org/abs/2503.18292)

#### 18. Rethinking Key-Value Cache Compression Techniques for LLM Serving — MLSys 2025
- Links: arXiv https://arxiv.org/abs/2503.24000 ; code https://github.com/LLMkvsys/rethink-kv-compression — [Source](https://arxiv.org/abs/2503.24000)
- Problem: KV compression algorithms are reported with memory-ratio and accuracy numbers, not serving metrics; the paper re-measures them from a serving perspective — [Source](https://arxiv.org/abs/2503.24000)
- Setup: "The majority of the experiments are conducted on A6000 GPUs" (also H800) with transformers, FlashAttention and LMDeploy; LLaMA and Mistral model families; ShareGPT for throughput/latency and LongBench for negative-sample analysis; algorithms KIVI, GEAR, StreamingLLM, H2O — [Source](https://arxiv.org/abs/2503.24000)
- Findings reported: compression implementations are not optimized for production serving and give suboptimal throughput; compression can lengthen outputs and so raise end-to-end latency; specific tasks show negative samples (accuracy loss) — [Source](https://arxiv.org/abs/2503.24000)
- Follow-ups: DiffKV (SOSP'25) integrates differentiated KV compression into vLLM with an on-GPU memory manager — [Source](https://arxiv.org/abs/2412.03131)

#### 19. Cake ("Compute Or Load KV Cache? Why Not Both?") — ICML 2025 (venue unverified)
- Links: arXiv https://arxiv.org/abs/2410.03065 ; no code link found in the PDF — [Source](https://arxiv.org/abs/2410.03065)
- Problem: on a prefix-cache hit stored remotely or on disk, loading KV is I/O-bound and recomputing is GPU-bound; Cake does both at once from opposite ends of the prefix — [Source](https://arxiv.org/abs/2410.03065)
- Models: LongAlpaca-7B/13B, LLaMA 3.1-8B, LLaMA 3.1-70B (FP8). Datasets: LongChat, TriviaQA, NarrativeQA; synthetic prompts of 4K-16K tokens — [Source](https://arxiv.org/abs/2410.03065)
- Hardware: server with 2x A100 80 GB (NVLink, EPYC 7763, 2 TB memory) and a single-H100 server; I/O bandwidth 7-100 Gbps, simulated by inserting delays computed from chunk size and bandwidth — [Source](https://arxiv.org/abs/2410.03065)
- Baselines: vLLM v0.6.2 chunked prefill (512 / 1024), LMCache v0.1.4. Headline: 2.6x average TTFT reduction — [Source](https://arxiv.org/abs/2410.03065)
- Stated future work: adaptive fallback to a single-resource mode when one resource dominates — [Source](https://arxiv.org/abs/2410.03065)

#### 20. CacheBlend — EuroSys 2025 (venue unverified from PDF/venue page)
- Links: arXiv https://arxiv.org/abs/2405.16444 ; code https://github.com/LMCache/LMCache ; about 3K lines on vLLM — [Source](https://arxiv.org/abs/2405.16444)
- Problem: RAG inputs concatenate several reusable chunks, so only the first is a prefix; CacheBlend reuses all chunk KVs and selectively recomputes a small token subset to restore cross-attention — [Source](https://arxiv.org/abs/2405.16444)
- Setup: Runpod with 2x Nvidia A40, 128 GB RAM, 1 TB NVMe (4.8 GB/s); Mistral-7B, Yi-34B, Llama-70B (8-bit quantized); 2WikiMQA (200 cases), Musique (150) — [Source](https://arxiv.org/abs/2405.16444)
- Headline: TTFT 2.2-3.3x lower than full recompute; throughput up to 5x — [Source](https://arxiv.org/abs/2405.16444)
- Stated limitations: "currently only applies to language models with transformer structures. We leave investigation of architectures other than transformer such as Mamba and Griffin for future work"; sharing KV across compute nodes not studied — [Source](https://arxiv.org/abs/2405.16444)
- Follow-ups: used as a baseline by DroidSpeak (NSDI'26) — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) and CacheSlide (FAST'26) — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)

#### Secondary entries
- CacheGen (SIGCOMM'24): KV cache encoding/streaming to cut network transfer; code https://github.com/UChi-JCL/CacheGen ; Mistral-7B, Llama-34B, Llama-70B; LongChat, TriviaQA, NarrativeQA, Wikitext (662 contexts); TTFT 3.1-4.7x lower, bandwidth 3.5-4.3x lower — [Source](https://arxiv.org/abs/2310.07240)
- InfiniGen (OSDI'24, venue unverified here): dynamic KV management keeping only essential tokens on GPU for offloading-based inference; baselines UVM, FlexGen, H2O, quantization; OPT-6.7B/13B/30B and Llama-2-7B/13B; up to 3.00x — [Source](https://arxiv.org/abs/2406.19707)
- PrefillOnly (SOSP'25): engine for prefill-only workloads (one output token), on vLLM; L4 / A100 / H100; post-recommendation and credit-verification datasets; 1.4-4.0x QPS; stated limitations: discarding suffix KV prevents its reuse, and CPU offload via LMCache is left to future work — [Source](https://arxiv.org/abs/2505.07203)
- Accelerating Model Loading in LLM Inference by Programmable Page Cache (FAST'26): Huawei Kunpeng 920 with 8 Ascend 910B2 NPUs, vLLM-Ascend 0.9.2; baselines Native, EagerLoad, PreCache, ServerlessLLM — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yubo)
- ServerlessLLM (OSDI'24, venue unverified here): fast checkpoint loading and locality-aware serverless LLM serving; used as a baseline by Aegaeon, Prism and the FAST'26 model-loading paper — [Source](https://arxiv.org/abs/2401.14351)
- SCBench (ICLR'25): benchmark of long-context methods across the KV-cache lifecycle (generation, compression, retrieval, loading) with shared-context multi-request tests, 12 tasks, two shared-context modes — [Source](https://arxiv.org/abs/2412.10319)
- HILOS (ASPLOS'26 per search only): near-storage (SmartSSD) KV processing for offline batched inference, up to 7.86x — [Source](https://arxiv.org/abs/2502.09921)

#### arXiv-only or lower-tier 2025-2026 items found by search (titles/abstracts only; not read, not peer-review-verified unless stated)
- KVDrive, GPU/DRAM/SSD multi-tier KV management — [Source](https://arxiv.org/abs/2605.18071)
- Swarm, multi-SSD KV offload — [Source](https://arxiv.org/abs/2603.17803)
- DUAL-BLADE (listed as ICDCS 2026 in search results) — [Source](https://arxiv.org/abs/2604.26557)
- PEEK — [Source](https://arxiv.org/abs/2607.02525)
- Beluga, CXL-based KV cache memory — [Source](https://arxiv.org/abs/2511.20172)
- Survey "From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving" (June 2026): reviews 30+ systems along locality, lifetime, ownership and substrate; names five archetypes (local-paged, disaggregated-pipeline, shared-store, memory-pool, hybrid-tier) — [Source](https://arxiv.org/abs/2607.02574)
- A second 2026 survey on KV cache systems — [Source](https://arxiv.org/abs/2607.08057)

### Inferences
Reproducibility on one node with about 4 idle 48 GB GPUs (RTX A6000 / RTX 6000 Ada, PCIe, no H100/A100), models up to about 30B. These verdicts are the researcher's inferences from the hardware and models each paper reports; none was tested. The target node's host DRAM and SSD configuration is unknown and matters for every offload paper.

| # | Paper | Verdict | What changes / why |
|---|---|---|---|
| 1 | Strata | scaled down | 8B and 14B single-GPU runs fit; drop Llama-3.1-70B; HBM tier is 48 GB instead of 141 GB (H200); DRAM (1.6 TB in paper) and NVMe (7 GB/s) tiers depend on the node; PCIe 5.0 bandwidth of the paper not available on A6000-class cards |
| 2 | ECHO | no | 671B model on 8x 96 GB GPUs with about 1 TB host KV pool |
| 3 | DirectKV | no for the headline | headline platform is GH200 with NVLink-C2C; the authors state PCIe-only platforms see limited throughput benefit. Models (8B-30B) themselves fit, so memory-reduction results could be attempted scaled down |
| 4 | Prism | scaled down | 4 GPUs instead of 32 H100; use the 1B-8B subset (51 of 58 models) and a shortened trace; public availability of the Hyperbolic/Novita traces not checked |
| 5 | DiffKV | yes | paper used 48 GB L40s (1 GPU for 7B/8B, 2 for 32B, 4 for 70B); code public on vLLM; Llama3-70B exceeds the 30B preference but matches the 4x 48 GB hardware used |
| 6 | Jenga | scaled down | H100 80 GB replaced by 48 GB (paper also reports L4 24 GB); skip Jamba 52B; no code link found, so the allocator would need reimplementation |
| 7 | Aegaeon | no (heavily scaled down at best) | 16 H800 with separate prefill/decode pools; no public code; 6B-14B models fit but the pool-level result does not transfer to 4 GPUs |
| 8 | DroidSpeak | scaled down | 7B/8B/24B pairs fit on 48 GB cards; drop 70B pairs; two-VM InfiniBand setup collapses to one node; no repository link found |
| 9 | CacheSlide | scaled down | A100 80 GB replaced by 48 GB; Mistral-7B fits, MPT-30B needs two GPUs or lower precision, drop 70B; needs CoPE LoRA adapters (availability in the repository not checked) |
| 10 | Bidaw | scaled down | single 80 GB A800 replaced by 48 GB (OPT-30B at fp16 would not fit on one card); modest SSD (1.5 GB/s) and 200 GB DRAM are commodity; system code not found, trace public |
| 11 | KunServe | scaled down | 4 GPUs in one node instead of 8 single-GPU servers over RDMA; Qwen-2.5-14B only; code public |
| 12 | Marconi | yes (cache-policy part) | token-hit-rate experiments are trace-driven with a public artifact (about 12 hours, about 20 GB disk); the Jamba-1.5-Mini TTFT part used 4x A100-40GB and a 52B-total model |
| 13 | Mooncake | no (cluster); traces reusable | RDMA cluster of 8x A800 nodes; the three open traces can drive a single-node cache study |
| 14 | IMPRESS | scaled down | single A100 80 GB replaced by 48 GB (OPT-6.7B/13B fit; OPT-30B needs two cards); one fast NVMe SSD required; no code |
| 15 | HCache | scaled down | paper's 4x A100-40G maps onto 4x 48 GB; needs several NVMe SSDs; DeepSpeed-MII base; no code |
| 16 | Pensieve | scaled down | up to 4x A100-80GB with 40 GB/GPU KV budget becomes a smaller KV budget; standalone C++/CUDA engine with no code |
| 17 | vAttention | yes / scaled down | Yi-6B on one GPU, Llama-3-8B and Yi-34B on two; code public but pinned to vLLM v0.2.7 |
| 18 | Rethinking KV compression | yes | majority of experiments on A6000 GPUs; code public |
| 19 | Cake | scaled down | 7B/8B/13B models fit; drop 70B FP8; I/O bandwidth is simulated by delays, so no special storage needed; no code link found |
| 20 | CacheBlend | yes | paper used 2x A40 (48 GB); code in LMCache; Llama-70B 8-bit exceeds the 30B preference |

- The Mooncake Tool&Agent trace is also used outside its own paper (Strata's delay-hit analysis), in addition to ShareGPT being near-universal. The workloads that recur across at least three catalogued papers are ShareGPT (Strata, ECHO, DirectKV, Aegaeon, Bidaw, KunServe, Marconi, Pensieve, Rethinking-KV, HCache via ShareGPT4), LongBench or its subsets (DiffKV, KunServe, Rethinking-KV, DroidSpeak) and NarrativeQA (Strata, Cake, CacheGen).
- AttentionStore/CachedAttention is repeatedly reimplemented as a baseline because it is closed-source (IMPRESS, HCache, Bidaw, Strata's HiCache).

### Gaps
- Stated limitations were not extracted for Mooncake, IMPRESS, HCache and Pensieve; no explicit limitations section was found by text search in Prism, Aegaeon, CacheSlide, Bidaw and Marconi. A manual read of the discussion/conclusion of each is still needed.
- Public availability and licence of the Prism traces (Hyperbolic, Novita) and whether the Arena-Chat/Arena-Battle traces are downloadable were not checked.
- Whether Jenga's allocator and Strata's hierarchical cache are present in upstream vLLM / SGLang releases was not verified against the repositories.
- Follow-up counts are from two targeted web searches only, not a citation-graph query (Semantic Scholar / Google Scholar "cited by" was not used); "none found" does not mean none exist.
- HiFC (flash-based KV swapping) surfaced in search without a confirmed URL or venue and is omitted from the lists above.
- The arXiv-only items (KVDrive, Swarm, DUAL-BLADE, PEEK, Beluga, the second survey) were not read; their setups and baselines are unrecorded.

## Key question 3: Which papers evaluate inside vLLM, SGLang, llama.cpp or TGI, and which use a standalone prototype?

### Takeaway
vLLM is the most common host engine (DiffKV, Jenga, vAttention, CacheSlide, CacheBlend, PrefillOnly, plus Marconi's and Bidaw's extended baselines) and SGLang hosts three of the four OSDI'26 papers (Strata, ECHO, Prism); no catalogued paper implements its system in llama.cpp or TGI.

### Cited Findings

#### Implemented inside vLLM
- DiffKV: on vLLM, about 4.5K lines CUDA/C++ plus 9K Python — [Source](https://arxiv.org/abs/2412.03131)
- Jenga: on vLLM, about 4,000 lines of Python with no CUDA kernel change; baseline vLLM v0.6.3; stated compatible with the 90 models of vLLM v0.6.4 — [Source](https://arxiv.org/abs/2503.18292)
- vAttention: inside vLLM v0.2.7 with FlashAttention-2 and FlashInfer back-ends — [Source](https://arxiv.org/abs/2405.04437)
- CacheSlide: on vLLM 0.8.5 — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yang)
- CacheBlend: about 3K lines on vLLM; code released through LMCache — [Source](https://arxiv.org/abs/2405.16444)
- PrefillOnly: built on vLLM — [Source](https://arxiv.org/abs/2505.07203)
- Bidaw: baselines CachedAttention and FlashGen reimplemented on vLLM; vLLM itself is a baseline — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- Marconi: compares against vLLM+ and SGLang+, which the authors extended because neither supported prefix caching for hybrid/SSM models at the time — [Source](https://arxiv.org/abs/2411.19379)
- FAST'26 model-loading paper: evaluated with vLLM-Ascend 0.9.2 on Ascend NPUs — [Source](https://www.usenix.org/conference/fast26/presentation/liu-yubo)

#### Implemented inside SGLang
- Strata: "Implemented as part of SGLang and deployed in production"; compares with SGLang v0.4.5, vLLM v0.8.5, vLLM + LMCache, TensorRT-LLM — [Source](https://arxiv.org/abs/2508.18572)
- ECHO: atop SGLang and DeepGEMM; compares with SGLang v0.5.4 and vLLM v0.11.1 — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Prism: on SGLang with a 22-line engine change; kvcached is a separate library — [Source](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
- SegLen (blog, not a paper): evaluated in SGLang with Qwen3.5-9B — [Source](https://abdelfattah-lab.github.io/blogs/seglen/)

#### Other engines
- HCache: built on DeepSpeed-MII — [Source](https://arxiv.org/abs/2410.05004)
- Rethinking KV compression: transformers, FlashAttention and LMDeploy — [Source](https://arxiv.org/abs/2503.24000)
- Cake: compares against vLLM v0.6.2 chunked prefill and LMCache v0.1.4 — [Source](https://arxiv.org/abs/2410.03065)

#### Standalone prototypes or own platforms
- DirectKV: about 5,300 lines CUDA/C++ on FlashAttention-3 with PyTorch bindings; SGLang is a baseline; positioned as complementary to vLLM/SGLang — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Pensieve: own engine, about 7K lines C++/CUDA; baselines vLLM v0.2.0 and TensorRT-LLM v0.12.0 — [Source](https://arxiv.org/abs/2312.05516)
- DroidSpeak: about 3K lines of Python on PyTorch — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)
- Mooncake: own production platform; baseline vLLM v0.5.1 — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Aegaeon: own production system (Alibaba Cloud Model Studio); baselines ServerlessLLM and MuxServe — [Source](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)

#### TGI and llama.cpp
- Jenga compares its memory manager against vLLM, SGLang and TGI in a breakdown experiment; this is the only use of TGI found in the catalogue — [Source](https://arxiv.org/abs/2503.18292)

### Inferences
- Baseline engine versions age quickly: the catalogue spans vLLM v0.2.0 (Pensieve), v0.2.7 (vAttention), v0.5.1 (Mooncake), v0.6.x (Jenga, Cake), v0.8.5 (Strata, CacheSlide) and v0.11.1 (ECHO); a reproduction on a current engine would not match the paper's baseline numbers directly.
- LMCache appears as a baseline or code vehicle in Strata, Cake, CacheSlide and CacheBlend, so it functions as the de facto open comparison point for KV offload on vLLM.
- No catalogued paper was found that implements its system in llama.cpp or TGI.

### Gaps
- The base engine of IMPRESS and KunServe was not captured from their PDFs; Bidaw's own implementation base (as opposed to its baselines) was not explicitly confirmed.
- Whether any of these systems has been merged upstream (vLLM, SGLang) was not checked against the repositories, except that Strata's paper states it is part of SGLang.

## Key question 4: Which problems do several papers name as still open?

### Takeaway
The problems stated as open by more than one paper are: (1) support beyond dense standard attention (sparse, linear, SSM/hybrid), (2) slow-tier I/O bandwidth and integration with multi-tier hierarchies, (3) the latency/SLO/fairness cost of offloading and cache-aware scheduling, (4) generality across models (multi-model memory sharing, cross-model KV reuse), (5) missing public real-world prefix-reuse traces, and (6) KV sharing across nodes.

### Cited Findings

#### (1) Architectures beyond dense standard attention
- Strata: "Strata targets hierarchical caching for the dense KV caches produced by standard attention. As hybrid mechanisms such as sparse and linear attention gain adoption, extending the same level of support to these models becomes increasingly important and raises distinct challenges in both memory management and scheduling-policy design." — [Source](https://arxiv.org/abs/2508.18572)
- CacheBlend: "currently only applies to language models with transformer structures. We leave investigation of architectures other than transformer such as Mamba and Griffin for future work" — [Source](https://arxiv.org/abs/2405.16444)
- ECHO: "DSA is currently the only publicly available native sparse attention validated on large-scale models"; co-design of sparse attention with prefetch-friendly selection left to future work — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- Jenga: Hamba not evaluated because it lacks GPU kernels and vLLM support — [Source](https://arxiv.org/abs/2503.18292)
- Marconi: vLLM and SGLang did not natively support prefix caching for hybrid/SSM models at the time of writing — [Source](https://arxiv.org/abs/2411.19379)
- SegLen blog: Marconi's FLOP-based eviction is "tightly coupled to model details and harder to maintain" across sequence mixers (Mamba, gated delta networks) and feed-forward designs (MoE) — [Source](https://abdelfattah-lab.github.io/blogs/seglen/)
- vLLM issue: prefix caching reported ineffective for Qwen3.5 hybrid models below a 528-token block — [Source](https://github.com/vllm-project/vllm/issues/40696)
- KunServe: implementation focuses on dense models; MoE only discussed — [Source](https://arxiv.org/abs/2412.18169)
- DiffKV: models with more aggressive training-time KV compression (higher queries-per-KV ratios in GQA) may need an additional precision level, left to future investigation — [Source](https://arxiv.org/abs/2412.03131)
- Bidaw: history-tensor caching fits MHA models; for GQA models "KV should be cached instead" — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)

#### (2) Slow-tier I/O bandwidth and multi-tier integration
- IMPRESS: SSD I/O is 51-98% of TTFT when prefix KVs live on SSD — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Bidaw: KV loading in a host-memory + SSD hierarchy raises latency up to 3.8x and cuts throughput up to 2.0x vs ideal — [Source](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- DirectKV: on PCIe-only platforms throughput benefits "are often limited by interconnect bandwidth"; integration with a multi-tier KV hierarchy and pageable/disk-backed tiers left to future work — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)
- Strata: hiding the latency of prefetching from slower storage tiers named in its discussion — [Source](https://arxiv.org/abs/2508.18572)
- PrefillOnly: CPU offload of KV via LMCache left to future work — [Source](https://arxiv.org/abs/2505.07203)
- Cake: adaptive fallback to single-resource mode when either compute or I/O dominates left as future work — [Source](https://arxiv.org/abs/2410.03065)

#### (3) Latency, SLO and fairness cost of offloading / cache-aware scheduling
- Strata: cache-aware batching "can still treat requests unevenly, risking SLO violations for individual requests" — [Source](https://arxiv.org/abs/2508.18572)
- ECHO: "offloading introduces latency overhead", so it is "most attractive for throughput-oriented long-context serving" — [Source](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
- KunServe: tail-latency gains come with an 18.3-22.7% increase in P50/P99 TPOT, and the lookahead scheduler assumes enough queued requests — [Source](https://arxiv.org/abs/2412.18169)

#### (4) Generality across models
- Jenga: "We leave the full support of serving multiple models as a future work" — [Source](https://arxiv.org/abs/2503.18292)
- DroidSpeak: KV reuse only between fine-tunes of the same foundation model; cross-foundation sharing is a "challenging problem that we leave for future work"; only two-model reuse evaluated; "may not work for all model pairs" — [Source](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)

#### (5) Missing public real-world traces for prefix/context reuse
- IMPRESS built prefixes from few-shot examples "Due to the lack of open-source, real-world datasets for prefix reuse" — [Source](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf)
- Mooncake open-sourced three production-derived traces (Conversation, Tool&Agent, Synthetic) with cache ratios 40-66% — [Source](https://www.usenix.org/system/files/fast25-qin.pdf)
- Bidaw released its own interactive conversation workload trace — [Source](https://github.com/ShipengHu-777/Interactive-conversation-workload)
- SCBench (ICLR'25) states existing benchmarks evaluate single requests and neglect the KV-cache lifecycle — [Source](https://arxiv.org/abs/2412.10319)

#### (6) KV sharing across nodes / scale-out
- CacheBlend: sharing KV caches across compute nodes not studied — [Source](https://arxiv.org/abs/2405.16444)
- DirectKV: "context-parallel scale-out" and "coordinating remotely owned KV blocks" named as main limitations — [Source](https://www.usenix.org/conference/osdi26/presentation/luo)

#### Open problems named by a 2026 survey (arXiv-only)
- The survey audits evaluations, identifies seven missing KV-specific measurements, and links them to "open problems in fault tolerance, isolation, tiered eviction, speculative decoding, MoE serving", plus shared-cache semantics — [Source](https://arxiv.org/abs/2607.02574)

### Inferences
- Theme (1) is the one stated most widely: an OSDI'26 paper (Strata), an EuroSys'25 paper (CacheBlend), a SOSP'25 paper (Jenga) and the MLSys'25 hybrid-caching paper (Marconi) each mark non-dense-attention or hybrid models as unsupported or future work, while the only follow-ups found for hybrid-state caching are engine pull requests and blog posts rather than peer-reviewed papers.
- Themes (2) and (3) are named mostly by the 2025-2026 offload papers, which also form the most populated cluster in the catalogue (Strata, ECHO, DirectKV, Bidaw, IMPRESS, HCache, Cake, plus arXiv items KVDrive and Swarm).
- Papers disagree by construction on what to store for restoration: HCache stores hidden states, Bidaw stores history tensors for MHA models but says KV should be cached for GQA models, and Strata/IMPRESS store KV; each evaluates on different model families (OPT/Llama2 MHA models vs Llama-3/Qwen GQA models), so results are not directly comparable.

### Gaps
- Only the abstract of the 2026 survey (arXiv 2607.02574) was read; its list of seven missing measurements and detailed open problems was not extracted.
- Open problems for Prism, Aegaeon, CacheSlide, Mooncake, IMPRESS, HCache and Pensieve are missing from this section because no explicit limitations statement was captured (see Key question 2 Gaps).
- No peer-reviewed 2026 paper was found that states open problems specifically for linear-attention or gated-delta-network hybrid state caching (Qwen3-Next / Qwen3.5 style); evidence there is limited to engine issues, pull requests and blogs.
