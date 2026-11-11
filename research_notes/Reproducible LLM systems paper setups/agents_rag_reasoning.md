# Systems for LLM agents, compound AI pipelines, RAG and reasoning-model workloads (2024-2026): problems and evaluation setups

Reading notes for the report writer (how these notes were made, applies to every entry):

- Venue was checked against the venue's own page where one is cited (USENIX, sigops.org, PMLR, neurips.cc, mlsys.org). Where the only evidence is the arXiv "Comments" field, the paper's own header, or a repository README, the entry says so.
- Setup and limitation details were extracted from the arXiv HTML version of each paper through a summarising fetch tool that reads only the first ~100k characters of a page. Quoted strings are as returned by that tool; they were not re-checked against the PDF, and the arXiv version can differ from the camera-ready. Anything that fell in the unread tail (usually appendices, sometimes the conclusion) is marked "not seen".
- Item 4 of each entry ("fits 4x48 GB?") is my assessment from the stated hardware and model sizes, not a statement by the authors. Memory arithmetic in it assumes 2 bytes per parameter for 16-bit weights.
- Item 6 ("follow-ups") lists papers I found that cite or position against the entry. Unless stated, I did not confirm that the follow-up reproduces or beats the entry's numbers.

## Which papers at top venues (2024-2026) address agent / compound-AI / RAG / reasoning-model serving, and what are their problem, evaluation setup, reproducibility, stated limitations and follow-ups?

### Takeaway
Eighteen peer-reviewed papers are catalogued below (OSDI '24/'26, NSDI '26, SOSP '25, ASPLOS '25, ISCA '25, HPCA '26, MLSys '25/'26, ICML '24, NeurIPS '25, IISWC '26), plus three arXiv-only items kept because they are notable follow-ups or the only source on a sub-topic (agent sandboxes, production agent traces). Most evaluations use one node, open models of 8B-32B, public QA/agent benchmarks and synthetic Poisson arrivals; the main obstacles to reproducing them on 4x48 GB are 70B+ model variants, H100-class memory, proprietary APIs used for planners/profilers/trace generation, and missing code for several of the most-cited systems (Agentix, METIS, Murakkab, PASCAL).

### Cited Findings

#### Venue confirmation summary

- SOSP 2025 accepted-papers page lists "Pie: A Programmable Serving System for Emerging LLM Applications", "HedraRAG: Co-Optimizing Generation and Retrieval for Heterogeneous RAG Workflows" and "METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation" — [SOSP 2025 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html)
- OSDI 2026 technical sessions list "Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms" (Chaudhry) and "Breaking the Reward Barrier: Accelerating Tree-of-Thought Reasoning via Speculative Exploration" (Zhong) — [OSDI '26 technical sessions](https://www.usenix.org/conference/osdi26/technical-sessions)
- NSDI 2026 lists "Agentix: An Efficient Serving Engine for LLM Agents as General Programs" (Luo et al.), pages 2443-2459; this is the published title of the arXiv preprint "Autellix" (same subtitle, same authors, same 4-15x claim) — [NSDI '26 presentation page](https://www.usenix.org/conference/nsdi26/presentation/luo); [arXiv 2502.13965](https://arxiv.org/html/2502.13965)
- Parrot is OSDI '24, pages 929-945, with three artifact-evaluation badges (labels inferred from badge filenames: Available, Functional, Reproduced) — [OSDI '24 presentation page](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)
- InferCept is ICML 2024, PMLR 235:81-95 — [PMLR v235 abhyankar24a](https://proceedings.mlr.press/v235/abhyankar24a.html)
- LLMCompiler is ICML 2024, PMLR 235:24370-24391 — [PMLR v235 kim24y](https://proceedings.mlr.press/v235/kim24y.html)
- "Efficiently Scaling LLM Reasoning Programs with Certaindex" (Dynasor) is a NeurIPS 2025 poster — [NeurIPS 2025 poster 116107](https://neurips.cc/virtual/2025/poster/116107)
- XGrammar's paper header reads "Proceedings of the 8th MLSys Conference, Santa Clara, CA, USA, 2025" (paper-stated; MLSys 2025 program page not opened) — [arXiv 2411.15100v3](https://arxiv.org/html/2411.15100v3)
- TeleRAG's paper header reads "Proceedings of the 9th MLSys Conference (MLSys 2026), Bellevue, WA, USA" (paper-stated; the MLSys 2026 papers page did not render in my fetch) — [arXiv 2502.20969](https://arxiv.org/html/2502.20969)
- OSWorld-Human is listed as an MLSys 2026 oral — [MLSys 2026 oral 3865](https://mlsys.org/virtual/2026/oral/3865)
- PASCAL and "The Cost of Dynamic Reasoning" both carry the arXiv comment "Accepted for publication at the 32nd IEEE International Symposium on High-Performance Computer Architecture (HPCA-32), 2026" (author-stated; HPCA program not opened) — [arXiv 2602.11530](https://arxiv.org/abs/2602.11530); [arXiv 2506.04301](https://arxiv.org/abs/2506.04301)
- Ayo (arXiv title "Teola") is reported as ASPLOS '25, Vol. 2, pages 1302-1316, DOI 10.1145/3676641.3716278 (from a search result and a curated list; ACM DL page not opened, so treat as unverified from the primary source) — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md); [arXiv 2407.00326](https://arxiv.org/abs/2407.00326v3)
- RAGO is described as an ISCA '25 paper by its official repository, DOI 10.1145/3695053.3731093 (ISCA program page not opened) — [github.com/google/rago](https://github.com/google/rago)
- "Agentic AI Workload Characteristics" carries the arXiv comment "IISWC 2026" (author-stated) — [arXiv 2605.26297](https://arxiv.org/abs/2605.26297)

#### A. Serving systems for agent programs and LLM applications

**A1. Parrot: Efficient Serving of LLM-based Applications with Semantic Variable — OSDI 2024**
1. Links: [paper page](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan), [arXiv 2405.19888](https://arxiv.org/html/2405.19888). Code: artifact badges are shown on the USENIX page but neither it nor the arXiv page links a repository; repository URL not confirmed in this session — [OSDI '24 page](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)
2. Problem: public LLM service providers see only individual requests, so applications "have to use the over-simplified request-level API provided by today's public LLM services", which causes chatty round trips, misaligned scheduling objectives and recomputation of shared prompt prefixes; the party hurt is the application developer (end-to-end latency), the party who can fix it is the service provider — [arXiv 2405.19888](https://arxiv.org/html/2405.19888)
3. Setup — [arXiv 2405.19888](https://arxiv.org/html/2405.19888):
   - Models: LLaMA 13B and LLaMA 7B.
   - Workloads: chain and map-reduce summarisation on the Arxiv-March dataset (public); Bing Copilot and GPTs system prompts with synthesised user queries (release not stated); multi-agent programming built on MetaGPT; chat from ShareGPT (public); mixed chat + map-reduce. GPT-4 responses were recorded so that the LLaMA models "generated text of similar length"; a random 200-300 ms delay emulates Internet latency.
   - Hardware: one server with 1x A100 80 GB, one server with 4x A6000 48 GB.
   - Baselines: LangChain applications served through FastChat with HuggingFace Transformers or vLLM as engine, in latency-centric and throughput-centric configurations; vLLM paged-attention prefix sharing.
   - Headline: "up to an order-of-magnitude improvement"; MetaGPT up to 11.7x vs latency-centric and 2.45x vs throughput-centric baseline; chain summarisation up to 1.38x / 1.88x vs vLLM / HuggingFace; GPTs serving up to 12x higher request rate.
4. Fits 4x48 GB? Yes (assessment): the multi-GPU server in the paper is 4x A6000 48 GB and both models fit a single 48 GB card; the Bing Copilot/GPTs prompt sets would need to be obtained or substituted.
5. Stated limitations — [arXiv 2405.19888](https://arxiv.org/html/2405.19888): "Parrot only supports cloud-side orchestration of LLM requests without involving dynamic control flow", with extensions left "as future works"; serving 32k or 1M-token contexts with satisfactory latency "is beyond the scope of this paper"; the paper focuses "on Parrot's mechanisms and a few use cases, leaving other optimizations as promising future works".
6. Follow-ups: Agentix/Autellix (A2), Ayo (A3); METIS uses a "Parrot*" baseline — [arXiv 2412.10543](https://arxiv.org/html/2412.10543); Dynasor compares against Parrot online — [arXiv 2412.20993](https://arxiv.org/html/2412.20993); HexAGenT (arXiv, May 2026) groups "Parrot, Hermes, Autellix, and Continuum" as program-aware systems — [arXiv 2605.16637](https://www.alphaxiv.org/abs/2605.16637)

**A2. Agentix: An Efficient Serving Engine for LLM Agents as General Programs (arXiv title: Autellix) — NSDI 2026**
1. Links: [NSDI '26 page](https://www.usenix.org/conference/nsdi26/presentation/luo), [PDF](https://www.usenix.org/system/files/nsdi26-luo.pdf), [arXiv 2502.13965](https://arxiv.org/html/2502.13965). Code: no artifact listed on the NSDI page and no repository linked in the arXiv version — [NSDI '26 page](https://www.usenix.org/conference/nsdi26/presentation/luo)
2. Problem: engines such as vLLM optimise single calls and ignore program context, so "programs submitted to LLM serving engines experience long cumulative wait times" from head-of-line blocking at call and program level; affects agent developers/users (slow programs) and serving providers (GPU throughput) — [arXiv 2502.13965](https://arxiv.org/html/2502.13965)
3. Setup (arXiv v1; the NSDI camera-ready was not read and may differ) — [arXiv 2502.13965](https://arxiv.org/html/2502.13965):
   - Models: LLaMA-3.1-8B on 1 GPU, LLaMA-3.1-70B on 4 GPUs, Falcon-180B on 8 GPUs.
   - Workloads: ShareGPT (public), Berkeley Function Calling Leaderboard v3 (public), LATS traces "derived from running MCTS on HotpotQA" (release not stated), and an equal mix; arrivals are synthesised by sampling programs with Poisson arrivals.
   - Hardware: one GCP a2-ultragpu-8g instance, 8x A100-SXM4-80GB with NVLink, 1360 GB host memory.
   - Baselines: vLLM v0.6.1 (FCFS), vLLM-opt (chunked prefill, prefix caching, multi-step scheduling), MLFQ on vLLM-opt; Round Robin and Least Used for multi-engine load balancing.
   - Headline: program throughput "4-15x" at the same latency vs vLLM; mixed workload up to 15x vs vLLM, 5.5x vs MLFQ, 4x vs vLLM-opt; load balancer up to 1.4x (body) / 1.5x (abstract).
4. Fits 4x48 GB? Scaled down (assessment): the 8B/1-GPU experiments carry over; the 70B (4x80 GB) and Falcon-180B (8x80 GB) experiments must be dropped or replaced with a model of at most ~30B; multi-engine experiments are limited to 4 replicas. The scheduler itself must be reimplemented because no code is public.
5. Stated limitations — [arXiv 2502.13965](https://arxiv.org/html/2502.13965): graph-level optimisations are deferred ("We defer such optimizations to future works"); "the current frontend lacks safeguards against user modification of the underlying package; addressing this limitation remains future work"; "a noticable gap remains between Autellix and SRPT", attributed to SRPT knowing program runtimes in advance; the paper has no dedicated limitations section.
6. Follow-ups (all arXiv-only unless noted):
   - Continuum (arXiv 2511.02230) argues Autellix uses end-of-turn eviction and ignores KV-cache retention across turns — [arXiv 2511.02230](https://arxiv.org/html/2511.02230v2)
   - ThunderAgent (arXiv 2602.13692) says Autellix "ignores workflow locality"; it compares experimentally against vLLM and Continuum, not Autellix — [arXiv 2602.13692](https://arxiv.org/html/2602.13692)
   - MARS (arXiv 2604.26963) says program-aware frameworks like Autellix "remain physically resource-agnostic" and reports up to 5.94x lower mean end-to-end latency than its baselines — [arXiv 2604.26963](https://arxiv.org/html/2604.26963v1)
   - HexAGenT (arXiv 2605.16637) and the AgentServeSim simulator (arXiv 2606.09613) position against Autellix and Continuum — [arXiv 2605.16637](https://www.alphaxiv.org/abs/2605.16637); [arXiv 2606.09613](https://arxiv.org/pdf/2606.09613)
   - KAIROS (arXiv 2604.16682), Nalar (arXiv 2601.05109) and Aragog (arXiv 2511.20975) are listed next to it in a curated list; their relation to Agentix was not checked — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md)

**A3. Towards End-to-End Optimization of LLM-based Applications with Ayo (arXiv title: Teola) — ASPLOS 2025 (venue not confirmed on ACM DL)**
1. Links: [arXiv 2407.00326](https://arxiv.org/html/2407.00326). Code: https://github.com/NetX-lab/Ayo — [arXiv 2407.00326](https://arxiv.org/html/2407.00326)
2. Problem: LlamaIndex/LangChain-style frameworks chain whole task modules, so optimisation stays inside each module and scheduling is per request; non-LLM modules can take "even more than 50%" of latency in document QA with RAG; affects developers who deploy LLM applications on locally hosted engines — [arXiv 2407.00326](https://arxiv.org/html/2407.00326)
3. Setup — [arXiv 2407.00326](https://arxiv.org/html/2407.00326):
   - Models: gemma-2-2B, llama-2-7B, llama-2-13B, "llama-30B", bge-large-en-v1.5 embeddings, bge-reranker-large, all half precision.
   - Workloads: four applications (search-engine-empowered generation, naive RAG, advanced RAG, contextual retrieval) with Poisson request streams built from web_question, HotpotQA, FinQABench and TruthfulQA (public datasets; the generated request traces are not stated to be released).
   - Hardware: RTX 3090 24 GB cards (1, 1 and 2 per instance for 2B, 7B, 13B) and 2x A800 80 GB per llama-30B instance; two instances per LLM, one per non-LLM engine; 100 Gbps between servers; node count not given.
   - Baselines: LlamaDist (Ray-based LlamaIndex), LlamaDistPC (parallel + prefix-cache reuse), AutoGen, each under a per-invocation-oriented and a throughput-oriented scheduler.
   - Headline: up to 2.09x end-to-end speedup (advanced RAG), 1.79x (search-engine app), 1.62-1.67x (naive RAG), 1.06-1.59x (contextual retrieval).
4. Fits 4x48 GB? Scaled down (assessment): the 2B/7B/13B configurations fit; "llama-30B" at half precision is about 60 GB, so each instance needs two 48 GB cards and the paper's two-instance setup would use all four GPUs, leaving none for embedding/reranker engines — run one instance or drop that model.
5. Stated limitations — [arXiv 2407.00326](https://arxiv.org/html/2407.00326): "Teola's ahead-of-time graph optimization would struggle with dynamic workflows", and "Adapting Teola for such dynamic workflows is a very promising avenue for future research"; "To enable finer-grained orchestration, we have to modify several engine-side mechanisms" (a performance/modularity trade-off); prioritising critical nodes "requires accurate online predictions of critical paths"; for co-located applications "This leaves out the potential for further optimization" such as cross-application KV-cache sharing; "finding the optimal split size is time-consuming".
6. Follow-ups: Murakkab (A4), Patchwork (arXiv 2505.07833) and Circinus (arXiv 2504.16397) appear in the same curated category; whether they compare against Ayo was not checked — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md)

**A4. Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms — OSDI 2026**
1. Links: [OSDI '26 sessions](https://www.usenix.org/conference/osdi26/technical-sessions), [arXiv 2508.18298](https://arxiv.org/html/2508.18298). Code: none stated in the arXiv version — [arXiv 2508.18298](https://arxiv.org/html/2508.18298)
2. Problem: agentic workflows are exposed to the platform as opaque sequences of model and tool calls that couple agent logic to model and hardware choices; developers must navigate a large configuration space without system visibility, end users care about cost, cloud providers about resource efficiency — [arXiv 2508.18298](https://arxiv.org/html/2508.18298)
3. Setup (arXiv v2, Sep 2025; the OSDI camera-ready was not read) — [arXiv 2508.18298](https://arxiv.org/html/2508.18298):
   - Models: Gemma-3-27B (static baseline), DeepSeek-Qwen-32B (code generation), Whisper, an object detector (text says OmDet, a listing says CLIP), LLaVA-OneVision-7B, NVLM-D-72B, Phi-4.
   - Workloads: video Q/A, code generation and math Q/A workflows, with quality measured on VideoMME, HumanEval and MATH. Because public agent-workflow traces were not available, arrivals are approximated with Azure LLM inference traces for chat and coding over 24 hours (2024-05-15 08:00 to 2024-05-16 08:00).
   - Hardware: Azure VMs with 8x A100 80 GB or 8x H100 80 GB; the static baseline uses 2,560 A100 GPUs; a constrained experiment uses 2,000 A100s plus 0-500 H100s. The optimiser is a MILP solved with Gurobi under a 300 s limit.
   - Baselines: a hand-crafted "Static" policy representing "existing systems like LangGraph"; Murakkab Optimized; Murakkab Optimized + Multiplexing.
   - Headline: GPU usage down "by up to 2.8x", energy "by 3.7x", cost "by 4.3x" while meeting SLOs (2,560 GPUs / 80.4 MWh / $201.5k vs 908 GPUs / 21.6 MWh / $46.5k).
4. Fits 4x48 GB? No for the headline result (assessment): it is a cluster-scale provisioning result over thousands of GPUs. Scaled down: per-configuration profiling of one workflow with the 27B/32B models is possible; Gurobi needs a licence.
5. Stated limitations — [arXiv 2508.18298](https://arxiv.org/html/2508.18298): no limitations section; future work is "extending adaptive co-optimization across larger clusters" and "exploring workload specialization for emerging classes of agentic applications"; on the baseline, "A non-hand-crafted baseline could be significantly more inefficient than our current baseline"; provisioning a new instance "is assumed to take 20 minutes".
6. Follow-ups: none found in this session.

**A5. Pie: A Programmable Serving System for Emerging LLM Applications — SOSP 2025**
1. Links: [SOSP 2025 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html), [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1), DOI 10.1145/3731569.3764814. Code: https://github.com/pie-project/pie — [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1)
2. Problem: vLLM/TGI-style systems have "an inflexible, monolithic token generation loop", which leaves three gaps — application-specific KV-cache control, customisable generation, and integrated computation and I/O; affects developers of agentic workflows and custom reasoning strategies — [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1)
3. Setup — [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1):
   - Models: Llama 3 at 1B, 3B and 8B, BF16.
   - Workloads: agentic workflows (ReACT, CodeACT, Swarm), text completion, Tree/Recursion/Graph/Skeleton-of-Thought, attention variants, beam search, EBNF-constrained decoding, speculative decoding; tasks are simplified versions of the original papers' tasks (arithmetic for ToT/RoT, document summarisation for GoT); no named dataset or trace.
   - Hardware: one GCP g2-standard-32 VM with 1x NVIDIA L4 24 GB.
   - Baselines: vLLM v0.6.0, SGLang v0.4.4, LMQL v0.7.3, StreamingLLM; Pie, vLLM and SGLang all use the FlashInfer backend.
   - Headline: 3-12% latency overhead on standard tasks; agentic workflows 1.3x-3.4x higher throughput (abstract; the introduction gives 1.1x-2.4x for latency); generation strategies up to 28% lower latency and 34% higher throughput.
4. Fits 4x48 GB? Yes (assessment): the whole evaluation is one 24 GB GPU with models of at most 8B, and code is public.
5. Stated limitations — [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1): "The Pie API is designed for Transformer-based LLMs" and handlers "currently support the Llama family of models"; "A comprehensive security treatment is beyond this paper's scope, but hardening is essential for commercial adoption of Pie"; "Real deployments must span multiple GPU nodes, raising challenges in load balancing", requiring "distributed coordination, globally-aware scheduling, and cache-aware placement policies"; the FCFS contention policy "terminates the most recently started inferlet to free resources", with richer policies (per-inferlet quotas, KvPage swapping) as "promising future work"; "more incremental, coarser-grained programmability layers that better fit existing designs remain an interesting avenue for future work".
6. Follow-ups: "Serve Programs, Not Prompts" (arXiv 2510.25412) and Vortex (arXiv 2606.06453) are listed nearby in a curated list; relation not checked — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md)

**A6. InferCept: Efficient Intercept Support for Augmented Large Language Model Inference — ICML 2024**
1. Links: [PMLR page](https://proceedings.mlr.press/v235/abhyankar24a.html), [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2). Code: https://github.com/WukLab/InferCept — [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2)
2. Problem: inference systems treat each tool/API call as the end of generation, discard the context and recompute it afterwards; this recomputation "accounts for 37-40% of total model forwarding time"; affects operators serving tool-augmented LLMs — [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2)
3. Setup — [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2):
   - Models: GPT-J-6B (1 GPU), Vicuna-13B (1 GPU, or 2 with tensor parallelism), Llama3-70B (4 GPUs).
   - Workloads: a uniform mix of six augmentation types — math (GSM8K-XL), multi-hop QA over Wikipedia (HotpotQA-based), virtual environment (ALFWorld), chatbot (ShareGPT), image generation (Stable Diffusion) and TTS (Bark) with ChatGPT-generated prompts (release of the last two not stated).
   - Hardware: NVIDIA A100 GPUs (memory size and node count not stated).
   - Baselines: vLLM (Discard), ImprovedDiscard, Preserve, Swap.
   - Headline: throughput "by 1.6x-2x" and "completes 2x more requests per second"; 70B: 2x higher arrival rate and 1.3x-12x lower normalised latency.
4. Fits 4x48 GB? Scaled down (assessment): the 6B and 13B experiments carry over; the 70B/4-GPU experiment exceeds the ~30B limit and should be replaced by a smaller model.
5. Stated limitations — [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2): "We leave the comparison of these scheduling policies to future work" (e.g. FastServe's MLFQ); gains on 13B/1 GPU are smaller because larger weights leave less KV-cache room; no limitations section.
6. Follow-ups: "Fast Inference for Augmented Large Language Models" (arXiv 2410.18248) is listed in the same category — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md); Continuum and ThunderAgent address the same pause-during-tool-call problem with KV time-to-live and tool-lifecycle awareness — [arXiv 2511.02230](https://arxiv.org/html/2511.02230v2); [arXiv 2602.13692](https://arxiv.org/html/2602.13692)

**A7. An LLM Compiler for Parallel Function Calling (LLMCompiler) — ICML 2024**
1. Links: [PMLR page](https://proceedings.mlr.press/v235/kim24y.html), [arXiv 2312.04511](https://arxiv.org/html/2312.04511). Code: https://github.com/SqueezeAILab/LLMCompiler — [arXiv 2312.04511](https://arxiv.org/html/2312.04511)
2. Problem: "current methods for function calling often require sequential reasoning and acting for each function", giving high latency and cost; targets users of black-box LLM services who cannot modify the model or the inference pipeline — [arXiv 2312.04511](https://arxiv.org/html/2312.04511)
3. Setup — [arXiv 2312.04511](https://arxiv.org/html/2312.04511):
   - Models: gpt-3.5-turbo (1106), gpt-4-turbo (1106), gpt-4 (0613), and LLaMA-2 70B served with vLLM.
   - Workloads: HotpotQA comparison dev set (~1.5k questions), Movie Recommendation (500), ParallelQA (113, the authors' own benchmark), Game of 24 (100), WebShop (500).
   - Hardware: LLaMA-2 70B "hosted on 2 A100-80GB GPUs".
   - Baselines: ReAct, ReAct with anti-looping prompts, OpenAI parallel function calling, Tree-of-Thoughts, LATS, LASER.
   - Headline: "latency speedup of up to 3.7x", "cost savings of up to 6.7x", accuracy "of up to ~9%" vs ReAct.
4. Fits 4x48 GB? Scaled down for the open-model rows (replace LLaMA-2 70B with a model of at most ~30B); no for the GPT rows, which need a proprietary API with specific dated model versions (assessment).
5. Stated limitations — [arXiv 2312.04511](https://arxiv.org/html/2312.04511): speedup is below N-fold "due to the overhead associated with LLMCompiler's Planner and final answering process that cannot be parallelized"; on WebShop it is "slightly slower than ReAct on this benchmark, mainly due to the Planner overhead"; for Game of 24 "solving this problem with a single static graph is not feasible" (hence replanning); future work includes "optimizing the structure of the agent scratchpad, rather than simply appending observations".
6. Follow-ups: used as one of five agent workloads in the HPCA '26 characterisation (D1) — [arXiv 2506.04301](https://arxiv.org/html/2506.04301); "Concurrency without Model Changes: Future-based Asynchronous Function Calling for LLMs" (arXiv 2605.15077) and "Agent JIT Compilation" (labelled ICML '26 in the list; venue not verified) continue the parallel/asynchronous tool-calling line — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md)

#### B. RAG serving systems

**B1. METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation — SOSP 2025**
1. Links: [SOSP 2025 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html), [arXiv 2412.10543](https://arxiv.org/html/2412.10543), DOI 10.1145/3731569.3764855. Code: no public code found (none on the arXiv page; a search returned no repository).
2. Problem: more retrieved context raises quality and delay together; prior systems optimise either delay (scheduling) or quality (configuration tuning) but not the trade-off per query; affects RAG service operators — [arXiv 2412.10543](https://arxiv.org/html/2412.10543)
3. Setup — [arXiv 2412.10543](https://arxiv.org/html/2412.10543):
   - Models: fine-tuned Mistral-7B-v3 (main) and Llama3.1-70B, both AWQ-quantised; query profilers GPT-4o and Llama-3.1-70B; Cohere embed-v3.0 embeddings; FAISS retrieval.
   - Workloads: Squad, Musique, KG RAG FinSec (LlamaHub) and QMSUM, 200 queries per dataset, Poisson arrivals.
   - Hardware: one server with 2x NVIDIA A40 (1 GPU for Mistral-7B, 2 for Llama3.1-70B).
   - Baselines: vLLM with fixed configurations, Parrot* (configuration-based batching), AdaptiveRAG* (query-complexity-based selection).
   - Headline: delay 1.64-2.54x lower than AdaptiveRAG* with no F1 loss; throughput 1.8-4.5x higher than fixed configurations at similar quality; F1 12-18% higher at similar delay.
4. Fits 4x48 GB? Yes for the Mistral-7B setting (the A40 is a 48 GB card); scaled down for the rest: Llama3.1-70B exceeds the ~30B limit, and the GPT-4o profiler and Cohere embeddings are proprietary APIs that need open substitutes (the paper also uses Llama-3.1-70B as profiler). The system must be reimplemented (assessment).
5. Stated limitations — [arXiv 2412.10543](https://arxiv.org/html/2412.10543): "Such 90% threshold can be tuned for better performance, and we leave it to future work"; "more work will be needed to complement it for better interpretability and robustness"; for highly under-specified queries it is hard for any model to estimate the profile; extension to agentic, multimodal and graph-based RAG is claimed possible but not evaluated.
6. Follow-ups: PCR, "A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving" (arXiv 2603.23049), surfaced as a 2026 RAG-serving preprint; its relation to METIS was not checked — [arXiv 2603.23049](https://arxiv.org/pdf/2603.23049)

**B2. HedraRAG: Co-Optimizing Generation and Retrieval for Heterogeneous RAG Workflows — SOSP 2025**
1. Links: [SOSP 2025 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html), [arXiv 2507.09138](https://arxiv.org/html/2507.09138) (arXiv title: "HedraRAG: Coordinating LLM Generation and Database Retrieval in Heterogeneous RAG Serving"). Code: an artifact-evaluation repository "Leo9660/HedraRAG_AE" appeared in search via a mirror; not opened, unverified.
2. Problem: multi-stage RAG workflows (multi-step, iterative, HyDE, RECOMP) on CPU-GPU systems suffer because "dynamic and imbalanced workloads often lead to misaligned execution and CPU-GPU hybrid system under-utilization"; affects operators of RAG serving — [arXiv 2507.09138](https://arxiv.org/html/2507.09138)
3. Setup — [arXiv 2507.09138](https://arxiv.org/html/2507.09138):
   - Models: Llama 3.1-8B (primary), Llama2-13B, OPT-30B.
   - Workloads: NaturalQuestions, 2WikiMultiHopQA, HotpotQA (open datasets) over a Wikipedia corpus of ~38M passages, e5_large 1024-d embeddings, IVF4096 index, nprobe 128/256/512; workflows One-shot, Multistep, IRG, HyDE, RECOMP.
   - Hardware: AMD EPYC 9534 64-core CPU for vector search, "NVIDIA H100 GPUs with 80 GB of memory" (count and node count not given).
   - Baselines: LangChain, FlashRAG; speculation compared with RaLMSpec and RAGCache re-implemented inside HedraRAG.
   - Headline: speedups "exceeding 1.5x" and "up to 5x"; online latency 2.2x-18.2x lower at the same request rate; offline 3.5x over LangChain and 1.3x over FlashRAG.
4. Fits 4x48 GB? Scaled down (assessment): 8B and 13B fit one 48 GB card and OPT-30B needs two; what changes is the GPU class (H100 to A6000), the CPU budget for IVF search (64 cores in the paper) and host RAM for a 38M x 1024-d index (about 156 GB as raw float32).
5. Stated limitations — [arXiv 2507.09138](https://arxiv.org/html/2507.09138): no limitations or future-work section; the authors note that "extremely large models paired with small databases tend to shift the performance bottleneck toward LLM inference".
6. Follow-ups: PCR (arXiv 2603.23049) and Patchwork (arXiv 2505.07833) are neighbouring RAG-serving preprints; relation not checked — [arXiv 2603.23049](https://arxiv.org/pdf/2603.23049); [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md)

**B3. RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving — ISCA 2025 (per the official repository)**
1. Links: [arXiv 2503.14649](https://arxiv.org/html/2503.14649). Code: https://github.com/google/rago (Apache 2.0; retrieval model based on ScaNN) — [github.com/google/rago](https://github.com/google/rago)
2. Problem: efficient RAG serving "remains an open challenge" because of many RAG variants with different workload characteristics; target is data-centre RAG serving ("As companies race to integrate RAG systems into their production pipelines, optimizing their performance has become increasingly critical") — [arXiv 2503.14649](https://arxiv.org/html/2503.14649)
3. Setup — [arXiv 2503.14649](https://arxiv.org/html/2503.14649):
   - Models: Llama-3 1B, 8B, 70B, 405B (8-bit), an 8B query rewriter, 120M reranker and encoder.
   - Workloads: four RAG paradigms ("cases") over a 64-billion-passage corpus (768-d, PQ-compressed, 5.6 TiB); sequence lengths "derived from QA datasets"; contexts of 100K-10M tokens in the long-context case.
   - Hardware: results come from a performance model, not measured GPUs — XPUs modelled on TPU v5p (96 GB HBM), "16 to 32 servers hosting 64 to 128 XPUs (4 XPUs per server)", AMD EPYC Milan hosts.
   - Baseline: a tuned "LLM-system extension" where RAG components are collocated with the LLM prefix stage.
   - Headline: "up to a 2x" QPS per chip and 55% lower TTFT.
4. Fits 4x48 GB? Yes in the sense that the published analysis is a simulator with public code and needs no GPUs; no if the goal is to measure the same configurations on real hardware (64B-passage index, 405B model). Validating the model on 4x A6000 would be a new, scaled-down study (assessment).
5. Stated limitations — [arXiv 2503.14649](https://arxiv.org/html/2503.14649): "we leave more complex strategies such as simultaneous execution as future work"; "While RAGSchema abstracts RAG serving workloads, it is not an abstraction for quality, as different models and databases of the same size can lead to varying quality"; "Collocation is restricted to consecutive neighbors to avoid excessively complicating the search space"; assumes "a data center model serving environment with abundant resources". The conclusion fell in the unread tail (not seen).
6. Follow-ups: HedraRAG and TeleRAG (measured systems on the same RAG pipelines) and Patchwork (arXiv 2505.07833) — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md)

**B4. TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval — MLSys 2026 (paper-stated)**
1. Links: [arXiv 2502.20969](https://arxiv.org/html/2502.20969). Code: https://github.com/uw-syfi/TeleRAG; artifact DOI https://doi.org/10.5281/zenodo.19361856; pre-generated traces and index at https://huggingface.co/datasets/lauyeeyu/TeleRAG-Dataset — [arXiv 2502.20969](https://arxiv.org/html/2502.20969)
2. Problem: in RAG pipelines "achieving high throughput and low latency is difficult, especially when GPU memory is limited", because large IVF indexes on the GPU take memory away from the LLM's KV cache; affects local single-GPU deployments and data centres — [arXiv 2502.20969](https://arxiv.org/html/2502.20969)
3. Setup — [arXiv 2502.20969](https://arxiv.org/html/2502.20969):
   - Models: Llama-3.2-3B, Llama-3-8B, Mistral-Small-22B; Contriever embeddings.
   - Workloads: NQ, HotpotQA, TriviaQA (1024 sampled queries each) over wiki_dpr (61 GB index, 4096 IVF clusters); six pipelines — HyDE, SubQ, Iter, IRG, FLARE, Self-RAG; pipeline traces generated with GPT-3.5-Turbo and released.
   - Hardware: desktop with 1x RTX 4090 24 GB (PCIe 4); server with 1x H100 80 GB (PCIe 5); server with 8x H200 (single node).
   - Baseline: CPU-offloaded Faiss retrieval with SGLang generation.
   - Headline: abstract "up to a 1.98x average end-to-end latency reduction (single-query) and 1.83x higher average throughput (batched)"; body gives a 1.53x average single-query speedup on the 4090 and 1.98x throughput at batch 8 on the H100 (the two do not line up in the extracted text).
4. Fits 4x48 GB? Scaled down (assessment): single-GPU experiments carry over (48 GB sits between the 24 GB and 80 GB cards used; the method moves IVF clusters over PCIe, so the PCIe generation changes the numbers); the 8-GPU scaling experiment stops at 4 GPUs.
5. Stated limitations — [arXiv 2502.20969](https://arxiv.org/html/2502.20969): "failure cases can occur (e.g., when the rewrite substantially shifts embedding vectors)"; "For cases where the budget is less than 2 GB, we observe a relatively low hit rate"; "The benefit on a single GPU is marginal because only a small cache space is used" (caching); "TeleRAG is designed for IVF-style retrieval indices" and HNSW "does not naturally expose explicit prefetchable units"; on unified-memory machines "TeleRAG is less useful as we can do GPU-only retrieval on those"; "We leave the exploration of different parallelisms for future work".
6. Follow-ups: PCR (arXiv 2603.23049) uses prefetching for RAG cache reuse; relation not checked — [arXiv 2603.23049](https://arxiv.org/pdf/2603.23049)

#### C. Reasoning-model serving and structured decoding

**C1. Breaking the Reward Barrier: Accelerating Tree-of-Thought Reasoning via Speculative Exploration (SPEX) — OSDI 2026**
1. Links: [OSDI '26 page](https://www.usenix.org/conference/osdi26/presentation/zhong), [arXiv 2605.10195](https://arxiv.org/html/2605.10195). Code: https://github.com/PKU-SEC-Lab/SPEX (built on SGLang) — [arXiv 2605.10195](https://arxiv.org/html/2605.10195)
2. Problem: reward-guided tree search is slowed by "a synchronization bottleneck caused by sequential reward-guided exploration" (the next thoughts cannot be generated until the current ones are scored); affects anyone serving ToT-style reasoning on math and programming tasks — [arXiv 2605.10195](https://arxiv.org/html/2605.10195)
3. Setup — [arXiv 2605.10195](https://arxiv.org/html/2605.10195):
   - Models: Llemma-7B, Llemma-34B, DeepSeek-R1-8B, Qwen3-30B-A3B.
   - Workloads: GSM8K and MATH-500 (Llemma); AIME 2024, AIME 2025, BRUMO, HMMT (DeepSeek, Qwen); algorithms REST-MCTS, RSTAR-MCTS, REBASE, ETS at two search widths.
   - Hardware: NVIDIA A6000 for Llemma-7B and DeepSeek-R1-8B; NVIDIA A100 for Llemma-34B and Qwen3-30B-A3B (counts not given).
   - Baseline: the unmodified ToT algorithm on the same engine (labelled "Baseline"; not defined more precisely in the extracted text).
   - Headline: 1.2-3x speedup across ToT algorithms; up to about 4.1x combined with token-level speculative decoding.
4. Fits 4x48 GB? Yes for the 7B/8B experiments, which were run on A6000 in the paper; scaled down for Llemma-34B and Qwen3-30B-A3B, which ran on A100 and at 16-bit need two 48 GB cards or quantisation (assessment).
5. Stated limitations — [arXiv 2605.10195](https://arxiv.org/html/2605.10195): no limitations section; on skewed trees SPEX "struggles to achieve significant acceleration"; the speculation hit rate "decreases as the prediction distance increases"; for DeepSeek and Qwen the authors "derived reward signals directly from intrinsic log probabilities to serve as confidence proxies" instead of a reward model; at larger batch sizes the system is compute-bound, "reducing SPEX's relative benefits".
6. Follow-ups: none found (the paper is from May 2026).

**C2. Efficiently Scaling LLM Reasoning Programs with Certaindex (Dynasor) — NeurIPS 2025**
1. Links: [NeurIPS poster](https://neurips.cc/virtual/2025/poster/116107), [arXiv 2412.20993](https://arxiv.org/html/2412.20993). Code: https://github.com/hao-ai-lab/Dynasor (built on SGLang v0.3.3) — [arXiv 2412.20993](https://arxiv.org/html/2412.20993)
2. Problem: test-time reasoning algorithms and reasoning models "can wastefully generate many tokens without improving accuracy"; affects serving operators and users paying for reasoning compute — [arXiv 2412.20993](https://arxiv.org/html/2412.20993)
3. Setup — [arXiv 2412.20993](https://arxiv.org/html/2412.20993):
   - Models: Qwen2.5 7B, DeepSeek-R1-distilled Qwen 7B/14B/32B, Llama 3.1 8B, QwQ, Gemma, Phi.
   - Workloads: MATH-500, AIME24, AMC23, GSM8K, ASDiv, LiveCodeBench, GAME24, MATH; reasoning programs SC, MCTS, REBASE, CoT; "3 online workloads" whose construction is in an appendix that was not seen.
   - Hardware: not found in the part of the paper that was read.
   - Baselines: SGLang and Parrot (online); even and length-based budget allocation (batch).
   - Headline: "up to 50% compute savings and 3.3x" higher sustainable request rate online with no accuracy drop; CoT token use down 11-29%.
4. Fits 4x48 GB? Probably yes (assessment): all models are 32B or smaller and code is public; the hardware used is a gap, so the original rates cannot be compared directly.
5. Stated limitations — [arXiv 2412.20993](https://arxiv.org/html/2412.20993): the authors "did not explore its integration with advanced serving techniques like PD disaggregation or chunked prefill"; the latency-accuracy impact "remains an open area for future research"; "Thresholding-based allocation is not strictly optimal"; risks named are "bias in early exits, security vulnerabilities through side-channel attack for certaindex in multi-tenant settings".
6. Follow-ups: SART ("Thinking Short and Right Over Thinking Long", redundant sampling with early stopping; venue not found) — [literature review](https://www.themoonlight.io/review/thinking-short-and-right-over-thinking-long-serving-llm-reasoning-efficiently-and-accurately); SPEX (C1) evaluates REBASE-style search on the same benchmarks — [arXiv 2605.10195](https://arxiv.org/html/2605.10195)

**C3. PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models — HPCA 2026 (author-stated)**
1. Links: [arXiv 2602.11530](https://arxiv.org/html/2602.11530). Code: no public code found — [arXiv 2602.11530](https://arxiv.org/abs/2602.11530)
2. Problem: reasoning models emit long hidden reasoning before the visible answer, inflating time to first (answer) token; "Existing LLM serving frameworks fail to distinguish between reasoning and answering phases"; affects users (responsiveness) and operators under GPU-memory limits — [arXiv 2602.11530](https://arxiv.org/html/2602.11530)
3. Setup — [arXiv 2602.11530](https://arxiv.org/html/2602.11530):
   - Model: DeepSeek-R1-Distill-Qwen-32B; OpenAI o4-mini queried through its API to obtain reasoning/answer token counts for the traces.
   - Workloads: AlpacaEval 2.0 and Arena-Hard prompts with o4-mini-derived length distributions; MATH-500, GPQA and LiveCodeBench for reasoning-heavy runs; Poisson arrivals of 300 requests for characterisation; trace release not stated.
   - Hardware: characterisation on 1x H100 (reported as 96 GB) with vLLM v0.6.1; cluster evaluation in a simulator of eight nodes, one H100 each, 100 Gbps fabric; the single-instance simulator is validated against the real system (1.62% MAPE end-to-end latency, 12.6% mean TTFT, 6.49% TPOT).
   - Baselines: FCFS (vLLM default) and round-robin. SLOs: 0.25 s target to first answer token, 100 ms TPOT, QoE threshold 0.95.
   - Headline: tail TTFT down by up to 72% vs FCFS (Arena-Hard) and 61% (AlpacaEval 2.0); throughput within 3% of baselines.
4. Fits 4x48 GB? Scaled down (assessment): the 32B model at 16-bit is about 64 GB, so one instance needs two 48 GB cards or a quantised model; the eight-node result is simulated and the simulator is not public; the length traces depend on a proprietary API unless regenerated from an open reasoning model.
5. Stated limitations — [arXiv 2602.11530](https://arxiv.org/html/2602.11530): "A deeper exploration of Pascal in such heterogeneous environments is beyond the scope of this work"; on partitioning resources between phases, "A deeper exploration of this design space is left for future work"; for reasoning-heavy workloads "Pascal's benefits may diminish because answering phases are too short"; occasional TTFT degradation from preemption among answering requests (worst case +13.30 s vs FCFS on AlpacaEval 2.0).
6. Follow-ups: none found (the paper is from February 2026).

**C4. XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models — MLSys 2025 (paper-stated)**
1. Links: [arXiv 2411.15100](https://arxiv.org/html/2411.15100v3). Code: the paper says "We are open-sourcing XGrammar and integrating it into major open-source LLM frameworks" but gives no URL; the MLC blog describes the open-source release — [arXiv 2411.15100](https://arxiv.org/html/2411.15100v3); [MLC blog](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)
2. Problem: "naively applying CFG to constrained decoding is not efficient because of its flexible nature" — per-token vocabulary checks and stack tracking dominate; motivated by agent/function-calling applications that need structured output — [arXiv 2411.15100](https://arxiv.org/html/2411.15100v3)
3. Setup — [arXiv 2411.15100](https://arxiv.org/html/2411.15100v3):
   - Models: Llama-3.1-8B-Instruct, DeepSeek-V2-Lite (16B MoE); Qwen-2.5-0.5B on iPhone and a 4-bit Llama-3.1-8B in the browser.
   - Workloads: JSON-mode-eval (NousResearch, public on Hugging Face); synthetic XML and Python-DSL datasets (release not stated); function calling and XML code generation for accuracy.
   - Hardware: mask generation on AMD Ryzen 9 7950X + RTX 4090; end-to-end serving on AMD EPYC 7R13 + NVIDIA H100.
   - Baselines: Outlines v1.0, llama.cpp grammar (b3998), lm-format-enforcer v0.10.9; vLLM v0.6.3 + Outlines, SGLang + Outlines, MLC-LLM.
   - Headline: "up to 3x speedup on JSON Schema and over 100x on CFG, compared to the best baseline in each case"; "up to 80x output token rate compared to existing solutions" end to end; per-token mask latency under 40 µs for JSON.
4. Fits 4x48 GB? Yes (assessment): mask generation is CPU-side and the end-to-end runs use one GPU with an 8B/16B model; absolute token rates will differ from the H100.
5. Stated limitations — [arXiv 2411.15100](https://arxiv.org/html/2411.15100v3): no limitations section; some batch-size-32 baseline results were omitted because "their API call time exceeded the API timeout limit of 600 seconds".
6. Follow-ups: XGrammar-2, "Dynamic and Efficient Structured Generation Engine for Agentic LLMs" (ACM CAIS '26 per arXiv comment), reports "over 6x faster compilation" and near-zero end-to-end overhead — [arXiv 2601.04426](https://arxiv.org/abs/2601.04426); Pie includes EBNF-constrained decoding among its workloads — [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1)

#### D. Workload characterisation and efficiency benchmarks for agents

**D1. The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective — HPCA 2026 (author-stated)**
1. Links: [arXiv 2506.04301](https://arxiv.org/html/2506.04301). Code: benchmark "Open-sourced at https://github.com/VIA-Research/AgentBench" — [arXiv 2506.04301](https://arxiv.org/html/2506.04301)
2. Problem: multi-turn agent workflows raise "serious concerns about system-level cost, efficiency, and sustainability", while "the infrastructure implications of dynamic reasoning workloads" are "underexplored"; audience is infrastructure/data-centre operators and agent developers — [arXiv 2506.04301](https://arxiv.org/html/2506.04301)
3. Setup — [arXiv 2506.04301](https://arxiv.org/html/2506.04301):
   - Models: Llama-3.1-8B-Instruct (default) and Llama-3.1-70B-Instruct on vLLM 0.6.6 with prefix caching.
   - Agents: CoT, ReAct, Reflexion, LATS, LLMCompiler (official implementations; LATS modified for concurrent calls).
   - Workloads: HotpotQA (Wikipedia APIs), WebShop, MATH (Wolfram Alpha and a Python calculator), HumanEval; ShareGPT as the non-agent baseline.
   - Hardware: GCP a2-highgpu-1g with 1x A100 40 GB (8B); a2-highgpu-8g with 8x A100 40 GB (70B).
   - Headline: agents make about 9.2x more LLM calls than CoT; LLM inference 69.4% and tools 30.2% of latency on average; GPU idle "up to 54.5%" of execution time; prefix caching cuts prefill latency 60.1% on average; per-query GPU energy 62.1x-136.5x that of a ShareGPT query.
4. Fits 4x48 GB? Yes for the 8B half (one 40 GB card in the paper); scaled down for the 70B half (replace with a model of at most ~30B). Wolfram Alpha needs an API key (assessment).
5. Stated limitations — [arXiv 2506.04301](https://arxiv.org/html/2506.04301): "we did not (and realistically could not) conduct our analysis with respect to a specific SLA target"; "A detailed exploration of efficient agent serving under SLA constraints is left as future work"; the energy analysis "includes only GPU energy, omitting overheads from CPU, memory, networking, storage, and cooling"; "our analysis does not account for LLM request batching".
6. Follow-ups: D2 below; "Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective" (arXiv 2511.00739, venue not found) — [arXiv 2511.00739](https://arxiv.org/html/2511.00739v3)

**D2. Agentic AI Workload Characteristics — IISWC 2026 (author-stated)**
1. Links: [arXiv 2605.26297](https://arxiv.org/html/2605.26297). Artifacts: source at https://doi.org/10.6084/m9.figshare.33188466.v1, raw and processed data at https://doi.org/10.6084/m9.figshare.33188042.v1 — [arXiv 2605.26297](https://arxiv.org/html/2605.26297)
2. Problem: how stateful multi-turn agent loops change LLM serving compared with single prompt-generation requests; audience is serving-system designers — [arXiv 2605.26297](https://arxiv.org/abs/2605.26297)
3. Setup — [arXiv 2605.26297](https://arxiv.org/html/2605.26297):
   - Models: Qwen3.6-27B and Gemma4-31B, each in thinking and non-thinking ("Instant") mode, on vLLM v0.20.0 with tensor parallelism 2 and OpenTelemetry/Jaeger tracing.
   - Harness: Claude Code launched through Harbor, with a gateway correlating requests to runs.
   - Workloads: ADE-Bench, DABStep (100 tasks), GAIA, SWE-bench Pro (100 tasks), Terminal-Bench 2.0.
   - Hardware: 2x NVIDIA H100 NVL, Intel Xeon Platinum 8592+.
   - Headline: cache-hit ratios 84.6-99.5% and decode 91.0-98.6% of LLM time without thrashing; LLM inference 71-98% of runtime (tools up to 28.7% on GAIA); under cache thrashing hit rate fell from about 98% to 16-26% and generation throughput dropped by up to 66%; failed runs have contexts up to 1.8x those of successful runs.
4. Fits 4x48 GB? Yes with caveats (assessment): the models are within the ~30B range and the TP=2 layout maps onto two 48 GB cards, but with about half the GPU memory of 2x H100 NVL the KV-cache headroom is smaller, so the thrashing thresholds will move.
5. Stated limitations — [arXiv 2605.26297](https://arxiv.org/html/2605.26297): the study "focuses on a representative set of models and mainstream ReAct-style agent execution"; "Advanced orchestration techniques" (speculative tool execution, context compaction, tree search) are "outside the scope of this study" — "These techniques may change tool latency, context evolution, and KV-cache behavior", and "we leave their characterization to future work"; the decode-dominated result holds for the no-thrashing regime.
6. Follow-ups: none found (May 2026).

**D3. OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents — MLSys 2026 (oral)**
1. Links: [MLSys 2026 oral](https://mlsys.org/virtual/2026/oral/3865), [arXiv 2506.16042](https://arxiv.org/html/2506.16042v1). Code/data: reported at github.com/WukLab/osworld-human (from a search summary; the arXiv v1 text only promises a release) — [arXiv 2506.16042](https://arxiv.org/html/2506.16042v1)
2. Problem: computer-use agents "are practically unusable due to extremely high end-to-end latency (e.g. tens of minutes)"; affects users and deployers of such agents — [arXiv 2506.16042](https://arxiv.org/html/2506.16042v1)
3. Setup (arXiv v1) — [arXiv 2506.16042](https://arxiv.org/html/2506.16042v1):
   - Agent: Agent S2 with GPT-4.1 as planner/reflection/retrieval model and UI-TARS-7B-DPO for grounding, the latter "hosted on a single NVIDIA A6000 GPU using SGLang"; max 50 steps.
   - Workloads: a 37-task (10%) subset of OSWorld for the latency study; OSWorld-Human adds human-minimal trajectories for all 369 OSWorld tasks; 16 agents are scored from their published leaderboard trajectories.
   - Headline: planning and reflection account for "75% to 94%" of task latency; later steps can take 3x longer; leading agents take "1.4-2.7x" more steps than necessary (the MLSys listing reportedly says 2.7-4.3x; not reconciled).
4. Fits 4x48 GB? No as published for the latency study (the planner is a proprietary API model); the step-efficiency scoring uses published trajectories and needs no GPU; a version with an open planner would be a changed setup (assessment).
5. Stated limitations — [arXiv 2506.16042](https://arxiv.org/html/2506.16042v1): no limitations section; "While our study focuses on Agent S2, its implications go beyond a single agent"; "addressing these latency issues by optimizing agent planning and execution strategies is crucial for the widespread adoption".
6. Follow-ups: "Agent JIT Compilation: Latency-Optimizing Web Agent Planning and Scheduling" (arXiv 2605.21470, labelled ICML '26 in a curated list; venue and relation not verified) — [LLMSys-PaperList](https://raw.githubusercontent.com/AmberLJC/LLMSys-PaperList/main/README.md)

#### E. arXiv-only items (not peer-reviewed as far as I could find)

**E1. ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System — arXiv 2602.13692 (v3, June 2026)**
- Code: https://github.com/ThunderAgent-org/ThunderAgent/ — [arXiv 2602.13692](https://arxiv.org/html/2602.13692)
- Problem: agent stacks "run agentic inference by loosely assembling isolated components" (e.g. vLLM plus Kubernetes), causing KV-cache thrashing, cross-node memory imbalance and tool-lifecycle obliviousness; also targets RL rollout, where "rollout accounts for over 70% of the total wall-clock time in reinforcement learning" — [arXiv 2602.13692](https://arxiv.org/html/2602.13692)
- Setup: GLM-4.6 (355B) and Qwen-3 (235B) at FP8 with TP8; OpenHands and mini-SWE-Agent on SWE-Bench Lite, HLE via ToolOrchestra, ScienceAgentBench; 8x H100 nodes (two for RL rollout); baselines vLLM, Continuum, vLLM + SGLang Gateway; headline 1.48-3.58x over vLLM and 1.17-3.31x over Continuum in steps per minute — [arXiv 2602.13692](https://arxiv.org/html/2602.13692)
- Fits 4x48 GB? No as published (200B+ models on 8x H100); the code is public, so a small-model rerun is a changed setup (assessment).
- Stated trade-off: while a program waits on a tool, "the GPU memory occupied by acting programs is idle" — [arXiv 2602.13692](https://arxiv.org/html/2602.13692)

**E2. AgentCgroup: Understanding and Controlling OS Resources of AI Agents — arXiv 2602.09345**
- Code: https://github.com/eunomia-bpf/agentcgroup — [arXiv 2602.09345](https://arxiv.org/html/2602.09345)
- Problem: multi-tenant platforms running sandboxed coding agents see tool calls with "distinct resource demands and rapid fluctuations" that container-level, user-space, history-based controls handle poorly — [arXiv 2602.09345](https://arxiv.org/html/2602.09345)
- Setup: Claude Code on 144 SWE-rebench tasks (111 with local GLM-4.7-Flash, 33 with Claude Haiku 4.5 via API) in Podman containers with official SWE-rebench images (2.9-17.3 GB) on a 24-core desktop CPU; prototype evaluated by replaying memory traces at 50x on a patched Linux 6.19-rc kernel against a no-isolation baseline — [arXiv 2602.09345](https://arxiv.org/html/2602.09345)
- Finding: OS-level execution (tool calls plus container/agent initialisation) accounts for 56-74% of end-to-end task latency and LLM reasoning for 26-44% — [arXiv 2602.09345](https://arxiv.org/html/2602.09345)
- Stated limitations: "Our current evaluation is limited to trace replay with a proof-of-concept prototype"; "the characterization covers one agent framework and one benchmark"; "challenges such as initialization overhead, large container images, and retry-induced accumulation remain open" — [arXiv 2602.09345](https://arxiv.org/html/2602.09345)
- Fits 4x48 GB? The characterisation with a local model is feasible; the prototype needs a custom kernel (assessment).

**E3. Agentic Coding in the Wild: Characterizing GitHub Copilot at Production Scale — arXiv 2608.00101 (Microsoft, July 2026)**
- Data: one week of sampled Copilot telemetry from June 2026 — about 13.5M sessions, 3.2M users, 760.5M LLM calls, 774.7M tool calls — [arXiv 2608.00101](https://arxiv.org/html/2608.00101)
- Findings: LLM and tool calls are close to 1:1 per session; KV-cache hit rate about 90% within a turn and about 55% across turn boundaries; 7.8% of sessions compact context but account for 44.2% of tokens; median user idle between turns 1,512 s; an idle-time predictor captures 86-90% of idle time — [arXiv 2608.00101](https://arxiv.org/html/2608.00101)
- Trace release: the paper says "We plan to release sanitized traces from our characterization data to the public soon" — [arXiv 2608.00101](https://arxiv.org/html/2608.00101); the Microsoft Research page links a dataset document in the Azure Public Dataset repository (https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md), which I did not open — [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/agentic-coding-in-the-wild-characterizing-github-copilot-at-production-scale/)
- Stated limitations: "Without prompt/response content, we cannot correlate infrastructure efficiency with task completion quality"; traces are client-side only; design directions named are "session-aware scheduling, cache lifecycle management, adaptive compaction, and archetype-aware SLOs" — [arXiv 2608.00101](https://arxiv.org/html/2608.00101)
- Fits 4x48 GB? No for the study itself (private production telemetry); the released trace, if available, could drive a replay (assessment).

### Inferences
- The setups closest to the student's hardware as published are Parrot (4x A6000 48 GB), METIS (2x A40 48 GB), SPEX's 7B/8B experiments (A6000), Pie (1x L4 24 GB), XGrammar (one GPU), TeleRAG's single-GPU experiments, and the IISWC '26 characterisation (27B/31B models at TP=2).
- Public code exists for Ayo, Pie, InferCept, LLMCompiler, RAGO, TeleRAG, SPEX, Dynasor, the HPCA '26 AgentBench, the IISWC '26 artifact, ThunderAgent and AgentCgroup; I found none for Agentix, METIS, Murakkab or PASCAL, so adopting those means reimplementing the system before reproducing it.
- By count of follow-ups found, program-level agent scheduling (Agentix line: Continuum, ThunderAgent, MARS, HexAGenT, KAIROS, Nalar, AgentServeSim) is the most crowded sub-topic in 2026; reasoning-phase scheduling (PASCAL), ToT speculation (SPEX) and agent sandbox resource control (AgentCgroup) had few or no follow-ups visible, which may reflect their recency rather than lack of interest.
- Several headline numbers disagree between abstract and body in the extracted text (Agentix load balancer 1.5x vs 1.4x; TeleRAG; Pie; OSWorld-Human step ratios; Copilot token totals). These should be re-read in the PDFs before being quoted in a report.

### Gaps
- Agent sandboxes: I found no paper at OSDI/SOSP/NSDI/EuroSys/ASPLOS/ATC 2024-2026 whose subject is agent-sandbox overhead. The only academic sources were arXiv preprints (AgentCgroup; SpecBox arXiv 2607.23933 and DeepSeek DSec arXiv 2609.22978, both known to me only from secondary summaries). The ATC '26 technical-sessions URL returned 404 and the EuroSys '26, ASPLOS '26, SIGMOD/VLDB, SC and SIGCOMM programs were not opened, so this absence is not established.
- The NSDI '26 and OSDI '26 session pages were read only partially (first ~200k and ~100k characters respectively); further relevant papers may be in the unread parts.
- ICML/ICLR/NeurIPS 2026 main tracks were not searched directly; "Agent JIT Compilation" (ICML '26) and several MLSys '26 entries (AgenticCache, FlashAgents, ContextPilot, Hippocampus) are known only from a curated list and were not read.
- EuroSys '26 entries in the curated list (AIMS; "From Imperative to Declarative") and VLDB/SIGMOD RAG-serving work (e.g. SemWeave SIGMOD '26) were not read.
- RAG caching (RAGCache, CacheBlend EuroSys '25) was left to the KV/prefix-cache researcher and is not catalogued here.
- Hardware for Dynasor, GPU counts for HedraRAG and SPEX, and all artifact appendices beyond the first 100k characters were not seen.
- Venue for Ayo (ASPLOS '25), RAGO (ISCA '25), XGrammar (MLSys '25), TeleRAG (MLSys '26), PASCAL and D1 (HPCA '26) and D2 (IISWC '26) rests on the paper, arXiv comment or repository rather than the venue's program page.
- Whether any follow-up reproduces and beats a catalogued paper's numbers on the same setup was not verified for any entry.

## Which public agent or RAG workloads, benchmarks and traces do systems papers use to measure latency, throughput or cost?

### Takeaway
Systems papers mostly reuse accuracy benchmarks as request sources and add their own arrival process (almost always Poisson); real agent arrival traces were unavailable to authors until the 2026 Copilot study, whose public release status is unclear. Released, replayable artifacts exist for TeleRAG (RAG pipeline traces), the IISWC '26 study (agent serving traces) and the HPCA '26 AgentBench.

### Cited Findings
- Agent/program serving: ShareGPT, Berkeley Function Calling Leaderboard v3, and LATS-on-HotpotQA program traces with Poisson arrivals (Agentix) — [arXiv 2502.13965](https://arxiv.org/html/2502.13965)
- LLM applications: Arxiv-March document summarisation (chain and map-reduce), MetaGPT multi-agent programming, ShareGPT chat, Bing Copilot/GPTs prompts with synthesised queries (Parrot) — [arXiv 2405.19888](https://arxiv.org/html/2405.19888)
- Tool-augmented serving: GSM8K-XL, HotpotQA-based multi-hop QA, ALFWorld, ShareGPT, plus image and TTS tool calls (InferCept) — [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2)
- Parallel function calling: HotpotQA comparison set, Movie Recommendation, ParallelQA, Game of 24, WebShop (LLMCompiler) — [arXiv 2312.04511](https://arxiv.org/html/2312.04511)
- Agent characterisation with open models: HotpotQA, WebShop, MATH, HumanEval under CoT/ReAct/Reflexion/LATS/LLMCompiler, packaged as https://github.com/VIA-Research/AgentBench — [arXiv 2506.04301](https://arxiv.org/html/2506.04301)
- Coding/terminal agents on self-hosted models: ADE-Bench, DABStep, GAIA, SWE-bench Pro, Terminal-Bench 2.0 driven by Claude Code via Harbor, with collected serving traces on figshare (https://doi.org/10.6084/m9.figshare.33188042.v1) — [arXiv 2605.26297](https://arxiv.org/html/2605.26297)
- Agentic inference and RL rollout: SWE-Bench Lite with mini-SWE-Agent and OpenHands, ScienceAgentBench, HLE via ToolOrchestra, measured in steps per minute (ThunderAgent) — [arXiv 2602.13692](https://arxiv.org/html/2602.13692)
- Sandbox-side measurement: SWE-rebench tasks and container images (https://huggingface.co/datasets/nebius/SWE-rebench) (AgentCgroup) — [arXiv 2602.09345](https://arxiv.org/html/2602.09345)
- Computer-use agents: OSWorld (369 tasks) with OSWorld-Human minimal human trajectories for step-efficiency metrics — [arXiv 2506.16042](https://arxiv.org/html/2506.16042v1)
- Agentic workflows at cloud scale: VideoMME, HumanEval and MATH for quality, with Azure LLM inference traces (24 h in May 2024) standing in for arrivals because public agent-workflow traces were not available (Murakkab) — [arXiv 2508.18298](https://arxiv.org/html/2508.18298)
- Production agent trace: GitHub Copilot telemetry (timestamps, durations, token counts, model and tool names, success status; no prompt or output text); release announced as planned — [arXiv 2608.00101](https://arxiv.org/html/2608.00101); a dataset document is linked from the publication page — [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/agentic-coding-in-the-wild-characterizing-github-copilot-at-production-scale/)
- RAG serving: NaturalQuestions, HotpotQA, 2WikiMultiHopQA over a ~38M-passage Wikipedia corpus with IVF indexes (HedraRAG) — [arXiv 2507.09138](https://arxiv.org/html/2507.09138); NQ, HotpotQA, TriviaQA over wiki_dpr with six pipelines, traces and index released at https://huggingface.co/datasets/lauyeeyu/TeleRAG-Dataset (TeleRAG) — [arXiv 2502.20969](https://arxiv.org/html/2502.20969); Squad, Musique, KG RAG FinSec, QMSUM with Poisson arrivals (METIS) — [arXiv 2412.10543](https://arxiv.org/html/2412.10543); web_question, HotpotQA, FinQABench, TruthfulQA with Poisson arrivals (Ayo) — [arXiv 2407.00326](https://arxiv.org/html/2407.00326)
- RAG performance modelling: RAGSchema workload descriptions and a simulator rather than a trace (RAGO) — [arXiv 2503.14649](https://arxiv.org/html/2503.14649); [github.com/google/rago](https://github.com/google/rago)
- Reasoning workloads: MATH-500, AIME24, AMC23, GSM8K, ASDiv, LiveCodeBench, GAME24 (Dynasor) — [arXiv 2412.20993](https://arxiv.org/html/2412.20993); GSM8K, MATH-500, AIME 2024/2025, BRUMO, HMMT (SPEX) — [arXiv 2605.10195](https://arxiv.org/html/2605.10195); AlpacaEval 2.0 and Arena-Hard prompts with reasoning/answer lengths taken from o4-mini, plus MATH-500, GPQA, LiveCodeBench (PASCAL) — [arXiv 2602.11530](https://arxiv.org/html/2602.11530)
- Structured decoding: JSON-mode-eval (NousResearch) plus synthetic XML and Python-DSL grammars (XGrammar) — [arXiv 2411.15100](https://arxiv.org/html/2411.15100v3)
- A 2026 preprint titled "Benchmarking LLM Serving Systems for Agentic AI Workloads with XPerf" exists (arXiv 2608.20370); I saw only its title and reference list — [arXiv 2608.20370](https://arxiv.org/pdf/2608.20370)

### Inferences
- HotpotQA and ShareGPT recur across agent-serving papers (Agentix, InferCept, LLMCompiler, HPCA '26, Ayo, HedraRAG, TeleRAG), and SWE-bench variants recur in the 2026 agent studies (IISWC '26, ThunderAgent, AgentCgroup); results on these are the easiest to line up against prior work.
- Arrival patterns are synthetic in almost every catalogued evaluation (Poisson in Agentix, Ayo, METIS, PASCAL; Azure chat/code traces repurposed in Murakkab), so none of the peer-reviewed systems here was evaluated on real agent arrival traces.
- Token-length distributions for reasoning workloads in PASCAL come from a proprietary model, whereas Dynasor, SPEX and IISWC '26 generate them with open models.

### Gaps
- I did not confirm that the Copilot trace is downloadable or what schema it has.
- I did not confirm public availability of the Azure LLM inference traces used by Murakkab, the LATS traces used by Agentix, or the Bing Copilot/GPTs prompts used by Parrot.
- XPerf and AgentServeSim (benchmark/simulator preprints) were not read, so what workloads they standardise is unknown.
- Dataset URLs are given only where a fetched source printed them; the well-known benchmarks are cited through the papers that use them.

## Which problems do several papers name as still open?

### Takeaway
The limitations that recur across papers are: dynamic (non-static) agent control flow, scheduling under real SLOs and unknown program lengths, multi-node and heterogeneous deployment, what to do with GPU and sandbox state while an agent waits on a tool or a user, the lack of joint quality-and-efficiency evaluation, and the absence of public agent traces.

### Cited Findings
- Dynamic workflows:
  - Parrot "only supports cloud-side orchestration of LLM requests without involving dynamic control flow" — [arXiv 2405.19888](https://arxiv.org/html/2405.19888)
  - Ayo's "ahead-of-time graph optimization would struggle with dynamic workflows" — [arXiv 2407.00326](https://arxiv.org/html/2407.00326)
  - LLMCompiler notes that for Game of 24 "solving this problem with a single static graph is not feasible" — [arXiv 2312.04511](https://arxiv.org/html/2312.04511)
  - Agentix/Autellix defers graph-level optimisations to future work — [arXiv 2502.13965](https://arxiv.org/html/2502.13965)
- Scheduling without knowing program length, and SLOs for agents:
  - "a noticable gap remains between Autellix and SRPT" — [arXiv 2502.13965](https://arxiv.org/html/2502.13965)
  - "A detailed exploration of efficient agent serving under SLA constraints is left as future work" — [arXiv 2506.04301](https://arxiv.org/html/2506.04301)
  - The Copilot study names "session-aware scheduling" and "archetype-aware SLOs" as design directions — [arXiv 2608.00101](https://arxiv.org/html/2608.00101)
  - Ayo says critical-path prioritisation "requires accurate online predictions of critical paths" — [arXiv 2407.00326](https://arxiv.org/html/2407.00326)
  - InferCept leaves comparison of alternative scheduling policies to future work — [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2)
- Multi-node, cluster-scale and heterogeneous hardware:
  - Pie: "Real deployments must span multiple GPU nodes, raising challenges in load balancing" — [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1)
  - PASCAL: heterogeneous environments are "beyond the scope of this work" — [arXiv 2602.11530](https://arxiv.org/html/2602.11530)
  - Murakkab: future work is "extending adaptive co-optimization across larger clusters" — [arXiv 2508.18298](https://arxiv.org/html/2508.18298)
  - Dynasor "did not explore its integration with advanced serving techniques like PD disaggregation or chunked prefill" — [arXiv 2412.20993](https://arxiv.org/html/2412.20993)
- State held while the agent is not generating (tool calls, user idle, compaction):
  - InferCept measures recomputation after tool calls at "37-40% of total model forwarding time" — [arXiv 2402.01869](https://arxiv.org/html/2402.01869v2)
  - ThunderAgent accepts that "the GPU memory occupied by acting programs is idle" — [arXiv 2602.13692](https://arxiv.org/html/2602.13692)
  - HPCA '26 measures GPU idle time "up to 54.5%" of execution time — [arXiv 2506.04301](https://arxiv.org/html/2506.04301)
  - Copilot: cache hit rate falls from about 90% within a turn to about 55% across turns, and compaction and model switches invalidate caches — [arXiv 2608.00101](https://arxiv.org/html/2608.00101)
  - IISWC '26: under thrashing, hit rate fell to 16-26% and throughput dropped by up to 66% — [arXiv 2605.26297](https://arxiv.org/html/2605.26297)
- Tool and sandbox side:
  - "challenges such as initialization overhead, large container images, and retry-induced accumulation remain open" — [arXiv 2602.09345](https://arxiv.org/html/2602.09345)
  - Speculative tool execution, context compaction and tree search are uncharacterised: "we leave their characterization to future work" — [arXiv 2605.26297](https://arxiv.org/html/2605.26297)
  - Tools account for 30.2% of agent latency on average (HPCA '26) and up to 28.7% of runtime on GAIA (IISWC '26) — [arXiv 2506.04301](https://arxiv.org/html/2506.04301); [arXiv 2605.26297](https://arxiv.org/html/2605.26297)
- Quality and efficiency evaluated together:
  - Copilot: "Without prompt/response content, we cannot correlate infrastructure efficiency with task completion quality" — [arXiv 2608.00101](https://arxiv.org/html/2608.00101)
  - RAGO: RAGSchema "is not an abstraction for quality" — [arXiv 2503.14649](https://arxiv.org/html/2503.14649)
  - METIS: "more work will be needed to complement it for better interpretability and robustness" — [arXiv 2412.10543](https://arxiv.org/html/2412.10543)
  - Dynasor: the latency-accuracy impact "remains an open area for future research" — [arXiv 2412.20993](https://arxiv.org/html/2412.20993)
- Reasoning workloads:
  - PASCAL's "benefits may diminish because answering phases are too short" on reasoning-heavy inputs, and phase-level resource partitioning "is left for future work" — [arXiv 2602.11530](https://arxiv.org/html/2602.11530)
  - SPEX "struggles to achieve significant acceleration" on skewed trees and loses relative benefit at large batch sizes — [arXiv 2605.10195](https://arxiv.org/html/2605.10195)
- RAG retrieval/generation balance:
  - TeleRAG "is designed for IVF-style retrieval indices" and HNSW "does not naturally expose explicit prefetchable units" — [arXiv 2502.20969](https://arxiv.org/html/2502.20969)
  - HedraRAG: "extremely large models paired with small databases tend to shift the performance bottleneck toward LLM inference" — [arXiv 2507.09138](https://arxiv.org/html/2507.09138)
  - RAGO leaves "simultaneous execution" strategies as future work — [arXiv 2503.14649](https://arxiv.org/html/2503.14649)
- Missing public traces: Murakkab approximated arrivals with Azure LLM traces because agent-workflow traces were not public — [arXiv 2508.18298](https://arxiv.org/html/2508.18298); the Copilot authors "plan to release sanitized traces" — [arXiv 2608.00101](https://arxiv.org/html/2608.00101)
- Security and multi-tenancy of programmable or signal-exposing serving: Pie says "A comprehensive security treatment is beyond this paper's scope" — [arXiv 2510.24051](https://arxiv.org/html/2510.24051v1); Dynasor names "security vulnerabilities through side-channel attack for certaindex in multi-tenant settings" — [arXiv 2412.20993](https://arxiv.org/html/2412.20993)

### Inferences
- The "state while waiting" theme is named by five independent sources (InferCept, ThunderAgent, HPCA '26, IISWC '26, Copilot) and overlaps with the KV-cache scope assigned to another researcher; the report writer should cross-reference rather than count it twice.
- Dynamic control flow is named as a limitation by the 2024-2025 graph-based systems (Parrot, Ayo, LLMCompiler) and is the stated motivation of the 2025-2026 program-level systems (Agentix, ThunderAgent), so the two groups describe the same gap from opposite sides.
- Several SOSP/OSDI papers here (HedraRAG, SPEX, Murakkab, XGrammar) have no limitations section in the text I read, so their open problems are under-represented relative to papers that state them explicitly.

### Gaps
- Limitations text in the unread tails (RAGO conclusion, appendices of Dynasor, PASCAL, ThunderAgent, D1) was not seen.
- Camera-ready versions of the 2026 USENIX papers (Agentix, Murakkab, SPEX) were not read; their discussion sections may differ from the arXiv versions quoted here.
- I found no survey or workshop report that itself enumerates open problems for agent serving, so the grouping above is assembled from individual papers' statements.
