# Checkpoint pilot (2026-10-27)

Run from the session scratchpad before the prefix-caching direction was set (EXPERIMENTS.md, "Prefix
caching for hybrid models"); moved here unchanged except for paths. Results: results/prefix/pilot/.

- ckpt_precision.py, ckpt_chain.py (config configs/prefix_pilot.yaml; HF transformers + fla, .venv-hybrid,
  GPUs 1-2): KL of compressed GDN checkpoints vs exact and vs the rechunk floor; chained rounding every
  528 tokens. summarize_precision.py / summarize_chain.py merge the shards.
- agent_loop.py (ReplaySSM vLLM fork; dense vs `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`), recompute_probe.py
  (prefill time vs recomputed tokens after a hit), pcie_bw.py (pinned H2D/D2H of one checkpoint). These
  three predate the config rule: their settings are constants at the top of each script and are stored in
  each result's provenance.
