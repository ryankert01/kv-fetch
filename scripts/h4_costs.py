"""H4 (EXPERIMENTS.md, "Hybrid models"): measured inputs for the tiered-KV cost model of a hybrid
model. Run with .venv-hybrid on one free GPU.

  CUDA_VISIBLE_DEVICES=4 .venv-hybrid/bin/python scripts/h4_costs.py configs/hybrid.yaml [--model qwen35_4b]

Measures, with CUDA graphs (as the engine decodes):
  layer times   one decoder layer per type at batch B and context n: a linear-attention (GDN) layer, a
                full-attention layer with dense attention and with page-sparse attention at 512 tokens
                (Quest scoring included); B in h4.batches, n in h4.contexts (cache contents are zeros:
                the cost does not depend on them),
  host -> GPU   copy bandwidth from pinned host memory: one contiguous copy per size, and the
                gather of scattered pages into a pinned staging buffer followed by one copy.
The cost model (in merge with H1's page churn) asks whether the pages a selector newly needs each
step fit in the window of the GDN layers that run between two attention layers.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kvfetch import sparse_decode as SD
from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import load_model
from kvfetch.provenance import provenance, write_json


def graph_time(fn, iters: int) -> float:
    """ms per call of fn replayed as a CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


@torch.inference_mode()
def layer_times(model, cfg, h4, device) -> list[dict]:
    defs = cfg["definitions"]
    c = model.config
    dec = model.model.layers
    types = c.layer_types
    lin = next(i for i, t in enumerate(types) if t == "linear_attention")
    att = next(i for i, t in enumerate(types) if t == "full_attention")
    out = []
    for B in h4["batches"]:
        for n in h4["contexts"]:
            if B * n * h4["kv_bytes_per_token"] > h4["max_kv_gib"] * 2 ** 30:
                continue
            for method in (SD.Method("full"), SD.Method("quest", 512)):
                eng = SD.SparseEngine(model, method, batch=B, max_len=n + 16, page_size=defs["page_size"],
                                      sink_pages=defs["pinned"]["sink_pages"], recent_tokens=defs["pinned"]["recent_tokens"],
                                      obs_window=cfg["trace"]["prefill_obs_window"], device=device)
                SD._ENGINE = eng
                eng.lens[:] = n - 1
                hs = torch.randn((B, 1, c.hidden_size), dtype=torch.bfloat16, device=device) * 0.1
                pos = eng.lens.clone()[:, None]
                pe = model.model.rotary_emb(hs, pos[None].expand(3, -1, -1))
                lc = eng.Lmax

                def step(layer):
                    eng.begin_step(lc)
                    o = dec[layer](hs, position_embeddings=pe, attention_mask=None, position_ids=pos,
                                   past_key_values=eng, use_cache=True)
                    eng.mode = None
                    eng.lens.fill_(n - 1)                                      # keep the context fixed
                    return o
                row = {"batch": B, "context": n, "method": method.label}
                row["attention_layer_ms"] = graph_time(lambda: step(att), h4["iters"])
                if method.name == "full":
                    row["gdn_layer_ms"] = graph_time(lambda: step(lin), h4["iters"])
                print(row, flush=True)
                out.append(row)
                SD._ENGINE = None
                del eng
                torch.cuda.empty_cache()
    return out


def copy_bandwidth(h4, device) -> dict:
    page = h4["page_bytes"]
    host = torch.empty(h4["host_pool_mib"] * 2 ** 20, dtype=torch.uint8).pin_memory()
    staging = torch.empty(max(h4["copy_mib"]) * 2 ** 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(max(h4["copy_mib"]) * 2 ** 20, dtype=torch.uint8, device=device)
    n_pages = host.numel() // page
    pages = host[:n_pages * page].view(n_pages, page)
    out = {"contiguous_gbps": {}, "gather_gbps": {}, "gather_cpu_ms": {}}
    for mib in h4["copy_mib"]:
        nb = mib * 2 ** 20
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            dst[:nb].copy_(host[:nb], non_blocking=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(h4["iters"]):
            dst[:nb].copy_(host[:nb], non_blocking=True)
        b.record()
        torch.cuda.synchronize()
        out["contiguous_gbps"][str(mib)] = nb * h4["iters"] / (a.elapsed_time(b) / 1e3) / 1e9
        k = nb // page                                                       # scattered pages -> staging -> GPU
        idx = torch.randperm(n_pages)[:k]
        sv = staging[:k * page].view(k, page)
        t0 = time.perf_counter()
        for _ in range(h4["iters"]):
            torch.index_select(pages, 0, idx, out=sv)
        cpu = (time.perf_counter() - t0) / h4["iters"]
        a.record()
        for _ in range(h4["iters"]):
            dst[:k * page].copy_(staging[:k * page], non_blocking=True)
        b.record()
        torch.cuda.synchronize()
        gpu = a.elapsed_time(b) / 1e3 / h4["iters"]
        out["gather_cpu_ms"][str(mib)] = 1e3 * cpu
        out["gather_gbps"][str(mib)] = k * page / (cpu + gpu) / 1e9
        print(mib, "MiB:", {k_: v[str(mib)] for k_, v in out.items()}, flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--model", default="qwen35_4b")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = config_from_args(args)
    h4, mc = cfg["hybrid"]["h4"], cfg["hybrid"]["models"][args.model]
    torch.set_num_threads(h4["cpu_threads"])
    SD.register_sparse_attention()
    model, _ = load_model(mc["name"], cfg, args.device, attn_implementation=SD.SPARSE_ATTN)
    res = {"copy": copy_bandwidth(h4, args.device), "layers": layer_times(model, cfg, h4, args.device)}
    write_json(Path(cfg["paths"]["results_dir"]) / f"h4_costs_{args.model}.json",
               {"provenance": provenance(cfg, model=mc["name"], context_length=None, h4=h4), **res})


if __name__ == "__main__":
    main()
