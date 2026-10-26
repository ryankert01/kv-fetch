"""H4 (EXPERIMENTS.md, "Hybrid models"): measured inputs for the tiered-KV cost model of a hybrid
model. Run with .venv-hybrid on one otherwise idle GPU.

  CUDA_VISIBLE_DEVICES=4 .venv-hybrid/bin/python scripts/h4_costs.py configs/hybrid.yaml [--model qwen35_4b]

Measures (cache contents are zeros: the costs do not depend on them), under CUDA graphs as the engine
decodes unless noted:
  layers     one decoder layer per type at batch B and context n (h4.batches x h4.contexts): a GDN layer;
             an attention layer with dense attention, H3's "ours" (hybrid.h2.engine_specs[chosen], the
             engine path H3 runs) and Quest, at h4.budget_tokens. The engine is built for a model cut
             after its first attention layer (3 GDN layers + 1 attention layer: all attention layers
             have the same shape), so batches beyond all-on-GPU capacity can be timed. The per-step page
             plan (begin_step, once per step for all layers) is timed apart.
  head       final norm, LM head and sampling with the presence penalty, at batch B,
  step       whole decode steps (all layers, the engine's step function) where all KV fits on the GPU
             (h4.max_kv_gib): calibrates the sum of the parts,
  lookahead  the H-c prefetch query: input norm, q path and RoPE of a hidden state, "ours" page scores
             over the layer's summaries, the top k x budget pages and their copy to pinned host memory
             (GPU part),
  copy       host -> GPU: contiguous copies; the staged fetch round trip of k scattered pages (wall clock:
             indices GPU -> pinned host, sync, CPU gather into pinned staging, one copy, sync); a zero-copy
             gather kernel (Triton) reading the pages from pinned host memory over PCIe (UVA), timed in
             a graph, with one program per page block or with h4.zero_copy_programs programs looping (fewer
             SMs taken from compute); a GDN layer's time while a contiguous copy or either gather kernel runs
             on another stream.
"""
import argparse
import copy as copy_
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import triton
import triton.language as tl

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
    return replay_time(g, iters)


def replay_time(g, iters: int) -> float:
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


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def cut_model(model, att: int):
    """The model's first att + 1 layers (GDN layers, then the attention layer att) as the engine sees it."""
    c = copy_.copy(model.config)
    c.layer_types = list(model.config.layer_types[:att + 1])
    c.num_hidden_layers = att + 1
    return types.SimpleNamespace(config=c, parameters=model.parameters)


def engine(model, method, B, max_len, cfg, device):
    defs = cfg["definitions"]
    return SD.SparseEngine(model, method, batch=B, max_len=max_len, page_size=defs["page_size"],
                           sink_pages=defs["pinned"]["sink_pages"], recent_tokens=defs["pinned"]["recent_tokens"],
                           obs_window=cfg["trace"]["prefill_obs_window"], device=device)


def methods(cfg) -> dict:
    h4, h2 = cfg["hybrid"]["h4"], cfg["hybrid"]["h2"]
    es = dict(h2["engine_specs"][h2["chosen"]])
    name = es.pop("method")
    spec = {"fuse": "sum", "history_weight": 0.0, "history": "last", **cfg["hybrid"]["h1"]["ours"], **es} if name == "ours" else es
    b = h4["budget_tokens"]
    return {"full": SD.Method("full"), f"ours@{b}": SD.Method(name, b, spec=spec), f"quest@{b}": SD.Method("quest", b)}


def decode_inputs(model, eng, B, n, device):
    eng.lens[:] = n - 1
    hs = torch.randn((B, 1, model.config.hidden_size), dtype=torch.bfloat16, device=device) * 0.1
    pos = eng.lens.clone()[:, None]
    pe = model.model.rotary_emb(hs, pos[None].expand(3, -1, -1))
    return hs, pos, pe


@torch.inference_mode()
def layer_times(model, cfg, h4, device) -> tuple[list[dict], dict]:
    types_ = model.config.layer_types
    att = types_.index("full_attention")
    lin = types_.index("linear_attention")
    cm = cut_model(model, att)
    dec = model.model.layers
    rows, plan = [], {}
    for B in h4["batches"]:
        for n in h4["contexts"]:
            for label, method in methods(cfg).items():
                row = {"batch": B, "context": n, "method": label}
                try:
                    eng = engine(cm, method, B, n + cfg["definitions"]["page_size"], cfg, device)
                    SD._ENGINE = eng
                    hs, pos, pe = decode_inputs(model, eng, B, n, device)
                    eng.begin_step(eng.Lmax)                                   # once: lens stay fixed
                    if label != "full":
                        plan.setdefault(str(B), {})[str(n)] = graph_time(lambda: eng.begin_step(eng.Lmax), h4["iters"])

                    def step(layer):
                        eng.mode = "decode"
                        dec[layer](hs, position_embeddings=pe, attention_mask=None, position_ids=pos,
                                   past_key_values=eng, use_cache=True)
                        eng.lens.fill_(n - 1)
                    row["attention_layer_ms"] = graph_time(lambda: step(att), h4["iters"])
                    if label == "full":
                        row["gdn_layer_ms"] = graph_time(lambda: step(lin), h4["iters"])
                except torch.OutOfMemoryError:
                    row["oom"] = True
                print(row, flush=True)
                rows.append(row)
                SD._ENGINE = None
                eng = None
                torch.cuda.empty_cache()
    return rows, plan


@torch.inference_mode()
def head_times(model, cfg, mc, h4, device) -> dict:
    out = {}
    V = model.config.vocab_size
    samp = mc["sampling"]
    for B in h4["batches"]:
        hs = torch.randn((B, 1, model.config.hidden_size), dtype=torch.bfloat16, device=device)
        u = torch.rand(B, device=device)
        ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        seen = torch.zeros((B, V), dtype=torch.bool, device=device)

        def fn():
            seen.scatter_(1, ids, True)
            logits = model.lm_head(model.model.norm(hs))[:, -1]
            return SD.sample_token(logits, u, **samp, seen=seen)
        out[str(B)] = graph_time(fn, h4["iters"])
        print("head", B, out[str(B)], flush=True)
    return out


@torch.inference_mode()
def step_times(model, cfg, mc, h4, device) -> list[dict]:
    """Whole decode steps as sparse_decode._run's step_fn (inactive slots: lengths stay fixed)."""
    rows = []
    V = model.config.vocab_size
    mask = {"full_attention": None, "linear_attention": None}
    for B in h4["batches"]:
        for n in h4["contexts"]:
            if B * n * h4["kv_bytes_per_token"] > h4["max_kv_gib"] * 2 ** 30:
                continue
            for label, method in methods(cfg).items():
                if label.startswith("quest"):
                    continue
                row = {"batch": B, "context": n, "method": label}
                try:
                    eng = engine(model, method, B, n + cfg["definitions"]["page_size"], cfg, device)
                    SD._ENGINE = eng
                    eng.lens[:] = n - 1
                    ids = torch.zeros((B, 1), dtype=torch.long, device=device)
                    u = torch.rand(B, device=device)
                    act = torch.zeros(B, dtype=torch.bool, device=device)
                    seen = torch.zeros((B, V), dtype=torch.bool, device=device)

                    def fn():
                        seen.scatter_(1, ids, True)
                        pos = eng.lens.clone()
                        eng.begin_step(eng.Lmax)
                        o = model(input_ids=ids, position_ids=pos[:, None], cache_position=pos, past_key_values=eng,
                                  attention_mask=mask, use_cache=True)
                        eng.end_step(act)
                        return SD.sample_token(o.logits[:, -1], u, **mc["sampling"], seen=seen)
                    row["step_ms"] = graph_time(fn, h4["iters"])
                except torch.OutOfMemoryError:
                    row["oom"] = True
                print(row, flush=True)
                rows.append(row)
                SD._ENGINE = None
                eng = None
                torch.cuda.empty_cache()
    return rows


def q_only(attn, x, cos, sin):
    """The attention layer's query path (projection, gate split, per-head norm, RoPE) for x [B, 1, D]."""
    shape, hd = x.shape[:-1], attn.head_dim
    qp = attn.q_proj(x)
    if qp.shape[-1] == 2 * attn.config.num_attention_heads * hd:
        q, _ = torch.chunk(qp.view(*shape, -1, hd * 2), 2, dim=-1)
    else:
        q = qp.view(*shape, -1, hd)
    q = attn.q_norm(q.reshape(*shape, -1, hd)).transpose(1, 2)
    return sys.modules[type(attn).__module__].apply_rotary_pos_emb(q, q, cos, sin)[0]


@torch.inference_mode()
def lookahead_times(model, cfg, h4, device) -> list[dict]:
    att = model.config.layer_types.index("full_attention")
    cm = cut_model(model, att)
    attn = model.model.layers[att].self_attn
    norm = model.model.layers[att].input_layernorm
    label, method = next((k, v) for k, v in methods(cfg).items() if k.startswith("ours"))
    ps = cfg["definitions"]["page_size"]
    rows = []
    for B in h4["batches"]:
        for n in h4["contexts"]:
            for mult in h4["lookahead_multiples"]:
                row = {"batch": B, "context": n, "multiple": mult}
                try:
                    eng = engine(cm, method, B, n + ps, cfg, device)
                    SD._ENGINE = eng
                    hs, pos, pe = decode_inputs(model, eng, B, n, device)
                    eng.begin_step(eng.Lmax)
                    k = int(mult * (eng.kmax_pages - eng.np_pinned + 1))       # the largest per-step budget
                    host_idx = torch.empty((B, k), dtype=torch.long).pin_memory()
                    li = eng.aslot[att]
                    elig = eng.plan["elig"]

                    def fn():
                        q = q_only(attn, norm(hs), *pe)                                 # [B, H, 1, d]
                        qg = q[:, :, 0].reshape(B, eng.S, eng.g, eng.d)
                        s = eng._mean_logit(li, qg, attn.scaling).masked_fill(~elig[:, None, :], float("-inf"))
                        idx = torch.softmax(s, -1).nan_to_num(0.0).sum(1).topk(k, -1).indices
                        host_idx.copy_(idx, non_blocking=True)
                    row["ms"] = graph_time(fn, h4["iters"])
                except torch.OutOfMemoryError:
                    row["oom"] = True
                print("lookahead", row, flush=True)
                rows.append(row)
                SD._ENGINE = None
                eng = None
                torch.cuda.empty_cache()
    return rows


@triton.jit
def _gather_pages(src, idx, dst, page_elems, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    p = tl.load(idx + i).to(tl.int64)
    offs = j * BLOCK + tl.arange(0, BLOCK)
    m = offs < page_elems
    tl.store(dst + i * page_elems + offs, tl.load(src + p * page_elems + offs, mask=m), mask=m)


@triton.jit
def _gather_pages_few(src, idx, dst, k, page_elems, NPROG: tl.constexpr, BLOCK: tl.constexpr):
    """As _gather_pages with a fixed number of programs, each looping over pages and blocks."""
    pid = tl.program_id(0)
    nblk = tl.cdiv(page_elems, BLOCK)
    for w in range(pid, k * nblk, NPROG):
        i = w // nblk
        j = w % nblk
        p = tl.load(idx + i).to(tl.int64)
        offs = j * BLOCK + tl.arange(0, BLOCK)
        m = offs < page_elems
        tl.store(dst + i * page_elems + offs, tl.load(src + p * page_elems + offs, mask=m), mask=m)


def zero_copy_gather(src_host: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, block: int, programs: int | None = None) -> None:
    """dst[i] = src_host[idx[i]] (rows of int64), read by the GPU from pinned host memory (UVA): one
    program per (page, block), or ``programs`` programs looping (fewer SMs taken from compute)."""
    k, pe = idx.shape[0], src_host.shape[1]
    if programs is None:
        _gather_pages[(k, triton.cdiv(pe, block))](src_host, idx, dst, pe, BLOCK=block)
    else:
        _gather_pages_few[(programs,)](src_host, idx, dst, k, pe, NPROG=programs, BLOCK=block)


def fill_pattern(host: torch.Tensor, chunk: int = 2 ** 24) -> None:
    """Distinct int64 values everywhere, so a gather that moves nothing cannot pass the check."""
    h = host.view(torch.int64)
    for a in range(0, h.numel(), chunk):
        n = min(chunk, h.numel() - a)
        h[a:a + n] = torch.arange(a, a + n, dtype=torch.int64)


@torch.inference_mode()
def copy_costs(model, cfg, h4, device) -> dict:
    """Every timed fetch reads a fresh random page set from the h4.host_pool_mib pool (far above the CPU
    last-level cache and the GPU L2), so neither cache serves repeated reads; sizes whose rate exceeds
    the contiguous copy rate are flagged."""
    page = h4["page_bytes"]
    host = torch.empty(h4["host_pool_mib"] * 2 ** 20, dtype=torch.uint8).pin_memory()
    fill_pattern(host)
    n_pages = host.numel() // page
    pages = host[:n_pages * page].view(n_pages, page)
    kmax = max(h4["fetch_pages"])
    staging = torch.empty((kmax, page), dtype=torch.uint8).pin_memory()
    rows_ = max(kmax, -(-max(h4["copy_mib"]) * 2 ** 20 // page))
    dst = torch.empty((rows_, page), dtype=torch.uint8, device=device)
    out = {"contiguous_gbps": {}, "staged_round_trip_ms": {}, "zero_copy_ms": {}, "zero_copy_few_ms": {},
           "zero_copy_check": None, "zero_copy_few_check": None, "zero_copy_programs": h4["zero_copy_programs"],
           "above_contiguous_rate": []}
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flat = dst.view(-1)
    for mib in h4["copy_mib"]:
        nb = mib * 2 ** 20
        for _ in range(3):
            flat[:nb].copy_(host[:nb], non_blocking=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(h4["iters"]):
            flat[:nb].copy_(host[:nb], non_blocking=True)
        b.record()
        torch.cuda.synchronize()
        out["contiguous_gbps"][str(mib)] = nb * h4["iters"] / (a.elapsed_time(b) / 1e3) / 1e9
    peak_gbps = max(out["contiguous_gbps"].values())
    out["graph_launch_us"] = graph_launch_us(device, h4["iters"])
    g = torch.Generator().manual_seed(0)
    src64, dst64 = pages.view(torch.int64), dst.view(torch.int64)
    for k in h4["fetch_pages"]:
        R = max(h4["iters"] + 8, -(-h4["cache_flush_mib"] * 2 ** 20 // (k * page)))   # fresh pages per call; a set comes back
        # only after h4.cache_flush_mib of other pages were read (CPU LLC, GPU L2)
        bank = torch.stack([torch.randperm(n_pages, generator=g)[:k] for _ in range(R)]).to(device)   # [R, k]
        idx_host = torch.empty(k, dtype=torch.long).pin_memory()
        sv = staging[:k]
        ctr = [0]

        def staged():
            idx_host.copy_(bank[ctr[0] % R], non_blocking=True)
            ctr[0] += 1
            torch.cuda.current_stream().synchronize()
            torch.index_select(pages, 0, idx_host, out=sv)
            dst[:k].copy_(sv, non_blocking=True)
            torch.cuda.current_stream().synchronize()
        for _ in range(3):
            staged()
        t0 = time.perf_counter()
        for _ in range(h4["iters"]):
            staged()
        out["staged_round_trip_ms"][str(k)] = 1e3 * (time.perf_counter() - t0) / h4["iters"]
        dctr = torch.zeros(1, dtype=torch.long, device=device)
        idx = torch.empty(k, dtype=torch.long, device=device)

        def rotate():                                                # in-graph: next page set
            idx.copy_(bank.index_select(0, dctr)[0])
            dctr.add_(1).remainder_(R)
        base = graph_time(rotate, h4["iters"])
        for key, progs in (("zero_copy", None), ("zero_copy_few", h4["zero_copy_programs"])):
            try:
                dst.zero_()
                ms = graph_time(lambda: (rotate(), zero_copy_gather(src64, idx, dst64[:k], h4["gather_block"], progs)), h4["iters"]) - base
                out[f"{key}_ms"][str(k)] = ms
                torch.cuda.synchronize()
                ok = bool(torch.equal(dst[:k].cpu(), pages[idx.cpu()]))
                out[f"{key}_check"] = ok if out[f"{key}_check"] is None else (out[f"{key}_check"] and ok)
            except Exception as e:                   # e.g. the launcher refuses a host pointer
                out[f"{key}_ms"][str(k)] = None
                out[f"{key}_error"] = repr(e)[:300]
        for key in ("staged_round_trip_ms", "zero_copy_ms", "zero_copy_few_ms"):
            v = out[key][str(k)]
            if v and k * page / (v / 1e3) / 1e9 > peak_gbps:
                out["above_contiguous_rate"].append([key, k])
        print("copy", k, "pages:", {key: out[key][str(k)] for key in ("staged_round_trip_ms", "zero_copy_ms", "zero_copy_few_ms")}, flush=True)
    out["overlap"] = overlap(model, cfg, h4, device, host, pages, dst)
    return out


def graph_launch_us(device, iters: int) -> float:
    """Wall-clock cost of launching one small CUDA graph and waiting for it: what splitting a step's
    graph at each attention layer (the staged fetch's host sync) adds per split."""
    x = torch.zeros(1, device=device)
    g = capture(lambda: x.add_(1))
    for _ in range(5):
        g.replay()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
        torch.cuda.synchronize()
    return 1e6 * (time.perf_counter() - t0) / iters


@torch.inference_mode()
def overlap(model, cfg, h4, device, host, pages, dst) -> list[dict]:
    """A GDN layer and a copy / gather on two streams, each slowed by the other: phase "gdn" times the
    GDN replays while the copies outlast them, phase "copy" times the copies while GDN replays outlast
    them."""
    types_ = model.config.layer_types
    att, lin = types_.index("full_attention"), types_.index("linear_attention")
    cm = cut_model(model, att)
    dec = model.model.layers
    side = torch.cuda.Stream()
    rows = []
    k = max(h4["fetch_pages"])
    idx_gpu = torch.randperm(pages.shape[0])[:k].to(device)
    src64, dst64 = pages.view(torch.int64), dst.view(torch.int64)
    nb = k * h4["page_bytes"]
    flat = dst.view(-1)
    loads = {"contiguous copy": lambda: flat[:nb].copy_(host[:nb], non_blocking=True),
             "zero-copy gather": lambda: zero_copy_gather(src64, idx_gpu, dst64[:k], h4["gather_block"]),
             "zero-copy gather (few programs)": lambda: zero_copy_gather(src64, idx_gpu, dst64[:k], h4["gather_block"],
                                                                         h4["zero_copy_programs"])}

    def timed(g, n, stream):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            a.record(stream)
            for _ in range(n):
                g.replay()
            b.record(stream)
        return a, b
    for B in h4["overlap_batches"]:
        eng = engine(cm, SD.Method("full"), B, cfg["definitions"]["page_size"] * 2, cfg, device)
        SD._ENGINE = eng
        hs, pos, pe = decode_inputs(model, eng, B, 1, device)
        eng.begin_step(eng.Lmax)

        def step():
            eng.mode = "decode"
            dec[lin](hs, position_embeddings=pe, attention_mask=None, position_ids=pos, past_key_values=eng, use_cache=True)
        gg = capture(step)
        alone = replay_time(gg, h4["iters"])
        row = {"batch": B, "gdn_alone_ms": alone}
        main = torch.cuda.current_stream()
        for name, fn in loads.items():
            try:
                lg = capture(fn)
                one = replay_time(lg, 5)
                for phase, (n_g, n_l, measure) in {"gdn": (h4["iters"], max(1, int(2 * alone * h4["iters"] / one)), "g"),
                                                   "copy": (max(1, int(2 * one * h4["iters"] / alone)), h4["iters"], "l")}.items():
                    torch.cuda.synchronize()
                    side.wait_stream(main)
                    el = timed(lg, n_l, side)
                    eg = timed(gg, n_g, main)
                    torch.cuda.synchronize()
                    if measure == "g":
                        row[f"gdn_with_{name}_ms"] = eg[0].elapsed_time(eg[1]) / n_g
                    else:
                        row[f"{name}_with_gdn_ms"] = el[0].elapsed_time(el[1]) / n_l
                row[f"{name}_alone_ms"] = one
            except Exception as e:
                row[f"gdn_with_{name}_ms"] = None
                row[f"{name}_error"] = repr(e)[:300]
        print("overlap", row, flush=True)
        rows.append(row)
        SD._ENGINE = None
        eng = None
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--model", default="qwen35_4b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only", nargs="*", default=None, help="subset of: copy layers head step lookahead (dry runs)")
    args = p.parse_args()
    cfg = config_from_args(args)
    h4, mc = cfg["hybrid"]["h4"], cfg["hybrid"]["models"][args.model]
    torch.set_num_threads(h4["cpu_threads"])
    SD.register_sparse_attention()
    model, _ = load_model(mc["name"], cfg, args.device, attn_implementation=SD.SPARSE_ATTN)
    parts = args.only or ["copy", "layers", "head", "step", "lookahead"]
    res = {}
    if "copy" in parts:
        res["copy"] = copy_costs(model, cfg, h4, args.device)
    if "layers" in parts:
        res["layers"], res["plan_ms"] = layer_times(model, cfg, h4, args.device)
    if "head" in parts:
        res["head_ms"] = head_times(model, cfg, mc, h4, args.device)
    if "step" in parts:
        res["steps"] = step_times(model, cfg, mc, h4, args.device)
    if "lookahead" in parts:
        res["lookahead"] = lookahead_times(model, cfg, h4, args.device)
    if args.only:
        print("dry run (--only): not written")
        return
    write_json(Path(cfg["paths"]["results_dir"]) / f"h4_costs_{args.model}.json",
               {"provenance": provenance(cfg, model=mc["name"], context_length=None, h4=h4,
                                         gpu_name=torch.cuda.get_device_name(), h2_chosen=cfg["hybrid"]["h2"]["chosen"]), **res})


if __name__ == "__main__":
    main()
