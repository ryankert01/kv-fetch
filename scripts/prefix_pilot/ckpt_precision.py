"""Does a GDN prefix-cache checkpoint survive being stored at low precision?

For each stored Qwen3.5 reasoning trace (H3's full-attention run, token ids) and checkpoint position P
(prompt + offset generated tokens): prefill ids[:P] exactly (fp32 GDN states), then continue
teacher-forced over ids[P:P+W] from (a) that cache and (b) the same cache with every GDN recurrent state
replaced by a compressed copy (conv states and attention KV untouched). Per token: KL(exact || variant),
top-1 agreement and the NLL of the true next token, binned by distance from the checkpoint.

References: "rechunk" computes the same prefix as [0, P-R) then [P-R, P) -- the numerical noise a cache
hit already incurs, since a hit's prefix was split into chunks differently from a cold run -- and "zero"
drops the state (what losing the checkpoint without recomputing would do).

Usage: ckpt_precision.py CONFIG OUT.json [--limit N]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch
import torch.nn.functional as F

from kvfetch.config import load_config
from kvfetch.models import encode, format_prompt, load_model
from kvfetch.provenance import provenance
from run_e3 import problems, thinking


def compress(S: torch.Tensor, name: str) -> torch.Tensor:
    """S: [B, H, K, V] fp32 recurrent state; returns its decompressed copy (fp32)."""
    if name == "identity":                           # run-to-run noise: an untouched copy
        return S.clone()
    if name == "bf16":
        return S.bfloat16().float()
    if name == "fp16":
        return S.half().float()
    if name.startswith("fp8"):
        dt = torch.float8_e4m3fn if "e4m3" in name else torch.float8_e5m2
        scale = S.abs().amax(-1, keepdim=True).clamp_min(1e-30) / torch.finfo(dt).max
        return (S / scale).to(dt).float() * scale
    if name.startswith("int"):                       # int8_row, int4_g32, int4_g128: symmetric, per group of V
        bits = int(name[3])
        g = S.shape[-1] if name.endswith("row") else int(name.split("_g")[1])
        qmax = 2 ** (bits - 1) - 1
        x = S.reshape(*S.shape[:-1], -1, g)
        scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / qmax
        return ((x / scale).round().clamp(-qmax, qmax) * scale).reshape(S.shape)
    if name.startswith("lowrank_r"):
        r = int(name[len("lowrank_r"):])
        U, s, Vh = torch.linalg.svd(S, full_matrices=False)
        return (U[..., :r] * s[..., None, :r]) @ Vh[..., :r, :]
    if name == "zero":
        return torch.zeros_like(S)
    raise ValueError(name)


def bytes_per_layer(name: str, H: int, K: int, V: int, scale_bytes: int) -> int:
    n = H * K * V
    if name in ("fp32", "rechunk", "identity"):
        return 4 * n
    if name in ("bf16", "fp16"):
        return 2 * n
    if name.startswith("fp8") or name == "int8_row":
        return n + H * K * scale_bytes
    if name.startswith("int4_g"):
        return n // 2 + (n // int(name.split("_g")[1])) * scale_bytes
    if name.startswith("lowrank_r"):
        return H * 2 * K * int(name[len("lowrank_r"):]) * 2    # bf16 factors U*s and Vh
    return 0


def linear_layers(model) -> list[int]:
    cfg = getattr(model.config, "text_config", model.config)
    return [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]


@torch.no_grad()
def prefill(model, ids: list[int], dev, split: int | None = None):
    """Cache after ids; ``split`` computes it as two forward calls (ids[:split], ids[split:])."""
    cache = None
    parts = [ids] if split is None else [ids[:split], ids[split:]]
    for p in parts:
        out = model(input_ids=torch.tensor([p], device=dev), past_key_values=cache, use_cache=True,
                    logits_to_keep=1)
        cache = out.past_key_values
    return cache


@torch.no_grad()
def continuation_logprobs(model, cache, cont: list[int], dev) -> torch.Tensor:
    out = model(input_ids=torch.tensor([cont], device=dev), past_key_values=cache, use_cache=True)
    return out.logits[0].float().log_softmax(-1)              # [W, V]


def compare(lp_ref, lp, targets, chunk: int) -> dict:
    kl, agree, dnll = [], [], []
    for a in range(0, lp_ref.shape[0], chunk):
        r, v = lp_ref[a:a + chunk], lp[a:a + chunk]
        kl.append((r.exp() * (r - v)).sum(-1))
        agree.append((r.argmax(-1) == v.argmax(-1)).float())
        t = targets[a:a + chunk, None]
        dnll.append(r.gather(-1, t)[:, 0] - v.gather(-1, t)[:, 0])
    return {"kl": torch.cat(kl), "agree": torch.cat(agree), "dnll": torch.cat(dnll)}


def summarize(per_tok: dict, bins: list[int], curve_step: int) -> dict:
    out = {}
    for k, x in per_tok.items():
        x = x.double()
        out[k] = {f"{a}-{b}": x[a:b].mean().item() for a, b in zip(bins[:-1], bins[1:])}
        out[k]["all"] = x.mean().item()
        out[k + "_curve"] = x[: len(x) // curve_step * curve_step].view(-1, curve_step).mean(1).tolist()
    out["kl"]["max"] = per_tok["kl"].max().item()
    return out


def state_stats(cache, layers, energy: float, variants: list[str]) -> dict:
    norms, ranks, rel = [], [], {v: [] for v in variants}
    for i in layers:
        S = cache.layers[i].recurrent_states[0].float()       # [1, H, K, V]
        norms.append(S.flatten(2).norm(dim=-1).mean().item())
        sv = torch.linalg.svdvals(S)                          # [1, H, min(K, V)]
        cum = (sv ** 2).cumsum(-1) / (sv ** 2).sum(-1, keepdim=True)
        ranks.append(((cum < energy).sum(-1) + 1).float().mean().item())
        for v in variants:
            rel[v].append(((compress(S, v) - S).norm() / S.norm()).item())
    return {"state_norm_mean": norms, "effective_rank_mean": ranks,
            "rel_error": {v: sum(e) / len(e) for v, e in rel.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1", help="i/n: this worker takes sequences i, i+n, ...")
    args = ap.parse_args()
    cfg = thinking(load_config(args.config))
    c = cfg["ckpt"]
    mc = cfg["hybrid"]["models"][c["model"]]
    dev = torch.device(args.device)
    model, tok = load_model(mc["name"], cfg, dev)
    layers = linear_layers(model)
    variants = c["variants"]
    seqs = []
    for s in problems({**cfg, "e3": cfg["hybrid"]["h3"]}, tok):
        f = REPO / c["runs"] / s.task / f"{s.index:04d}_s{c['sample']}.json"
        if f.exists():
            r = json.loads(f.read_text())
            prompt = encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"]
            assert len(prompt) == r["prompt_tokens"], f
            seqs.append({"task": s.task, "index": s.index, "ids": prompt + r["token_ids"], "pl": len(prompt)})
    i, n = map(int, args.shard.split("/"))
    seqs = seqs[: args.limit][i::n]
    W, R = c["window"], c["rechunk_tokens"]
    rows, t0 = [], time.time()
    shape = None
    for s in seqs:
        for off in c["offsets"]:
            P = s["pl"] + off
            if len(s["ids"]) < P + W + 1:
                continue
            cont = s["ids"][P:P + W]
            targets = torch.tensor(s["ids"][P + 1:P + W + 1], device=dev)
            base = prefill(model, s["ids"][:P], dev)
            S0 = base.layers[layers[0]].recurrent_states[0]
            shape = list(S0.shape)
            assert S0.dtype == torch.float32, S0.dtype
            stats = state_stats(base, layers, c["rank_energy"], [v for v in variants if v not in ("zero", "identity")])
            lp_ref = continuation_logprobs(model, copy.deepcopy(base), cont, dev)
            row = {"task": s["task"], "index": s["index"], "P": P, "offset": off, "stats": stats, "variants": {}}
            rc = prefill(model, s["ids"][:P], dev, split=P - R)
            row["variants"]["rechunk"] = summarize(
                compare(lp_ref, continuation_logprobs(model, rc, cont, dev), targets, c["logit_chunk"]),
                c["bins"], c["curve_step"])
            del rc
            for v in variants:
                q = copy.deepcopy(base)
                for i in layers:
                    st = q.layers[i].recurrent_states[0]
                    st.copy_(compress(st, v))
                row["variants"][v] = summarize(
                    compare(lp_ref, continuation_logprobs(model, q, cont, dev), targets, c["logit_chunk"]),
                    c["bins"], c["curve_step"])
                del q
            rows.append(row)
            del base, lp_ref
            torch.cuda.empty_cache()
            kl = {v: f"{row['variants'][v]['kl']['all']:.2e}" for v in ["rechunk"] + variants}
            print(f"[{time.time() - t0:6.0f}s] {s['task']} {s['index']} P={P}: {kl}", flush=True)
    H, K, V = shape[1:]
    sizes = {v: bytes_per_layer(v, H, K, V, c["scale_bytes"]) for v in ["fp32"] + variants}
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()),
                      settings=c, gdn_layers=layers, state_shape=shape, n_rows=len(rows))
    json.dump({"provenance": prov, "bytes_per_layer": sizes, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
