"""E2 input (EXPERIMENTS.md): full post-RoPE keys at the traced layers, rebuilt by replaying a
trace's tokens: prompt prefill, then one generated token per forward, as in the traced run. (One
forward over the whole sequence takes a different bf16 path: deep-layer keys drift by ~0.5%,
which moves up to ~0.4 of a head's attention mass at layers 29/32.)

  CUDA_VISIBLE_DEVICES=1 python scripts/dump_keys.py configs/reasoning.yaml [--split explore] [--shard 0/3]

Writes keys [L, KVH, N, d] (fp16) to e2.keys_dir/<task>/<idx>.safetensors, N = prompt + steps - 1
(the keys the last traced query attends). Checks each file against its trace: the attention mass
recomputed from the recorded queries and these keys must match the traced mass at e2.check_steps.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import DynamicCache

from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import load_model
from kvfetch.provenance import provenance
from kvfetch.tracing.schema import load_meta, load_trace


def recomputed_mass(q: np.ndarray, keys: torch.Tensor, n_keys: int, page_size: int, n_pages: int) -> np.ndarray:
    """q [H, d] (one step), keys [KVH, N, d] -> KV-group mean attention mass per page [KVH, n_pages]."""
    H, d = q.shape
    KVH = keys.shape[0]
    k = keys[:, :n_keys].float().repeat_interleave(H // KVH, 0)                    # [H, n, d]
    p = torch.softmax(torch.einsum("hd,hnd->hn", torch.from_numpy(q).float().to(k.device), k) * d ** -0.5, -1)
    pad = n_pages * page_size - n_keys
    pages = torch.nn.functional.pad(p, (0, pad)).reshape(H, n_pages, page_size).sum(-1)
    return pages.reshape(KVH, H // KVH, n_pages).mean(1).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="explore", choices=["explore", "confirm"])
    p.add_argument("--shard", default="0/1", help="i/n: this worker takes every n-th trace from i")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = config_from_args(args)
    sw, e2, ps = cfg["sweep"], cfg["e2"], cfg["definitions"]["page_size"]
    i, n = map(int, args.shard.split("/"))
    paths = [pth for pth in sorted(Path(sw["trace_dir"]).rglob("*.safetensors"))
             if load_meta(pth).get("split") == args.split][i::n]
    model, _ = load_model(sw["model"], cfg, args.device)
    prov = provenance(cfg, model=sw["model"], context_length=sw["max_context"])
    for pth in paths:
        out = Path(e2["keys_dir"]) / pth.parent.name / pth.name
        if out.exists():
            continue
        tr = load_trace(pth)
        meta, layers = tr.meta, tr.meta["layers"]
        T, P = tr.mass.shape[0], tr.mass.shape[-1]
        ids = np.concatenate([tr.prompt_ids, tr.tokens[:-1]]).astype(np.int64)
        cache = DynamicCache()
        x = torch.from_numpy(ids)[None].to(args.device)
        pl = meta["prompt_len"]
        with torch.no_grad():
            model(x[:, :pl], past_key_values=cache, use_cache=True, logits_to_keep=1)
            for a in range(pl, ids.size):
                model(x[:, a:a + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
        keys = torch.stack([cache.layers[l].keys[0] for l in layers])                # [L, KVH, N, d]
        del cache
        diffs = []
        for s in np.linspace(0, T - 1, e2["check_steps"]).astype(int):
            for li in range(len(layers)):
                got = recomputed_mass(tr.extra["q"][s, li], keys[li], meta["prompt_len"] + s, ps, P)
                diffs.append(float(np.abs(got - tr.mass[s, li].astype(np.float32)).max()))
        if max(diffs) > e2["check_atol"]:
            raise RuntimeError(f"{pth}: recomputed mass differs from the trace by {max(diffs):.4f}")
        out.parent.mkdir(parents=True, exist_ok=True)
        info = {**prov, "trace": str(pth), "layers": layers, "n_keys": int(ids.size), "rope": "post",
                "check_max_abs_diff": max(diffs)}
        save_file({"keys": keys.to(torch.float16).cpu().contiguous()}, str(out), metadata={"meta": json.dumps(info)})
        print(f"[keys] {pth.parent.name}/{pth.name} N={ids.size} check max|dm|={max(diffs):.2e}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
