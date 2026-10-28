"""Chained store -> resume -> store: does checkpoint quantization error accumulate?

Same traces and checkpoint positions as ckpt_precision.py. The 2,048-token continuation runs in chunks of
``chain_tokens`` (one vLLM block); a variant rounds every GDN recurrent state through its format before
each chunk -- the checkpoint at P and again at every later block boundary, as an agent loop or a cached
decode region would store and resume it. The reference runs the same chunks without rounding; "split" is
the reference's own difference from one unchunked call (the floor for this chunking). One-shot rounding
(only at P) is included for comparison with the chained version.

Usage: ckpt_chain.py CONFIG OUT.json [--shard i/n]
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from ckpt_precision import REPO, compare, compress, continuation_logprobs, linear_layers, prefill, summarize
from kvfetch.config import load_config
from kvfetch.models import encode, format_prompt, load_model
from kvfetch.provenance import provenance
from run_e3 import problems, thinking


@torch.no_grad()
def chunked_logprobs(model, cache, cont, dev, chunk: int, layers, fmt: str | None, every: bool):
    out = []
    for a in range(0, len(cont), chunk):
        if fmt is not None and (every or a == 0):
            for i in layers:
                st = cache.layers[i].recurrent_states[0]
                st.copy_(compress(st, fmt))
        o = model(input_ids=torch.tensor([cont[a:a + chunk]], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        out.append(o.logits[0].float().log_softmax(-1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    cfg = thinking(load_config(args.config))
    c = cfg["ckpt"]
    mc = cfg["hybrid"]["models"][c["model"]]
    dev = torch.device(args.device)
    model, tok = load_model(mc["name"], cfg, dev)
    layers = linear_layers(model)
    seqs = []
    for s in problems({**cfg, "e3": cfg["hybrid"]["h3"]}, tok):
        f = REPO / c["runs"] / s.task / f"{s.index:04d}_s{c['sample']}.json"
        if f.exists():
            r = json.loads(f.read_text())
            prompt = encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"]
            assert len(prompt) == r["prompt_tokens"], f
            seqs.append({"task": s.task, "index": s.index, "ids": prompt + r["token_ids"], "pl": len(prompt)})
    i, n = map(int, args.shard.split("/"))
    seqs = seqs[i::n]
    W, K = c["window"], c["chain_tokens"]
    rows, t0 = [], time.time()
    for s in seqs:
        for off in c["offsets"]:
            P = s["pl"] + off
            if len(s["ids"]) < P + W + 1:
                continue
            cont = s["ids"][P:P + W]
            targets = torch.tensor(s["ids"][P + 1:P + W + 1], device=dev)
            base = prefill(model, s["ids"][:P], dev)
            ref = chunked_logprobs(model, copy.deepcopy(base), cont, dev, K, layers, None, False)
            row = {"task": s["task"], "index": s["index"], "P": P, "offset": off, "variants": {}}
            one = continuation_logprobs(model, copy.deepcopy(base), cont, dev)
            row["variants"]["split"] = summarize(compare(ref, one, targets, c["logit_chunk"]), c["bins"], c["curve_step"])
            del one
            for v in c["chain_variants"]:
                for every in (False, True):
                    lp = chunked_logprobs(model, copy.deepcopy(base), cont, dev, K, layers, v, every)
                    row["variants"][f"{v}_{'chain' if every else 'once'}"] = summarize(
                        compare(ref, lp, targets, c["logit_chunk"]), c["bins"], c["curve_step"])
                    del lp
            rows.append(row)
            del base, ref
            torch.cuda.empty_cache()
            kl = {k: f"{d['kl']['all']:.2e}" for k, d in row["variants"].items()}
            print(f"[{time.time() - t0:6.0f}s] {s['task']} {s['index']} P={P}: {kl}", flush=True)
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()),
                      settings=c, gdn_layers=layers, n_rows=len(rows))
    json.dump({"provenance": prov, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
