"""E2 (EXPERIMENTS.md): published selectors vs ours on exact token attention, equal token budgets.

  CUDA_VISIBLE_DEVICES=1 python scripts/compare_baselines.py configs/reasoning.yaml [--split explore] [--shard 0/1]

Needs the replayed keys (scripts/dump_keys.py). Per (trace, layer, budget): missing attention for
each method, and for token/chunk-level methods how many 16-token pages their selection touches per
page of budget (1 = page-aligned; transfers get scattered as it grows). Writes one JSON per shard;
--merge combines the shards into results/baselines_<split>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from safetensors.torch import load_file

from kvfetch.baselines import (arkvale_scores, attention_by_group, expand, landmark_scores, partial_channel_scores,
                               pick_channels, quest_scores, shared_token_set, topk_mask, unit_summaries)
from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.provenance import provenance, write_json
from kvfetch.selectors import run_selector
from kvfetch.sim import coverage, existing_pages
from kvfetch.tracing.schema import load_meta, load_trace

TOKEN_METHODS = ["exact_token", "infinigen", "lessismore_same", "lessismore_prev", "shadowkv"]
PAGE_METHODS = ["true_page", "quest", "arkvale"]


def fit_pages(score: torch.Tensor, P: int) -> torch.Tensor:
    """Page scores from the replayed keys cover pages with at least one key; pad to the trace's P."""
    return torch.nn.functional.pad(score, (0, max(0, P - score.shape[-1])), value=float("-inf"))[..., :P]


def analyze(path: Path, cfg: dict, device: str) -> dict:
    e2, defs, sc = cfg["e2"], cfg["definitions"], cfg["selectors"]
    ps = defs["page_size"]
    tr = load_trace(path)
    meta = tr.meta
    keys_all = load_file(str(Path(e2["keys_dir"]) / path.parent.name / path.name))["keys"]   # [L, S, N, d]
    T, _, S, P = tr.mass.shape
    pl, N = meta["prompt_len"], keys_all.shape[2]
    elig = eligibility_matrix(pl, T, P, ps, defs["pinned"])
    exist = existing_pages(pl, T, P, ps)
    pinned_pg = exist & ~elig
    b_pages = np.maximum(0, np.array(e2["budgets_tokens"])[:, None] // ps - pinned_pg.sum(-1)[None])   # [nb, T]
    tok_page = np.arange(N) // ps
    n_keys = pl + np.arange(T)
    present = np.arange(N)[None, :] < n_keys[:, None]
    allowed_tok = torch.from_numpy(elig[:, tok_page] & present).to(device)
    pinned_tok = torch.from_numpy(pinned_pg[:, tok_page] & present).to(device)
    allowed_pg = torch.from_numpy(elig).to(device)
    nb = len(e2["budgets_tokens"])
    out = {"task": meta["task"], "sample_id": meta["sample_id"], "n_steps": T, "score": meta.get("score"), "layers": {}}
    prev_sets = None                                       # LessIsMore sets chosen at the previous traced layer
    for li, layer in enumerate(meta["layers"]):
        k = keys_all[li].to(device).float()
        q = torch.from_numpy(tr.extra["q"][:, li].astype(np.float32)).to(device)
        H, d = q.shape[1:]
        scaling = d ** -0.5
        pg, ch = unit_summaries(k, ps), unit_summaries(k, e2["shadowkv"]["chunk"])
        chans = pick_channels(q[0], k[:, :pl], max(1, int(round(e2["infinigen"]["partial_ratio"] * d))))
        acc = {m: np.zeros(nb) for m in TOKEN_METHODS + PAGE_METHODS}
        spread = {m: np.zeros(nb) for m in TOKEN_METHODS}
        m_rec = np.zeros((T, S, P), np.float32)
        qb_rec = np.zeros((T, S, P), np.float32)
        qm_rec = np.zeros((T, S, P), np.float32)
        cur_sets = torch.zeros((nb, T, N), dtype=torch.bool)
        # InfiniGen + observed history (token level, stateful): history starts from the prefill
        # window's page mass spread over each page's tokens, then holds each token's renormalised
        # attention from the last step it was attended.
        hws = e2["infinigen"]["history_weights"]
        init_tok = torch.from_numpy(tr.obs_mass[:, li].astype(np.float32).mean(0) / ps).to(device)[:, tok_page]
        hist = init_tok[None, None].repeat(nb, len(hws), 1, 1)                     # [nb, nw, S, N]
        acc_h = np.zeros((nb, len(hws)))
        cs = e2["step_chunk"]
        for a0 in range(0, T, cs):
            sl = slice(a0, min(T, a0 + cs))
            c = sl.stop - sl.start
            nk = torch.from_numpy(n_keys[sl]).to(device)
            att = attention_by_group(q[sl], k, nk, scaling)                       # [c, S, g, N]
            a = att.mean(2)
            pad = P * ps - N
            m_c = torch.nn.functional.pad(a, (0, pad)).reshape(c, S, P, ps).sum(-1)
            m_rec[sl] = m_c.cpu().numpy()
            qpg = fit_pages(quest_scores(q[sl], pg, scaling), P)
            qb_rec[sl] = qpg.cpu().numpy()
            qm_rec[sl] = fit_pages(landmark_scores(q[sl], pg, scaling), P).cpu().numpy()   # mean-key logits
            ark = fit_pages(arkvale_scores(q[sl], pg, scaling), P)
            lm = landmark_scores(q[sl], ch, scaling)
            spec = partial_channel_scores(q[sl], k, chans, scaling)
            spec = torch.softmax(spec.masked_fill(~(allowed_tok[sl] | pinned_tok[sl])[:, None, None, :], float("-inf")), -1).mean(2)
            n_ch = lm.shape[-1]
            ch_page = torch.arange(n_ch, device=device) * e2["shadowkv"]["chunk"] // ps
            allowed_ch = allowed_pg[sl][:, ch_page.clamp(max=P - 1)]
            heads = att.flatten(1, 2)                                             # [c, H, N]
            base = (a * pinned_tok[sl][:, None, :]).sum(-1)                       # pinned mass [c, S]
            tot = a.sum(-1)
            for bi in range(nb):
                bp = torch.from_numpy(b_pages[bi, sl]).to(device)
                bt = bp * ps
                masks = {}
                masks["exact_token"] = topk_mask(a, bt, allowed_tok[sl])
                masks["infinigen"] = topk_mask(spec, bt, allowed_tok[sl])
                n_rec = torch.round(e2["lessismore"]["recent_share"] * bt).long()
                recent = topk_mask(torch.arange(N, device=device, dtype=torch.float32).expand(c, 1, N), n_rec, allowed_tok[sl])[:, 0]
                shared = shared_token_set(heads, bt - n_rec, allowed_tok[sl] & ~recent) | recent
                masks["lessismore_same"] = shared[:, None, :].expand(c, S, N)
                cur_sets[bi, sl] = shared.cpu()
                if prev_sets is not None:
                    masks["lessismore_prev"] = prev_sets[bi, sl].to(device)[:, None, :].expand(c, S, N) & allowed_tok[sl][:, None, :]
                n_out = torch.round(e2["shadowkv"]["outlier_share"] * bt / e2["shadowkv"]["chunk"]).long()
                outl = topk_mask(-ch["min_cos"][None].expand(c, -1, -1), n_out, allowed_ch)
                bc = bt // e2["shadowkv"]["chunk"] - n_out
                land = topk_mask(lm.masked_fill(outl, float("-inf")), bc, allowed_ch) | outl
                masks["shadowkv"] = expand(land, e2["shadowkv"]["chunk"], N)
                masks["true_page"] = expand(topk_mask(m_c, bp, allowed_pg[sl]), ps, N)
                masks["quest"] = expand(topk_mask(qpg, bp, allowed_pg[sl]), ps, N)
                masks["arkvale"] = expand(topk_mask(ark, bp, allowed_pg[sl]), ps, N)
                for wi, w in enumerate(hws):
                    for j in range(c):
                        f = (1.0 - w) * spec[j] + w * hist[bi, wi]
                        sel = topk_mask(f[None], bt[j:j + 1], allowed_tok[sl][j:j + 1])[0]
                        att_j = sel | pinned_tok[sl][j][None, :]
                        mj = a[j] * att_j
                        acc_h[bi, wi] += float((mj.sum(-1) / tot[j]).sum())
                        obs = mj / mj.sum(-1, keepdim=True).clamp(min=torch.finfo(mj.dtype).tiny)
                        hist[bi, wi] = torch.where(att_j, obs, hist[bi, wi])
                for name, msk in masks.items():
                    cov = (base + (a * (msk & allowed_tok[sl][:, None, :])).sum(-1)) / tot
                    acc[name][bi] += float(cov.sum())
                    if name in spread:
                        touched = torch.nn.functional.pad(msk, (0, pad)).reshape(c, S, P, ps).any(-1).sum(-1).float()
                        spread[name][bi] += float((touched / bp.clamp(min=1)[:, None]).sum())
        res = {}
        init = tr.obs_mass[:, li].astype(np.float32).mean(0)
        for bi, btok in enumerate(e2["budgets_tokens"]):
            r = {}
            for name in TOKEN_METHODS + PAGE_METHODS:
                if name == "lessismore_prev" and prev_sets is None:
                    continue
                r[name] = {"missing": 1.0 - acc[name][bi] / (T * S)}
                if name in spread:
                    r[name]["pages_touched_per_budget_page"] = spread[name][bi] / (T * S)
            for wi, w in enumerate(hws):
                r[f"infinigen+hist_w{w:g}"] = {"missing": 1.0 - acc_h[bi, wi] / (T * S)}
            for name in e2["ours"]:
                sel = run_selector(sc["candidates"][name], m=m_rec, qb=qb_rec, init=init, eligible=elig, exist=exist,
                                   b=b_pages[bi], qm=qm_rec)
                r[f"ours:{name}"] = {"missing": 1.0 - coverage(sel | pinned_pg[:, None, :], m_rec)}
            res[str(btok)] = r
        out["layers"][str(layer)] = res
        prev_sets = cur_sets
        del k, q, att
        torch.cuda.empty_cache()
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    groups = {t: [r for r in rows if r["task"] == t] for t in sorted({r["task"] for r in rows})}
    groups["ALL"] = rows
    for task, rs in groups.items():
        s = {"n_samples": len(rs)}
        for btok in rs[0]["layers"][next(iter(rs[0]["layers"]))]:
            names = sorted({n for r in rs for l in r["layers"].values() for n in l[btok]})
            s[btok] = {}
            for n in names:
                vals = [l[btok][n] for r in rs for l in r["layers"].values() if n in l[btok]]
                s[btok][n] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        out[task] = s
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="explore", choices=["explore", "confirm"])
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge", action="store_true", help="combine shard outputs and print the table")
    args = p.parse_args()
    cfg = config_from_args(args)
    e2 = cfg["e2"]
    shard_dir = Path(e2["shard_dir"])
    if args.merge:
        rows = [r for f in sorted(shard_dir.glob(f"{args.split}_*.json")) for r in json.load(open(f))["per_sample"]]
        summary = summarize(rows)
        a = summary["ALL"]
        print(f"split={args.split} traces={len(rows)}")
        names = sorted(a[str(e2["budgets_tokens"][0])], key=lambda n: a[str(e2["budgets_tokens"][0])][n]["missing"])
        print(f"{'method':>24s} " + " ".join(f"{b:>16s}" for b in map(str, e2["budgets_tokens"])))
        for n in names:
            cells = []
            for b in map(str, e2["budgets_tokens"]):
                v = a[b].get(n)
                cells.append("-" if v is None else f"{100 * v['missing']:6.2f}%" + (f" x{v['pages_touched_per_budget_page']:4.1f}" if "pages_touched_per_budget_page" in v else "      "))
            print(f"{n:>24s} " + " ".join(f"{c:>16s}" for c in cells))
        write_json(f"results/baselines_{args.split}.json",
                   {"provenance": provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"],
                                             split=args.split, n_traces=len(rows)), "summary": summary, "per_sample": rows})
        print(f"\nwrote results/baselines_{args.split}.json")
        return
    i, n = map(int, args.shard.split("/"))
    paths = [pth for pth in sorted(Path(cfg["sweep"]["trace_dir"]).rglob("*.safetensors"))
             if load_meta(pth).get("split") == args.split][i::n]
    rows = []
    for pth in paths:
        with torch.no_grad():
            rows.append(analyze(pth, cfg, args.device))
        print(f"[e2] {pth.parent.name}/{pth.name} done", flush=True)
    write_json(shard_dir / f"{args.split}_{i}of{n}.json",
               {"provenance": provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"],
                                         split=args.split), "per_sample": rows})


if __name__ == "__main__":
    main()
