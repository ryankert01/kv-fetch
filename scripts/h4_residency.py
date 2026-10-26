"""H4 (EXPERIMENTS.md, "Hybrid models"): the pages a host-resident KV must move per decode step, on
Qwen3.5-4B's own sequences, for H3's selector ("ours" = hybrid.h2.chosen) at h4.budget_tokens, and a
step-time / capacity model from the measured costs (scripts/h4_costs.py). Run with .venv-hybrid.

  CUDA_VISIBLE_DEVICES=4 .venv-hybrid/bin/python scripts/h4_residency.py configs/hybrid.yaml screen
  .venv-hybrid/bin/python scripts/h4_residency.py configs/hybrid.yaml merge

Sequences: sample h1.sample of H3's full-attention run, every h4.residency.every-th explore problem, up
to h4.residency.max_gen_tokens ("main"); and, as a cross-check of the trajectory (H4 audit), a few of
H3's ours@512 samples ("cross", h4.residency.cross_check). Queries and attention come from one dense
replay forward (scripts/h1_screen.capture), as in H1.

Policies, per attention layer and step. The sink and recent pages (the engine's pinned set) stay on the
GPU; a page that ages out of the recent window is resident (it was just written); after prefill every
prompt page is resident and the cache is trimmed to its capacity; the GPU page cache holds a constant
c x (the largest per-step budget) pages, LRU, evicted at the end of each step:
  exact c      attend the pages selected with the step's own query (as H3); the selected pages not
               resident are fetched before the layer can attend (synchronous),
  la3 c+k      exact, plus a prefetch of the top k x budget pages ranked with the query predicted from
               the hidden state three layers earlier (H-c), issued during the GDN layers before the
               attention layer: prefetch traffic and the residual synchronous misses; selection
               (accuracy) is unchanged,
  lag1 c       step t attends the pages ranked with step t-1's query over step t's eligible pages and
               budget (the next step's eligibility is known in advance): the fetch overlaps a whole
               step; the cost is the missing attention of the stale ranking vs the exact one.
Per-step counts (synchronous misses, prefetch traffic, context length) are stored per shard (.npz);
merge samples batches of B (sequence, step) pairs from the steps whose context falls in each
h4.contexts bucket and prices every sample with the measured round-trip / zero-copy fetch times,
GDN-window lengths and slowdowns under concurrent copies (expected overheads, not means inside max).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from kvfetch import selectors_gpu as G
from kvfetch import sparse_decode as SD
from kvfetch.baselines import attention_by_group, landmark_scores, unit_summaries
from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.models import encode, format_prompt, load_model
from kvfetch.provenance import provenance, write_json
from kvfetch.sim import existing_pages
from compare_pages import Scorer, fit_pages
from run_e3 import problems, thinking
import h1_screen as H1


def policies(rc: dict) -> list[dict]:
    la = rc["prefetch"]["lookahead"]
    out = [{"name": f"exact c{c:g}", "need": "exact", "cache": c, "pre": 0} for c in rc["cache_multiples"]]
    out += [{"name": f"la{la} c{c:g}+k{k:g}", "need": "exact", "cache": c, "pre": k} for c, k in rc["prefetch"]["cache_plus"]]
    out += [{"name": f"lag1 c{c:g}", "need": "lag1", "cache": c, "pre": 0} for c in rc["lag1_cache_multiples"]]
    return out


def simulate(need: torch.Tensor, pre: torch.Tensor, aged: torch.Tensor, cap: torch.Tensor):
    """LRU page cache per policy. need, pre [T, K, P] (pages attended / prefetched at step t), aged
    [T, P] (pages leaving the pinned recent window at t, or every prompt page at t = 0: resident),
    cap [K] (cache pages). Returns per step the synchronous misses and the prefetch traffic, each
    [T, K], and the most pages resident at once (before the end-of-step eviction) [K]. Use order
    within a step: aged pages, then prefetched, then attended; ties evict the lower page index."""
    T, K, P = need.shape
    dev = need.device
    last = torch.full((K, P), -1, dtype=torch.long, device=dev)          # 3 t + {0, 1, 2}; -1: not resident
    sync = torch.zeros((T, K), dtype=torch.long, device=dev)
    traffic = torch.zeros_like(sync)
    peak = torch.zeros(K, dtype=torch.long, device=dev)
    ar = torch.arange(P, device=dev).expand(K, P)
    big = torch.iinfo(torch.long).max
    for t in range(T):                                    # no host syncs: the loop stays on the GPU queue
        last = torch.where(aged[t][None], 3 * t, last)
        traffic[t] = (pre[t] & (last < 0)).sum(-1)
        last = torch.where(pre[t], 3 * t + 1, last)
        sync[t] = (need[t] & (last < 0)).sum(-1)
        last = torch.where(need[t], 3 * t + 2, last)
        res = last >= 0
        n_res = res.sum(-1)
        peak = torch.maximum(peak, n_res - aged[t].sum() * (t == 0))   # prompt pages at t = 0 are not cache traffic
        excess = (n_res - cap).clamp(min=0)
        order = torch.argsort(torch.where(res, last, big), dim=-1, stable=True)
        rank = torch.empty_like(order).scatter_(-1, order, ar)
        last = last.masked_fill(res & (rank < excess[:, None]), -1)
    return sync, traffic, peak


def selections(score: torch.Tensor, score_la: torch.Tensor, elig: torch.Tensor, b: torch.Tensor, pols: list[dict]):
    """Page sets per step [T, P] for every policy: exact (the step's own query), lag1 (step t - 1's query
    ranked over step t's eligible pages and budget; step 0, the prompt's last query in the dense
    prefill, is exact) and the lookahead prefetches (top k x b by the predicted query)."""
    sel = G.select_from_ranks(G.static_ranks(score, elig, "logit"), elig, b, "sum")[:, 0]
    lag_score = torch.cat([score[:1], score[:-1]], 0)
    lag = G.select_from_ranks(G.static_ranks(lag_score, elig, "logit"), elig, b, "sum")[:, 0]
    la_ranks = G.static_ranks(score_la, elig, "logit")
    pre = {k: G.select_from_ranks(la_ranks, elig, (k * b).long(), "sum")[:, 0] for k in sorted({p["pre"] for p in pols if p["pre"]})}
    return sel, lag, pre


def summary_scores(q: torch.Tensor, summ: dict, P: int, chunk: int, fmt: str) -> torch.Tensor:
    """Mean-key logits [T, S, P] in the H2 format of hybrid.h2.chosen (mean16 / mean8)."""
    mean = {"mean16": summ["mean"].half().float(), "mean8": summ["mean"].to(torch.float8_e4m3fn).float()}[fmt]
    T, S = q.shape[0], mean.shape[0]
    out = torch.empty((T, S, P), device=q.device)
    for a in range(0, T, chunk):
        out[a:a + chunk] = fit_pages(landmark_scores(q[a:a + chunk].float(), {"mean": mean}, q.shape[-1] ** -0.5), P)
    return out


def page_mass(q: torch.Tensor, k: torch.Tensor, pl: int, P: int, ps: int, chunk: int) -> torch.Tensor:
    T = q.shape[0]
    n_keys = torch.arange(T, device=q.device) + pl
    m = torch.empty((T, k.shape[0], P), device=q.device)
    pad = P * ps - k.shape[1]
    for a in range(0, T, chunk):
        sl = slice(a, min(T, a + chunk))
        att = torch.nn.functional.pad(attention_by_group(q[sl].float(), k, n_keys[sl], q.shape[-1] ** -0.5), (0, pad))
        m[sl] = att.reshape(*att.shape[:-1], P, ps).sum(-1).mean(2)
    return m


def analyze_layer(c: dict, pl: int, cfg: dict, device: str) -> tuple[dict, dict]:
    """Returns the layer's summary and its per-step arrays (sync, traffic [T, K] int16)."""
    h4, defs, ps = cfg["hybrid"]["h4"], cfg["definitions"], cfg["definitions"]["page_size"]
    rc = h4["residency"]
    fmt = cfg["hybrid"]["h2"]["chosen"]
    q, k = c["q"], c["k"]
    T = q.shape[0]
    P = math.ceil((pl + T - 1) / ps)
    elig_np = eligibility_matrix(pl, T, P, ps, defs["pinned"])
    exist_np = existing_pages(pl, T, P, ps)
    elig, exist = torch.from_numpy(elig_np).to(device), torch.from_numpy(exist_np).to(device)
    b = H1.budgets_for([h4["budget_tokens"]], ps, exist_np, elig_np, device)[str(h4["budget_tokens"])]
    summ = unit_summaries(k, ps)
    m = page_mass(q, k, pl, P, ps, rc["step_chunk"])
    S = m.shape[1]
    score = Scorer(m, torch.zeros_like(m), (exist & ~elig)[:, None, :])
    pols = policies(rc)
    sel, lag, pre = selections(summary_scores(q, summ, P, rc["step_chunk"], fmt),
                               summary_scores(c["la"][rc["prefetch"]["lookahead"]], summ, P, rc["step_chunk"], fmt), elig, b, pols)
    aged = elig.clone()
    aged[1:] &= ~elig[:-1]
    need = torch.stack([sel if p["need"] == "exact" else lag for p in pols], 1)
    pre_t = torch.stack([pre[p["pre"]] if p["pre"] else torch.zeros_like(sel) for p in pols], 1)
    bmax = int(b.max())
    cap = torch.tensor([int(p["cache"] * bmax) for p in pols], device=device)
    sync, traffic, peak = simulate(need, pre_t, aged, cap)
    if int((lag.sum(-1) > b).sum()) or int((sel.sum(-1) > b).sum()):
        raise RuntimeError("a selection exceeds its step's budget")
    out = {"n_steps": T, "n_pages": P, "budget_pages_max": bmax,
           "missing": {"exact": score.whole(sel[:, None].expand(T, S, P), b, False)["missing"],
                       "lag1": score.whole(lag[:, None].expand(T, S, P), b, False)["missing"]},
           "policies": {}}
    for i, p in enumerate(pols):
        out["policies"][p["name"]] = {"sync_mean": float(sync[:, i].float().mean()), "prefetch_mean": float(traffic[:, i].float().mean()),
                                      "peak_resident": int(peak[i]), "cache_pages": int(cap[i]), **p}
    arrays = {"sync": sync.to(torch.int16).cpu().numpy(), "traffic": traffic.to(torch.int16).cpu().numpy()}
    del m, score, need, pre_t
    torch.cuda.empty_cache()
    return out, arrays


def model_sizes(model, cfg, device) -> dict:
    """Byte counts from the engine's own buffers (no constants)."""
    defs = cfg["definitions"]
    eng = SD.SparseEngine(model, SD.Method("full"), batch=1, max_len=defs["page_size"], page_size=defs["page_size"],
                          sink_pages=defs["pinned"]["sink_pages"], recent_tokens=defs["pinned"]["recent_tokens"],
                          obs_window=cfg["trace"]["prefill_obs_window"], device=device)
    types_ = model.config.layer_types
    gdn_before = []                                       # GDN layers right before each attention layer
    for l in eng.attn_layers:
        j = l - 1
        while j >= 0 and types_[j] == "linear_attention":
            j -= 1
        gdn_before.append(l - 1 - j)
    out = {"weights": sum(t.numel() * t.element_size() for t in model.parameters()),
           "state_per_seq": sum(t.numel() * t.element_size() for l in eng.lin_layers
                                for t in (eng.layers[l].conv_states[0], eng.layers[l].recurrent_states[0])),
           "kv_per_token": 2 * eng.k[:, 0, :, 0].numel() * eng.k.element_size(),
           "n_layers": model.config.num_hidden_layers, "n_attn": eng.L, "head_dim": eng.d, "np_pinned": eng.np_pinned,
           "vocab": model.config.vocab_size, "gdn_before_attn": gdn_before}
    del eng
    return out


def screen(cfg: dict, args) -> None:
    h1, h4 = cfg["hybrid"]["h1"], cfg["hybrid"]["h4"]
    rc = h4["residency"]
    mc = cfg["hybrid"]["models"][rc["model"]]
    model, tok = load_model(mc["name"], cfg, args.device, attn_implementation=H1.CAPTURE)
    prov = provenance(cfg, model=mc["name"], context_length=None, residency=rc, budget_tokens=h4["budget_tokens"],
                      h2_chosen=cfg["hybrid"]["h2"]["chosen"])
    sizes = model_sizes(model, cfg, args.device)
    if sizes["kv_per_token"] != h4["kv_bytes_per_token"]:
        raise RuntimeError(f"h4.kv_bytes_per_token {h4['kv_bytes_per_token']} != the engine's {sizes['kv_per_token']}")
    if rc["prefetch"]["lookahead"] > min(sizes["gdn_before_attn"]):
        raise RuntimeError(f"lookahead {rc['prefetch']['lookahead']} exceeds the GDN window {min(sizes['gdn_before_attn'])}")
    probs = problems({**cfg, "e3": cfg["hybrid"]["h3"]}, tok)
    sets = {"main": (rc["sequences"], probs[::rc["every"]]),
            "cross": (rc["cross_check"]["sequences"], probs[::rc["cross_check"]["every"]])}
    for kind, (src, plist) in sets.items():
        plist = plist[:args.limit]
        if not args.max_gen and not args.limit:
            write_json(Path(rc["shard_dir"]) / kind / "manifest.json",
                       {"provenance": prov, "expected": [f"{s.task}/{s.index:04d}" for s in plist]})
        for s in plist:
            out = Path(rc["shard_dir"]) / kind / s.task / f"{s.index:04d}.json"
            f = Path(src) / s.task / f"{s.index:04d}_s{h1['sample']}.json"
            if out.exists():
                continue
            if not f.exists():
                print(f"[h4] {kind} {s.task}/{s.index}: no sequence at {f}", flush=True)
                continue
            r = json.loads(f.read_text())
            prompt = encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"]
            if len(prompt) != r["prompt_tokens"]:
                raise RuntimeError(f"{f}: prompt has {len(prompt)} tokens, the run had {r['prompt_tokens']}")
            gen = r["token_ids"][:args.max_gen or rc["max_gen_tokens"]]
            pl = len(prompt)
            cap = H1.capture(model, prompt + gen[:-1], pl, mc["layers"], [rc["prefetch"]["lookahead"]], gdn=False)
            layers, arrays = {}, {}
            for l in mc["layers"]:
                if cap[l]["q_self_err"] > rc["q_self_err_max"]:
                    raise RuntimeError(f"recomputed queries differ from the model's at layer {l}: {cap[l]['q_self_err']}")
                layers[str(l)], arr = analyze_layer(cap[l], pl, cfg, args.device)
                arrays.update({f"{n}_{l}": v for n, v in arr.items()})
                print(f"[h4] {kind} {s.task}/{s.index} layer {l}: " + ", ".join(
                    f"{n} {v['sync_mean']:.2f}+{v['prefetch_mean']:.2f}" for n, v in layers[str(l)]["policies"].items()), flush=True)
            del cap
            torch.cuda.empty_cache()
            if args.max_gen:
                print("dry run (--max-gen): not written")
                continue
            np.savez_compressed(out.with_suffix(".npz"), context=pl + np.arange(len(gen), dtype=np.int32), **arrays)
            write_json(out, {"provenance": prov, "sizes": sizes, "task": s.task, "index": s.index, "prompt_len": pl,
                             "n_gen": len(gen), "stopped": r["stopped"], "score": r["score"], "run": r["run"], "layers": layers})


# ---------------------------------------------------------------------------- merge: cost model
def interp(table: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear in x (pages) over a measured {pages: ms} table, linear extension of the last
    segment beyond it (flagged); x = 0 costs nothing. Returns (ms, extrapolated) arrays like x."""
    ks = sorted((k for k in table if table[k] is not None), key=float)
    xs, ys = np.array([float(k) for k in ks]), np.array([table[k] for k in ks])
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xs, ys)
    ext = x > xs[-1]
    y = np.where(ext, ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1]), y)
    return np.where(x > 0, y, 0.0), ext


def load_shards(rc: dict, kind: str) -> tuple[list[dict], list[dict], dict]:
    root = Path(rc["shard_dir"]) / kind
    man = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {"expected": []}
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/*.json"))]
    arrs = [dict(np.load(Path(p).with_suffix(".npz"))) for p in sorted(root.glob("*/*.json"))]
    got = {f"{r['task']}/{r['index']:04d}" for r in rows}
    keys = {(r["provenance"]["h2_chosen"], r["provenance"]["budget_tokens"], r["provenance"]["git"]["commit"],
             tuple(next(iter(r["layers"].values()))["policies"])) for r in rows}
    check = {"expected": len(man["expected"]), "present": len(got & set(man["expected"])),
             "complete": bool(man["expected"]) and set(man["expected"]) <= got, "consistent": len(keys) <= 1}
    return rows, arrs, check


def merge(cfg: dict) -> None:
    h4 = cfg["hybrid"]["h4"]
    rc = h4["residency"]
    mkey = rc["model"]
    rows, arrs, check = load_shards(rc, "main")
    if not rows:
        raise SystemExit("no residency shards")
    if not check["consistent"]:
        raise SystemExit("shards disagree on h2_chosen / budget / commit / policies")
    sz = rows[0]["sizes"]
    lname = list(rows[0]["layers"])
    pol_names = list(rows[0]["layers"][lname[0]]["policies"])
    pinfo = {n: {k: rows[0]["layers"][lname[0]]["policies"][n][k] for k in ("need", "cache", "pre")} for n in pol_names}
    steps = np.array([r["layers"][l]["n_steps"] for r in rows for l in lname], dtype=float)

    def wmean(key, sub=None):                             # step-weighted over sequences x layers
        v = np.array([(r["layers"][l][key] if sub is None else r["layers"][l][key][sub]) for r in rows for l in lname], dtype=float)
        return float((v * steps).sum() / steps.sum())
    pol = {n: {"sync_mean": float(sum(a[f"sync_{l}"][:, i].sum() for a in arrs for l in lname) / steps.sum()),
               "prefetch_mean": float(sum(a[f"traffic_{l}"][:, i].sum() for a in arrs for l in lname) / steps.sum()),
               "peak_resident_max": max(r["layers"][l]["policies"][n]["peak_resident"] for r in rows for l in lname), **pinfo[n]}
           for i, n in enumerate(pol_names)}
    miss = {k: wmean("missing", k) for k in ("exact", "lag1")}
    cross_rows, cross_arrs, cross_check = load_shards(rc, "cross")
    cross = None
    if cross_rows:
        cs = np.array([r["layers"][l]["n_steps"] for r in cross_rows for l in lname], dtype=float)
        cross = {"n_sequences": len(cross_rows), "check": cross_check,
                 "sync_mean": {n: float(sum(a[f"sync_{l}"][:, i].sum() for a in cross_arrs for l in lname) / cs.sum())
                               for i, n in enumerate(pol_names)}}

    cpath = Path(cfg["paths"]["results_dir"]) / f"h4_costs_{mkey}.json"
    costs = json.loads(cpath.read_text())
    cp = costs["copy"]
    # a link is priced only with a measured table (>= 2 sizes) and, for the zero-copy gathers, a passing
    # data check; the staged path needs the step's graph split at each attention layer (host sync)
    split_ms = cp["graph_launch_us"] / 1e3
    links = {"staged": {"fetch": cp["staged_round_trip_ms"], "slow": "contiguous copy", "check": True, "split_ms": split_ms},
             "zero-copy": {"fetch": cp["zero_copy_ms"], "slow": "zero-copy gather", "check": cp.get("zero_copy_check"), "split_ms": 0.0},
             "zero-copy few": {"fetch": cp.get("zero_copy_few_ms", {}), "slow": "zero-copy gather (few programs)",
                               "check": cp.get("zero_copy_few_check"), "split_ms": 0.0}}
    dropped = {k: "check failed" if v["check"] is not True else "too few sizes" for k, v in links.items()
               if v["check"] is not True or sum(x is not None for x in v["fetch"].values()) < 2}
    links = {k: v for k, v in links.items() if k not in dropped}
    ov = costs["copy"]["overlap"]

    def slowdown(B, which, of):
        """Slowdown of the GDN layer (of = "gdn") or of the copy (of = "copy") when both run at once."""
        key, base = (f"gdn_with_{which}_ms", "gdn_alone_ms") if of == "gdn" else (f"{which}_with_gdn_ms", f"{which}_alone_ms")
        xs = [o["batch"] for o in ov if o.get(key)]
        ys = [o[key] / o[base] for o in ov if o.get(key)]
        return float(np.interp(B, xs, ys)) if xs else 1.0
    lay = costs["layers"]
    ours = next(m for m in {r["method"] for r in lay} if m.startswith("ours"))

    def t(B, n, method, key="attention_layer_ms"):
        r = next((r for r in lay if r["batch"] == B and r["context"] == n and r["method"] == method), None)
        return None if r is None or r.get("oom") or key not in r else r[key]
    n_attn, n_gdn = sz["n_attn"], sz["n_layers"] - sz["n_attn"]
    win = min(sz["gdn_before_attn"])
    rng = np.random.default_rng(rc["mc_seed"])
    ctx = np.concatenate([a["context"] for a in arrs])
    cat = {l: {k: np.concatenate([a[f"{k}_{l}"] for a in arrs]).astype(np.int64) for k in ("sync", "traffic")} for l in lname}
    edges = [0] + list(h4["contexts"])
    model = []
    beyond = float((ctx > edges[-1]).mean())             # steps past the last bucket: priced with it (open-ended)
    for B in h4["batches"]:
        for lo, n in zip(edges[:-1], edges[1:]):
            gdn, attn = t(B, n, "full", "gdn_layer_ms"), t(B, n, ours)
            if gdn is None or attn is None:
                continue
            plan = costs["plan_ms"][str(B)][str(n)]
            head = costs["head_ms"][str(B)]
            step = plan + n_gdn * gdn + n_attn * attn + head                     # all KV on the GPU, same selector
            last = n == edges[-1]
            pool = np.nonzero((ctx > lo) & ((ctx <= n) | last))[0]
            if not len(pool):
                continue
            row = {"batch": B, "context": n, "context_bucket": [lo, None if last else n], "bucket_steps": int(len(pool)),
                   "writeback_ms": B * sz["kv_per_token"] / (max(cp["contiguous_gbps"].values()) * 1e9) * 1e3,   # GPU -> host, other direction
                   "missing_costs": [],
                   "step_gpu_ms": step, "parts_ms": {"plan": plan, "gdn": gdn, "attn": attn, "head": head},
                   "step_gpu_full_ms": plan + n_gdn * gdn + n_attn * (t(B, n, "full") or float("nan")) + head,
                   "gdn_window_ms": win * gdn, "policies": {}}
            draws = pool[rng.integers(0, len(pool), size=(rc["mc_samples"], B))]     # B (sequence, step) pairs per sample
            for name, p in pol.items():
                i_pol = pol_names.index(name)
                la = next((r["ms"] for r in costs["lookahead"] if r["batch"] == B and r["context"] == n
                           and r["multiple"] == p["pre"] and not r.get("oom")), None) if p["pre"] else 0.0
                if la is None:
                    row["missing_costs"].append(f"lookahead x{p['pre']:g} ({name})")
                    continue
                for link, L in links.items():
                    f = slowdown(B, L["slow"], "gdn")
                    fc = slowdown(B, L["slow"], "copy")
                    o = np.zeros(len(draws))
                    tot = np.zeros(len(draws))
                    n_ext, n_all = 0, 0
                    for l in lname:
                        ts, e1 = interp(L["fetch"], cat[l]["sync"][draws, i_pol].sum(-1))
                        n_ext, n_all = n_ext + int(e1.sum()), n_all + e1.size
                        if p["need"] == "lag1":
                            tot += ts
                            continue
                        o += ts + L["split_ms"]
                        if p["pre"]:
                            tp, e2 = interp(L["fetch"], cat[l]["traffic"][draws, i_pol].sum(-1))
                            n_ext, n_all = n_ext + int(e2.sum()), n_all + e2.size
                            tp = tp * fc                                     # the copy slowed by the GDN layers it overlaps
                            w = win * gdn
                            o += la + np.maximum(0.0, tp - w) + (f - 1) * np.minimum(w, tp)
                    if p["need"] == "lag1":
                        tot = tot * fc
                        o = np.maximum(0.0, tot - step) + (f - 1) * np.minimum(step, tot)
                    eo = float(o.mean())
                    row["policies"].setdefault(name, {})[link] = {
                        "overhead_ms": eo, "throughput_ratio": step / (step + eo), "tokens_per_s": 1e3 * B / (step + eo),
                        "gdn_slowdown": f, "copy_slowdown": fc, "extrapolated_share": n_ext / n_all}
            model.append(row)
    calib = []
    for r in costs.get("steps", []):
        if r.get("oom") or not r["method"].startswith("ours"):
            continue
        m = next((x for x in model if x["batch"] == r["batch"] and x["context"] == r["context"]), None)
        if m:
            calib.append({"batch": r["batch"], "context": r["context"], "measured_ms": r["step_ms"],
                          "parts_ms": m["step_gpu_ms"], "ratio": r["step_ms"] / m["step_gpu_ms"]})
    # capacity per context: GPU bytes per sequence, all-on-GPU vs tiered (deployable fp8 means)
    share = H1.H2_BYTES[cfg["hybrid"]["h2"]["chosen"]](sz["head_dim"], cfg["definitions"]["page_size"]) / (2 * sz["head_dim"] * 2)
    page = h4["page_bytes"]
    free = (rc["gpu_mem_gib"] - rc["reserve_gib"]) * 2 ** 30 - sz["weights"]
    kmax_pages = h4["budget_tokens"] // cfg["definitions"]["page_size"]
    extras = (sz["vocab"] * (4 + 1)                       # fp32 logits + presence-penalty mask per slot
              + (kmax_pages + sz["np_pinned"]) * cfg["definitions"]["page_size"] * sz["kv_per_token"] // sz["n_attn"])   # gathered K, V
    st = sz["state_per_seq"]
    cap = {"free_gpu_bytes": free, "weights_bytes": sz["weights"], "state_bytes": st, "metadata_share": share,
           "per_seq_extras_bytes": extras, "by_context": {}}
    for n in h4["contexts"]:
        kv = n * sz["kv_per_token"]
        d = {"kv_bytes": kv, "all_on_gpu": {"per_seq_bytes": kv + share * kv + st + extras,
                                            "max_batch": int(free // (kv + share * kv + st + extras))}}
        for name, p in pol.items():
            per = st + share * kv + extras + n_attn * page * (sz["np_pinned"] + p["peak_resident_max"])
            d[name] = {"per_seq_bytes": per, "max_batch_gpu": int(free // per),
                       "max_batch_host": int(rc["host_kv_gib"] * 2 ** 30 // kv),
                       "max_batch": int(min(free // per, rc["host_kv_gib"] * 2 ** 30 // kv))}
        cap["by_context"][str(n)] = d
    tol = h4["gate"]["calibration_tol"]
    calib_ok = bool(calib) and all(abs(c["ratio"] - 1) <= tol for c in calib)
    # H4 stretch gate: tiered throughput >= h4.gate.min_ratio x all-on-GPU wherever both fit, and a larger
    # max batch at the longest context; throughput at each mode's capacity (largest timed batch within it)
    nmax = max(h4["contexts"])
    rows_n = [m for m in model if m["context"] == nmax]
    cn = cap["by_context"][str(nmax)]
    at_cap, gate = {}, {}
    fit = [m for m in rows_n if m["batch"] <= cn["all_on_gpu"]["max_batch"]]
    if fit:
        g = max(fit, key=lambda m: m["batch"])
        at_cap["all_on_gpu"] = {"batch": g["batch"], "tokens_per_s": 1e3 * g["batch"] / g["step_gpu_ms"]}
    for name, p in pol.items():
        for link in links:
            within = [m for m in rows_n if m["batch"] <= cn[name]["max_batch"] and name in m["policies"]]
            if within:
                g = max(within, key=lambda m: m["batch"])
                at_cap[f"{name} | {link}"] = {"batch": g["batch"], "tokens_per_s": g["policies"][name][link]["tokens_per_s"]}
            both = [m["policies"][name][link]["throughput_ratio"] for m in model
                    if name in m["policies"] and m["batch"] <= cap["by_context"][str(m["context"])]["all_on_gpu"]["max_batch"]]
            if both:
                ext = max(m["policies"][name][link]["extrapolated_share"] for m in model if name in m["policies"])
                gate[f"{name} | {link}"] = {"min_ratio_where_both_fit": min(both), "ratio_ok": min(both) >= h4["gate"]["min_ratio"],
                                            "larger_max_batch": cn[name]["max_batch"] > cn["all_on_gpu"]["max_batch"],
                                            "changes_selection": p["need"] == "lag1", "max_extrapolated_share": ext,
                                            "valid": bool(check["complete"] and ext == 0.0 and calib_ok)}
    res = {"provenance": provenance(cfg, model=cfg["hybrid"]["models"][mkey]["name"], context_length=None, residency=rc, h4=h4),
           "costs_file": {"path": str(cpath), "sha256": hashlib.sha256(cpath.read_bytes()).hexdigest(),
                          "provenance": {k: costs["provenance"].get(k) for k in ("git", "timestamp", "gpu", "gpu_name", "host", "h2_chosen")}},
           "check": check, "calibration_ok": calib_ok, "links_dropped": dropped, "steps_beyond_last_bucket": beyond,
           "n_sequences": len(rows), "mean_gen_tokens": float(np.mean([r["n_gen"] for r in rows])),
           "missing": miss, "policies": pol, "cross_check": cross, "links": list(links),
           "step_model": model, "calibration": calib, "capacity": cap, "at_capacity": at_cap, "gate_h4": gate,
           "notes": ["step_gpu = plan + GDN + attention (H3's ours engine path) + head, from per-layer graph timings",
                     "staged = measured wall-clock round trip incl. index readback and syncs; zero-copy = measured "
                     "in-graph GPU gather from pinned host memory",
                     "prefetch / lag1 overlap: copy hidden behind compute up to the window; the copy and the GDN layers "
                     "each slowed by their measured factors when run together (lag1: the factor applied to the whole step)",
                     "staged: one graph launch + sync per attention layer added (the step graph is split for the host sync)",
                     "new KV written back GPU -> host each step (writeback_ms, the other PCIe direction) is not added",
                     "steps beyond the longest context are priced in the last bucket (steps_beyond_last_bucket)",
                     "gate_h4 valid only with all sequences, no extrapolated fetch sizes and calibration within "
                     "h4.gate.calibration_tol"]}
    print(f"{len(rows)} sequences ({check}), mean {res['mean_gen_tokens']:.0f} generated tokens; missing at "
          f"{h4['budget_tokens']}: exact {100 * miss['exact']:.2f}%, lag1 {100 * miss['lag1']:.2f}%")
    for n, p in pol.items():
        print(f"{n:16s} sync {p['sync_mean']:.2f} prefetch {p['prefetch_mean']:.2f} pages/step/layer, peak resident {p['peak_resident_max']}"
              + (f" | cross {cross['sync_mean'][n]:.2f}" if cross else ""))
    for m in model:
        print(f"B {m['batch']:3d} n<={m['context']:6d}: step {m['step_gpu_ms']:.2f} ms (window {m['gdn_window_ms']:.2f}) | " +
              "  ".join(f"{n} " + "/".join(f"{v[k]['throughput_ratio']:.2f}" for k in v) for n, v in m["policies"].items()))
    for c in calib:
        print("calibration", c)
    print(json.dumps(cap["by_context"][str(nmax)], indent=1))
    print("at capacity:", json.dumps(at_cap, indent=1))
    print("gate_h4:", json.dumps(gate, indent=1))
    write_json(Path(cfg["paths"]["results_dir"]) / f"h4_residency_{mkey}.json", res)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("mode", choices=["screen", "merge"])
    p.add_argument("--limit", type=int, default=None, help="first n sequences per set (dry runs)")
    p.add_argument("--max-gen", type=int, default=None, help="dry run: cut generated tokens, write nothing")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = thinking(config_from_args(args))
    H1.register()
    {"screen": screen, "merge": lambda c, a: merge(c)}[args.mode](cfg, args)


if __name__ == "__main__":
    main()
