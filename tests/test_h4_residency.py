"""scripts/h4_residency: the vectorised LRU page cache against a plain Python LRU, the lag-1 selection's
eligibility and budget, analyze_layer end to end and merge on synthetic shards (CPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import h4_residency as R
from kvfetch.config import load_config

CFG = Path(__file__).resolve().parent.parent / "configs" / "hybrid.yaml"


def _reference(need, pre, aged, cap):
    T, P = need.shape
    last = {}                                             # page -> use order (higher = more recent)
    sync, traffic, peak = [], [], 0
    for t in range(T):
        for p in range(P):
            if aged[t, p]:
                last[p] = 3 * t
        traffic.append(sum(1 for p in range(P) if pre[t, p] and p not in last))
        for p in range(P):
            if pre[t, p]:
                last[p] = 3 * t + 1
        sync.append(sum(1 for p in range(P) if need[t, p] and p not in last))
        for p in range(P):
            if need[t, p]:
                last[p] = 3 * t + 2
        peak = max(peak, len(last) - (int(aged[0].sum()) if t == 0 else 0))
        excess = len(last) - cap
        if excess > 0:
            for p in sorted(last, key=lambda p: (last[p], p))[:excess]:
                del last[p]
    return sync, traffic, peak


def test_simulate_matches_reference():
    g = torch.Generator().manual_seed(0)
    T, K, P = 60, 3, 24
    need = torch.rand((T, K, P), generator=g) < 0.2
    pre = torch.rand((T, K, P), generator=g) < 0.15
    pre[:, 0] = False                                     # a policy without prefetch
    aged = torch.rand((T, P), generator=g) < 0.05
    aged[0] = torch.rand(P, generator=g) < 0.5            # prompt pages resident after prefill
    cap = torch.tensor([3, 7, 11])
    sync, traffic, peak = R.simulate(need, pre, aged, cap)
    for k in range(K):
        s, tr, pk = _reference(need[:, k], pre[:, k], aged, int(cap[k]))
        assert sync[:, k].tolist() == s
        assert traffic[:, k].tolist() == tr
        assert int(peak[k]) == pk


def test_lag1_uses_current_eligibility_and_budget():
    """A page that ages out at step t can be chosen by the stale ranking at t, and no step goes over budget."""
    T, P = 6, 8
    elig = torch.zeros((T, P), dtype=torch.bool)
    elig[:, 1:4] = True
    elig[3:, 4] = True                                    # page 4 ages out of the recent window at step 3
    b = torch.tensor([2, 2, 2, 3, 3, 3])
    score = torch.zeros((T, 1, P))
    score[:, 0, 4] = 5.0                                  # page 4 is the most attended once eligible
    score[:, 0, 1] = 1.0
    pols = [{"name": "lag1 c1", "need": "lag1", "cache": 1, "pre": 0}, {"name": "la c2+k1", "need": "exact", "cache": 2, "pre": 1}]
    sel, lag, pre = R.selections(score, score, elig, b, pols)
    assert bool(lag[3, 4]) and bool(sel[3, 4])
    assert (lag.sum(-1) <= b).all() and (lag <= elig).all()
    assert torch.equal(lag[0], sel[0])
    assert (pre[1].sum(-1) <= b).all()


def _cfg(tmp: Path) -> dict:
    cfg = load_config(CFG)
    cfg["paths"]["results_dir"] = str(tmp)
    cfg["hybrid"]["h4"]["residency"]["shard_dir"] = str(tmp / "shards")
    cfg["hybrid"]["h4"]["residency"]["mc_samples"] = 16
    return cfg


def test_analyze_layer_runs_on_cpu():
    cfg = load_config(CFG)
    g = torch.Generator().manual_seed(1)
    pl, T, H, S, d = 40, 300, 16, 4, 32
    k = torch.randn((S, pl + T - 1, d), generator=g)
    q = torch.randn((T, H, d), generator=g).to(torch.bfloat16)
    c = {"q": q, "k": k, "la": {3: (q.float() + 0.3 * torch.randn((T, H, d), generator=g)).to(torch.bfloat16)}, "q_self_err": 0.0}
    out, arr = R.analyze_layer(c, pl, cfg, "cpu")
    names = [p["name"] for p in R.policies(cfg["hybrid"]["h4"]["residency"])]
    assert list(out["policies"]) == names
    assert arr["sync"].shape == (T, len(names)) and arr["traffic"].shape == (T, len(names))
    assert 0.0 <= out["missing"]["exact"] <= 1.0 and 0.0 <= out["missing"]["lag1"] <= 1.0
    for n, p in out["policies"].items():
        assert p["peak_resident"] >= p["cache_pages"] or p["sync_mean"] == 0.0
        if p["pre"]:
            assert p["sync_mean"] <= out["policies"][f"exact c{p['cache']:g}"]["sync_mean"] + 1e-9   # prefetch only helps


def test_merge_on_synthetic_shards(tmp_path):
    cfg = _cfg(tmp_path)
    h4 = cfg["hybrid"]["h4"]
    rc = h4["residency"]
    names = [p["name"] for p in R.policies(rc)]
    layers = cfg["hybrid"]["models"][rc["model"]]["layers"]
    sizes = {"weights": 9 * 2 ** 30, "state_per_seq": 50 * 2 ** 20, "kv_per_token": h4["kv_bytes_per_token"], "n_layers": 32,
             "n_attn": 8, "head_dim": 256, "np_pinned": 10, "vocab": 248320, "gdn_before_attn": [3] * 8}
    rng = np.random.default_rng(0)
    root = Path(rc["shard_dir"]) / "main"
    exp = []
    for i in range(2):
        T, pl = 20000, 100
        prov = {"h2_chosen": "mean8", "budget_tokens": 512, "git": {"commit": "x"}}
        lay = {str(l): {"n_steps": T, "missing": {"exact": 0.2, "lag1": 0.25},
                        "policies": {n: {"peak_resident": 60, "need": "lag1" if n.startswith("lag1") else "exact",
                                         "cache": 1, "pre": 1 if n.startswith("la3") else 0} for n in names}} for l in layers}
        (root / "aime24").mkdir(parents=True, exist_ok=True)
        (root / "aime24" / f"{i:04d}.json").write_text(json.dumps({"provenance": prov, "sizes": sizes, "task": "aime24", "index": i,
                                                                 "n_gen": T, "layers": lay}))
        np.savez_compressed(root / "aime24" / f"{i:04d}.npz", context=pl + np.arange(T, dtype=np.int32),
                            **{f"{k}_{l}": rng.integers(0, 6, (T, len(names))).astype(np.int16) for k in ("sync", "traffic") for l in layers})
        exp.append(f"aime24/{i:04d}")
    (root / "manifest.json").write_text(json.dumps({"expected": exp}))
    ms = {str(k): 0.01 + 0.006 * k for k in h4["fetch_pages"]}
    lay = [{"batch": B, "context": n, "method": m, "attention_layer_ms": 0.5, **({"gdn_layer_ms": 0.4} if m == "full" else {})}
           for B in h4["batches"] for n in h4["contexts"] for m in ("full", "ours@512", "quest@512")]
    costs = {"provenance": {"git": {"commit": "x"}}, "layers": lay,
             "plan_ms": {str(B): {str(n): 0.05 for n in h4["contexts"]} for B in h4["batches"]},
             "head_ms": {str(B): 1.0 for B in h4["batches"]}, "steps": [],
             "lookahead": [{"batch": B, "context": n, "multiple": m, "ms": 0.1} for B in h4["batches"] for n in h4["contexts"] for m in (1, 2)],
             "copy": {"staged_round_trip_ms": {k: v + 0.1 for k, v in ms.items()}, "zero_copy_ms": ms,
                      "overlap": [{"batch": 8, "gdn_alone_ms": 0.4, "gdn_with_contiguous copy_ms": 0.42, "gdn_with_zero-copy gather_ms": 0.5}]}}
    (tmp_path / "h4_costs_qwen35_4b.json").write_text(json.dumps(costs))
    R.merge(cfg)
    res = json.loads((tmp_path / "h4_residency_qwen35_4b.json").read_text())
    assert res["check"]["complete"] and res["check"]["consistent"]
    rows = [m for m in res["step_model"] if m["context"] == 32768]
    assert rows and all(0 < v[k]["throughput_ratio"] <= 1 for m in rows for v in m["policies"].values() for k in v)
    assert res["capacity"]["by_context"]["32768"]["all_on_gpu"]["max_batch"] > 0
