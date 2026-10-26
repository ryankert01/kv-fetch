"""Run the online SparseEngine step by step on a synthetic sequence and compare its page sets with
h1_screen's offline replicas (snapkv_evict, quest|vote, ours|sum, oracle|sum, H2 formats).
Written by the H1 audit agent (2026-10-26); every method must differ on 0 steps.

  CUDA_VISIBLE_DEVICES=4 .venv-hybrid/bin/python scripts/check_h1_replicas.py [prompt_len ...]
"""
import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from kvfetch import sparse_decode as SD
from kvfetch import selectors_gpu as G
from kvfetch.baselines import attention_by_group, unit_summaries, quest_scores, landmark_scores
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.sim import existing_pages
from compare_pages import fit_pages, Scorer
import h1_screen as H

DEV = "cuda"
S, Hq, d = 2, 4, 16
ps, R, sink = 4, 8, 1


def make_seq(pl, T, seed, head_skew=True):
    g = torch.Generator().manual_seed(seed)
    N = pl + T - 1
    k = (torch.randn((1, S, N, d), generator=g) * 1.5)
    q = torch.randn((1, Hq, N, d), generator=g)
    if head_skew:   # heads within a group look at different pages
        q[:, 1::2] *= 3.0
    v = torch.randn((1, S, N, d), generator=g)
    return q.to(torch.bfloat16).to(DEV), k.to(torch.bfloat16).to(DEV), v.to(torch.bfloat16).to(DEV)


def run_engine(method, q, k, v, pl, T):
    N = pl + T - 1
    cfg = types.SimpleNamespace(num_hidden_layers=1, num_key_value_heads=S, num_attention_heads=Hq, head_dim=d, hidden_size=32)
    model = types.SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1, dtype=torch.bfloat16)]))
    eng = SD.SparseEngine(model, method, batch=1, max_len=N + 1, page_size=ps, sink_pages=sink, recent_tokens=R,
                          obs_window=method.window, device=DEV)
    eng.record = {0: []}
    sc = d ** -0.5
    eng.begin_prefill(0)
    eng.update(k[:, :, :pl], v[:, :, :pl], 0)
    eng._prefill(0, q[:, :, :pl], k[:, :, :pl], v[:, :, :pl], sc)
    eng.end_prefill(pl)
    for t in range(1, T):
        pos = pl - 1 + t
        eng.begin_step(eng.Lmax)
        K, V = eng.update(k[:, :, pos:pos + 1], v[:, :, pos:pos + 1], 0)
        eng.attend(0, q[:, :, pos:pos + 1], K, V, sc)
        eng.end_step(torch.ones(1, dtype=torch.bool, device=DEV))
    return torch.stack([r[0] for r in eng.record[0]])        # [T-1, P]


def offline(pl, T, q, k, btok):
    N = pl + T - 1
    P = -(-N // ps)
    pinned_def = {"sink_pages": sink, "decode_region": "aged", "recent_tokens": R}
    elig_np = eligibility_matrix(pl, T, P, ps, pinned_def)
    exist_np = existing_pages(pl, T, P, ps)
    elig, exist = torch.from_numpy(elig_np).to(DEV), torch.from_numpy(exist_np).to(DEV)
    b = H.budgets_for([btok], ps, exist_np, elig_np, DEV)[str(btok)]
    qd = q[0, :, pl - 1:].transpose(0, 1).contiguous()          # [T, H, d] as captured
    kk = k[0].float()
    summ = unit_summaries(kk, ps)
    sc = H.page_scores(qd, kk, pl, P, ps, 64, summ)
    return dict(P=P, elig=elig, exist=exist, b=b, sc=sc, qd=qd, kk=kk, summ=summ)


def compare(name, on, off):
    P = off.shape[1]
    assert not on[:, P:].any()
    on = on[:, :P]
    T1 = on.shape[0]
    off = off[1:].cpu()
    diff = (on != off).any(-1)
    print(f"{name:10s}: steps differing {int(diff.sum())}/{T1}; first at t={int(diff.nonzero()[0]) + 1 if diff.any() else None}")
    return diff


if __name__ == "__main__":
  for pl in [int(x) for x in sys.argv[1:]] or [41, 120, 123]:
    T, btok = 300, 48
    print(f"### prompt_len {pl}")
    for seed in range(3):
        print(f"--- seed {seed}")
        q, k, v = make_seq(pl, T, seed)
        o = offline(pl, T, q, k, btok)
        m, elig, exist, b, P = o["sc"]["m"], o["elig"], o["exist"], o["b"], o["P"]
        # snapkv
        for W in (4, 32):
            on = run_engine(SD.Method("snapkv", btok, window=W, protect_prompt=True), q, k, v, pl, T)
            off = H.snapkv_evict(m, o["sc"]["mh"], elig, exist, b, W, -(-pl // ps), True)
            dsn = compare(f"snapkv W{W}", on, off)
            score = Scorer(m, torch.zeros_like(m), (exist & ~elig)[:, None, :])
            on_full = torch.cat([off[:1].cpu(), on[:, :P]]).to(DEV)
            mo = score.whole(on_full[:, None, :].expand(T, S, P) & elig[:, None, :], b, False)["missing"]
            mf = score.whole(off[:, None, :].expand(T, S, P) & elig[:, None, :], b, False)["missing"]
            print(f"   missing online-sets {100*mo:.3f}  offline-sets {100*mf:.3f}")
        # quest vote
        on = run_engine(SD.Method("quest", btok), q, k, v, pl, T)
        off = G.select_from_ranks(G.static_ranks(o["sc"]["quest"], elig, "logit"), elig, b, "vote")[:, 0]
        compare("quest", on, off)
        # ours sum (H3 spec)
        spec = {"fuse": "sum", "history_weight": 0.0, "history": "last", "mean_mix": 0.5}
        on = run_engine(SD.Method("ours", btok, spec=spec), q, k, v, pl, T)
        mm = 0.5
        ours = (1 - mm) * o["sc"]["quest"] + mm * o["sc"]["mean"]
        off = G.select_from_ranks(G.static_ranks(ours, elig, "logit"), elig, b, "sum")[:, 0]
        compare("ours", on, off)
        # oracle
        on = run_engine(SD.Method("oracle", btok), q, k, v, pl, T)
        off = G.mask_from_ranks(G.rank_desc(m.sum(1, keepdim=True), elig), b, elig)[:, 0]
        compare("oracle", on, off)
        # H2 formats
        h2 = H.h2_scores(o["qd"], o["kk"], pl, P, ps, 64, o["summ"])
        for nm, es, off_s, mode in [
            ("quest4", {"bound_fmt": "int4"}, h2["quest4"], "vote"),
            ("quest4p2", {"bound_fmt": "int4p2"}, h2["quest4p2"], "vote"),
            ("mean8", {"bound_fmt": "none", "mean_fmt": "fp8"}, h2["mean8"], "sum"),
            ("meanc", {"bound_fmt": "none", "mean_fmt": "chan"}, h2["meanc"], "sum"),
            ("lite", {"bound_fmt": "int4p2", "mean_fmt": "chan"}, (1 - mm) * h2["quest4p2"] + mm * h2["meanc"], "sum"),
        ]:
            meth = SD.Method("quest", btok, spec=es) if nm.startswith("quest") else SD.Method("ours", btok, spec={**spec, **es})
            on = run_engine(meth, q, k, v, pl, T)
            off = G.select_from_ranks(G.static_ranks(off_s, elig, "logit"), elig, b, mode)[:, 0]
            compare(nm, on, off)
