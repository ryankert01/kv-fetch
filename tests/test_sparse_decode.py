import types

import numpy as np
import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from kvfetch import sparse_decode as SD
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.sim import existing_pages

PINNED = {"sink_pages": 1, "decode_region": "aged", "recent_tokens": 128}


def _tiny_model():
    cfg = types.SimpleNamespace(num_hidden_layers=1, num_key_value_heads=2, num_attention_heads=4, head_dim=8,
                                hidden_size=32)
    return types.SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1, dtype=torch.bfloat16)]))


def test_step_plan_matches_offline_eligibility():
    ps, btok, max_len = 16, 1024, 3000
    eng = SD.SparseEngine(_tiny_model(), SD.Method("quest", btok), batch=1, max_len=max_len, page_size=ps,
                          sink_pages=PINNED["sink_pages"], recent_tokens=PINNED["recent_tokens"], obs_window=32,
                          device="cpu")
    P = eng.P
    elig = eligibility_matrix(1, max_len - 1, P, ps, PINNED)          # row t: keys [0, 1 + t) visible
    exist = existing_pages(1, max_len - 1, P, ps)
    for n in list(range(1, 400)) + [1000, 1007, 1008, 1009, 2047, 2999]:
        eng.lens[:] = n - 1
        eng.begin_step(eng.Lmax)
        pl = eng.plan
        t = n - 1
        assert np.array_equal(pl["elig"][0, :P].numpy(), elig[t]), n
        pinned = np.zeros(P + 1, bool)
        pinned[pl["pinned"][0].numpy()] = True
        assert np.array_equal(pinned[:P], exist[t] & ~elig[t]), n
        b = max(0, btok // ps - int((exist[t] & ~elig[t]).sum()))
        assert int(pl["b"][0]) == b and int(pl["k_sel"][0]) == min(b, int(elig[t].sum())), n


def test_sample_token_matches_hf_warpers():
    torch.manual_seed(0)
    logits = torch.randn(3, 500) * 4
    warp = LogitsProcessorList([TemperatureLogitsWarper(0.6), TopKLogitsWarper(20), TopPLogitsWarper(0.95)])
    want = torch.softmax(warp(None, logits.clone()), -1)
    n = 20000
    u = (torch.arange(n, dtype=torch.float32) + 0.5) / n                     # inverse CDF on a uniform grid
    for r in range(3):
        toks = SD.sample_token(logits[r].expand(n, -1), u, temperature=0.6, top_k=20, top_p=0.95)
        freq = torch.bincount(toks, minlength=500).float() / n
        assert float((freq - want[r]).abs().max()) < 2e-4


def test_sample_token_greedy():
    logits = torch.tensor([[0.0, 3.0, 1.0], [5.0, 0.0, 1.0]])
    assert SD.sample_token(logits, torch.zeros(2), temperature=0, top_k=1, top_p=1.0).tolist() == [1, 0]
