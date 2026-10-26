"""The engine's H2 summary formats (kvfetch/sparse_decode.py) score pages as scripts/h1_screen.py's
offline versions do, given the same keys and query."""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from kvfetch import sparse_decode as SD
from kvfetch.baselines import landmark_scores, quest_scores, unit_summaries
import h1_screen as H


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _engine(spec, S=2, H_=4, d=16, n=96, ps=16):
    cfg = types.SimpleNamespace(num_hidden_layers=1, num_key_value_heads=S, num_attention_heads=H_, head_dim=d, hidden_size=32)
    model = types.SimpleNamespace(config=cfg, parameters=lambda: iter([torch.zeros(1, dtype=torch.bfloat16)]))
    eng = SD.SparseEngine(model, SD.Method("ours", 64, spec=spec), batch=1, max_len=n + 1, page_size=ps, sink_pages=1,
                          recent_tokens=32, obs_window=4, device=DEV)
    torch.manual_seed(0)
    k = torch.randn((1, S, n, d), dtype=torch.bfloat16, device=DEV) * 2
    q = torch.randn((1, H_, n, d), dtype=torch.bfloat16, device=DEV)
    eng.begin_prefill(0)
    eng.update(k, k, 0)
    eng._prefill(0, q, k, k, d ** -0.5)
    eng.end_prefill(n)
    return eng, q[0, :, -1].float(), k[0].float(), q[0].transpose(0, 1).float()


@pytest.mark.skipif(DEV == "cpu", reason="bmm with out_dtype needs CUDA")
def test_formats_match_offline():
    ps = 16
    for bf, mf in [("int4", "fp32"), ("int4p2", "chan"), ("none", "fp8"), ("none", "fp16"), ("none", "chan")]:
        eng, q, k, qall = _engine({"fuse": "sum", "history_weight": 0.0, "history": "last", "mean_mix": 0.5,
                                   "bound_fmt": bf, "mean_fmt": mf})
        S, n, d = k.shape
        P = n // ps
        qg = q.view(1, S, -1, d).to(torch.bfloat16)
        sc = d ** -0.5
        summ = unit_summaries(k, ps)
        if bf != "none":
            on = eng._bound(0, qg, sc)[0, :, :P]
            if bf == "int4":
                off = quest_scores(q[None], H.q4_bounds(summ), sc)[0]
            else:
                off = quest_scores(q[None], H.q4_bounds(unit_summaries(k, 2 * ps)), sc)[0].repeat_interleave(2, -1)[:, :P]
            assert torch.allclose(on, off, atol=2e-2, rtol=2e-2), (bf, (on - off).abs().max())
        on = eng._mean_logit(0, qg, sc)[0, :, :P]
        if mf == "chan":
            imp = qall[-1].abs().view(S, -1, d).mean(1) * k.abs().mean(1)
            ch = torch.topk(imp, d // 4, dim=-1).indices
            off = H.partial_mean_scores(q[None], summ["mean"], ch, sc)[0]
        else:
            m = summ["mean"].half().float() if mf == "fp16" else summ["mean"].to(torch.float8_e4m3fn).float()
            off = landmark_scores(q[None], {"mean": m}, sc)[0]
        assert torch.allclose(on, off, atol=2e-2, rtol=2e-2), (mf, (on - off).abs().max())
