"""Validate query recording, per-page key summaries and multi-turn (fed) decoding.

  python scripts/check_multiturn_q.py configs/smoke_2k.yaml --device cuda:0

Checks: (1) attention recomputed from the stored fp16 query and the final key cache matches
the recorded page mass at every step, generated and fed; (2) key summaries equal the page
mean/min/max of the cache; (3) fed tokens are exactly the follow-up ids, turns are found;
(4) every recorded row sums to 1.
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, followup_turns, load_model, model_dims, stop_token_ids
from kvfetch.provenance import provenance, write_json
from kvfetch.tasks.ruler import RulerNiah
from kvfetch.tracing.decode import FED, GEN, generate
from kvfetch.tracing.tracer import AttentionTracer, attention_probs, page_reduce


def topb_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.05) -> float:
    B = max(1, int(np.ceil(frac * a.shape[-1])))
    ia = np.argpartition(-a, B - 1, -1)[..., :B]
    ib = np.argpartition(-b, B - 1, -1)[..., :B]
    return float(np.mean([len(set(x) & set(y)) / B for x, y in zip(ia.reshape(-1, B), ib.reshape(-1, B))]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = config_from_args(args)
    sc = cfg["smoke_multiturn"]
    cfg = copy.deepcopy(cfg)
    cfg["trace"]["record_q"] = cfg["trace"]["key_summary"] = True

    model, tok = load_model(sc["model"], cfg, args.device, traced=True)
    dims = model_dims(model)
    stops = stop_token_ids(sc["model"], tok, cfg)
    sample = RulerNiah(sc["task"], cfg).samples(tok, max_seq_length=sc["context_length"], num_samples=1, seed=0)[0]
    ids = torch.tensor(encode(tok, sample.prompt)["input_ids"], device=args.device)
    first_user = sample.prompt  # only used for template rendering of later turns
    followups, eot = followup_turns(tok, first_user, sc["followups"], cfg)
    layers = list(range(dims.num_layers))
    tr = AttentionTracer(layers=layers, num_heads=dims.num_heads, num_kv_heads=dims.num_kv_heads,
                         page_size=cfg["definitions"]["page_size"], prompt_len=ids.numel(),
                         decode_steps=sc["max_new_tokens"], obs_window=cfg["trace"]["prefill_obs_window"],
                         flush_every=cfg["trace"]["flush_every"], device=args.device,
                         head_dim=dims.head_dim, record_q=True)
    with tr:
        res, cache = generate(model, ids, max_new_tokens=sc["max_new_tokens"], stop_ids=stops, suppress_stop=False,
                              prefill_chunk_size=cfg["models"]["prefill_chunk_size"], tracer=tr,
                              followups=followups, eot_ids=eot, turn_max_new=sc["turn_max_new"], return_cache=True)
    tr.finalize_key_summary(cache)
    T = len(res.tokens)
    n = ids.numel()

    # (1) recompute every step from the stored query
    scaling = model.model.layers[0].self_attn.scaling
    worst, overlaps = 0.0, []
    for slot, layer in enumerate(layers):
        k = cache.layers[layer].keys[0]
        rec = np.stack([page_reduce(attention_probs(torch.tensor(tr.q[s, slot], device=k.device)[:, None, :].float(),
                                                    k[:, :n + s], scaling)[:, 0], tr.page_size, tr.n_pages).cpu().numpy()
                        for s in range(T)])                                           # [T, H, P]
        got = tr.mass[:T, slot].astype(np.float32)
        worst = max(worst, float(np.abs(rec - got).max()))
        overlaps.append(topb_overlap(rec, got))
    # (2) key summaries vs the cache
    ks_err, ks_mag = 0.0, 0.0
    for slot, layer in enumerate(layers):
        k = cache.layers[layer].keys[0].float()
        full = (k.shape[1] // tr.page_size) * tr.page_size
        pk = k[:, :full].reshape(k.shape[0], -1, tr.page_size, k.shape[2])
        ref = torch.stack([pk.mean(2), pk.amin(2), pk.amax(2)]).cpu().numpy()          # [3, KVH, Pfull, d]
        got = tr.key_summary[slot][:, :, :ref.shape[2]].astype(np.float32)
        # fp16 storage: error relative to magnitude (Qwen3 post-RoPE keys reach the hundreds)
        ks_err = max(ks_err, float((np.abs(got - ref) / (sc["key_summary_atol"] + sc["key_summary_rtol"] * np.abs(ref))).max()))
        ks_mag = max(ks_mag, float(np.abs(ref).max()))
    # (3) turns and fed tokens
    fed = [t for t, kd in zip(res.tokens, res.kinds) if kd == FED]
    expected_fed = []
    for i, t in enumerate(res.turns[:-1]):
        expected_fed += ([] if t["natural_stop"] is not None else eot) + followups[i]
    sums = tr.mass[:T].astype(np.float32).sum(-1)

    report = {
        "q_recompute_max_abs_diff": worst, "q_recompute_topb_overlap_min": min(overlaps),
        "key_summary_err_over_bound": ks_err, "key_summary_max_abs_key": ks_mag,
        "n_turns": len(res.turns), "turns": res.turns, "n_fed": len(fed), "fed_tokens_match": fed == expected_fed,
        "max_row_sum_err": float(np.abs(sums - 1).max()),
        "turn_texts": [tok.decode(res.turn_tokens(i)) for i in range(len(res.turns))],
        "answers": sample.answers,
    }
    report["pass"] = bool(worst <= sc["q_mass_atol"] and min(overlaps) >= sc["q_min_topb_overlap"]
                          and ks_err <= 1.0 and report["fed_tokens_match"]
                          and len(res.turns) == len(sc["followups"]) + 1 and report["max_row_sum_err"] < 5e-3)
    # (5) head_reduce=kv_mean stores the GQA-group mean of exactly these masses
    tr2 = AttentionTracer(layers=layers, num_heads=dims.num_heads, num_kv_heads=dims.num_kv_heads,
                          page_size=cfg["definitions"]["page_size"], prompt_len=n, decode_steps=sc["max_new_tokens"],
                          obs_window=cfg["trace"]["prefill_obs_window"], flush_every=cfg["trace"]["flush_every"],
                          device=args.device, head_reduce="kv_mean")
    with tr2:
        res2 = generate(model, ids, max_new_tokens=sc["max_new_tokens"], stop_ids=stops, suppress_stop=False,
                        prefill_chunk_size=cfg["models"]["prefill_chunk_size"], tracer=tr2,
                        followups=followups, eot_ids=eot, turn_max_new=sc["turn_max_new"])
    g = dims.num_heads // dims.num_kv_heads
    full = tr.mass[:T].astype(np.float32)
    want = full.reshape(T, len(layers), dims.num_kv_heads, g, -1).mean(3)
    report["kv_mean_same_tokens"] = res2.tokens == res.tokens
    report["kv_mean_max_abs_diff"] = float(np.abs(tr2.mass[:T].astype(np.float32) - want).max())
    report["pass"] = bool(report["pass"] and report["kv_mean_same_tokens"] and report["kv_mean_max_abs_diff"] < 1e-3)
    write_json(sc["out"], {"provenance": provenance(cfg, model=sc["model"], context_length=n), **report})
    print({k: v for k, v in report.items() if k not in ("turn_texts", "turns")})
    for i, t in enumerate(report["turn_texts"]):
        print(f"--- turn {i}: {t[:160]!r}")


if __name__ == "__main__":
    main()
