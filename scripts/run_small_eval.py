"""Small eval (spec §5): probe -> trace -> analyze, end to end on one GPU, timed.

  python scripts/run_small_eval.py configs/small_eval.yaml --device cuda:0 \
      [--set small_eval.model=Qwen/Qwen3-4B --set small_eval.context_length=32768]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from kvfetch.analysis import analyze_trace, pool_results
from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, load_model, model_dims, stop_token_ids
from kvfetch.provenance import provenance, write_json
from kvfetch.tasks.longgen import LongGen
from kvfetch.tasks.ruler import RulerNiah, char_spans_to_token_spans, find_spans, string_match_all
from kvfetch.tracing.capture import capture, trace_meta
from kvfetch.tracing.schema import load_trace, save_trace


def needle_pages(tok, sample, page_size: int) -> list[int]:
    enc = encode(tok, sample.prompt, offsets=True)
    spans = char_spans_to_token_spans(enc["offset_mapping"], find_spans(sample.prompt, sample.query_needles))
    return sorted({p for a, b in spans for p in range(a // page_size, (b - 1) // page_size + 1)})


def probe_retrieval_layer(model, tok, cfg: dict, se: dict, device, stops) -> dict:
    """Score every (layer, head) by attention mass on the queried needle's pages while
    the answer is emitted; the layer with the highest single-head score wins."""
    pc = se["layers"]["probe"]
    dims = model_dims(model)
    ps = cfg["definitions"]["page_size"]
    samples = RulerNiah(pc["ruler_task"], cfg).samples(tok, max_seq_length=se["context_length"],
                                                       num_samples=pc["samples"], seed=pc["seed"])
    score = np.zeros((dims.num_layers, dims.num_heads))
    for s in samples:
        ids = torch.tensor(encode(tok, s.prompt)["input_ids"], device=device)
        _, tr = capture(model, ids, layers=list(range(dims.num_layers)), cfg=cfg, decode_steps=pc["steps"],
                        stop_ids=stops, suppress_stop=True)
        score += tr.mass[:, :, :, needle_pages(tok, s, ps)].astype(np.float32).sum(-1).mean(0)
    score /= len(samples)
    layer_score = score.max(1)
    top = np.dstack(np.unravel_index(np.argsort(-score, axis=None)[:8], score.shape))[0]
    return {"layer": int(layer_score.argmax()), "layer_scores": layer_score.round(4).tolist(),
            "top_heads": [{"layer": int(l), "head": int(h), "score": float(score[l, h])} for l, h in top]}


def select_layers(num_layers: int, lc: dict, probe: dict | None) -> list[int]:
    layers = {int(round(f * (num_layers - 1))) for f in lc["fracs"]}
    if probe is not None:
        layers.add(probe["layer"])
    return sorted(layers)


def build_samples(task: dict, tok, cfg: dict, se: dict):
    if task["kind"] == "ruler":
        return RulerNiah(task["ruler_task"], cfg).samples(
            tok, max_seq_length=se["context_length"], num_samples=se["samples_per_task"], seed=se["seed"],
            tokens_to_generate=se["decode_steps"])
    if task["kind"] == "longgen":
        return LongGen(cfg).samples(tok, max_seq_length=se["context_length"], decode_steps=se["decode_steps"],
                                    num_samples=se["samples_per_task"])
    raise ValueError(task["kind"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = config_from_args(args)
    se = cfg["small_eval"]
    ps = cfg["definitions"]["page_size"]
    tag = f"{se['model'].replace('/', '__')}__{se['context_length']}"
    trace_dir = Path(se["trace_dir"]) / tag
    timing: dict = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    model, tok = load_model(se["model"], cfg, args.device, traced=True)
    dims = model_dims(model)
    stops = stop_token_ids(se["model"], tok, cfg)
    timing["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probe = probe_retrieval_layer(model, tok, cfg, se, args.device, stops) if se["layers"]["mode"] == "auto" else None
    layers = select_layers(dims.num_layers, se["layers"], probe)
    timing["probe_s"] = time.perf_counter() - t0
    print(f"[small_eval] {se['model']}: traced layers {layers} (retrieval probe -> {probe and probe['layer']})", flush=True)

    prov = provenance(cfg, model=se["model"], context_length=se["context_length"], decode_steps=se["decode_steps"],
                      traced_layers=layers, trace_format=cfg["trace"]["format"])
    task_out: dict = {}
    for task in se["tasks"]:
        t0 = time.perf_counter()
        samples = build_samples(task, tok, cfg, se)
        t_build = time.perf_counter() - t0
        recs, paths = [], []
        t0 = time.perf_counter()
        for s in samples:
            ids = torch.tensor(encode(tok, s.prompt)["input_ids"], device=args.device)
            res, tr = capture(model, ids, layers=layers, cfg=cfg, decode_steps=se["decode_steps"],
                              stop_ids=stops, suppress_stop=task["suppress_stop"])
            pred = tok.decode(res.natural_tokens(), skip_special_tokens=True)
            score = string_match_all(pred, s.answers) if s.answers else None
            extra = {"answers": s.answers, "pred": pred, "score": score, **s.extra}
            if s.query_needles:
                extra["query_needle_pages"] = needle_pages(tok, s, ps)
            meta = trace_meta(prov=prov, tracer=tr, res=res, task=task["name"], sample_id=f"{task['name']}-{s.index}", **extra)
            n = len(res.tokens)
            paths.append(save_trace(trace_dir / task["name"] / f"{s.index:04d}.safetensors", meta=meta,
                                    mass=tr.mass[:n], obs_mass=tr.obs_numpy(), tokens=np.array(res.tokens),
                                    prompt_ids=ids.cpu().numpy(), fmt=cfg["trace"]["format"], topk=cfg["trace"]["topk"]))
            recs.append({"index": s.index, "prompt_tokens": int(ids.numel()), "natural_stop_step": res.natural_stop_step,
                         "score": score, "pred": pred[:200], "prefill_s": res.prefill_s, "decode_s": res.decode_s})
        t_trace = time.perf_counter() - t0
        print(f"[small_eval] {task['name']}: traced {len(samples)} samples in {t_trace:.0f}s", flush=True)

        t0 = time.perf_counter()
        analyses = [analyze_trace(load_trace(pth), cfg, se["analysis"]) for pth in paths]
        pooled = pool_results(analyses, cfg, se["analysis"])
        t_an = time.perf_counter() - t0
        scores = [r["score"] for r in recs if r["score"] is not None]
        stops_at = [r["natural_stop_step"] for r in recs]
        task_out[task["name"]] = {
            "metrics": pooled,
            "task_score": 100 * float(np.mean(scores)) if scores else None,
            "natural_stop_steps": stops_at,
            "samples": recs,
            "timing": {"build_s": t_build, "trace_s": t_trace, "analysis_s": t_an,
                       "mean_prefill_s": float(np.mean([r["prefill_s"] for r in recs])),
                       "mean_decode_step_ms": 1e3 * float(np.mean([r["decode_s"] / max(1, se["decode_steps"] - 1) for r in recs]))},
        }
        print(f"[small_eval] {task['name']}: analysis {t_an:.0f}s", flush=True)

    timing["total_s"] = time.perf_counter() - t_all
    timing["within_budget"] = timing["total_s"] <= se["max_wall_s"]
    out = Path(se["out_dir"]) / f"{tag}.json"
    write_json(out, {"provenance": prov, "probe": probe, "traced_layers": layers, "timing": timing,
                     "trace_dir": str(trace_dir), "tasks": task_out})
    print(f"[small_eval] total {timing['total_s'] / 60:.1f} min (budget {se['max_wall_s'] / 60:.0f} min) -> {out}")
    for name, t in task_out.items():
        q = t["metrics"]["query"]
        print(f"  {name}: score={t['task_score']} recall@5% last k1={q['recall_last_step_k1']:.3f} "
              f"k4={q['recall_last_step_k4']:.3f} | ema k1={q['recall_ema_k1']:.3f} k4={q['recall_ema_k4']:.3f} "
              f"| lead median last={q['lead_last_step']['median']} ema={q['lead_ema']['median']} "
              f"| gradual share of onsets={q['trajectory']['gradual_share_of_onsets']:.3f}")


if __name__ == "__main__":
    main()
