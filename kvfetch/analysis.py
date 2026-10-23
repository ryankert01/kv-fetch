"""Per-trace metrics and pooling across samples (used by the small-eval runner)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from kvfetch.metrics import predictors
from kvfetch.metrics.importance import (budget_pages, eligible_pages, flatten_series, future_mass, head_view,
                                        topb_mask)
from kvfetch.metrics.selection import (CENSORED, MISSED, OK, churn, lead_summary, lead_times, recall_at_k,
                                       sets_for_next_step)
from kvfetch.metrics.trajectory import CLASSES, TrajectoryResult, class_shares, classify
from kvfetch.tracing.schema import Trace


def _predictor_scores(m: np.ndarray, names: list[str], cfg: dict) -> dict[str, np.ndarray]:
    table = {"last_step": lambda: predictors.last_step(m),
             "ema": lambda: predictors.ema(m, cfg["reference_predictors"]["ema_alpha"])}
    return {n: table[n]() for n in names}


def topk_coverage(m: np.ndarray, ks: list[int]) -> dict[int, float]:
    """Mean fraction of a row's total mass held by its K heaviest pages (all pages)."""
    x = m.astype(np.float32)
    P = x.shape[-1]
    srt = -np.sort(-x, axis=-1)
    total = srt.sum(-1)
    return {k: float((srt[..., :min(k, P)].sum(-1) / total).mean()) for k in ks}


def analyze_trace(tr: Trace, cfg: dict, acfg: dict) -> dict:
    defs, meta = cfg["definitions"], tr.meta
    ps, w, rounding = defs["page_size"], defs["w"], defs["budget_rounding"]
    m_q = tr.dense()
    elig = eligible_pages(meta["prompt_len"], meta["n_pages"], ps, defs["pinned"])
    n_elig = int(elig.sum())
    B = budget_pages(acfg["budget_frac"], n_elig, rounding)
    B_lead = budget_pages(cfg["lead_time"]["budget_frac"], n_elig, rounding)
    out = {"summary": {"n_steps": tr.n_steps, "n_eligible_pages": n_elig, "B": B,
                       "topk_coverage": topk_coverage(m_q, acfg["topk_coverage_K"])},
           "pool": {}}
    for view in defs["head_views"]:
        m = flatten_series(head_view(m_q, view, meta["num_kv_heads"]).astype(np.float32))  # [T, S, P]
        Mw = future_mass(m, w)
        s, pool = {}, {}
        s["pinned_mass_share"] = float(m[..., ~elig].sum(-1).mean())
        s["true_topB_churn_Mw"] = float(churn(topb_mask(Mw, B, elig), B).mean())
        true_sets = topb_mask(m if cfg["lead_time"]["truth"] == "m" else Mw, B_lead, elig)
        for name, score in _predictor_scores(m, acfg["predictors"], cfg).items():
            for k in acfg["k"]:
                s[f"recall_{name}_k{k}"] = float(np.nanmean(recall_at_k(score, Mw, k, B, elig)))
            pred_for = sets_for_next_step(score, B_lead, elig)
            lead, status = lead_times(pred_for, true_sets, elig, mode=cfg["lead_time"]["mode"])
            s[f"lead_{name}"] = lead_summary(lead, status, cfg["lead_time"]["required_lead"])
            s[f"churn_{name}"] = float(churn(pred_for[1:], B_lead).mean())
            pool[f"lead_{name}"] = (lead[status == OK], int((status == MISSED).sum()),
                                    int((status == CENSORED).sum()))
        traj = classify(m, elig, cfg["trajectory"], rounding)
        s["trajectory"] = class_shares(traj)
        pool["traj"] = traj
        if view == "query" and meta.get("query_needle_pages"):
            # sanity: the queried needle's pages, across all traced (layer, head) series
            S = m.shape[1]
            elig_idx = {p: i for i, p in enumerate(np.flatnonzero(elig))}
            cols = [elig_idx[p] for p in meta["query_needle_pages"] if p in elig_idx]
            sel = (np.arange(S)[:, None] * n_elig + np.array(cols)[None]).ravel()
            s["needle_pages_trajectory"] = {c: float((traj.cls[sel] == i).mean()) for i, c in enumerate(CLASSES)}
        out["summary"][view] = s
        out["pool"][view] = pool
    return out


def pool_results(results: list[dict], cfg: dict, acfg: dict) -> dict:
    """Pool per-sample analyses: means of per-sample scalars; lead/rise statistics over
    the union of events."""
    agg: dict = {"n_samples": len(results)}
    agg["topk_coverage"] = {k: float(np.mean([r["summary"]["topk_coverage"][k] for r in results]))
                            for k in acfg["topk_coverage_K"]}
    for view in cfg["definitions"]["head_views"]:
        sums = [r["summary"][view] for r in results]
        v: dict = {}
        for key, val in sums[0].items():
            if isinstance(val, float):
                v[key] = float(np.mean([x[key] for x in sums]))
        for name in acfg["predictors"]:
            leads = np.concatenate([r["pool"][view][f"lead_{name}"][0] for r in results])
            missed = sum(r["pool"][view][f"lead_{name}"][1] for r in results)
            censored = sum(r["pool"][view][f"lead_{name}"][2] for r in results)
            n = leads.size + missed
            req = cfg["lead_time"]["required_lead"]
            v[f"lead_{name}"] = {
                "events": n, "censored": censored, "missed_rate": missed / n if n else float("nan"),
                "median": float(np.median(leads)) if leads.size else float("nan"),
                "p10": float(np.percentile(leads, 10)) if leads.size else float("nan"),
                "p90": float(np.percentile(leads, 90)) if leads.size else float("nan"),
                "on_time_rate": float((leads >= req).sum() / n) if n else float("nan"),
            }
        trajs = [r["pool"][view]["traj"] for r in results]
        cat = TrajectoryResult(**{f: np.concatenate([getattr(t, f) for t in trajs])
                                  for f in TrajectoryResult.__dataclass_fields__})
        v["trajectory"] = class_shares(cat)
        if "needle_pages_trajectory" in sums[0]:
            v["needle_pages_trajectory"] = {c: float(np.mean([x["needle_pages_trajectory"][c] for x in sums]))
                                            for c in CLASSES}
        agg[view] = v
    return agg
