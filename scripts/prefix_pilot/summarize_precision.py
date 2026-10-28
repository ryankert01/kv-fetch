"""Merge ckpt_precision shards and summarize per variant: KL (mean over rows, bootstrap CI), ratio to the
rechunk floor (paired, per row), top-1 agreement, NLL change, bytes per checkpoint.
Usage: summarize_precision.py OUT.json SHARD.json [SHARD.json ...]"""
import json
import sys

import numpy as np

BOOT, SEED = 2000, 0


def ci(x, rng):
    x = np.asarray(x)
    m = rng.choice(x, size=(BOOT, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    shards = [json.load(open(p)) for p in sys.argv[2:]]
    rows = [r for s in shards for r in s["rows"]]
    sizes = shards[0]["bytes_per_layer"]
    layers = len(shards[0]["provenance"]["gdn_layers"])
    rng = np.random.default_rng(SEED)
    variants = list(rows[0]["variants"])
    bins = [k for k in rows[0]["variants"]["rechunk"]["kl"] if k not in ("all", "max")]
    out = {"n_rows": len(rows), "n_seqs": len({(r["task"], r["index"]) for r in rows}), "offsets": sorted({r["offset"] for r in rows}),
           "rows_per_offset": {str(o): sum(r["offset"] == o for r in rows) for o in sorted({r["offset"] for r in rows})},
           "variants": {}}
    for v in variants:
        kl = [r["variants"][v]["kl"]["all"] for r in rows]
        ratio = [r["variants"][v]["kl"]["all"] / r["variants"]["rechunk"]["kl"]["all"] for r in rows]
        d = {"kl_all": ci(kl, rng),
             "kl_ratio_to_rechunk": ci(ratio, rng),
             "kl_bins": {b: float(np.mean([r["variants"][v]["kl"][b] for r in rows])) for b in bins},
             "kl_max_mean": float(np.mean([r["variants"][v]["kl"]["max"] for r in rows])),
             "agree_all": float(np.mean([r["variants"][v]["agree"]["all"] for r in rows])),
             "agree_bins": {b: float(np.mean([r["variants"][v]["agree"][b] for r in rows])) for b in bins},
             "dnll_all": ci([r["variants"][v]["dnll"]["all"] for r in rows], rng),
             "kl_curve": np.mean([r["variants"][v]["kl_curve"] for r in rows], 0).tolist(),
             "kl_by_offset": {str(o): float(np.mean([r["variants"][v]["kl"]["all"] for r in rows if r["offset"] == o]))
                              for o in out["offsets"]}}
        if v in sizes:
            d["checkpoint_MiB"] = sizes[v] * layers / 2 ** 20
        if v not in ("rechunk", "identity", "zero"):
            d["rel_error"] = float(np.mean([r["stats"]["rel_error"][v] for r in rows]))
        out["variants"][v] = d
    out["fp32_checkpoint_MiB"] = sizes["fp32"] * layers / 2 ** 20
    out["effective_rank_mean"] = float(np.mean([np.mean(r["stats"]["effective_rank_mean"]) for r in rows]))
    out["effective_rank_by_layer"] = np.mean([r["stats"]["effective_rank_mean"] for r in rows], 0).tolist()
    out["provenance"] = {"shards": sys.argv[2:], "shard_provenance": [s["provenance"] for s in shards],
                         "bootstrap": BOOT, "seed": SEED}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for v, d in out["variants"].items():
        print(f"{v:14s} KL {d['kl_all'][0]:.2e} [{d['kl_all'][1]:.2e},{d['kl_all'][2]:.2e}]  x floor "
              f"{d['kl_ratio_to_rechunk'][0]:6.2f}  agree {d['agree_all']:.4f}  dNLL {d['dnll_all'][0]:+.2e}  "
              f"MiB {d.get('checkpoint_MiB', float('nan')):6.2f}  bins " +
              " ".join(f"{x:.1e}" for x in d["kl_bins"].values()))
    print("effective rank (99% energy) mean", out["effective_rank_mean"])


if __name__ == "__main__":
    main()
