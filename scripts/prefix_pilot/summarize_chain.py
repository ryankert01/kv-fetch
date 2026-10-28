"""Merge ckpt_chain shards: per variant mean KL, ratio to the split floor (paired per checkpoint, bootstrap CI),
chained / one-shot ratio, and the KL curve. Usage: summarize_chain.py OUT.json SHARD.json [...]"""
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
    rng = np.random.default_rng(SEED)
    out = {"n_rows": len(rows), "variants": {}, "provenance": {"shards": sys.argv[2:], "bootstrap": BOOT, "seed": SEED,
                                                               "shard_provenance": [s["provenance"] for s in shards]}}
    for v in rows[0]["variants"]:
        kl = [r["variants"][v]["kl"]["all"] for r in rows]
        d = {"kl_all": ci(kl, rng),
             "ratio_to_split": ci([r["variants"][v]["kl"]["all"] / r["variants"]["split"]["kl"]["all"] for r in rows], rng),
             "agree_all": float(np.mean([r["variants"][v]["agree"]["all"] for r in rows])),
             "kl_curve": np.mean([r["variants"][v]["kl_curve"] for r in rows], 0).tolist()}
        if v.endswith("_chain"):
            once = v[:-len("_chain")] + "_once"
            d["chain_over_once"] = ci([r["variants"][v]["kl"]["all"] / r["variants"][once]["kl"]["all"] for r in rows], rng)
        out["variants"][v] = d
        print(f"{v:22s} KL {d['kl_all'][0]:.2e}  x split {d['ratio_to_split'][0]:5.2f} [{d['ratio_to_split'][1]:.2f},{d['ratio_to_split'][2]:.2f}]"
              + (f"  chain/once {d['chain_over_once'][0]:.2f}" if "chain_over_once" in d else ""))
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
