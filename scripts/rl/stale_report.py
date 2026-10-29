"""Kill test R2: summarize scripts/rl/stale_state.py results for a hybrid and a dense model and apply the gate.

Per step and cut, means over probes of KL(F || X) for every condition X, rho = KL(F||S) / KL(F||P), the same ratio
with the kernel-path floor subtracted from both terms, KL(F||S) / floor, and KL(F||S) by continuation position.
The gate (config `stale.gate`, EXPERIMENTS.md): the hybrid is alive at (step, cut) if KL(F||S) >= floor_multiple
x floor and either rho >= rho_alone or rho >= max(rho_min, rho_vs_dense x the dense model's rho there).

Usage: stale_report.py CONFIG --hybrid STALE.json --dense STALE.json --out OUT.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402


def summarize(path: str) -> dict:
    d = json.load(open(path))
    out = {}
    for res in d["results"]:
        for cut in sorted({r["cut"] for r in res["rows"]}):
            rs = [r for r in res["rows"] if r["cut"] == cut]
            conds = [c for c in rs[0] if c not in ("cut", "probe")]
            m = {c: {k: sum(r[c][k] for r in rs) / len(rs) for k in rs[0][c]} for c in conds}
            fl = m["floor"]["kl"]
            out[(res["step"], cut)] = {
                "probes": len(rs), "kl": {c: m[c]["kl"] for c in conds}, "logratio": {c: m[c]["logratio"] for c in conds},
                "kl_S_by_position": {k: v for k, v in m["S"].items() if k.startswith("kl_")},
                "rho": m["S"]["kl"] / m["P"]["kl"],
                "rho_minus_floor": (m["S"]["kl"] - fl) / (m["P"]["kl"] - fl) if m["P"]["kl"] > fl else None,
                "S_over_floor": m["S"]["kl"] / fl}
    return {"steps": d["steps"], "cells": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--hybrid", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    g = load_config(args.config, [])["stale"]["gate"]
    h, dn = summarize(args.hybrid), summarize(args.dense)
    alive = []
    print(f"{'step':>4} {'cut':>5} | {'hyb KL(F||S)':>12} {'KL(F||P)':>9} {'floor':>8} {'S/floor':>7} {'rho':>5} {'rho-fl':>6} "
          f"{'S_state':>8} {'S_kv':>8} | {'dense rho':>9} {'rho-fl':>6} {'S/floor':>7} | alive")
    for key in sorted(h["cells"]):
        c = h["cells"][key]
        dc = dn["cells"].get(key)
        rho_d = dc["rho"] if dc else None
        ok = key[0] <= g["max_step"] and c["S_over_floor"] >= g["floor_multiple"] and (
            c["rho"] >= g["rho_alone"] or (rho_d is not None and c["rho"] >= max(g["rho_min"], g["rho_vs_dense"] * rho_d)))
        c["alive"] = ok
        if ok:
            alive.append(key)
        fmt = lambda x: "  --" if x is None else f"{x:6.2f}"
        print(f"{key[0]:>4} {key[1]:>5} | {c['kl']['S']:12.3g} {c['kl']['P']:9.3g} {c['kl']['floor']:8.3g} {c['S_over_floor']:7.2f} "
              f"{c['rho']:5.2f} {fmt(c['rho_minus_floor'])} {c['kl'].get('S_state', float('nan')):8.3g} {c['kl'].get('S_kv', float('nan')):8.3g} | "
              f"{fmt(rho_d):>9} {fmt(dc['rho_minus_floor'] if dc else None)} {fmt(dc['S_over_floor'] if dc else None):>7} | {ok}")
    print("verdict:", "ALIVE at " + ", ".join(map(str, alive)) if alive else "KILLED (no step/cut passes the gate)")
    js = lambda s: {"steps": s["steps"], "cells": [{"step": k[0], "cut": k[1], **v} for k, v in sorted(s["cells"].items())]}
    json.dump({"sources": {"hybrid": args.hybrid, "dense": args.dense}, "gate": g, "alive": [list(a) for a in alive],
               "hybrid": js(h), "dense": js(dn)}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
