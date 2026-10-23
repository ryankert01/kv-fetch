"""Aggregate gate job files into a table + pass/fail per (model, length).

  python scripts/summarize_model_gate.py configs/model_gate.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvfetch.config import add_config_args, config_from_args
from kvfetch.provenance import provenance, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    args = p.parse_args()
    cfg = config_from_args(args)
    g = cfg["gate"]
    out_dir = Path(g["out_dir"])
    rows, missing = [], []
    for model in g["candidates"] + g["reference_candidates"]:
        for L in g["lengths"]:
            f = out_dir / f"{model.replace('/', '__')}__{L}.json"
            if not f.exists():
                missing.append((model, L))
                continue
            d = json.load(open(f))
            rows.append({"model": model, "length": L, "role": "reference" if model in g["reference_candidates"] else "iteration",
                         **{t: d["per_task"][t] for t in g["tasks"]}, **{f"group_{k}": v for k, v in d["groups"].items()},
                         "passed": d["passed"], "wall_min": d["wall_s"] / 60,
                         "mean_prefill_s": sum(r["prefill_s"] for r in d["records"]) / len(d["records"]),
                         "git": d["provenance"]["git"]["commit"][:7]})
    longest = {}
    for r in rows:
        if r["passed"]:
            longest[r["model"]] = max(longest.get(r["model"], 0), r["length"])

    short = {t: t.replace("niah_", "").replace("single_", "s").replace("multikey_", "mk") for t in g["tasks"]}
    cols = ["model", "length"] + [short[t] for t in g["tasks"]] + [f"{k} avg" for k in g["groups"]] + ["pass", "min"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [r["model"], f"{r['length'] // 1024}K"] + [f"{r[t]:.0f}" for t in g["tasks"]] + \
                [f"{r[f'group_{k}']:.1f}" for k in g["groups"]] + ["yes" if r["passed"] else "no", f"{r['wall_min']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    print(f"\npass rule: every group mean >= {g['pass_threshold']}; {g['num_samples']} samples/task")
    print("longest passing length:", longest or "none")
    if missing:
        print("missing jobs:", missing)
    write_json(out_dir / "summary.json", {"provenance": provenance(cfg, model=None, context_length=None),
                                          "rows": rows, "longest_passing_length": longest, "missing": missing})
    (out_dir / "summary.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
