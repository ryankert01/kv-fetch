"""E3: are truncated generations verbatim loops, or just long?

For every truncated output (no stop token within the generation cap) take the last ``tail_chars``
characters and compute the share of character n-grams (every position) that already occurred earlier
in that tail. An output counts as looping when that share exceeds a threshold. Reported per run for each
threshold in ``e3.repetition.thresholds`` (the main one is ``e3.repetition.threshold``).

  python scripts/e3_repetition.py configs/reasoning.yaml   ->  results/e3_repetition_<split>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from kvfetch.config import add_config_args, config_from_args
from kvfetch.provenance import provenance, write_json


def repeat_share(text: str, tail: int, n: int) -> float:
    t = text[-tail:]
    seen, rep, tot = set(), 0, 0
    for i in range(len(t) - n + 1):
        g = t[i:i + n]
        tot += 1
        rep += g in seen
        seen.add(g)
    return rep / max(tot, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    cfg = config_from_args(p.parse_args())
    e3, rc = cfg["e3"], cfg["e3"]["repetition"]
    labels = [l for tier in e3["runs"] for l in tier]
    out, prov = {}, None
    for label in labels:
        shares = {"truncated": [], "stopped": []}
        for f in sorted((Path(e3["out_dir"]) / label.replace("@", "_")).glob("*/*.json")):
            r = json.loads(f.read_text())
            prov = prov or r["provenance"]
            shares["stopped" if r["stopped"] else "truncated"].append(repeat_share(r["text"], rc["tail_chars"], rc["ngram_chars"]))
        if not shares["truncated"]:
            continue
        tr = np.array(shares["truncated"])
        n_all = len(tr) + len(shares["stopped"])
        out[label] = {
            "n_samples": n_all, "n_truncated": len(tr),
            "looping_share_of_truncated": {str(th): float((tr > th).mean()) for th in rc["thresholds"]},
            "looping_share_of_all": {str(th): float((tr > th).sum() / n_all) for th in rc["thresholds"]},
            "median_repeat_share": {"truncated": float(np.median(tr)),
                                    "stopped": float(np.median(shares["stopped"])) if shares["stopped"] else None},
        }
    th = str(rc["threshold"])
    print(f"{'run':16s} {'trunc':>6s}  looping share of truncated at thresholds {rc['thresholds']}")
    for label, s in out.items():
        print(f"{label:16s} {s['n_truncated']:6d}  " + "  ".join(f"{100 * v:5.1f}%" for v in s["looping_share_of_truncated"].values()))
    write_json(Path(cfg["paths"]["results_dir"]) / f"e3_repetition_{e3['split']}.json", {
        "provenance": provenance(cfg, model=prov["model"], context_length=prov["context_length"], repetition=rc,
                                 main_threshold=th),
        "runs": out})


if __name__ == "__main__":
    main()
