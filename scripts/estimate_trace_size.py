"""Print and save trace-size estimates. Usage: python scripts/estimate_trace_size.py configs/trace_size.yaml"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvfetch.config import add_config_args, config_from_args
from kvfetch.provenance import provenance, write_json
from kvfetch.trace_size import estimate_all, human_bytes

COLUMNS = ["scenario", "ps", "heads", "L", "P", "dense/sample", "top-K/sample", "top-K+M_w/sample",
           "dense total", "top-K total", "K >= max B?"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--out", default=None, help="JSON output path (default: <results_dir>/trace_size.json)")
    args = p.parse_args()
    cfg = config_from_args(args)

    rows = estimate_all(cfg)
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        d = r.as_dict()
        cover = "yes" if r.topk_covers_max_budget else f"NO (max B={r.max_budget_pages})"
        cells = [r.scenario, r.page_size, f"{r.heads} {r.head_kind}", r.layers, r.pages,
                 human_bytes(r.dense_bytes_per_sample), human_bytes(r.topk_bytes_per_sample),
                 human_bytes(r.topk_with_mw_bytes_per_sample), human_bytes(d["dense_total_bytes"]),
                 human_bytes(d["topk_total_bytes"]), cover]
        print("| " + " | ".join(map(str, cells)) + " |")

    out = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "trace_size.json"
    write_json(out, {"provenance": provenance(cfg, model=None, context_length=None),
                     "rows": [r.as_dict() for r in rows]})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
