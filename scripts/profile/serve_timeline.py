"""Engine state over time from a serve_bench run: the server's 10-second stats lines, per benchmark run.

Each `vllm serve` log line "Engine 000: Avg prompt throughput: X tokens/s, Avg generation throughput: Y
tokens/s, Running: R reqs, Waiting: W reqs, GPU KV cache usage: K%[, Prefix cache hit rate: H%]" becomes a
point; points are assigned to the benchmark run whose time window holds them. Run windows come from
serve_index.json (`started`, `seconds`), or for older runs from each result file's mtime minus its duration.

Usage: serve_timeline.py RUN_DIR [RUN_DIR ...]   (writes RUN_DIR/serve_timeline.json)
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine \d+: Avg prompt throughput: ([\d.]+) tokens/s, "
                  r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                  r"GPU KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?")


def points(log: Path, year: int) -> list[dict]:
    out = []
    for m in LINE.finditer(log.read_text(errors="replace")):
        t = dt.datetime.strptime(f"{year}-{m[1]}", "%Y-%m-%d %H:%M:%S").timestamp()
        out.append({"t": t, "prompt_tps": float(m[2]), "gen_tps": float(m[3]), "running": int(m[4]), "waiting": int(m[5]),
                    "kv_pct": float(m[6]), "hit_pct": float(m[7]) if m[7] else None})
    return out


def main():
    for d in sys.argv[1:]:
        rd = Path(d)
        index = json.load(open(rd / "serve_index.json"))
        year = dt.datetime.fromtimestamp((rd / "serve_index.json").stat().st_mtime).year
        res = {"source": str(rd), "runs": []}
        for r in index["runs"]:
            if "started" in r:
                t0 = r["started"]
            else:                                 # older runs: result written at the end
                t0 = (rd / r["result"]).stat().st_mtime - r["seconds"]
            pts = [p for p in points(rd / "logs" / f"serve_{r['model']}_{r['setting']}.log", year) if t0 <= p["t"] <= t0 + r["seconds"]]
            for p in pts:
                p["t"] = round(p["t"] - t0)
            res["runs"].append({k: r[k] for k in ("model", "setting", "workload", "concurrency", "result", "seconds")} | {"points": pts})
            print(f"{rd.name}: {r['model']} {r['setting']} {r['workload']} c={r['concurrency']}: {len(pts)} points")
        json.dump(res, open(rd / "serve_timeline.json", "w"), indent=1)


if __name__ == "__main__":
    main()
