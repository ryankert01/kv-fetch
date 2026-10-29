"""P4: online serving benchmarks on upstream vLLM (`vllm serve` + `vllm bench serve`).

For each model and server setting, start one server, run every workload at every concurrency (random
prompts, fixed lengths unless the workload sets its own range_ratio, ignore_eos; num_prompts =
prompts_per_concurrency x concurrency), save each `vllm bench serve` result together with the change of the
server's Prometheus counters over the run (preemptions, prompt / cached tokens: `metrics` in
serve_index.json, rewritten after every run), stop the server. Each (workload, concurrency) gets its own
bench seed (serve.seed + its index), the same for every model and setting: with one seed, a run's first
prompts repeat the previous run's (the random dataset draws start offsets in sequence) and hit the prefix
cache the previous run left behind, the more so the larger the model's KV pool. The server port is the
first free one from serve.port + 10 x the GPU index, so jobs on different GPUs do not share a server; a
run is ok only when the bench completed requests with none failed, and the script exits non-zero if any
run was not ok.

Usage (venv-vllm32, CUDA_VISIBLE_DEVICES set to one GPU): serve_bench.py CONFIG OUT_DIR [--models a,b] [--settings x,y] [--workloads w1,w2] [--set k=v]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def dataset_args(w: dict, sc: dict) -> list[str]:
    if w.get("dataset", "random") == "prefix_repetition":   # shared prefixes + short new suffixes: multi-turn-like reuse
        return ["--dataset-name", "prefix_repetition", "--prefix-repetition-prefix-len", str(w["prefix"]),
                "--prefix-repetition-suffix-len", str(w["suffix"]), "--prefix-repetition-num-prefixes", str(w["num_prefixes"]),
                "--prefix-repetition-output-len", str(w["output"])]
    return ["--dataset-name", "random", "--random-input-len", str(w["input"]), "--random-output-len", str(w["output"]),
            "--random-range-ratio", str(w.get("range_ratio", sc["range_ratio"]))]


def counters(port: int) -> dict[str, float]:
    """vLLM's Prometheus counters and histogram sums/counts, summed over label sets."""
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30).read().decode()
    res: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{")[0]
        if name.endswith(("_total", "_sum", "_count")):
            res[name] = res.get(name, 0.0) + float(value)
    return res


def free_port(start: int, tries: int = 10) -> int:
    """First port from `start` nobody listens on (a range per GPU, so parallel jobs never share a server;
    other users' services hold some ports on this machine)."""
    for port in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in {start}..{start + tries - 1}")


def wait_ready(port: int, proc, name: str, timeout: float):
    """Until our server answers; the model check guards against another job's server on the same port."""
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            models = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5))
        except Exception:
            time.sleep(5)
            continue
        if name not in [m["id"] for m in models["data"]]:
            raise RuntimeError(f"port {port} serves {models['data']}, not {name}")
        return
    raise TimeoutError("server not ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--models", default=None)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--workloads", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    sc = cfg["profile"]["serve"]
    out = Path(args.out_dir).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    models = args.models.split(",") if args.models else sc["models"]
    settings = args.settings.split(",") if args.settings else list(sc["settings"])
    port = free_port(sc["port"] + 10 * int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))
    runs = []
    import vllm
    prov = provenance(cfg, model=None, context_length=None, script=str(Path(__file__).resolve()), settings=sc,
                      vllm=vllm.__version__, load_avg=list(os.getloadavg()))

    points = [(wname, c) for wname, w in sc["workloads"].items() for c in w["concurrency"]]

    def write_index():                    # after every run, so a long job's finished runs are readable
        json.dump({"provenance": prov, "runs": runs}, open(out / "serve_index.json", "w"), indent=1)
    for mkey in models:
        name = cfg["profile"]["models"][mkey]["name"]
        for sname in settings:
            log = open(out / "logs" / f"serve_{mkey}_{sname}.log", "w")
            cmd = ["vllm", "serve", name, "--port", str(port), *sc["server_args"],
                   *cfg["profile"]["models"][mkey].get("server_args", []), *sc["settings"][sname]]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            try:
                wait_ready(port, proc, name, sc["ready_timeout_s"])
                for wname, w in sc["workloads"].items():
                    if args.workloads and wname not in args.workloads.split(","):
                        continue
                    for c in w["concurrency"]:
                        fname = f"{mkey}_{sname}_{wname}_c{c}.json"
                        bench = ["vllm", "bench", "serve", "--backend", "vllm", "--port", str(port), "--model", name,
                                 *dataset_args(w, sc),
                                 "--num-prompts", str(w["prompts_per_concurrency"] * c), "--max-concurrency", str(c),
                                 "--ignore-eos", "--seed", str(sc["seed"] + points.index((wname, c))),
                                 "--percentile-metrics", "ttft,tpot,itl,e2el",
                                 "--metric-percentiles", "50,90,99", "--save-result", "--result-dir", str(out),
                                 "--result-filename", fname, "--disable-tqdm"]
                        t = time.time()
                        before = counters(port)
                        r = subprocess.run(bench, capture_output=True, text=True)
                        after = counters(port)
                        (out / "logs" / f"bench_{fname[:-5]}.log").write_text(r.stdout + r.stderr)
                        res = json.load(open(out / fname)) if (out / fname).exists() else {}
                        ok = r.returncode == 0 and res.get("completed", 0) > 0 and res.get("failed", 1) == 0
                        runs.append({"model": mkey, "setting": sname, "workload": wname, "concurrency": c,
                                     "seed": sc["seed"] + points.index((wname, c)), "result": fname,
                                     "ok": ok, "started": t, "seconds": time.time() - t,
                                     "metrics": {k: v - before.get(k, 0.0) for k, v in after.items()
                                                 if v != before.get(k, 0.0)}})
                        write_index()
                        print(f"{mkey} {sname} {wname} c={c}: {'ok' if ok else 'FAILED'} "
                              f"({time.time() - t:.0f}s)", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
                time.sleep(10)
    write_index()
    if not all(r["ok"] for r in runs):
        sys.exit("some benchmark runs failed (serve_index.json: ok false)")


if __name__ == "__main__":
    main()
