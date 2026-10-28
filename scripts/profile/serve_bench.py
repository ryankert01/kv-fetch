"""P4: online serving benchmarks on upstream vLLM (`vllm serve` + `vllm bench serve`).

For each model and server setting, start one server, run every workload at every concurrency (random
prompts, fixed lengths, ignore_eos; num_prompts = prompts_per_concurrency x concurrency), save each
`vllm bench serve` result, stop the server.

Usage (venv-vllm32, CUDA_VISIBLE_DEVICES set to one GPU): serve_bench.py CONFIG OUT_DIR [--models a,b] [--settings x,y]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def wait_ready(port: int, proc, timeout: float):
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError("server not ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--models", default=None)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    sc = cfg["profile"]["serve"]
    out = Path(args.out_dir).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    models = args.models.split(",") if args.models else sc["models"]
    settings = args.settings.split(",") if args.settings else list(sc["settings"])
    port = sc["port"]
    runs = []
    for mkey in models:
        name = cfg["profile"]["models"][mkey]["name"]
        for sname in settings:
            log = open(out / "logs" / f"serve_{mkey}_{sname}.log", "w")
            cmd = ["vllm", "serve", name, "--port", str(port), *sc["server_args"],
                   *cfg["profile"]["models"][mkey].get("server_args", []), *sc["settings"][sname]]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            try:
                wait_ready(port, proc, sc["ready_timeout_s"])
                for wname, w in sc["workloads"].items():
                    for c in w["concurrency"]:
                        fname = f"{mkey}_{sname}_{wname}_c{c}.json"
                        bench = ["vllm", "bench", "serve", "--backend", "vllm", "--port", str(port), "--model", name,
                                 "--dataset-name", "random", "--random-input-len", str(w["input"]),
                                 "--random-output-len", str(w["output"]), "--random-range-ratio", str(sc["range_ratio"]),
                                 "--num-prompts", str(w["prompts_per_concurrency"] * c), "--max-concurrency", str(c),
                                 "--ignore-eos", "--seed", "0", "--percentile-metrics", "ttft,tpot,itl,e2el",
                                 "--metric-percentiles", "50,90,99", "--save-result", "--result-dir", str(out),
                                 "--result-filename", fname, "--disable-tqdm"]
                        t = time.time()
                        r = subprocess.run(bench, capture_output=True, text=True)
                        (out / "logs" / f"bench_{fname[:-5]}.log").write_text(r.stdout + r.stderr)
                        runs.append({"model": mkey, "setting": sname, "workload": wname, "concurrency": c,
                                     "result": fname, "ok": r.returncode == 0, "seconds": time.time() - t})
                        print(f"{mkey} {sname} {wname} c={c}: {'ok' if r.returncode == 0 else 'FAILED'} "
                              f"({time.time() - t:.0f}s)", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
                time.sleep(10)
    import vllm
    prov = provenance(cfg, model=None, context_length=None, script=str(Path(__file__).resolve()), settings=sc,
                      vllm=vllm.__version__, load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "runs": runs}, open(out / "serve_index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
