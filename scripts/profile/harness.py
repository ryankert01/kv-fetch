"""Long-running profiling harness: a job queue that runs on free GPUs and re-analyses after every job.

Jobs are JSON files in <root>/{pending,running,done,failed}/<id>.json. A job runs one command with
CUDA_VISIBLE_DEVICES set to the GPU it was given, in <runs>/<id>/ (log in job.log; a retry's in
job.attempt<N>.log, so the failed attempt's log survives). Kinds:
  engine  scripts/profile/engine_profile.py on one model (+ --set overrides, flags), then parse_trace.py
  serve   scripts/profile/serve_bench.py (models, settings)
  cmd     any command; "{out}" in it becomes the run directory
Each job runs in its own process group, killed as a whole on timeout.
GPU policy (configs/harness.yaml): only `allowed` GPUs; a GPU is free when its memory use and utilisation
stay under the thresholds over several samples; while the `watch` GPU (GPU 0) is in use by others, at
most `max_concurrent_when_watch_busy` jobs run, otherwise `max_concurrent`.

Usage (venv-vllm32):
  harness.py CONFIG [--set k=v] enqueue PLAN.yaml        add the plan's jobs not already known
  harness.py CONFIG run --until ISO_TIME     run the queue (keeps polling when idle) until that time
  harness.py CONFIG wait --timeout-min N     block until a job finishes, the queue goes idle, or timeout
  harness.py CONFIG status
"""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from kvfetch.config import load_config  # noqa: E402

STATES = ("pending", "running", "done", "failed")


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Queue:
    def __init__(self, hc: dict):
        self.root = REPO / hc["queue_dir"]
        self.runs = REPO / hc["runs_dir"]
        for s in STATES:
            (self.root / s).mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)

    def jobs(self, state: str) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.root / state).glob("*.json"))]

    def known(self) -> set[str]:
        return {p.stem for s in STATES for p in (self.root / s).glob("*.json")}

    def move(self, job: dict, src: str, dst: str):
        (self.root / dst / f"{job['id']}.json").write_text(json.dumps(job, indent=1))
        p = self.root / src / f"{job['id']}.json"
        if p.exists():
            p.unlink()


def gpu_sample() -> dict[int, tuple[int, int]]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        i, mem, util = (int(x) for x in line.split(","))
        res[i] = (mem, util)
    return res


def gpu_state(hc: dict, mine: set[int]) -> tuple[list[int], bool]:
    """Free allowed GPUs (not running our jobs) and whether the watched GPU is busy."""
    g = hc["gpus"]
    samples = []
    for k in range(g["samples"]):
        samples.append(gpu_sample())
        if k + 1 < g["samples"]:
            time.sleep(g["sample_interval_s"])
    busy = lambda i: any(s[i][0] > g["max_mem_MiB"] or s[i][1] > g["max_util"] for s in samples)
    free = [i for i in g["allowed"] if i not in mine and i in samples[0] and not busy(i)]
    return free, busy(g["watch"])


def command(job: dict, out: Path, hc: dict) -> list[str]:
    py = sys.executable
    if job["kind"] == "engine":
        sets = [x for s in job.get("sets", []) for x in ("--set", s)]
        prof = [py, "scripts/profile/engine_profile.py", hc["profile_config"], str(out), "--model", job["model"],
                *sets, *job.get("flags", [])]
        parse = [py, "scripts/profile/parse_trace.py", str(out), job["model"]]
        return ["bash", "-c", " ".join(map(sh, prof)) + " && " + " ".join(map(sh, parse))]
    if job["kind"] == "serve":
        sets = [x for s in job.get("sets", []) for x in ("--set", s)]
        wl = ["--workloads", ",".join(job["workloads"])] if job.get("workloads") else []
        return [py, "scripts/profile/serve_bench.py", hc["profile_config"], str(out), "--models", ",".join(job["models"]),
                "--settings", ",".join(job["settings"]), *wl, *sets]
    if job["kind"] == "cmd":
        return ["bash", "-c", job["cmd"].replace("{out}", str(out)).replace("{py}", py)]
    raise ValueError(job["kind"])


def sh(s: str) -> str:
    return "'" + str(s).replace("'", "'\\''") + "'"


def run(cfg: dict, until: dt.datetime):
    hc = cfg["harness"]
    q = Queue(hc)
    for j in q.jobs("running"):            # a previous runner died: requeue its jobs
        q.move(j, "running", "pending")
    procs: dict[str, tuple[subprocess.Popen, dict, float]] = {}
    log = open(q.root / "runner.log", "a")

    def note(msg):
        log.write(f"{now()} {msg}\n")
        log.flush()

    note(f"runner start pid {os.getpid()} until {until.isoformat()}")
    while True:
        for jid, (p, job, t0) in list(procs.items()):
            if p.poll() is None:
                if time.time() - t0 > job.get("timeout_s", hc["default_timeout_s"]):
                    os.killpg(p.pid, signal.SIGKILL)      # the whole job, servers and engine processes included
                    note(f"{jid} timeout, killed")
                continue
            job.update(returncode=p.returncode, finished=now(), seconds=round(time.time() - t0))
            ok = p.returncode == 0
            if not ok and job.get("attempts", 1) < hc["max_attempts"]:
                job["attempts"] = job.get("attempts", 1) + 1
                q.move(job, "running", "pending")
                note(f"{jid} failed rc={p.returncode}, requeued (attempt {job['attempts']})")
            else:
                q.move(job, "running", "done" if ok else "failed")
                note(f"{jid} {'done' if ok else 'FAILED'} rc={p.returncode} in {job['seconds']}s")
            del procs[jid]
            subprocess.run([sys.executable, "scripts/profile/findings.py", hc["profile_config"]], cwd=REPO,
                           stdout=open(q.root / "findings.log", "w"), stderr=subprocess.STDOUT)
        pending = sorted(q.jobs("pending"), key=lambda j: (j.get("priority", 50), j["id"]))
        done = {p.stem for p in (q.root / "done").glob("*.json")}
        ready = [j for j in pending if all(d in done for d in j.get("after", []))]
        if ready and dt.datetime.now() < until:
            mine = {job["gpu"] for _, job, _ in procs.values()}
            free, watch_busy = gpu_state(hc, mine)
            cap = hc["max_concurrent_when_watch_busy"] if watch_busy else hc["max_concurrent"]
            while ready and free and len(procs) < cap:
                job, gpu = ready.pop(0), free.pop(0)
                out = q.runs / job["id"]
                out.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, **{k: str(v) for k, v in job.get("env", {}).items()}, "CUDA_VISIBLE_DEVICES": str(gpu)}
                job.update(gpu=gpu, started=now(), out=str(out.relative_to(REPO)), watch_busy=watch_busy)
                n = job.get("attempts", 1)
                p = subprocess.Popen(command(job, out, hc), cwd=REPO, env=env,
                                     stdout=open(out / ("job.log" if n == 1 else f"job.attempt{n}.log"), "w"),
                                     stderr=subprocess.STDOUT, start_new_session=True)
                procs[job["id"]] = (p, job, time.time())
                q.move(job, "pending", "running")
                note(f"{job['id']} start on GPU {gpu} (watch busy {watch_busy})")
        state = {"updated": now(), "pid": os.getpid(), "until": until.isoformat(),
                 "running": [{"id": j, "gpu": job["gpu"], "since": job["started"]} for j, (_, job, _) in procs.items()],
                 "pending": len(pending), "done": len(done), "failed": len(list((q.root / "failed").glob("*.json")))}
        (q.root / "state.json").write_text(json.dumps(state, indent=1))
        if dt.datetime.now() >= until and not procs:
            note("runner stop (time)")
            return
        time.sleep(hc["poll_s"])


def wait(cfg: dict, timeout_min: float):
    q = Queue(cfg["harness"])
    start = {p.stem for s in ("done", "failed") for p in (q.root / s).glob("*.json")}
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        finished = [p.stem for s in ("done", "failed") for p in (q.root / s).glob("*.json") if p.stem not in start]
        idle = not list((q.root / "pending").glob("*.json")) and not list((q.root / "running").glob("*.json"))
        if finished or idle:
            break
        time.sleep(30)
    status(cfg)
    print("finished since wait:", [p.stem for s in ("done", "failed") for p in (q.root / s).glob("*.json") if p.stem not in start])


def status(cfg: dict):
    q = Queue(cfg["harness"])
    st = q.root / "state.json"
    print(st.read_text() if st.exists() else "no runner state")
    for s in STATES:
        print(s, [j["id"] for j in q.jobs(s)])


def enqueue(cfg: dict, plan_path: str):
    q = Queue(cfg["harness"])
    plan = yaml.safe_load(open(plan_path))
    known = q.known()
    added = 0
    for job in plan["jobs"]:
        if job["id"] in known:
            continue
        job.setdefault("priority", 50)
        job["queued"] = now()
        job["plan"] = str(Path(plan_path).resolve().relative_to(REPO)) if str(Path(plan_path).resolve()).startswith(str(REPO)) else plan_path
        (q.root / "pending" / f"{job['id']}.json").write_text(json.dumps(job, indent=1))
        added += 1
    print(f"enqueued {added} job(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enqueue")
    e.add_argument("plan")
    r = sub.add_parser("run")
    r.add_argument("--until", required=True)
    w = sub.add_parser("wait")
    w.add_argument("--timeout-min", type=float, default=120)
    sub.add_parser("status")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.cmd == "enqueue":
        enqueue(cfg, args.plan)
    elif args.cmd == "run":
        run(cfg, dt.datetime.fromisoformat(args.until))
    elif args.cmd == "wait":
        wait(cfg, args.timeout_min)
    else:
        status(cfg)


if __name__ == "__main__":
    main()
