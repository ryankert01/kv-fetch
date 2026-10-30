"""Run one GRPO rollout step inside vime and record its time, work and vLLM preemptions.

Checks the vime PR (results/profile/upstream/vime_pr_draft.md) in vime rather than on vLLM directly: vime's own
train.py with --debug-rollout-only (vime's data loader and chat template, router, vLLM server launch, GRPO
sampling and reward; no Megatron training step) on one engine over config vime.num_gpus GPUs (tensor parallel),
optionally with speculative decoding (vime.speculative_config, as vime's hybrid recipes run MTP), on the R1 rollout
step of configs/rl_rollout.yaml (the same prompts, num_prompts x n, max_tokens, temperature), at one
`--vllm-watermark`. vime and its pinned, patched vLLM run from the env in config `vime`, under a private Ray head
(own ports and temp dir, this job's GPUs only) that is torn down at the end. While the step runs, the vLLM server's
/metrics is polled for preemptions and draft acceptance, and /server_info gives the config the engine actually
runs with.

Usage (venv-vllm32, vime.num_gpus GPUs via CUDA_VISIBLE_DEVICES):
  vime_verify.py CONFIG OUT_DIR --model KEY --watermark W [--set k=v]
Writes OUT_DIR/prompts.jsonl, train.log, ray_head.log, metrics.jsonl, vime_verify.json; a retry first renames an
earlier attempt's logs to NAME.attemptN.
"""
import argparse
import ast
import fcntl
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402
from rollout_trace import load_prompts  # noqa: E402

SERVER_RE = re.compile(r"Starting vLLM server on (http://\S+)")
PERF_RE = re.compile(r"perf (\d+): (\{.*\})")
NP_RE = re.compile(r"np\.\w+\(([^()]*)\)")
METRICS = ("vllm:num_preemptions_total", "vllm:num_requests_running", "vllm:num_requests_waiting",
           "vllm:kv_cache_usage_perc", "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
           "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_total")


def write_prompts(rc: dict, path: Path) -> int:
    """The R1 prompts (rollout_trace.load_prompts), with each problem's answer as vime's label."""
    pc = rc["prompts"]
    rows = [json.loads(line) for line in open(pc["jsonl"])]
    rows = [r for r in rows if r.get("level") in pc["levels"]]
    random.Random(rc["seed"]).shuffle(rows)
    rows = rows[:rc["num_prompts"]]
    prompts = [f"{r[pc['field']]}\n{pc['instruction']}" for r in rows]
    assert prompts == load_prompts(rc), "prompt selection drifted from rollout_trace.load_prompts"
    with open(path, "w") as f:
        for p, r in zip(prompts, rows):
            f.write(json.dumps({"prompt": p, "label": r["answer"]}) + "\n")
    return len(prompts)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def model_args(repo: Path, rel: str) -> list[str]:
    out = subprocess.run(["bash", "-c", f'source "{repo / rel}"; printf "%s\\n" "${{MODEL_ARGS[@]}}"'],
                         capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def parse_dict(text: str) -> dict:
    """A dict vime logged with %s / f-string (numpy scalars print as np.float64(x))."""
    return ast.literal_eval(NP_RE.sub(r"\1", text))


def scrape(url: str) -> dict:
    vals = {}
    for line in requests.get(f"{url}/metrics", timeout=10).text.splitlines():
        name = line.split("{")[0].split(" ")[0]
        if name in METRICS:
            vals[name] = vals.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return vals


class Poller(threading.Thread):
    """Follows train.log for the vLLM server URL, then polls its /metrics; keeps /server_info once."""

    def __init__(self, log: Path, out: Path, every_s: float):
        super().__init__(daemon=True)
        self.log, self.out, self.every_s = log, out, every_s
        self.stop = threading.Event()
        self.url, self.server_info, self.last = None, None, {}

    def run(self):
        with open(self.out, "w") as f:
            while not self.stop.is_set():
                if self.url is None and self.log.exists():
                    m = SERVER_RE.search(self.log.read_text(errors="replace"))
                    if m:
                        self.url = m.group(1)
                if self.url is not None:
                    try:
                        vals = scrape(self.url)
                        if vals:
                            self.last = vals
                            f.write(json.dumps({"t": time.time(), **vals}) + "\n")
                            f.flush()
                        if self.server_info is None:
                            r = requests.get(f"{self.url}/server_info?config_format=json", timeout=10)
                            if r.ok:
                                self.server_info = r.json()
                    except requests.RequestException:
                        pass                    # server still starting, or gone at the end
                self.stop.wait(self.every_s)


def find_key(obj, key):
    """First value of `key` anywhere in a nested dict/list (the server_info layout varies across vLLM)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def kill_session(sid: int, grace_s: float):
    """SIGTERM, then SIGKILL, every process of the private Ray head's session (it never leaves it)."""
    def members():
        pids = []
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    if os.getsid(int(p.name)) == sid:
                        pids.append(int(p.name))
                except (ProcessLookupError, PermissionError):
                    pass
        return pids
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5)):
        for pid in members():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + wait
        while members() and time.time() < deadline:
            time.sleep(1)
    return members()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--watermark", type=float, required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, rc, vc = cfg["profile"], cfg["rollout"], cfg["vime"]
    mc = pc["models"][args.model]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train.log", "ray_head.log", "metrics.jsonl"):     # a retry keeps the earlier attempt's logs
        if (out / name).exists():
            (out / name).rename(next(out / f"{name}.attempt{i}" for i in range(1, 100)
                                     if not (out / f"{name}.attempt{i}").exists()))
    repo, py = Path(vc["repo"]), vc["python"]
    ng, spec = vc["num_gpus"], vc["speculative_config"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    assert len(visible) == ng, f"vime.num_gpus is {ng}, CUDA_VISIBLE_DEVICES has {visible}"

    from huggingface_hub import snapshot_download
    hf = snapshot_download(mc["name"], local_files_only=True)
    n_prompts = write_prompts(rc, out / "prompts.jsonl")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
           "PYTHONPATH": f"{vc['megatron']}:{repo}", "CUDA_HOME": vc["cuda_home"],
           "PATH": f"{Path(py).parent}:{vc['cuda_home']}/bin:{os.environ['PATH']}",
           "no_proxy": "localhost,127.0.0.1,0.0.0.0",
           "VIME_HOST_IP": "127.0.0.1"}        # vime's router binds here; its vLLM server still takes the LAN IP (route probe)
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "RAY_ADDRESS"):
        env.pop(k, None)
    port = free_port()
    ray_tmp = Path(vc["ray"]["temp_root"]) / f"rv_{os.getpid()}"
    head = subprocess.Popen(
        [str(Path(py).parent / "ray"), "start", "--head", "--block", "--port", str(port), "--node-ip-address", "127.0.0.1",
         "--temp-dir", str(ray_tmp),
         "--num-gpus", str(ng), "--num-cpus", str(vc["ray"]["num_cpus"]), "--object-store-memory",
         str(vc["ray"]["object_store_bytes"]), "--include-dashboard", "false", "--disable-usage-stats"],
        env=env, stdout=open(out / "ray_head.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)

    train_args = [
        "--debug-rollout-only", "--rollout-num-gpus", str(ng), "--rollout-num-gpus-per-engine", str(ng),
        "--num-gpus-per-node", str(ng), "--hf-checkpoint", hf, *model_args(repo, vc["model_args"][args.model]),
        "--prompt-data", str(out / "prompts.jsonl"), "--input-key", "prompt", "--label-key", "label",
        "--apply-chat-template", "--rm-type", vc["rm_type"], "--num-rollout", "1",
        "--rollout-batch-size", str(n_prompts), "--n-samples-per-prompt", str(rc["n"]),
        "--rollout-max-response-len", str(rc["max_tokens"]), "--rollout-temperature", str(rc["temperature"]),
        "--rollout-top-p", str(rc["top_p"]), "--global-batch-size", str(vc["global_batch_size"]),
        "--vllm-gpu-memory-utilization", str(vc["gpu_memory_utilization"]), "--vllm-watermark", str(args.watermark),
    ] + (["--vllm-speculative-config", json.dumps(spec)] if spec else [])
    log = out / "train.log"
    poller = Poller(log, out / "metrics.jsonl", vc["metrics_poll_s"])
    rc_train, leftover = None, []
    try:
        deadline = time.time() + 120
        while time.time() < deadline:            # the head is up once its GCS port accepts connections
            try:
                socket.create_connection(("127.0.0.1", port), timeout=2).close()
                break
            except OSError:
                if head.poll() is not None:
                    raise RuntimeError(f"ray head exited, see {out / 'ray_head.log'}")
                time.sleep(2)
        time.sleep(5)
        poller.start()
        t0 = time.time()
        with open(log, "w") as f, open(vc["start_lock"], "w") as lock:
            # vime picks its vLLM port by probing up from 15000 well before the server binds it, so concurrent runs
            # take turns from train.py start until this run's server is up (outside vime's rollout_time)
            fcntl.flock(lock, fcntl.LOCK_EX)
            proc = subprocess.Popen([py, "train.py", *train_args], cwd=repo, stdout=f, stderr=subprocess.STDOUT,
                                    env={**env, "RAY_ADDRESS": f"127.0.0.1:{port}"})
            while proc.poll() is None and poller.url is None:
                time.sleep(2)
            fcntl.flock(lock, fcntl.LOCK_UN)
            rc_train = proc.wait()
        wall = time.time() - t0
    finally:
        poller.stop.set()
        poller.join(timeout=30)
        leftover = kill_session(head.pid, vc["stop_grace_s"])      # session leader: its pid is the sid

    text = log.read_text(errors="replace")
    perf = [parse_dict(m.group(2)) for m in PERF_RE.finditer(text)]
    if rc_train != 0 or not perf:
        sys.exit(f"train.py exit {rc_train}, {len(perf)} perf lines; see {log}")
    p = perf[-1]
    rollout_time = p["perf/rollout_time"]
    samples = n_prompts * rc["n"]
    out_tokens = p["perf/tokens_per_gpu_per_sec"] * rollout_time * ng      # vime divides by rollout GPUs
    info = poller.server_info or {}
    sched_wm, engine_tp = find_key(info, "watermark"), find_key(info, "tensor_parallel_size")
    drafts = poller.last.get("vllm:spec_decode_num_draft_tokens_total")
    accepted = poller.last.get("vllm:spec_decode_num_accepted_tokens_total")
    summary = {"model": args.model, "watermark": args.watermark, "num_gpus": ng, "speculative_config": spec,
               "samples": samples,
               "rollout_time_s": rollout_time, "train_py_wall_s": wall, "output_tokens": out_tokens,
               "output_tok_s": out_tokens / rollout_time, "response_len_mean": p.get("rollout/response_len/mean"),
               "response_len_max": p.get("rollout/response_len/max"), "truncated_ratio": p.get("rollout/truncated_ratio"),
               "preemptions": poller.last.get("vllm:num_preemptions_total"),
               "generation_tokens_vllm": poller.last.get("vllm:generation_tokens_total"),
               "prompt_tokens_vllm": poller.last.get("vllm:prompt_tokens_total"),
               "spec_draft_tokens": drafts, "spec_accepted_tokens": accepted,
               "spec_acceptance": accepted / drafts if drafts else None}
    evidence = {"server_url": poller.url, "server_info_scheduler_watermark": sched_wm,
                "server_info_tensor_parallel_size": engine_tp, "leftover_pids": leftover}
    print(json.dumps({**summary, "server_info_scheduler_watermark": sched_wm}, indent=1), flush=True)
    vime_git = {k: subprocess.run(["git", *cmd], cwd=repo, capture_output=True, text=True).stdout.strip()
                for k, cmd in (("commit", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"]))}
    vllm_version = subprocess.run([py, "-c", "import vllm; print(vllm.__version__)"], capture_output=True,
                                  text=True, env=env).stdout.strip()
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()),
                      settings={"rollout": rc, "vime": vc, "watermark": args.watermark}, vime=vime_git,
                      vime_vllm=vllm_version, gpu_env=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      train_args=train_args, load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "summary": summary, "evidence": evidence, "perf": p,
               "server_info": poller.server_info}, open(out / "vime_verify.json", "w"), indent=1)
    if sched_wm is not None and abs(float(sched_wm) - args.watermark) > 1e-9:
        sys.exit(f"engine runs watermark {sched_wm}, asked for {args.watermark}")
    if engine_tp is not None and int(engine_tp) != ng:
        sys.exit(f"engine runs tensor parallel {engine_tp}, asked for {ng}")
    if spec and not drafts:
        sys.exit("speculative decoding configured, but the engine reports no draft tokens")


if __name__ == "__main__":
    main()
