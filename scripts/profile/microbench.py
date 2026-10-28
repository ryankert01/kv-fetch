"""P3: measured hardware peaks on this GPU (the roofline's ceilings) and model weight bytes.

Peaks: device-to-device copy bandwidth (read + write bytes / time), read-only bandwidth (a sum over a
buffer), bf16 GEMM throughput (square matmuls) and skinny bf16 GEMMs at decode shapes (M = batch rows
against a weight matrix), each the best of `repeats` timed runs after warm-up. Buffers and the skinny
GEMMs' rotating weight sets are several times the GPU's L2 so the rates are DRAM rates. Weight bytes per module class come from the safetensors headers
(no weights loaded), for the bytes a decode step must read.

Usage (venv-vllm32): microbench.py CONFIG OUT.json
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "U8": 1, "I32": 4, "F64": 8, "I64": 8}
WEIGHT_CLASSES = [   # first match wins, on the tensor name
    ("vision", r"visual|vision|mm_projector|merger"),
    ("mtp", r"(^|\.)mtp\."),
    ("embed", r"embed_tokens"),
    ("lm_head", r"lm_head"),
    ("gdn", r"linear_attn"),
    ("attention", r"self_attn"),
    ("moe_experts", r"experts"),
    ("mlp", r"mlp"),
    ("norm", r"norm"),
]


def weight_bytes(model_dir: Path) -> dict:
    out = Counter()
    for f in sorted(model_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, t in header.items():
            if name == "__metadata__":
                continue
            size = DTYPE_BYTES[t["dtype"]]
            for d in t["shape"]:
                size *= d
            out[next((c for c, p in WEIGHT_CLASSES if re.search(p, name)), "other")] += size
    return dict(out)


def best_ms(fn, repeats: int, warmup: int = 3) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mb = cfg["profile"]["microbench"]
    import torch
    from huggingface_hub import snapshot_download

    dev = torch.device("cuda")
    res = {"copy": [], "read": [], "gemm": [], "skinny_gemm": [], "weights": {}}
    for mib in mb["copy_MiB"]:
        x = torch.empty(mib << 20, dtype=torch.uint8, device=dev)
        y = torch.empty_like(x)
        ms = best_ms(lambda: y.copy_(x), mb["repeats"])
        res["copy"].append({"MiB": mib, "ms": ms, "GBps": 2 * (mib << 20) / ms / 1e6})
        z = x.view(torch.float32)
        ms = best_ms(lambda: z.sum(), mb["repeats"])
        res["read"].append({"MiB": mib, "ms": ms, "GBps": (mib << 20) / ms / 1e6})
        del x, y, z
    for n in mb["gemm_sizes"]:
        a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
        b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
        ms = best_ms(lambda: a @ b, mb["repeats"])
        res["gemm"].append({"n": n, "ms": ms, "TFLOPs": 2 * n ** 3 / ms / 1e9})
        del a, b
    for k, n in mb["skinny_kn"]:
        count = -(-(mb["skinny_set_MiB"] << 20) // (n * k * 2))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device=dev) for _ in range(count)]
        for m in mb["skinny_m"]:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
            ms = best_ms(lambda: [torch.nn.functional.linear(x, w) for w in ws], mb["repeats"]) / count
            res["skinny_gemm"].append({"m": m, "k": k, "n": n, "ms": ms, "weights_in_set": count,
                                       "weight_GBps": n * k * 2 / ms / 1e6, "TFLOPs": 2 * m * n * k / ms / 1e9})
        del ws
    for key, m in cfg["profile"]["models"].items():
        d = Path(snapshot_download(m["name"], local_files_only=True))
        res["weights"][key] = weight_bytes(d)
    for r in res["copy"] + res["read"]:
        print(f"{'copy' if r in res['copy'] else 'read'} {r['MiB']} MiB: {r['GBps']:.0f} GB/s")
    for r in res["skinny_gemm"]:
        print(f"skinny m={r['m']} {r['k']}x{r['n']}: {r['ms'] * 1e3:.1f} us, {r['weight_GBps']:.0f} GB/s, {r['TFLOPs']:.1f} TFLOP/s")
    for r in res["gemm"]:
        print(f"gemm {r['n']}: {r['TFLOPs']:.1f} TFLOP/s")
    for key, w in res["weights"].items():
        print(key, {c: f"{v / 2**30:.2f} GiB" for c, v in w.items()})
    prov = provenance(cfg, model=None, context_length=None, script=str(Path(__file__).resolve()), settings=mb,
                      cuda=torch.version.cuda)
    json.dump({"provenance": prov, **res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
