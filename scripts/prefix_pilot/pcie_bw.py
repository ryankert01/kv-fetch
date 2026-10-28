"""Host<->GPU copy time for one Qwen3.5-4B GDN checkpoint (24 layers x 2,146,304 B), pinned host memory.
Usage: pcie_bw.py OUT.json"""
import json, socket, sys, time
import torch

SETTINGS = {"layer_bytes": 2146304, "layers": 24, "repeats": 50, "batch_checkpoints": [1, 4, 16]}


def timed(fn, reps):
    torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    s = SETTINGS
    rows = []
    for n in s["batch_checkpoints"]:
        nbytes = n * s["layers"] * s["layer_bytes"]
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        per_layer_h = host.view(n * s["layers"], s["layer_bytes"])
        per_layer_d = dev.view(n * s["layers"], s["layer_bytes"])
        h2d = timed(lambda: dev.copy_(host, non_blocking=True), s["repeats"])
        d2h = timed(lambda: host.copy_(dev, non_blocking=True), s["repeats"])
        def per_layer():
            for i in range(per_layer_h.shape[0]):
                per_layer_d[i].copy_(per_layer_h[i], non_blocking=True)
        h2d_layers = timed(per_layer, s["repeats"])
        rows.append({"checkpoints": n, "bytes": nbytes, "h2d_ms": h2d, "d2h_ms": d2h, "h2d_per_layer_copies_ms": h2d_layers,
                     "h2d_GBps": nbytes / h2d / 1e6, "d2h_GBps": nbytes / d2h / 1e6})
        print(rows[-1], flush=True)
    prov = {"settings": s, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump({"provenance": prov, "rows": rows}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
