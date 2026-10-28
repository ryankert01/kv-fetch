"""GDN chunked prefill vs fused recurrent update for T tokens of one sequence (vLLM 0.32's own kernels).

Times `layers` back-to-back calls (one per GDN layer) with a sync at the end: wall time includes the
host-side launch cost that dominates small prefills, CUDA-event time is the GPU timeline over the same
calls. Both paths start from the same fp32 state; their outputs and final states are compared.

Usage (venv-vllm32, one GPU): gdn_prefill_micro.py CONFIG OUT.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gc = cfg["profile"]["gdn_micro"]
    import torch
    import torch.nn.functional as F
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule, fused_sigmoid_gating_delta_rule_update

    torch.manual_seed(gc["seed"])
    dev = torch.device("cuda")
    H, HV, D = gc["heads_k"], gc["heads_v"], gc["head_dim"]
    A_log = torch.randn(HV, device=dev).abs().log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    rows = []
    for T in gc["tokens"]:
        q = F.normalize(torch.randn(1, T, H, D, device=dev), dim=-1).bfloat16()
        k = F.normalize(torch.randn(1, T, H, D, device=dev), dim=-1).bfloat16()
        v = torch.randn(1, T, HV, D, device=dev).bfloat16()
        a = torch.randn(T, HV, device=dev).bfloat16()
        b = torch.randn(T, HV, device=dev).bfloat16()
        g = (-A_log.exp() * F.softplus(a.float() + dt_bias)).unsqueeze(0)      # the gating the fused kernel applies
        beta = b.float().sigmoid().unsqueeze(0).bfloat16()
        h0 = torch.randn(1, HV, D, D, device=dev) * 0.05
        cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
        idx = torch.zeros(1, dtype=torch.int32, device=dev)

        def chunk():
            return chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0.clone(), output_final_state=True,
                                          cu_seqlens=cu, use_qk_l2norm_in_kernel=False)

        state = h0.clone()

        def recurrent():
            state.copy_(h0)
            return fused_sigmoid_gating_delta_rule_update(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
                                                          initial_state=state, inplace_final_state=True, cu_seqlens=cu,
                                                          ssm_state_indices=idx, use_qk_l2norm_in_kernel=False)

        oc, hc = chunk()
        orr, _ = recurrent()
        o_err = ((oc.float() - orr.float().reshape(oc.shape)).norm() / oc.float().norm()).item()
        h_err = ((hc.float() - state.float()).norm() / hc.float().norm()).item()
        row = {"tokens": T, "out_rel_err": o_err, "state_rel_err": h_err}
        for name, fn in (("chunk", chunk), ("recurrent", recurrent)):
            for _ in range(gc["warmup"]):
                fn()
            torch.cuda.synchronize()
            walls, gpus = [], []
            for _ in range(gc["repeats"]):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t = time.perf_counter()
                e0.record()
                for _ in range(gc["layers"]):
                    fn()
                e1.record()
                torch.cuda.synchronize()
                walls.append((time.perf_counter() - t) * 1e3)
                gpus.append(e0.elapsed_time(e1))
            row[f"{name}_wall_ms"] = statistics.median(walls)
            row[f"{name}_gpu_ms"] = statistics.median(gpus)
        rows.append(row)
        print(f"T={T:5d}: chunk {row['chunk_wall_ms']:7.2f} ms wall / {row['chunk_gpu_ms']:7.2f} gpu | recurrent "
              f"{row['recurrent_wall_ms']:7.2f} / {row['recurrent_gpu_ms']:7.2f} | rel err out {o_err:.1e} state {h_err:.1e}", flush=True)
    prov = provenance(cfg, model=None, context_length=None, script=str(Path(__file__).resolve()), settings=gc,
                      cuda=torch.version.cuda)
    json.dump({"provenance": prov, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
