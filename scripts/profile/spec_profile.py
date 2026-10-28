"""Speculative decoding on a hybrid vs a dense model: what verifying k draft tokens costs per engine step.

One engine per (model, k): the model's own drafter (Qwen3.5: its MTP layer; Qwen3-4B: an EAGLE3 head) with
k = num_speculative_tokens, k = 0 without speculation. For each batch size b, b prompts (real text, the
config's parquet column, chat template applied) generate `output` tokens each (ignore_eos, greedy).
Per point: wall time, engine steps (`vllm:iteration_tokens_total` count), ms per step, output tokens per
step, draft acceptance (accepted / draft tokens) and mean tokens per draft. The hybrid-specific question is
the step cost at k against k = 0: every verify step runs k + 1 tokens per sequence through the GDN layers
and keeps the states needed to roll back rejected ones.

Usage (venv-vllm32, one GPU): spec_profile.py CONFIG OUT_DIR --model KEY --k K [--set k=v]
Writes OUT_DIR/<model>_k<K>.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402

NAMES = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")


def counters(llm) -> dict[str, float]:
    res = {}
    for m in llm.get_metrics():
        if m.name in NAMES:
            res[m.name] = res.get(m.name, 0) + m.value
        elif m.name == "vllm:iteration_tokens_total":
            res["steps"] = res.get("steps", 0) + m.count
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, sc = cfg["profile"], cfg["profile"]["spec"]
    mc = pc["models"][args.model]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    import vllm
    from vllm import LLM, SamplingParams

    spec = None
    if args.k > 0:
        spec = {**sc["drafters"][args.model], "num_speculative_tokens": args.k}
    llm = LLM(model=mc["name"], **{**pc["engine"], **mc.get("engine", {}), **sc["engine"]}, speculative_config=spec,
              disable_log_stats=False)
    tok = llm.get_tokenizer()
    texts = pq.read_table(sc["prompts"]["parquet"]).column(sc["prompts"]["column"]).to_pylist()
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
               for t in texts]
    sp = SamplingParams(max_tokens=sc["output"], ignore_eos=True, temperature=0)
    llm.generate(prompts[:2], SamplingParams(max_tokens=32, ignore_eos=True, temperature=0), use_tqdm=False)

    points = []
    for b in sc["batches"]:
        batch = [prompts[i % len(prompts)] for i in range(b)]
        c0 = counters(llm)
        t = time.perf_counter()
        llm.generate(batch, sp, use_tqdm=False)
        wall = time.perf_counter() - t
        c1 = counters(llm)
        d = {k: c1.get(k, 0) - c0.get(k, 0) for k in set(c0) | set(c1)}
        steps = d.get("steps", 0)
        drafts, drafted, accepted = (d.get(n, 0) for n in NAMES)
        p = {"batch": b, "wall_s": wall, "output_tokens": b * sc["output"], "steps": steps,
             "ms_per_step": wall / steps * 1e3 if steps else None, "tokens_per_step": b * sc["output"] / steps if steps else None,
             "tok_s": b * sc["output"] / wall, "drafts": drafts, "draft_tokens": drafted, "accepted_tokens": accepted,
             "acceptance": accepted / drafted if drafted else None, "tokens_per_draft": 1 + accepted / drafts if drafts else None}
        points.append(p)
        print(f"{args.model} k={args.k} b={b}: {p['tok_s']:.0f} tok/s, {steps} steps x {p['ms_per_step'] or 0:.2f} ms, "
              f"{p['tokens_per_step'] or 0:.1f} tok/step, acceptance {p['acceptance'] or 0:.2f}", flush=True)
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()), settings=sc,
                      vllm=vllm.__version__, k=args.k, speculative_config=spec, load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "points": points}, open(out / f"{args.model}_k{args.k}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
