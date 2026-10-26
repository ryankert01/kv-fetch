"""H0 checks (EXPERIMENTS.md, "Hybrid models"): the sparse decoding engine on a hybrid model, before
any H3 job. Run with .venv-hybrid.

  CUDA_VISIBLE_DEVICES=1 .venv-hybrid/bin/python scripts/check_hybrid.py configs/hybrid.yaml logits [--model qwen35_4b] [--steps 300]
  CUDA_VISIBLE_DEVICES=1 .venv-hybrid/bin/python scripts/check_hybrid.py configs/hybrid.yaml graphs [--model qwen35_4b] [--steps 400]

logits: teacher-force HF's own greedy continuation (SDPA + DynamicCache, one token per forward) of six
  AIME prompts, forced to different lengths so that slots retire and are refilled mid-decode; every
  method with a budget larger than the context must give HF's next-token logits up to kernel noise,
  at batch 1 and on 4 slots (as check_e3.py does for Qwen3). Tolerance basis: HF's own eager vs SDPA
  paths differ by up to ~6 in max |dlogit| on Qwen3.5 (H0 audit).
graphs: free-running decoding of the six prompts on 4 slots, stopping at a blank line so that slots
  retire and refill at different steps, CUDA graphs on vs off with the same dense-length buckets
  (identical tokens expected), greedy and with the model's sampling settings (presence penalty
  included), for full, quest@512 and snapkv@512; greedy full also reports the first step where each
  job differs from HF's greedy tokens (bf16 near-ties flip after some hundred steps on any kernel change).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import DynamicCache

from kvfetch import sparse_decode as SD
from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, format_prompt, load_model
from kvfetch.tasks.sweep import LOADERS
from check_e3 import make_engine, thinking


def hybrid_methods(cfg, budget):
    h3 = cfg["hybrid"]["h3"]
    return [SD.Method("quest", budget), SD.Method("ours", budget, spec={"fuse": "sum", "history_weight": 0.0,
                                                                    "history": "last", **cfg["hybrid"]["h1"]["ours"]}),
            SD.Method("snapkv", budget, window=h3["snapkv_window"], protect_prompt=h3["snapkv_protect_prompt"]),
            SD.Method("oracle", budget)]


def prompts_and_refs(cfg, mkey, device, steps):
    name = cfg["hybrid"]["models"][mkey]["name"]
    model, tok = load_model(name, cfg, device, attn_implementation=SD.SPARSE_ATTN)
    ref, _ = load_model(name, cfg, device)
    samples = LOADERS["aime"](tok, cfg["sweep"]["tasks"]["aime24"], 0, 12)[::2]
    prompts = [encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"] for s in samples[:6]]
    refs, ref_logits = [], []
    with torch.inference_mode():
        for p in prompts:
            cache, toks, lg = DynamicCache(config=ref.config), [], []
            out = ref(input_ids=torch.tensor([p], device=device), past_key_values=cache, use_cache=True)
            for _ in range(steps):
                lg.append(out.logits[0, -1].float().cpu())
                toks.append(int(lg[-1].argmax()))
                out = ref(input_ids=torch.tensor([[toks[-1]]], device=device), past_key_values=cache, use_cache=True)
            refs.append(toks)
            ref_logits.append(torch.stack(lg))
    del ref
    torch.cuda.empty_cache()
    print(f"model {name}", flush=True)
    return model, prompts, refs, ref_logits, tok


def check_logits(cfg, args, device):
    model, prompts, refs, ref_logits, _ = prompts_and_refs(cfg, args.model, device, args.steps)
    refs = [r[:args.steps - 37 * i] for i, r in enumerate(refs)]                     # different lengths: refills
    max_len = max(map(len, prompts)) + args.steps + 1
    greedy = {"temperature": 0, "top_k": 1, "top_p": 1.0}
    ps = cfg["definitions"]["page_size"]
    big = ps * (max_len // ps + 1)
    for method, batch in [(SD.Method("full"), 1), (SD.Method("full"), 4)] + [(m, 4) for m in hybrid_methods(cfg, big)]:
        eng = make_engine(model, cfg, method, batch, max_len, device)
        lo = {}
        jobs = [SD.Job(key=(i,), prompt_ids=p, seed=[i], forced=refs[i]) for i, p in enumerate(prompts)]
        SD.run_batch(model, eng, jobs, stop_ids=[], max_new_tokens=args.steps, sampling=greedy,
                     on_done=lambda *a: None, logits_out=lo)
        cols = []
        for i in range(len(prompts)):
            a = torch.stack(lo[(i,)])
            b = ref_logits[i][:len(a)]
            cols.append(f"{float((a - b).abs().mean()):.4f}/{float((a - b).abs().max()):.2f}/"
                        f"{float((a.argmax(-1) == b.argmax(-1)).float().mean()):.3f}")
        print(f"{method.label:14s} batch {batch}  mean|dlogit|/max/argmax agree: " + "  ".join(cols), flush=True)
        del eng
        torch.cuda.empty_cache()


def check_graphs(cfg, args, device):
    model, prompts, refs, _, tok = prompts_and_refs(cfg, args.model, device, args.steps)
    max_len = max(map(len, prompts)) + args.steps + 1
    greedy = {"temperature": 0, "top_k": 1, "top_p": 1.0}
    sampled = cfg["hybrid"]["models"][args.model]["sampling"]
    stop = tok("\n\n", add_special_tokens=False)["input_ids"]
    h3 = cfg["hybrid"]["h3"]
    methods = [SD.Method("full"), SD.Method("quest", 512),
               SD.Method("snapkv", 512, window=h3["snapkv_window"], protect_prompt=h3["snapkv_protect_prompt"])]
    for method in methods:
        for label, sampling in [("greedy", greedy), ("sampled", sampled)]:
            got = {}
            for graphs in (True, False):            # off: same dense-length buckets as the graphs
                eng = make_engine(model, cfg, method, 4, max_len, device)
                out = {}
                jobs = [SD.Job(key=(i,), prompt_ids=p, seed=[7, i]) for i, p in enumerate(prompts)]
                SD.run_batch(model, eng, jobs, stop_ids=stop, max_new_tokens=args.steps, sampling=sampling, graphs=graphs,
                             lc_unit=1024, on_done=lambda job, toks, stopped: out.__setitem__(job.key[0], toks))
                got[graphs] = out
                del eng
                torch.cuda.empty_cache()
            diff = [next((j for j in range(min(len(a), len(b))) if a[j] != b[j]),
                         None if len(a) == len(b) else min(len(a), len(b)))
                    for a, b in ((got[True][i], got[False][i]) for i in range(len(prompts)))]
            line = (f"{method.label:12s} {label}: lengths {[len(got[True][i]) for i in range(len(prompts))]}; "
                    f"graphs on vs off first diff {diff}")
            if label == "greedy" and method.name == "full":
                line += "; vs HF greedy first diff " + str([next((j for j in range(len(got[True][i])) if got[True][i][j] != refs[i][j]), None)
                                                            for i in range(len(prompts))])
            print(line, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("check", choices=["logits", "graphs"])
    p.add_argument("--model", default="qwen35_4b")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = thinking(config_from_args(args))
    SD.register_sparse_attention()
    {"logits": check_logits, "graphs": check_graphs}[args.check](cfg, args, args.device)


if __name__ == "__main__":
    main()
