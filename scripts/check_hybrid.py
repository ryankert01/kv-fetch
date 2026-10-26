"""H0 checks (EXPERIMENTS.md, "Hybrid models"): the sparse decoding engine on a hybrid model, before
any H3 job. Run with .venv-hybrid.

  CUDA_VISIBLE_DEVICES=1 .venv-hybrid/bin/python scripts/check_hybrid.py configs/hybrid.yaml logits [--model qwen35_4b] [--steps 300]
  CUDA_VISIBLE_DEVICES=1 .venv-hybrid/bin/python scripts/check_hybrid.py configs/hybrid.yaml graphs [--model qwen35_4b] [--steps 400]

logits: teacher-force HF's own greedy continuation (SDPA + DynamicCache, one token per forward) of a
  few AIME prompts; every method with a budget larger than the context must give HF's next-token
  logits up to kernel noise, at batch 1 and in a mixed-length batch (as check_e3.py does for Qwen3).
graphs: free-running greedy decoding of the same prompts in a mixed-length batch, CUDA graphs on vs
  off with the same dense-length buckets (identical tokens expected), and the first step where each
  differs from HF's greedy tokens (bf16 near-ties flip after some hundred steps on any kernel change);
  then the same with the model's sampling settings (presence penalty included), graphs on vs off.
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
    samples = LOADERS["aime"](tok, cfg["sweep"]["tasks"]["aime24"], 0, 6)
    prompts = [encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"] for s in samples[::2][:3]]
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
    return model, prompts, refs, ref_logits


def check_logits(cfg, args, device):
    model, prompts, refs, ref_logits = prompts_and_refs(cfg, args.model, device, args.steps)
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
            a, b = torch.stack(lo[(i,)]), ref_logits[i]
            cols.append(f"{float((a - b).abs().mean()):.4f}/{float((a - b).abs().max()):.2f}/"
                        f"{float((a.argmax(-1) == b.argmax(-1)).float().mean()):.3f}")
        print(f"{method.label:14s} batch {batch}  mean|dlogit|/max/argmax agree: " + "  ".join(cols), flush=True)
        del eng
        torch.cuda.empty_cache()


def check_graphs(cfg, args, device):
    model, prompts, refs, _ = prompts_and_refs(cfg, args.model, device, args.steps)
    max_len = max(map(len, prompts)) + args.steps + 1
    greedy = {"temperature": 0, "top_k": 1, "top_p": 1.0}
    sampled = cfg["hybrid"]["models"][args.model]["sampling"]
    for label, sampling in [("greedy", greedy), ("sampled", sampled)]:
        got = {}
        for graphs in (True, False):                # off: same dense-length buckets as the graphs
            eng = make_engine(model, cfg, SD.Method("full"), 4, max_len, device)
            out = {}
            jobs = [SD.Job(key=(i,), prompt_ids=p, seed=[7, i]) for i, p in enumerate(prompts)]
            SD.run_batch(model, eng, jobs, stop_ids=[], max_new_tokens=args.steps, sampling=sampling, graphs=graphs,
                         lc_unit=1024, on_done=lambda job, toks, stopped: out.__setitem__(job.key[0], toks))
            got[graphs] = out
            del eng
            torch.cuda.empty_cache()
        for i in range(len(prompts)):
            a, b = got[True][i], got[False][i]
            same = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
            line = f"{label} prompt {i}: graphs on vs off first diff at {same} (len {len(a)})"
            if label == "greedy":
                hf = next((j for j in range(len(a)) if a[j] != refs[i][j]), None)
                line += f"; vs HF greedy first diff at {hf}"
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
