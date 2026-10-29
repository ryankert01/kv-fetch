"""Kill test R2, step 2: real GRPO updates, then how much a cache built under the old weights distorts the continuation.

Updates (config `stale.update`): from theta_0, GRPO steps on the update rollouts of scripts/rl/stale_rollouts.py,
only groups with mixed rewards, group-normalized advantages, token-level mean of the clipped surrogate (clip-higher)
with theta_0 as the old policy, AdamW with fp32 master weights and optimizer state on the host and the bf16 copy on
the GPU, as a trainer that syncs bf16 weights to the rollout engine. The log-softmax over the vocabulary runs in
chunks on a detached hidden state, so a long sequence's logits are never all alive at once.

Measurement (`stale.measure`), after each step in at_steps, for every probe long enough for a cut T: the prefix is
prompt + T response tokens, the continuation the next `continuation` sampled tokens, teacher-forced. A cache is
built over prefix[:-1] by theta_0 or theta_k, then a model runs [prefix[-1]] + continuation[:-1] on a copy of it,
giving the next-token distribution at every continuation position:
    F  theta_k on theta_k's cache   (recompute after the update)
    S  theta_k on theta_0's cache   (keep everything)
    S_state / S_kv (hybrid only)    theta_k's cache with the linear-attention layers' (conv + recurrent) state, or
                                    the attention layers' KV, taken from theta_0's cache
    P  theta_0 on theta_0's cache   (the old policy)
    floor  theta_k over prefix + continuation in one pass (same weights as F, other kernel path)
Per condition X: KL(F || X) per position (mean over position bins) and |log F(y) - log X(y)| of the sampled token.

Usage (.venv-hybrid, one GPU): stale_state.py CONFIG OUT_DIR --model KEY --rollouts ROLLOUTS.json [--set k=v]
Writes OUT_DIR/stale_<model>.json.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def load(name: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, local_files_only=True).cuda()


def padded(ids: list[int], mult: int, pad: int) -> torch.Tensor:
    n = -(-len(ids) // mult) * mult
    return torch.tensor([ids + [pad] * (n - len(ids))], device="cuda")


def token_logprobs(model, h: torch.Tensor, ids: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """log p(ids[t]) for t in [lo + 1, hi + 1) from hidden states h[:, lo:hi]"""
    return model.lm_head(h[:, lo:hi]).float().log_softmax(-1).gather(-1, ids[:, lo + 1:hi + 1, None])[0, :, 0]


def grpo_step(model, ref, batch: list[dict], uc: dict, pad: int) -> dict:
    n_tok = sum(len(s["token_ids"]) for s in batch)
    stats = {"tokens": n_tok, "ratio_mean": 0.0, "clipped": 0, "loss": 0.0}
    for s in batch:
        P, L = len(s["prompt"]), len(s["prompt"]) + len(s["token_ids"])
        ids = padded(s["prompt"] + s["token_ids"], uc["pad_to"], pad)
        with torch.no_grad():
            hr = ref.model(input_ids=ids, use_cache=False).last_hidden_state
            old = torch.cat([token_logprobs(ref, hr, ids, a, min(a + uc["logit_chunk"], L - 1))
                             for a in range(P - 1, L - 1, uc["logit_chunk"])])
            del hr
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state
        hd = h.detach().requires_grad_()
        for a in range(P - 1, L - 1, uc["logit_chunk"]):
            b = min(a + uc["logit_chunk"], L - 1)
            lp = token_logprobs(model, hd, ids, a, b)
            ratio = (lp - old[a - P + 1:b - P + 1]).exp()
            A = s["advantage"]
            surr = torch.minimum(ratio * A, ratio.clamp(1 - uc["clip_low"], 1 + uc["clip_high"]) * A)
            loss = -surr.sum() / n_tok
            loss.backward()
            stats["loss"] += loss.item()
            stats["ratio_mean"] += ratio.sum().item() / n_tok
            stats["clipped"] += int(((ratio < 1 - uc["clip_low"]) | (ratio > 1 + uc["clip_high"])).sum())
        h.backward(hd.grad)
        del h, hd
    return stats


@torch.no_grad()
def build_cache(model, prefix: torch.Tensor):
    return model(input_ids=prefix, use_cache=True, logits_to_keep=1).past_key_values


@torch.no_grad()
def cont_logprobs(model, cache, inp: torch.Tensor) -> torch.Tensor:
    return model(input_ids=inp, past_key_values=copy.deepcopy(cache), use_cache=True).logits[0].float().log_softmax(-1)


def mixed(base, donor, linear: bool):
    """copy of `base` with the linear-attention layers (linear=True) or the attention layers taken from `donor`"""
    c = copy.deepcopy(base)
    for i, layer in enumerate(donor.layers):
        if (type(layer).__name__ == "LinearAttentionLayer") == linear:
            c.layers[i] = copy.deepcopy(layer)
    return c


def compare(F: torch.Tensor, X: torch.Tensor, y: torch.Tensor, bins: list[int]) -> dict:
    kl = (F.exp() * (F - X)).sum(-1)
    lr = (F.gather(-1, y[:, None]) - X.gather(-1, y[:, None]))[:, 0].abs()
    out, lo = {}, 0
    for hi in bins:
        out[f"kl_{lo + 1}_{hi}"] = kl[lo:hi].mean().item()
        lo = hi
    out["kl"] = kl.mean().item()
    out["logratio"] = lr.mean().item()
    return out


def measure(model, ref, probes: list[dict], mc_: dict, hybrid: bool) -> list[dict]:
    rows = []
    K = mc_["continuation"]
    for T in mc_["cuts"]:
        long_enough = [p for p in probes if len(p["token_ids"]) >= T + K][:mc_["max_probes_per_cut"]]
        for p in long_enough:
            prefix = p["prompt"] + p["token_ids"][:T]
            cont = p["token_ids"][T:T + K]
            pre = torch.tensor([prefix[:-1]], device="cuda")
            inp = torch.tensor([prefix[-1:] + cont[:-1]], device="cuda")
            y = torch.tensor(cont, device="cuda")
            c0, ck = build_cache(ref, pre), build_cache(model, pre)
            F = cont_logprobs(model, ck, inp)
            row = {"cut": T, "probe": p["id"]}
            row["S"] = compare(F, cont_logprobs(model, c0, inp), y, mc_["bins"])
            row["P"] = compare(F, cont_logprobs(ref, c0, inp), y, mc_["bins"])
            if hybrid:
                row["S_state"] = compare(F, cont_logprobs(model, mixed(ck, c0, True), inp), y, mc_["bins"])
                row["S_kv"] = compare(F, cont_logprobs(model, mixed(ck, c0, False), inp), y, mc_["bins"])
            with torch.no_grad():
                one = model(input_ids=torch.tensor([prefix + cont[:-1]], device="cuda"), logits_to_keep=K).logits[0]
            row["floor"] = compare(F, one.float().log_softmax(-1), y, mc_["bins"])
            rows.append(row)
            del c0, ck, F, one
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, sc = cfg["profile"], cfg["stale"]
    uc, mc_ = sc["update"], sc["measure"]
    mcfg = pc["models"][args.model]
    hybrid = mcfg["kind"] == "hybrid"
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["rollout"]["seed"])
    torch.set_num_threads(int(os.environ.get("STALE_CPU_THREADS", "32")))

    ro = json.load(open(args.rollouts))
    groups = []
    for g in ro["update_rollouts"]:
        r = torch.tensor([s["reward"] for s in g["samples"]])
        if 0 < r.sum() < len(r):
            adv = (r - r.mean()) / (r.std() + 1e-6)
            groups.append([{"prompt": g["prompt_token_ids"], "token_ids": s["token_ids"], "advantage": a.item()}
                           for s, a in zip(g["samples"], adv)])
    probes = [{"id": f"{g['unique_id']}#{j}", "prompt": g["prompt_token_ids"], "token_ids": s["token_ids"]}
              for g in ro["probes"] for j, s in enumerate(g["samples"])]
    print(f"{len(groups)} mixed groups, {len(probes)} probes", flush=True)

    ref = load(mcfg["name"]).eval()
    model = load(mcfg["name"])
    model.gradient_checkpointing_enable()
    pad = model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else model.config.eos_token_id[0]
    params = [p for p in model.parameters() if p.requires_grad]
    master = [p.detach().float().cpu() for p in params]
    opt = torch.optim.AdamW(master, lr=uc["lr"], betas=tuple(uc["betas"]), eps=uc["eps"], weight_decay=uc["weight_decay"])

    steps, results = [], []
    t0 = time.perf_counter()
    for k in range(1, uc["steps"] + 1):
        batch = [s for i in range(uc["groups_per_step"]) for s in groups[((k - 1) * uc["groups_per_step"] + i) % len(groups)]]
        model.train()
        st = grpo_step(model, ref, batch, uc, pad)
        st["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, uc["grad_clip"]).item()
        for p, m in zip(params, master):
            m.grad = p.grad.float().cpu()
            p.grad = None
        opt.step()
        with torch.no_grad():
            changed = total = 0
            for p, m, p0 in zip(params, master, ref.parameters()):
                p.copy_(m.to(torch.bfloat16))
                changed += int((p != p0).sum())
                total += p.numel()
        st.update({"step": k, "bf16_changed_frac": changed / total, "t_s": time.perf_counter() - t0})
        print(json.dumps(st), flush=True)
        steps.append(st)
        if k in mc_["at_steps"]:
            model.eval()
            torch.cuda.empty_cache()
            rows = measure(model, ref, probes, mc_, hybrid)
            results.append({"step": k, "rows": rows})
            for T in mc_["cuts"]:
                rs = [r for r in rows if r["cut"] == T]
                if rs:
                    mean = {c: sum(r[c]["kl"] for r in rs) / len(rs) for c in rs[0] if c not in ("cut", "probe")}
                    print(f"step {k} cut {T} ({len(rs)} probes): " + "  ".join(f"KL(F||{c}) {v:.3g}" for c, v in mean.items())
                          + f"  rho {mean['S'] / mean['P']:.3g}", flush=True)
    prov = provenance(cfg, model=mcfg["name"], context_length=None, script=str(Path(__file__).resolve()), settings=sc,
                      rollouts=str(Path(args.rollouts).resolve()), torch=torch.__version__, load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "groups_used": len(groups), "probes": len(probes), "steps": steps, "results": results},
              open(out / f"stale_{args.model}.json", "w"))


if __name__ == "__main__":
    main()
