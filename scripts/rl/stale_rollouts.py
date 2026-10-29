"""Kill test R2, step 1: sample the update rollouts and the probe rollouts under the released weights (theta_0).

Update rollouts (config `stale.update_rollouts`): MATH problems of the given levels, n samples each at the RL
temperature, scored 1 if the last \\boxed{} matches the reference answer after normalization. They feed the
GRPO steps in scripts/rl/stale_state.py. Probes (`stale.probes`): harder problems (R1's levels, R1's 64 skipped),
long samples whose prefixes are cut and continued under updated weights.

Usage (venv-vllm32, one GPU): stale_rollouts.py CONFIG OUT_DIR --model KEY [--set k=v]
Writes OUT_DIR/rollouts_<model>.json.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def problems(rc: dict, levels: list[int], skip: int, count: int) -> list[dict]:
    pc = rc["prompts"]
    rows = [json.loads(line) for line in open(pc["jsonl"])]
    rows = [r for r in rows if r.get("level") in levels]
    random.Random(rc["seed"]).shuffle(rows)
    return rows[skip:skip + count]


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def normalize(ans: str) -> str:
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", ans)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("dfrac", "frac"), ("tfrac", "frac"),
                 ("^\\circ", ""), ("^{\\circ}", ""), ("\\$", ""), ("$", ""), (" ", "")):
        s = s.replace(a, b)
    return s.rstrip(".")


def reward(text: str, answer: str) -> float:
    got = last_boxed(text)
    if got is None:
        return 0.0
    a, b = normalize(got), normalize(answer)
    if a == b:
        return 1.0
    try:
        return float(abs(float(a) - float(b)) < 1e-6)
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, rc, sc = cfg["profile"], cfg["rollout"], cfg["stale"]
    mc = pc["models"][args.model]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=mc["name"], **{**pc["engine"], **mc.get("engine", {}), **sc["engine"]})
    tok = llm.get_tokenizer()
    instr = rc["prompts"]["instruction"]
    enc = lambda p: tok.encode(tok.apply_chat_template([{"role": "user", "content": f"{p[rc['prompts']['field']]}\n{instr}"}],
                                                       tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    res, summary = {}, {}
    for part, skip in (("update_rollouts", 0), ("probes", sc["probes"]["skip"])):
        c = sc[part]
        probs = problems(rc, c["levels"], skip, c["num_prompts"])
        prompts = [enc(p) for p in probs]
        sp = SamplingParams(n=c["n"], temperature=rc["temperature"], top_p=rc["top_p"], max_tokens=c["max_tokens"], seed=rc["seed"])
        t = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
        wall = time.perf_counter() - t
        groups = []
        for prob, p, o in zip(probs, prompts, outs):
            samples = [{"token_ids": list(s.token_ids), "finish": s.finish_reason,
                        "reward": reward(s.text, prob["answer"])} for s in o.outputs]
            groups.append({"unique_id": prob["unique_id"], "level": prob["level"], "answer": prob["answer"],
                           "prompt_token_ids": p, "samples": samples})
        lens = [len(s["token_ids"]) for g in groups for s in g["samples"]]
        rw = [[s["reward"] for s in g["samples"]] for g in groups]
        summary[part] = {"groups": len(groups), "samples": len(lens), "output_mean": sum(lens) / len(lens),
                         "output_max": max(lens), "capped": sum(s["finish"] == "length" for g in groups for s in g["samples"]),
                         "reward_mean": sum(map(sum, rw)) / len(lens), "mixed_groups": sum(0 < sum(r) < len(r) for r in rw),
                         "wall_s": wall}
        res[part] = groups
    print(json.dumps(summary, indent=1), flush=True)
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()), settings=sc,
                      rollout=rc, vllm=vllm.__version__, load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "summary": summary, **res}, open(out / f"rollouts_{args.model}.json", "w"))


if __name__ == "__main__":
    main()
