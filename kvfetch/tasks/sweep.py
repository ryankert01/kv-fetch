"""Dataset loaders for the revival sweep. Each returns SweepSamples: the user turns (the first
carries the context), per-turn reference answers, and whether Qwen3 thinking mode is on.

Context is fitted to ``budget_tokens`` without dropping what the questions need: SCBench KV
keeps every queried key, SCBench RepoQA keeps a window around every target function, other
long documents are truncated in the middle (LongBench practice)."""
from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field

from kvfetch.models import encode

DATA = "/share2/huggingface/datasets"


@dataclass
class SweepSample:
    task: str
    index: int
    turns: list[str]                 # user messages; turns[0] includes the context
    answers: list[list[str]]         # per turn
    think: bool = False
    extra: dict = field(default_factory=dict)


def _arrow(pattern: str):
    from datasets import Dataset

    files = sorted(glob.glob(f"{DATA}/{pattern}", recursive=True))
    if not files:
        raise FileNotFoundError(pattern)
    return Dataset.from_file(files[0])


def truncate_middle(tok, text: str, max_tokens: int) -> str:
    ids = encode(tok, text)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    return tok.decode(ids[:half]) + "\n...\n" + tok.decode(ids[-half:])


def keep_spans(tok, text: str, spans: list[tuple[int, int]], max_tokens: int) -> str:
    """Keep the token ranges covering the char spans, widened evenly by the largest padding
    that fits the budget (binary search: kept length only grows with padding). Gaps between
    kept ranges are marked with '...'."""
    enc = encode(tok, text, offsets=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) <= max_tokens:
        return text
    ranges = []
    for a, b in spans:
        i = next(k for k, o in enumerate(offs) if o[1] > a)
        j = next((k for k in range(i, len(offs)) if offs[k][0] >= b), len(offs))
        ranges.append((i, j))

    def kept(pad: int) -> list[list[int]]:
        return _merge([[max(0, i - pad), min(len(ids), j + pad)] for i, j in ranges])

    if sum(j - i for i, j in kept(0)) > max_tokens:
        raise ValueError("targets alone exceed the budget")
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(j - i for i, j in kept(mid)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return "\n...\n".join(tok.decode(ids[i:j]) for i, j in kept(lo))


def _merge(rs: list[list[int]]) -> list[list[int]]:
    out = []
    for i, j in sorted(rs):
        if out and i <= out[-1][1]:
            out[-1][1] = max(out[-1][1], j)
        else:
            out.append([i, j])
    return out


# ------------------------------------------------------------------ loaders
def math500(tok, sc: dict, budget_tokens: int, n: int) -> list[SweepSample]:
    d = _arrow("HuggingFaceH4___math-500/**/*.arrow")
    out = []
    for r in d:
        if r["level"] != sc["level"]:
            continue
        out.append(SweepSample("math500", len(out), [r["problem"] + sc["instruction"]], [[r["answer"]]],
                               think=True, extra={"unique_id": r["unique_id"]}))
        if len(out) == n:
            break
    return out


def longbench_v2(tok, sc: dict, budget_tokens: int, n: int) -> list[SweepSample]:
    d = _arrow("THUDM___long_bench-v2/**/*.arrow")
    head_tail = sc["template"].format(context="", question="{q}", a="", b="", c="", d="")
    out = []
    for r in d:
        if r["length"] != sc["length"]:
            continue
        overhead = len(encode(tok, head_tail.replace("{q}", r["question"]) + r["choice_A"] + r["choice_B"]
                              + r["choice_C"] + r["choice_D"])["input_ids"])
        ctx = truncate_middle(tok, r["context"], budget_tokens - overhead)
        text = sc["template"].format(context=ctx, question=r["question"], a=r["choice_A"], b=r["choice_B"],
                                     c=r["choice_C"], d=r["choice_D"])
        out.append(SweepSample("longbench_v2", len(out), [text], [[r["answer"]]], think=True,
                               extra={"_id": r["_id"], "domain": r["domain"]}))
        if len(out) == n:
            break
    return out


def scbench_kv(tok, sc: dict, budget_tokens: int, n: int) -> list[SweepSample]:
    d = _arrow("MInference___sc_bench/multi_turn_kv/**/*.arrow")
    out = []
    for r in d:
        pairs = json.loads(r["context"][r["context"].index("{"):])
        keys = [re.search(r'Key: "([^"]+)"', t["input"]).group(1) for t in r["multi_turns"]]
        fillers = [k for k in pairs if k not in set(keys)]
        overhead = len(encode(tok, sc["instruction"] + "".join(t["input"] for t in r["multi_turns"]))["input_ids"])

        def build(k: int) -> str:  # queried keys + the first k other pairs, in original order
            kept = set(keys) | set(fillers[:k])
            return "JSON data:\n" + json.dumps({key: pairs[key] for key in pairs if key in kept})

        lo, hi = 0, len(fillers)  # largest k that fits (token count grows with k)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(encode(tok, build(mid))["input_ids"]) <= budget_tokens - overhead:
                lo = mid
            else:
                hi = mid - 1
        ctx = build(lo)
        turns = [sc["instruction"] + "\n\n" + ctx + "\n" + r["multi_turns"][0]["input"].strip()]
        turns += [t["input"].strip() for t in r["multi_turns"][1:]]
        out.append(SweepSample("scbench_kv", len(out), turns, [[t["answer"]] for t in r["multi_turns"]],
                               extra={"n_pairs": len(keys) + lo}))
        if len(out) == n:
            break
    return out


def scbench_repoqa(tok, sc: dict, budget_tokens: int, n: int) -> list[SweepSample]:
    d = _arrow("microsoft___sc_bench/scbench_repoqa/**/*.arrow")
    out = []
    for r in d:
        if sc.get("languages") and r["lang"] not in sc["languages"]:
            continue
        spans = []
        for t in r["multi_turns"]:
            head = t["answer"].strip().splitlines()[0]
            i = r["context"].find(head)
            if i < 0:
                break
            spans.append((i, i + len(t["answer"])))
        if len(spans) != len(r["multi_turns"]):
            continue  # a target function was not found verbatim
        overhead = len(encode(tok, sc["instruction"] + "".join(t["input"] for t in r["multi_turns"]))["input_ids"])
        try:
            ctx = keep_spans(tok, r["context"], spans, budget_tokens - overhead)
        except ValueError:
            continue
        turns = [sc["instruction"] + "\n\n<repo>\n" + ctx + "\n</repo>\n" + r["multi_turns"][0]["input"].strip()]
        turns += [t["input"].strip() for t in r["multi_turns"][1:]]
        names = [t["answer"].strip().splitlines()[0] for t in r["multi_turns"]]
        out.append(SweepSample("scbench_repoqa", len(out), turns, [[nm] for nm in names],
                               extra={"repo": r["repo"], "lang": r["lang"]}))
        if len(out) == n:
            break
    return out


def qmsum(tok, sc: dict, budget_tokens: int, n: int) -> list[SweepSample]:
    d = _arrow("THUDM___long_bench/qmsum/**/*.arrow")
    out = []
    for r in d:
        overhead = len(encode(tok, sc["template"].format(context="", input=r["input"]))["input_ids"])
        ctx = truncate_middle(tok, r["context"], budget_tokens - overhead)
        out.append(SweepSample("qmsum", len(out), [sc["template"].format(context=ctx, input=r["input"])],
                               [list(r["answers"])], extra={"_id": r["_id"]}))
        if len(out) == n:
            break
    return out


def pg19_summary(tok, sc: dict, budget_tokens: int, n: int) -> list[SweepSample]:
    d = _arrow("emozilla___pg19-test/**/*.arrow")
    overhead = len(encode(tok, sc["template"].format(document=""))["input_ids"])
    room = budget_tokens - overhead
    out = []
    for i, r in enumerate(d):
        ids = encode(tok, r["text"][: room * sc["chars_per_token_upper"]])["input_ids"]
        if len(ids) < room:
            continue  # book shorter than the budget
        doc = tok.decode(ids[:room])
        out.append(SweepSample("pg19_summary", len(out), [sc["template"].format(document=doc)], [[]],
                               extra={"book_index": i, "title": r["short_book_title"]}))
        if len(out) == n:
            break
    return out


LOADERS = {"math500": math500, "longbench_v2": longbench_v2, "scbench_kv": scbench_kv,
           "scbench_repoqa": scbench_repoqa, "qmsum": qmsum, "pg19_summary": pg19_summary}


def score(task: str, turn_texts: list[str], answers: list[list[str]]) -> float | None:
    """Rough per-sample accuracy where a cheap check exists (secondary to the traces)."""
    if task in ("pg19_summary", "qmsum"):
        return None
    hits = []
    for text, refs in zip(turn_texts, answers):
        if not refs:
            continue
        if task == "math500":
            boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
            got = boxed[-1] if boxed else ""
            hits.append(float(re.sub(r"\s", "", got) == re.sub(r"\s", "", refs[0])))
        elif task == "longbench_v2":
            m = re.findall(r"correct answer is \(?([A-D])\)?", text)
            hits.append(float(bool(m) and m[-1] == refs[0]))
        else:
            hits.append(float(any(ref.strip().lower() in text.lower() for ref in refs)))
    return sum(hits) / len(hits) if hits else None
