"""RULER needle-in-a-haystack generator (re-implementation of NVIDIA/RULER niah.py).

Differences from upstream, all intentional and small:
  * sentences are split with a regex rather than nltk.sent_tokenize;
  * keys are drawn as random adjective + random noun (upstream draws from the sorted
    set of all adj-noun pairs: the same distribution up to rare collisions);
  * each sample uses its own seeded RNG so samples are reproducible independently.
"""
from __future__ import annotations

import random
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from kvfetch.models import encode, format_prompt

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Sample:
    task: str
    index: int
    prompt: str                 # fully formatted (chat template + answer prefix)
    answers: list[str]
    needles: list[str] = field(default_factory=list)  # verbatim inserted needle sentences
    query_needles: list[str] = field(default_factory=list)  # needles holding the queried values
    extra: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def _essay_words(path: str) -> tuple[str, ...]:
    from datasets import Dataset

    text = " ".join(Dataset.from_file(path)["text"])
    return tuple(re.sub(r"\s+", " ", text).split(" "))


@lru_cache(maxsize=None)
def _wordlists() -> tuple[tuple[str, ...], tuple[str, ...]]:
    from wonderwords import random_word

    nouns = random_word._get_words_from_text_file("nounlist.txt")
    adjs = random_word._get_words_from_text_file("adjectivelist.txt")
    return tuple(sorted(set(adjs))), tuple(sorted(set(nouns)))


def string_match_all(pred: str, refs: list[str]) -> float:
    pred = re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip().lower()
    return sum(r.lower() in pred for r in refs) / len(refs)


class RulerNiah:
    def __init__(self, task: str, cfg: dict):
        self.task = task
        self.rc = cfg["ruler"]
        self.tc = self.rc["tasks"][task]
        self.cfg = cfg
        self.depths = list(np.round(np.linspace(0, 100, num=self.rc["num_depths"], endpoint=True)).astype(int))
        if self.tc["type_haystack"] == "essay":
            self.essay = _essay_words(cfg["data"]["paul_graham_essays"])
        self.adjs, self.nouns = _wordlists()

    # -- random atoms -------------------------------------------------------
    def _rand(self, kind: str, rng: random.Random) -> str:
        if kind == "numbers":
            d = self.rc["number_digits"]
            return str(rng.randint(10 ** (d - 1), 10 ** d - 1))
        if kind == "words":
            return f"{rng.choice(self.adjs)}-{rng.choice(self.nouns)}"
        if kind == "uuids":
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))
        raise ValueError(kind)

    # -- one sample, fixed haystack size ------------------------------------
    def build(self, num_haystack: int, rng: random.Random) -> tuple[str, str, list[str], list[str], list[str]]:
        tc, rc = self.tc, self.rc
        nk = max(tc["num_needle_k"], tc["num_needle_q"])
        tv = tc["type_needle_v"]
        keys, values, needles, needle_of = [], [], [], {}
        for _ in range(nk):
            keys.append(self._rand(tc["type_needle_k"], rng))
            vals = []
            for _ in range(tc["num_needle_v"]):
                vals.append(self._rand(tv, rng))
                needles.append(rc["needle"].format(type_needle_v=tv, key=keys[-1], value=vals[-1]))
                needle_of[(len(keys) - 1, len(vals) - 1)] = needles[-1]
            values.append(vals)
        random.Random(rc["shuffle_seed"]).shuffle(needles)

        hay = tc["type_haystack"]
        if hay == "essay":
            words = self.essay
            if num_haystack > len(words):
                words = words * ((num_haystack + len(words) - 1) // len(words))
            sents = _SENT_SPLIT.split(" ".join(words[:num_haystack]).strip())
            pos = [0] + sorted(int(len(sents) * (d / 100)) for d in rng.sample(self.depths, len(needles))) + [len(sents)]
            parts = []
            for i in range(1, len(pos)):
                parts.append(" ".join(sents[pos[i - 1]:pos[i]]))
                if i - 1 < len(needles):
                    parts.append(needles[i - 1])
            context = " ".join(parts)
        else:
            if hay == "noise":
                sentences = [rc["noise_sentence"]] * num_haystack
            elif hay == "needle":
                sentences = [rc["needle"].format(type_needle_v=tv, key=self._rand(tc["type_needle_k"], rng),
                                                 value=self._rand(tv, rng)) for _ in range(num_haystack)]
            else:
                raise ValueError(hay)
            for idx, el in zip(sorted(rng.sample(range(num_haystack), len(needles)), reverse=True), needles):
                sentences.insert(idx, el)
            context = "\n".join(sentences)

        qidx = rng.sample(range(nk), tc["num_needle_q"])
        queries = [keys[i] for i in qidx]
        answers = [a for i in qidx for a in values[i]]
        query_needles = [needle_of[(i, j)] for i in qidx for j in range(len(values[i]))]
        query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]

        template, prefix, tvs = rc["template"], rc["answer_prefix"], tv
        if tc["num_needle_q"] * tc["num_needle_v"] == 1:
            def singular(s: str) -> str:
                return (s.replace("Some", "A").replace("are all", "is").replace("are", "is")
                        .replace("answers", "answer"))
            template, prefix, tvs = singular(template), singular(prefix), tv[:-1]
        user = template.format(type_needle_v=tvs, context=context, query=query)
        return user, prefix.format(type_needle_v=tvs, query=query), answers, needles, query_needles

    # -- length fitting (upstream: binary search, then per-sample back-off) --
    def _n_tokens(self, tok, user: str, prefix: str) -> int:
        return len(encode(tok, format_prompt(tok, user, prefix, self.cfg))["input_ids"])

    def fit_haystack(self, tok, max_seq_length: int, tokens_to_generate: int, seed: int) -> int:
        inc = self.rc["incremental"][self.tc["type_haystack"]]
        if self.tc["type_haystack"] != "essay" and max_seq_length < 4096:
            inc = 5
        u, p, *_ = self.build(inc, random.Random(seed))
        per = self._n_tokens(tok, u, p) / inc
        lo, hi, best = inc, max(int(max_seq_length / per * 3), inc * 2), None
        while lo <= hi:
            mid = (lo + hi) // 2
            u, p, *_ = self.build(mid, random.Random(seed))
            if self._n_tokens(tok, u, p) + tokens_to_generate <= max_seq_length:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return best if best is not None else inc

    def samples(self, tok, *, max_seq_length: int, num_samples: int, seed: int,
                tokens_to_generate: int | None = None) -> list[Sample]:
        ttg = self.rc["tokens_to_generate"] if tokens_to_generate is None else tokens_to_generate
        inc = self.rc["incremental"][self.tc["type_haystack"]]
        n_hay = self.fit_haystack(tok, max_seq_length, ttg, seed)
        out = []
        for i in range(num_samples):
            used = n_hay
            while True:
                rng = random.Random(f"{seed}:{self.task}:{max_seq_length}:{i}")
                user, prefix, answers, needles, qn = self.build(used, rng)
                prompt = format_prompt(tok, user, prefix, self.cfg)
                n = len(encode(tok, prompt)["input_ids"])
                if n + ttg <= max_seq_length or used <= inc:
                    break
                used -= inc
            out.append(Sample(task=self.task, index=i, prompt=prompt, answers=answers, needles=needles,
                              query_needles=qn, extra={"num_haystack": used, "prompt_tokens": n}))
        return out


def char_spans_to_token_spans(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Map [start, end) char spans to [start, end) token spans using tokenizer offsets."""
    starts = np.array([o[0] for o in offsets])
    ends = np.array([o[1] for o in offsets])
    out = []
    for a, b in spans:
        ti = int(np.searchsorted(ends, a, side="right"))
        tj = int(np.searchsorted(starts, b, side="left"))
        out.append((ti, max(tj, ti + 1)))
    return out


def find_spans(text: str, needles: list[str]) -> list[tuple[int, int]]:
    spans = []
    for nd in needles:
        i = text.find(nd)
        if i < 0:
            raise ValueError(f"needle not found: {nd[:60]}")
        spans.append((i, i + len(nd)))
    return spans
