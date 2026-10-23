"""Chunked prefill + greedy decode, optionally traced and multi-turn. Batch size 1."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from kvfetch.tracing.tracer import AttentionTracer

GEN, FED = 0, 1  # token kinds: chosen by the model / forced (a follow-up user turn)


@dataclass
class GenResult:
    tokens: list[int]
    natural_stop_step: int | None  # first step whose argmax was a stop token (None = never)
    prefill_s: float
    decode_s: float
    kinds: list[int] = field(default_factory=list)   # GEN / FED per token
    turns: list[dict] = field(default_factory=list)  # per assistant turn: start, end (exclusive), natural_stop

    def natural_tokens(self) -> list[int]:
        return self.tokens if self.natural_stop_step is None else self.tokens[:self.natural_stop_step]

    def turn_tokens(self, i: int) -> list[int]:
        """Tokens the model produced in assistant turn i, without its stop token."""
        t = self.turns[i]
        end = t["natural_stop"] if t["natural_stop"] is not None else t["end"]
        return self.tokens[t["start"]:end]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, input_ids: torch.Tensor, *, max_new_tokens: int, stop_ids: list[int],
             suppress_stop: bool, prefill_chunk_size: int, tracer: AttentionTracer | None = None,
             followups: list[list[int]] | None = None, eot_ids: list[int] | None = None,
             turn_max_new: int | None = None, return_cache: bool = False):
    """Greedy decode, at most ``max_new_tokens`` tokens in total.

    Single turn: with ``suppress_stop`` exactly ``max_new_tokens`` tokens (stop tokens masked,
    the step where one would have won is recorded), else until a stop token.
    Multi-turn (``followups``): when an assistant turn ends (stop token, or ``turn_max_new``
    tokens), the next follow-up's token ids are fed one per step (kind FED, traced like any
    decode step); ``eot_ids`` is prepended when the turn ended without a stop token.
    Step s of the tracer records the query producing token s+1."""
    if followups and suppress_stop:
        raise ValueError("multi-turn decoding needs stop tokens to end turns")
    device = input_ids.device
    if input_ids.dim() == 1:
        input_ids = input_ids[None]
    n = input_ids.shape[1]
    cache = DynamicCache()
    stop = torch.tensor(stop_ids, device=device)

    _sync(device)
    t0 = time.perf_counter()
    for start in range(0, n, prefill_chunk_size):
        if tracer is not None:
            tracer.begin_prefill_chunk(start)
        out = model(input_ids=input_ids[:, start:start + prefill_chunk_size], past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
    if tracer is not None:
        tracer.finish_prefill()
    logits = out.logits[0, -1].float()
    _sync(device)
    t1 = time.perf_counter()

    pending = [list(f) for f in followups or []]
    queue: list[int] = []
    tokens: list[int] = []
    kinds: list[int] = []
    turns: list[dict] = []
    turn = {"start": 0, "natural_stop": None}
    gen_in_turn = 0
    natural_stop = None
    for s in range(max_new_tokens):
        ended_by_stop = False
        if queue:
            tok, kind = queue.pop(0), FED
        else:
            kind = GEN
            tok = int(torch.argmax(logits))
            is_stop = tok in stop_ids
            if is_stop and natural_stop is None:
                natural_stop = s
            if is_stop and turn["natural_stop"] is None:
                turn["natural_stop"] = s
            if is_stop and suppress_stop:
                logits[stop] = float("-inf")
                tok = int(torch.argmax(logits))
            ended_by_stop = is_stop and not suppress_stop
            gen_in_turn += 1
        tokens.append(tok)  # one token per step, stop token included when not suppressed
        kinds.append(kind)
        if kind == GEN and (ended_by_stop or (turn_max_new is not None and gen_in_turn >= turn_max_new)):
            turn["end"] = len(tokens)
            turns.append(turn)
            if not pending:  # last turn finished (stop token or its own cap): conversation over
                break
            queue = ([] if ended_by_stop else list(eot_ids or [])) + pending.pop(0)
            turn = {"start": len(tokens) + len(queue), "natural_stop": None}
            gen_in_turn = 0
        if s == max_new_tokens - 1:
            break
        if tracer is not None:
            tracer.begin_decode_step(s + 1)
        out = model(input_ids=torch.tensor([[tok]], device=device), past_key_values=cache, use_cache=True)
        if tracer is not None:
            tracer.end_decode_step()
        logits = out.logits[0, -1].float()
    if not turns or turns[-1] is not turn:
        turn["end"] = len(tokens)
        turns.append(turn)
    if tracer is not None:
        tracer.flush()
    _sync(device)
    t2 = time.perf_counter()
    res = GenResult(tokens=tokens, natural_stop_step=natural_stop, prefill_s=t1 - t0, decode_s=t2 - t1,
                    kinds=kinds, turns=turns)
    return (res, cache) if return_cache else res
