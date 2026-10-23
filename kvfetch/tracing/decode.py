"""Chunked prefill + greedy decode, optionally traced. Batch size 1."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from kvfetch.tracing.tracer import AttentionTracer


@dataclass
class GenResult:
    tokens: list[int]
    natural_stop_step: int | None  # first step whose argmax was a stop token (None = never)
    prefill_s: float
    decode_s: float

    def natural_tokens(self) -> list[int]:
        return self.tokens if self.natural_stop_step is None else self.tokens[:self.natural_stop_step]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, input_ids: torch.Tensor, *, max_new_tokens: int, stop_ids: list[int],
             suppress_stop: bool, prefill_chunk_size: int,
             tracer: AttentionTracer | None = None) -> GenResult:
    """Greedy decode of exactly ``max_new_tokens`` tokens when ``suppress_stop`` (stop
    tokens are masked out, but the step where one would have won is recorded), else
    until a stop token. Step s of the tracer records the query producing token s+1."""
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

    tokens: list[int] = []
    natural_stop = None
    for s in range(max_new_tokens):
        tok = int(torch.argmax(logits))
        is_stop = tok in stop_ids
        if is_stop and natural_stop is None:
            natural_stop = s
        if is_stop and suppress_stop:
            logits[stop] = float("-inf")
            tok = int(torch.argmax(logits))
        tokens.append(tok)  # one token per step, stop token included when not suppressed
        if (is_stop and not suppress_stop) or s == max_new_tokens - 1:
            break
        if tracer is not None:
            tracer.begin_decode_step(s + 1)
        out = model(input_ids=torch.tensor([[tok]], device=device), past_key_values=cache, use_cache=True)
        if tracer is not None:
            tracer.end_decode_step()
        logits = out.logits[0, -1].float()
    if tracer is not None:
        tracer.flush()
    _sync(device)
    t2 = time.perf_counter()
    return GenResult(tokens=tokens, natural_stop_step=natural_stop, prefill_s=t1 - t0, decode_s=t2 - t1)
