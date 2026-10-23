"""Long-generation task: summarize the beginning of a PG19 test book. Placeholder for the
"gradual" task until the custom multi-turn re-reference task is specified (§9.2)."""
from __future__ import annotations

from kvfetch.models import encode, format_prompt
from kvfetch.tasks.ruler import Sample


class LongGen:
    CHARS_PER_TOKEN_UPPER = 8  # generous: read at most budget * this many characters per book

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lc = cfg["longgen"]

    def samples(self, tok, *, max_seq_length: int, decode_steps: int, num_samples: int) -> list[Sample]:
        from datasets import Dataset

        books = Dataset.from_file(self.cfg["data"]["pg19_test"])
        overhead = len(encode(tok, format_prompt(tok, self.lc["template"].format(document=""),
                                                 self.lc["answer_prefix"], self.cfg))["input_ids"])
        budget = max_seq_length - decode_steps - overhead
        out = []
        for i, book in enumerate(books):
            ids = encode(tok, book["text"][: budget * self.CHARS_PER_TOKEN_UPPER])["input_ids"]
            if len(ids) < budget * self.lc["min_book_tokens_slack"]:
                continue
            n = budget
            while True:  # decode/re-encode can shift the count slightly; trim until it fits
                doc = tok.decode(ids[:n])
                prompt = format_prompt(tok, self.lc["template"].format(document=doc), self.lc["answer_prefix"], self.cfg)
                n_tok = len(encode(tok, prompt)["input_ids"])
                if n_tok + decode_steps <= max_seq_length:
                    break
                n -= n_tok + decode_steps - max_seq_length
            out.append(Sample(task="longgen", index=len(out), prompt=prompt, answers=[],
                              extra={"book_index": i, "title": book["short_book_title"], "prompt_tokens": n_tok}))
            if len(out) == num_samples:
                break
        if len(out) < num_samples:
            raise ValueError(f"only {len(out)} PG19 books are long enough for {max_seq_length} tokens")
        return out
