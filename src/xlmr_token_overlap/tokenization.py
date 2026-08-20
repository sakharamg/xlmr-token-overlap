"""Exact tokenizer loading and auditable token counting."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer

from .io import Record, portable_path, sha256


@dataclass(slots=True)
class LanguageTokens:
    language_code: str
    examples: int
    encoded_token_count: int
    analysis_token_count: int
    unique_analysis_token_count: int
    special_token_count: int
    control_token_count: int
    unknown_token_count: int
    counts: Counter[int]
    split_examples: Counter[str]
    split_encoded_tokens: Counter[str]
    split_analysis_tokens: Counter[str]


class AuditedTokenizer:
    def __init__(self, tokenizer_json: Path) -> None:
        self.path = tokenizer_json
        self.raw = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        self.special_ids = {
            int(item["id"])
            for item in self.raw.get("added_tokens", [])
            if item.get("special")
        }
        model = self.raw.get("model", {})
        self.unknown_id = model.get("unk_id")
        if self.unknown_id is None:
            self.unknown_id = self.tokenizer.token_to_id("<unk>")
        if self.unknown_id is not None:
            self.special_ids.add(int(self.unknown_id))
        self.eligible_vocab_size = self.vocab_size - len(self.special_ids)
        self.checksum = sha256(tokenizer_json)

    def metadata(self) -> dict:
        return {
            "path": portable_path(self.path),
            "sha256": self.checksum,
            "vocab_size_with_added_tokens": self.vocab_size,
            "eligible_vocab_size": self.eligible_vocab_size,
            "special_token_ids_excluded": sorted(self.special_ids),
            "special_tokens_excluded": {
                str(token_id): self.tokenizer.id_to_token(token_id)
                for token_id in sorted(self.special_ids)
            },
            "unknown_token_id": self.unknown_id,
            "external_normalization": "none",
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
        }

    def count(self, language_code: str, records: Iterable[Record]) -> LanguageTokens:
        rows = list(records)
        encodings = self.tokenizer.encode_batch(
            [record.text for record in rows], add_special_tokens=False
        )
        counts: Counter[int] = Counter()
        split_examples: Counter[str] = Counter()
        split_encoded_tokens: Counter[str] = Counter()
        split_analysis_tokens: Counter[str] = Counter()
        encoded_total = 0
        special_total = 0
        control_total = 0
        unknown_total = 0

        for record, encoding in zip(rows, encodings, strict=True):
            ids = encoding.ids
            encoded_total += len(ids)
            split_examples[record.split] += 1
            split_encoded_tokens[record.split] += len(ids)
            for token_id in ids:
                if token_id in self.special_ids:
                    special_total += 1
                    if self.unknown_id is not None and token_id == self.unknown_id:
                        unknown_total += 1
                    else:
                        control_total += 1
                else:
                    counts[token_id] += 1
                    split_analysis_tokens[record.split] += 1

        return LanguageTokens(
            language_code=language_code,
            examples=len(rows),
            encoded_token_count=encoded_total,
            analysis_token_count=sum(counts.values()),
            unique_analysis_token_count=len(counts),
            special_token_count=special_total,
            control_token_count=control_total,
            unknown_token_count=unknown_total,
            counts=counts,
            split_examples=split_examples,
            split_encoded_tokens=split_encoded_tokens,
            split_analysis_tokens=split_analysis_tokens,
        )
