"""Frozen overlap definitions shared by all data passes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import LANGUAGE_BY_CODE, Language
from .tokenization import AuditedTokenizer, LanguageTokens


@dataclass(slots=True)
class ConditionMetrics:
    condition: str
    languages: tuple[Language, ...]
    language_tokens: dict[str, LanguageTokens]
    language_stats: pd.DataFrame
    split_stats: pd.DataFrame
    token_frequencies: pd.DataFrame
    pairwise: pd.DataFrame
    type_iou: pd.DataFrame
    type_iou_upper: pd.DataFrame
    shared_count: pd.DataFrame
    frequency_overlap: pd.DataFrame


def compute_metrics(
    condition: str,
    languages: tuple[Language, ...],
    language_tokens: dict[str, LanguageTokens],
    tokenizer: AuditedTokenizer,
) -> ConditionMetrics:
    codes = [language.code for language in languages]

    language_rows: list[dict] = []
    split_rows: list[dict] = []
    frequency_rows: list[dict] = []
    for language in languages:
        tokens = language_tokens[language.code]
        language_rows.append(
            {
                "condition": condition,
                "language_code": language.code,
                "language": language.name,
                "label": language.label,
                "script": language.script,
                "family": language.family,
                "visual_group": language.visual_group,
                "examples": tokens.examples,
                "encoded_token_count": tokens.encoded_token_count,
                "analysis_token_count": tokens.analysis_token_count,
                "unique_analysis_token_count": tokens.unique_analysis_token_count,
                "vocabulary_coverage_percent": (
                    100.0 * tokens.unique_analysis_token_count / tokenizer.eligible_vocab_size
                ),
                "special_token_count": tokens.special_token_count,
                "control_token_count": tokens.control_token_count,
                "unknown_token_count": tokens.unknown_token_count,
                "unknown_rate_percent": (
                    100.0 * tokens.unknown_token_count / tokens.encoded_token_count
                    if tokens.encoded_token_count
                    else 0.0
                ),
                "mean_analysis_tokens_per_example": (
                    tokens.analysis_token_count / tokens.examples if tokens.examples else 0.0
                ),
            }
        )
        for split in sorted(tokens.split_examples):
            split_rows.append(
                {
                    "condition": condition,
                    "language_code": language.code,
                    "split": split,
                    "examples": tokens.split_examples[split],
                    "encoded_token_count": tokens.split_encoded_tokens[split],
                    "analysis_token_count": tokens.split_analysis_tokens[split],
                }
            )
        for token_id, count in sorted(tokens.counts.items()):
            frequency_rows.append(
                {
                    "condition": condition,
                    "language_code": language.code,
                    "token_id": token_id,
                    "token": tokenizer.tokenizer.id_to_token(token_id),
                    "count": count,
                    "frequency_percent": 100.0 * count / tokens.analysis_token_count,
                }
            )

    iou = np.zeros((len(codes), len(codes)), dtype=float)
    shared = np.zeros((len(codes), len(codes)), dtype=np.int64)
    directional = np.zeros((len(codes), len(codes)), dtype=float)
    pairwise_rows: list[dict] = []

    for source_index, source_code in enumerate(codes):
        source = language_tokens[source_code]
        source_types = set(source.counts)
        for target_index, target_code in enumerate(codes):
            target = language_tokens[target_code]
            target_types = set(target.counts)
            intersection = source_types & target_types
            union = source_types | target_types
            shared_types = len(intersection)
            union_types = len(union)
            type_iou = 100.0 * shared_types / union_types if union_types else float("nan")
            source_shared_occurrences = sum(source.counts[token_id] for token_id in intersection)
            frequency = (
                100.0 * source_shared_occurrences / source.analysis_token_count
                if source.analysis_token_count
                else float("nan")
            )
            iou[source_index, target_index] = type_iou
            shared[source_index, target_index] = shared_types
            directional[source_index, target_index] = frequency
            pairwise_rows.append(
                {
                    "condition": condition,
                    "source_language_code": source_code,
                    "target_language_code": target_code,
                    "source_script": LANGUAGE_BY_CODE[source_code].script,
                    "target_script": LANGUAGE_BY_CODE[target_code].script,
                    "shared_token_count": shared_types,
                    "union_token_count": union_types,
                    "type_iou_percent": type_iou,
                    "source_shared_token_occurrences": source_shared_occurrences,
                    "source_analysis_token_count": source.analysis_token_count,
                    "frequency_overlap_percent": frequency,
                }
            )

    type_iou = pd.DataFrame(iou, index=codes, columns=codes)
    type_iou.index.name = "language_code"
    type_iou_upper = type_iou.mask(np.tril(np.ones(type_iou.shape, dtype=bool), k=-1))
    shared_count = pd.DataFrame(shared, index=codes, columns=codes)
    shared_count.index.name = "language_code"
    frequency_overlap = pd.DataFrame(directional, index=codes, columns=codes)
    frequency_overlap.index.name = "source_language_code"

    return ConditionMetrics(
        condition=condition,
        languages=languages,
        language_tokens=language_tokens,
        language_stats=pd.DataFrame(language_rows),
        split_stats=pd.DataFrame(split_rows),
        token_frequencies=pd.DataFrame(frequency_rows),
        pairwise=pd.DataFrame(pairwise_rows),
        type_iou=type_iou,
        type_iou_upper=type_iou_upper,
        shared_count=shared_count,
        frequency_overlap=frequency_overlap,
    )

