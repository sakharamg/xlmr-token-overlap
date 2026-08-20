"""Pinned, coverage-aware MTEB Multilingual v2 corpus construction.

This module builds an auditable tokenizer-analysis corpus, not an embedding
benchmark score.  It uses a controlled subset of official
``MTEB(Multilingual, v2)`` tasks chosen to maximize coverage of the frozen 24
languages without allowing English-only tasks to dominate a task family.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import (
    LANGUAGE_BY_CODE,
    LANGUAGE_CODES,
    LANGUAGES,
    MTEB_BENCHMARK,
    MTEB_FAMILIES,
    MTEB_SOURCE_REVISION,
    MTEB_VERSION,
)
from .io import Record, portable_path, sha256
from .pipeline import run_analysis
from .plotting import plot_matrix
from .tokenization import AuditedTokenizer


MASSIVE_REVISION = "4672e20407010da34463acc759c162ca9734bca6"
GUJARATI_NEWS_REVISION = "7f438910e374d37b9136d4ded9888edb3551a494"
SIB200_REVISION = "a74d7350ea12af010cfb1c21e34f1f81fd2e615b"
BELEBELE_REVISION = "979a211276faa22f671e69d096634193567cfd05"
STS22_REVISION = "d31f33a128469b20e357535c39b82fb3c3f6f2bd"
STS17_REVISION = "faeb762787bd10488a50c8b5be4a3b82e411949c"
SEMREL24_REVISION = "f5146bf724c55899ca20502851dc98c5405b67f9"
INDIC_STS_REVISION = "f0366eb5a20087355c0e131162bbed943ba54b51"
JSICK_REVISION = "729cfe4a16d3c2b61c6aa9f9f6c8a96bb5512868"
WIKIPEDIA_RERANK_REVISION = "803771c366038ed587b21e3d8fe25f8f73134fad"
WEBLINX_REVISION = "107fdc2402d2c4bfb2a720dfcfe1f6ff9d21151b"
ALLOPROF_REVISION = "a7d2d793f2e5ba55139bb10088c2e8ee2df2ce02"
VOYAGE_MMARCO_REVISION = "bd2050c52b480e48c51372b4ec98a1cbbc4515f2"
RUBQ_REVISION = "e8233e2234f8b24ab47f203b69d1161c3c0bc5a1"
T2_REVISION = "a34fe6bc0dff185af1228e49a0f6fb1de1565627"


MASSIVE_CONFIGS: Mapping[str, str] = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "zh-CN": "zho_Hans",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "ro": "ron_Latn",
    "tl": "tgl_Latn",
    "ms": "zsm_Latn",
    "hu": "hun_Latn",
}

STS22_CONFIGS: Mapping[str, tuple[str, str]] = {
    "en": ("eng_Latn", "eng_Latn"),
    "de": ("deu_Latn", "deu_Latn"),
    "es": ("spa_Latn", "spa_Latn"),
    "pl": ("pol_Latn", "pol_Latn"),
    "tr": ("tur_Latn", "tur_Latn"),
    "ar": ("arb_Arab", "arb_Arab"),
    "ru": ("rus_Cyrl", "rus_Cyrl"),
    "zh": ("zho_Hans", "zho_Hans"),
    "fr": ("fra_Latn", "fra_Latn"),
    "it": ("ita_Latn", "ita_Latn"),
}

STS17_CONFIGS: Mapping[str, tuple[str, str]] = {
    "ko-ko": ("kor_Hang", "kor_Hang"),
    "nl-en": ("nld_Latn", "eng_Latn"),
}

SEMREL_CONFIGS: Mapping[str, tuple[str, str]] = {
    "ind": ("ind_Latn", "ind_Latn"),
    "hin": ("hin_Deva", "hin_Deva"),
}

INDIC_STS_CONFIGS: Mapping[str, tuple[str, str]] = {
    "en-gu": ("eng_Latn", "guj_Gujr"),
}

WIKIPEDIA_RERANK_CONFIGS: Mapping[str, str] = {
    "de": "deu_Latn",
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "sv": "swe_Latn",
}

EXPECTED_FAMILY_LANGUAGES: Mapping[str, frozenset[str]] = {
    "classification": frozenset(LANGUAGE_CODES),
    "clustering": frozenset(LANGUAGE_CODES),
    "retrieval": frozenset(LANGUAGE_CODES),
    "sts": frozenset(
        {
            "arb_Arab",
            "deu_Latn",
            "eng_Latn",
            "spa_Latn",
            "fra_Latn",
            "guj_Gujr",
            "hin_Deva",
            "ind_Latn",
            "ita_Latn",
            "jpn_Jpan",
            "kor_Hang",
            "nld_Latn",
            "pol_Latn",
            "rus_Cyrl",
            "tur_Latn",
            "zho_Hans",
        }
    ),
    "reranking": frozenset(
        {
            "deu_Latn",
            "eng_Latn",
            "fra_Latn",
            "hin_Deva",
            "ita_Latn",
            "jpn_Jpan",
            "nld_Latn",
            "por_Latn",
            "ron_Latn",
            "rus_Cyrl",
            "swe_Latn",
            "zho_Hans",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class Candidate:
    family: str
    language_code: str
    text: str
    task: str
    subset: str
    split: str
    role: str
    example_id: str
    analysis_tokens: int


@dataclass(frozen=True, slots=True)
class SourceTask:
    name: str
    family: str
    path: str
    revision: str
    selected_scope: str
    languages: tuple[str, ...]


SOURCE_TASKS: tuple[SourceTask, ...] = (
    SourceTask(
        "MassiveIntentClassification",
        "classification",
        "mteb/amazon_massive_intent",
        MASSIVE_REVISION,
        "validation and test for 23 supported locales",
        tuple(code for code in LANGUAGE_CODES if code != "guj_Gujr"),
    ),
    SourceTask(
        "GujaratiNewsClassification",
        "classification",
        "mteb/GujaratiNewsClassification",
        GUJARATI_NEWS_REVISION,
        "test",
        ("guj_Gujr",),
    ),
    SourceTask(
        "SIB200ClusteringS2S",
        "clustering",
        "mteb/sib200",
        SIB200_REVISION,
        "train, validation and test for all 24 languages",
        tuple(LANGUAGE_CODES),
    ),
    SourceTask(
        "BelebeleRetrieval",
        "retrieval",
        "mteb/belebele",
        BELEBELE_REVISION,
        "deduplicated test questions and passages for all 24 languages",
        tuple(LANGUAGE_CODES),
    ),
    SourceTask(
        "STS22.v2",
        "sts",
        "mteb/sts22-crosslingual-sts",
        STS22_REVISION,
        "ten monolingual test subsets",
        tuple(sorted({code for pair in STS22_CONFIGS.values() for code in pair})),
    ),
    SourceTask(
        "STS17",
        "sts",
        "mteb/sts17-crosslingual-sts",
        STS17_REVISION,
        "ko-ko and nl-en test subsets",
        tuple(sorted({code for pair in STS17_CONFIGS.values() for code in pair})),
    ),
    SourceTask(
        "SemRel24STS",
        "sts",
        "mteb/SemRel24STS",
        SEMREL24_REVISION,
        "Indonesian and Hindi test subsets",
        ("hin_Deva", "ind_Latn"),
    ),
    SourceTask(
        "IndicCrosslingualSTS",
        "sts",
        "mteb/IndicCrosslingualSTS",
        INDIC_STS_REVISION,
        "en-gu test subset",
        ("eng_Latn", "guj_Gujr"),
    ),
    SourceTask(
        "JSICK",
        "sts",
        "mteb/JSICK",
        JSICK_REVISION,
        "test",
        ("jpn_Jpan",),
    ),
    SourceTask(
        "WikipediaRerankingMultilingual",
        "reranking",
        "mteb/WikipediaRerankingMultilingual",
        WIKIPEDIA_RERANK_REVISION,
        "test queries and corpus for eight requested languages",
        tuple(WIKIPEDIA_RERANK_CONFIGS.values()),
    ),
    SourceTask(
        "WebLINXCandidatesReranking",
        "reranking",
        "mteb/WebLINXCandidatesReranking",
        WEBLINX_REVISION,
        "test_iid queries and corpus",
        ("eng_Latn",),
    ),
    SourceTask(
        "AlloprofReranking",
        "reranking",
        "mteb/AlloprofReranking",
        ALLOPROF_REVISION,
        "test queries and corpus",
        ("fra_Latn",),
    ),
    SourceTask(
        "VoyageMMarcoReranking",
        "reranking",
        "mteb/VoyageMMarcoReranking",
        VOYAGE_MMARCO_REVISION,
        "test queries and corpus",
        ("jpn_Jpan",),
    ),
    SourceTask(
        "RuBQReranking",
        "reranking",
        "mteb/RuBQReranking",
        RUBQ_REVISION,
        "test queries and corpus",
        ("rus_Cyrl",),
    ),
    SourceTask(
        "T2Reranking",
        "reranking",
        "mteb/T2Reranking",
        T2_REVISION,
        "dev queries and corpus",
        ("zho_Hans",),
    ),
)


def _stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dataset_rows(
    path: str,
    revision: str,
    config: str | None,
    requested_splits: Sequence[str],
    cache_dir: Path,
) -> Iterator[tuple[str, int, Mapping]]:
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as error:  # pragma: no cover - exercised by the CLI environment
        raise RuntimeError(
            "MTEB loading requires the 'mteb' optional dependencies; "
            "install requirements-mteb.txt"
        ) from error

    kwargs = {"revision": revision}
    if config is not None:
        kwargs["config_name"] = config
    available = get_dataset_split_names(path, **kwargs)
    selected = [split for split in requested_splits if split in available]
    if not selected and len(available) == 1:
        selected = [str(available[0])]
    if not selected:
        raise ValueError(
            f"No requested split for {path}/{config}: requested={list(requested_splits)}, "
            f"available={available}"
        )

    for split in selected:
        dataset = load_dataset(
            path,
            config,
            split=split,
            revision=revision,
            streaming=True,
            cache_dir=str(cache_dir),
        )
        for index, row in enumerate(dataset):
            yield split, index, row


def _text(row: Mapping, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string field {field!r}; columns={sorted(row)}")
    return value


class _Collector:
    def __init__(self, tokenizer: AuditedTokenizer, task_token_cap: int) -> None:
        self.tokenizer = tokenizer
        self.task_token_cap = task_token_cap
        self.candidates: list[Candidate] = []
        self.task_tokens: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.seen_text: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    def full(self, task: str, language_code: str) -> bool:
        return self.task_tokens[(task, language_code)] >= self.task_token_cap

    def add(
        self,
        *,
        family: str,
        language_code: str,
        text: str,
        task: str,
        subset: str,
        split: str,
        role: str,
        row_id: object,
    ) -> None:
        if self.full(task, language_code):
            return
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seen = self.seen_text[(family, language_code)]
        if digest in seen:
            return
        token_count = self.tokenizer.analysis_length(text)
        if token_count == 0:
            return
        seen.add(digest)
        example_id = f"{task}:{subset}:{split}:{role}:{row_id}"
        self.candidates.append(
            Candidate(
                family=family,
                language_code=language_code,
                text=text,
                task=task,
                subset=subset,
                split=split,
                role=role,
                example_id=example_id,
                analysis_tokens=token_count,
            )
        )
        self.task_tokens[(task, language_code)] += token_count


def _collect_single_text_task(
    collector: _Collector,
    *,
    family: str,
    task: str,
    path: str,
    revision: str,
    configs: Mapping[str | None, str],
    splits: Sequence[str],
    field: str,
    cache_dir: Path,
) -> None:
    for config, language_code in configs.items():
        for split, index, row in _dataset_rows(path, revision, config, splits, cache_dir):
            collector.add(
                family=family,
                language_code=language_code,
                text=_text(row, field),
                task=task,
                subset=config or "default",
                split=split,
                role=field,
                row_id=row.get("id", index),
            )
            if collector.full(task, language_code):
                break


def _collect_sts_task(
    collector: _Collector,
    *,
    task: str,
    path: str,
    revision: str,
    configs: Mapping[str | None, tuple[str, str]],
    cache_dir: Path,
) -> None:
    for config, languages in configs.items():
        for split, index, row in _dataset_rows(path, revision, config, ("test",), cache_dir):
            for field, language_code in zip(("sentence1", "sentence2"), languages, strict=True):
                collector.add(
                    family="sts",
                    language_code=language_code,
                    text=_text(row, field),
                    task=task,
                    subset=config or "default",
                    split=split,
                    role=field,
                    row_id=row.get("id", index),
                )
            if all(collector.full(task, code) for code in set(languages)):
                break


def _collect_belebele(collector: _Collector, cache_dir: Path) -> None:
    for language_code in LANGUAGE_CODES:
        for split, index, row in _dataset_rows(
            "mteb/belebele",
            BELEBELE_REVISION,
            language_code,
            ("test",),
            cache_dir,
        ):
            collector.add(
                family="retrieval",
                language_code=language_code,
                text=_text(row, "question"),
                task="BelebeleRetrieval",
                subset=language_code,
                split=split,
                role="query",
                row_id=row.get("id", index),
            )
            collector.add(
                family="retrieval",
                language_code=language_code,
                text=_text(row, "flores_passage"),
                task="BelebeleRetrieval",
                subset=language_code,
                split=split,
                role="corpus",
                row_id=row.get("link", index),
            )
            if collector.full("BelebeleRetrieval", language_code):
                break


def _collect_retrieval_task(
    collector: _Collector,
    *,
    task: str,
    path: str,
    revision: str,
    language_code: str,
    splits: Sequence[str],
    cache_dir: Path,
    subset: str | None = None,
) -> None:
    prefix = f"{subset}-" if subset else ""
    for role in ("queries", "corpus"):
        config = f"{prefix}{role}"
        for split, index, row in _dataset_rows(path, revision, config, splits, cache_dir):
            row_id = row.get("id", row.get("_id", index))
            fields = ("text",) if role == "queries" else ("title", "text")
            for field in fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    collector.add(
                        family="reranking",
                        language_code=language_code,
                        text=value,
                        task=task,
                        subset=subset or "default",
                        split=split,
                        role=f"{role}:{field}",
                        row_id=row_id,
                    )
            if collector.full(task, language_code):
                break


def collect_candidates(
    tokenizer: AuditedTokenizer,
    cache_dir: Path,
    requested_token_budget: int,
) -> list[Candidate]:
    """Download pinned task slices and return de-duplicated candidate texts."""

    collector = _Collector(tokenizer, task_token_cap=max(10_000, requested_token_budget * 2))
    _collect_single_text_task(
        collector,
        family="classification",
        task="MassiveIntentClassification",
        path="mteb/amazon_massive_intent",
        revision=MASSIVE_REVISION,
        configs=MASSIVE_CONFIGS,
        splits=("validation", "test"),
        field="text",
        cache_dir=cache_dir,
    )
    _collect_single_text_task(
        collector,
        family="classification",
        task="GujaratiNewsClassification",
        path="mteb/GujaratiNewsClassification",
        revision=GUJARATI_NEWS_REVISION,
        configs={None: "guj_Gujr"},
        splits=("test",),
        field="text",
        cache_dir=cache_dir,
    )
    _collect_single_text_task(
        collector,
        family="clustering",
        task="SIB200ClusteringS2S",
        path="mteb/sib200",
        revision=SIB200_REVISION,
        configs={code: code for code in LANGUAGE_CODES},
        splits=("train", "validation", "test"),
        field="text",
        cache_dir=cache_dir,
    )
    _collect_belebele(collector, cache_dir)

    _collect_sts_task(
        collector,
        task="STS22.v2",
        path="mteb/sts22-crosslingual-sts",
        revision=STS22_REVISION,
        configs=STS22_CONFIGS,
        cache_dir=cache_dir,
    )
    _collect_sts_task(
        collector,
        task="STS17",
        path="mteb/sts17-crosslingual-sts",
        revision=STS17_REVISION,
        configs=STS17_CONFIGS,
        cache_dir=cache_dir,
    )
    _collect_sts_task(
        collector,
        task="SemRel24STS",
        path="mteb/SemRel24STS",
        revision=SEMREL24_REVISION,
        configs=SEMREL_CONFIGS,
        cache_dir=cache_dir,
    )
    _collect_sts_task(
        collector,
        task="IndicCrosslingualSTS",
        path="mteb/IndicCrosslingualSTS",
        revision=INDIC_STS_REVISION,
        configs=INDIC_STS_CONFIGS,
        cache_dir=cache_dir,
    )
    _collect_sts_task(
        collector,
        task="JSICK",
        path="mteb/JSICK",
        revision=JSICK_REVISION,
        configs={None: ("jpn_Jpan", "jpn_Jpan")},
        cache_dir=cache_dir,
    )

    for subset, language_code in WIKIPEDIA_RERANK_CONFIGS.items():
        _collect_retrieval_task(
            collector,
            task="WikipediaRerankingMultilingual",
            path="mteb/WikipediaRerankingMultilingual",
            revision=WIKIPEDIA_RERANK_REVISION,
            language_code=language_code,
            splits=("test",),
            cache_dir=cache_dir,
            subset=subset,
        )
    for task, path, revision, code, splits in (
        (
            "WebLINXCandidatesReranking",
            "mteb/WebLINXCandidatesReranking",
            WEBLINX_REVISION,
            "eng_Latn",
            ("test_iid",),
        ),
        ("AlloprofReranking", "mteb/AlloprofReranking", ALLOPROF_REVISION, "fra_Latn", ("test",)),
        (
            "VoyageMMarcoReranking",
            "mteb/VoyageMMarcoReranking",
            VOYAGE_MMARCO_REVISION,
            "jpn_Jpan",
            ("test",),
        ),
        ("RuBQReranking", "mteb/RuBQReranking", RUBQ_REVISION, "rus_Cyrl", ("test",)),
        ("T2Reranking", "mteb/T2Reranking", T2_REVISION, "zho_Hans", ("dev",)),
    ):
        _collect_retrieval_task(
            collector,
            task=task,
            path=path,
            revision=revision,
            language_code=code,
            splits=splits,
            cache_dir=cache_dir,
        )
    return collector.candidates


def select_token_budget(
    candidates: Iterable[Candidate],
    target_tokens: int,
    *,
    seed: int,
    namespace: str,
) -> list[Candidate]:
    """Select complete records deterministically without exceeding a token budget."""

    ranked = sorted(
        candidates,
        key=lambda item: _stable_digest(
            seed,
            namespace,
            item.family,
            item.language_code,
            item.example_id,
            hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
        ),
    )
    selected: list[Candidate] = []
    total = 0
    for candidate in ranked:
        if total + candidate.analysis_tokens <= target_tokens:
            selected.append(candidate)
            total += candidate.analysis_tokens
    return selected


def _records(selected: Iterable[Candidate], condition: str) -> list[Record]:
    return [
        Record(
            condition=condition,
            language_code=item.language_code,
            text=item.text,
            example_id=item.example_id,
            split=f"{item.task}|{item.split}|{item.role}",
        )
        for item in selected
    ]


def build_balanced_records(
    candidates: list[Candidate],
    *,
    requested_family_tokens: int,
    requested_overall_tokens: int,
    seed: int,
) -> tuple[list[Record], dict, list[Candidate]]:
    """Balance each family and construct a full-coverage overall condition."""

    by_family_language: defaultdict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_family_language[(candidate.family, candidate.language_code)].append(candidate)

    selected_family: list[Candidate] = []
    family_budgets: dict[str, int] = {}
    coverage: dict[str, list[str]] = {}
    for family in MTEB_FAMILIES:
        observed = {
            code
            for (candidate_family, code), rows in by_family_language.items()
            if candidate_family == family and rows
        }
        expected = EXPECTED_FAMILY_LANGUAGES[family]
        if observed != expected:
            raise ValueError(
                f"Unexpected {family} coverage: missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}"
            )
        available = {
            code: sum(item.analysis_tokens for item in by_family_language[(family, code)])
            for code in observed
        }
        effective_budget = min(requested_family_tokens, min(available.values()))
        family_budgets[family] = effective_budget
        coverage[family] = [code for code in LANGUAGE_CODES if code in observed]
        for code in LANGUAGE_CODES:
            if code not in observed:
                continue
            selected_family.extend(
                select_token_budget(
                    by_family_language[(family, code)],
                    effective_budget,
                    seed=seed,
                    namespace=f"family:{family}:{code}",
                )
            )

    by_language: defaultdict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected_family:
        by_language[candidate.language_code].append(candidate)
    missing_overall = set(LANGUAGE_CODES) - by_language.keys()
    if missing_overall:
        raise ValueError(f"Overall MTEB pool is missing languages: {sorted(missing_overall)}")
    overall_available = {
        code: sum(item.analysis_tokens for item in by_language[code]) for code in LANGUAGE_CODES
    }
    effective_overall = min(requested_overall_tokens, min(overall_available.values()))
    selected_overall: list[Candidate] = []
    for code in LANGUAGE_CODES:
        selected_overall.extend(
            select_token_budget(
                by_language[code],
                effective_overall,
                seed=seed,
                namespace=f"overall:{code}",
            )
        )

    records: list[Record] = []
    for family in MTEB_FAMILIES:
        records.extend(_records((item for item in selected_family if item.family == family), family))
    records.extend(_records(selected_overall, "overall"))

    audit = {
        "requested_family_token_budget": requested_family_tokens,
        "effective_family_token_budgets": family_budgets,
        "requested_overall_token_budget": requested_overall_tokens,
        "effective_overall_token_budget": effective_overall,
        "seed": seed,
        "coverage": coverage,
    }
    return records, audit, selected_family + selected_overall


def _write_frame(frame: pd.DataFrame, base: Path, *, index: bool = False) -> None:
    frame.to_csv(base.with_suffix(".csv"), index=index, float_format="%.8f")
    frame.to_parquet(base.with_suffix(".parquet"), index=index)


def _write_audit_files(
    output_dir: Path,
    audit: dict,
    records: list[Record],
    selected: list[Candidate],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = pd.DataFrame([asdict(task) for task in SOURCE_TASKS])
    tasks["languages"] = tasks["languages"].map(lambda values: ",".join(values))
    tasks.to_csv(output_dir / "task_inventory.csv", index=False)

    candidate_lookup = {
        (item.language_code, item.example_id): item for item in selected
    }
    rows = []
    for record in records:
        item = candidate_lookup[(record.language_code, record.example_id)]
        rows.append({"condition": record.condition, **asdict(item)})
    selected_frame = pd.DataFrame(rows)
    contributions = (
        selected_frame.groupby(
            [
                "condition",
                "family",
                "language_code",
                "task",
                "subset",
                "split",
                "role",
            ],
            dropna=False,
        )
        .agg(examples=("example_id", "size"), analysis_tokens=("analysis_tokens", "sum"))
        .reset_index()
    )
    _write_frame(contributions, output_dir / "source_contributions")
    by_language = (
        selected_frame.groupby(["condition", "language_code"])
        .agg(
            examples=("example_id", "size"),
            analysis_tokens=("analysis_tokens", "sum"),
            tasks=("task", "nunique"),
        )
        .reset_index()
    )
    _write_frame(by_language, output_dir / "sampling_by_language_family")

    coverage = pd.DataFrame(False, index=MTEB_FAMILIES, columns=LANGUAGE_CODES)
    for family, codes in audit["coverage"].items():
        coverage.loc[family, codes] = True
    coverage.index.name = "task_family"
    coverage.to_csv(output_dir / "coverage_matrix.csv")

    audit_payload = {
        **audit,
        "benchmark": MTEB_BENCHMARK,
        "mteb_version": MTEB_VERSION,
        "mteb_source_revision": MTEB_SOURCE_REVISION,
        "selection": "stable SHA-256 ordering of complete, unmodified records",
        "candidate_deduplication": "exact UTF-8 text SHA-256 within family and language",
        "raw_text_committed": False,
    }
    (output_dir / "sampling_manifest.json").write_text(
        json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_all_language_views(condition_dir: Path) -> list[str]:
    stats = pd.read_csv(condition_dir / "language_stats.csv")
    observed = set(stats["language_code"].astype(str))
    missing = [code for code in LANGUAGE_CODES if code not in observed]
    matrix_names = (
        "type_iou_percent",
        "type_iou_percent_upper",
        "frequency_overlap_percent",
        "shared_token_count",
    )
    full_matrices: dict[str, pd.DataFrame] = {}
    for name in matrix_names:
        frame = pd.read_csv(condition_dir / f"{name}.csv", index_col=0)
        full = frame.reindex(index=LANGUAGE_CODES, columns=LANGUAGE_CODES)
        if name == "type_iou_percent_upper":
            full = full.mask(np.tril(np.ones(full.shape, dtype=bool), k=-1))
        _write_frame(full, condition_dir / f"{name}_all_languages", index=True)
        full_matrices[name] = full

    rows = []
    by_code = stats.set_index("language_code").to_dict("index")
    for language in LANGUAGES:
        if language.code in by_code:
            rows.append(
                {
                    "language_code": language.code,
                    **by_code[language.code],
                    "available": True,
                }
            )
        else:
            rows.append(
                {
                    "condition": stats["condition"].iloc[0],
                    "language_code": language.code,
                    "language": language.name,
                    "label": language.label,
                    "script": language.script,
                    "family": language.family,
                    "visual_group": language.visual_group,
                    "examples": 0,
                    "encoded_token_count": 0,
                    "analysis_token_count": 0,
                    "unique_analysis_token_count": 0,
                    "vocabulary_coverage_percent": np.nan,
                    "special_token_count": 0,
                    "control_token_count": 0,
                    "unknown_token_count": 0,
                    "unknown_rate_percent": np.nan,
                    "mean_analysis_tokens_per_example": np.nan,
                    "available": False,
                }
            )
    _write_frame(pd.DataFrame(rows), condition_dir / "language_stats_all_languages")
    (condition_dir / "missing_languages.json").write_text(
        json.dumps({"missing_languages": missing}, indent=2) + "\n", encoding="utf-8"
    )

    heatmaps = condition_dir / "heatmaps_all_languages"
    condition = str(stats["condition"].iloc[0])
    plot_matrix(
        full_matrices["type_iou_percent"].to_numpy(),
        LANGUAGES,
        heatmaps / "type_iou_upper.png",
        f"MTEB {condition}: XLM-R token-type IoU (coverage masked)",
        "Token-type IoU (%)",
        upper_triangle=True,
    )
    plot_matrix(
        full_matrices["frequency_overlap_percent"].to_numpy(),
        LANGUAGES,
        heatmaps / "frequency_overlap_directional.png",
        f"MTEB {condition}: directional frequency overlap (coverage masked)",
        "Source occurrences covered (%)",
    )
    plot_matrix(
        full_matrices["shared_token_count"].to_numpy(),
        LANGUAGES,
        heatmaps / "shared_token_count.png",
        f"MTEB {condition}: shared observed XLM-R token types (coverage masked)",
        "Shared token types",
        integer_annotations=True,
    )
    return missing


def _pair_vector(frame: pd.DataFrame, codes: Sequence[str]) -> pd.Series:
    values = {
        f"{left}|{right}": float(frame.loc[left, right])
        for left, right in combinations(codes, 2)
        if pd.notna(frame.loc[left, right])
    }
    return pd.Series(values, dtype=float)


def _spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """Return Spearman's rho without pandas' optional SciPy dependency."""

    aligned = pd.concat(
        [left.rename("left"), right.rename("right")], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    ranks = aligned.rank(method="average")
    return float(ranks["left"].corr(ranks["right"]))


def _write_cross_pass(output_dir: Path, flores_dir: Path) -> None:
    flores = pd.read_csv(flores_dir / "type_iou_percent.csv", index_col=0)
    overall = pd.read_csv(output_dir / "overall" / "type_iou_percent.csv", index_col=0)
    summary_rows = []
    pair_rows = []
    vectors = {"flores": _pair_vector(flores, LANGUAGE_CODES)}
    for condition in ("overall", *MTEB_FAMILIES):
        frame = pd.read_csv(output_dir / condition / "type_iou_percent.csv", index_col=0)
        codes = [code for code in LANGUAGE_CODES if code in frame.index]
        condition_vector = _pair_vector(frame, codes)
        vectors[condition] = condition_vector
        flores_vector = vectors["flores"].reindex(condition_vector.index)
        difference = condition_vector - flores_vector
        summary_rows.append(
            {
                "condition": condition,
                "languages": len(codes),
                "pairs": len(condition_vector),
                "spearman_vs_flores": _spearman_correlation(
                    condition_vector, flores_vector
                ),
                "mean_absolute_difference_points": difference.abs().mean(),
                "mean_signed_difference_points": difference.mean(),
            }
        )
        for pair, value in condition_vector.items():
            left, right = pair.split("|", 1)
            pair_rows.append(
                {
                    "condition": condition,
                    "language_i": left,
                    "language_j": right,
                    "flores_type_iou_percent": flores_vector[pair],
                    "mteb_type_iou_percent": value,
                    "difference_points": value - flores_vector[pair],
                }
            )
    _write_frame(pd.DataFrame(summary_rows), output_dir / "cross_pass_summary")
    _write_frame(pd.DataFrame(pair_rows), output_dir / "flores_pairwise_comparison")

    names = ["flores", "overall", *MTEB_FAMILIES]
    correlations = pd.DataFrame(np.nan, index=names, columns=names)
    for left in names:
        for right in names:
            common = vectors[left].index.intersection(vectors[right].index)
            correlations.loc[left, right] = _spearman_correlation(
                vectors[left].loc[common], vectors[right].loc[common]
            )
    correlations.index.name = "condition"
    correlations.to_csv(output_dir / "condition_spearman.csv", float_format="%.8f")


def _refresh_manifest(condition_dir: Path) -> None:
    path = condition_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    outputs = []
    for output in sorted(condition_dir.rglob("*")):
        if output.is_file() and output.name != "manifest.json":
            outputs.append(
                {
                    "path": str(output.relative_to(condition_dir)),
                    "bytes": output.stat().st_size,
                    "sha256": sha256(output),
                }
            )
    manifest["outputs"] = outputs
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_mteb(output_dir: Path) -> dict:
    checks = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_conditions = {"overall", *MTEB_FAMILIES}
    observed_conditions = {path.name for path in output_dir.iterdir() if path.is_dir()}
    check(
        "all conditions emitted",
        expected_conditions <= observed_conditions,
        f"expected={sorted(expected_conditions)}, observed={sorted(observed_conditions)}",
    )
    for condition in sorted(expected_conditions):
        condition_dir = output_dir / condition
        validation = json.loads((condition_dir / "validation_report.json").read_text())
        check(f"{condition}: native validation", validation["status"] == "passed")
        stats = pd.read_csv(condition_dir / "language_stats_all_languages.csv")
        check(f"{condition}: 24 language-stat rows", len(stats) == len(LANGUAGE_CODES))
        supported = set(stats.loc[stats["available"].astype(bool), "language_code"])
        expected_supported = (
            set(LANGUAGE_CODES) if condition == "overall" else set(EXPECTED_FAMILY_LANGUAGES[condition])
        )
        check(
            f"{condition}: coverage matches protocol",
            supported == expected_supported,
            f"observed={sorted(supported)}",
        )
        for name in (
            "type_iou_percent",
            "frequency_overlap_percent",
            "shared_token_count",
        ):
            frame = pd.read_csv(
                condition_dir / f"{name}_all_languages.csv", index_col=0
            )
            check(f"{condition}/{name}: 24x24", frame.shape == (24, 24))
            missing = [code for code in LANGUAGE_CODES if code not in supported]
            missing_is_nan = all(frame.loc[code].isna().all() and frame[code].isna().all() for code in missing)
            check(f"{condition}/{name}: missing coverage masked", missing_is_nan)
    failures = [item for item in checks if not item["passed"]]
    report = {
        "status": "passed" if not failures else "failed",
        "checks_run": len(checks),
        "checks_passed": len(checks) - len(failures),
        "failures": failures,
        "checks": checks,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise ValueError("MTEB validation failed: " + ", ".join(row["name"] for row in failures))
    return report


def _write_root_report(output_dir: Path, audit: dict, validation: dict) -> None:
    cross = pd.read_csv(output_dir / "cross_pass_summary.csv")
    summary_rows = []
    for condition in ("overall", *MTEB_FAMILIES):
        summary = json.loads((output_dir / condition / "summary.json").read_text())
        pair = summary["strongest_type_iou_pair"]
        summary_rows.append(
            f"| {condition} | {summary['languages']} | "
            f"{pair['language_i']}–{pair['language_j']} | {pair['type_iou_percent']:.2f}% |"
        )
    cross_rows = []
    for row in cross.itertuples(index=False):
        cross_rows.append(
            f"| {row.condition} | {row.languages} | {row.pairs} | "
            f"{row.spearman_vs_flores:.3f} | {row.mean_absolute_difference_points:.2f} |"
        )
    text = f"""# MTEB Multilingual v2 tokenizer-overlap pass

This is a coverage-balanced tokenizer audit over pinned official tasks from
`{MTEB_BENCHMARK}` as resolved in `mteb=={MTEB_VERSION}`. It is not an MTEB
embedding-score run and it does not claim that overlap causes performance
changes.

## Coverage and sampling

- Frozen language inventory: 24
- Full-coverage families: classification, clustering, retrieval
- STS coverage: {len(EXPECTED_FAMILY_LANGUAGES['sts'])}/24
- Reranking coverage: {len(EXPECTED_FAMILY_LANGUAGES['reranking'])}/24
- Requested family token budget: {audit['requested_family_token_budget']:,}
- Effective family budgets: {json.dumps(audit['effective_family_token_budgets'], sort_keys=True)}
- Requested overall token budget: {audit['requested_overall_token_budget']:,}
- Effective overall token budget: {audit['effective_overall_token_budget']:,}
- Selection seed: {audit['seed']}

Missing languages are represented by blank/NA cells in the `*_all_languages`
24×24 matrices. They are never imputed from another task or language.

## Strongest observed pair by condition

| Condition | Languages | Pair | Type IoU |
|---|---:|---|---:|
{chr(10).join(summary_rows)}

## Stability against FLORES

| Condition | Languages | Pairs | Spearman | Mean absolute difference |
|---|---:|---:|---:|---:|
{chr(10).join(cross_rows)}

## Validation

All {validation['checks_run']} aggregate MTEB checks pass, in addition to the
28 native matrix invariants run independently inside each condition.

See `task_inventory.csv`, `sampling_manifest.json`,
`sampling_by_language_family.csv`, and `source_contributions.csv` for the exact
task revisions, coverage, and selected token counts. Raw benchmark text is not
committed.
"""
    (output_dir / "REPORT.md").write_text(text, encoding="utf-8")


def run_mteb(
    *,
    tokenizer_json: Path,
    cache_dir: Path,
    output_dir: Path,
    flores_dir: Path,
    family_token_budget: int = 20_000,
    overall_token_budget: int = 30_000,
    seed: int = 1729,
) -> dict:
    """Prepare, balance, analyze, compare, and validate the MTEB pass."""

    tokenizer = AuditedTokenizer(tokenizer_json)
    candidates = collect_candidates(tokenizer, cache_dir, family_token_budget)
    records, audit, selected = build_balanced_records(
        candidates,
        requested_family_tokens=family_token_budget,
        requested_overall_tokens=overall_token_budget,
        seed=seed,
    )
    provenance = {
        "dataset": MTEB_BENCHMARK,
        "mteb_version": MTEB_VERSION,
        "mteb_source_revision": MTEB_SOURCE_REVISION,
        "scope": "coverage-balanced tokenizer-audit task slice",
        "task_count": len(SOURCE_TASKS),
        "task_revisions": [asdict(task) for task in SOURCE_TASKS],
        "sampling": audit,
        "cache_dir": portable_path(cache_dir),
        "raw_text_committed": False,
    }
    destinations = run_analysis(
        records,
        tokenizer_json,
        output_dir,
        provenance,
        require_all_languages=False,
    )
    _write_audit_files(output_dir, audit, records, selected)
    missing = {condition: _write_all_language_views(path) for condition, path in destinations.items()}
    _write_cross_pass(output_dir, flores_dir)
    validation = _validate_mteb(output_dir)
    _write_root_report(output_dir, audit, validation)
    for path in destinations.values():
        _refresh_manifest(path)

    result = {
        "status": "passed",
        "conditions": {condition: portable_path(path) for condition, path in destinations.items()},
        "coverage": {condition: 24 - len(codes) for condition, codes in missing.items()},
        "validation": {
            "checks_run": validation["checks_run"],
            "checks_passed": validation["checks_passed"],
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
