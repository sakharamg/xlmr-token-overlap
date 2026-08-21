"""Full-corpus loaders for the local Pass-3 and SQE canonical datasets.

Every accepted text cell is passed to the exact XLM-R tokenizer without
character clipping, tokenizer truncation, row sampling, or token-budget
sampling. Natural corpus-size differences are retained and reported.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .constants import LANGUAGE_CODES, LANGUAGES
from .io import Record, portable_path, sha256
from .suite_utils import (
    refresh_manifest,
    validate_condition_suite,
    write_all_language_views,
    write_cross_pass,
    write_frame,
)


LABEL_TO_CODE = {language.label: language.code for language in LANGUAGES}

SQE_LOCALE_ALIASES = {
    "arabic": "AR",
    "arabic_saudi_arabia": "AR",
    "german": "DE",
    "german_germany": "DE",
    "english": "EN",
    "english_united_states": "EN",
    "spanish": "ES",
    "spanish_spain": "ES",
    "french": "FR",
    "french_france": "FR",
    "gujarati": "GU",
    "gujarati_india": "GU",
    "hindi": "HI",
    "hindi_india": "HI",
    "hungarian": "HU",
    "hungarian_hungary": "HU",
    "indonesian": "ID",
    "indonesian_indonesia": "ID",
    "italian": "IT",
    "italian_italy": "IT",
    "japanese": "JA",
    "japanese_japan": "JA",
    "korean": "KO",
    "korean_republic_of_korea": "KO",
    "malay": "MS",
    "malay_malaysia": "MS",
    "dutch": "NL",
    "dutch_netherlands": "NL",
    "polish": "PL",
    "polish_poland": "PL",
    "portuguese": "PT",
    "portuguese_brazil": "PT",
    "romanian": "RO",
    "romanian_romania": "RO",
    "russian": "RU",
    "russian_russian_federation": "RU",
    "swedish": "SV",
    "swedish_sweden": "SV",
    "thai": "TH",
    "thai_thailand": "TH",
    "tagalog": "TL",
    "filipino": "TL",
    "filipino_philippines": "TL",
    "turkish": "TR",
    "turkish_turkey": "TR",
    "vietnamese": "VI",
    "vietnamese_viet_nam": "VI",
    "chinese": "ZH",
    "chinese_simplified": "ZH",
    "chinese_simplified_china": "ZH",
}

SQE_VARIANTS = {
    "tc.xlsx": "standard",
    "contextual_tc.xlsx": "contextual",
    "contextual_drop_time_tc.xlsx": "contextual_drop_time",
}

EXPECTED_SQE_LABEL_COVERAGE = {
    "calendar": frozenset({"KO"}),
    "call_recording": frozenset({"KO"}),
    "notes": frozenset(
        {
            "AR",
            "DE",
            "EN",
            "ES",
            "FR",
            "HI",
            "ID",
            "IT",
            "JA",
            "KO",
            "PL",
            "PT",
            "RU",
            "TH",
            "VI",
            "ZH",
        }
    ),
    "reminder": frozenset({"KO"}),
    "settings": frozenset(LABEL_TO_CODE),
    "voice_recording": frozenset({"KO"}),
}

EXPECTED_SQE_VARIANTS = {
    "calendar": frozenset({"standard", "contextual", "contextual_drop_time"}),
    "call_recording": frozenset({"standard", "contextual", "contextual_drop_time"}),
    "notes": frozenset({"standard", "contextual", "contextual_drop_time"}),
    "reminder": frozenset({"standard", "contextual", "contextual_drop_time"}),
    "settings": frozenset({"standard"}),
    "voice_recording": frozenset({"standard", "contextual", "contextual_drop_time"}),
}


@dataclass
class CorpusCollection:
    records: list[Record]
    contributions: dict[tuple[str, str, str, str, str], dict[str, int]]

    @classmethod
    def empty(cls) -> "CorpusCollection":
        return cls(records=[], contributions={})

    def add(
        self,
        *,
        condition: str,
        language_code: str,
        text: Any,
        example_id: str,
        split: str,
        dataset: str,
        role: str,
        source: str,
    ) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Expected a non-empty text string for {source}/{example_id}/{role}"
            )
        if language_code not in LANGUAGE_CODES:
            raise ValueError(f"Unsupported language code: {language_code}")
        self.records.append(
            Record(
                condition=condition,
                language_code=language_code,
                text=text,
                example_id=example_id,
                split=split,
            )
        )
        key = (condition, language_code, dataset, role, source)
        row = self.contributions.setdefault(
            key,
            {"examples": 0, "characters": 0},
        )
        row["examples"] += 1
        row["characters"] += len(text)

    def clone_condition(
        self,
        *,
        source_conditions: Iterable[str],
        target_condition: str,
    ) -> None:
        sources = set(source_conditions)
        original = [
            record for record in self.records if record.condition in sources
        ]
        for record in original:
            dataset, role = _dataset_role_from_split(record.split)
            self.add(
                condition=target_condition,
                language_code=record.language_code,
                text=record.text,
                example_id=f"{target_condition}:{record.example_id}",
                split=record.split,
                dataset=dataset,
                role=role,
                source="derived-union",
            )

    def contributions_frame(self) -> pd.DataFrame:
        rows = []
        for (
            condition,
            language_code,
            dataset,
            role,
            source,
        ), counts in sorted(self.contributions.items()):
            rows.append(
                {
                    "condition": condition,
                    "language_code": language_code,
                    "dataset": dataset,
                    "role": role,
                    "source": source,
                    **counts,
                }
            )
        return pd.DataFrame(rows)


def _dataset_role_from_split(split: str) -> tuple[str, str]:
    parts = split.split("|")
    if len(parts) < 2:
        return split, "text"
    return parts[0], parts[-1]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_sqe_locale(value: str) -> str:
    upper = value.upper()
    if upper in LABEL_TO_CODE:
        return LABEL_TO_CODE[upper]
    normalized = _normalize_name(value)
    label = SQE_LOCALE_ALIASES.get(normalized)
    if label is None:
        for alias, candidate in SQE_LOCALE_ALIASES.items():
            if normalized.startswith(alias + "_") or normalized.endswith("_" + alias):
                label = candidate
                break
    if label is None:
        raise ValueError(f"Unrecognized SQE locale directory: {value}")
    return LABEL_TO_CODE[label]


def _domain_key(directory_name: str) -> str:
    normalized = _normalize_name(directory_name)
    if normalized.startswith("sqe_"):
        normalized = normalized[4:]
    if normalized.endswith("_search"):
        normalized = normalized[:-7]
    if normalized not in EXPECTED_SQE_LABEL_COVERAGE:
        raise ValueError(f"Unrecognized SQE domain directory: {directory_name}")
    return normalized


def _required_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    path: Path,
    sheet: str | None = None,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        location = f"{path}#{sheet}" if sheet else str(path)
        raise ValueError(
            f"Missing columns in {location}: {missing}; "
            f"observed={list(frame.columns)}"
        )


def _file_entry(
    path: Path,
    root: Path,
    frame: pd.DataFrame,
    *,
    sheet: str | None = None,
) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": len(frame),
        "columns": [str(column) for column in frame.columns],
        "sheet": sheet,
    }


def _id_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_gt_ids(value: Any) -> list[str]:
    """Parse SQE gt_ids cells without assuming one serialization format."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, Mapping):
        for key in (
            "gt_ids",
            "ids",
            "id",
            "doc_ids",
            "document_ids",
            "relevant_ids",
        ):
            if key in value:
                return parse_gt_ids(value[key])
        return [_id_key(key) for key in value]
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(parse_gt_ids(item))
        return [item for item in result if item]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        key = _id_key(value)
        return [key] if key else []

    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if parsed != value:
            return parse_gt_ids(parsed)
    if any(separator in text for separator in (",", ";", "|", "\n")):
        return [
            item.strip()
            for item in re.split(r"[,;|\n]+", text)
            if item.strip()
        ]
    return [text]


def _sequence_digest(rows: Iterable[tuple[Any, ...]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, default=str).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def collect_pass3_records(
    datasets_root: Path,
) -> tuple[CorpusCollection, dict[str, Any]]:
    """Load every text from translated STS and Belebele retrieval."""

    root = datasets_root.resolve()
    collection = CorpusCollection.empty()
    source_files = []
    sts_audit: dict[str, Any] = {}
    retrieval_audit: dict[str, Any] = {}

    for label, language_code in sorted(LABEL_TO_CODE.items()):
        sts_path = root / "sts" / label / "canonical" / "data.parquet"
        if not sts_path.is_file():
            raise FileNotFoundError(f"Missing Pass-3 STS file: {sts_path}")
        sts = pd.read_parquet(sts_path)
        _required_columns(sts, ("sentence1", "sentence2", "score"), sts_path)
        numeric_scores = pd.to_numeric(sts["score"], errors="raise")
        source_files.append(_file_entry(sts_path, root, sts))
        sts_audit[language_code] = {
            "rows": len(sts),
            "duplicate_pairs": int(
                sts.duplicated(["sentence1", "sentence2"]).sum()
            ),
            "score_min": float(numeric_scores.min()),
            "score_max": float(numeric_scores.max()),
            "score_sequence_sha256": _sequence_digest(
                (score,) for score in numeric_scores.tolist()
            ),
        }
        for row_index, row in sts.iterrows():
            for field in ("sentence1", "sentence2"):
                collection.add(
                    condition="sts",
                    language_code=language_code,
                    text=row[field],
                    example_id=f"sts:{label}:{row_index}:{field}",
                    split=f"sts|{field}",
                    dataset="sts",
                    role=field,
                    source=f"sts/{label}/canonical/data.parquet",
                )

        canonical = root / "belebele_retrieval" / label / "canonical"
        query_path = canonical / "queries.parquet"
        corpus_path = canonical / "corpus.parquet"
        qrels_path = canonical / "qrels.parquet"
        for path in (query_path, corpus_path, qrels_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing Belebele file: {path}")
        queries = pd.read_parquet(query_path)
        corpus = pd.read_parquet(corpus_path)
        qrels = pd.read_parquet(qrels_path)
        _required_columns(queries, ("id", "text"), query_path)
        _required_columns(corpus, ("id", "text"), corpus_path)
        _required_columns(qrels, ("query-id", "corpus-id", "score"), qrels_path)
        source_files.extend(
            (
                _file_entry(query_path, root, queries),
                _file_entry(corpus_path, root, corpus),
                _file_entry(qrels_path, root, qrels),
            )
        )

        query_ids = [_id_key(value) for value in queries["id"]]
        corpus_ids = [_id_key(value) for value in corpus["id"]]
        if any(not value for value in query_ids):
            raise ValueError(f"Empty query ID in {query_path}")
        if any(not value for value in corpus_ids):
            raise ValueError(f"Empty corpus ID in {corpus_path}")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError(f"Duplicate query IDs in {query_path}")
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ValueError(f"Duplicate corpus IDs in {corpus_path}")

        qrel_query_ids = {_id_key(value) for value in qrels["query-id"]}
        qrel_corpus_ids = {_id_key(value) for value in qrels["corpus-id"]}
        orphan_queries = qrel_query_ids - set(query_ids)
        orphan_corpus = qrel_corpus_ids - set(corpus_ids)
        queries_without_qrels = set(query_ids) - qrel_query_ids
        if orphan_queries or orphan_corpus or queries_without_qrels:
            raise ValueError(
                f"Belebele integrity failure for {label}: "
                f"orphan_queries={len(orphan_queries)}, "
                f"orphan_corpus={len(orphan_corpus)}, "
                f"queries_without_qrels={len(queries_without_qrels)}"
            )
        retrieval_audit[language_code] = {
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": len(qrels),
            "orphan_query_ids": 0,
            "orphan_corpus_ids": 0,
            "queries_without_qrels": 0,
            "unreferenced_corpus_ids": len(set(corpus_ids) - qrel_corpus_ids),
        }
        for row_index, row in queries.iterrows():
            collection.add(
                condition="retrieval",
                language_code=language_code,
                text=row["text"],
                example_id=f"belebele:{label}:query:{_id_key(row['id']) or row_index}",
                split="belebele_retrieval|query",
                dataset="belebele_retrieval",
                role="query",
                source=f"belebele_retrieval/{label}/canonical/queries.parquet",
            )
        for row_index, row in corpus.iterrows():
            collection.add(
                condition="retrieval",
                language_code=language_code,
                text=row["text"],
                example_id=f"belebele:{label}:corpus:{_id_key(row['id']) or row_index}",
                split="belebele_retrieval|corpus",
                dataset="belebele_retrieval",
                role="corpus",
                source=f"belebele_retrieval/{label}/canonical/corpus.parquet",
            )

    collection.clone_condition(
        source_conditions=("sts", "retrieval"),
        target_condition="overall",
    )
    coverage = {
        condition: sorted(
            {
                record.language_code
                for record in collection.records
                if record.condition == condition
            }
        )
        for condition in ("sts", "retrieval", "overall")
    }
    expected = set(LANGUAGE_CODES)
    for condition, codes in coverage.items():
        if set(codes) != expected:
            raise ValueError(
                f"Pass-3 {condition} coverage mismatch: "
                f"missing={sorted(expected - set(codes))}"
            )

    audit = {
        "dataset": "balanced translated STS + Belebele retrieval",
        "analysis_scope": "all non-empty text cells",
        "sampling": "none",
        "row_limit": None,
        "character_limit": None,
        "tokenizer_truncation": False,
        "conditions": coverage,
        "sts": sts_audit,
        "retrieval": retrieval_audit,
        "source_files": source_files,
    }
    return collection, audit


def collect_sqe_records(
    datasets_root: Path,
    *,
    domains: Sequence[str] | None = None,
    allow_coverage_drift: bool = False,
    strict_ground_truth: bool = False,
) -> tuple[CorpusCollection, dict[str, Any], dict[str, set[str]]]:
    """Load every SQE data text and test-case query as independent conditions."""

    root = datasets_root.resolve()
    sqe_root = root / "sqe"
    if not sqe_root.is_dir():
        raise FileNotFoundError(f"Missing SQE directory: {sqe_root}")
    requested = set(domains or EXPECTED_SQE_LABEL_COVERAGE)
    unknown = requested - EXPECTED_SQE_LABEL_COVERAGE.keys()
    if unknown:
        raise ValueError(f"Unknown SQE domains: {sorted(unknown)}")

    collection = CorpusCollection.empty()
    source_files = []
    integrity: dict[str, Any] = {}
    data_id_integrity: dict[str, Any] = {}
    condition_coverage: defaultdict[str, set[str]] = defaultdict(set)
    observed_domain_coverage: defaultdict[str, set[str]] = defaultdict(set)
    observed_variants: defaultdict[str, set[str]] = defaultdict(set)

    domain_dirs = {}
    for path in sorted(sqe_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("sqe_"):
            continue
        key = _domain_key(path.name)
        if key in requested:
            domain_dirs[key] = path
    missing_domains = requested - domain_dirs.keys()
    if missing_domains:
        raise FileNotFoundError(
            f"Missing SQE domain directories: {sorted(missing_domains)}"
        )

    for domain, domain_dir in sorted(domain_dirs.items()):
        for locale_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            language_code = normalize_sqe_locale(locale_dir.name)
            label = next(
                language.label
                for language in LANGUAGES
                if language.code == language_code
            )
            observed_domain_coverage[domain].add(label)
            canonical = locale_dir / "canonical"
            data_path = canonical / "data.xlsx"
            if not data_path.is_file():
                raise FileNotFoundError(f"Missing SQE data workbook: {data_path}")
            data = pd.read_excel(
                data_path,
                sheet_name="data",
                engine="openpyxl",
            )
            _required_columns(data, ("id", "text"), data_path, "data")
            source_files.append(
                _file_entry(data_path, root, data, sheet="data")
            )
            data_ids = [_id_key(value) for value in data["id"]]
            nonempty_data_ids = [value for value in data_ids if value]
            data_id_counts = Counter(nonempty_data_ids)
            empty_data_id_rows = len(data_ids) - len(nonempty_data_ids)
            duplicate_data_id_values = sum(
                count > 1 for count in data_id_counts.values()
            )
            duplicate_data_id_extra_rows = sum(
                count - 1
                for count in data_id_counts.values()
                if count > 1
            )
            has_data_id_warning = bool(
                empty_data_id_rows or duplicate_data_id_extra_rows
            )
            if strict_ground_truth and has_data_id_warning:
                raise ValueError(
                    f"SQE data-ID integrity failure in {data_path}: "
                    f"empty_data_id_rows={empty_data_id_rows}, "
                    f"duplicate_data_id_values={duplicate_data_id_values}, "
                    "duplicate_data_id_extra_rows="
                    f"{duplicate_data_id_extra_rows}"
                )
            if has_data_id_warning:
                print(
                    "[warning] SQE data-ID metadata anomaly in "
                    f"{data_path}: empty_data_id_rows={empty_data_id_rows}, "
                    f"duplicate_data_id_values={duplicate_data_id_values}, "
                    "duplicate_data_id_extra_rows="
                    f"{duplicate_data_id_extra_rows}; retaining every "
                    "non-empty data text for tokenizer analysis"
                )
            data_id_set = set(nonempty_data_ids)
            data_id_integrity[f"{domain}:{language_code}"] = {
                "status": "warning" if has_data_id_warning else "passed",
                "data_rows": len(data),
                "nonempty_data_ids": len(nonempty_data_ids),
                "unique_nonempty_data_ids": len(data_id_set),
                "empty_data_id_rows": empty_data_id_rows,
                "duplicate_data_id_values": duplicate_data_id_values,
                "duplicate_data_id_extra_rows": duplicate_data_id_extra_rows,
            }

            found_variant = False
            for filename, variant in SQE_VARIANTS.items():
                tc_path = canonical / filename
                if not tc_path.is_file():
                    continue
                found_variant = True
                observed_variants[domain].add(variant)
                condition = f"{domain}_{variant}"
                test_cases = pd.read_excel(
                    tc_path,
                    sheet_name="tc",
                    engine="openpyxl",
                )
                _required_columns(test_cases, ("query", "gt_ids"), tc_path, "tc")
                source_files.append(
                    _file_entry(tc_path, root, test_cases, sheet="tc")
                )

                missing_references: set[str] = set()
                missing_reference_occurrences = 0
                rows_with_missing_references = 0
                empty_ground_truth_rows = 0
                parsed_reference_count = 0
                for value in test_cases["gt_ids"]:
                    references = parse_gt_ids(value)
                    if not references:
                        empty_ground_truth_rows += 1
                    parsed_reference_count += len(references)
                    row_missing = [
                        reference
                        for reference in references
                        if reference not in data_id_set
                    ]
                    if row_missing:
                        rows_with_missing_references += 1
                        missing_reference_occurrences += len(row_missing)
                        missing_references.update(row_missing)
                has_ground_truth_warning = bool(
                    empty_ground_truth_rows or missing_reference_occurrences
                )
                if strict_ground_truth and has_ground_truth_warning:
                    raise ValueError(
                        f"SQE ground-truth integrity failure in {tc_path}: "
                        f"empty_gt_rows={empty_ground_truth_rows}, "
                        f"missing_reference_occurrences="
                        f"{missing_reference_occurrences}"
                    )
                if has_ground_truth_warning:
                    print(
                        "[warning] SQE ground-truth metadata anomaly in "
                        f"{tc_path}: empty_gt_rows={empty_ground_truth_rows}, "
                        "missing_reference_occurrences="
                        f"{missing_reference_occurrences}; retaining every "
                        "non-empty query for tokenizer analysis"
                    )

                source_data = (
                    f"sqe/{domain_dir.name}/{locale_dir.name}/canonical/data.xlsx"
                )
                for row_index, row in data.iterrows():
                    collection.add(
                        condition=condition,
                        language_code=language_code,
                        text=row["text"],
                        example_id=(
                            f"sqe:{domain}:{variant}:{label}:data:"
                            f"{row_index}:"
                            f"{_id_key(row['id']) or 'no-id'}"
                        ),
                        split=f"sqe_{domain}|{variant}|data",
                        dataset=f"sqe_{domain}",
                        role="data",
                        source=source_data,
                    )
                source_tc = (
                    f"sqe/{domain_dir.name}/{locale_dir.name}/canonical/{filename}"
                )
                for row_index, row in test_cases.iterrows():
                    collection.add(
                        condition=condition,
                        language_code=language_code,
                        text=row["query"],
                        example_id=(
                            f"sqe:{domain}:{variant}:{label}:query:{row_index}"
                        ),
                        split=f"sqe_{domain}|{variant}|query",
                        dataset=f"sqe_{domain}",
                        role="query",
                        source=source_tc,
                    )
                condition_coverage[condition].add(language_code)
                integrity[f"{condition}:{language_code}"] = {
                    "status": (
                        "warning" if has_ground_truth_warning else "passed"
                    ),
                    "data_rows": len(data),
                    "query_rows": len(test_cases),
                    "data_ids": len(data_id_set),
                    "parsed_gt_references": parsed_reference_count,
                    "empty_gt_rows": empty_ground_truth_rows,
                    "rows_with_missing_gt_references": (
                        rows_with_missing_references
                    ),
                    "missing_gt_reference_occurrences": (
                        missing_reference_occurrences
                    ),
                    "unique_missing_gt_references": len(missing_references),
                }
            if not found_variant:
                raise FileNotFoundError(
                    f"No SQE test-case workbook found in {canonical}"
                )

    if not allow_coverage_drift:
        for domain in requested:
            expected_labels = set(EXPECTED_SQE_LABEL_COVERAGE[domain])
            observed_labels = observed_domain_coverage[domain]
            if observed_labels != expected_labels:
                raise ValueError(
                    f"SQE {domain} locale coverage drift: "
                    f"missing={sorted(expected_labels - observed_labels)}, "
                    f"unexpected={sorted(observed_labels - expected_labels)}"
                )
            expected_variants = set(EXPECTED_SQE_VARIANTS[domain])
            if observed_variants[domain] != expected_variants:
                raise ValueError(
                    f"SQE {domain} variant drift: "
                    f"missing={sorted(expected_variants - observed_variants[domain])}, "
                    f"unexpected={sorted(observed_variants[domain] - expected_variants)}"
                )

    expected_coverage = {
        condition: set(codes)
        for condition, codes in condition_coverage.items()
    }
    warning_conditions = sorted(
        key
        for key, details in integrity.items()
        if details["status"] == "warning"
    )
    ground_truth_summary = {
        "strict_mode": strict_ground_truth,
        "conditions_with_warnings": len(warning_conditions),
        "warning_conditions": warning_conditions,
        "empty_gt_rows": sum(
            details["empty_gt_rows"] for details in integrity.values()
        ),
        "rows_with_missing_gt_references": sum(
            details["rows_with_missing_gt_references"]
            for details in integrity.values()
        ),
        "missing_gt_reference_occurrences": sum(
            details["missing_gt_reference_occurrences"]
            for details in integrity.values()
        ),
    }
    data_id_warning_slices = sorted(
        key
        for key, details in data_id_integrity.items()
        if details["status"] == "warning"
    )
    data_id_summary = {
        "strict_mode": strict_ground_truth,
        "slices_with_warnings": len(data_id_warning_slices),
        "warning_slices": data_id_warning_slices,
        "empty_data_id_rows": sum(
            details["empty_data_id_rows"]
            for details in data_id_integrity.values()
        ),
        "duplicate_data_id_values": sum(
            details["duplicate_data_id_values"]
            for details in data_id_integrity.values()
        ),
        "duplicate_data_id_extra_rows": sum(
            details["duplicate_data_id_extra_rows"]
            for details in data_id_integrity.values()
        ),
    }
    audit = {
        "dataset": "SQE canonical workbooks",
        "analysis_scope": "all non-empty data.text and test-case query cells",
        "sampling": "none",
        "row_limit": None,
        "character_limit": None,
        "tokenizer_truncation": False,
        "condition_coverage": {
            condition: [
                code for code in LANGUAGE_CODES if code in codes
            ]
            for condition, codes in sorted(condition_coverage.items())
        },
        "domain_label_coverage": {
            domain: sorted(labels)
            for domain, labels in sorted(observed_domain_coverage.items())
        },
        "variants": {
            domain: sorted(variants)
            for domain, variants in sorted(observed_variants.items())
        },
        "data_id_summary": data_id_summary,
        "data_id_integrity": data_id_integrity,
        "ground_truth_summary": ground_truth_summary,
        "ground_truth_integrity": integrity,
        "source_files": source_files,
    }
    return collection, audit, expected_coverage


def _write_corpus_audit(
    output_dir: Path,
    collection: CorpusCollection,
    audit: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(
        collection.contributions_frame(),
        output_dir / "source_contributions",
    )
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_suite(
    *,
    suite_name: str,
    title_prefix: str,
    collection: CorpusCollection,
    audit: Mapping[str, Any],
    expected_coverage: Mapping[str, set[str] | frozenset[str]],
    tokenizer_json: Path,
    output_dir: Path,
    flores_dir: Path,
) -> tuple[dict[str, Path], pd.DataFrame, dict]:
    from .pipeline import run_analysis

    provenance = {
        "dataset": audit["dataset"],
        "scope": audit["analysis_scope"],
        "sampling": "none; every valid source row is retained",
        "full_text": True,
        "character_limit": None,
        "tokenizer_truncation": False,
        "dataset_root": audit.get("dataset_root"),
        "raw_text_committed": False,
    }
    destinations = run_analysis(
        collection.records,
        tokenizer_json,
        output_dir,
        provenance,
        require_all_languages=False,
        force_condition_subdirs=True,
    )
    _write_corpus_audit(output_dir, collection, audit)
    for path in destinations.values():
        write_all_language_views(path, title_prefix)
    cross = write_cross_pass(output_dir, flores_dir, destinations)
    validation = validate_condition_suite(
        output_dir,
        destinations,
        expected_coverage,
        suite_name,
    )
    for path in destinations.values():
        refresh_manifest(path)
    return destinations, cross, validation


def _strongest_pair(condition_dir: Path) -> str:
    summary = json.loads(
        (condition_dir / "summary.json").read_text(encoding="utf-8")
    )
    pair = summary.get("strongest_type_iou_pair")
    if not pair:
        return "NA (fewer than two languages)"
    return (
        f"{pair['language_i']}–{pair['language_j']} "
        f"({pair['type_iou_percent']:.2f}%)"
    )


def _write_pass3_report(
    output_dir: Path,
    audit: Mapping[str, Any],
    destinations: Mapping[str, Path],
    cross: pd.DataFrame,
    validation: Mapping[str, Any],
) -> None:
    cross_by_condition = cross.set_index("condition").to_dict("index")
    rows = []
    for condition in ("overall", "sts", "retrieval"):
        comparison = cross_by_condition[condition]
        rows.append(
            f"| {condition} | 24 | {_strongest_pair(destinations[condition])} | "
            f"{comparison['spearman_vs_flores']:.3f} |"
        )
    sts_rows = {
        code: details["rows"] for code, details in audit["sts"].items()
    }
    text = f"""# Pass 3: local balanced STS and retrieval tokenizer overlap

Every non-empty source text is analyzed in full. There is no character limit,
tokenizer truncation, row sampling, or token-budget sampling. The natural
corpus-size differences remain visible in per-language statistics.

## Conditions

| Condition | Languages | Strongest pair | Spearman vs FLORES |
|---|---:|---|---:|
{chr(10).join(rows)}

- STS text roles: sentence1 and sentence2.
- Retrieval text roles: each query and each corpus item exactly once.
- Overall: the complete STS and retrieval record union.
- Qrels validate retrieval linkage but are not repeated as text.
- STS source row counts: {json.dumps(sts_rows, sort_keys=True)}.

STS has no stable ID column and row/score order differs across some translated
locales. That prevents a content-aligned parallel interpretation, but does not
invalidate this full-corpus tokenizer-distribution condition.

All {validation['checks_run']} aggregate suite checks pass in addition to the
native matrix invariants for each condition. See corpus_manifest.json and
source_contributions.csv for exact source checksums, roles, row counts, and
character counts. Raw source text is not written to the results.
"""
    (output_dir / "REPORT.md").write_text(text, encoding="utf-8")


def _write_sqe_report(
    output_dir: Path,
    audit: Mapping[str, Any],
    destinations: Mapping[str, Path],
    cross: pd.DataFrame,
    validation: Mapping[str, Any],
) -> None:
    cross_by_condition = cross.set_index("condition").to_dict("index")
    rows = []
    for condition, path in sorted(destinations.items()):
        languages = len(audit["condition_coverage"][condition])
        spearman = cross_by_condition[condition]["spearman_vs_flores"]
        spearman_text = f"{spearman:.3f}" if pd.notna(spearman) else "NA"
        rows.append(
            f"| {condition} | {languages} | {_strongest_pair(path)} | "
            f"{spearman_text} |"
        )
    ground_truth = audit["ground_truth_summary"]
    data_ids = audit["data_id_summary"]
    text = f"""# SQE tokenizer-overlap suite

Every non-empty data.text and test-case query cell is analyzed in full. There
is no character limit, tokenizer truncation, row sampling, or token-budget
sampling. The gt_ids field is used only to validate query-to-data linkage.
Ground-truth metadata warnings are non-blocking for tokenizer analysis: the
audit found {ground_truth['empty_gt_rows']} empty gt_ids row(s) and
{ground_truth['missing_gt_reference_occurrences']} missing-reference
occurrence(s) across {ground_truth['conditions_with_warnings']}
condition-language slice(s). Every non-empty query is retained.
The data-ID audit found {data_ids['empty_data_id_rows']} empty ID row(s) and
{data_ids['duplicate_data_id_extra_rows']} extra row(s) carrying duplicate IDs
across {data_ids['slices_with_warnings']} domain-language slice(s). IDs are
linkage metadata, so every non-empty data text is retained without
deduplication.

| Condition | Languages | Strongest pair | Spearman vs FLORES |
|---|---:|---|---:|
{chr(10).join(rows)}

Each domain and query variant is independent. There is intentionally no SQE
overall condition: Korean has six domains, the notes domain has 16 languages,
and settings has 24, so pooling their union would confound language with
domain. The settings_standard condition is the complete 24-language SQE view;
notes conditions are emitted as coverage-masked 24×24 matrices. Korean-only
conditions retain full tokenization diagnostics but have no off-diagonal
language pairs.

All {validation['checks_run']} aggregate suite checks pass in addition to the
native matrix invariants for each condition. See corpus_manifest.json and
source_contributions.csv for exact source checksums, roles, rows, characters,
coverage, and gt_ids integrity. Raw workbook text is not written to results.
"""
    (output_dir / "REPORT.md").write_text(text, encoding="utf-8")


def _result_payload(
    output_dir: Path,
    destinations: Mapping[str, Path],
    validation: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "status": "passed",
        "analysis_scope": audit["analysis_scope"],
        "sampling": audit["sampling"],
        "character_limit": audit["character_limit"],
        "tokenizer_truncation": audit["tokenizer_truncation"],
        "conditions": {
            condition: portable_path(path)
            for condition, path in sorted(destinations.items())
        },
        "validation": {
            "checks_run": validation["checks_run"],
            "checks_passed": validation["checks_passed"],
        },
    }
    if "ground_truth_summary" in audit:
        result["ground_truth_summary"] = audit["ground_truth_summary"]
    if "data_id_summary" in audit:
        result["data_id_summary"] = audit["data_id_summary"]
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_pass3(
    *,
    datasets_root: Path,
    tokenizer_json: Path,
    output_dir: Path,
    flores_dir: Path,
) -> dict[str, Any]:
    collection, audit = collect_pass3_records(datasets_root)
    audit["dataset_root"] = portable_path(datasets_root)
    expected = {condition: set(LANGUAGE_CODES) for condition in ("sts", "retrieval", "overall")}
    destinations, cross, validation = _run_suite(
        suite_name="pass3",
        title_prefix="Pass 3",
        collection=collection,
        audit=audit,
        expected_coverage=expected,
        tokenizer_json=tokenizer_json,
        output_dir=output_dir,
        flores_dir=flores_dir,
    )
    _write_pass3_report(output_dir, audit, destinations, cross, validation)
    return _result_payload(
        output_dir,
        destinations,
        validation,
        audit,
    )


def run_sqe(
    *,
    datasets_root: Path,
    tokenizer_json: Path,
    output_dir: Path,
    flores_dir: Path,
    domains: Sequence[str] | None = None,
    allow_coverage_drift: bool = False,
    strict_ground_truth: bool = False,
) -> dict[str, Any]:
    collection, audit, expected = collect_sqe_records(
        datasets_root,
        domains=domains,
        allow_coverage_drift=allow_coverage_drift,
        strict_ground_truth=strict_ground_truth,
    )
    audit["dataset_root"] = portable_path(datasets_root)
    destinations, cross, validation = _run_suite(
        suite_name="sqe",
        title_prefix="SQE",
        collection=collection,
        audit=audit,
        expected_coverage=expected,
        tokenizer_json=tokenizer_json,
        output_dir=output_dir,
        flores_dir=flores_dir,
    )
    _write_sqe_report(output_dir, audit, destinations, cross, validation)
    return _result_payload(
        output_dir,
        destinations,
        validation,
        audit,
    )
