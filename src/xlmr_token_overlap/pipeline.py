"""End-to-end condition-aware analysis pipeline."""

from __future__ import annotations

import json
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tokenizers

from . import __version__
from .constants import LANGUAGES
from .io import Record, sha256
from .metrics import ConditionMetrics, compute_metrics
from .plotting import write_heatmaps
from .reporting import build_descriptive_outputs
from .tokenization import AuditedTokenizer
from .validation import validate_results


def _safe_condition(condition: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", condition).strip("._-")
    return value or "condition"


def _write_frame(frame: pd.DataFrame, base: Path, *, index: bool = False) -> None:
    frame.to_csv(base.with_suffix(".csv"), index=index, float_format="%.8f")
    frame.to_parquet(base.with_suffix(".parquet"), index=index)


def _write_outputs(metrics: ConditionMetrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(metrics.language_stats, output_dir / "language_stats")
    _write_frame(metrics.split_stats, output_dir / "split_stats")
    _write_frame(metrics.token_frequencies, output_dir / "token_frequencies")
    _write_frame(metrics.pairwise, output_dir / "pairwise_long")
    _write_frame(metrics.type_iou, output_dir / "type_iou_percent", index=True)
    _write_frame(metrics.type_iou_upper, output_dir / "type_iou_percent_upper", index=True)
    _write_frame(metrics.shared_count, output_dir / "shared_token_count", index=True)
    _write_frame(
        metrics.frequency_overlap,
        output_dir / "frequency_overlap_percent",
        index=True,
    )
    language_key = pd.DataFrame(
        [
            {
                "order": index,
                "language_code": language.code,
                "language": language.name,
                "label": language.label,
                "script": language.script,
                "family": language.family,
                "visual_group": language.visual_group,
            }
            for index, language in enumerate(metrics.languages, start=1)
        ]
    )
    language_key.to_csv(output_dir / "language_key.csv", index=False)
    write_heatmaps(metrics, output_dir)


def _output_checksums(output_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def run_analysis(
    records: list[Record],
    tokenizer_json: Path,
    output_dir: Path,
    provenance: dict,
    *,
    require_all_languages: bool = True,
) -> dict[str, Path]:
    tokenizer = AuditedTokenizer(tokenizer_json)
    by_condition: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_condition[record.condition].append(record)

    destinations: dict[str, Path] = {}
    multiple = len(by_condition) > 1
    for condition, condition_records in sorted(by_condition.items()):
        by_language: dict[str, list[Record]] = defaultdict(list)
        for record in condition_records:
            by_language[record.language_code].append(record)

        observed = set(by_language)
        expected = {language.code for language in LANGUAGES}
        if require_all_languages and observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"Condition {condition!r} does not contain the frozen 24-language inventory; "
                f"missing={missing}, unexpected={unexpected}"
            )
        languages = tuple(language for language in LANGUAGES if language.code in observed)
        if not languages:
            raise ValueError(f"Condition {condition!r} has no supported languages")

        token_counts = {}
        for index, language in enumerate(languages, start=1):
            print(f"[tokenize {index:02d}/{len(languages):02d}] {condition}: {language.code}")
            token_counts[language.code] = tokenizer.count(
                language.code, by_language[language.code]
            )
        metrics = compute_metrics(condition, languages, token_counts, tokenizer)

        destination = output_dir / _safe_condition(condition) if multiple else output_dir
        _write_outputs(metrics, destination)
        summary = build_descriptive_outputs(metrics, destination)
        validation = validate_results(destination, write_report=True)

        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package": {"name": "xlmr-token-overlap", "version": __version__},
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "tokenizers": tokenizers.__version__,
                "pandas": pd.__version__,
            },
            "condition": condition,
            "language_order": [language.code for language in languages],
            "tokenizer": tokenizer.metadata(),
            "overlap_protocol": {
                "analysis_vocabulary": "observed token IDs excluding all tokenizer special IDs",
                "shared_token_count": "|V_i ∩ V_j|",
                "type_iou_percent": "100 × |V_i ∩ V_j| / |V_i ∪ V_j|",
                "frequency_overlap_percent": (
                    "100 × sum_{t in V_i ∩ V_j} c_i(t) / sum_{t in V_i} c_i(t)"
                ),
                "frequency_direction": "matrix row/source i → column/target j",
            },
            "provenance": provenance,
            "summary": summary,
            "validation": {
                "status": validation["status"],
                "checks_run": validation["checks_run"],
            },
        }
        manifest["outputs"] = _output_checksums(destination)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        destinations[condition] = destination
        print(f"[complete] {condition}: {destination}")

    return destinations
