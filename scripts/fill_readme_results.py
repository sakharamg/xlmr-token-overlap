#!/usr/bin/env python3
"""Fill README Pass 3 and SQE result blocks from private-run summaries.

This script reads only aggregate CSV/JSON outputs. It never reads source
datasets, workbook cells, token-frequency tables, or model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PASS3_START = "<!-- BEGIN GENERATED: PASS3_RESULTS -->"
PASS3_END = "<!-- END GENERATED: PASS3_RESULTS -->"
SQE_START = "<!-- BEGIN GENERATED: SQE_RESULTS -->"
SQE_END = "<!-- END GENERATED: SQE_RESULTS -->"

PASS3_CONDITIONS = (
    ("overall", "Overall"),
    ("sts", "STS"),
    ("retrieval", "Retrieval"),
)
SQE_CONDITIONS = (
    ("settings_standard", "Settings standard"),
    ("notes_standard", "Notes standard"),
    ("notes_contextual", "Notes contextual"),
    ("notes_contextual_drop_time", "Notes contextual-drop-time"),
)


class ResultSyncError(RuntimeError):
    """Raised when a generated result cannot safely populate the README."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResultSyncError(f"Required file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultSyncError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultSyncError(f"Expected a JSON object in {path}")
    return value


def _read_cross_pass(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise ResultSyncError(f"Required file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ResultSyncError(f"Could not read CSV file {path}: {exc}") from exc
    by_condition: dict[str, dict[str, str]] = {}
    for row in rows:
        condition = row.get("condition", "")
        if not condition:
            raise ResultSyncError(f"Missing condition value in {path}")
        if condition in by_condition:
            raise ResultSyncError(f"Duplicate condition {condition!r} in {path}")
        by_condition[condition] = row
    return by_condition


def _language_names(readme: Path) -> dict[str, str]:
    language_key = readme.parent / "results" / "flores" / "language_key.csv"
    if not language_key.is_file():
        return {}
    with language_key.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["language_code"]: row["language"]
            for row in csv.DictReader(handle)
            if row.get("language_code") and row.get("language")
        }


def _number(value: object, digits: int, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}{suffix}"


def _signed_points(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:+.2f} pp"


def _pair_cells(
    summary: Mapping[str, Any], names: Mapping[str, str]
) -> tuple[str, str]:
    pair = summary.get("strongest_type_iou_pair")
    if not isinstance(pair, dict):
        return "NA", "NA"
    left_code = str(pair.get("language_i", ""))
    right_code = str(pair.get("language_j", ""))
    if not left_code or not right_code:
        raise ResultSyncError("Strongest-pair record is missing language codes")
    left = names.get(left_code, left_code)
    right = names.get(right_code, right_code)
    return f"{left}–{right}", _number(pair.get("type_iou_percent"), 2, "%")


def _validation(path: Path) -> tuple[int, int]:
    report = _read_json(path)
    if report.get("status") != "passed":
        raise ResultSyncError(
            f"Refusing to publish a non-passing result: {path} "
            f"has status={report.get('status')!r}"
        )
    try:
        return int(report["checks_passed"]), int(report["checks_run"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultSyncError(f"Validation counts are missing in {path}") from exc


def _condition_row(
    result_dir: Path,
    cross: Mapping[str, Mapping[str, str]],
    condition: str,
    label: str,
    names: Mapping[str, str],
) -> str:
    comparison = cross.get(condition)
    if comparison is None:
        raise ResultSyncError(
            f"Condition {condition!r} is missing from "
            f"{result_dir / 'cross_pass_summary.csv'}"
        )
    summary = _read_json(result_dir / condition / "summary.json")
    pair, iou = _pair_cells(summary, names)
    try:
        languages = int(float(comparison["languages"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultSyncError(
            f"Condition {condition!r} has no valid languages count"
        ) from exc
    return (
        f"| {label} | {languages}/24 | {pair} | {iou} | "
        f"{_number(comparison.get('spearman_vs_flores'), 3)} | "
        f"{_signed_points(comparison.get('mean_signed_difference_points'))} |"
    )


def _table(
    result_dir: Path,
    conditions: Sequence[tuple[str, str]],
    names: Mapping[str, str],
) -> list[str]:
    cross = _read_cross_pass(result_dir / "cross_pass_summary.csv")
    lines = [
        "| Condition | Language coverage | Strongest pair | Type IoU | "
        "Spearman vs. FLORES | Mean IoU Δ vs. FLORES |",
        "|---|---:|---|---:|---:|---:|",
    ]
    lines.extend(
        _condition_row(result_dir, cross, condition, label, names)
        for condition, label in conditions
    )
    return lines


def render_pass3(result_dir: Path, names: Mapping[str, str]) -> str:
    passed, total = _validation(result_dir / "validation_report.json")
    lines = [
        f"The private full-corpus run passes all **{passed}/{total} aggregate "
        "checks**. Its comparative results are:",
        "",
        *_table(result_dir, PASS3_CONDITIONS, names),
    ]
    return "\n".join(lines)


def _audit_value(summary: Mapping[str, Any], group: str, field: str) -> int:
    values = summary.get(group, {})
    if not isinstance(values, dict):
        return 0
    try:
        return int(values.get(field, 0))
    except (TypeError, ValueError):
        return 0


def render_sqe(result_dir: Path, names: Mapping[str, str]) -> str:
    passed, total = _validation(result_dir / "validation_report.json")
    run_summary = _read_json(result_dir / "run_summary.json")
    empty_gt = _audit_value(run_summary, "ground_truth_summary", "empty_gt_rows")
    missing_refs = _audit_value(
        run_summary,
        "ground_truth_summary",
        "missing_gt_reference_occurrences",
    )
    duplicate_ids = _audit_value(
        run_summary,
        "data_id_summary",
        "duplicate_data_id_extra_rows",
    )
    skipped_data = _audit_value(
        run_summary,
        "text_summary",
        "skipped_empty_or_nontext_data_rows",
    )
    skipped_queries = _audit_value(
        run_summary,
        "text_summary",
        "skipped_empty_or_nontext_query_rows",
    )
    lines = [
        f"The private full-text run passes all **{passed}/{total} aggregate "
        "checks**. The metadata audit reports "
        f"**{empty_gt}** empty ground-truth row(s), "
        f"**{missing_refs}** unresolved ground-truth reference occurrence(s), "
        f"and **{duplicate_ids}** duplicate-ID extra row(s). It skipped only "
        f"**{skipped_data}** blank/non-text data cell(s) and "
        f"**{skipped_queries}** blank/non-text query cell(s).",
        "",
        *_table(result_dir, SQE_CONDITIONS, names),
    ]
    return "\n".join(lines)


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ResultSyncError(
            f"README must contain exactly one {start!r} and one {end!r} marker"
        )
    start_index = text.index(start) + len(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + "\n" + body.rstrip() + "\n" + text[end_index:]


def _heatmap_mappings(
    readme: Path, pass3_dir: Path, sqe_dir: Path
) -> list[tuple[Path, Path]]:
    repo_root = readme.parent
    mappings: list[tuple[Path, Path]] = []
    for condition, _ in PASS3_CONDITIONS:
        relative = Path(condition) / "heatmaps_all_languages" / "type_iou_upper.png"
        mappings.append(
            (pass3_dir / relative, repo_root / "results" / "pass3" / relative)
        )
    for condition, _ in SQE_CONDITIONS:
        relative = Path(condition) / "heatmaps_all_languages" / "type_iou_upper.png"
        mappings.append(
            (sqe_dir / relative, repo_root / "results" / "sqe" / relative)
        )
    return mappings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fill README Pass 3/SQE aggregate tables from generated private-run "
            "summaries. No source text is read."
        )
    )
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--pass3-dir", type=Path, default=Path("results/pass3"))
    parser.add_argument("--sqe-dir", type=Path, default=Path("results/sqe"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if README is stale; do not write it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    readme = args.readme.resolve()
    pass3_dir = args.pass3_dir.resolve()
    sqe_dir = args.sqe_dir.resolve()
    try:
        current = readme.read_text(encoding="utf-8")
        names = _language_names(readme)
        updated = _replace_block(
            current,
            PASS3_START,
            PASS3_END,
            render_pass3(pass3_dir, names),
        )
        updated = _replace_block(
            updated,
            SQE_START,
            SQE_END,
            render_sqe(sqe_dir, names),
        )
    except (OSError, ResultSyncError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if current != updated:
            print("README result blocks are stale", file=sys.stderr)
            return 1
        print("README result blocks are current")
    elif current == updated:
        print(f"README result blocks already current: {readme}")
    else:
        readme.write_text(updated, encoding="utf-8")
        print(f"Updated aggregate result blocks: {readme}")

    missing = [
        (source, destination)
        for source, destination in _heatmap_mappings(readme, pass3_dir, sqe_dir)
        if not destination.is_file()
    ]
    if missing:
        print(
            f"warning: {len(missing)} README heatmap destination(s) are missing; "
            "copy these files:",
            file=sys.stderr,
        )
        for source, destination in missing:
            print(f"  {source} -> {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
