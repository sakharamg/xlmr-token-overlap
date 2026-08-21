"""Shared helpers for multi-condition local tokenizer-analysis suites."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import LANGUAGE_CODES, LANGUAGES
from .io import sha256
from .plotting import plot_matrix


def write_frame(frame: pd.DataFrame, base: Path, *, index: bool = False) -> None:
    frame.to_csv(base.with_suffix(".csv"), index=index, float_format="%.8f")
    frame.to_parquet(base.with_suffix(".parquet"), index=index)


def write_all_language_views(condition_dir: Path, title_prefix: str) -> list[str]:
    """Write frozen-axis 24-language views, masking unavailable languages."""

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
        write_frame(full, condition_dir / f"{name}_all_languages", index=True)
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
    write_frame(pd.DataFrame(rows), condition_dir / "language_stats_all_languages")
    (condition_dir / "missing_languages.json").write_text(
        json.dumps({"missing_languages": missing}, indent=2) + "\n",
        encoding="utf-8",
    )

    heatmaps = condition_dir / "heatmaps_all_languages"
    condition = str(stats["condition"].iloc[0])
    plot_matrix(
        full_matrices["type_iou_percent"].to_numpy(),
        LANGUAGES,
        heatmaps / "type_iou_upper.png",
        f"{title_prefix} {condition}: XLM-R token-type IoU (coverage masked)",
        "Token-type IoU (%)",
        upper_triangle=True,
    )
    plot_matrix(
        full_matrices["frequency_overlap_percent"].to_numpy(),
        LANGUAGES,
        heatmaps / "frequency_overlap_directional.png",
        f"{title_prefix} {condition}: directional frequency overlap (coverage masked)",
        "Source occurrences covered (%)",
    )
    plot_matrix(
        full_matrices["shared_token_count"].to_numpy(),
        LANGUAGES,
        heatmaps / "shared_token_count.png",
        f"{title_prefix} {condition}: shared observed XLM-R token types (coverage masked)",
        "Shared token types",
        integer_annotations=True,
    )
    return missing


def pair_vector(frame: pd.DataFrame, codes: Sequence[str]) -> pd.Series:
    values = {
        f"{left}|{right}": float(frame.loc[left, right])
        for left, right in combinations(codes, 2)
        if pd.notna(frame.loc[left, right])
    }
    return pd.Series(values, dtype=float)


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat(
        [left.rename("left"), right.rename("right")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    ranks = aligned.rank(method="average")
    return float(ranks["left"].corr(ranks["right"]))


def write_cross_pass(
    output_dir: Path,
    flores_dir: Path,
    condition_dirs: Mapping[str, Path],
) -> pd.DataFrame:
    """Compare each condition's off-diagonal IoU ordering with FLORES."""

    flores = pd.read_csv(flores_dir / "type_iou_percent.csv", index_col=0)
    flores_vector = pair_vector(flores, LANGUAGE_CODES)
    vectors = {"flores": flores_vector}
    summary_rows = []
    pair_rows = []
    for condition, condition_dir in sorted(condition_dirs.items()):
        frame = pd.read_csv(condition_dir / "type_iou_percent.csv", index_col=0)
        codes = [code for code in LANGUAGE_CODES if code in frame.index]
        condition_vector = pair_vector(frame, codes)
        vectors[condition] = condition_vector
        aligned_flores = flores_vector.reindex(condition_vector.index)
        difference = condition_vector - aligned_flores
        summary_rows.append(
            {
                "condition": condition,
                "languages": len(codes),
                "pairs": len(condition_vector),
                "spearman_vs_flores": spearman_correlation(
                    condition_vector, aligned_flores
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
                    "flores_type_iou_percent": aligned_flores[pair],
                    "condition_type_iou_percent": value,
                    "difference_points": value - aligned_flores[pair],
                }
            )

    summary = pd.DataFrame(summary_rows)
    write_frame(summary, output_dir / "cross_pass_summary")
    pair_comparison = pd.DataFrame(
        pair_rows,
        columns=(
            "condition",
            "language_i",
            "language_j",
            "flores_type_iou_percent",
            "condition_type_iou_percent",
            "difference_points",
        ),
    )
    write_frame(pair_comparison, output_dir / "flores_pairwise_comparison")

    names = ["flores", *sorted(condition_dirs)]
    correlations = pd.DataFrame(np.nan, index=names, columns=names)
    for left in names:
        for right in names:
            common = vectors[left].index.intersection(vectors[right].index)
            correlations.loc[left, right] = spearman_correlation(
                vectors[left].loc[common],
                vectors[right].loc[common],
            )
    correlations.index.name = "condition"
    correlations.to_csv(
        output_dir / "condition_spearman.csv",
        float_format="%.8f",
    )
    return summary


def _available_codes(stats: pd.DataFrame) -> set[str]:
    values = stats["available"]
    if values.dtype == bool:
        mask = values
    else:
        mask = values.astype(str).str.lower().isin({"true", "1", "yes"})
    return set(stats.loc[mask, "language_code"].astype(str))


def validate_condition_suite(
    output_dir: Path,
    condition_dirs: Mapping[str, Path],
    expected_coverage: Mapping[str, set[str] | frozenset[str]],
    suite_name: str,
) -> dict:
    checks = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "condition inventory",
        set(condition_dirs) == set(expected_coverage),
        f"observed={sorted(condition_dirs)}, expected={sorted(expected_coverage)}",
    )
    for condition, condition_dir in sorted(condition_dirs.items()):
        validation = json.loads(
            (condition_dir / "validation_report.json").read_text(encoding="utf-8")
        )
        check(f"{condition}: native validation", validation["status"] == "passed")
        stats = pd.read_csv(condition_dir / "language_stats_all_languages.csv")
        check(
            f"{condition}: 24 language-stat rows",
            len(stats) == len(LANGUAGE_CODES),
        )
        supported = _available_codes(stats)
        expected = set(expected_coverage[condition])
        check(
            f"{condition}: coverage",
            supported == expected,
            f"observed={sorted(supported)}, expected={sorted(expected)}",
        )
        for name in (
            "type_iou_percent",
            "frequency_overlap_percent",
            "shared_token_count",
        ):
            frame = pd.read_csv(
                condition_dir / f"{name}_all_languages.csv",
                index_col=0,
            )
            check(f"{condition}/{name}: 24x24", frame.shape == (24, 24))
            missing = [code for code in LANGUAGE_CODES if code not in supported]
            masked = all(
                frame.loc[code].isna().all() and frame[code].isna().all()
                for code in missing
            )
            check(f"{condition}/{name}: missing coverage masked", masked)

    failures = [item for item in checks if not item["passed"]]
    report = {
        "suite": suite_name,
        "status": "passed" if not failures else "failed",
        "checks_run": len(checks),
        "checks_passed": len(checks) - len(failures),
        "failures": failures,
        "checks": checks,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise ValueError(
            f"{suite_name} validation failed: "
            + ", ".join(item["name"] for item in failures)
        )
    return report


def refresh_manifest(condition_dir: Path) -> None:
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
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
