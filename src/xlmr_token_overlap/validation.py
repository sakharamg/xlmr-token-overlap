"""Mechanical validation of result invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def validate_results(results_dir: Path, write_report: bool = True) -> dict:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    language_stats = pd.read_csv(results_dir / "language_stats.csv")
    codes = language_stats["language_code"].astype(str).tolist()
    n = len(codes)
    iou = _matrix(results_dir / "type_iou_percent.csv")
    upper = _matrix(results_dir / "type_iou_percent_upper.csv")
    shared = _matrix(results_dir / "shared_token_count.csv")
    frequency = _matrix(results_dir / "frequency_overlap_percent.csv")
    pairwise = pd.read_csv(results_dir / "pairwise_long.csv")
    token_frequencies = pd.read_parquet(results_dir / "token_frequencies.parquet")

    for name, frame in (
        ("type IoU", iou),
        ("type IoU upper", upper),
        ("shared count", shared),
        ("frequency overlap", frequency),
    ):
        check(f"{name}: shape", frame.shape == (n, n), f"observed {frame.shape}, expected {(n, n)}")
        check(f"{name}: row order", frame.index.tolist() == codes)
        check(f"{name}: column order", frame.columns.tolist() == codes)

    check("language inventory is unique", len(codes) == len(set(codes)))
    check("type IoU is symmetric", np.allclose(iou, iou.T, atol=1e-10, equal_nan=True))
    check("type IoU diagonal is 100", np.allclose(np.diag(iou), 100.0, atol=1e-10))
    check("type IoU is bounded", np.nanmin(iou.to_numpy()) >= 0 and np.nanmax(iou.to_numpy()) <= 100)
    check("shared count is symmetric", np.array_equal(shared.to_numpy(), shared.to_numpy().T))
    check(
        "shared diagonal equals unique-token counts",
        np.array_equal(
            np.diag(shared).astype(int),
            language_stats["unique_analysis_token_count"].to_numpy(dtype=int),
        ),
    )
    check("frequency diagonal is 100", np.allclose(np.diag(frequency), 100.0, atol=1e-10))
    check(
        "frequency overlap is bounded",
        np.nanmin(frequency.to_numpy()) >= 0 and np.nanmax(frequency.to_numpy()) <= 100,
    )
    lower = upper.to_numpy()[np.tril_indices(n, k=-1)]
    check("upper matrix lower triangle is empty", np.isnan(lower).all())
    upper_indices = np.triu_indices(n)
    check(
        "upper matrix matches full IoU",
        np.allclose(
            upper.to_numpy()[upper_indices], iou.to_numpy()[upper_indices], atol=1e-10
        ),
    )
    check("pairwise long table has n² rows", len(pairwise) == n * n, f"observed {len(pairwise)}")
    check(
        "pairwise directional rows are unique",
        not pairwise.duplicated(["source_language_code", "target_language_code"]).any(),
    )

    token_sums = token_frequencies.groupby("language_code")["count"].sum()
    expected_sums = language_stats.set_index("language_code")["analysis_token_count"]
    check(
        "token-frequency counts sum to language totals",
        token_sums.reindex(codes).astype(int).equals(expected_sums.reindex(codes).astype(int)),
    )
    check(
        "encoded token identity holds",
        np.array_equal(
            language_stats["encoded_token_count"].to_numpy(dtype=int),
            (
                language_stats["analysis_token_count"]
                + language_stats["special_token_count"]
            ).to_numpy(dtype=int),
        ),
    )

    # Ensure the Parquet copies are not stale or schema-divergent.
    parquet_stats = pd.read_parquet(results_dir / "language_stats.parquet")
    parquet_pairs = pd.read_parquet(results_dir / "pairwise_long.parquet")
    check("language Parquet row count matches CSV", len(parquet_stats) == len(language_stats))
    check("pairwise Parquet row count matches CSV", len(parquet_pairs) == len(pairwise))

    failures = [item for item in checks if not item["passed"]]
    report = {
        "status": "passed" if not failures else "failed",
        "checks_run": len(checks),
        "checks_passed": len(checks) - len(failures),
        "failures": failures,
        "checks": checks,
    }
    if write_report:
        (results_dir / "validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if failures:
        names = ", ".join(item["name"] for item in failures)
        raise ValueError(f"Result validation failed: {names}")
    return report

