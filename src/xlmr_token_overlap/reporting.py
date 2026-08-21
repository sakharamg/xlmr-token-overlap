"""Descriptive summaries that deliberately precede performance correlation."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from .constants import LANGUAGE_BY_CODE


def _relation(left: str, right: str) -> str:
    left_latin = LANGUAGE_BY_CODE[left].script == "Latin"
    right_latin = LANGUAGE_BY_CODE[right].script == "Latin"
    if left_latin and right_latin:
        return "within Latin-script"
    if left_latin != right_latin:
        return "Latin ↔ non-Latin"
    return "within non-Latin scripts"


def build_descriptive_outputs(metrics, output_dir: Path) -> dict:
    codes = [language.code for language in metrics.languages]
    unordered_rows: list[dict] = []
    asymmetry_rows: list[dict] = []
    for left, right in combinations(codes, 2):
        forward = float(metrics.frequency_overlap.loc[left, right])
        reverse = float(metrics.frequency_overlap.loc[right, left])
        unordered_rows.append(
            {
                "language_i": left,
                "language_j": right,
                "relation": _relation(left, right),
                "type_iou_percent": float(metrics.type_iou.loc[left, right]),
                "shared_token_count": int(metrics.shared_count.loc[left, right]),
                "frequency_i_to_j_percent": forward,
                "frequency_j_to_i_percent": reverse,
            }
        )
        asymmetry_rows.append(
            {
                "language_i": left,
                "language_j": right,
                "frequency_i_to_j_percent": forward,
                "frequency_j_to_i_percent": reverse,
                "absolute_gap_points": abs(forward - reverse),
            }
        )
    unordered = pd.DataFrame(
        unordered_rows,
        columns=(
            "language_i",
            "language_j",
            "relation",
            "type_iou_percent",
            "shared_token_count",
            "frequency_i_to_j_percent",
            "frequency_j_to_i_percent",
        ),
    )
    asymmetry = pd.DataFrame(
        asymmetry_rows,
        columns=(
            "language_i",
            "language_j",
            "frequency_i_to_j_percent",
            "frequency_j_to_i_percent",
            "absolute_gap_points",
        ),
    ).sort_values("absolute_gap_points", ascending=False)
    top_pairs = unordered.sort_values("type_iou_percent", ascending=False).head(30)
    if unordered.empty:
        relation_summary = pd.DataFrame(
            columns=(
                "relation",
                "pairs",
                "mean_type_iou_percent",
                "median_type_iou_percent",
                "mean_shared_token_count",
            )
        )
    else:
        relation_summary = (
            unordered.groupby("relation", sort=False)
            .agg(
                pairs=("type_iou_percent", "size"),
                mean_type_iou_percent=("type_iou_percent", "mean"),
                median_type_iou_percent=("type_iou_percent", "median"),
                mean_shared_token_count=("shared_token_count", "mean"),
            )
            .reset_index()
        )

    group_rows: list[dict] = []
    for row in unordered_rows:
        left_group = LANGUAGE_BY_CODE[row["language_i"]].visual_group
        right_group = LANGUAGE_BY_CODE[row["language_j"]].visual_group
        group_i, group_j = sorted((left_group, right_group))
        group_rows.append({**row, "group_i": group_i, "group_j": group_j})
    if group_rows:
        group_summary = (
            pd.DataFrame(group_rows)
            .groupby(["group_i", "group_j"], sort=False)
            .agg(
                pairs=("type_iou_percent", "size"),
                mean_type_iou_percent=("type_iou_percent", "mean"),
                median_type_iou_percent=("type_iou_percent", "median"),
            )
            .reset_index()
        )
    else:
        group_summary = pd.DataFrame(
            columns=(
                "group_i",
                "group_j",
                "pairs",
                "mean_type_iou_percent",
                "median_type_iou_percent",
            )
        )

    top_pairs.to_csv(output_dir / "top_type_iou_pairs.csv", index=False, float_format="%.6f")
    asymmetry.head(30).to_csv(
        output_dir / "top_frequency_asymmetries.csv", index=False, float_format="%.6f"
    )
    relation_summary.to_csv(
        output_dir / "script_relation_summary.csv", index=False, float_format="%.6f"
    )
    group_summary.to_csv(
        output_dir / "group_pair_summary.csv", index=False, float_format="%.6f"
    )

    relation_lookup = relation_summary.set_index("relation").to_dict("index")
    stats = metrics.language_stats
    summary = {
        "condition": metrics.condition,
        "languages": len(codes),
        "examples_per_language_min": int(stats["examples"].min()),
        "examples_per_language_max": int(stats["examples"].max()),
        "analysis_tokens_min": int(stats["analysis_token_count"].min()),
        "analysis_tokens_max": int(stats["analysis_token_count"].max()),
        "unique_tokens_min": int(stats["unique_analysis_token_count"].min()),
        "unique_tokens_max": int(stats["unique_analysis_token_count"].max()),
        "unknown_tokens_total": int(stats["unknown_token_count"].sum()),
        "relation_summaries": relation_lookup,
        "strongest_type_iou_pair": (
            top_pairs.iloc[0].to_dict() if not top_pairs.empty else None
        ),
        "largest_frequency_asymmetry": (
            asymmetry.iloc[0].to_dict() if not asymmetry.empty else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(summary, top_pairs, output_dir / "REPORT.md")
    return summary


def _write_report(summary: dict, top_pairs: pd.DataFrame, path: Path) -> None:
    relations = summary["relation_summaries"]
    strongest = summary["strongest_type_iou_pair"]

    table_lines = [
        "| Pair | Type IoU | Shared types |",
        "|---|---:|---:|",
    ]
    for row in top_pairs.head(10).itertuples(index=False):
        table_lines.append(
            f"| {row.language_i} – {row.language_j} | {row.type_iou_percent:.2f}% | {row.shared_token_count:,} |"
        )

    if strongest is None:
        pair_analysis = """## Pairwise structure

This condition contains fewer than two languages. Full tokenization and
per-language vocabulary statistics are emitted, but no off-diagonal
cross-language pair exists.
"""
    else:
        latin = relations.get("within Latin-script")
        cross = relations.get("Latin ↔ non-Latin")
        non_latin = relations.get("within non-Latin scripts")
        relation_lines = []
        for label, value in (
            ("within Latin-script languages", latin),
            ("for Latin/non-Latin pairs", cross),
            ("within the non-Latin set", non_latin),
        ):
            if value is not None:
                relation_lines.append(
                    f"- {label}: {value['mean_type_iou_percent']:.2f}%"
                )
        pair_analysis = f"""## Script-level structure

Mean token-type IoU across available relation groups:

{chr(10).join(relation_lines)}

The strongest observed pair is `{strongest['language_i']}`–`{strongest['language_j']}`
at {strongest['type_iou_percent']:.2f}% type IoU. This is descriptive evidence
about tokenizer sharing in this condition's observed text; it is not evidence
that overlap caused the Stage-2 score changes.

## Highest token-type overlaps

{chr(10).join(table_lines)}
"""

    text = f"""# {summary['condition']} descriptive report

## Run coverage

- Languages: {summary['languages']}
- Examples per language: {summary['examples_per_language_min']:,}–{summary['examples_per_language_max']:,}
- Analysis tokens per language: {summary['analysis_tokens_min']:,}–{summary['analysis_tokens_max']:,}
- Unique observed analysis tokens: {summary['unique_tokens_min']:,}–{summary['unique_tokens_max']:,}
- Unknown-token occurrences: {summary['unknown_tokens_total']:,}

{pair_analysis}

## Interpretation boundary

These matrices should be frozen and compared across independent data
conditions before Stage-1 → Stage-2 deltas are joined. Script, family,
tokenizer coverage, training exposure, and task/domain distribution remain
confounded at this stage.
"""
    path.write_text(text, encoding="utf-8")
