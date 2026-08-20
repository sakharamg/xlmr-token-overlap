"""Grouped matrix visualizations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Work in read-only-home containers without Matplotlib trying to create
# ~/.config. The directory is a disposable render cache, not a study input.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xlmr-token-overlap-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .constants import Language


GROUP_COLORS = {
    "Germanic Latin": "#386cb0",
    "Romance Latin": "#ef7c3b",
    "Austronesian Latin": "#42a96f",
    "Other Latin": "#b26bb2",
    "Other scripts": "#6c757d",
}


def _boundaries(languages: tuple[Language, ...]) -> list[int]:
    return [
        index
        for index in range(1, len(languages))
        if languages[index - 1].visual_group != languages[index].visual_group
    ]


def plot_matrix(
    values: np.ndarray,
    languages: tuple[Language, ...],
    output: Path,
    title: str,
    colorbar_label: str,
    *,
    upper_triangle: bool = False,
    integer_annotations: bool = False,
) -> None:
    matrix = np.asarray(values, dtype=float).copy()
    if upper_triangle:
        matrix[np.tril_indices_from(matrix, k=-1)] = np.nan

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(15, 12.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, aspect="equal", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label(colorbar_label)

    labels = [language.label for language in languages]
    positions = np.arange(len(labels))
    ax.set_xticks(positions, labels=labels, rotation=0, fontsize=9)
    ax.set_yticks(positions, labels=labels, fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for tick, language in zip(ax.get_xticklabels(), languages, strict=True):
        tick.set_color(GROUP_COLORS[language.visual_group])
        tick.set_fontweight("bold")
    for tick, language in zip(ax.get_yticklabels(), languages, strict=True):
        tick.set_color(GROUP_COLORS[language.visual_group])
        tick.set_fontweight("bold")

    for boundary in _boundaries(languages):
        ax.axhline(boundary - 0.5, color="white", linewidth=2.2)
        ax.axvline(boundary - 0.5, color="white", linewidth=2.2)
        ax.axhline(boundary - 0.5, color="#222222", linewidth=0.65)
        ax.axvline(boundary - 0.5, color="#222222", linewidth=0.65)

    # Compact cell labels make the PNG independently inspectable; CSV/Parquet
    # remain the authoritative full-precision values.
    threshold = np.nanmedian(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                continue
            label = f"{int(round(value))}" if integer_annotations else f"{value:.1f}"
            normalized = image.norm(value)
            color = "white" if normalized > 0.58 else "#111111"
            if value < threshold and normalized > 0.45:
                color = "white"
            ax.text(column, row, label, ha="center", va="center", fontsize=5.1, color=color)

    handles = [Patch(color=color, label=group) for group, color in GROUP_COLORS.items()]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    ax.set_title(title, pad=24, fontsize=15, fontweight="bold")
    ax.set_xlabel("Target language" if not upper_triangle else "Language j", labelpad=12)
    ax.set_ylabel("Source language" if not upper_triangle else "Language i")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_heatmaps(metrics, output_dir: Path) -> None:
    heatmaps = output_dir / "heatmaps"
    plot_matrix(
        metrics.type_iou.to_numpy(),
        metrics.languages,
        heatmaps / "type_iou_upper.png",
        f"{metrics.condition}: XLM-R token-type IoU (upper triangle)",
        "Token-type IoU (%)",
        upper_triangle=True,
    )
    plot_matrix(
        metrics.frequency_overlap.to_numpy(),
        metrics.languages,
        heatmaps / "frequency_overlap_directional.png",
        f"{metrics.condition}: directional frequency-weighted overlap",
        "Source occurrences covered (%)",
    )
    plot_matrix(
        metrics.shared_count.to_numpy(),
        metrics.languages,
        heatmaps / "shared_token_count.png",
        f"{metrics.condition}: shared observed XLM-R token types",
        "Shared token types",
        integer_annotations=True,
    )
