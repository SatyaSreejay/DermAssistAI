"""Plots describing the dataset and the splits."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (Colab, servers, tests)
import matplotlib.pyplot as plt
import pandas as pd

from src.data.splitting import SPLIT_NAMES

# Fixed colours so a split always has the same colour in every figure.
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SINGLE_COLOR = "#2a78d6"
INK, INK_SECONDARY = "#0b0b0b", "#52514e"


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK_SECONDARY)
    ax.spines["bottom"].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_distribution(class_counts: dict[str, int], out_path: str | Path, title: str) -> Path:
    """Bar chart of images per class with the count printed above each bar."""
    out_path = Path(out_path)
    classes = list(class_counts)
    counts = [class_counts[c] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    bars = ax.bar(classes, counts, color=SINGLE_COLOR, width=0.6)
    ax.bar_label(bars, labels=[f"{n:,}" for n in counts], padding=2, color=INK, fontsize=8)
    ax.set_ylabel("Images", color=INK)
    ax.set_xlabel("Diagnosis (dx)", color=INK)
    ax.set_title(title, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_split_proportions(summary: pd.DataFrame, out_path: str | Path, title: str) -> Path:
    """Grouped bars: share of each class within each split.

    Similar bar heights across splits mean the stratification worked.
    """
    out_path = Path(out_path)
    data = summary.drop(index="TOTAL", errors="ignore")
    splits = [s for s in SPLIT_NAMES if f"{s}_pct" in data.columns]
    classes = list(data.index)
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for i, s in enumerate(splits):
        xs = [x + (i - (len(splits) - 1) / 2) * width for x in range(len(classes))]
        ax.bar(xs, data[f"{s}_pct"], width=width * 0.92, color=SPLIT_COLORS[s], label=s)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel("Share of images in split (%)", color=INK)
    ax.set_xlabel("Diagnosis (dx)", color=INK)
    ax.set_title(title, color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, ncol=len(splits), loc="lower right", bbox_to_anchor=(1, 1))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
