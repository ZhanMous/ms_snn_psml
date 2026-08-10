#!/usr/bin/env python3
"""Create Fig. 3: FASD-Net alarm-mechanism validation.

Chart contract
--------------
Analytical question
    Does the spiking alarm pathway improve timing over dense/prior controls,
    and does its performance depend on temporal order?
Takeaway
    The SNN alarm head has the lowest alarm MAE among available heads and
    priors; shuffling or reversing tokens increases its error by 30x and 59x.
Chart family and variant
    Comparison/ranking: ranked horizontal bars for head/prior comparison and
    a log-scale dot-reference comparison for temporal-order controls.
Data sufficiency
    Four comparable alarm-producing methods and three order conditions from
    the existing PSML alarm ablation. DPMixer-only is retained in the source as
    a categorical N/A because it has no alarm output.
Surface and footprint
    Static full-width IEEE conference figure; export PNG, PDF, and SVG.
Palette and non-color distinction
    Blue focal marks, orange stress-test marks, and neutral priors. Direct
    labels, marker shapes, and open/filled markers remain readable in grayscale.
Final QA surface
    The compiled IEEE two-column manuscript at final printed scale.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT / "results" / "fasdnet" / "controls" / "alarm_mechanism.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"

INK = "#1F2430"
MUTED = "#626A76"
GRID = "#E1E4E8"
AXIS = "#B8BEC7"
BLUE = "#5477C4"
BLUE_DARK = "#2E4780"
BLUE_LIGHT = "#CEDFFE"
ORANGE = "#F0986E"
ORANGE_DARK = "#804126"
NEUTRAL_LIGHT = "#E2E5EA"
NEUTRAL_MID = "#7A828F"
NEUTRAL_DARK = "#464C55"


@dataclass(frozen=True)
class AlarmMetric:
    condition: str
    analysis_group: str
    alarm_mae: float | None
    params: int
    availability: str


def load_alarm_metrics(path: Path = DEFAULT_SOURCE) -> list[AlarmMetric]:
    """Load and validate the alarm values visualized in Fig. 3."""
    rows: list[AlarmMetric] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_mae = row["alarm_mae"].strip()
            rows.append(
                AlarmMetric(
                    condition=row["condition"].strip(),
                    analysis_group=row["analysis_group"].strip(),
                    alarm_mae=float(raw_mae) if raw_mae else None,
                    params=int(row["params"]),
                    availability=row["availability"].strip(),
                )
            )

    by_name = {row.condition: row for row in rows}
    expected = {
        "DPMixer-only",
        "SNN alarm head",
        "Dense causal alarm head",
        "Global-mean prior",
        "Per-class mean prior",
        "Ordered tokens",
        "Shuffled tokens",
        "Reversed tokens",
    }
    if set(by_name) != expected:
        raise ValueError("alarm-mechanism source has missing or unexpected conditions")
    if by_name["DPMixer-only"].alarm_mae is not None:
        raise ValueError("DPMixer-only must be represented as no alarm output")
    for name, metric in by_name.items():
        if name != "DPMixer-only" and (
            metric.alarm_mae is None or metric.alarm_mae <= 0
        ):
            raise ValueError(f"{name} requires a positive alarm MAE")
    return rows


def _use_ieee_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def draw_alarm_mechanism_figure(
    metrics: list[AlarmMetric],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    """Render the full-width alarm-head and order-sensitivity comparison."""
    _use_ieee_style()
    by_name = {row.condition: row for row in metrics}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.25),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )
    fig.subplots_adjust(left=0.19, right=0.965, bottom=0.17, top=0.88, wspace=0.42)

    # (a) Four available alarm-producing methods. The categorical N/A row for
    # DPMixer-only stays in the source but is not encoded as a numeric zero.
    ax = axes[0]
    head_order = [
        "SNN alarm head",
        "Dense causal alarm head",
        "Global-mean prior",
        "Per-class mean prior",
    ]
    head_df = pd.DataFrame(
        {
            "condition": head_order,
            "alarm_mae": [by_name[name].alarm_mae for name in head_order],
        }
    )
    palette = {
        "SNN alarm head": BLUE,
        "Dense causal alarm head": ORANGE,
        "Global-mean prior": NEUTRAL_LIGHT,
        "Per-class mean prior": NEUTRAL_LIGHT,
    }
    edge_colors = {
        "SNN alarm head": BLUE_DARK,
        "Dense causal alarm head": ORANGE_DARK,
        "Global-mean prior": NEUTRAL_DARK,
        "Per-class mean prior": NEUTRAL_DARK,
    }
    sns.barplot(
        data=head_df,
        x="alarm_mae",
        y="condition",
        order=head_order,
        hue="condition",
        hue_order=head_order,
        palette=palette,
        legend=False,
        dodge=False,
        ax=ax,
        linewidth=0.8,
    )
    for patch, name in zip(ax.patches, head_order, strict=True):
        patch.set_edgecolor(edge_colors[name])
        value = float(by_name[name].alarm_mae)
        ax.text(
            value + 1.0,
            patch.get_y() + patch.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=edge_colors[name],
            fontweight="bold" if name == "SNN alarm head" else "normal",
        )
    ax.set_title(
        "(a) Alarm-head comparison", loc="left", fontsize=8.7, fontweight="bold", pad=7
    )
    ax.set_xlim(0, 51)
    ax.set_xlabel("Alarm MAE (timesteps)", fontsize=7.8)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="x", labelsize=6.8, colors=MUTED, length=3)
    ax.tick_params(axis="y", labelsize=7.0, colors=INK, length=0, pad=5)
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    # (b) Same trained SNN alarm head under temporal-order interventions.
    ax = axes[1]
    order_names = ["Ordered tokens", "Shuffled tokens", "Reversed tokens"]
    order_labels = ["Ordered", "Shuffled", "Reversed"]
    values = [float(by_name[name].alarm_mae) for name in order_names]
    baseline = values[0]
    y = list(range(len(order_names)))[::-1]
    ax.axvline(baseline, color=BLUE_LIGHT, linestyle=":", linewidth=1.0, zorder=1)
    for row_y, label, value in zip(y, order_labels, values, strict=True):
        is_ordered = label == "Ordered"
        ax.hlines(
            row_y,
            baseline,
            value,
            color=BLUE_LIGHT if is_ordered else NEUTRAL_MID,
            linewidth=1.0,
            zorder=2,
        )
        marker = "o" if is_ordered else ("s" if label == "Shuffled" else "^")
        face = BLUE if is_ordered else ("white" if label == "Shuffled" else ORANGE)
        edge = BLUE_DARK if is_ordered else ORANGE_DARK
        ax.scatter(
            [value],
            [row_y],
            marker=marker,
            s=38,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
            zorder=4,
        )
        value_label = f"{value:.2f}" if is_ordered else f"{value:.0f}"
        ratio_label = "" if is_ordered else f"  ({value / baseline:.0f}×)"
        ax.annotate(
            value_label + ratio_label,
            (value, row_y),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.0,
            color=edge,
            fontweight="bold" if is_ordered else "normal",
            clip_on=False,
        )
    ax.set_title(
        "(b) Temporal-order stress test",
        loc="left",
        fontsize=8.7,
        fontweight="bold",
        pad=7,
    )
    ax.set_xscale("log")
    ax.set_xlim(8, 1150)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y, labels=order_labels)
    ax.set_xlabel("Alarm MAE (log scale)", fontsize=7.8)
    ax.xaxis.set_major_locator(mticker.FixedLocator([10, 100, 1000]))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=6.8, colors=MUTED, length=3)
    ax.tick_params(axis="y", labelsize=7.0, colors=INK, length=0, pad=5)
    ax.grid(axis="x", which="major", zorder=0)
    ax.grid(axis="x", which="minor", visible=False)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig3_fasdnet_alarm_mechanism"
    paths = (
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
    )
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    saved = draw_alarm_mechanism_figure(load_alarm_metrics())
    print(f"Saved: {[str(path) for path in saved]}")
