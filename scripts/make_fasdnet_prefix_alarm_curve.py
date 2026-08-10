#!/usr/bin/env python3
"""Create the FASD-Net prefix-length alarm sensitivity curve.

Chart contract
--------------
Analytical question
    How does FASD-Net alarm MAE change when only a causal prefix is available?
Takeaway
    Alarm MAE falls rapidly as prefix length increases and reaches its lowest
    audited value near 320 timesteps, supporting early causal alarm inference.
Chart family and variant
    Single-series line chart over evaluated prefix settings.
Data sufficiency
    One selected-seed sensitivity audit on the PSML test split. The curve is
    not a replacement for the main full-protocol comparison.
Surface and footprint
    Static single-column IEEE conference figure; export PNG, PDF, and SVG.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "results" / "fasdnet" / "controls" / "prefix_alarm_curve"
)

INK = "#1F2430"
MUTED = "#626A76"
GRID = "#E1E4E8"
AXIS = "#B8BEC7"
BLUE = "#5477C4"
BLUE_DARK = "#2E4780"
ACCENT = "#B45B4D"


@dataclass(frozen=True)
class PrefixAlarmPoint:
    prefix: int
    alarm_mae_all_test: float


PREFIX_ALARM_CURVE: tuple[PrefixAlarmPoint, ...] = (
    PrefixAlarmPoint(32, 88.01),
    PrefixAlarmPoint(64, 59.93),
    PrefixAlarmPoint(96, 38.66),
    PrefixAlarmPoint(128, 23.47),
    PrefixAlarmPoint(160, 13.29),
    PrefixAlarmPoint(192, 7.88),
    PrefixAlarmPoint(240, 6.50),
    PrefixAlarmPoint(320, 6.34),
    PrefixAlarmPoint(480, 7.01),
    PrefixAlarmPoint(960, 11.71),
)


def write_source_csv(
    points: tuple[PrefixAlarmPoint, ...],
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> Path:
    """Write the exact plotted values for manuscript traceability."""
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / "prefix_alarm_curve.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prefix", "alarm_mae_all_test"])
        for point in points:
            writer.writerow([point.prefix, f"{point.alarm_mae_all_test:.2f}"])
    return path


def use_ieee_style() -> None:
    plt.rcParams.update(
        {
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
        }
    )


def draw_prefix_alarm_curve(
    points: tuple[PrefixAlarmPoint, ...] = PREFIX_ALARM_CURVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    """Render the single-series prefix sensitivity curve."""
    use_ieee_style()
    prefixes = [point.prefix for point in points]
    maes = [point.alarm_mae_all_test for point in points]
    x_positions = list(range(len(points)))

    fig, ax = plt.subplots(figsize=(3.48, 2.32))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.30, top=0.91)

    ax.plot(
        x_positions,
        maes,
        color=BLUE,
        linewidth=1.55,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=BLUE_DARK,
        markeredgewidth=0.9,
        zorder=3,
    )
    min_idx = min(range(len(maes)), key=maes.__getitem__)
    ax.scatter(
        [x_positions[min_idx]],
        [maes[min_idx]],
        s=32,
        marker="o",
        facecolor=ACCENT,
        edgecolor=BLUE_DARK,
        linewidth=0.8,
        zorder=4,
    )

    ax.set_xlabel("Prefix length (timesteps)", fontsize=8.2)
    ax.set_ylabel("Alarm MAE (timesteps)", fontsize=8.2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(prefix) for prefix in prefixes], rotation=45, ha="right")
    ax.set_xlim(-0.35, len(points) - 0.65)
    ax.set_ylim(0, 95)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="both", labelsize=7.2, colors=MUTED, length=3)
    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)

    ax.text(
        x_positions[min_idx] + 0.18,
        maes[min_idx] + 7.5,
        "min 6.34",
        fontsize=7.2,
        color=ACCENT,
        ha="left",
        va="bottom",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig4_prefix_alarm_curve"
    paths = (
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".svg"),
    )
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    args = parser.parse_args()
    write_source_csv(PREFIX_ALARM_CURVE, args.source_dir)
    draw_prefix_alarm_curve(PREFIX_ALARM_CURVE, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
