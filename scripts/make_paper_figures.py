#!/usr/bin/env python3
"""Generate the FASD-Net comparison figure used by the final manuscript."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "figures"
ANN_BASELINES_CSV = (
    REPO_ROOT / "results" / "official_ann_baselines_current_env" / "summary.csv"
)
FASDNET_SUMMARY_JSON = (
    REPO_ROOT / "results" / "fasdnet" / "controls" / "fasdnet_summary.json"
)
FASDNET_CONTROL_CSV = (
    REPO_ROOT / "results" / "fasdnet" / "controls" / "alarm_control_table.csv"
)


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "fasdnet_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_fasdnet_results(
        read_csv_rows(ANN_BASELINES_CSV),
        read_json(FASDNET_SUMMARY_JSON),
        read_csv_rows(FASDNET_CONTROL_CSV),
    )
    print(OUTPUT_DIR)
    return 0


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: Any, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def draw_fasdnet_results(
    ann_rows: list[dict[str, str]],
    fasdnet_summary: dict[str, Any],
    fasdnet_controls: list[dict[str, str]],
) -> None:
    """Draw diagnosis trade-off, alarm error, and parameter count."""
    import matplotlib.pyplot as plt

    del fasdnet_controls  # retained in the API for source-table provenance
    ann_by_variant = {row["variant"]: row for row in ann_rows}
    display_order = ["resnet", "inception_time", "mlstm_fcn"]
    palette = {
        "resnet": "#D08C60",
        "inception_time": "#6B9AC4",
        "mlstm_fcn": "#7BAE7F",
    }
    fasdnet = {
        "cls_ba": float(fasdnet_summary["cls_ba_mean"]),
        "cls_std": float(fasdnet_summary.get("cls_ba_std", 0.0)),
        "loc_ba": float(fasdnet_summary["loc_ba_mean"]),
        "loc_std": float(fasdnet_summary.get("loc_ba_std", 0.0)),
        "alarm_mae": float(fasdnet_summary["alarm_mae_mean"]),
        "alarm_std": float(fasdnet_summary.get("alarm_mae_std", 0.0)),
        "params": float(fasdnet_summary["params_mean"]),
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.6, 3.9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.18, 1.0, 1.0]},
    )

    tradeoff_axis = axes[0]
    for variant in display_order:
        row = ann_by_variant[variant]
        x = float(row["localization_balanced_acc"])
        y = float(row["classification_balanced_acc"])
        tradeoff_axis.scatter(
            x,
            y,
            s=70,
            color=palette[variant],
            edgecolor="#30343B",
            linewidth=0.55,
            zorder=2,
        )
        tradeoff_axis.annotate(
            row["display_name"].replace(" port", ""),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
        )
    tradeoff_axis.scatter(
        fasdnet["loc_ba"],
        fasdnet["cls_ba"],
        marker="*",
        s=180,
        color="#111111",
        edgecolor="#111111",
        zorder=4,
    )
    if fasdnet["cls_std"] > 0 or fasdnet["loc_std"] > 0:
        tradeoff_axis.errorbar(
            fasdnet["loc_ba"],
            fasdnet["cls_ba"],
            xerr=fasdnet["loc_std"],
            yerr=fasdnet["cls_std"],
            fmt="none",
            ecolor="#111111",
            elinewidth=0.9,
            capsize=2.5,
            zorder=4,
        )
    tradeoff_axis.annotate(
        "FASD-Net",
        (fasdnet["loc_ba"], fasdnet["cls_ba"]),
        xytext=(-8, 12),
        textcoords="offset points",
        fontsize=8.5,
        weight="bold",
        ha="right",
    )
    tradeoff_axis.set_xlabel("localization balanced accuracy")
    tradeoff_axis.set_ylabel("classification balanced accuracy")
    tradeoff_axis.set_title("(a) Classification-localization", weight="bold")
    tradeoff_axis.set_xlim(0.50, 0.80)
    tradeoff_axis.set_ylim(0.76, 0.86)
    tradeoff_axis.grid(alpha=0.24)

    alarm_rows = [
        ("FASD-Net", fasdnet["alarm_mae"], fasdnet["alarm_std"]),
        (
            "MLSTM-FCN",
            float(ann_by_variant["mlstm_fcn"]["detection_macro_mae"]),
            None,
        ),
        (
            "InceptionTime",
            float(ann_by_variant["inception_time"]["detection_macro_mae"]),
            None,
        ),
        ("ResNet", float(ann_by_variant["resnet"]["detection_macro_mae"]), None),
    ]
    alarm_rows.sort(key=lambda item: item[1])
    labels = [item[0] for item in alarm_rows]
    values = [item[1] for item in alarm_rows]
    colors = ["#111111" if label == "FASD-Net" else "#C7CCD4" for label in labels]
    positions = np.arange(len(alarm_rows))
    bars = axes[1].barh(positions, values, color=colors, height=0.62)
    for index, (_, value, error) in enumerate(alarm_rows):
        if error:
            axes[1].errorbar(value, index, xerr=error, fmt="none", capsize=2)
    axes[1].set_yticks(positions, labels, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("alarm MAE (timesteps)")
    axes[1].set_title("(b) Alarm timing error", weight="bold")
    axes[1].grid(axis="x", alpha=0.24)
    axes[1].set_xlim(0, max(values) * 1.22)
    for bar, value, label in zip(bars, values, labels, strict=True):
        is_ours = label == "FASD-Net"
        axes[1].text(
            value * 0.57 if is_ours else value + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="center" if is_ours else "left",
            color="white" if is_ours else "#111111",
            weight="bold" if is_ours else "normal",
        )

    param_rows = [
        ("FASD-Net", fasdnet["params"] / 1000.0),
        ("ResNet", float(ann_by_variant["resnet"]["parameter_count_total"]) / 1000),
        (
            "MLSTM-FCN",
            float(ann_by_variant["mlstm_fcn"]["parameter_count_total"]) / 1000,
        ),
        (
            "InceptionTime",
            float(ann_by_variant["inception_time"]["parameter_count_total"]) / 1000,
        ),
    ]
    param_rows.sort(key=lambda item: item[1])
    labels = [item[0] for item in param_rows]
    values = [item[1] for item in param_rows]
    colors = ["#111111" if label == "FASD-Net" else "#C7CCD4" for label in labels]
    positions = np.arange(len(param_rows))
    bars = axes[2].barh(positions, values, color=colors, height=0.62)
    axes[2].set_yticks(positions, labels, fontsize=7.5)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("parameters (K)")
    axes[2].set_title("(c) Model size", weight="bold")
    axes[2].grid(axis="x", alpha=0.24)
    axes[2].set_xlim(0, max(values) * 1.18)
    for bar, value, label in zip(bars, values, labels, strict=True):
        is_ours = label == "FASD-Net"
        axes[2].text(
            value * 0.52 if is_ours else value + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}K" if value >= 100 else f"{value:.1f}K",
            va="center",
            ha="center" if is_ours else "left",
            color="white" if is_ours else "#111111",
            weight="bold" if is_ours else "normal",
        )

    save_figure(fig, "fig2_fasdnet_results")


if __name__ == "__main__":
    raise SystemExit(main())
