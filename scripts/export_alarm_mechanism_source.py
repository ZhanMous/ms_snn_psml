#!/usr/bin/env python3
"""Export ``results/fasdnet/controls/alarm_mechanism.csv`` (Fig. 3 source).

The manuscript's Fig. 3 source table is derived from the committed
``alarm_control_table.csv``, which itself records the values of the selected
five-seed alarm-control rerun. This script reshapes that table into the
``condition / analysis_group / alarm_mae / params / availability`` format
consumed by ``make_fasdnet_alarm_mechanism_figure.py``, so the figure source
stays reproducible without a full retraining rerun.

Usage:
    python scripts/export_alarm_mechanism_source.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fasdnet_controls import (  # noqa: E402
    AlarmMechanismSource,
    write_alarm_mechanism_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_TABLE = (
    REPO_ROOT / "results" / "fasdnet" / "controls" / "alarm_control_table.csv"
)
OUT_DIR = REPO_ROOT / "results" / "fasdnet" / "controls"

TABLE_MODEL_TO_ALARM_SOURCE = {
    "DPMixer-only": "backbone",
    "DPMixer + dense ANN alarm head": "dense",
    "FASD-Net (DPMixer + SNN alarm head)": "snn",
    "SNN alarm w/o temporal order (shuffle)": "shuffle",
    "SNN alarm w/o temporal order (reverse)": "reverse",
    "global-mean prior": "global",
    "per-class mean prior": "per_class",
}


def load_committed_source(control_table: Path) -> AlarmMechanismSource:
    """Build the Fig. 3 source from the committed alarm-control table."""
    rows = {}
    with control_table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["model"].strip()] = row

    def mae(model: str) -> float:
        value = rows[model]["alarm_mae"].strip()
        if not value:
            msg = f"alarm_control_table.csv has no alarm MAE for {model!r}"
            raise ValueError(msg)
        return float(value)

    def params(model: str) -> int:
        return int(rows[model]["params"])

    return AlarmMechanismSource(
        snn_alarm_mae=mae("FASD-Net (DPMixer + SNN alarm head)"),
        snn_params=params("FASD-Net (DPMixer + SNN alarm head)"),
        dense_alarm_mae=mae("DPMixer + dense ANN alarm head"),
        dense_params=params("DPMixer + dense ANN alarm head"),
        backbone_params=params("DPMixer-only"),
        global_prior_mae=mae("global-mean prior"),
        per_class_prior_mae=mae("per-class mean prior"),
        shuffle_mae=mae("SNN alarm w/o temporal order (shuffle)"),
        reverse_mae=mae("SNN alarm w/o temporal order (reverse)"),
    )


def main() -> int:
    source = load_committed_source(CONTROL_TABLE)
    path = write_alarm_mechanism_source(source, OUT_DIR)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
