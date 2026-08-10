#!/usr/bin/env python3
"""FASD-Net alarm control rerun — produces all rows of alarm_control_table.csv
from a single reproducible script.

Variants (all built from FASDNET_CONFIG, differing only in snn_branch):
  - DPMixer-only      : snn_branch="none"     (no alarm head)
  - DPMixer + dense   : snn_branch="ann_alarm" (dense causal alarm head)
  - FASD-Net          : snn_branch="plif"      (spiking alarm head, main model)

For the FASD-Net trained model, alarm MAE is also evaluated under two
temporal-order controls (snn_control="shuffle" / "reverse") on the same
test set, using the trained weights (no retraining).

Mean-prior baselines (global-mean, per-class-mean) are recomputed
deterministically from the official PSML split.

Outputs (to results/fasdnet/controls/):
  - fasdnet_controls_raw.csv        : per-seed rows for every variant + control
  - fasdnet_controls_summary.json   : 5-seed aggregates + mean priors + metadata
  - alarm_control_table.csv         : the manuscript control table (replaces hand note)

Usage:
    python scripts/run_fasdnet_controls.py --seeds 0 1 2 3 4 --alarm-weight 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from group_exp_utils import load_psml, set_seed
from run_fasdnet import train_eval_task_specific

from fasdnet.models.fasdnet import FASDNET_CONFIG, FASDNet

SEEDS_DEFAULT = [0, 1, 2, 3, 4]
ALARM_WEIGHT_DEFAULT = 0.1
OUT_DIR = Path("results/fasdnet/controls")
N_CLASSES = 5


# ──────────────────────────────────────────────────────────────────────────
# Alarm control evaluation (extends evaluate_controls to report alarm MAE)
# ──────────────────────────────────────────────────────────────────────────


def evaluate_alarm_controls(model, tex, ty_alarm, seed_for_shuffle):
    """Return alarm MAE under snn_control in {none, shuffle, reverse}.

    The model is already trained; this is inference-only. ``seed_for_shuffle``
    makes the shuffle permutation reproducible per seed.
    """
    model.eval()
    out = {}
    with torch.inference_mode():
        for ctrl in ["none", "shuffle", "reverse"]:
            if ctrl == "shuffle":
                torch.manual_seed(seed_for_shuffle * 100 + 7)
            o = model(tex, snn_control=ctrl)
            if o["alarm_pred"] is not None:
                mae = float((o["alarm_pred"] - ty_alarm).abs().mean().item())
            else:
                mae = None
            out[ctrl] = mae
    return out


# ──────────────────────────────────────────────────────────────────────────
# Mean-prior baselines (deterministic, no training)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AlarmMechanismSource:
    """Alarm-head and temporal-order values exported for the Fig. 3 source."""

    snn_alarm_mae: float
    snn_params: int
    dense_alarm_mae: float
    dense_params: int
    backbone_params: int
    global_prior_mae: float
    per_class_prior_mae: float
    shuffle_mae: float
    reverse_mae: float


def write_alarm_mechanism_source(source: AlarmMechanismSource, out_dir: Path) -> Path:
    """Write ``alarm_mechanism.csv``, the source table for Fig. 3.

    The table is a reshaped view of the alarm control rerun: one row per
    condition shown in the manuscript figure. ``DPMixer-only`` has no alarm
    output and is encoded with an empty ``alarm_mae`` (categorical N/A), never
    as a numeric zero.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "alarm_mechanism.csv"
    rows = [
        ("DPMixer-only", "head_control", "", source.backbone_params, "no_alarm_output"),
        (
            "SNN alarm head",
            "alarm_head",
            f"{source.snn_alarm_mae:.2f}",
            source.snn_params,
            "yes",
        ),
        (
            "Dense causal alarm head",
            "alarm_head",
            f"{source.dense_alarm_mae:.2f}",
            source.dense_params,
            "yes",
        ),
        (
            "Global-mean prior",
            "prior",
            f"{source.global_prior_mae:.2f}",
            0,
            "no",
        ),
        (
            "Per-class mean prior",
            "prior",
            f"{source.per_class_prior_mae:.2f}",
            0,
            "no",
        ),
        (
            "Ordered tokens",
            "temporal_order",
            f"{source.snn_alarm_mae:.2f}",
            source.snn_params,
            "yes",
        ),
        (
            "Shuffled tokens",
            "temporal_order",
            f"{source.shuffle_mae:.0f}",
            source.snn_params,
            "yes",
        ),
        (
            "Reversed tokens",
            "temporal_order",
            f"{source.reverse_mae:.0f}",
            source.snn_params,
            "yes",
        ),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["condition", "analysis_group", "alarm_mae", "params", "availability"]
        )
        writer.writerows(rows)
    return path


def compute_mean_priors(y, train_idx, test_idx):
    """Global-mean and per-class-mean alarm-time priors evaluated on test."""
    ay = y[:, 2].astype(float)
    cls = y[:, 0].astype(int)
    tr, te = train_idx, test_idx
    global_mean = ay[tr].mean()
    global_mae = float(np.abs(ay[te] - global_mean).mean())
    pc_mean = np.array([ay[tr][cls[tr] == k].mean() for k in range(N_CLASSES)])
    pc_pred = pc_mean[cls[te]]
    pc_mae = float(np.abs(ay[te] - pc_pred).mean())
    return {
        "global_mean_prior_mae": global_mae,
        "per_class_mean_prior_mae": pc_mae,
        "global_mean_value": float(global_mean),
        "per_class_mean_values": pc_mean.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main rerun
# ──────────────────────────────────────────────────────────────────────────


def run(seeds, alarm_weight, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_psml()
    train_x = data["x"][data["train_idx"]]
    test_x = data["x"][data["test_idx"]]
    train_y = data["y"][data["train_idx"]]
    test_y = data["y"][data["test_idx"]]
    mu = train_x.reshape(-1, 91).mean(axis=0)
    sd = train_x.reshape(-1, 91).std(axis=0) + 1e-8
    train_x_s = (train_x - mu) / sd
    test_x_s = (test_x - mu) / sd
    tex = torch.from_numpy(test_x_s.astype(np.float32)).to(dev)
    ty_alarm = torch.from_numpy(test_y[:, 2].astype(np.float32)).to(dev)

    variants = [
        ("DPMixer-only", "none"),
        ("DPMixer_dense_alarm", "ann_alarm"),
        ("FASD-Net", "plif"),
    ]

    raw_rows = []
    for seed in seeds:
        for vname, branch in variants:
            set_seed(seed)
            cfg = dict(FASDNET_CONFIG)
            cfg["snn_branch"] = branch
            model = FASDNet(**cfg)
            r = train_eval_task_specific(
                model, train_x_s, train_y, test_x_s, test_y, alarm_weight=alarm_weight
            )
            r["variant"] = vname
            r["snn_branch"] = branch
            r["seed"] = seed
            r["alarm_weight"] = alarm_weight

            # FASD-Net: evaluate shuffle/reverse alarm MAE on trained weights
            if branch == "plif":
                ctrl = evaluate_alarm_controls(model, tex, ty_alarm, seed)
                r["alarm_mae_shuffle"] = ctrl["shuffle"]
                r["alarm_mae_reverse"] = ctrl["reverse"]
                r["alarm_mae_none"] = ctrl["none"]
            else:
                r["alarm_mae_shuffle"] = None
                r["alarm_mae_reverse"] = None
                r["alarm_mae_none"] = None

            raw_rows.append(r)
            spk = f"{r['spike_rate']:.4f}" if r["spike_rate"] is not None else "—"
            ctrl_str = ""
            if r.get("alarm_mae_shuffle") is not None:
                ctrl_str = (
                    f"  shuffle={r['alarm_mae_shuffle']:.1f}"
                    f"  reverse={r['alarm_mae_reverse']:.1f}"
                )
            print(
                f"  {vname:24s} seed={seed}  cls={r['cls_ba']:.4f}  "
                f"loc={r['loc_ba']:.4f}  almMAE={r['alarm_mae']}"
                f"  spkR={spk}  ep={r['epoch']}  p={r['params']:,}"
                f"  t={r['train_time_sec']:.1f}s{ctrl_str}",
                flush=True,
            )

    # ── mean priors ──
    priors = compute_mean_priors(data["y"], data["train_idx"], data["test_idx"])
    print(f"\n  global-mean prior MAE = {priors['global_mean_prior_mae']:.4f}")
    print(f"  per-class-mean prior MAE = {priors['per_class_mean_prior_mae']:.4f}")

    # ── write raw CSV ──
    raw_fields = [
        "variant",
        "snn_branch",
        "seed",
        "alarm_weight",
        "params",
        "epoch",
        "cls_ba",
        "loc_ba",
        "alarm_mae",
        "spike_rate",
        "spike_count",
        "alarm_mae_none",
        "alarm_mae_shuffle",
        "alarm_mae_reverse",
        "train_time_sec",
    ]
    raw_path = out_dir / "fasdnet_controls_raw.csv"
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        w.writeheader()
        for r in raw_rows:
            w.writerow({k: r.get(k, "") for k in raw_fields})
    print(f"\nSaved: {raw_path}")

    # ── compute 5-seed aggregates ──
    def agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    summary = {
        "alarm_weight": alarm_weight,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "config": "FASDNET_CONFIG (src/fasdnet/models/fasdnet.py)",
    }
    for vname, branch in variants:
        vrows = [r for r in raw_rows if r["variant"] == vname]
        cls_m, cls_s = agg(vrows, "cls_ba")
        loc_m, loc_s = agg(vrows, "loc_ba")
        alm_m, alm_s = agg(vrows, "alarm_mae")
        spk_m, spk_s = agg(vrows, "spike_rate")
        params = vrows[0]["params"] if vrows else None
        entry = {
            "cls_ba_mean": cls_m,
            "cls_ba_std": cls_s,
            "loc_ba_mean": loc_m,
            "loc_ba_std": loc_s,
            "alarm_mae_mean": alm_m,
            "alarm_mae_std": alm_s,
            "spike_rate_mean": spk_m,
            "spike_rate_std": spk_s,
            "params": params,
        }
        if branch == "plif":
            sh_m, sh_s = agg(vrows, "alarm_mae_shuffle")
            rv_m, rv_s = agg(vrows, "alarm_mae_reverse")
            entry["alarm_mae_shuffle_mean"] = sh_m
            entry["alarm_mae_shuffle_std"] = sh_s
            entry["alarm_mae_reverse_mean"] = rv_m
            entry["alarm_mae_reverse_std"] = rv_s
        summary[vname] = entry
    summary["mean_priors"] = priors

    sum_path = out_dir / "fasdnet_controls_summary.json"
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {sum_path}")

    # ── write alarm_control_table.csv (manuscript table) ──
    def fmt_mae(m, s):
        if m is None:
            return ""
        if s is None:
            return f"{m:.2f}"
        return f"{m:.2f} ± {s:.2f}"

    fasd = summary["FASD-Net"]
    dense = summary["DPMixer_dense_alarm"]
    dponly = summary["DPMixer-only"]
    ctrl_rows = [
        {
            "model": "DPMixer-only",
            "role": "slow diagnostic baseline",
            "cls_loc_available": "yes",
            "alarm_mae": "",
            "alarm_mae_std": "",
            "params": dponly["params"],
            "source": "fasdnet_controls_summary.json (snn_branch=none)",
            "interpretation": (
                "classification/localization backbone without an explicit alarm head"
            ),
        },
        {
            "model": "DPMixer + dense ANN alarm head",
            "role": "dense alarm control",
            "cls_loc_available": "yes",
            "alarm_mae": f"{dense['alarm_mae_mean']:.2f}",
            "alarm_mae_std": f"{dense['alarm_mae_std']:.2f}",
            "params": dense["params"],
            "source": "fasdnet_controls_summary.json (snn_branch=ann_alarm)",
            "interpretation": (
                "dense causal alarm head improves over mean priors but remains weaker "
                "than the spiking alarm pathway"
            ),
        },
        {
            "model": "FASD-Net (DPMixer + SNN alarm head)",
            "role": "main model",
            "cls_loc_available": "yes",
            "alarm_mae": f"{fasd['alarm_mae_mean']:.2f}",
            "alarm_mae_std": f"{fasd['alarm_mae_std']:.2f}",
            "params": fasd["params"],
            "source": "fasdnet_controls_summary.json (snn_branch=plif)",
            "interpretation": "five-seed rerun of FASDNET_CONFIG with alarm_weight "
            f"{alarm_weight}",
        },
        {
            "model": "SNN alarm w/o temporal order (shuffle)",
            "role": "temporal-order control",
            "cls_loc_available": "yes",
            "alarm_mae": f"{fasd['alarm_mae_shuffle_mean']:.1f}",
            "alarm_mae_std": f"{fasd['alarm_mae_shuffle_std']:.1f}",
            "params": fasd["params"],
            "source": "fasdnet_controls_summary.json (plif, snn_control=shuffle)",
            "interpretation": "shuffling token order collapses alarm timing",
        },
        {
            "model": "SNN alarm w/o temporal order (reverse)",
            "role": "temporal-order control",
            "cls_loc_available": "yes",
            "alarm_mae": f"{fasd['alarm_mae_reverse_mean']:.1f}",
            "alarm_mae_std": f"{fasd['alarm_mae_reverse_std']:.1f}",
            "params": fasd["params"],
            "source": "fasdnet_controls_summary.json (plif, snn_control=reverse)",
            "interpretation": "reversing token order collapses alarm timing",
        },
        {
            "model": "global-mean prior",
            "role": "alarm prior",
            "cls_loc_available": "no",
            "alarm_mae": f"{priors['global_mean_prior_mae']:.2f}",
            "alarm_mae_std": "",
            "params": 0,
            "source": "data/PSML/processed_dataset/classification.pkl (deterministic)",
            "interpretation": (
                "global training alarm-time mean evaluated on the official test split"
            ),
        },
        {
            "model": "per-class mean prior",
            "role": "alarm prior",
            "cls_loc_available": "no",
            "alarm_mae": f"{priors['per_class_mean_prior_mae']:.2f}",
            "alarm_mae_std": "",
            "params": 0,
            "source": "data/PSML/processed_dataset/classification.pkl (deterministic)",
            "interpretation": (
                "class-conditional training alarm-time means evaluated on the official "
                "test split"
            ),
        },
    ]
    ctrl_fields = [
        "model",
        "role",
        "cls_loc_available",
        "alarm_mae",
        "alarm_mae_std",
        "params",
        "source",
        "interpretation",
    ]
    ctrl_path = out_dir / "alarm_control_table.csv"
    with ctrl_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ctrl_fields)
        w.writeheader()
        for row in ctrl_rows:
            w.writerow(row)
    print(f"Saved: {ctrl_path}")

    # ── Fig. 3 source table (alarm mechanism) ──
    fig3_source = AlarmMechanismSource(
        snn_alarm_mae=fasd["alarm_mae_mean"],
        snn_params=int(fasd["params"]),
        dense_alarm_mae=dense["alarm_mae_mean"],
        dense_params=int(dense["params"]),
        backbone_params=int(dponly["params"]),
        global_prior_mae=priors["global_mean_prior_mae"],
        per_class_prior_mae=priors["per_class_mean_prior_mae"],
        shuffle_mae=fasd["alarm_mae_shuffle_mean"],
        reverse_mae=fasd["alarm_mae_reverse_mean"],
    )
    fig3_path = write_alarm_mechanism_source(fig3_source, out_dir)
    print(f"Saved: {fig3_path}")

    # ── print final table ──
    print(f"\n{'=' * 100}")
    print(f"  FASD-Net alarm control table (5-seed, alarm_weight={alarm_weight})")
    print(f"{'=' * 100}")
    print(f"{'Model':<42s} {'Cls/Loc':>7s} {'Alarm MAE':>16s} {'Params':>10s}")
    print("-" * 100)
    for row in ctrl_rows:
        mae = row["alarm_mae"]
        if row["alarm_mae_std"]:
            mae = f"{mae} ± {row['alarm_mae_std']}"
        params = f"{row['params'] / 1000:.1f}K" if row["params"] else "0K"
        print(
            f"{row['model']:<42s} {row['cls_loc_available']:>7s} "
            f"{mae:>16s} {params:>10s}"
        )
    print("=" * 100)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--alarm-weight", type=float, default=ALARM_WEIGHT_DEFAULT)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = p.parse_args()
    run(args.seeds, args.alarm_weight, Path(args.out_dir))
