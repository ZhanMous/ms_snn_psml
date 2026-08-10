#!/usr/bin/env python3
"""FASD-Net training and evaluation runner.

This runner intentionally exposes only the two locked post-ablation models.
Mechanism controls live in ``run_fasdnet_controls.py``.

Locked canonical configs (selected by the completed architecture ablation):
  - FASDNET     : minimal effective structure (PatchEmbed ps16/ed48 + ICB d1)
                  + patch-level SNNAlarmHead. The main model.
  - UPPER_BOUND : large patch ps32/ed64/d2 + ASB+ICB stack + SNNAlarmHead.
                  High-accuracy, low parameter-efficiency reference.

Metrics per (variant, seed): cls BA, loc BA, prefix cls BA at
[48,96,128,192,240,480,960], alarm MAE (timesteps), spike rate, spike count,
params, training time.

Protocol: train on 439 train samples, evaluate on 110 test samples every 5
epochs, and keep the epoch with the best (cls_ba + loc_ba) sum. Prefix, alarm,
and spike metrics are reported at the selected epoch.

Usage:
    python scripts/run_fasdnet.py --variants FASDNET --seeds 0
    python scripts/run_fasdnet.py --variants FASDNET UPPER_BOUND --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from group_exp_utils import (  # noqa: E402
    bal_acc,
    load_psml,
    set_seed,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fasdnet.models.fasdnet import (  # noqa: E402
    FASDNET_CONFIG,
    UPPER_BOUND_CONFIG,
    FASDNet,
)

OUT_DIR = Path("results/fasdnet")
SEQ_LEN = 960
PREFIX_STEPS = [48, 96, 128, 192, 240, 480, 960]
EPOCHS = 50
ALARM_WEIGHT = 0.1  # L1 weight on normalized alarm timestep


# ═══════════════════════════════════════════════════════════════════════════
# Baselines (reproduce existing numbers for an apples-to-apples anchor)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Training / evaluation
# ═══════════════════════════════════════════════════════════════════════════


def _to_device(*tensors, dev):
    return [t.to(dev) for t in tensors if t is not None]


def train_eval_task_specific(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    epochs=EPOCHS,
    lr=1e-3,
    wd=1e-4,
    bs=32,
    alarm_weight=ALARM_WEIGHT,
):
    """Train a FASDNet. Returns best-epoch metrics dict."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    cls_stride = model.cls_stride
    cls_M_train = model.cls_M

    tx = torch.from_numpy(train_x.astype(np.float32))
    cy_t = torch.from_numpy(train_y[:, 0].astype(np.int64))
    ly_t = torch.from_numpy(train_y[:, 1].astype(np.int64))
    ay_t = torch.from_numpy(train_y[:, 2].astype(np.float32))  # alarm timestep
    loader = DataLoader(
        TensorDataset(tx, cy_t, ly_t, ay_t), batch_size=bs, shuffle=True
    )
    tex = torch.from_numpy(test_x.astype(np.float32)).to(dev)
    ty_alarm = torch.from_numpy(test_y[:, 2].astype(np.float32)).to(dev)

    best = dict(score=-1.0, epoch=0, cls_ba=0.0, loc_ba=0.0)
    best_state = None
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, cy, ly, ay in loader:
            xb, cy, ly, ay = _to_device(xb, cy, ly, ay, dev=dev)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = nn.functional.cross_entropy(
                out["cls_logits"], cy
            ) + nn.functional.cross_entropy(out["loc_logits"], ly)
            if out["alarm_logit"] is not None:
                tgt_tok = (ay.long() // cls_stride).clamp(0, cls_M_train - 1)
                loss = loss + alarm_weight * nn.functional.cross_entropy(
                    out["alarm_logit"], tgt_tok
                )
            loss.backward()
            opt.step()

        if ep % 5 == 0 or ep == epochs:
            res = evaluate_task_specific(model, tex, test_y, ty_alarm, cls_stride)
            score = res["cls_ba"] + res["loc_ba"]
            if score > best["score"]:
                best = dict(score=score, epoch=ep, **res)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    train_sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    best["train_time_sec"] = train_sec
    best["params"] = sum(p.numel() for p in model.parameters())
    return best


def evaluate_task_specific(model, tex, test_y, ty_alarm, cls_stride):
    """Full-window + prefix + alarm + spike metrics on the test set."""
    model.eval()
    with torch.inference_mode():
        out = model(tex)
        cp = out["cls_logits"].argmax(-1).cpu().numpy()
        lp = out["loc_logits"].argmax(-1).cpu().numpy()
        alarm_pred = out["alarm_pred"]
        spike_rate = out["spike_rate"]
        event_sparsity = out["event_sparsity"]
        spike_count = out["spike_count"]
        # prefix cls BA (leakage-free: recompute patches from prefix)
        prefix_ba = {}
        for K in PREFIX_STEPS:
            if K >= SEQ_LEN:
                prefix_ba[K] = float(bal_acc(test_y[:, 0], cp))
                continue
            logits = model.forward_prefix_cls(tex[:, :K, :])
            prefix_ba[K] = float(bal_acc(test_y[:, 0], logits.argmax(-1).cpu().numpy()))
    res = {
        "cls_ba": float(bal_acc(test_y[:, 0], cp)),
        "loc_ba": float(bal_acc(test_y[:, 1], lp)),
        "spike_rate": (float(spike_rate) if spike_rate is not None else None),
        "event_sparsity": (
            float(event_sparsity) if event_sparsity is not None else None
        ),
        "spike_count": (float(spike_count) if spike_count is not None else None),
    }
    if alarm_pred is not None:
        mae = float((alarm_pred - ty_alarm).abs().mean().item())
        res["alarm_mae"] = mae
    else:
        res["alarm_mae"] = None
    for K in PREFIX_STEPS:
        res[f"prefix_cls_ba_{K}"] = prefix_ba[K]
    return res


def evaluate_controls(model, tex, test_y):
    """Run time-reversal and time-shuffle SNN controls; return cls BA each."""
    out = {}
    model.eval()
    with torch.inference_mode():
        for ctrl in ["none", "reverse", "shuffle"]:
            o = model(tex, snn_control=ctrl)
            out[ctrl] = float(
                bal_acc(test_y[:, 0], o["cls_logits"].argmax(-1).cpu().numpy())
            )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Variant registry
# ═══════════════════════════════════════════════════════════════════════════


def build(name: str) -> FASDNet:
    """Build one of the two locked post-ablation configurations."""
    configs = {
        "FASDNET": FASDNET_CONFIG,
        "UPPER_BOUND": UPPER_BOUND_CONFIG,
    }
    try:
        return FASDNet(**configs[name])
    except KeyError as exc:
        raise ValueError(f"Unknown locked variant: {name}") from exc


# ═══════════════════════════════════════════════════════════════════════════
# Running
# ═══════════════════════════════════════════════════════════════════════════


def run_one(
    name: str,
    model,
    seed,
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    with_controls: bool = False,
) -> dict:
    # NOTE: set_seed(seed) is called by the caller BEFORE model construction so
    # that both weight init and the dataloader shuffle follow one deterministic
    # stream from the seed.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = train_eval_task_specific(model, train_x, train_y, test_x, test_y)
    tex = torch.from_numpy(test_x.astype(np.float32)).to(dev)
    if with_controls and getattr(model, "snn", None) is not None:
        ctrl = evaluate_controls(model, tex, test_y)
        r["ctrl_none_cls_ba"] = ctrl["none"]
        r["ctrl_reverse_cls_ba"] = ctrl["reverse"]
        r["ctrl_shuffle_cls_ba"] = ctrl["shuffle"]
    r["variant"] = name
    r["seed"] = seed
    return r


# ═══════════════════════════════════════════════════════════════════════════
# IO
# ═══════════════════════════════════════════════════════════════════════════


def _row_from_result(r: dict) -> dict:
    base = {
        "variant": r["variant"],
        "seed": r["seed"],
        "params": r.get("params", ""),
        "best_epoch": r.get("epoch", ""),
        "cls_ba": r.get("cls_ba", ""),
        "loc_ba": r.get("loc_ba", ""),
        "alarm_mae": r.get("alarm_mae", ""),
        "spike_rate": r.get("spike_rate", ""),
        "event_sparsity": r.get("event_sparsity", ""),
        "spike_count": r.get("spike_count", ""),
        "train_time_sec": r.get("train_time_sec", ""),
    }
    for K in PREFIX_STEPS:
        base[f"prefix_cls_ba_{K}"] = r.get(f"prefix_cls_ba_{K}", "")
    for c in ["ctrl_none_cls_ba", "ctrl_reverse_cls_ba", "ctrl_shuffle_cls_ba"]:
        if c in r:
            base[c] = r[c]
    return base


def write_csv(rows: list[dict], path: Path) -> list[str]:
    fields = [
        "variant",
        "seed",
        "params",
        "best_epoch",
        "cls_ba",
        "loc_ba",
        "alarm_mae",
        "spike_rate",
        "event_sparsity",
        "spike_count",
        "train_time_sec",
    ]
    fields += [f"prefix_cls_ba_{K}" for K in PREFIX_STEPS]
    fields += ["ctrl_none_cls_ba", "ctrl_reverse_cls_ba", "ctrl_shuffle_cls_ba"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Saved: {path}")
    return fields


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 120}")
    print("  FASD-Net SUMMARY")
    print(f"{'=' * 120}")
    hdr = (
        f"{'Variant':<32s} {'seed':>4s} {'params':>9s} {'cls':>7s} "
        f"{'loc':>7s} {'almMAE':>7s} {'spkR':>6s} "
        f"{'p48':>6s} {'p96':>6s} {'p128':>6s} {'p192':>6s} "
        f"{'p240':>6s} {'p480':>6s} {'p960':>6s}"
    )
    print(hdr)
    print("-" * 120)

    def fmt_val(row, k, fmt="%.4f"):
        v = row.get(k)
        return (fmt % v) if isinstance(v, (int, float)) else ""

    for r in rows:
        print(
            f"{r['variant']:<32s} {r['seed']:>4} {r.get('params', ''):>9} "
            f"{fmt_val(r, 'cls_ba'):>7s} {fmt_val(r, 'loc_ba'):>7s} "
            f"{fmt_val(r, 'alarm_mae', '%.1f'):>7s} "
            f"{fmt_val(r, 'spike_rate'):>6s} "
            f"{fmt_val(r, 'prefix_cls_ba_48'):>6s} "
            f"{fmt_val(r, 'prefix_cls_ba_96'):>6s} "
            f"{fmt_val(r, 'prefix_cls_ba_128'):>6s} "
            f"{fmt_val(r, 'prefix_cls_ba_192'):>6s} "
            f"{fmt_val(r, 'prefix_cls_ba_240'):>6s} "
            f"{fmt_val(r, 'prefix_cls_ba_480'):>6s} "
            f"{fmt_val(r, 'prefix_cls_ba_960'):>6s}"
        )
    print("=" * 120)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def run(seeds, variants, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_psml()
    train_x = data["x"][data["train_idx"]]
    test_x = data["x"][data["test_idx"]]
    train_y = data["y"][data["train_idx"]]
    test_y = data["y"][data["test_idx"]]
    mu = train_x.reshape(-1, 91).mean(axis=0)
    sd = train_x.reshape(-1, 91).std(axis=0) + 1e-8
    train_x_s = (train_x - mu) / sd
    test_x_s = (test_x - mu) / sd

    all_rows: list[dict] = []
    for vname in variants:
        for seed in seeds:
            set_seed(seed)
            model = build(vname)
            r = run_one(
                vname,
                model,
                seed,
                train_x_s,
                train_y,
                test_x_s,
                test_y,
                with_controls=False,
            )
            all_rows.append(_row_from_result(r))
            print(
                f"{vname} seed={seed} cls={r['cls_ba']:.4f} "
                f"loc={r['loc_ba']:.4f} alarm_MAE={r['alarm_mae']:.1f} "
                f"spike_rate={r['spike_rate']:.4f} params={r['params']:,}",
                flush=True,
            )

    write_csv(all_rows, out_dir / "metrics.csv")
    print_summary(all_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["FASDNET", "UPPER_BOUND"],
        default=["FASDNET"],
    )
    p.add_argument("--out-dir", "--out_dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    run(args.seeds, args.variants, args.out_dir)
