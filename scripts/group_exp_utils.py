"""Shared data and reproducibility helpers for retained FASD-Net scripts."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "PSML" / "processed_dataset" / "classification.pkl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_psml() -> dict[str, np.ndarray | list[str]]:
    """Load the official processed classification split used by the paper."""
    with DATA_PATH.open("rb") as handle:
        data = pickle.load(handle)
    features = np.asarray(data["feature_list"])[:, :, 1:]
    labels = np.asarray(data["label_list"])
    return {
        "x": features,
        "y": labels,
        "names": data["feature_names"][1:],
        "train_idx": np.asarray(data["data_split"]["train"]),
        "test_idx": np.asarray(data["data_split"]["test"]),
    }


def bal_acc(target: np.ndarray, prediction: np.ndarray) -> float:
    """Balanced accuracy without introducing another metrics dependency."""
    classes = np.unique(target)
    return float(np.mean([np.mean(prediction[target == c] == c) for c in classes]))
