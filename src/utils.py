import json
import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_loss(history: dict, output_path: Path | None = None) -> Path:
    output_path = output_path or Path("artifacts") / "loss_curve.png"
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 5))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_error_distribution(
    train_errors: np.ndarray, test_errors: np.ndarray, threshold: float, output_path: Path | None = None
) -> Path:
    output_path = output_path or Path("artifacts") / "error_distribution.png"
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 5))
    plt.hist(train_errors, bins=40, alpha=0.6, label="train_errors")
    plt.hist(test_errors, bins=40, alpha=0.6, label="test_errors")
    plt.axvline(threshold, color="red", linestyle="--", label="threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path