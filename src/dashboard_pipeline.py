"""Reusable data, training, persistence, and inference helpers for the web UI and CLI."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from src.evaluate import evaluate_predictions, reconstruction_error
from src.model import build_autoencoder
from src.train import train_autoencoder
from src.utils import ensure_dir, plot_error_distribution, plot_loss, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
LABEL_CANDIDATES = ("fraud", "class", "label", "is_fraud", "target", "anomaly")


def detect_label_column(df: pd.DataFrame) -> str | None:
    """Find a likely binary target column, case-insensitively."""
    by_lower = {str(column).strip().lower(): column for column in df.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in by_lower:
            return str(by_lower[candidate])
    return None


def _clean_numeric_data(df: pd.DataFrame, label_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    if label_col is not None:
        labels = pd.to_numeric(df[label_col], errors="coerce")
        observed = set(labels.dropna().unique().tolist())
        if not observed.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Label column '{label_col}' must contain binary values 0 and 1. "
                f"Found values such as: {sorted(observed)[:8]}"
            )
    else:
        labels = None

    excluded = [label_col] if label_col is not None else []
    numeric = df.drop(columns=excluded).select_dtypes(include=[np.number, "bool"]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("No numeric feature columns were found. The autoencoder requires numeric input features.")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid_rows = numeric.notna().all(axis=1)
    if labels is not None:
        valid_rows &= labels.notna()
        labels = labels.loc[valid_rows].astype(int)
    cleaned_df = df.loc[valid_rows].copy()
    numeric = numeric.loc[valid_rows].astype(float)
    if len(numeric) < 20:
        raise ValueError("At least 20 valid rows are required to train and evaluate the model.")
    return cleaned_df, numeric, labels


def prepare_dataset(
    source_df: pd.DataFrame,
    max_rows: int | None = 20_000,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Clean, optionally sample, split, and scale a dataset for anomaly detection.

    If a binary label is available, the model is trained only on rows labelled 0.
    """
    if source_df.empty:
        raise ValueError("The uploaded dataset is empty.")

    label_col = detect_label_column(source_df)
    clean_df, features, labels = _clean_numeric_data(source_df, label_col)
    cleaned_rows = len(clean_df)

    if max_rows is not None and len(clean_df) > max_rows:
        if labels is not None and labels.nunique() > 1 and labels.value_counts().min() >= 2:
            sampled_df, _ = train_test_split(
                clean_df,
                train_size=max_rows,
                random_state=random_state,
                stratify=labels,
            )
            clean_df = sampled_df.sort_index()
        else:
            clean_df = clean_df.sample(n=max_rows, random_state=random_state).sort_index()
        features = features.loc[clean_df.index]
        if labels is not None:
            labels = labels.loc[clean_df.index]

    if labels is not None:
        if not (labels == 0).any():
            raise ValueError(
                f"Label column '{label_col}' has no rows labelled 0. This project treats 0 as the normal class."
            )
        stratify = labels if labels.nunique() > 1 and labels.value_counts().min() >= 2 else None
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        normal_train = X_train_raw.loc[y_train == 0]
        y_test_arr: np.ndarray | None = y_test.to_numpy(dtype=np.int64)
    else:
        X_train_raw, X_test_raw = train_test_split(
            features,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        normal_train = X_train_raw
        y_test_arr = None

    if len(normal_train) < 10:
        raise ValueError("Too few normal training rows remain. Add more normal examples or increase the dataset sample size.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_train).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    label_distribution: dict[str, int] = {}
    if labels is not None:
        label_distribution = {str(int(k)): int(v) for k, v in labels.value_counts().sort_index().items()}

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test_arr,
        "scaler": scaler,
        "feature_columns": list(features.columns),
        "label_column": label_col,
        "df_test": clean_df.loc[X_test_raw.index].copy(),
        "train_rows": int(len(normal_train)),
        "test_rows": int(len(X_test_raw)),
        "valid_source_rows": int(cleaned_rows),
        "used_rows": int(len(clean_df)),
        "label_distribution": label_distribution,
    }


def train_pipeline(
    prepared: dict[str, Any],
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    latent_dim: int = 8,
    patience: int = 5,
    threshold_method: str = "percentile",
    threshold_setting: float = 99.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the model and return metrics, predictions, history, and model bundle."""
    X_train = np.asarray(prepared["X_train"], dtype=np.float32)
    X_test = np.asarray(prepared["X_test"], dtype=np.float32)
    if len(X_train) < 10 or len(X_test) < 2:
        raise ValueError("Not enough normal training or test samples. Try a larger dataset sample.")

    set_seed(random_state)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    train_array, val_array = train_test_split(
        X_train,
        test_size=0.15,
        random_state=random_state,
        shuffle=True,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(train_array))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(val_array))),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    hidden_dims = [64, 32]
    cfg = Config(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        early_stopping_patience=patience,
        random_state=random_state,
        seed=random_state,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_autoencoder(X_train.shape[1], hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
    history = train_autoencoder(model, train_loader, val_loader, cfg, device)

    train_errors = reconstruction_error(model, X_train)
    test_errors = reconstruction_error(model, X_test)
    if threshold_method == "percentile":
        threshold = float(np.percentile(train_errors, threshold_setting))
        threshold_description = f"{threshold_setting:g}th percentile of normal training errors"
    elif threshold_method == "mean_std":
        threshold = float(np.mean(train_errors) + threshold_setting * np.std(train_errors))
        threshold_description = f"Mean + {threshold_setting:g} × standard deviation"
    else:
        raise ValueError("Unsupported threshold method.")

    predictions = (test_errors > threshold).astype(np.int64)
    metrics: dict[str, Any] | None = None
    y_test = prepared["y_test"]
    if y_test is not None:
        evaluated = evaluate_predictions(y_test, predictions)
        cm = np.asarray(evaluated["confusion_matrix"])
        # Keep the confusion matrix consistently 2x2 for binary classification displays.
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        metrics = {
            "accuracy": float(evaluated["accuracy"]),
            "precision": float(evaluated["precision"]),
            "recall": float(evaluated["recall"]),
            "f1": float(evaluated["f1"]),
            "confusion_matrix": cm.astype(int).tolist(),
        }

    test_results = prepared["df_test"].copy().reset_index(drop=True)
    test_results["reconstruction_error"] = test_errors
    test_results["predicted_anomaly"] = predictions
    test_results["prediction"] = np.where(predictions == 1, "Anomaly", "Normal")

    result: dict[str, Any] = {
        "model": model,
        "scaler": prepared["scaler"],
        "threshold": threshold,
        "threshold_method": threshold_method,
        "threshold_setting": float(threshold_setting),
        "threshold_description": threshold_description,
        "history": history,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "predictions": predictions,
        "test_results": test_results,
        "metrics": metrics,
        "feature_columns": prepared["feature_columns"],
        "label_column": prepared["label_column"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "anomalies_detected": int(predictions.sum()),
        "device": str(device),
        "data_summary": {
            "used_rows": prepared["used_rows"],
            "valid_source_rows": prepared["valid_source_rows"],
            "label_distribution": prepared["label_distribution"],
        },
        "input_defaults": prepared["scaler"].mean_.astype(float).tolist(),
        "hidden_dims": hidden_dims,
        "latent_dim": int(latent_dim),
    }
    save_artifacts(result)
    return result


def save_artifacts(result: dict[str, Any]) -> None:
    """Persist the model, scaler, metadata, plots, and summary metrics."""
    ensure_dir(ARTIFACT_DIR)
    model = result["model"]
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(cpu_state, ARTIFACT_DIR / "autoencoder.pt")
    with (ARTIFACT_DIR / "scaler.pkl").open("wb") as file:
        pickle.dump(result["scaler"], file)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": float(result["threshold"]),
        "threshold_method": result["threshold_method"],
        "threshold_setting": float(result["threshold_setting"]),
        "threshold_description": result["threshold_description"],
        "feature_columns": result["feature_columns"],
        "label_column": result["label_column"],
        "hidden_dims": result["hidden_dims"],
        "latent_dim": result["latent_dim"],
        "input_defaults": result["input_defaults"],
    }
    with (ARTIFACT_DIR / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    summary = {
        "threshold": float(result["threshold"]),
        "threshold_description": result["threshold_description"],
        "train_samples": int(result["train_samples"]),
        "test_samples": int(result["test_samples"]),
        "anomalies_detected": int(result["anomalies_detected"]),
        "device": result["device"],
        "metrics": result["metrics"],
        "history": result["history"],
    }
    with (ARTIFACT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plot_loss(result["history"], ARTIFACT_DIR / "loss_curve.png")
    plot_error_distribution(
        result["train_errors"], result["test_errors"], result["threshold"],
        ARTIFACT_DIR / "error_distribution.png",
    )


def load_saved_bundle() -> dict[str, Any] | None:
    """Load the most recently saved model for inference after an app restart."""
    model_path = ARTIFACT_DIR / "autoencoder.pt"
    scaler_path = ARTIFACT_DIR / "scaler.pkl"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    if not (model_path.exists() and scaler_path.exists() and metadata_path.exists()):
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        with scaler_path.open("rb") as file:
            scaler = pickle.load(file)
        model = build_autoencoder(
            input_dim=len(metadata["feature_columns"]),
            hidden_dims=metadata.get("hidden_dims", [64, 32]),
            latent_dim=int(metadata.get("latent_dim", 8)),
        )
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:  # Compatibility with earlier PyTorch versions.
            state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return {
            "model": model,
            "scaler": scaler,
            "threshold": float(metadata["threshold"]),
            "threshold_method": metadata["threshold_method"],
            "threshold_setting": float(metadata["threshold_setting"]),
            "threshold_description": metadata["threshold_description"],
            "feature_columns": metadata["feature_columns"],
            "label_column": metadata.get("label_column"),
            "input_defaults": metadata.get("input_defaults", scaler.mean_.astype(float).tolist()),
            "saved_at": metadata.get("created_at_utc"),
            "model_loaded_from_disk": True,
        }
    except (OSError, ValueError, KeyError, TypeError, pickle.PickleError, json.JSONDecodeError):
        return None


def score_dataframe(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Score transactions from a dataframe with the model's required features."""
    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    numeric = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    invalid_rows = numeric.isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(
            f"{int(invalid_rows.sum())} rows contain missing or non-numeric values in required features. "
            "Clean those rows before scoring."
        )

    scaled = bundle["scaler"].transform(numeric.astype(float)).astype(np.float32)
    errors = reconstruction_error(bundle["model"], scaled)
    predicted = (errors > float(bundle["threshold"])).astype(np.int64)
    scored = df.copy()
    scored["reconstruction_error"] = errors
    scored["predicted_anomaly"] = predicted
    scored["prediction"] = np.where(predicted == 1, "Anomaly", "Normal")
    return scored
