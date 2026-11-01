from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config


def build_dataloaders(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X_train, dtype=np.float32))), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X_val, dtype=np.float32))), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X_test, dtype=np.float32))), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


def _epoch_loss(model: torch.nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for (batch,) in data_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def train_autoencoder(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(cfg.num_epochs):
        model.train()
        batch_losses = []

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss = _epoch_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_model(model: torch.nn.Module, X_train: np.ndarray) -> Dict[str, list[float]]:
    cfg = Config()

    X_train = np.asarray(X_train, dtype=np.float32)
    if len(X_train) < 2:
        raise ValueError("Need at least 2 training samples to create a validation split.")

    train_split, val_split = train_test_split(
        X_train,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_split)), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_split)), batch_size=cfg.batch_size, shuffle=False
    )

    return train_autoencoder(model, train_loader, val_loader, cfg, device)