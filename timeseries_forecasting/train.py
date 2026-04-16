from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingResult:
    best_state: Dict[str, torch.Tensor]
    val_loss: float
    train_time: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
) -> TrainingResult:
    criterion = nn.MSELoss()
    best_state = None
    best_val = math.inf
    no_improve = 0
    start_time = time.time()

    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_losses.append(criterion(outputs, targets).item())
        avg_val = float(np.mean(val_losses)) if val_losses else math.inf
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    train_time = time.time() - start_time
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return TrainingResult(best_state=best_state, val_loss=best_val, train_time=train_time)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            preds.append(outputs)
            trues.append(targets.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
