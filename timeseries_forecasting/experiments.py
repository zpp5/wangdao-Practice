from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import (
    apply_scaler,
    build_scaler_params,
    create_windows,
    fit_scaler,
    inverse_scale_targets,
    split_dataframe,
    WindowedDataset,
)
from .models import build_model
from .train import TrainingResult, compute_metrics, predict, train_model


@dataclass
class ExperimentConfig:
    seq_len: int
    pred_len: int
    model_name: str
    input_cols: List[str]
    target_cols: List[str]
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    weight_decay: float


@dataclass
class ExperimentOutcome:
    record: Dict[str, object]
    predictions: np.ndarray
    targets: np.ndarray


def build_dataloaders(
    df: pd.DataFrame,
    input_cols: List[str],
    target_cols: List[str],
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    scale_cols: List[str],
) -> Tuple[Dict[str, DataLoader], Dict[str, Dict[str, float]]]:
    train_df, val_df, test_df = split_dataframe(df, train_ratio, val_ratio)
    scaler = fit_scaler(train_df, scale_cols)
    scaler_params = build_scaler_params(scaler, scale_cols)

    train_scaled = apply_scaler(train_df, scaler, scale_cols)
    val_scaled = apply_scaler(val_df, scaler, scale_cols)
    test_scaled = apply_scaler(test_df, scaler, scale_cols)

    train_x, train_y = create_windows(
        train_scaled[input_cols].values,
        train_scaled[target_cols].values,
        seq_len,
        pred_len,
    )
    val_x, val_y = create_windows(
        val_scaled[input_cols].values,
        val_scaled[target_cols].values,
        seq_len,
        pred_len,
    )
    test_x, test_y = create_windows(
        test_scaled[input_cols].values,
        test_scaled[target_cols].values,
        seq_len,
        pred_len,
    )

    train_loader = DataLoader(WindowedDataset(train_x, train_y), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(WindowedDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WindowedDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    return {"train": train_loader, "val": val_loader, "test": test_loader}, scaler_params


def generate_param_grid(
    hidden_sizes: Iterable[int],
    num_layers: Iterable[int],
    dropouts: Iterable[float],
    learning_rates: Iterable[float],
    batch_sizes: Iterable[int],
) -> List[Tuple[int, int, float, float, int]]:
    return list(itertools.product(hidden_sizes, num_layers, dropouts, learning_rates, batch_sizes))


def run_single_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
    train_ratio: float,
    val_ratio: float,
    scale_cols: List[str],
    device: torch.device,
) -> Tuple[Dict[str, object], ExperimentOutcome]:
    loaders, scaler_params = build_dataloaders(
        df,
        config.input_cols,
        config.target_cols,
        config.seq_len,
        config.pred_len,
        train_ratio,
        val_ratio,
        config.batch_size,
        scale_cols,
    )

    model = build_model(
        config.model_name,
        input_size=len(config.input_cols),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        output_size=len(config.target_cols),
        pred_len=config.pred_len,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    training_result: TrainingResult = train_model(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        device,
        epochs=config.epochs,
        patience=config.patience,
    )
    model.load_state_dict(training_result.best_state)

    val_true, val_pred = predict(model, loaders["val"], device)
    test_true, test_pred = predict(model, loaders["test"], device)

    val_true_rescaled = inverse_scale_targets(val_true, scaler_params, config.target_cols)
    val_pred_rescaled = inverse_scale_targets(val_pred, scaler_params, config.target_cols)
    test_true_rescaled = inverse_scale_targets(test_true, scaler_params, config.target_cols)
    test_pred_rescaled = inverse_scale_targets(test_pred, scaler_params, config.target_cols)

    val_metrics = compute_metrics(val_true_rescaled, val_pred_rescaled)
    test_metrics = compute_metrics(test_true_rescaled, test_pred_rescaled)

    record = {
        "model": config.model_name,
        "task": f"{len(config.input_cols)}in-{len(config.target_cols)}out",
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_mape": val_metrics["mape"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_mape": test_metrics["mape"],
        "train_time_sec": training_result.train_time,
    }

    outcome = ExperimentOutcome(record=record, predictions=test_pred_rescaled, targets=test_true_rescaled)
    return record, outcome


def build_task_configs(all_cols: List[str], target_col: str) -> Dict[str, Tuple[List[str], List[str]]]:
    if target_col not in all_cols:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    return {
        "single-to-single": ([target_col], [target_col]),
        "single-to-multi": ([target_col], all_cols),
        "multi-to-multi": (all_cols, all_cols),
    }


def run_experiments(
    df: pd.DataFrame,
    all_cols: List[str],
    target_col: str,
    seq_lens: Iterable[int],
    pred_lens: Iterable[int],
    model_names: Iterable[str],
    param_grid: List[Tuple[int, int, float, float, int]],
    epochs: int,
    patience: int,
    train_ratio: float,
    val_ratio: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, ExperimentOutcome]]:
    task_configs = build_task_configs(all_cols, target_col)
    results: List[Dict[str, object]] = []
    best_outcomes: Dict[str, ExperimentOutcome] = {}

    for task_name, (input_cols, target_cols) in task_configs.items():
        best_val = float("inf")
        for seq_len, pred_len, model_name, (hidden_size, num_layers, dropout, lr, batch_size) in itertools.product(
            seq_lens, pred_lens, model_names, param_grid
        ):
            config = ExperimentConfig(
                seq_len=seq_len,
                pred_len=pred_len,
                model_name=model_name,
                input_cols=input_cols,
                target_cols=target_cols,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=lr,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                weight_decay=weight_decay,
            )
            record, outcome = run_single_experiment(
                df,
                config,
                train_ratio,
                val_ratio,
                scale_cols=all_cols,
                device=device,
            )
            results.append(record)
            if record["val_rmse"] < best_val:
                best_val = record["val_rmse"]
                best_outcomes[task_name] = outcome
    return results, best_outcomes
