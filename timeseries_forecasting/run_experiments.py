from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data import clip_outliers, detect_outliers_iqr, impute_missing, load_csv, set_seed
from .experiments import generate_param_grid, run_experiments


def parse_list(value: str, dtype=float) -> List:
    return [dtype(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM/GRU forecasting experiments.")
    parser.add_argument("--data-path", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--time-col", default=None, help="Optional time column name.")
    parser.add_argument("--columns", default=None, help="Comma-separated list of sensor columns to use.")
    parser.add_argument("--target-col", required=True, help="Target column for single-variable tasks.")
    parser.add_argument("--output-dir", default="experiment_outputs", help="Output directory.")
    parser.add_argument("--seq-lens", default="24,48,96,168", help="Comma-separated input window sizes.")
    parser.add_argument("--pred-lens", default="1,24,48", help="Comma-separated prediction lengths.")
    parser.add_argument("--models", default="lstm,gru", help="Comma-separated models (lstm, gru).")
    parser.add_argument("--hidden-sizes", default="64,128", help="Comma-separated hidden sizes.")
    parser.add_argument("--num-layers", default="1,2", help="Comma-separated layer counts.")
    parser.add_argument("--dropouts", default="0.0,0.1", help="Comma-separated dropout values.")
    parser.add_argument("--learning-rates", default="0.001", help="Comma-separated learning rates.")
    parser.add_argument("--batch-sizes", default="64", help="Comma-separated batch sizes.")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def plot_forecast(targets, predictions, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    true_series = targets[:, 0, 0]
    pred_series = predictions[:, 0, 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(true_series, label="True")
    ax.plot(pred_series, label="Predicted")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.data_path, time_col=args.time_col)
    if args.columns:
        all_cols = [col.strip() for col in args.columns.split(",") if col.strip()]
    else:
        all_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not all_cols:
        raise ValueError("No numeric columns found for experiments.")

    outlier_stats = detect_outliers_iqr(df[all_cols], all_cols)
    cleaned = clip_outliers(impute_missing(df[all_cols]), outlier_stats)

    seq_lens = [int(x) for x in parse_list(args.seq_lens, int)]
    pred_lens = [int(x) for x in parse_list(args.pred_lens, int)]
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    hidden_sizes = [int(x) for x in parse_list(args.hidden_sizes, int)]
    num_layers = [int(x) for x in parse_list(args.num_layers, int)]
    dropouts = [float(x) for x in parse_list(args.dropouts, float)]
    learning_rates = [float(x) for x in parse_list(args.learning_rates, float)]
    batch_sizes = [int(x) for x in parse_list(args.batch_sizes, int)]

    param_grid = generate_param_grid(hidden_sizes, num_layers, dropouts, learning_rates, batch_sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results, best_outcomes = run_experiments(
        cleaned,
        all_cols,
        target_col=args.target_col,
        seq_lens=seq_lens,
        pred_lens=pred_lens,
        model_names=model_names,
        param_grid=param_grid,
        epochs=args.epochs,
        patience=args.patience,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        weight_decay=args.weight_decay,
        device=device,
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    best_records = []
    for task, outcome in best_outcomes.items():
        best_records.append(outcome.record | {"task_name": task})
        plot_forecast(
            outcome.targets,
            outcome.predictions,
            output_dir / "best_forecasts" / f"forecast_{task}.png",
            title=f"Best Forecast - {task}",
        )

    if best_records:
        pd.DataFrame(best_records).to_csv(output_dir / "best_results.csv", index=False)


if __name__ == "__main__":
    main()
