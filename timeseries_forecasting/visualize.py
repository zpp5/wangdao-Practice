from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_timeseries(df: pd.DataFrame, columns: Iterable[str], output_path: Path) -> None:
    cols = list(columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.5 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], label=col)
        ax.set_title(f"Time Series - {col}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, columns: Iterable[str], output_path: Path) -> None:
    cols = list(columns)
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cols)), labels=cols)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, columns: Iterable[str], output_path: Path) -> None:
    cols = list(columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.5 * len(cols)))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=50, alpha=0.7, color="steelblue")
        ax.set_title(f"Distribution - {col}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_acf_pacf(
    df: pd.DataFrame,
    columns: Iterable[str],
    output_dir: Path,
    lags: int = 48,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(df[col].dropna(), ax=axes[0], lags=lags)
        plot_pacf(df[col].dropna(), ax=axes[1], lags=lags, method="ywm")
        axes[0].set_title(f"ACF - {col}")
        axes[1].set_title(f"PACF - {col}")
        fig.tight_layout()
        fig.savefig(output_dir / f"acf_pacf_{col}.png", dpi=150)
        plt.close(fig)
