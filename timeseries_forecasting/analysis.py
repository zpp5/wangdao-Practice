from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from .data import detect_outliers_iqr, summarize_missing


def compute_basic_stats(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Dict[str, int]]:
    missing = summarize_missing(df, columns)
    outliers = detect_outliers_iqr(df, columns)
    return {
        "missing": missing,
        "outliers": {col: stats["count"] for col, stats in outliers.items()},
    }


def compute_seasonal_stats(
    df: pd.DataFrame, columns: Iterable[str], period: int
) -> Dict[str, Dict[str, float]]:
    seasonal_stats: Dict[str, Dict[str, float]] = {}
    for col in columns:
        series = df[col].dropna()
        if len(series) < period * 2:
            continue
        decomposition = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
        seasonal_stats[col] = {
            "trend_std": float(decomposition.trend.dropna().std()),
            "seasonal_std": float(decomposition.seasonal.dropna().std()),
            "residual_std": float(decomposition.resid.dropna().std()),
        }
    return seasonal_stats


def save_summary(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
