import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv(path: str, time_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    return df


def summarize_missing(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, int]:
    return df[list(columns)].isna().sum().astype(int).to_dict()


def detect_outliers_iqr(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in columns:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        stats[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
        }
    return stats


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        interpolated = df.interpolate(method="time")
    else:
        interpolated = df.interpolate(method="linear")
    return interpolated.ffill().bfill()


def clip_outliers(df: pd.DataFrame, outlier_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    clipped = df.copy()
    for col, stats in outlier_stats.items():
        clipped[col] = clipped[col].clip(lower=stats["lower"], upper=stats["upper"])
    return clipped


def split_dataframe(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame, scale_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[scale_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, scale_cols: List[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaled[scale_cols] = scaler.transform(df[scale_cols])
    return scaled


def build_scaler_params(scaler: StandardScaler, scale_cols: List[str]) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}
    for col, mean, scale in zip(scale_cols, scaler.mean_, scaler.scale_):
        params[col] = {"mean": float(mean), "scale": float(scale)}
    return params


def inverse_scale_targets(
    values: np.ndarray, params: Dict[str, Dict[str, float]], target_cols: List[str]
) -> np.ndarray:
    restored = values.copy()
    for idx, col in enumerate(target_cols):
        mean = params[col]["mean"]
        scale = params[col]["scale"]
        restored[..., idx] = restored[..., idx] * scale + mean
    return restored


def create_windows(
    inputs: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(inputs) != len(targets):
        raise ValueError("Inputs and targets must have the same length.")
    if len(inputs) < seq_len + pred_len:
        raise ValueError("Not enough data to create windows.")
    x_samples = []
    y_samples = []
    for start in range(0, len(inputs) - seq_len - pred_len + 1):
        end = start + seq_len
        target_end = end + pred_len
        x_samples.append(inputs[start:end])
        y_samples.append(targets[end:target_end])
    return np.asarray(x_samples, dtype=np.float32), np.asarray(y_samples, dtype=np.float32)


class WindowedDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])
