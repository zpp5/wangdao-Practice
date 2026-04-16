from __future__ import annotations

import argparse
from pathlib import Path

from statsmodels.tsa.seasonal import seasonal_decompose

from .analysis import compute_basic_stats, compute_seasonal_stats, save_summary
from .data import clip_outliers, detect_outliers_iqr, impute_missing, load_csv
from .visualize import plot_acf_pacf, plot_correlation_heatmap, plot_distributions, plot_timeseries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ETT transformer oil temperature dataset.")
    parser.add_argument("--data-path", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--time-col", default=None, help="Optional time column name.")
    parser.add_argument("--columns", default=None, help="Comma-separated list of sensor columns to use.")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory for analysis outputs.")
    parser.add_argument("--period", type=int, default=24, help="Seasonal decomposition period.")
    parser.add_argument("--lags", type=int, default=48, help="Lags for ACF/PACF plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = load_csv(args.data_path, time_col=args.time_col)

    if args.columns:
        columns = [col.strip() for col in args.columns.split(",") if col.strip()]
    else:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns found for analysis.")

    basic_stats = compute_basic_stats(df, columns)
    outlier_stats = detect_outliers_iqr(df, columns)
    cleaned = clip_outliers(impute_missing(df[columns]), outlier_stats)

    seasonal_stats = compute_seasonal_stats(cleaned, columns, args.period)
    summary = {"basic": basic_stats, "seasonal": seasonal_stats}
    save_summary(output_dir / "summary.json", summary)

    plot_timeseries(cleaned, columns, output_dir / "timeseries.png")
    plot_correlation_heatmap(cleaned, columns, output_dir / "correlation_heatmap.png")
    plot_distributions(cleaned, columns, output_dir / "distributions.png")
    plot_acf_pacf(cleaned, columns, output_dir / "acf_pacf", lags=args.lags)

    decomposition_dir = output_dir / "decomposition"
    decomposition_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        series = cleaned[col].dropna()
        if len(series) < args.period * 2:
            continue
        decomposition = seasonal_decompose(series, model="additive", period=args.period, extrapolate_trend="freq")
        fig = decomposition.plot()
        fig.suptitle(f"Seasonal Decomposition - {col}", y=1.02)
        fig.savefig(decomposition_dir / f"decomposition_{col}.png", dpi=150, bbox_inches="tight")
        fig.clf()


if __name__ == "__main__":
    main()
