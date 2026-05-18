import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_research import evaluate_prediction_frame, prepare_market_dataframe


def build_parser():
    parser = argparse.ArgumentParser(description="Run market_daily baselines")
    parser.add_argument("--parquet_path", type=str, default="./market_daily.parquet")
    parser.add_argument("--cache_path", type=str, default="./cache/market_daily_features.parquet")
    parser.add_argument("--output_dir", type=str, default="./results/market_baselines")
    parser.add_argument("--fold_years", type=str, default="2019,2021")
    parser.add_argument("--market_start_year", type=int, default=2010)
    parser.add_argument("--market_min_history", type=int, default=120)
    parser.add_argument("--market_min_avg_amount", type=float, default=2e7)
    return parser


def fold_test_frame(frame, fold_year):
    test_start = f"{fold_year}-01-01"
    test_end = f"{fold_year}-12-31"
    if fold_year == 2024:
        test_end = "2024-09-02"
    test_frame = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)].copy()
    return test_frame


def run_factor(frame, score_column):
    pred_frame = frame[["date", "code", "label"]].copy()
    pred_frame["pred"] = frame[score_column]
    pred_frame = pred_frame.rename(columns={"label": "true"})
    pred_frame = pred_frame.dropna(subset=["pred", "true"])
    return evaluate_prediction_frame(pred_frame)


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    frame = prepare_market_dataframe(
        parquet_path=args.parquet_path,
        start_date=f"{args.market_start_year}-01-01",
        min_history=args.market_min_history,
        min_avg_amount=args.market_min_avg_amount,
        cache_path=args.cache_path,
    )
    fold_years = [int(item) for item in args.fold_years.split(",") if item]
    factors = {
        "momentum_20": "ret_20",
        "reversal_5": "ret_5",
        "turnover": "turnover_rate",
        "risk_adjusted_momentum": "ret_20",
    }

    summary = {}
    for fold_year in fold_years:
        fold_frame = fold_test_frame(frame, fold_year)
        fold_summary = {}
        for factor_name, score_column in factors.items():
            score_frame = fold_frame.copy()
            if factor_name == "reversal_5":
                score_frame[score_column] = -score_frame[score_column]
            if factor_name == "risk_adjusted_momentum":
                score_frame["ret_20"] = score_frame["ret_20"] / (score_frame["vol_20"].abs() + 1e-6)
            fold_summary[factor_name] = run_factor(score_frame, score_column)
        summary[str(fold_year)] = fold_summary

    output_path = os.path.join(args.output_dir, "baseline_summary.json")
    with open(output_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
