#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_research import (
    build_head_candidate_diagnostics,
    build_monthly_head_diagnostics,
    build_state_slice_diagnostics,
    build_top_pick_feature_profile,
    evaluate_topk_returns,
    evaluate_topk_rollover_returns,
)


def find_prediction_path(year, model, variant):
    pattern = f"long_term_forecast_market_{year}_{variant}_{model}_*/top1_predictions.csv"
    matches = sorted((ROOT / "test_results").glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No prediction file found for year={year}, model={model}, variant={variant}"
        )
    return matches[0]


def load_feature_frame(cache_path, year, columns):
    frame = pd.read_parquet(cache_path, columns=["date", "code", *columns])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[frame["date"].dt.year == int(year)].copy()


def attach_state_columns(prediction_frame, feature_frame, state_columns):
    keep_columns = ["date", "code", *state_columns]
    state_frame = feature_frame[keep_columns].drop_duplicates(subset=["date", "code"])
    merged = prediction_frame.merge(state_frame, on=["date", "code"], how="left")
    return merged


def build_rollover_rows(frame, topk_list):
    rows = []
    for top_k in topk_list:
        strict_metrics = evaluate_topk_returns(frame, top_k=top_k)
        rollover_metrics = evaluate_topk_rollover_returns(frame, top_k=top_k)
        rows.append(
            {
                "top_k": int(top_k),
                "strict_mean_return": float(strict_metrics["mean_return"]),
                "strict_cumulative_return": float(strict_metrics["cumulative_return"]),
                "strict_sharpe": float(strict_metrics["sharpe"]),
                "rollover_mean_return": float(rollover_metrics["mean_return"]),
                "rollover_cumulative_return": float(rollover_metrics["cumulative_return"]),
                "rollover_sharpe": float(rollover_metrics["sharpe"]),
                "rollover_positive_rate": float(rollover_metrics["positive_rate"]),
            }
        )
    return pd.DataFrame(rows)


def build_worst_day_frame(daily_rows, top_n):
    if not daily_rows:
        return pd.DataFrame()
    daily = pd.DataFrame(daily_rows).copy()
    return (
        daily.sort_values(
            ["top1_true", "regret_vs_best_tradable", "hit_top1_in_true20"],
            ascending=[True, False, True],
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one_variant(
    year,
    model,
    variant,
    output_dir,
    cache_path,
    feature_columns,
    state_columns,
    pred_topk_list,
    true_topk_list,
    rollover_topk_list,
    worst_day_top_n,
    top_pick_top_n,
):
    prediction_path = find_prediction_path(year, model, variant)
    prediction_frame = pd.read_csv(prediction_path, parse_dates=["date"])
    diagnostics = build_head_candidate_diagnostics(
        prediction_frame,
        pred_topk_list=pred_topk_list,
        true_topk_list=true_topk_list,
    )
    load_columns = sorted(set(feature_columns) | set(state_columns))
    feature_frame = load_feature_frame(cache_path, year, columns=load_columns)
    prediction_with_state = attach_state_columns(prediction_frame, feature_frame, state_columns)
    monthly = build_monthly_head_diagnostics(diagnostics["daily"])
    state_slices = build_state_slice_diagnostics(
        prediction_with_state,
        state_columns=tuple(state_columns),
        topk_list=tuple(rollover_topk_list),
    )
    top_pick_profile = build_top_pick_feature_profile(
        prediction_frame,
        feature_frame,
        feature_columns=tuple(feature_columns),
        top_n=top_pick_top_n,
    )
    rollover = build_rollover_rows(prediction_frame, rollover_topk_list)
    worst_days = build_worst_day_frame(diagnostics["daily"], top_n=worst_day_top_n)

    variant_dir = output_dir / f"{model}_{year}_{variant}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(diagnostics["daily"]).to_csv(variant_dir / "daily.csv", index=False)
    monthly.to_csv(variant_dir / "monthly.csv", index=False)
    top_pick_profile.to_csv(variant_dir / "top_pick_profile.csv", index=False)
    rollover.to_csv(variant_dir / "rollover.csv", index=False)
    worst_days.to_csv(variant_dir / "worst_days.csv", index=False)
    write_json(variant_dir / "summary.json", diagnostics["summary"])
    write_json(variant_dir / "state_slices.json", state_slices)
    write_json(
        variant_dir / "meta.json",
        {
            "year": int(year),
            "model": model,
            "variant": variant,
            "prediction_path": str(prediction_path),
            "cache_path": str(cache_path),
            "feature_columns": list(feature_columns),
            "state_columns": list(state_columns),
        },
    )

    summary_row = {
        "year": int(year),
        "model": model,
        "variant": variant,
        "prediction_path": str(prediction_path),
        **diagnostics["summary"],
    }
    for _, row in rollover.iterrows():
        top_k = int(row["top_k"])
        summary_row[f"rollover_top{top_k}_mean_return"] = float(row["rollover_mean_return"])
        summary_row[f"rollover_top{top_k}_sharpe"] = float(row["rollover_sharpe"])
    return summary_row


def main():
    parser = argparse.ArgumentParser(description="Diagnose regime/style failure for market daily prediction results")
    parser.add_argument("--year", type=int, action="append", required=True, help="Repeatable target year")
    parser.add_argument("--variant", action="append", required=True, help="Repeatable variant name")
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--cache_path", default="cache/market_daily_features_full2010.parquet")
    parser.add_argument("--output_dir", default="logs/regime_failure_diagnostics")
    parser.add_argument("--feature_columns", default="log_amount,avg_amount_20,turnover_rate,avg_turnover_20,amplitude,avg_amplitude_20,ret_20,ret_60,vol_20,vol_60,market_cc_mean,market_amount_top10_share")
    parser.add_argument("--state_columns", default="market_cc_mean,market_cc_std,market_amount_top10_share,market_amplitude_mean,market_vol_20_mean")
    parser.add_argument("--pred_topk_list", default="10,20")
    parser.add_argument("--true_topk_list", default="10,20")
    parser.add_argument("--rollover_topk_list", default="1,3,5,10")
    parser.add_argument("--worst_day_top_n", type=int, default=15)
    parser.add_argument("--top_pick_top_n", type=int, default=20)
    args = parser.parse_args()

    feature_columns = [item.strip() for item in args.feature_columns.split(",") if item.strip()]
    state_columns = [item.strip() for item in args.state_columns.split(",") if item.strip()]
    pred_topk_list = tuple(int(item) for item in args.pred_topk_list.split(",") if item.strip())
    true_topk_list = tuple(int(item) for item in args.true_topk_list.split(",") if item.strip())
    rollover_topk_list = tuple(int(item) for item in args.rollover_topk_list.split(",") if item.strip())

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant in args.variant:
        for year in args.year:
            summary_rows.append(
                run_one_variant(
                    year=year,
                    model=args.model,
                    variant=variant,
                    output_dir=output_dir,
                    cache_path=ROOT / args.cache_path,
                    feature_columns=feature_columns,
                    state_columns=state_columns,
                    pred_topk_list=pred_topk_list,
                    true_topk_list=true_topk_list,
                    rollover_topk_list=rollover_topk_list,
                    worst_day_top_n=args.worst_day_top_n,
                    top_pick_top_n=args.top_pick_top_n,
                )
            )

    summary_frame = pd.DataFrame(summary_rows).sort_values(["year", "variant"]).reset_index(drop=True)
    summary_path = output_dir / f"{args.model}_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(json.dumps({"summary_path": str(summary_path), "rows": int(summary_frame.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
