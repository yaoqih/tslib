#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_live_proxy import (
    apply_live_trading_proxy,
    build_daily_top1_strategy_frame,
    build_state_gated_top1_strategy_from_daily_state,
    summarize_live_proxy,
)
from utils.market_research import residualize_prediction_scores


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


def merge_features(prediction_frame, feature_frame, columns):
    feature_keep = feature_frame[["date", "code", *columns]].drop_duplicates(subset=["date", "code"])
    return prediction_frame.merge(feature_keep, on=["date", "code"], how="left")


def build_style_neutral_frame(prediction_frame, feature_frame, style_columns):
    residual_frame = residualize_prediction_scores(
        prediction_frame=prediction_frame,
        feature_frame=feature_frame,
        feature_columns=tuple(style_columns),
    ).copy()
    residual_frame["pred"] = residual_frame["pred_resid"].astype(float)
    keep_columns = ["date", "code", "pred", "true"]
    if "tradable" in residual_frame.columns:
        keep_columns.append("tradable")
    return residual_frame[keep_columns].copy()


def build_strategy_comparison_artifacts(strategy_frames):
    yearly_rows = []
    curve_parts = []
    summary_rows = []
    for strategy_name, year_map in strategy_frames.items():
        strategy_years = []
        for year, strategy_frame in sorted(year_map.items()):
            proxy = apply_live_trading_proxy(strategy_frame, buy_cost_bps=0.0, sell_cost_bps=0.0)
            summary = summarize_live_proxy(proxy)
            gated_off_rate = (
                float(strategy_frame["state_gated_off"].mean())
                if "state_gated_off" in strategy_frame.columns and not strategy_frame.empty
                else 0.0
            )
            yearly_rows.append(
                {
                    "strategy_name": strategy_name,
                    "year": int(year),
                    "gated_off_rate": gated_off_rate,
                    **summary,
                }
            )
            current_curve = proxy[["date", "code", "net_return"]].copy()
            current_curve["strategy_name"] = strategy_name
            current_curve["year"] = int(year)
            strategy_years.append(current_curve)

        if strategy_years:
            combined = pd.concat(strategy_years, ignore_index=True).sort_values("date").reset_index(drop=True)
            combined["cumulative_return"] = (1.0 + combined["net_return"]).cumprod() - 1.0
            curve_parts.append(combined)
            summary_rows.append(
                {
                    "strategy_name": strategy_name,
                    "mean_return_avg": float(pd.DataFrame(yearly_rows).loc[pd.DataFrame(yearly_rows)["strategy_name"] == strategy_name, "mean_return"].mean()),
                    "cumulative_return_final": float(combined["cumulative_return"].iloc[-1]) if not combined.empty else 0.0,
                    "sharpe_avg": float(pd.DataFrame(yearly_rows).loc[pd.DataFrame(yearly_rows)["strategy_name"] == strategy_name, "sharpe"].mean()),
                    "switch_count_total": int(pd.DataFrame(yearly_rows).loc[pd.DataFrame(yearly_rows)["strategy_name"] == strategy_name, "switch_count"].sum()),
                    "same_code_streak_days_total": int(pd.DataFrame(yearly_rows).loc[pd.DataFrame(yearly_rows)["strategy_name"] == strategy_name, "same_code_streak_days"].sum()),
                    "gated_off_rate_avg": float(pd.DataFrame(yearly_rows).loc[pd.DataFrame(yearly_rows)["strategy_name"] == strategy_name, "gated_off_rate"].mean()),
                }
            )

    yearly_frame = pd.DataFrame(yearly_rows).sort_values(["strategy_name", "year"]).reset_index(drop=True)
    curve_frame = pd.concat(curve_parts, ignore_index=True).sort_values(["strategy_name", "date"]).reset_index(drop=True) if curve_parts else pd.DataFrame()
    summary_frame = pd.DataFrame(summary_rows).sort_values("strategy_name").reset_index(drop=True)
    return yearly_frame, curve_frame, summary_frame


def plot_cumulative_curves(curve_frame, output_path):
    plt.figure(figsize=(12, 6))
    for strategy_name, group in curve_frame.groupby("strategy_name", sort=False):
        plt.plot(group["date"], group["cumulative_return"], label=strategy_name, linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.legend()
    plt.title("2018-2025 Cumulative Return Curves")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare final plain/guarded strategies across 2018-2025")
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--variant", default="single_head_csrank_topq_v1")
    parser.add_argument("--cache_path", default="cache/market_daily_features_full2010.parquet")
    parser.add_argument("--style_columns", default="log_amount,turnover_rate,amplitude,ret_20,vol_20")
    parser.add_argument("--output_dir", default="logs/final_state_guard_compare")
    args = parser.parse_args()

    years = [int(y) for y in args.year]
    style_columns = [item.strip() for item in args.style_columns.split(",") if item.strip()]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_frames = {
        "baseline_plain_top1": {},
        "baseline_amplitude_low_cash": {},
        "style_neutral_amountshare_low_cash": {},
    }

    for year in years:
        prediction_path = find_prediction_path(year, args.model, args.variant)
        prediction_frame = pd.read_csv(prediction_path, parse_dates=["date"])
        feature_frame = load_feature_frame(
            ROOT / args.cache_path,
            year,
            columns=sorted(set(style_columns) | {"market_amplitude_mean", "market_amount_top10_share"}),
        )

        baseline_frame = prediction_frame.copy()
        style_neutral_frame = build_style_neutral_frame(prediction_frame, feature_frame, style_columns)

        baseline_with_state = merge_features(baseline_frame, feature_frame, ["market_amplitude_mean"])
        style_with_state = merge_features(style_neutral_frame, feature_frame, ["market_amount_top10_share"])

        amp_daily_state = (
            baseline_with_state.groupby("date", sort=True)["market_amplitude_mean"]
            .mean()
            .reset_index()
            .rename(columns={"market_amplitude_mean": "state_value"})
        )
        amt_daily_state = (
            style_with_state.groupby("date", sort=True)["market_amount_top10_share"]
            .mean()
            .reset_index()
            .rename(columns={"market_amount_top10_share": "state_value"})
        )

        strategy_frames["baseline_plain_top1"][year] = build_daily_top1_strategy_frame(baseline_frame)
        strategy_frames["baseline_amplitude_low_cash"][year] = build_state_gated_top1_strategy_from_daily_state(
            prediction_frame=baseline_with_state,
            daily_state_frame=amp_daily_state,
            threshold=3.095808,
            bad_side="low",
            fallback="cash",
            fallback_top_k=3,
        )
        strategy_frames["style_neutral_amountshare_low_cash"][year] = build_state_gated_top1_strategy_from_daily_state(
            prediction_frame=style_with_state,
            daily_state_frame=amt_daily_state,
            threshold=0.742913,
            bad_side="low",
            fallback="cash",
            fallback_top_k=3,
        )

    yearly_frame, curve_frame, summary_frame = build_strategy_comparison_artifacts(strategy_frames)
    yearly_path = output_dir / f"{args.model}_{args.variant}_yearly.csv"
    curve_path = output_dir / f"{args.model}_{args.variant}_curve.csv"
    summary_path = output_dir / f"{args.model}_{args.variant}_summary.csv"
    png_path = output_dir / f"{args.model}_{args.variant}_curve.png"
    yearly_frame.to_csv(yearly_path, index=False)
    curve_frame.to_csv(curve_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    plot_cumulative_curves(curve_frame, png_path)
    meta = {
        "years": years,
        "model": args.model,
        "variant": args.variant,
        "baseline_amplitude_low_cash": {"threshold": 3.095808, "bad_side": "low"},
        "style_neutral_amountshare_low_cash": {"threshold": 0.742913, "bad_side": "low"},
    }
    (output_dir / f"{args.model}_{args.variant}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "yearly_path": str(yearly_path),
                "curve_path": str(curve_path),
                "summary_path": str(summary_path),
                "png_path": str(png_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
