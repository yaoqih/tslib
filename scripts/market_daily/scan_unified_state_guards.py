#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

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


def build_score_frames(prediction_frame, feature_frame, style_columns):
    residual_frame = residualize_prediction_scores(
        prediction_frame=prediction_frame,
        feature_frame=feature_frame,
        feature_columns=tuple(style_columns),
    ).copy()
    residual_frame["pred"] = residual_frame["pred_resid"].astype(float)
    frames = {
        "baseline": prediction_frame.copy(),
        "style_neutral": residual_frame[["date", "code", "pred", "true", *([col for col in residual_frame.columns if col == "tradable"])]].copy()
        if "tradable" in residual_frame.columns
        else residual_frame[["date", "code", "pred", "true"]].copy(),
    }
    return frames


def summarize_strategy_frame(strategy_frame):
    proxy = apply_live_trading_proxy(strategy_frame, buy_cost_bps=0.0, sell_cost_bps=0.0)
    return summarize_live_proxy(proxy)


def collect_year_data(years, model, variant, cache_path, style_columns, state_columns):
    per_year = {}
    for year in years:
        prediction_path = find_prediction_path(year, model, variant)
        prediction_frame = pd.read_csv(prediction_path, parse_dates=["date"])
        feature_frame = load_feature_frame(cache_path, year, columns=sorted(set(style_columns) | set(state_columns)))
        per_year[int(year)] = {
            "prediction_path": str(prediction_path),
            "prediction_frame": prediction_frame,
            "feature_frame": feature_frame,
            "score_frames": build_score_frames(prediction_frame, feature_frame, style_columns),
        }
    return per_year


def build_daily_state_table(per_year, score_name, state_columns):
    rows = []
    for year, payload in per_year.items():
        score_frame = payload["score_frames"][score_name]
        state_frame = merge_features(score_frame, payload["feature_frame"], state_columns)
        daily_state = state_frame.groupby("date", sort=True).agg({column: "mean" for column in state_columns}).reset_index()
        daily_state["year"] = int(year)
        rows.append(daily_state)
    return pd.concat(rows, ignore_index=True)


def evaluate_rule(per_year, score_name, state_column, threshold, bad_side, fallback, fallback_top_k):
    year_rows = []
    for year, payload in per_year.items():
        score_frame = payload["score_frames"][score_name]
        state_frame = merge_features(score_frame, payload["feature_frame"], [state_column])
        daily_state = (
            state_frame.groupby("date", sort=True)[state_column]
            .mean()
            .reset_index()
            .rename(columns={state_column: "state_value"})
        )
        strategy = build_state_gated_top1_strategy_from_daily_state(
            prediction_frame=state_frame,
            daily_state_frame=daily_state,
            threshold=threshold,
            bad_side=bad_side,
            fallback=fallback,
            fallback_top_k=fallback_top_k,
        )
        summary = summarize_strategy_frame(strategy)
        summary["year"] = int(year)
        summary["gated_off_rate"] = float(strategy["state_gated_off"].mean()) if not strategy.empty else 0.0
        year_rows.append(summary)

    year_frame = pd.DataFrame(year_rows).sort_values("year").reset_index(drop=True)
    return {
        "year_frame": year_frame,
        "summary": {
            "score_name": score_name,
            "state_column": state_column,
            "threshold": float(threshold),
            "bad_side": bad_side,
            "fallback": fallback,
            "fallback_top_k": int(fallback_top_k),
            "mean_of_year_means": float(year_frame["mean_return"].mean()),
            "median_of_year_means": float(year_frame["mean_return"].median()),
            "mean_sharpe": float(year_frame["sharpe"].mean()),
            "negative_years": int((year_frame["mean_return"] < 0).sum()),
            "positive_years": int((year_frame["mean_return"] > 0).sum()),
            "worst_year_mean": float(year_frame["mean_return"].min()),
            "best_year_mean": float(year_frame["mean_return"].max()),
            "avg_gated_off_rate": float(year_frame["gated_off_rate"].mean()),
            "year_2025_mean": float(year_frame.loc[year_frame["year"] == 2025, "mean_return"].iloc[0]) if (year_frame["year"] == 2025).any() else 0.0,
            "year_2024_mean": float(year_frame.loc[year_frame["year"] == 2024, "mean_return"].iloc[0]) if (year_frame["year"] == 2024).any() else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Scan unified state-guard rules across multiple years")
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--variant", default="single_head_csrank_topq_v1")
    parser.add_argument("--cache_path", default="cache/market_daily_features_full2010.parquet")
    parser.add_argument("--style_columns", default="log_amount,turnover_rate,amplitude,ret_20,vol_20")
    parser.add_argument("--state_columns", default="market_cc_mean,market_cc_std,market_amplitude_mean,market_amount_top10_share")
    parser.add_argument("--quantiles", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--fallbacks", default="cash")
    parser.add_argument("--topk_list", default="3")
    parser.add_argument("--output_dir", default="logs/unified_state_guards")
    args = parser.parse_args()

    years = [int(year) for year in args.year]
    style_columns = [item.strip() for item in args.style_columns.split(",") if item.strip()]
    state_columns = [item.strip() for item in args.state_columns.split(",") if item.strip()]
    quantiles = [float(item) for item in args.quantiles.split(",") if item.strip()]
    fallbacks = [item.strip() for item in args.fallbacks.split(",") if item.strip()]
    topk_list = [int(item) for item in args.topk_list.split(",") if item.strip()]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_year = collect_year_data(
        years=years,
        model=args.model,
        variant=args.variant,
        cache_path=ROOT / args.cache_path,
        style_columns=style_columns,
        state_columns=state_columns,
    )

    all_summary_rows = []
    detailed_frames = []
    for score_name in ("baseline", "style_neutral"):
        daily_state_table = build_daily_state_table(per_year, score_name, state_columns)
        baseline_rows = []
        for year, payload in per_year.items():
            strategy = build_daily_top1_strategy_frame(payload["score_frames"][score_name])
            summary = summarize_strategy_frame(strategy)
            summary.update(
                {
                    "year": int(year),
                    "score_name": score_name,
                    "state_column": "none",
                    "threshold": float("nan"),
                    "bad_side": "none",
                    "fallback": "plain_top1",
                    "fallback_top_k": 1,
                    "gated_off_rate": 0.0,
                }
            )
            baseline_rows.append(summary)
        baseline_frame = pd.DataFrame(baseline_rows).sort_values("year").reset_index(drop=True)
        detailed_frames.append(baseline_frame)
        all_summary_rows.append(
            {
                "score_name": score_name,
                "state_column": "none",
                "threshold": float("nan"),
                "bad_side": "none",
                "fallback": "plain_top1",
                "fallback_top_k": 1,
                "mean_of_year_means": float(baseline_frame["mean_return"].mean()),
                "median_of_year_means": float(baseline_frame["mean_return"].median()),
                "mean_sharpe": float(baseline_frame["sharpe"].mean()),
                "negative_years": int((baseline_frame["mean_return"] < 0).sum()),
                "positive_years": int((baseline_frame["mean_return"] > 0).sum()),
                "worst_year_mean": float(baseline_frame["mean_return"].min()),
                "best_year_mean": float(baseline_frame["mean_return"].max()),
                "avg_gated_off_rate": 0.0,
                "year_2025_mean": float(baseline_frame.loc[baseline_frame["year"] == 2025, "mean_return"].iloc[0]) if (baseline_frame["year"] == 2025).any() else 0.0,
                "year_2024_mean": float(baseline_frame.loc[baseline_frame["year"] == 2024, "mean_return"].iloc[0]) if (baseline_frame["year"] == 2024).any() else 0.0,
            }
        )

        for state_column in state_columns:
            state_series = daily_state_table[state_column].dropna()
            for q in quantiles:
                threshold = float(state_series.quantile(q))
                for bad_side in ("high", "low"):
                    for fallback in fallbacks:
                        for top_k in topk_list:
                            if fallback == "cash" and top_k != topk_list[0]:
                                continue
                            result = evaluate_rule(
                                per_year=per_year,
                                score_name=score_name,
                                state_column=state_column,
                                threshold=threshold,
                                bad_side=bad_side,
                                fallback=fallback,
                                fallback_top_k=top_k,
                            )
                            detail = result["year_frame"].copy()
                            detail["score_name"] = score_name
                            detail["state_column"] = state_column
                            detail["threshold"] = threshold
                            detail["bad_side"] = bad_side
                            detail["fallback"] = fallback
                            detail["fallback_top_k"] = int(top_k)
                            detailed_frames.append(detail)
                            all_summary_rows.append(result["summary"])

    summary_frame = pd.DataFrame(all_summary_rows).sort_values(
        ["negative_years", "year_2025_mean", "mean_of_year_means", "mean_sharpe"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    detail_frame = pd.concat(detailed_frames, ignore_index=True)
    summary_path = output_dir / f"{args.model}_{args.variant}_unified_summary.csv"
    detail_path = output_dir / f"{args.model}_{args.variant}_unified_detail.csv"
    summary_frame.to_csv(summary_path, index=False)
    detail_frame.to_csv(detail_path, index=False)
    meta = {
        "years": years,
        "model": args.model,
        "variant": args.variant,
        "style_columns": style_columns,
        "state_columns": state_columns,
        "quantiles": quantiles,
        "fallbacks": fallbacks,
        "topk_list": topk_list,
    }
    (output_dir / f"{args.model}_{args.variant}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary_path": str(summary_path), "detail_path": str(detail_path), "rows": int(summary_frame.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
