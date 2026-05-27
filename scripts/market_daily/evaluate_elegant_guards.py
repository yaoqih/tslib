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
    build_state_gated_top1_strategy_frame,
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


def summarize_strategy(strategy_frame, tag):
    proxy = apply_live_trading_proxy(strategy_frame, buy_cost_bps=0.0, sell_cost_bps=0.0)
    summary = summarize_live_proxy(proxy)
    summary["strategy"] = tag
    return summary


def merge_features(prediction_frame, feature_frame, columns):
    feature_keep = feature_frame[["date", "code", *columns]].drop_duplicates(subset=["date", "code"])
    merged = prediction_frame.merge(feature_keep, on=["date", "code"], how="left")
    return merged


def build_baseline_and_residual_frames(prediction_frame, feature_frame, style_columns):
    residual_frame = residualize_prediction_scores(
        prediction_frame=prediction_frame,
        feature_frame=feature_frame,
        feature_columns=tuple(style_columns),
    ).copy()
    residual_frame["pred"] = residual_frame["pred_resid"].astype(float)
    return {
        "baseline": prediction_frame.copy(),
        "style_neutral": residual_frame[["date", "code", "pred", "true", *([col for col in residual_frame.columns if col == "tradable"])]].copy()
        if "tradable" in residual_frame.columns
        else residual_frame[["date", "code", "pred", "true"]].copy(),
    }


def evaluate_year(
    year,
    model,
    variant,
    cache_path,
    style_columns,
    state_columns,
    topk_list,
):
    prediction_path = find_prediction_path(year, model, variant)
    prediction_frame = pd.read_csv(prediction_path, parse_dates=["date"])
    feature_columns = sorted(set(style_columns) | set(state_columns))
    feature_frame = load_feature_frame(cache_path, year, columns=feature_columns)
    prediction_with_state = merge_features(prediction_frame, feature_frame, state_columns)
    frame_map = build_baseline_and_residual_frames(prediction_frame, feature_frame, style_columns)
    results = []

    for score_name, score_frame in frame_map.items():
        state_frame = merge_features(score_frame, feature_frame, state_columns)
        baseline_strategy = build_daily_top1_strategy_frame(score_frame)
        results.append(
            {
                "year": int(year),
                "model": model,
                "variant": variant,
                "score_name": score_name,
                "guard_name": "plain_top1",
                **summarize_strategy(baseline_strategy, f"{score_name}:plain_top1"),
            }
        )

        for state_column in state_columns:
            daily_state = (
                prediction_with_state.groupby("date", sort=True)[state_column]
                .mean()
                .reset_index()
                .rename(columns={state_column: "state_value"})
            )
            threshold = float(daily_state["state_value"].median())
            baseline_daily = baseline_strategy.rename(columns={"true": "top1_true"})[["date", "top1_true"]]
            daily_joined = daily_state.merge(baseline_daily, on="date", how="left")
            low_mean = float(daily_joined.loc[daily_joined["state_value"] <= threshold, "top1_true"].mean())
            high_mean = float(daily_joined.loc[daily_joined["state_value"] > threshold, "top1_true"].mean())
            bad_side = "high" if high_mean <= low_mean else "low"

            for fallback in ("cash", "topk_rollover"):
                for top_k in tuple(topk_list):
                    if fallback == "cash" and top_k != topk_list[0]:
                        continue
                    strategy = build_state_gated_top1_strategy_frame(
                        prediction_frame=state_frame,
                        state_column=state_column,
                        threshold=threshold,
                        bad_side=bad_side,
                        fallback=fallback,
                        fallback_top_k=top_k,
                    )
                    results.append(
                        {
                            "year": int(year),
                            "model": model,
                            "variant": variant,
                            "score_name": score_name,
                            "guard_name": f"{state_column}:{bad_side}:{fallback}:top{top_k}",
                            "state_column": state_column,
                            "bad_side": bad_side,
                            "threshold": threshold,
                            "fallback": fallback,
                            "fallback_top_k": int(top_k),
                            **summarize_strategy(
                                strategy,
                                f"{score_name}:{state_column}:{bad_side}:{fallback}:top{top_k}",
                            ),
                        }
                    )
    return pd.DataFrame(results), str(prediction_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate simple elegant guard rails on prediction files")
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--variant", default="single_head_csrank_topq_v1")
    parser.add_argument("--cache_path", default="cache/market_daily_features_full2010.parquet")
    parser.add_argument("--style_columns", default="log_amount,turnover_rate,amplitude,ret_20,vol_20")
    parser.add_argument("--state_columns", default="market_cc_std,market_amplitude_mean,market_amount_top10_share,market_cc_mean")
    parser.add_argument("--topk_list", default="3,5")
    parser.add_argument("--output_dir", default="logs/elegant_guards")
    args = parser.parse_args()

    style_columns = [item.strip() for item in args.style_columns.split(",") if item.strip()]
    state_columns = [item.strip() for item in args.state_columns.split(",") if item.strip()]
    topk_list = tuple(int(item) for item in args.topk_list.split(",") if item.strip())
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    meta_rows = []
    for year in args.year:
        frame, prediction_path = evaluate_year(
            year=year,
            model=args.model,
            variant=args.variant,
            cache_path=ROOT / args.cache_path,
            style_columns=style_columns,
            state_columns=state_columns,
            topk_list=topk_list,
        )
        frame.to_csv(output_dir / f"{args.model}_{args.variant}_{year}_guards.csv", index=False)
        best = frame.sort_values(["mean_return", "sharpe"], ascending=[False, False]).head(1).copy()
        best.to_csv(output_dir / f"{args.model}_{args.variant}_{year}_best.csv", index=False)
        all_rows.append(frame)
        meta_rows.append({"year": int(year), "prediction_path": prediction_path})

    merged = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    merged.to_csv(output_dir / f"{args.model}_{args.variant}_summary.csv", index=False)
    (output_dir / f"{args.model}_{args.variant}_meta.json").write_text(
        json.dumps(meta_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_path": str(output_dir / f"{args.model}_{args.variant}_summary.csv"),
                "rows": int(merged.shape[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
