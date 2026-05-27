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


def summarize_strategy(name, strategy_frame, buy_cost_bps, sell_cost_bps, extra=None):
    proxy = apply_live_trading_proxy(
        strategy_frame[["date", "code", "pred", "true"]].copy(),
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
    )
    row = {"strategy": name, **summarize_live_proxy(proxy)}
    if extra:
        row.update(extra)
    return row, proxy


def main():
    parser = argparse.ArgumentParser(description="Evaluate residualized ranking and simple state gating")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--cache_path", default="cache/market_daily_features_full2010.parquet")
    parser.add_argument(
        "--feature_columns",
        default="log_amount,turnover_rate,amplitude,ret_20,vol_20",
    )
    parser.add_argument("--state_column", default="market_cc_std")
    parser.add_argument("--state_quantile", type=float, default=0.8)
    parser.add_argument("--bad_side", default="high")
    parser.add_argument("--buy_cost_bps", type=float, default=0.0)
    parser.add_argument("--sell_cost_bps", type=float, default=0.0)
    parser.add_argument("--output_dir", default="logs/residual_gating_eval")
    args = parser.parse_args()

    feature_columns = [item.strip() for item in args.feature_columns.split(",") if item.strip()]
    needed_columns = sorted(set(feature_columns + [args.state_column]))

    prediction_path = find_prediction_path(args.year, args.model, args.variant)
    prediction_frame = pd.read_csv(prediction_path, parse_dates=["date"])
    feature_frame = load_feature_frame(ROOT / args.cache_path, args.year, needed_columns)

    merged = prediction_frame.merge(
        feature_frame[["date", "code", *needed_columns]],
        on=["date", "code"],
        how="left",
    )
    residualized = residualize_prediction_scores(
        prediction_frame=prediction_frame,
        feature_frame=feature_frame,
        feature_columns=tuple(feature_columns),
    )
    residualized["pred"] = residualized["pred_resid"]
    residualized = residualized.merge(
        feature_frame[["date", "code", args.state_column]],
        on=["date", "code"],
        how="left",
    )

    state_daily = residualized.groupby("date", sort=True)[args.state_column].mean()
    threshold = float(state_daily.quantile(args.state_quantile))

    baseline_strategy = build_daily_top1_strategy_frame(merged)
    residual_strategy = build_daily_top1_strategy_frame(residualized)
    gated_strategy = build_state_gated_top1_strategy_frame(
        prediction_frame=residualized,
        state_column=args.state_column,
        threshold=threshold,
        bad_side=args.bad_side,
        fallback="cash",
    )

    output_dir = ROOT / args.output_dir / f"{args.model}_{args.year}_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, strategy, extra in (
        ("baseline", baseline_strategy, {}),
        ("residualized", residual_strategy, {}),
        (
            "residualized_state_gated_cash",
            gated_strategy,
            {
                "state_column": args.state_column,
                "state_quantile": float(args.state_quantile),
                "state_threshold": threshold,
                "bad_side": args.bad_side,
                "gated_off_rate": float(gated_strategy["state_gated_off"].mean()),
            },
        ),
    ):
        strategy.to_csv(output_dir / f"{name}_daily.csv", index=False)
        row, proxy = summarize_strategy(
            name=name,
            strategy_frame=strategy,
            buy_cost_bps=args.buy_cost_bps,
            sell_cost_bps=args.sell_cost_bps,
            extra=extra,
        )
        proxy.to_csv(output_dir / f"{name}_proxy_daily.csv", index=False)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("strategy").reset_index(drop=True)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "year": int(args.year),
        "model": args.model,
        "variant": args.variant,
        "prediction_path": str(prediction_path),
        "cache_path": str(ROOT / args.cache_path),
        "feature_columns": feature_columns,
        "state_column": args.state_column,
        "state_quantile": float(args.state_quantile),
        "state_threshold": threshold,
        "summary_path": str(summary_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
