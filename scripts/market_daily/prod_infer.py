import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.prod_common import (
    build_strategy_frame,
    ensure_predictions,
    filter_to_date,
    latest_common_date,
    load_live_config,
    load_prediction_frame,
    prediction_csv_path,
    summarize_strategy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Daily live inference entry for production strategy")
    parser.add_argument("--config", type=str, default="configs/market_live_strategy.json")
    parser.add_argument("--as_of_date", type=str, required=True)
    parser.add_argument("--fold_year", type=int, default=0)
    parser.add_argument("--strategy_names", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--force_refresh", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="logs/market_live_prod/daily")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_live_config(args.config)
    fold_year = args.fold_year or pd.Timestamp(args.as_of_date).year
    strategy_names = [item for item in args.strategy_names.split(",") if item]
    if not strategy_names:
        strategy_names = [config["primary_strategy"], *config.get("backup_strategies", [])]

    model_keys = sorted({model for name in strategy_names for model in config["strategies"][name]["models"]})
    ensure_predictions(
        config=config,
        model_keys=model_keys,
        fold_year=fold_year,
        market_test_end=args.as_of_date,
        python_bin=args.python,
        gpu=args.gpu,
        force=args.force_refresh,
    )

    full_frames = {key: load_prediction_frame(prediction_csv_path(config, key, fold_year)) for key in model_keys}
    target_date = latest_common_date(full_frames, args.as_of_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for strategy_name in strategy_names:
        strategy_cfg = config["strategies"][strategy_name]
        strategy_frame = build_strategy_frame(strategy_cfg, full_frames)
        strategy_top1, live_metrics = summarize_strategy(strategy_frame, config["scenarios"])
        date_frame = filter_to_date(strategy_frame, target_date)
        ranked = date_frame.copy()
        if "tradable" in ranked.columns:
            ranked = ranked[ranked["tradable"]].copy()
        ranked = ranked.sort_values("pred", ascending=False).reset_index(drop=True)
        if ranked.empty:
            raise ValueError(f"No tradable candidates for strategy {strategy_name} on {target_date.date()}")
        top1 = ranked.iloc[0]
        ranked.head(20).to_csv(output_dir / f"{strategy_name}_{target_date.date()}_top20.csv", index=False)
        results.append(
            {
                "strategy": strategy_name,
                "trade_date": str(target_date.date()),
                "code": str(top1["code"]),
                "pred": float(top1["pred"]),
                "expected_return": float(top1["true"]) if "true" in top1 else None,
                "live_proxy": live_metrics,
                "top20_csv": str(output_dir / f"{strategy_name}_{target_date.date()}_top20.csv"),
            }
        )

    payload = {
        "as_of_date": args.as_of_date,
        "trade_date": str(target_date.date()),
        "fold_year": fold_year,
        "results": results,
    }
    output_path = output_dir / f"daily_signal_{target_date.date()}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"output_json": str(output_path), "trade_date": str(target_date.date())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
