import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.prod_common import build_strategy_frame, load_live_config, prediction_csv_path, summarize_strategy
from scripts.market_daily.evaluate_ensembles import load_prediction_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Replay production strategies on stored model predictions")
    parser.add_argument("--config", type=str, default="configs/market_live_strategy.json")
    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default="2024-12-31")
    parser.add_argument("--strategy_names", type=str, default="")
    parser.add_argument("--fold_years", type=str, default="2015,2016,2017,2018,2019,2020,2021,2022,2023,2024")
    parser.add_argument("--output_dir", type=str, default="logs/market_live_prod/replay")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_live_config(args.config)
    strategy_names = [item for item in args.strategy_names.split(",") if item]
    if not strategy_names:
        strategy_names = [config["primary_strategy"], *config.get("backup_strategies", [])]
    fold_years = [int(item) for item in args.fold_years.split(",") if item]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = sorted({model for name in strategy_names for model in config["strategies"][name]["models"]})
    full_frames = {key: [] for key in model_keys}
    for fold_year in fold_years:
        for key in model_keys:
            pred_path = prediction_csv_path(config, key, fold_year)
            if pred_path.exists():
                frame = load_prediction_frame(pred_path)
                frame = frame[
                    (pd.to_datetime(frame["date"]) >= pd.Timestamp(args.start_date))
                    & (pd.to_datetime(frame["date"]) <= pd.Timestamp(args.end_date))
                ].copy()
                if not frame.empty:
                    full_frames[key].append(frame)
    full_frames = {key: pd.concat(parts, ignore_index=True) for key, parts in full_frames.items() if parts}

    summary_rows = []
    for strategy_name in strategy_names:
        strategy_cfg = config["strategies"][strategy_name]
        strategy_frame = build_strategy_frame(strategy_cfg, full_frames)
        top1_frame, live_metrics = summarize_strategy(strategy_frame, config["scenarios"])
        top1_frame.to_csv(output_dir / f"{strategy_name}_daily.csv", index=False)
        for scenario_name, metrics in live_metrics.items():
            summary_rows.append(
                {
                    "strategy": strategy_name,
                    "scenario": scenario_name,
                    **metrics,
                    "daily_csv": str(output_dir / f"{strategy_name}_daily.csv"),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["scenario", "mean_return", "sharpe"], ascending=[True, False, False])
    summary_path = output_dir / "replay_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "summary_csv": str(summary_path),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "strategies": strategy_names,
    }
    (output_dir / "replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
