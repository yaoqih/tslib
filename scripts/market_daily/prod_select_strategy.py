import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.prod_common import build_strategy_frame, load_live_config, prediction_csv_path
from scripts.market_daily.evaluate_ensembles import load_prediction_frame
from utils.market_live_proxy import apply_live_trading_proxy, build_daily_top1_strategy_frame, summarize_live_proxy


def parse_args():
    parser = argparse.ArgumentParser(description="Choose active live strategy with automatic switch rules")
    parser.add_argument("--config", type=str, default="configs/market_live_strategy.json")
    parser.add_argument("--as_of_date", type=str, required=True)
    parser.add_argument("--fold_year", type=int, default=0)
    parser.add_argument("--strategy_names", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def load_state(state_path):
    path = Path(state_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(state_path, payload):
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    config = load_live_config(args.config)
    fold_year = args.fold_year or pd.Timestamp(args.as_of_date).year
    strategy_names = [item for item in args.strategy_names.split(",") if item]
    if not strategy_names:
        strategy_names = [config["primary_strategy"], *config.get("backup_strategies", [])]

    model_keys = sorted({model for name in strategy_names for model in config["strategies"][name]["models"]})
    frames = {
        key: load_prediction_frame(prediction_csv_path(config, key, fold_year))
        for key in model_keys
    }
    as_of_ts = pd.Timestamp(args.as_of_date)
    rules = config["switch_rules"]
    scenario_name = rules["scenario"]
    lookback_start = as_of_ts - pd.Timedelta(days=int(rules["lookback_days"]) * 2)

    rows = []
    for strategy_name in strategy_names:
        strategy_frame = build_strategy_frame(config["strategies"][strategy_name], frames)
        strategy_frame = strategy_frame[pd.to_datetime(strategy_frame["date"]) <= as_of_ts].copy()
        strategy_frame = strategy_frame[pd.to_datetime(strategy_frame["date"]) >= lookback_start].copy()
        strategy_top1 = build_daily_top1_strategy_frame(strategy_frame)
        if len(strategy_top1) > int(rules["lookback_days"]):
            strategy_top1 = strategy_top1.tail(int(rules["lookback_days"])).copy()
        proxy_frame = apply_live_trading_proxy(
            strategy_top1,
            buy_cost_bps=config["scenarios"][scenario_name]["buy_cost_bps"],
            sell_cost_bps=config["scenarios"][scenario_name]["sell_cost_bps"],
        )
        metrics = summarize_live_proxy(proxy_frame)
        rows.append({"strategy": strategy_name, **metrics})

    ranking = pd.DataFrame(rows).sort_values(
        ["mean_return", "sharpe", "switch_count"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best = ranking.iloc[0].to_dict()
    state = load_state(rules["state_path"])
    current_strategy = state.get("active_strategy", config["primary_strategy"])
    current_since = state.get("active_since")
    current_row = ranking[ranking["strategy"] == current_strategy]
    chosen_strategy = current_strategy if not current_row.empty else best["strategy"]
    switch_reason = "keep_current"

    hold_ok = True
    if current_since:
        hold_days = (as_of_ts - pd.Timestamp(current_since)).days
        hold_ok = hold_days >= int(rules["min_hold_days"])

    if current_row.empty:
        chosen_strategy = best["strategy"]
        switch_reason = "current_missing"
    else:
        current_metrics = current_row.iloc[0].to_dict()
        better_return = best["mean_return"] - current_metrics["mean_return"]
        better_sharpe = best["sharpe"] - current_metrics["sharpe"]
        if (
            best["strategy"] != current_strategy
            and hold_ok
            and better_return >= float(rules["min_excess_mean_return"])
            and better_sharpe >= float(rules["min_excess_sharpe"])
        ):
            chosen_strategy = best["strategy"]
            switch_reason = "switch_to_better_recent_strategy"
        else:
            chosen_strategy = current_strategy

    if chosen_strategy != current_strategy:
        active_since = str(as_of_ts.date())
    else:
        active_since = current_since or str(as_of_ts.date())

    decision = {
        "as_of_date": str(as_of_ts.date()),
        "fold_year": fold_year,
        "scenario": scenario_name,
        "lookback_days": int(rules["lookback_days"]),
        "ranking": ranking.to_dict(orient="records"),
        "active_strategy": chosen_strategy,
        "previous_strategy": current_strategy,
        "active_since": active_since,
        "switch_reason": switch_reason,
    }
    save_state(rules["state_path"], decision)
    output_json = args.output_json or str(Path(config["logs_dir"]) / f"strategy_decision_{as_of_ts.date()}.json")
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    print(json.dumps({"output_json": str(output_path), "active_strategy": chosen_strategy}, ensure_ascii=False))


if __name__ == "__main__":
    main()
