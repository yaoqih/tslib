import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.evaluate_ensembles import build_confidence_selector_frame, load_prediction_frame
from scripts.market_daily.run_p1_research import DLINEAR_MAINLINE, TIMESNET_VARIANTS
from utils.market_live_proxy import apply_live_trading_proxy, build_daily_top1_strategy_frame, summarize_live_proxy
from utils.market_selector_audit import build_selector_audit_frame, build_threshold_gated_strategy_frame


def build_parser():
    parser = argparse.ArgumentParser(description="Validate P1 strategy candidates with live-trading proxy costs")
    parser.add_argument("--decision_json", type=str, default="logs/market_p1/p1_decision.json")
    parser.add_argument("--quantiles", type=str, default="0.8,0.9")
    parser.add_argument("--fallback_source", type=str, default="left")
    parser.add_argument("--selector_method", type=str, default="top1_gap")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="low:5:10,base:8:13,high:12:17",
        help="Comma-separated NAME:BUY_BPS:SELL_BPS",
    )
    parser.add_argument("--output_dir", type=str, default="logs/market_live_proxy")
    return parser


def parse_scenarios(raw_value):
    scenarios = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        name, buy_bps, sell_bps = item.split(":")
        scenarios.append(
            {
                "scenario": name,
                "buy_cost_bps": float(buy_bps),
                "sell_cost_bps": float(sell_bps),
            }
        )
    if not scenarios:
        raise ValueError("At least one scenario is required")
    return scenarios


def load_many(mapping, candidate_years):
    frames = []
    for fold_year in candidate_years:
        frame = load_prediction_frame(Path(mapping[fold_year]))
        frame["fold_year"] = fold_year
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def summarize_gross(strategy_frame):
    daily_returns = pd.Series(strategy_frame["true"], dtype=float)
    mean_return = float(daily_returns.mean()) if not daily_returns.empty else 0.0
    cumulative_return = float((1.0 + daily_returns).prod() - 1.0) if not daily_returns.empty else 0.0
    std_return = float(daily_returns.std(ddof=0)) if len(daily_returns) > 1 else 0.0
    sharpe = 0.0 if std_return == 0.0 else (mean_return / std_return) * math.sqrt(252)
    return {
        "num_days": int(daily_returns.shape[0]),
        "mean_return": mean_return,
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
    }


def summarize_live_proxy_by_year(proxy_frame):
    rows = []
    frame = proxy_frame.copy()
    frame["fold_year"] = pd.to_datetime(frame["date"]).dt.year
    for fold_year, group in frame.groupby("fold_year"):
        rows.append({"fold_year": int(fold_year), **summarize_live_proxy(group)})
    return pd.DataFrame(rows).sort_values("fold_year").reset_index(drop=True)


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision = json.loads(Path(args.decision_json).read_text())
    chosen_variant = decision["chosen_timesnet_variant"]
    candidate_years = sorted(int(year) for year in decision["candidate_years"])
    quantiles = [float(item) for item in args.quantiles.split(",") if item]
    scenarios = parse_scenarios(args.scenarios)

    left_frame = load_many(DLINEAR_MAINLINE, candidate_years)
    right_frame = load_many(TIMESNET_VARIANTS[chosen_variant], candidate_years)

    strategy_frames = {
        "dlinear_a01": build_daily_top1_strategy_frame(left_frame),
        chosen_variant: build_daily_top1_strategy_frame(right_frame),
        f"selector_{args.selector_method}": build_daily_top1_strategy_frame(
            build_confidence_selector_frame(left_frame, right_frame, method=args.selector_method)
        ),
    }

    audit_daily = build_selector_audit_frame(left_frame, right_frame)
    threshold_rows = []
    for quantile in quantiles:
        threshold = float(audit_daily["confidence_edge"].abs().quantile(quantile))
        strategy_name = f"gated_q{int(quantile * 100):02d}_{args.fallback_source}"
        strategy_frame = build_threshold_gated_strategy_frame(
            daily_audit_frame=audit_daily,
            fallback_source=args.fallback_source,
            min_abs_edge=threshold,
        )
        strategy_frames[strategy_name] = strategy_frame[["date", "code", "pred", "true"]].copy()
        threshold_rows.append(
            {
                "strategy": strategy_name,
                "quantile": quantile,
                "threshold": threshold,
                "fallback_source": args.fallback_source,
            }
        )
        strategy_frame.to_csv(output_dir / f"{strategy_name}_daily.csv", index=False)

    gross_rows = []
    proxy_rows = []
    proxy_year_rows = []
    for strategy_name, strategy_frame in strategy_frames.items():
        gross_rows.append({"strategy": strategy_name, **summarize_gross(strategy_frame)})
        strategy_frame.to_csv(output_dir / f"{strategy_name}_daily.csv", index=False)
        for scenario in scenarios:
            proxy_frame = apply_live_trading_proxy(
                strategy_frame,
                buy_cost_bps=scenario["buy_cost_bps"],
                sell_cost_bps=scenario["sell_cost_bps"],
            )
            proxy_frame.to_csv(output_dir / f"{strategy_name}_{scenario['scenario']}_proxy_daily.csv", index=False)
            proxy_rows.append({"strategy": strategy_name, **scenario, **summarize_live_proxy(proxy_frame)})
            by_year = summarize_live_proxy_by_year(proxy_frame)
            if not by_year.empty:
                by_year.insert(0, "strategy", strategy_name)
                by_year.insert(1, "scenario", scenario["scenario"])
                by_year.insert(2, "buy_cost_bps", scenario["buy_cost_bps"])
                by_year.insert(3, "sell_cost_bps", scenario["sell_cost_bps"])
                proxy_year_rows.append(by_year)

    gross_summary = pd.DataFrame(gross_rows).sort_values(["mean_return", "sharpe"], ascending=[False, False])
    proxy_summary = pd.DataFrame(proxy_rows).sort_values(
        ["scenario", "mean_return", "sharpe"], ascending=[True, False, False]
    )
    proxy_year_summary = (
        pd.concat(proxy_year_rows, ignore_index=True)
        if proxy_year_rows
        else pd.DataFrame(columns=["strategy", "scenario", "buy_cost_bps", "sell_cost_bps", "fold_year"])
    )
    threshold_frame = pd.DataFrame(threshold_rows)

    gross_summary.to_csv(output_dir / "gross_summary.csv", index=False)
    proxy_summary.to_csv(output_dir / "proxy_summary.csv", index=False)
    proxy_year_summary.to_csv(output_dir / "proxy_year_summary.csv", index=False)
    threshold_frame.to_csv(output_dir / "gated_thresholds.csv", index=False)

    payload = {
        "chosen_timesnet_variant": chosen_variant,
        "candidate_years": candidate_years,
        "selector_method": args.selector_method,
        "fallback_source": args.fallback_source,
        "quantiles": quantiles,
        "scenarios": scenarios,
        "gross_summary_csv": str(output_dir / "gross_summary.csv"),
        "proxy_summary_csv": str(output_dir / "proxy_summary.csv"),
        "proxy_year_summary_csv": str(output_dir / "proxy_year_summary.csv"),
        "gated_thresholds_csv": str(output_dir / "gated_thresholds.csv"),
    }
    (output_dir / "validation_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
