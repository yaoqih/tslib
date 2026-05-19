import argparse
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.evaluate_ensembles import (  # noqa: E402
    build_confidence_selector_frame,
    load_prediction_frame,
)
from utils.market_live_proxy import (  # noqa: E402
    apply_live_trading_proxy,
    build_daily_top1_strategy_frame,
    summarize_live_proxy,
)
from utils.market_research import combine_prediction_frames, evaluate_prediction_frame  # noqa: E402
from utils.market_selector_audit import (  # noqa: E402
    build_selector_audit_frame,
    build_threshold_gated_strategy_frame,
)


MODEL_PATTERNS = {
    "dlinear": "test_results/long_term_forecast_market_*_20_fsA_DLinear_*market_round1_fsA_0/top1_predictions.csv",
    "itransformer": "test_results/long_term_forecast_market_*_20_fsA_iTransformer_*market_round1_fsA_0/top1_predictions.csv",
    "patchtst": "test_results/long_term_forecast_market_*_20_fsA_PatchTST_*market_round1_fsA_0/top1_predictions.csv",
    "timemixer": "test_results/long_term_forecast_market_*_20_fsA_TimeMixer_*market_round1_fsA_0/top1_predictions.csv",
    "timesnet": "test_results/long_term_forecast_market_*_20_fsA_TimesNet_*market_round1_fsA_0/top1_predictions.csv",
    "tsmixer": "test_results/long_term_forecast_market_*_20_fsA_TSMixer_*market_round1_fsA_0/top1_predictions.csv",
    "wpmixer": "test_results/long_term_forecast_market_*_40_fsA_WPMixer_*market_fullroll_wpmixer_sl40_p16_lr3e4_0/top1_predictions.csv",
}

COMBINE_METHODS = ("rank_mean", "zscore_mean", "mean")
PAIR_SELECTOR_METHODS = ("top1_gap",)
PAIR_GATED_QUANTILES = (0.5,)
SCENARIOS = {
    "low": {"buy_cost_bps": 5.0, "sell_cost_bps": 10.0},
    "base": {"buy_cost_bps": 8.0, "sell_cost_bps": 13.0},
    "high": {"buy_cost_bps": 12.0, "sell_cost_bps": 17.0},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unified-version live-ready combination matrix")
    parser.add_argument("--min_size", type=int, default=1)
    parser.add_argument("--max_size", type=int, default=4)
    parser.add_argument(
        "--models",
        type=str,
        default="tsmixer,timesnet,timemixer,itransformer,patchtst,dlinear",
    )
    parser.add_argument("--max_workers", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="logs/unified_live_matrix")
    return parser.parse_args()


def discover_model_paths(root: Path, models):
    result = {}
    for model_name in models:
        year_map = {}
        for path in sorted(root.glob(MODEL_PATTERNS[model_name])):
            year = int(path.parent.name.split("_")[4])
            year_map[year] = str(path)
        result[model_name] = year_map
    return result


def common_years(model_to_paths, combo):
    years = None
    for name in combo:
        current = set(model_to_paths[name].keys())
        years = current if years is None else (years & current)
    return sorted(years or [])


def load_combo_frames(model_to_paths, combo, years):
    yearly = {}
    for year in years:
        yearly[year] = {name: load_prediction_frame(model_to_paths[name][year]) for name in combo}
    return yearly


def summarize_gross_from_strategy(strategy_frame):
    daily = strategy_frame["true"].astype(float)
    mean_return = float(daily.mean()) if not daily.empty else 0.0
    cumulative_return = float((1.0 + daily).prod() - 1.0) if not daily.empty else 0.0
    std_return = float(daily.std(ddof=0)) if len(daily) > 1 else 0.0
    sharpe = 0.0 if std_return == 0.0 else (mean_return / std_return) * math.sqrt(252)
    return {
        "num_days": int(daily.shape[0]),
        "gross_mean_return": mean_return,
        "gross_cumulative_return": cumulative_return,
        "gross_sharpe": sharpe,
    }


def dominant_weight_vectors(size):
    if size == 2:
        base = 0.6
    elif size == 3:
        base = 0.5
    else:
        base = 0.4
    rest = (1.0 - base) / (size - 1)
    vectors = []
    for idx in range(size):
        weights = [rest] * size
        weights[idx] = base
        vectors.append((idx, weights))
    return vectors


def build_pair_gated_variants(left_name, right_name, left_frame, right_frame):
    daily_audit = build_selector_audit_frame(left_frame, right_frame)
    rows = []
    for quantile in PAIR_GATED_QUANTILES:
        threshold = float(daily_audit["confidence_edge"].abs().quantile(quantile))
        for fallback_source in ("left", "right"):
            strategy_frame = build_threshold_gated_strategy_frame(
                daily_audit_frame=daily_audit,
                fallback_source=fallback_source,
                min_abs_edge=threshold,
            )
            rows.append(
                (
                    f"gated_q{int(quantile * 100):02d}_{fallback_source}",
                    strategy_frame[["date", "code", "pred", "true"]].copy(),
                )
            )
    return rows


def evaluate_strategy_rows(strategy_name, combo_name, size, variant_kind, variant_method, strategy_frame):
    strategy_frame = strategy_frame.sort_values("date").reset_index(drop=True)
    gross = summarize_gross_from_strategy(strategy_frame)
    rows = []
    for scenario_name, scenario in SCENARIOS.items():
        proxy_frame = apply_live_trading_proxy(
            strategy_frame,
            buy_cost_bps=scenario["buy_cost_bps"],
            sell_cost_bps=scenario["sell_cost_bps"],
        )
        live = summarize_live_proxy(proxy_frame)
        rows.append(
            {
                "strategy": strategy_name,
                "combo": combo_name,
                "size": size,
                "variant_kind": variant_kind,
                "variant_method": variant_method,
                "scenario": scenario_name,
                **gross,
                **live,
            }
        )
    return rows


def evaluate_single_or_combo(combo, yearly_frames):
    combo_name = "+".join(combo)
    size = len(combo)
    aggregate_prediction_frames = {}

    for year in sorted(yearly_frames):
        per_model = yearly_frames[year]
        frames = [per_model[name] for name in combo]

        if size == 1:
            aggregate_prediction_frames.setdefault(("single", combo[0]), []).append(frames[0])
            continue

        for method in COMBINE_METHODS:
            combined = combine_prediction_frames(frames, method=method)
            aggregate_prediction_frames.setdefault(("combo", method), []).append(combined)

        for weight_idx, weights in dominant_weight_vectors(size):
            combined = combine_prediction_frames(frames, method="rank_mean", weights=weights)
            tag = json.dumps(
                {"dominant": combo[weight_idx], "weights": [round(x, 4) for x in weights]},
                ensure_ascii=False,
            )
            aggregate_prediction_frames.setdefault(("weighted_rank", tag), []).append(combined)

        if size == 2:
            left_name, right_name = combo
            left_frame, right_frame = frames
            for selector_method in PAIR_SELECTOR_METHODS:
                combined = build_confidence_selector_frame(left_frame, right_frame, method=selector_method)
                aggregate_prediction_frames.setdefault(("selector", selector_method), []).append(combined)
            for gated_name, gated_frame in build_pair_gated_variants(left_name, right_name, left_frame, right_frame):
                aggregate_prediction_frames.setdefault(("gated", gated_name), []).append(gated_frame)

    rows = []
    for (variant_kind, variant_method), frames in sorted(aggregate_prediction_frames.items()):
        merged = pd.concat(frames, ignore_index=True)
        if variant_kind == "single":
            strategy_name = combo[0]
        else:
            strategy_name = f"{combo_name}::{variant_kind}::{variant_method}"
        strategy_frame = build_daily_top1_strategy_frame(merged)
        rows.extend(
            evaluate_strategy_rows(
                strategy_name=strategy_name,
                combo_name=combo_name,
                size=size,
                variant_kind=variant_kind,
                variant_method=variant_method,
                strategy_frame=strategy_frame,
            )
        )
    return rows


def evaluate_combo_task(task):
    combo = tuple(task["combo"])
    years = task["years"]
    model_to_paths = task["model_to_paths"]
    yearly_frames = load_combo_frames(model_to_paths, combo, years)
    return evaluate_single_or_combo(combo, yearly_frames)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    model_to_paths = discover_model_paths(Path("."), models)
    manifest = {}
    for name, year_map in model_to_paths.items():
        manifest[name] = {
            "years": sorted(year_map.keys()),
            "paths": year_map,
        }

    tasks = []
    for size in range(args.min_size, args.max_size + 1):
        for combo in itertools.combinations(models, size):
            years = common_years(model_to_paths, combo)
            if len(years) < 10:
                continue
            tasks.append({"combo": combo, "years": years, "model_to_paths": model_to_paths})

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(evaluate_combo_task, task) for task in tasks]
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            rows.extend(future.result())
            done += 1
            print(json.dumps({"done_tasks": done, "total_tasks": total}, ensure_ascii=False), flush=True)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["scenario", "mean_return", "sharpe", "gross_mean_return", "gross_sharpe"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    output_csv = output_dir / "unified_live_matrix.csv"
    summary.to_csv(output_csv, index=False)

    base_rank = summary[summary["scenario"] == "base"].copy()
    base_rank = base_rank.sort_values(
        ["mean_return", "sharpe", "gross_mean_return", "switch_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    base_csv = output_dir / "unified_live_matrix_base_rank.csv"
    base_rank.to_csv(base_csv, index=False)

    gross_rank = summary[summary["scenario"] == "base"].copy()
    gross_rank = gross_rank.sort_values(
        ["gross_mean_return", "gross_sharpe", "mean_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    gross_csv = output_dir / "unified_live_matrix_gross_rank.csv"
    gross_rank.to_csv(gross_csv, index=False)

    manifest_path = output_dir / "unified_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "base_rank_csv": str(base_csv),
                "gross_rank_csv": str(gross_csv),
                "manifest_json": str(manifest_path),
                "num_rows": int(summary.shape[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
