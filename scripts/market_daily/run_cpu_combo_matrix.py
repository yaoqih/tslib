import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.evaluate_ensembles import (
    build_confidence_selector_frame,
    load_prediction_frame,
)
from utils.market_research import combine_prediction_frames, evaluate_prediction_frame


MODEL_PATTERNS = {
    "dlinear": "test_results/long_term_forecast_market_*_120_fsA_DLinear_*market_round1_fsA_0/top1_predictions.csv",
    "itransformer": "test_results/long_term_forecast_market_*_20_fsA_iTransformer_*market_round1_fsA_0/top1_predictions.csv",
    "patchtst": "test_results/long_term_forecast_market_*_60_fsA_PatchTST_*market_round1_fsA_0/top1_predictions.csv",
    "timemixer": "test_results/long_term_forecast_market_*_20_fsA_TimeMixer_*market_round1_fsA_0/top1_predictions.csv",
    "timesnet": "test_results/long_term_forecast_market_*_20_fsA_TimesNet_*market_round1_fsA_0/top1_predictions.csv",
    "tsmixer": "test_results/long_term_forecast_market_*_20_fsA_TSMixer_*market_round1_fsA_0/top1_predictions.csv",
    "wpmixer": "test_results/long_term_forecast_market_*_40_fsA_WPMixer_*market_fullroll_wpmixer_sl40_p16_lr3e4_0/top1_predictions.csv",
}

PAIR_METHODS = ("rank_mean", "zscore_mean", "mean")
SELECTOR_METHODS = ("top1_gap", "top1_gap_rank")


def compact_metrics(metrics):
    return {
        "num_days": metrics["num_days"],
        "mean_return": metrics["mean_return"],
        "cumulative_return": metrics["cumulative_return"],
        "sharpe": metrics["sharpe"],
        "ic": metrics["ic"],
        "rank_ic": metrics["rank_ic"],
    }


def discover_model_paths(root: Path, models):
    mapping = {}
    for model_name in models:
        pattern = MODEL_PATTERNS[model_name]
        year_map = {}
        for path in sorted(root.glob(pattern)):
            year = int(path.parent.name.split("_")[4])
            year_map[year] = str(path)
        mapping[model_name] = year_map
    return mapping


def load_combo_frames(model_to_paths, combo):
    common_years = None
    for model_name in combo:
        years = set(model_to_paths[model_name].keys())
        common_years = years if common_years is None else (common_years & years)
    common_years = sorted(common_years or [])
    yearly_frames = {}
    for year in common_years:
        per_model = {}
        for model_name in combo:
            per_model[model_name] = load_prediction_frame(model_to_paths[model_name][year])
        yearly_frames[year] = per_model
    return yearly_frames


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
        vectors.append(weights)
    return vectors


def evaluate_combo_task(task):
    combo = tuple(task["combo"])
    model_to_paths = task["model_to_paths"]
    yearly_frames = load_combo_frames(model_to_paths, combo)
    rows = []
    aggregate_frames = {}

    for year, per_model in sorted(yearly_frames.items()):
        frames = [per_model[name] for name in combo]

        for method in PAIR_METHODS:
            combined = combine_prediction_frames(frames, method=method)
            metrics = compact_metrics(evaluate_prediction_frame(combined))
            rows.append(
                {
                    "combo": "+".join(combo),
                    "size": len(combo),
                    "method": method,
                    "weight_tag": "equal",
                    "fold_year": year,
                    **metrics,
                }
            )
            aggregate_frames.setdefault((method, "equal"), []).append(combined)

        for weight_idx, weights in enumerate(dominant_weight_vectors(len(combo))):
            combined = combine_prediction_frames(frames, method="rank_mean", weights=weights)
            metrics = compact_metrics(evaluate_prediction_frame(combined))
            weight_tag = json.dumps(
                {"dominant": combo[weight_idx], "weights": [round(x, 4) for x in weights]},
                ensure_ascii=False,
            )
            rows.append(
                {
                    "combo": "+".join(combo),
                    "size": len(combo),
                    "method": "weighted_rank",
                    "weight_tag": weight_tag,
                    "fold_year": year,
                    **metrics,
                }
            )
            aggregate_frames.setdefault(("weighted_rank", weight_tag), []).append(combined)

        if len(combo) == 2:
            left = per_model[combo[0]]
            right = per_model[combo[1]]
            for selector_method in SELECTOR_METHODS:
                combined = build_confidence_selector_frame(left, right, method=selector_method)
                metrics = compact_metrics(evaluate_prediction_frame(combined))
                rows.append(
                    {
                        "combo": "+".join(combo),
                        "size": 2,
                        "method": selector_method,
                        "weight_tag": "selector",
                        "fold_year": year,
                        **metrics,
                    }
                )
                aggregate_frames.setdefault((selector_method, "selector"), []).append(combined)

    if not rows:
        return []

    aggregate_rows = []
    frame = pd.DataFrame(rows)
    for (method, weight_tag), frames in aggregate_frames.items():
        merged = pd.concat(frames, ignore_index=True)
        metrics = compact_metrics(evaluate_prediction_frame(merged))
        year_slice = frame[(frame["method"] == method) & (frame["weight_tag"] == weight_tag)]
        aggregate_rows.append(
            {
                "combo": "+".join(combo),
                "size": len(combo),
                "method": method,
                "weight_tag": weight_tag,
                "fold_year": -1,
                **metrics,
                "positive_years": int((year_slice["mean_return"] > 0).sum()),
            }
        )

    detailed = frame.copy()
    detailed["positive_years"] = np.nan
    return pd.concat([detailed, pd.DataFrame(aggregate_rows)], ignore_index=True).to_dict("records")


def build_tasks(model_to_paths, min_size, max_size):
    models = sorted(model_to_paths.keys())
    tasks = []
    for size in range(min_size, max_size + 1):
        for combo in itertools.combinations(models, size):
            common_years = None
            for model_name in combo:
                years = set(model_to_paths[model_name].keys())
                common_years = years if common_years is None else (common_years & years)
            if common_years and len(common_years) >= 8:
                tasks.append({"combo": combo, "model_to_paths": model_to_paths})
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run CPU-parallel ensemble matrix on tradable-only top1 predictions")
    parser.add_argument("--models", type=str, default="itransformer,patchtst,wpmixer,timemixer,timesnet,tsmixer,dlinear")
    parser.add_argument("--min_size", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=max(4, math.floor((os.cpu_count() or 8) * 0.8)))
    parser.add_argument("--output_csv", type=str, default="logs/day1_matrix/cpu_combo_matrix/all_combo_summary.csv")
    args = parser.parse_args()

    root = Path(".")
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    model_to_paths = discover_model_paths(root, model_names)
    tasks = build_tasks(model_to_paths, args.min_size, args.max_size)

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(evaluate_combo_task, task) for task in tasks]
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            rows.extend(future.result())
            done += 1
            print(
                json.dumps(
                    {"done_tasks": done, "total_tasks": total, "progress": round(done / total, 4)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["fold_year", "sharpe", "mean_return", "size", "combo", "method"],
        ascending=[True, False, False, True, True, True],
    ).reset_index(drop=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    aggregate = frame[frame["fold_year"] == -1].copy()
    aggregate = aggregate.sort_values(
        ["mean_return", "sharpe", "positive_years", "size"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    aggregate_path = output_path.with_name(output_path.stem + "_aggregate.csv")
    aggregate.to_csv(aggregate_path, index=False)

    print(json.dumps({"tasks": len(tasks), "output_csv": str(output_path), "aggregate_csv": str(aggregate_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
