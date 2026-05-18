import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_research import combine_prediction_frames, evaluate_prediction_frame


def parse_candidate_arg(raw_value):
    if "=" not in raw_value:
        raise ValueError(f"Candidate spec must be NAME=PATH, got: {raw_value}")
    name, path = raw_value.split("=", 1)
    return name.strip(), Path(path.strip())


def load_prediction_frame(pred_path):
    usecols = ["date", "code", "pred", "true"]
    sample = pd.read_csv(pred_path, nrows=1)
    if "tradable" in sample.columns:
        usecols.append("tradable")
    frame = pd.read_csv(pred_path, usecols=usecols)
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def load_named_candidates(candidate_args):
    candidates = []
    for raw_value in candidate_args:
        name, pred_path = parse_candidate_arg(raw_value)
        frame = load_prediction_frame(pred_path)
        fold_years = sorted(pd.to_datetime(frame["date"]).dt.year.unique().tolist())
        candidates.append(
            {
                "name": name,
                "pred_path": str(pred_path),
                "frame": frame,
                "fold_years": fold_years,
            }
        )
    return candidates


def split_candidates_by_year(candidates):
    yearly = {}
    for item in candidates:
        frame = item["frame"].copy()
        frame["fold_year"] = pd.to_datetime(frame["date"]).dt.year
        for fold_year, group in frame.groupby("fold_year"):
            payload = {
                "name": item["name"],
                "fold_year": int(fold_year),
                "pred_path": item["pred_path"],
                "frame": group.drop(columns=["fold_year"]).sort_values(["date", "code"]).reset_index(drop=True),
            }
            yearly.setdefault(int(fold_year), []).append(payload)
    return yearly


def daily_top1_gap(frame, pred_column):
    ranked = frame.sort_values(["date", pred_column], ascending=[True, False]).copy()
    ranked["row_id"] = ranked.groupby("date").cumcount()
    top2 = ranked[ranked["row_id"] < 2][["date", pred_column, "row_id"]]
    pivot = top2.pivot(index="date", columns="row_id", values=pred_column)
    if 0 not in pivot.columns:
        raise ValueError("Missing top1 prediction when computing confidence gap")
    top1 = pivot[0]
    top2 = pivot[1] if 1 in pivot.columns else pd.Series(0.0, index=pivot.index)
    return (top1 - top2).fillna(0.0)


def build_confidence_selector_frame(first_frame, second_frame, method="top1_gap"):
    merged = first_frame.rename(columns={"pred": "pred_left"}).merge(
        second_frame.rename(columns={"pred": "pred_right"}),
        on=["date", "code", "true"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    if "tradable_left" in merged.columns and "tradable_right" in merged.columns:
        merged["tradable"] = merged["tradable_left"] & merged["tradable_right"]
    elif "tradable_left" in merged.columns:
        merged["tradable"] = merged["tradable_left"]
    elif "tradable_right" in merged.columns:
        merged["tradable"] = merged["tradable_right"]

    if method == "top1_gap":
        left_conf = daily_top1_gap(merged[["date", "code", "pred_left"]], "pred_left")
        right_conf = daily_top1_gap(merged[["date", "code", "pred_right"]], "pred_right")
    elif method == "top1_gap_rank":
        ranked = merged.copy()
        ranked["pred_left_rank"] = ranked.groupby("date")["pred_left"].rank(method="average", pct=True)
        ranked["pred_right_rank"] = ranked.groupby("date")["pred_right"].rank(method="average", pct=True)
        left_conf = daily_top1_gap(ranked[["date", "code", "pred_left_rank"]], "pred_left_rank")
        right_conf = daily_top1_gap(ranked[["date", "code", "pred_right_rank"]], "pred_right_rank")
    else:
        raise ValueError(f"Unsupported selector method: {method}")

    selector = pd.DataFrame({"date": left_conf.index, "left_conf": left_conf.values, "right_conf": right_conf.values})
    selector["use_left"] = selector["left_conf"] >= selector["right_conf"]
    merged = merged.merge(selector[["date", "use_left"]], on="date", how="left")
    merged["pred"] = np.where(merged["use_left"], merged["pred_left"], merged["pred_right"])
    columns = ["date", "code", "pred", "true"]
    if "tradable" in merged.columns:
        columns.append("tradable")
    return merged[columns].sort_values(["date", "code"]).reset_index(drop=True)


def evaluate_named_frame(name, fold_year, frame, kind, method, members):
    metrics = evaluate_prediction_frame(frame)
    return {
        "kind": kind,
        "name": name,
        "fold_year": int(fold_year),
        "members": members,
        "method": method,
        **metrics,
    }


def evaluate_single_candidates(yearly_candidates):
    rows = []
    aggregate_frames = {}
    for fold_year, items in sorted(yearly_candidates.items()):
        for item in items:
            rows.append(
                evaluate_named_frame(
                    name=f"{fold_year}_{item['name']}",
                    fold_year=fold_year,
                    frame=item["frame"],
                    kind="single",
                    method="single",
                    members=item["name"],
                )
            )
            aggregate_frames.setdefault(item["name"], []).append(item["frame"])

    for name, frames in sorted(aggregate_frames.items()):
        merged = pd.concat(frames, ignore_index=True)
        rows.append(
            evaluate_named_frame(
                name=f"aggregate_{name}",
                fold_year=-1,
                frame=merged,
                kind="single_aggregate",
                method="single",
                members=name,
            )
        )
    return rows


def evaluate_pair_candidates(yearly_candidates, methods):
    rows = []
    aggregate_frames = {}
    for fold_year, items in sorted(yearly_candidates.items()):
        if len(items) < 2:
            continue
        for idx in range(len(items)):
            for jdx in range(idx + 1, len(items)):
                first = items[idx]
                second = items[jdx]
                pair_name = f"{first['name']}+{second['name']}"
                for method in methods:
                    frame = combine_prediction_frames([first["frame"], second["frame"]], method=method)
                    rows.append(
                        evaluate_named_frame(
                            name=f"{fold_year}_{method}_{pair_name}",
                            fold_year=fold_year,
                            frame=frame,
                            kind="pair",
                            method=method,
                            members=pair_name,
                        )
                    )
                    aggregate_frames.setdefault((method, pair_name), []).append(frame)

    for (method, pair_name), frames in sorted(aggregate_frames.items()):
        merged = pd.concat(frames, ignore_index=True)
        rows.append(
            evaluate_named_frame(
                name=f"aggregate_{method}_{pair_name}",
                fold_year=-1,
                frame=merged,
                kind="pair_aggregate",
                method=method,
                members=pair_name,
            )
        )
    return rows


def evaluate_selector_candidates(yearly_candidates, selector_methods):
    rows = []
    aggregate_frames = {}
    for fold_year, items in sorted(yearly_candidates.items()):
        if len(items) != 2:
            continue
        first, second = items
        pair_name = f"{first['name']}+{second['name']}"
        for method in selector_methods:
            frame = build_confidence_selector_frame(first["frame"], second["frame"], method=method)
            rows.append(
                evaluate_named_frame(
                    name=f"{fold_year}_{method}_{pair_name}",
                    fold_year=fold_year,
                    frame=frame,
                    kind="selector",
                    method=method,
                    members=pair_name,
                )
            )
            aggregate_frames.setdefault((method, pair_name), []).append(frame)

    for (method, pair_name), frames in sorted(aggregate_frames.items()):
        merged = pd.concat(frames, ignore_index=True)
        rows.append(
            evaluate_named_frame(
                name=f"aggregate_{method}_{pair_name}",
                fold_year=-1,
                frame=merged,
                kind="selector_aggregate",
                method=method,
                members=pair_name,
            )
        )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate tradable-only market ensemble and selector candidates")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec in NAME=/path/to/top1_predictions.csv format; repeatable",
    )
    parser.add_argument("--pair_methods", type=str, default="rank_mean,zscore_mean,mean")
    parser.add_argument("--selector_methods", type=str, default="top1_gap,top1_gap_rank")
    parser.add_argument("--output_csv", type=str, default="logs/market_p1/ensemble_summary.csv")
    return parser


def main():
    args = build_parser().parse_args()
    if len(args.candidate) < 2:
        raise ValueError("At least two --candidate entries are required")

    candidates = load_named_candidates(args.candidate)
    yearly_candidates = split_candidates_by_year(candidates)
    pair_methods = [item for item in args.pair_methods.split(",") if item]
    selector_methods = [item for item in args.selector_methods.split(",") if item]

    rows = []
    rows.extend(evaluate_single_candidates(yearly_candidates))
    rows.extend(evaluate_pair_candidates(yearly_candidates, pair_methods))
    rows.extend(evaluate_selector_candidates(yearly_candidates, selector_methods))

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["kind", "fold_year", "sharpe", "mean_return"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))
    print(json.dumps({"output_csv": str(output_path), "num_rows": int(frame.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
