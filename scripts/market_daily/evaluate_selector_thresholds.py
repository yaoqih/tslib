import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.evaluate_ensembles import load_prediction_frame
from utils.market_selector_audit import (
    build_selector_audit_frame,
    build_threshold_gated_audit_frame,
    summarize_threshold_gated_audit,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate threshold-gated selector variants")
    parser.add_argument("--left", action="append", default=[], help="Repeatable left prediction csv")
    parser.add_argument("--right", action="append", default=[], help="Repeatable right prediction csv")
    parser.add_argument("--fallback_sources", type=str, default="right,left")
    parser.add_argument("--quantiles", type=str, default="0.0,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--output_csv", type=str, default="logs/market_p1_selector_audit/threshold_grid.csv")
    return parser


def load_many(paths):
    frames = [load_prediction_frame(Path(path)) for path in paths]
    return pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def main():
    args = build_parser().parse_args()
    if len(args.left) != len(args.right):
        raise ValueError("--left and --right must have the same number of entries")

    left_frame = load_many(args.left)
    right_frame = load_many(args.right)
    daily = build_selector_audit_frame(left_frame, right_frame)
    quantiles = [float(item) for item in args.quantiles.split(",") if item]
    fallback_sources = [item for item in args.fallback_sources.split(",") if item]

    rows = []
    for quantile in quantiles:
        threshold = float(daily["confidence_edge"].abs().quantile(quantile))
        for fallback_source in fallback_sources:
            gated = build_threshold_gated_audit_frame(
                daily_audit_frame=daily,
                fallback_source=fallback_source,
                min_abs_edge=threshold,
            )
            row = {
                "quantile": quantile,
                "threshold": threshold,
                "fallback_source": fallback_source,
                **summarize_threshold_gated_audit(gated),
            }
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        ["mean_return", "sharpe", "fallback_rate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))
    print(json.dumps({"output_csv": str(output_path), "num_rows": int(frame.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
