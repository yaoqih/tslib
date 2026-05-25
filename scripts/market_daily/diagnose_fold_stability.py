import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_research import build_market_diagnostics  # noqa: E402


def parse_candidate_arg(raw_value):
    if "=" not in raw_value:
        raise ValueError(f"Candidate spec must be NAME=PATH, got: {raw_value}")
    name, path = raw_value.split("=", 1)
    return name.strip(), Path(path.strip())


def load_prediction_frame(pred_path):
    sample = pd.read_csv(pred_path, nrows=1)
    usecols = ["date", "code", "pred", "true"]
    optional = [
        "tradable",
        "market_cc_mean",
        "market_amount_top10_share",
    ]
    usecols.extend([col for col in optional if col in sample.columns])
    frame = pd.read_csv(pred_path, usecols=usecols, parse_dates=["date"])
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose fold stability for market prediction artifacts")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec in NAME=/path/to/top1_predictions.csv format; repeatable",
    )
    parser.add_argument("--output_dir", type=str, default="logs/diagnostics/fold_stability")
    parser.add_argument("--score_debias", type=str, default="none")
    parser.add_argument("--score_debias_strength", type=float, default=1.0)
    parser.add_argument("--topk_list", type=str, default="1,3,5,10,20")
    parser.add_argument("--score_bucket_count", type=int, default=10)
    parser.add_argument("--high_repeat_top_n", type=int, default=20)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.candidate:
        raise ValueError("At least one --candidate entry is required")

    topk_list = tuple(int(item.strip()) for item in args.topk_list.split(",") if item.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for raw_candidate in args.candidate:
        name, pred_path = parse_candidate_arg(raw_candidate)
        frame = load_prediction_frame(pred_path)
        diagnostics = build_market_diagnostics(
            frame,
            topk_list=topk_list,
            score_debias=args.score_debias,
            score_debias_strength=args.score_debias_strength,
            score_bucket_count=args.score_bucket_count,
            high_repeat_top_n=args.high_repeat_top_n,
        )

        summary = {"name": name, "prediction_path": str(pred_path), **diagnostics["summary"]}
        summary_rows.append(summary)

        safe_name = name.replace("/", "__").replace(" ", "_")
        (output_dir / f"{safe_name}_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, default=str),
            encoding="utf-8",
        )
        pd.DataFrame(diagnostics["score_buckets"]).to_csv(output_dir / f"{safe_name}_score_buckets.csv", index=False)
        pd.DataFrame(diagnostics["top_repeated_picks"]).to_csv(
            output_dir / f"{safe_name}_top_repeated_picks.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "slice_name": slice_name,
                    "median": payload["median"],
                    "low_num_days": payload["low"]["num_days"],
                    "low_top1_mean_return": payload["low"]["top1_mean_return"],
                    "high_num_days": payload["high"]["num_days"],
                    "high_top1_mean_return": payload["high"]["top1_mean_return"],
                }
                for slice_name, payload in diagnostics["market_slices"].items()
            ]
        ).to_csv(output_dir / f"{safe_name}_market_slices.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    print(output_dir)


if __name__ == "__main__":
    main()
