import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.evaluate_ensembles import load_prediction_frame
from utils.market_selector_audit import build_selector_audit_frame, summarize_selector_audit, summarize_selector_audit_by_year


def build_parser():
    parser = argparse.ArgumentParser(description="Audit selector stability for tradable-only market top1 strategy")
    parser.add_argument("--left_name", type=str, required=True)
    parser.add_argument("--right_name", type=str, required=True)
    parser.add_argument("--left", action="append", default=[], help="Repeatable path to left model prediction csv")
    parser.add_argument("--right", action="append", default=[], help="Repeatable path to right model prediction csv")
    parser.add_argument("--output_dir", type=str, default="logs/market_p1_audit")
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
    summary = summarize_selector_audit(daily)
    by_year = summarize_selector_audit_by_year(daily)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_path = output_dir / "selector_daily_audit.csv"
    year_path = output_dir / "selector_year_audit.csv"
    summary_path = output_dir / "selector_summary.json"
    daily.to_csv(daily_path, index=False)
    by_year.to_csv(year_path, index=False)
    payload = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "summary": summary,
        "daily_csv": str(daily_path),
        "year_csv": str(year_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
