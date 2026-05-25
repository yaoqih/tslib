#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_research import build_head_candidate_diagnostics


def find_prediction_path(year, model, variant):
    pattern = f"long_term_forecast_market_{year}_{variant}_{model}_*/top1_predictions.csv"
    matches = sorted((ROOT / "test_results").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No prediction file found for year={year}, model={model}, variant={variant}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Build daily head-candidate diagnostics from prediction files")
    parser.add_argument("--year", type=int, action="append", required=True, help="Repeatable target year")
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--variant", default="stage2topheavy_topk_csrank_topq_v2")
    parser.add_argument("--output_dir", default="logs/head_candidate_diagnostics")
    parser.add_argument("--pred_topk_list", default="10,20,50")
    parser.add_argument("--true_topk_list", default="10,20,50")
    args = parser.parse_args()

    pred_topk_list = tuple(int(item) for item in args.pred_topk_list.split(",") if item.strip())
    true_topk_list = tuple(int(item) for item in args.true_topk_list.split(",") if item.strip())

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for year in args.year:
        prediction_path = find_prediction_path(year, args.model, args.variant)
        frame = pd.read_csv(prediction_path, parse_dates=["date"])
        diagnostics = build_head_candidate_diagnostics(
            frame,
            pred_topk_list=pred_topk_list,
            true_topk_list=true_topk_list,
        )

        daily_path = output_dir / f"{args.model}_{year}_{args.variant}_daily.csv"
        summary_path = output_dir / f"{args.model}_{year}_{args.variant}_summary.json"
        pd.DataFrame(diagnostics["daily"]).to_csv(daily_path, index=False)
        summary_path.write_text(json.dumps(diagnostics["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

        row = {"year": year, "model": args.model, "variant": args.variant, "prediction_path": str(prediction_path)}
        row.update(diagnostics["summary"])
        summary_rows.append(row)
        print(json.dumps({"year": year, "daily_path": str(daily_path), "summary_path": str(summary_path)}, ensure_ascii=False))

    summary_csv = output_dir / f"{args.model}_{args.variant}_summary_table.csv"
    pd.DataFrame(summary_rows).sort_values("year").to_csv(summary_csv, index=False)
    print(json.dumps({"summary_table": str(summary_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
