import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize market round1 experiments")
    parser.add_argument("--manifest", type=str, default="./logs/market_round1/job_manifest.json")
    parser.add_argument("--output_csv", type=str, default="./logs/market_round1/round1_summary.csv")
    return parser


def parse_metrics(metrics_path):
    metrics = {}
    for line in metrics_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = ast.literal_eval(value)
        except Exception:
            metrics[key] = value
    return metrics


def get_job_des(job):
    if job.get("des"):
        return job["des"]
    command = job.get("command", [])
    if "--des" in command:
        index = command.index("--des")
        if index + 1 < len(command):
            return command[index + 1]
    return "market_round1"


def find_metrics_file(job):
    des = get_job_des(job)
    pattern = (
        f"test_results/long_term_forecast_market_{job['fold_year']}_{job['seq_len']}_*_"
        f"{job['model']}_market_daily_*_{des}_0/market_metrics.txt"
    )
    matches = sorted(Path(".").glob(pattern))
    return matches[-1] if matches else None


def main():
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    jobs = json.loads(manifest_path.read_text())

    rows = []
    for job in jobs:
        row = {
            "model": job["model"],
            "fold_year": job["fold_year"],
            "seq_len": job["seq_len"],
            "gpu": job["gpu"],
            "status": job.get("status", "unknown"),
            "return_code": job.get("return_code"),
            "des": get_job_des(job),
            "log_path": job.get("log_path", ""),
        }
        metrics_path = find_metrics_file(job)
        row["metrics_path"] = str(metrics_path) if metrics_path else ""
        if metrics_path and metrics_path.exists():
            metrics = parse_metrics(metrics_path)
            for key in ("num_days", "mean_return", "cumulative_return", "sharpe", "ic", "rank_ic"):
                row[key] = metrics.get(key)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["model", "fold_year", "seq_len"]).reset_index(drop=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
