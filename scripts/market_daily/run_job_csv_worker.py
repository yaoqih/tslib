import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_jobs(csv_path, visible_gpu, poll_seconds=30):
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)
    env["PYTHONUNBUFFERED"] = "1"

    failed = []
    for idx, row in enumerate(rows, start=1):
        command = json.loads(row["command"])
        if "--gpu" in command:
            gpu_index = command.index("--gpu")
            if gpu_index + 1 < len(command):
                command[gpu_index + 1] = "0"
        log_dir = csv_path.parent / f"gpu{visible_gpu}_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{row['year']}_{row['model']}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            rc = proc.wait()
        print(json.dumps({
            "event": "finished",
            "gpu": visible_gpu,
            "index": idx,
            "total": len(rows),
            "year": row["year"],
            "model": row["model"],
            "returncode": rc,
            "log_path": str(log_path),
        }, ensure_ascii=False), flush=True)
        if rc != 0:
            failed.append({
                "year": row["year"],
                "model": row["model"],
                "returncode": rc,
                "log_path": str(log_path),
            })
    if failed:
        print(json.dumps({
            "event": "worker_failed_jobs",
            "gpu": visible_gpu,
            "failed": failed,
        }, ensure_ascii=False), flush=True)
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="Run one GPU shard from a job CSV")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--visible_gpu", required=True)
    parser.add_argument("--poll_seconds", type=int, default=30)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_jobs(args.csv_path, args.visible_gpu, args.poll_seconds)
