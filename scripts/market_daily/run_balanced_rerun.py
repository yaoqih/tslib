#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYBIN = "/huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python"

MODEL_MATRIX = {
    "DLinear": {"d_model": 64, "d_ff": 128, "e_layers": 2, "n_heads": 4},
    "PatchTST": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "iTransformer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "TimeMixer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "TSMixer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "TimesNet": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
}

MODEL_EPOCHS = {
    "DLinear": 2,
    "PatchTST": 2,
    "iTransformer": 2,
    "TimeMixer": 3,
    "TimesNet": 2,
    "TSMixer": 1,
}


def build_command(job: dict) -> list[str]:
    model = job["model"]
    fold_year = job["fold_year"]
    feature_set = job.get("feature_set", "A")
    model_cfg = MODEL_MATRIX[model]
    epochs = MODEL_EPOCHS[model]
    seq_len = 20
    feature_count = 24
    command = [
        PYBIN,
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", f"market_{fold_year}_{seq_len}_fs{feature_set}",
        "--model", model,
        "--data", "market_daily",
        "--root_path", ".",
        "--data_path", "market_daily.parquet",
        "--features", "MS",
        "--target", "label",
        "--freq", "d",
        "--seq_len", str(seq_len),
        "--label_len", "0",
        "--pred_len", "1",
        "--enc_in", str(feature_count),
        "--dec_in", str(feature_count),
        "--c_out", str(feature_count),
        "--d_model", str(model_cfg["d_model"]),
        "--d_ff", str(model_cfg["d_ff"]),
        "--e_layers", str(model_cfg["e_layers"]),
        "--n_heads", str(model_cfg["n_heads"]),
        "--factor", "3",
        "--dropout", "0.1",
        "--learning_rate", "0.0003",
        "--train_epochs", str(epochs),
        "--patience", "2",
        "--batch_size", "4096",
        "--num_workers", "8",
        "--loss", "Huber",
        "--huber_delta", "1.0",
        "--gpu", str(job["gpu"]),
        "--checkpoints", "./checkpoints",
        "--market_fold_year", str(fold_year),
        "--market_feature_set", feature_set,
        "--market_cache_path", "./cache/market_daily_features.parquet",
        "--market_start_year", "2010",
        "--market_min_history", "120",
        "--market_min_avg_amount", "20000000.0",
        "--train_mode", "fixed_epoch",
        "--market_train_full_window",
        "--des", "market_round1_fsA",
    ]
    if model != "TimesNet":
        command.append("--use_amp")
    if model == "TimeMixer":
        command.extend([
            "--embed", "fixed",
            "--down_sampling_layers", "3",
            "--down_sampling_method", "avg",
            "--down_sampling_window", "2",
        ])
    if model == "TimesNet":
        command.extend(["--embed", "fixed"])
    if model in {"iTransformer", "PatchTST", "TSMixer"}:
        command.extend(["--embed", "timeF"])
    return command


def load_jobs(manifest_path: Path) -> list[dict]:
    jobs = json.loads(manifest_path.read_text())
    return jobs


def run_worker(gpu: int, jobs: list[dict], out_dir: Path) -> None:
    log_path = out_dir / f"gpu{gpu}_worker.log"
    status_path = out_dir / f"gpu{gpu}_status.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    state = {"gpu": gpu, "started_at": int(time.time()), "jobs": []}
    with log_path.open("a", buffering=1) as log_file:
        log_file.write(f"[launcher] gpu={gpu} jobs={len(jobs)} start={state['started_at']}\n")
        for idx, job in enumerate(jobs, 1):
            cmd = build_command(job)
            record = {
                "idx": idx,
                "model": job["model"],
                "fold_year": job["fold_year"],
                "gpu": gpu,
                "status": "running",
                "command": cmd,
                "started_at": int(time.time()),
            }
            state["jobs"].append(record)
            status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log_file.write(
                f"[start] {idx}/{len(jobs)} gpu={gpu} model={job['model']} fold={job['fold_year']}\n"
            )
            log_file.write(" ".join(cmd) + "\n")
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
            record["return_code"] = proc.returncode
            record["finished_at"] = int(time.time())
            record["status"] = "done" if proc.returncode == 0 else "failed"
            state["jobs"][-1] = record
            status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            if proc.returncode != 0:
                log_file.write(
                    f"[fail] gpu={gpu} model={job['model']} fold={job['fold_year']} rc={proc.returncode}\n"
                )
                raise SystemExit(proc.returncode)
        state["finished_at"] = int(time.time())
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        log_file.write(f"[launcher] gpu={gpu} done\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="logs/fold_rerun_balanced_4gpu_run/job_manifest.json",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/fold_rerun_balanced_4gpu_run",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(manifest_path)
    gpu_ids = [int(x) for x in args.gpus.split(",") if x]
    by_gpu = {gpu: [] for gpu in gpu_ids}
    for job in jobs:
        by_gpu[job["gpu"]].append(job)

    for gpu in gpu_ids:
        gpu_jobs = by_gpu[gpu]
        if not gpu_jobs:
            continue
        pid = os.fork()
        if pid == 0:
            try:
                run_worker(gpu, gpu_jobs, out_dir)
            except SystemExit as exc:
                os._exit(int(exc.code or 0))
            except Exception as exc:  # pragma: no cover
                err_path = out_dir / f"gpu{gpu}_worker.err"
                err_path.write_text(repr(exc) + "\n")
                os._exit(1)
            os._exit(0)

    while True:
        try:
            pid, _ = os.wait()
        except ChildProcessError:
            break
        if pid == 0:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
