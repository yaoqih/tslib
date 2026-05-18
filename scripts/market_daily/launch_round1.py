import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_research import get_feature_columns


MODEL_MATRIX = {
    "DLinear": {"d_model": 64, "d_ff": 128, "e_layers": 2, "n_heads": 4},
    "PatchTST": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "iTransformer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "TimeMixer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "TSMixer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
    "TimesNet": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4},
}


def build_parser():
    parser = argparse.ArgumentParser(description="Launch round1 market experiments")
    parser.add_argument("--fold_years", type=str, default="2019,2021")
    parser.add_argument("--seq_lens", type=str, default="20,60,120")
    parser.add_argument("--feature_set", type=str, default="A")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--max_jobs_per_gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--models", type=str, default="DLinear,PatchTST,iTransformer,TimeMixer")
    parser.add_argument("--output_dir", type=str, default="./logs/market_round1")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--loss", type=str, default="Huber")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--market_aux_cls", action="store_true", default=False)
    parser.add_argument("--market_cls_weight", type=float, default=0.5)
    parser.add_argument("--market_rank_loss", action="store_true", default=False)
    parser.add_argument("--market_rank_weight", type=float, default=0.1)
    parser.add_argument("--market_rank_margin", type=float, default=0.0)
    parser.add_argument("--des", type=str, default="")
    parser.add_argument(
        "--launch_mode",
        type=str,
        default="queued",
        choices=["queued", "immediate"],
        help="queued waits for free GPU slots before starting the next job",
    )
    parser.add_argument("--poll_interval", type=int, default=30)
    return parser


def build_command(model, seq_len, fold_year, feature_set, epochs, batch_size, python_bin, loss, huber_delta,
                  market_aux_cls, market_cls_weight, market_rank_loss, market_rank_weight, market_rank_margin,
                  des_override):
    feature_count = len(get_feature_columns(feature_set))
    model_cfg = MODEL_MATRIX[model]
    model_id = f"market_{fold_year}_{seq_len}_fs{feature_set}"
    des = des_override or f"market_round1_fs{feature_set}"
    command = [
        python_bin,
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", model_id,
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
        "--batch_size", str(batch_size),
        "--num_workers", "8",
        "--loss", loss,
        "--huber_delta", str(huber_delta),
        "--gpu", "0",
        "--market_fold_year", str(fold_year),
        "--market_feature_set", feature_set,
        "--market_cache_path", "./cache/market_daily_features.parquet",
        "--des", des,
    ]
    if model != "TimesNet":
        command.append("--use_amp")
    if market_aux_cls:
        command.extend(
            [
                "--market_aux_cls",
                "--market_cls_weight", str(market_cls_weight),
            ]
        )
    if market_rank_loss:
        command.extend(
            [
                "--market_rank_loss",
                "--market_rank_weight", str(market_rank_weight),
                "--market_rank_margin", str(market_rank_margin),
            ]
        )
    if model == "TimeMixer":
        command.extend(
            [
                "--embed", "fixed",
                "--down_sampling_layers", "3",
                "--down_sampling_method", "avg",
                "--down_sampling_window", "2",
            ]
        )
    if model == "TimesNet":
        command.extend(["--embed", "fixed"])
    return command


def write_manifest(manifest_path, manifest):
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)


def launch_job(job, output_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    log_path = Path(output_dir) / f'{job["model"]}_fold{job["fold_year"]}_sl{job["seq_len"]}.log'
    job["log_path"] = str(log_path)

    print(f'[GPU {job["gpu"]}] {" ".join(job["command"])}')
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        job["command"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        cwd=str(ROOT),
    )
    job["status"] = "running"
    job["pid"] = process.pid
    job["started_at"] = int(time.time())
    return process, log_file


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    models = [item for item in args.models.split(",") if item]
    seq_lens = [int(item) for item in args.seq_lens.split(",") if item]
    fold_years = [int(item) for item in args.fold_years.split(",") if item]
    gpu_ids = [int(item) for item in args.gpus.split(",") if item]

    jobs = []
    for model in models:
        for fold_year in fold_years:
            for seq_len in seq_lens:
                jobs.append((model, seq_len, fold_year))

    manifest = []
    for index, job in enumerate(jobs):
        gpu_id = gpu_ids[(index // args.max_jobs_per_gpu) % len(gpu_ids)]
        model, seq_len, fold_year = job
        command = build_command(
            model=model,
            seq_len=seq_len,
            fold_year=fold_year,
            feature_set=args.feature_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            python_bin=args.python,
            loss=args.loss,
            huber_delta=args.huber_delta,
            market_aux_cls=args.market_aux_cls,
            market_cls_weight=args.market_cls_weight,
            market_rank_loss=args.market_rank_loss,
            market_rank_weight=args.market_rank_weight,
            market_rank_margin=args.market_rank_margin,
            des_override=args.des,
        )
        manifest.append(
            {
                "gpu": gpu_id,
                "model": model,
                "fold_year": fold_year,
                "seq_len": seq_len,
                "feature_set": args.feature_set,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "market_aux_cls": args.market_aux_cls,
                "market_cls_weight": args.market_cls_weight,
                "market_rank_loss": args.market_rank_loss,
                "market_rank_weight": args.market_rank_weight,
                "market_rank_margin": args.market_rank_margin,
                "des": args.des or f"market_round1_fs{args.feature_set}",
                "command": command,
                "status": "pending",
            }
        )

    manifest_path = Path(args.output_dir) / "job_manifest.json"
    write_manifest(manifest_path, manifest)

    if args.dry_run:
        for job in manifest:
            log_path = Path(args.output_dir) / f'{job["model"]}_fold{job["fold_year"]}_sl{job["seq_len"]}.log'
            job["log_path"] = str(log_path)
            print(f'[GPU {job["gpu"]}] {" ".join(job["command"])}')
        write_manifest(manifest_path, manifest)
        return

    if args.launch_mode == "immediate":
        launched = []
        try:
            for job in manifest:
                process, log_file = launch_job(job, args.output_dir)
                launched.append((process, log_file, job))
            write_manifest(manifest_path, manifest)
        finally:
            for _, log_file, _ in launched:
                log_file.close()
        return

    pending_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
    for job in manifest:
        pending_by_gpu[job["gpu"]].append(job)

    active = {gpu_id: [] for gpu_id in gpu_ids}
    while True:
        made_progress = False

        for gpu_id in gpu_ids:
            next_active = []
            for process, log_file, job in active[gpu_id]:
                return_code = process.poll()
                if return_code is None:
                    next_active.append((process, log_file, job))
                    continue
                log_file.close()
                job["return_code"] = return_code
                job["finished_at"] = int(time.time())
                job["status"] = "done" if return_code == 0 else "failed"
                made_progress = True
            active[gpu_id] = next_active

        for gpu_id in gpu_ids:
            while pending_by_gpu[gpu_id] and len(active[gpu_id]) < args.max_jobs_per_gpu:
                job = pending_by_gpu[gpu_id].pop(0)
                process, log_file = launch_job(job, args.output_dir)
                active[gpu_id].append((process, log_file, job))
                made_progress = True

        write_manifest(manifest_path, manifest)

        if not any(active.values()) and not any(pending_by_gpu.values()):
            break

        if not made_progress:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
