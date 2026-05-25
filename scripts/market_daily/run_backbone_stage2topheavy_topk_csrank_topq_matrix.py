import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_research import evaluate_prediction_file


DEFAULT_YEARS = (2021, 2022, 2023, 2024, 2025)
DEFAULT_MODELS = ("PatchTST", "iTransformer", "TimeMixer", "TimesNet", "WPMixer", "Transformer", "FreTS")
DEFAULT_FEATURE_SET = "A"
DEFAULT_SEQ_LEN = 20
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_CACHE_PATH = "./cache/market_daily_features_full2010.parquet"

MODEL_CONFIGS = {
    "PatchTST": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4, "epochs": 20, "speed": 2},
    "iTransformer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4, "epochs": 20, "speed": 2},
    "TimeMixer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4, "epochs": 20, "speed": 5},
    "TimesNet": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4, "epochs": 20, "speed": 3},
    "WPMixer": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4, "epochs": 20, "speed": 6},
    "Transformer": {"d_model": 64, "d_ff": 128, "e_layers": 2, "n_heads": 4, "epochs": 20, "speed": 2},
    "FreTS": {"d_model": 128, "d_ff": 256, "e_layers": 2, "n_heads": 4, "epochs": 20, "speed": 4},
}


def parse_int_list(raw_value):
    return tuple(int(item.strip()) for item in raw_value.split(",") if item.strip())


def parse_str_list(raw_value):
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def build_setting(year, model, seq_len, d_model, d_ff, e_layers, n_heads, embed, distil, des, model_id):
    return (
        f"long_term_forecast_{model_id}_{model}_market_daily"
        f"_ftMS_sl{seq_len}_ll0_pl1_dm{d_model}_nh{n_heads}_el{e_layers}_dl1_df{d_ff}"
        f"_expand2_dc4_fc3_eb{embed}_dt{distil}_{des}_0"
    )


def build_result_paths(year, model, seq_len, d_model, d_ff, e_layers, n_heads, embed, distil, des, model_id):
    base = ROOT / "test_results" / build_setting(
        year, model, seq_len, d_model, d_ff, e_layers, n_heads, embed, distil, des, model_id
    )
    return base / "top1_predictions.csv", base / "market_metrics.txt"


def build_job_record(year, model, feature_set, seq_len, batch_size, cache_path, learning_rate, des, market_cs_recent_k):
    model_cfg = MODEL_CONFIGS[model]
    embed = "fixed" if model in {"TimesNet", "TimeMixer"} else "timeF"
    distil = "True"
    model_id = f"market_{year}_{des}_{model}"
    prediction_path, metrics_path = build_result_paths(
        year, model, seq_len, model_cfg["d_model"], model_cfg["d_ff"], model_cfg["e_layers"], model_cfg["n_heads"],
        embed, distil, des, model_id
    )
    command = [
        sys.executable,
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
        "--enc_in", "24",
        "--dec_in", "24",
        "--c_out", "24",
        "--d_model", str(model_cfg["d_model"]),
        "--d_ff", str(model_cfg["d_ff"]),
        "--e_layers", str(model_cfg["e_layers"]),
        "--n_heads", str(model_cfg["n_heads"]),
        "--factor", "3",
        "--dropout", "0.1",
        "--learning_rate", str(learning_rate),
        "--train_epochs", str(model_cfg["epochs"]),
        "--patience", "3",
        "--batch_size", str(batch_size),
        "--num_workers", "8",
        "--loss", "Huber",
        "--huber_delta", "1.0",
        "--gpu", "0",
        "--market_fold_year", str(year),
        "--market_feature_set", feature_set,
        "--market_cache_path", cache_path,
        "--market_start_year", "2010",
        "--market_min_history", "120",
        "--market_min_avg_amount", "20000000.0",
        "--market_train_full_window",
        "--market_cross_section_batches",
        "--market_train_horizons", "1,3,5",
        "--market_train_horizon_weights", "2.0,0.25,0.25",
        "--market_target_mode", "cross_section_rank",
        "--market_pred_topq_ratio", "0.1",
        "--market_pred_topq_weight", "2.0",
        "--market_rank_loss",
        "--market_rank_weight", "0.1",
        "--market_rank_margin", "0.0",
        "--market_topk_loss",
        "--market_topk_weight", "0.03",
        "--market_topk_k", "3",
        "--market_topk_temperature", "1.0",
        "--market_topk_target_mode", "soft",
        "--train_mode", "train_loss_plateau",
        "--train_plateau_metric", "loss",
        "--train_plateau_patience", "3",
        "--train_plateau_ema_decay", "0.7",
        "--des", des,
        "--stage2_epochs", "20",
        "--stage2_train_mode", "train_loss_plateau",
        "--stage2_train_plateau_metric", "loss",
        "--stage2_rank_weight", "0.1",
        "--stage2_topk_weight", "0.03",
        "--stage2_topk_k", "3",
        "--stage2_topk_temperature", "1.0",
        "--stage2_topk_target_mode", "soft",
        "--stage2_head_concentration_weight", "0.01",
        "--stage2_head_concentration_temperature", "0.7",
        "--stage2_static_bias_weight", "0.01",
        "--stage2_static_bias_topk", "3",
        "--market_cs_recent_k", str(market_cs_recent_k),
    ]
    if model == "TimeMixer":
        command.extend([
            "--embed", "fixed",
            "--down_sampling_layers", "3",
            "--down_sampling_method", "avg",
            "--down_sampling_window", "2",
        ])
    elif model == "TimesNet":
        command.extend(["--embed", "fixed"])
    elif model == "WPMixer":
        command.extend(["--patch_len", "16"])
    elif model == "FreTS":
        command.extend(["--embed", "timeF", "--channel_independence", "0"])
    else:
        command.extend(["--embed", "timeF"])

    return {
        "year": int(year),
        "model": model,
        "feature_set": feature_set,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "des": des,
        "speed": int(model_cfg["speed"]),
        "model_id": model_id,
        "setting": build_setting(
            year, model, seq_len, model_cfg["d_model"], model_cfg["d_ff"], model_cfg["e_layers"], model_cfg["n_heads"],
            embed, distil, des, model_id
        ),
        "prediction_path": str(prediction_path),
        "metrics_path": str(metrics_path),
        "command": json.dumps(command),
    }


def ensure_dirs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_logs").mkdir(parents=True, exist_ok=True)


def build_manifest(years, models, feature_set, seq_len, batch_size, cache_path, learning_rate, des, market_cs_recent_k):
    rows = []
    for year, model in itertools.product(years, models):
        rows.append(
            build_job_record(
                year=year,
                model=model,
                feature_set=feature_set,
                seq_len=seq_len,
                batch_size=batch_size,
                cache_path=cache_path,
                learning_rate=learning_rate,
                des=des,
                market_cs_recent_k=market_cs_recent_k,
            )
        )
    return pd.DataFrame(rows)


def write_manifest(output_dir, manifest):
    ensure_dirs(output_dir)
    manifest_csv = output_dir / "manifest.csv"
    manifest_json = output_dir / "manifest.json"
    manifest.to_csv(manifest_csv, index=False)
    manifest_json.write_text(manifest.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    return manifest_csv, manifest_json


def load_manifest(output_dir):
    manifest_csv = output_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    return pd.read_csv(manifest_csv)


def prediction_exists(row):
    return Path(row["prediction_path"]).exists()


def build_balanced_queue(rows):
    pending = sorted(rows, key=lambda row: (-int(row["speed"]), int(row["year"]), row["model"]))
    gpu_loads = [0, 0, 0, 0]
    assignments = [[] for _ in range(4)]
    for row in pending:
        gpu_idx = min(range(4), key=lambda idx: (gpu_loads[idx], idx))
        assignments[gpu_idx].append(row)
        gpu_loads[gpu_idx] += int(row["speed"])
    queue = []
    for round_idx in range(max(len(items) for items in assignments)):
        for gpu_idx in range(4):
            if round_idx < len(assignments[gpu_idx]):
                queue.append(assignments[gpu_idx][round_idx])
    return queue


def launch_jobs(output_dir, manifest, gpus, dry_run=False, rerun_completed=False, poll_seconds=30):
    ensure_dirs(output_dir)
    pending_rows = [row for row in manifest.to_dict("records") if rerun_completed or not prediction_exists(row)]
    if not pending_rows:
        print(json.dumps({"pending_jobs": 0, "status": "all_completed"}, ensure_ascii=False))
        return

    gpu_ids = [item.strip() for item in gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    if len(gpu_ids) != 4:
        raise ValueError("This launcher expects exactly 4 GPU ids for balanced scheduling")

    queue = build_balanced_queue(pending_rows)

    if dry_run:
        for row in queue:
            print(json.dumps({"year": row["year"], "model": row["model"], "speed": row["speed"], "command": json.loads(row["command"])}, ensure_ascii=False))
        return

    active = {}

    while queue or active:
        finished_gpus = []
        for gpu_id, payload in list(active.items()):
            proc = payload["process"]
            if proc.poll() is not None:
                payload["log_file"].close()
                finished_gpus.append(gpu_id)
                print(json.dumps({
                    "event": "finished",
                    "gpu": gpu_id,
                    "year": payload["row"]["year"],
                    "model": payload["row"]["model"],
                    "returncode": proc.returncode,
                }, ensure_ascii=False), flush=True)
        for gpu_id in finished_gpus:
            active.pop(gpu_id, None)

        for gpu_id in gpu_ids:
            if gpu_id in active or not queue:
                continue
            row = queue.pop(0)
            command = json.loads(row["command"])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            log_path = output_dir / "run_logs" / f"{row['year']}_{row['model']}.log"
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[gpu_id] = {"process": proc, "row": row, "log_file": log_file}
            print(json.dumps({
                "event": "started",
                "gpu": gpu_id,
                "pid": proc.pid,
                "year": row["year"],
                "model": row["model"],
                "speed": row["speed"],
                "log_path": str(log_path),
            }, ensure_ascii=False), flush=True)

        if active:
            time.sleep(poll_seconds)


def evaluate_prediction_artifact(prediction_path):
    metrics = evaluate_prediction_file(
        prediction_path,
        topk_list=(1, 3, 5),
        score_debias="none",
        score_debias_strength=1.0,
    )
    debias = evaluate_prediction_file(
        prediction_path,
        topk_list=(1,),
        score_debias="expanding_mean",
        score_debias_strength=0.15,
    )
    top_codes = [code for _, code in metrics.get("top_picks", [])]
    return {
        "top1_mean": metrics["top1_mean_return"],
        "top1_sharpe": metrics["top1_sharpe"],
        "top3_mean": metrics["top3_mean_return"],
        "top5_mean": metrics["top5_mean_return"],
        "ic": metrics["ic"],
        "rank_ic": metrics["rank_ic"],
        "unique": len(set(top_codes)),
        "max_rep": max((top_codes.count(code) for code in set(top_codes)), default=0),
        "debias015_top1_mean": debias["top1_mean_return"],
        "debias015_top1_sharpe": debias["top1_sharpe"],
        "num_days": metrics["top1_num_days"],
    }


def summarize_jobs(output_dir, manifest):
    rows = []
    for row in manifest.to_dict("records"):
        pred_path = Path(row["prediction_path"])
        record = {
            "year": int(row["year"]),
            "model": row["model"],
            "feature_set": row["feature_set"],
            "seq_len": int(row["seq_len"]),
            "prediction_path": str(pred_path),
            "completed": pred_path.exists(),
        }
        if pred_path.exists():
            record.update(evaluate_prediction_artifact(pred_path))
        rows.append(record)

    summary = pd.DataFrame(rows)
    sort_columns = [column for column in ("year", "top1_mean", "top1_sharpe") if column in summary.columns]
    if sort_columns:
        ascending = [True] + [False] * (len(sort_columns) - 1)
        summary = summary.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    summary.to_csv(output_dir / "summary_raw.csv", index=False)

    completed = summary[summary["completed"]].copy()
    if not completed.empty:
        aggregate = (
            completed.groupby(["model"], dropna=False)
            .agg(
                years=("year", "count"),
                avg_top1_mean=("top1_mean", "mean"),
                avg_top1_sharpe=("top1_sharpe", "mean"),
                avg_top3_mean=("top3_mean", "mean"),
                avg_top5_mean=("top5_mean", "mean"),
                avg_ic=("ic", "mean"),
                avg_rank_ic=("rank_ic", "mean"),
                avg_unique=("unique", "mean"),
                avg_max_rep=("max_rep", "mean"),
                avg_debias015_top1_mean=("debias015_top1_mean", "mean"),
                min_top1_mean=("top1_mean", "min"),
                min_top1_sharpe=("top1_sharpe", "min"),
            )
            .reset_index()
            .sort_values(["avg_top1_mean", "avg_top1_sharpe"], ascending=[False, False])
        )
        aggregate.to_csv(output_dir / "summary_aggregate.csv", index=False)

        per_year = (
            completed.sort_values(["year", "top1_mean", "top1_sharpe"], ascending=[True, False, False])
            .groupby("year", as_index=False)
            .first()[["year", "model", "top1_mean", "top1_sharpe", "unique", "max_rep", "debias015_top1_mean"]]
        )
        per_year.to_csv(output_dir / "summary_year_winners.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "summary_aggregate.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "summary_year_winners.csv", index=False)

    print(json.dumps({
        "summary_raw": str(output_dir / "summary_raw.csv"),
        "summary_aggregate": str(output_dir / "summary_aggregate.csv"),
        "summary_year_winners": str(output_dir / "summary_year_winners.csv"),
    }, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Run backbone stage2topheavy_topk csrank_topq market matrix")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output_dir", default="logs/backbone_stage2topheavy_topk_csrank_topq_matrix")
    manifest.add_argument("--years", default="2021,2022,2023,2024,2025")
    manifest.add_argument("--models", default="PatchTST,iTransformer,TimeMixer,TimesNet,WPMixer,Transformer,FreTS")
    manifest.add_argument("--feature_set", default=DEFAULT_FEATURE_SET)
    manifest.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    manifest.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    manifest.add_argument("--cache_path", default=DEFAULT_CACHE_PATH)
    manifest.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    manifest.add_argument("--des", default="stage2topheavy_topk_csrank_topq_v2")
    manifest.add_argument("--market_cs_recent_k", type=int, default=5)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--output_dir", default="logs/backbone_stage2topheavy_topk_csrank_topq_matrix")
    launch.add_argument("--years", default="2021,2022,2023,2024,2025")
    launch.add_argument("--models", default="PatchTST,iTransformer,TimeMixer,TimesNet,WPMixer,Transformer,FreTS")
    launch.add_argument("--feature_set", default=DEFAULT_FEATURE_SET)
    launch.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    launch.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    launch.add_argument("--cache_path", default=DEFAULT_CACHE_PATH)
    launch.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    launch.add_argument("--des", default="stage2topheavy_topk_csrank_topq_v2")
    launch.add_argument("--market_cs_recent_k", type=int, default=5)
    launch.add_argument("--gpus", default="0,1,2,3")
    launch.add_argument("--dry_run", action="store_true", default=False)
    launch.add_argument("--rerun_completed", action="store_true", default=False)
    launch.add_argument("--poll_seconds", type=int, default=30)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output_dir", default="logs/backbone_stage2topheavy_topk_csrank_topq_matrix")

    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.cmd == "manifest":
        years = parse_int_list(args.years)
        models = parse_str_list(args.models)
        manifest = build_manifest(
            years=years,
            models=models,
            feature_set=args.feature_set,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            cache_path=args.cache_path,
            learning_rate=args.learning_rate,
            des=args.des,
            market_cs_recent_k=args.market_cs_recent_k,
        )
        write_manifest(output_dir, manifest)
        print(json.dumps({"manifest_rows": len(manifest)}, ensure_ascii=False))
        return

    if args.cmd == "launch":
        if not (output_dir / "manifest.csv").exists():
            years = parse_int_list(args.years)
            models = parse_str_list(args.models)
            manifest = build_manifest(
                years=years,
                models=models,
                feature_set=args.feature_set,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                cache_path=args.cache_path,
                learning_rate=args.learning_rate,
                des=args.des,
                market_cs_recent_k=args.market_cs_recent_k,
            )
            write_manifest(output_dir, manifest)
        manifest = load_manifest(output_dir)
        launch_jobs(
            output_dir=output_dir,
            manifest=manifest,
            gpus=args.gpus,
            dry_run=args.dry_run,
            rerun_completed=args.rerun_completed,
            poll_seconds=args.poll_seconds,
        )
        return

    if args.cmd == "summarize":
        manifest = load_manifest(output_dir)
        summarize_jobs(output_dir, manifest)
        return


if __name__ == "__main__":
    raise SystemExit(main())
