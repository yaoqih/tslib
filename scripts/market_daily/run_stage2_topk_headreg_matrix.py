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


DEFAULT_YEARS = (2021, 2022, 2023, 2024)
DEFAULT_TOPK_WEIGHTS = (0.03, 0.05, 0.08)
DEFAULT_HEAD_CONC_WEIGHTS = (0.00, 0.01, 0.02)
DEFAULT_STATIC_BIAS_WEIGHTS = (0.00, 0.01, 0.02)

MODEL_NAME = "Transformer"
LOSS_TAG_PREFIX = "s2hg"
BASELINE_VARIANT = "stage2topheavy_topk_lossstop_fullyear"
HEADREG_VARIANT = "headreg_lossstop_fullyear"


def parse_float_list(raw_value):
    return tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())


def parse_int_list(raw_value):
    return tuple(int(item.strip()) for item in raw_value.split(",") if item.strip())


def compact_float_tag(value):
    scaled = int(round(float(value) * 1000))
    return f"{scaled:03d}"


def build_job_tag(topk_weight, head_conc_weight, static_bias_weight):
    return (
        f"{LOSS_TAG_PREFIX}"
        f"_tw{compact_float_tag(topk_weight)}"
        f"_hc{compact_float_tag(head_conc_weight)}"
        f"_sb{compact_float_tag(static_bias_weight)}"
        f"_fy"
    )


def build_model_id(year, job_tag):
    return f"market_{year}_csattn_transformer_fp32_{job_tag}"


def build_setting(year, job_tag):
    model_id = build_model_id(year, job_tag)
    return (
        f"long_term_forecast_{model_id}_{MODEL_NAME}_market_daily"
        f"_ftMS_sl20_ll0_pl1_dm64_nh4_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_{job_tag}_0"
    )


def build_prediction_path(year, job_tag):
    return ROOT / "test_results" / build_setting(year, job_tag) / "top1_predictions.csv"


def build_metrics_path(year, job_tag):
    return ROOT / "test_results" / build_setting(year, job_tag) / "market_metrics.txt"


def build_log_path(output_dir, year, job_tag):
    return output_dir / "run_logs" / f"{year}_{job_tag}.log"


def build_run_command(year, topk_weight, head_conc_weight, static_bias_weight):
    job_tag = build_job_tag(topk_weight, head_conc_weight, static_bias_weight)
    model_id = build_model_id(year, job_tag)
    return [
        sys.executable,
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", MODEL_NAME,
        "--data", "market_daily",
        "--root_path", ".",
        "--data_path", "market_daily.parquet",
        "--features", "MS",
        "--target", "label",
        "--freq", "d",
        "--seq_len", "20",
        "--label_len", "0",
        "--pred_len", "1",
        "--enc_in", "24",
        "--dec_in", "24",
        "--c_out", "24",
        "--d_model", "64",
        "--n_heads", "4",
        "--e_layers", "2",
        "--d_layers", "1",
        "--d_ff", "128",
        "--factor", "3",
        "--dropout", "0.1",
        "--embed", "timeF",
        "--batch_size", "4096",
        "--learning_rate", "0.0001",
        "--train_epochs", "20",
        "--patience", "3",
        "--lradj", "type1",
        "--des", job_tag,
        "--itr", "1",
        "--num_workers", "8",
        "--gpu", "0",
        "--market_fold_year", str(year),
        "--market_start_year", "2010",
        "--market_cache_path", "./cache/market_daily_features_full2010.parquet",
        "--market_train_full_window",
        "--market_cross_section_batches",
        "--market_train_horizons", "1,3,5",
        "--market_train_horizon_weights", "2.0,0.25,0.25",
        "--market_rank_loss",
        "--market_rank_weight", "0.1",
        "--market_rank_margin", "0.0",
        "--market_topk_loss",
        "--market_topk_weight", "0.0",
        "--market_topk_k", "3",
        "--market_topk_temperature", "1.0",
        "--market_topk_target_mode", "soft",
        "--train_mode", "train_loss_plateau",
        "--train_plateau_metric", "loss",
        "--train_plateau_patience", "3",
        "--train_plateau_ema_decay", "0.7",
        "--stage2_epochs", "20",
        "--stage2_train_mode", "train_loss_plateau",
        "--stage2_train_plateau_metric", "loss",
        "--stage2_rank_weight", "0.1",
        "--stage2_topk_weight", str(topk_weight),
        "--stage2_topk_k", "3",
        "--stage2_topk_temperature", "1.0",
        "--stage2_topk_target_mode", "soft",
        "--stage2_head_concentration_weight", str(head_conc_weight),
        "--stage2_head_concentration_temperature", "0.7",
        "--stage2_static_bias_weight", str(static_bias_weight),
        "--stage2_static_bias_topk", "3",
    ]


def build_manifest(years, topk_weights, head_conc_weights, static_bias_weights):
    rows = []
    for year, topk_weight, head_conc_weight, static_bias_weight in itertools.product(
        years,
        topk_weights,
        head_conc_weights,
        static_bias_weights,
    ):
        job_tag = build_job_tag(topk_weight, head_conc_weight, static_bias_weight)
        rows.append(
            {
                "year": int(year),
                "job_tag": job_tag,
                "topk_weight": float(topk_weight),
                "head_concentration_weight": float(head_conc_weight),
                "static_bias_weight": float(static_bias_weight),
                "setting": build_setting(year, job_tag),
                "prediction_path": str(build_prediction_path(year, job_tag)),
                "metrics_path": str(build_metrics_path(year, job_tag)),
                "command": json.dumps(build_run_command(year, topk_weight, head_conc_weight, static_bias_weight)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["year", "topk_weight", "head_concentration_weight", "static_bias_weight"]
    ).reset_index(drop=True)


def ensure_output_dirs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_logs").mkdir(parents=True, exist_ok=True)


def write_manifest(output_dir, manifest):
    ensure_output_dirs(output_dir)
    manifest_csv = output_dir / "manifest.csv"
    manifest_json = output_dir / "manifest.json"
    manifest.to_csv(manifest_csv, index=False)
    manifest_json.write_text(
        manifest.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_csv, manifest_json


def load_manifest(output_dir):
    manifest_csv = output_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    return pd.read_csv(manifest_csv)


def prediction_exists(row):
    return Path(row["prediction_path"]).exists()


def pid_is_running(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def load_active_jobs_from_launcher_log(output_dir, manifest):
    launcher_log = output_dir / "launcher.log"
    if not launcher_log.exists():
        return {}, set()

    manifest_lookup = {
        (int(row["year"]), row["job_tag"]): row
        for row in manifest.to_dict("records")
    }
    started = {}
    active = {}
    running_keys = set()

    with launcher_log.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (int(event.get("year", -1)), event.get("job_tag"))
            if event.get("event") == "started":
                started[key] = event
            elif event.get("event") == "finished":
                started.pop(key, None)

    for key, event in started.items():
        pid = event.get("pid")
        gpu = str(event.get("gpu"))
        row = manifest_lookup.get(key)
        if row is None or not pid_is_running(pid):
            continue
        active[gpu] = {
            "process": None,
            "pid": int(pid),
            "row": row,
            "log_file": None,
        }
        running_keys.add(key)

    return active, running_keys


def launch_jobs(output_dir, manifest, gpus, dry_run=False, rerun_completed=False, poll_seconds=30):
    ensure_output_dirs(output_dir)
    pending_rows = []
    for row in manifest.to_dict("records"):
        if rerun_completed or not prediction_exists(row):
            pending_rows.append(row)

    if not pending_rows:
        print(json.dumps({"pending_jobs": 0, "status": "all_completed"}, ensure_ascii=False))
        return

    gpu_ids = [item.strip() for item in gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")

    if dry_run:
        for row in pending_rows:
            print(json.dumps({"year": row["year"], "job_tag": row["job_tag"], "command": json.loads(row["command"])}, ensure_ascii=False))
        return

    active, running_keys = load_active_jobs_from_launcher_log(output_dir, manifest)
    queue = [
        row for row in pending_rows
        if (int(row["year"]), row["job_tag"]) not in running_keys
    ]

    for gpu_id, payload in sorted(active.items()):
        print(
            json.dumps(
                {
                    "event": "resumed",
                    "gpu": gpu_id,
                    "pid": payload["pid"],
                    "year": payload["row"]["year"],
                    "job_tag": payload["row"]["job_tag"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    while queue or active:
        finished_gpus = []
        for gpu_id, payload in active.items():
            proc = payload["process"]
            if proc is not None:
                is_finished = proc.poll() is not None
                returncode = proc.returncode if is_finished else None
            else:
                is_finished = not pid_is_running(payload["pid"])
                returncode = None
            if is_finished:
                if payload["log_file"] is not None:
                    payload["log_file"].close()
                finished_gpus.append(gpu_id)
                print(
                    json.dumps(
                        {
                            "event": "finished",
                            "gpu": gpu_id,
                            "year": payload["row"]["year"],
                            "job_tag": payload["row"]["job_tag"],
                            "returncode": returncode,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
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
            log_path = build_log_path(output_dir, int(row["year"]), row["job_tag"])
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[gpu_id] = {
                "process": proc,
                "pid": proc.pid,
                "row": row,
                "log_file": log_file,
            }
            print(
                json.dumps(
                    {
                        "event": "started",
                        "gpu": gpu_id,
                        "pid": proc.pid,
                        "year": row["year"],
                        "job_tag": row["job_tag"],
                        "log_path": str(log_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if active:
            time.sleep(poll_seconds)


def evaluate_prediction_artifact(prediction_path):
    raw = evaluate_prediction_file(
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
    top_codes = [code for _, code in raw.get("top_picks", [])]
    return {
        "top1_mean": raw["top1_mean_return"],
        "top1_cum": raw["top1_cumulative_return"],
        "top1_sharpe": raw["top1_sharpe"],
        "top3_mean": raw["top3_mean_return"],
        "top5_mean": raw["top5_mean_return"],
        "ic": raw["ic"],
        "rank_ic": raw["rank_ic"],
        "unique": len(set(top_codes)),
        "max_rep": max((top_codes.count(code) for code in set(top_codes)), default=0),
        "debias015_top1_mean": debias["top1_mean_return"],
        "debias015_top1_sharpe": debias["top1_sharpe"],
        "num_days": raw["top1_num_days"],
    }


def find_baseline_prediction(year, variant):
    matches = list(
        (ROOT / "test_results").glob(
            f"long_term_forecast_market_{year}_*_{variant}_0/top1_predictions.csv"
        )
    )
    if not matches:
        return None
    return matches[0]


def summarize_jobs(output_dir, manifest):
    rows = []
    for row in manifest.to_dict("records"):
        pred_path = Path(row["prediction_path"])
        record = {
            "year": int(row["year"]),
            "job_tag": row["job_tag"],
            "topk_weight": float(row["topk_weight"]),
            "head_concentration_weight": float(row["head_concentration_weight"]),
            "static_bias_weight": float(row["static_bias_weight"]),
            "prediction_path": str(pred_path),
            "completed": pred_path.exists(),
        }
        if pred_path.exists():
            record.update(evaluate_prediction_artifact(pred_path))
        rows.append(record)

    summary = pd.DataFrame(rows)
    if "top1_mean" not in summary.columns:
        summary["top1_mean"] = pd.NA
    if "top1_sharpe" not in summary.columns:
        summary["top1_sharpe"] = pd.NA
    summary = summary.sort_values(
        ["year", "top1_mean", "top1_sharpe", "job_tag"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    summary_path = output_dir / "summary_raw.csv"
    summary.to_csv(summary_path, index=False)

    mainline_rows = []
    headreg_rows = []
    for year in sorted(summary["year"].dropna().unique()):
        mainline_path = find_baseline_prediction(int(year), BASELINE_VARIANT)
        headreg_path = find_baseline_prediction(int(year), HEADREG_VARIANT)
        if mainline_path is not None:
            metrics = evaluate_prediction_artifact(mainline_path)
            mainline_rows.append({"year": int(year), **metrics})
        if headreg_path is not None:
            metrics = evaluate_prediction_artifact(headreg_path)
            headreg_rows.append({"year": int(year), **metrics})

    mainline_df = pd.DataFrame(mainline_rows)
    headreg_df = pd.DataFrame(headreg_rows)

    if not mainline_df.empty:
        merged = summary.merge(mainline_df, on="year", how="left", suffixes=("", "_mainline"))
        for col in ("top1_mean", "top1_sharpe", "unique", "max_rep", "debias015_top1_mean"):
            rhs = f"{col}_mainline"
            if col in merged.columns and rhs in merged.columns:
                merged[f"delta_{col}_vs_mainline"] = merged[col] - merged[rhs]
        merged.to_csv(output_dir / "summary_vs_mainline.csv", index=False)

    if not headreg_df.empty:
        merged = summary.merge(headreg_df, on="year", how="left", suffixes=("", "_headreg"))
        for col in ("top1_mean", "top1_sharpe", "unique", "max_rep", "debias015_top1_mean"):
            rhs = f"{col}_headreg"
            if col in merged.columns and rhs in merged.columns:
                merged[f"delta_{col}_vs_headreg"] = merged[col] - merged[rhs]
        merged.to_csv(output_dir / "summary_vs_headreg.csv", index=False)

    completed = summary[summary["completed"]].copy()
    if not completed.empty:
        aggregate = (
            completed.groupby(
                ["topk_weight", "head_concentration_weight", "static_bias_weight"],
                dropna=False,
            )
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
            .sort_values(
                ["avg_top1_mean", "avg_top1_sharpe", "avg_unique"],
                ascending=[False, False, False],
            )
        )
        aggregate.to_csv(output_dir / "summary_aggregate.csv", index=False)

    print(
        json.dumps(
            {
                "summary_raw_csv": str(summary_path),
                "completed_jobs": int(summary["completed"].sum()),
                "total_jobs": int(summary.shape[0]),
            },
            ensure_ascii=False,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Stage2 top-k/head-regularizer matrix runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("manifest", "launch", "summarize"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--output_dir", type=str, default="logs/stage2_topk_headreg_matrix")
        sub.add_argument("--years", type=str, default="2021,2022,2023,2024")
        sub.add_argument("--topk_weights", type=str, default="0.03,0.05,0.08")
        sub.add_argument("--head_concentration_weights", type=str, default="0.00,0.01,0.02")
        sub.add_argument("--static_bias_weights", type=str, default="0.00,0.01,0.02")

    launch = subparsers.choices["launch"]
    launch.add_argument("--gpus", type=str, default="0,1,2,3")
    launch.add_argument("--dry_run", action="store_true", default=False)
    launch.add_argument("--rerun_completed", action="store_true", default=False)
    launch.add_argument("--poll_seconds", type=int, default=30)

    return parser


def main():
    args = build_parser().parse_args()
    output_dir = ROOT / args.output_dir
    years = parse_int_list(args.years)
    topk_weights = parse_float_list(args.topk_weights)
    head_conc_weights = parse_float_list(args.head_concentration_weights)
    static_bias_weights = parse_float_list(args.static_bias_weights)

    manifest = build_manifest(
        years=years or DEFAULT_YEARS,
        topk_weights=topk_weights or DEFAULT_TOPK_WEIGHTS,
        head_conc_weights=head_conc_weights or DEFAULT_HEAD_CONC_WEIGHTS,
        static_bias_weights=static_bias_weights or DEFAULT_STATIC_BIAS_WEIGHTS,
    )

    if args.command == "manifest":
        manifest_csv, manifest_json = write_manifest(output_dir, manifest)
        print(
            json.dumps(
                {
                    "manifest_csv": str(manifest_csv),
                    "manifest_json": str(manifest_json),
                    "jobs": int(manifest.shape[0]),
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "launch":
        write_manifest(output_dir, manifest)
        launch_jobs(
            output_dir=output_dir,
            manifest=manifest,
            gpus=args.gpus,
            dry_run=args.dry_run,
            rerun_completed=args.rerun_completed,
            poll_seconds=args.poll_seconds,
        )
        return

    if args.command == "summarize":
        write_manifest(output_dir, manifest)
        summarize_jobs(output_dir=output_dir, manifest=manifest)
        return


if __name__ == "__main__":
    main()
