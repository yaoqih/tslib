import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.evaluate_ensembles import build_confidence_selector_frame, load_prediction_frame
from utils.market_live_proxy import apply_live_trading_proxy, build_daily_top1_strategy_frame, summarize_live_proxy
from utils.market_research import combine_prediction_frames, get_feature_columns
from utils.market_selector_audit import build_selector_audit_frame, build_threshold_gated_strategy_frame


def load_live_config(config_path):
    return json.loads(Path(config_path).read_text())


def build_model_id(model_cfg, fold_year):
    return f"market_{fold_year}_{model_cfg['seq_len']}_fs{model_cfg['feature_set']}"


def resolved_des(config, model_cfg):
    experiment_tag = config.get("experiment_tag", "")
    des = model_cfg["des"]
    if experiment_tag:
        des = f"{des}_{experiment_tag}"
    return des


def build_setting(config, model_cfg, fold_year):
    return (
        f"long_term_forecast_{build_model_id(model_cfg, fold_year)}_{model_cfg['model']}_market_daily"
        f"_ftMS_sl{model_cfg['seq_len']}_ll0_pl1_dm{model_cfg['d_model']}_nh{model_cfg['n_heads']}"
        f"_el{model_cfg['e_layers']}_dl{model_cfg['d_layers']}_df{model_cfg['d_ff']}_expand2_dc4"
        f"_fc{model_cfg['factor']}_eb{model_cfg['embed']}_dtTrue_{resolved_des(config, model_cfg)}_0"
    )


def build_run_command(config, model_key, fold_year, is_training, market_test_end="", python_bin=None, gpu=0):
    model_cfg = config["models"][model_key]
    python_bin = python_bin or sys.executable
    feature_count = len(get_feature_columns(model_cfg["feature_set"]))
    command = [
        python_bin,
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1" if is_training else "0",
        "--model_id",
        build_model_id(model_cfg, fold_year),
        "--model",
        model_cfg["model"],
        "--data",
        "market_daily",
        "--root_path",
        ".",
        "--data_path",
        Path(config["parquet_path"]).name,
        "--features",
        "MS",
        "--target",
        "label",
        "--freq",
        "d",
        "--seq_len",
        str(model_cfg["seq_len"]),
        "--label_len",
        "0",
        "--pred_len",
        "1",
        "--enc_in",
        str(feature_count),
        "--dec_in",
        str(feature_count),
        "--c_out",
        str(feature_count),
        "--d_model",
        str(model_cfg["d_model"]),
        "--d_ff",
        str(model_cfg["d_ff"]),
        "--e_layers",
        str(model_cfg["e_layers"]),
        "--d_layers",
        str(model_cfg["d_layers"]),
        "--n_heads",
        str(model_cfg["n_heads"]),
        "--factor",
        str(model_cfg["factor"]),
        "--dropout",
        str(model_cfg["dropout"]),
        "--embed",
        model_cfg["embed"],
        "--learning_rate",
        str(config["learning_rate"]),
        "--train_epochs",
        str(config["train_epochs"]),
        "--patience",
        str(config["patience"]),
        "--batch_size",
        str(config["batch_size"]),
        "--num_workers",
        str(config["num_workers"]),
        "--loss",
        str(config["loss"]),
        "--huber_delta",
        str(config["huber_delta"]),
        "--lradj",
        str(config.get("lradj", "type1")),
        "--train_mode",
        str(config.get("train_mode", "best_val")),
        "--gpu",
        str(gpu),
        "--checkpoints",
        config["checkpoints_dir"],
        "--market_fold_year",
        str(fold_year),
        "--market_feature_set",
        model_cfg["feature_set"],
        "--market_cache_path",
        config["cache_path"],
        "--market_start_year",
        str(config["market_start_year"]),
        "--market_min_history",
        str(config["market_min_history"]),
        "--market_min_avg_amount",
        str(config["market_min_avg_amount"]),
        "--des",
        resolved_des(config, model_cfg),
    ]
    if config.get("market_train_full_window", False):
        command.append("--market_train_full_window")
    if market_test_end:
        command.extend(["--market_test_end", market_test_end])
    if model_cfg.get("use_amp", False):
        command.append("--use_amp")
    for key, value in model_cfg.get("extra_args", {}).items():
        command.extend([f"--{key}", str(value)])
    return command


def checkpoint_path(config, model_key, fold_year):
    return Path(config["checkpoints_dir"]) / build_setting(config, config["models"][model_key], fold_year) / "checkpoint.pth"


def prediction_csv_path(config, model_key, fold_year):
    return Path(config["test_results_dir"]) / build_setting(config, config["models"][model_key], fold_year) / "top1_predictions.csv"


def ensure_predictions(config, model_keys, fold_year, market_test_end="", python_bin=None, gpu=0, force=False):
    outputs = {}
    for model_key in model_keys:
        pred_path = prediction_csv_path(config, model_key, fold_year)
        if force or not pred_path.exists():
            command = build_run_command(
                config=config,
                model_key=model_key,
                fold_year=fold_year,
                is_training=False,
                market_test_end=market_test_end,
                python_bin=python_bin,
                gpu=gpu,
            )
            subprocess.run(command, cwd=str(ROOT), check=True)
        outputs[model_key] = pred_path
    return outputs


def latest_common_date(frames_by_model, as_of_date):
    as_of_ts = pd.Timestamp(as_of_date)
    common = None
    for frame in frames_by_model.values():
        dates = set(pd.to_datetime(frame["date"]))
        dates = {item for item in dates if item <= as_of_ts}
        common = dates if common is None else (common & dates)
    if not common:
        raise ValueError(f"No common trading date found on or before {as_of_date}")
    return max(common)


def filter_to_date(frame, target_date):
    target_ts = pd.Timestamp(target_date)
    mask = pd.to_datetime(frame["date"]) == target_ts
    return frame.loc[mask].sort_values(["date", "code"]).reset_index(drop=True)


def build_strategy_frame(strategy_cfg, model_frames):
    model_keys = strategy_cfg["models"]
    frames = [model_frames[name] for name in model_keys]
    kind = strategy_cfg["kind"]
    if kind == "combo":
        return combine_prediction_frames(frames, method=strategy_cfg["method"])
    if kind == "selector":
        if len(frames) != 2:
            raise ValueError("selector strategy requires exactly two models")
        return build_confidence_selector_frame(frames[0], frames[1], method=strategy_cfg["method"])
    if kind == "gated":
        if len(frames) != 2:
            raise ValueError("gated strategy requires exactly two models")
        daily_audit = build_selector_audit_frame(frames[0], frames[1])
        threshold = float(daily_audit["confidence_edge"].abs().quantile(float(strategy_cfg["quantile"])))
        return build_threshold_gated_strategy_frame(
            daily_audit_frame=daily_audit,
            fallback_source=strategy_cfg["fallback_source"],
            min_abs_edge=threshold,
        )[["date", "code", "pred", "true"]]
    raise ValueError(f"Unsupported strategy kind: {kind}")


def summarize_strategy(strategy_frame, scenarios):
    top1_frame = build_daily_top1_strategy_frame(strategy_frame)
    payload = {}
    for name, scenario in scenarios.items():
        proxy_frame = apply_live_trading_proxy(
            top1_frame,
            buy_cost_bps=scenario["buy_cost_bps"],
            sell_cost_bps=scenario["sell_cost_bps"],
        )
        payload[name] = summarize_live_proxy(proxy_frame)
    return top1_frame, payload
