import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.evaluate_ensembles import load_prediction_frame
from utils.market_research import evaluate_prediction_frame


TIMESNET_VARIANTS = {
    "timesnet_base": {
        2015: "test_results/long_term_forecast_market_2015_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_class_timesnet20_probe_v3_0/top1_predictions.csv",
        2017: "test_results/long_term_forecast_market_2017_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_class_timesnet20_probe_tail_0/top1_predictions.csv",
        2019: "test_results/long_term_forecast_market_2019_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_class_timesnet20_probe_v3_0/top1_predictions.csv",
        2021: "test_results/long_term_forecast_market_2021_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_class_timesnet20_probe_v3_0/top1_predictions.csv",
        2022: "test_results/long_term_forecast_market_2022_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_class_timesnet20_probe_tail_0/top1_predictions.csv",
        2024: "test_results/long_term_forecast_market_2024_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_class_timesnet20_probe_v3_0/top1_predictions.csv",
    },
    "timesnet_hbce_a01": {
        2015: "test_results/long_term_forecast_market_2015_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a01_broadA_0/top1_predictions.csv",
        2017: "test_results/long_term_forecast_market_2017_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a01_broadA_0/top1_predictions.csv",
        2019: "test_results/long_term_forecast_market_2019_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a01_broadA_0/top1_predictions.csv",
        2021: "test_results/long_term_forecast_market_2021_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a01_broadB_0/top1_predictions.csv",
        2022: "test_results/long_term_forecast_market_2022_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a01_broadB_0/top1_predictions.csv",
        2024: "test_results/long_term_forecast_market_2024_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a01_broadB_0/top1_predictions.csv",
    },
    "timesnet_hbce_a02": {
        2015: "test_results/long_term_forecast_market_2015_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a02_broadA_0/top1_predictions.csv",
        2017: "test_results/long_term_forecast_market_2017_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a02_broadA_0/top1_predictions.csv",
        2019: "test_results/long_term_forecast_market_2019_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a02_broadA_0/top1_predictions.csv",
        2021: "test_results/long_term_forecast_market_2021_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a02_broadB_0/top1_predictions.csv",
        2022: "test_results/long_term_forecast_market_2022_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a02_broadB_0/top1_predictions.csv",
        2024: "test_results/long_term_forecast_market_2024_20_fsA_TimesNet_market_daily_ftMS_sl20_ll0_pl1_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebfixed_dtTrue_market_loss_timesnet20_hbce_a02_broadB_0/top1_predictions.csv",
    },
}

DLINEAR_MAINLINE = {
    2015: "test_results/long_term_forecast_market_2015_120_fsA_DLinear_market_daily_ftMS_sl120_ll0_pl1_dm64_nh4_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_market_hbce_a01_fsA_0/top1_predictions.csv",
    2017: "test_results/long_term_forecast_market_2017_120_fsA_DLinear_market_daily_ftMS_sl120_ll0_pl1_dm64_nh4_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_market_hbce_a01_fsA_0/top1_predictions.csv",
    2019: "test_results/long_term_forecast_market_2019_120_fsA_DLinear_market_daily_ftMS_sl120_ll0_pl1_dm64_nh4_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_market_hbce_a01_fsA_0/top1_predictions.csv",
    2021: "test_results/long_term_forecast_market_2021_120_fsA_DLinear_market_daily_ftMS_sl120_ll0_pl1_dm64_nh4_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_market_hbce_a01_fsA_0/top1_predictions.csv",
    2022: "test_results/long_term_forecast_market_2022_120_fsA_DLinear_market_daily_ftMS_sl120_ll0_pl1_dm64_nh4_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_market_hbce_a01_fsA_0/top1_predictions.csv",
    2024: "test_results/long_term_forecast_market_2024_120_fsA_DLinear_market_daily_ftMS_sl120_ll0_pl1_dm64_nh4_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_market_hbce_a01_fsA_0/top1_predictions.csv",
}


def wait_for_pid(pid, poll_seconds):
    while True:
        proc = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True, check=False)
        if proc.returncode != 0 or len(proc.stdout.strip().splitlines()) <= 1:
            return
        time.sleep(poll_seconds)


def evaluate_variant(name, mapping):
    rows = []
    aggregate_frames = []
    for fold_year, path in sorted(mapping.items()):
        pred_path = Path(path)
        if not pred_path.exists():
            continue
        frame = load_prediction_frame(pred_path)
        metrics = evaluate_prediction_frame(frame)
        rows.append({"variant": name, "fold_year": fold_year, **metrics, "pred_path": str(pred_path)})
        aggregate_frames.append(frame)
    if aggregate_frames:
        merged = pd.concat(aggregate_frames, ignore_index=True)
        metrics = evaluate_prediction_frame(merged)
        rows.append({"variant": name, "fold_year": -1, **metrics, "pred_path": "aggregate"})
    return rows


def choose_timesnet_variant(summary_frame):
    aggregate = summary_frame[summary_frame["fold_year"] == -1].copy()
    aggregate["positive_years"] = summary_frame[summary_frame["fold_year"] != -1].groupby("variant")["mean_return"].apply(
        lambda x: int((x > 0).sum())
    )
    aggregate = aggregate.sort_values(["mean_return", "sharpe", "positive_years"], ascending=[False, False, False])
    return aggregate.iloc[0]["variant"]


def run_command(command):
    subprocess.run(command, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Wait for P0 closure, then run P1 selector and ensemble research")
    parser.add_argument("--wait_pid", type=int, default=18181)
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--output_dir", type=str, default="logs/market_p1")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wait_for_pid(args.wait_pid, args.poll_seconds)

    challenger_rows = []
    for name, mapping in TIMESNET_VARIANTS.items():
        challenger_rows.extend(evaluate_variant(name, mapping))
    challenger_summary = pd.DataFrame(challenger_rows)
    challenger_summary.to_csv(output_dir / "timesnet_challenger_summary.csv", index=False)

    chosen_variant = choose_timesnet_variant(challenger_summary)
    chosen_paths = TIMESNET_VARIANTS[chosen_variant]

    candidate_args = []
    for fold_year in sorted(set(DLINEAR_MAINLINE) & set(chosen_paths)):
        candidate_args.append(f"dlinear_a01={DLINEAR_MAINLINE[fold_year]}")
        candidate_args.append(f"{chosen_variant}={chosen_paths[fold_year]}")

    ensemble_output = output_dir / "p1_ensemble_summary.csv"
    run_command(
        [
            args.python,
            "scripts/market_daily/evaluate_ensembles.py",
            *sum([["--candidate", item] for item in candidate_args], []),
            "--output_csv",
            str(ensemble_output),
        ]
    )

    decision = {
        "chosen_timesnet_variant": chosen_variant,
        "timesnet_summary_csv": str(output_dir / "timesnet_challenger_summary.csv"),
        "p1_ensemble_summary_csv": str(ensemble_output),
        "candidate_years": sorted(set(DLINEAR_MAINLINE) & set(chosen_paths)),
    }
    (output_dir / "p1_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
