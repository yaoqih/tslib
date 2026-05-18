import math

import numpy as np
import pandas as pd

from utils.market_research import evaluate_prediction_frame


def _daily_top_rows(frame, pred_column):
    ranked = frame.sort_values(["date", pred_column], ascending=[True, False]).copy()
    ranked["row_id"] = ranked.groupby("date").cumcount()
    return ranked[ranked["row_id"] < 2].copy()


def _daily_top1_gap(frame, pred_column):
    top2 = _daily_top_rows(frame, pred_column)[["date", pred_column, "row_id"]]
    pivot = top2.pivot(index="date", columns="row_id", values=pred_column)
    top1 = pivot[0]
    top2 = pivot[1] if 1 in pivot.columns else pd.Series(0.0, index=pivot.index)
    return (top1 - top2).fillna(0.0)


def _daily_pick_details(frame, pred_column, prefix):
    tradable = frame.copy()
    if "tradable" in tradable.columns:
        tradable = tradable[tradable["tradable"]].copy()
    ranked = tradable.sort_values(["date", pred_column], ascending=[True, False]).copy()
    top1 = ranked.groupby("date").head(1).copy()
    top1 = top1.rename(
        columns={
            "code": f"{prefix}_code",
            pred_column: f"{prefix}_pred",
            "true": f"{prefix}_true",
        }
    )
    keep = ["date", f"{prefix}_code", f"{prefix}_pred", f"{prefix}_true"]
    return top1[keep].reset_index(drop=True)


def build_selector_audit_frame(first_frame, second_frame):
    merged = first_frame.rename(columns={"pred": "pred_left"}).merge(
        second_frame.rename(columns={"pred": "pred_right"}),
        on=["date", "code", "true"],
        how="inner",
        suffixes=("_left", "_right"),
    )

    if "tradable_left" in merged.columns and "tradable_right" in merged.columns:
        merged["tradable"] = merged["tradable_left"] & merged["tradable_right"]
    elif "tradable_left" in merged.columns:
        merged["tradable"] = merged["tradable_left"]
    elif "tradable_right" in merged.columns:
        merged["tradable"] = merged["tradable_right"]
    else:
        merged["tradable"] = True

    left_gap = _daily_top1_gap(merged[["date", "code", "pred_left"]], "pred_left")
    right_gap = _daily_top1_gap(merged[["date", "code", "pred_right"]], "pred_right")
    selector = pd.DataFrame(
        {
            "date": left_gap.index,
            "left_gap": left_gap.values,
            "right_gap": right_gap.values,
        }
    )
    selector["selected_source"] = np.where(selector["left_gap"] >= selector["right_gap"], "left", "right")
    selector["confidence_edge"] = selector["left_gap"] - selector["right_gap"]

    left_pick = _daily_pick_details(merged[["date", "code", "pred_left", "true", "tradable"]], "pred_left", "left")
    right_pick = _daily_pick_details(merged[["date", "code", "pred_right", "true", "tradable"]], "pred_right", "right")

    selected_frame = merged[["date", "code", "true", "tradable", "pred_left", "pred_right"]].copy()
    selected_frame = selected_frame.merge(selector[["date", "selected_source"]], on="date", how="left")
    selected_frame["pred"] = np.where(
        selected_frame["selected_source"] == "left",
        selected_frame["pred_left"],
        selected_frame["pred_right"],
    )
    selected_pick = _daily_pick_details(selected_frame[["date", "code", "pred", "true", "tradable"]], "pred", "selected")

    daily = selector.merge(left_pick, on="date", how="left").merge(right_pick, on="date", how="left").merge(
        selected_pick, on="date", how="left"
    )
    daily["fold_year"] = pd.to_datetime(daily["date"]).dt.year
    daily["disagreement"] = daily["left_code"] != daily["right_code"]
    daily["selected_is_left"] = daily["selected_source"] == "left"
    daily["selected_matches_left_pick"] = daily["selected_code"] == daily["left_code"]
    daily["selected_matches_right_pick"] = daily["selected_code"] == daily["right_code"]
    daily["top1_change_vs_left"] = daily["selected_code"] != daily["left_code"]
    daily["top1_change_vs_right"] = daily["selected_code"] != daily["right_code"]
    daily["selected_return"] = daily["selected_true"].fillna(0.0)
    daily["left_return"] = daily["left_true"].fillna(0.0)
    daily["right_return"] = daily["right_true"].fillna(0.0)
    daily["selector_alpha_vs_left"] = daily["selected_return"] - daily["left_return"]
    daily["selector_alpha_vs_right"] = daily["selected_return"] - daily["right_return"]
    daily["switch_from_prev"] = daily["selected_source"].ne(daily["selected_source"].shift(1)).fillna(False)
    if not daily.empty:
        daily.loc[daily.index[0], "switch_from_prev"] = False
    daily["is_top_decile_day"] = daily["selected_return"].rank(method="first", pct=True) >= 0.9
    daily["is_bottom_decile_day"] = daily["selected_return"].rank(method="first", pct=True) <= 0.1
    return daily.sort_values("date").reset_index(drop=True)


def build_threshold_gated_audit_frame(daily_audit_frame, fallback_source="right", min_abs_edge=0.0):
    if fallback_source not in {"left", "right"}:
        raise ValueError("fallback_source must be 'left' or 'right'")
    frame = daily_audit_frame.copy()
    frame["gated_source"] = np.where(frame["confidence_edge"].abs() >= min_abs_edge, frame["selected_source"], fallback_source)
    frame["gated_code"] = np.where(frame["gated_source"] == "left", frame["left_code"], frame["right_code"])
    frame["gated_pred"] = np.where(frame["gated_source"] == "left", frame["left_pred"], frame["right_pred"])
    frame["gated_true"] = np.where(frame["gated_source"] == "left", frame["left_true"], frame["right_true"])
    frame["gated_switch_from_prev"] = frame["gated_source"].ne(frame["gated_source"].shift(1)).fillna(False)
    if not frame.empty:
        frame.loc[frame.index[0], "gated_switch_from_prev"] = False
    frame["gated_top1_change_vs_left"] = frame["gated_code"] != frame["left_code"]
    frame["gated_top1_change_vs_right"] = frame["gated_code"] != frame["right_code"]
    frame["gated_alpha_vs_left"] = frame["gated_true"] - frame["left_true"]
    frame["gated_alpha_vs_right"] = frame["gated_true"] - frame["right_true"]
    frame["gated_active_selector"] = frame["confidence_edge"].abs() >= min_abs_edge
    return frame


def build_threshold_gated_strategy_frame(daily_audit_frame, fallback_source="right", min_abs_edge=0.0):
    frame = build_threshold_gated_audit_frame(
        daily_audit_frame=daily_audit_frame,
        fallback_source=fallback_source,
        min_abs_edge=min_abs_edge,
    )
    strategy = frame[
        [
            "date",
            "gated_code",
            "gated_pred",
            "gated_true",
            "gated_source",
            "gated_active_selector",
            "confidence_edge",
        ]
    ].copy()
    strategy = strategy.rename(
        columns={
            "gated_code": "code",
            "gated_pred": "pred",
            "gated_true": "true",
            "gated_source": "selected_source",
        }
    )
    return strategy.sort_values("date").reset_index(drop=True)


def _return_stats(daily_returns):
    daily_returns = pd.Series(daily_returns, dtype=np.float64)
    if daily_returns.empty:
        return {"mean_return": 0.0, "cumulative_return": 0.0, "sharpe": 0.0}
    mean_return = float(daily_returns.mean())
    cumulative_return = float((1.0 + daily_returns).prod() - 1.0)
    std_return = float(daily_returns.std(ddof=0)) if len(daily_returns) > 1 else 0.0
    sharpe = 0.0 if std_return == 0 else (mean_return / std_return) * math.sqrt(252)
    return {
        "mean_return": mean_return,
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
    }


def _contribution_metrics(daily_returns):
    series = pd.Series(daily_returns, dtype=np.float64).sort_values(ascending=False).reset_index(drop=True)
    positive = series[series > 0]
    negative = series[series < 0]
    positive_sum = float(positive.sum())
    negative_sum = float(negative.sum())

    def positive_share(top_n):
        if positive.empty or positive_sum == 0.0:
            return 0.0
        return float(positive.head(top_n).sum() / positive_sum)

    def negative_share(bottom_n):
        if negative.empty or negative_sum == 0.0:
            return 0.0
        return float(negative.tail(bottom_n).sum() / negative_sum)

    return {
        "top1_positive_share": positive_share(1),
        "top5_positive_share": positive_share(5),
        "top10_positive_share": positive_share(10),
        "bottom1_negative_share": negative_share(1),
        "bottom5_negative_share": negative_share(5),
        "bottom10_negative_share": negative_share(10),
    }


def summarize_selector_audit(daily_audit_frame):
    frame = daily_audit_frame.sort_values("date").reset_index(drop=True)
    stats = _return_stats(frame["selected_return"])
    metrics = evaluate_prediction_frame(
        frame[["date", "selected_code", "selected_return"]]
        .rename(columns={"selected_code": "code", "selected_return": "true"})
        .assign(pred=frame["selected_return"].to_numpy())
        [["date", "code", "pred", "true"]]
    )
    switch_count = int(frame["switch_from_prev"].sum())
    disagreement_days = int(frame["disagreement"].sum())
    metrics.update(
        {
            "num_days": int(frame.shape[0]),
            "switch_count": switch_count,
            "switch_rate": 0.0 if frame.shape[0] <= 1 else float(switch_count / (frame.shape[0] - 1)),
            "left_usage_rate": float((frame["selected_source"] == "left").mean()) if not frame.empty else 0.0,
            "right_usage_rate": float((frame["selected_source"] == "right").mean()) if not frame.empty else 0.0,
            "disagreement_rate": float(disagreement_days / frame.shape[0]) if not frame.empty else 0.0,
            "top1_change_vs_left_rate": float(frame["top1_change_vs_left"].mean()) if not frame.empty else 0.0,
            "top1_change_vs_right_rate": float(frame["top1_change_vs_right"].mean()) if not frame.empty else 0.0,
            "disagreement_mean_return": float(frame.loc[frame["disagreement"], "selected_return"].mean())
            if disagreement_days
            else 0.0,
            "agreement_mean_return": float(frame.loc[~frame["disagreement"], "selected_return"].mean())
            if (~frame["disagreement"]).any()
            else 0.0,
            "switch_day_mean_return": float(frame.loc[frame["switch_from_prev"], "selected_return"].mean())
            if switch_count
            else 0.0,
            "non_switch_day_mean_return": float(frame.loc[~frame["switch_from_prev"], "selected_return"].mean())
            if (~frame["switch_from_prev"]).any()
            else 0.0,
            "alpha_vs_left_mean": float(frame["selector_alpha_vs_left"].mean()) if not frame.empty else 0.0,
            "alpha_vs_right_mean": float(frame["selector_alpha_vs_right"].mean()) if not frame.empty else 0.0,
            "confidence_edge_mean": float(frame["confidence_edge"].mean()) if not frame.empty else 0.0,
            "confidence_edge_abs_mean": float(frame["confidence_edge"].abs().mean()) if not frame.empty else 0.0,
            **stats,
            **_contribution_metrics(frame["selected_return"]),
        }
    )
    return metrics


def summarize_selector_audit_by_year(daily_audit_frame):
    rows = []
    for fold_year, group in daily_audit_frame.groupby("fold_year"):
        row = {"fold_year": int(fold_year), **summarize_selector_audit(group)}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold_year").reset_index(drop=True)


def summarize_threshold_gated_audit(gated_audit_frame):
    frame = gated_audit_frame.sort_values("date").reset_index(drop=True)
    daily_returns = frame["gated_true"]
    stats = _return_stats(daily_returns)
    switch_count = int(frame["gated_switch_from_prev"].sum())
    num_days = int(frame.shape[0])
    active_rate = float(frame["gated_active_selector"].mean()) if num_days else 0.0
    return {
        "num_days": num_days,
        "switch_count": switch_count,
        "switch_rate": 0.0 if num_days <= 1 else float(switch_count / (num_days - 1)),
        "active_selector_rate": active_rate,
        "fallback_rate": 1.0 - active_rate,
        "left_usage_rate": float((frame["gated_source"] == "left").mean()) if num_days else 0.0,
        "right_usage_rate": float((frame["gated_source"] == "right").mean()) if num_days else 0.0,
        "mean_return": stats["mean_return"],
        "cumulative_return": stats["cumulative_return"],
        "sharpe": stats["sharpe"],
        "alpha_vs_left_mean": float(frame["gated_alpha_vs_left"].mean()) if num_days else 0.0,
        "alpha_vs_right_mean": float(frame["gated_alpha_vs_right"].mean()) if num_days else 0.0,
        "active_day_mean_return": float(frame.loc[frame["gated_active_selector"], "gated_true"].mean())
        if frame["gated_active_selector"].any()
        else 0.0,
        "fallback_day_mean_return": float(frame.loc[~frame["gated_active_selector"], "gated_true"].mean())
        if (~frame["gated_active_selector"]).any()
        else 0.0,
        **_contribution_metrics(daily_returns),
    }
