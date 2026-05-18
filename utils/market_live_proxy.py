import math

import pandas as pd


def build_daily_top1_strategy_frame(prediction_frame):
    frame = prediction_frame.copy()
    if "tradable" in frame.columns:
        frame = frame[frame["tradable"]].copy()
    ranked = frame.sort_values(["date", "pred"], ascending=[True, False]).copy()
    top1 = ranked.groupby("date").head(1).copy()
    return top1[["date", "code", "pred", "true"]].sort_values("date").reset_index(drop=True)


def apply_live_trading_proxy(daily_strategy_frame, buy_cost_bps=0.0, sell_cost_bps=0.0):
    frame = daily_strategy_frame.sort_values("date").reset_index(drop=True).copy()
    if frame.empty:
        frame["entry_cost_bps"] = []
        frame["exit_cost_bps"] = []
        frame["net_return"] = []
        frame["gross_return"] = []
        frame["prev_code"] = []
        frame["next_code"] = []
        frame["switch_in"] = []
        frame["switch_out"] = []
        return frame

    frame["gross_return"] = frame["true"].astype(float)
    frame["prev_code"] = frame["code"].shift(1)
    frame["next_code"] = frame["code"].shift(-1)
    frame["switch_in"] = frame["code"].ne(frame["prev_code"]).fillna(True)
    frame["switch_out"] = frame["code"].ne(frame["next_code"]).fillna(True)
    frame["entry_cost_bps"] = frame["switch_in"].astype(float) * float(buy_cost_bps)
    frame["exit_cost_bps"] = frame["switch_out"].astype(float) * float(sell_cost_bps)

    entry_rate = frame["entry_cost_bps"] / 10000.0
    exit_rate = frame["exit_cost_bps"] / 10000.0
    frame["net_return"] = ((1.0 - entry_rate) * (1.0 + frame["gross_return"]) * (1.0 - exit_rate)) - 1.0
    return frame


def summarize_live_proxy(proxy_frame):
    frame = proxy_frame.sort_values("date").reset_index(drop=True)
    daily_returns = frame["net_return"] if "net_return" in frame.columns else pd.Series(dtype=float)
    if daily_returns.empty:
        return {
            "num_days": 0,
            "mean_return": 0.0,
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "switch_count": 0,
            "same_code_streak_days": 0,
            "avg_entry_cost_bps": 0.0,
            "avg_exit_cost_bps": 0.0,
        }

    mean_return = float(daily_returns.mean())
    cumulative_return = float((1.0 + daily_returns).prod() - 1.0)
    std_return = float(daily_returns.std(ddof=0)) if len(daily_returns) > 1 else 0.0
    sharpe = 0.0 if std_return == 0 else (mean_return / std_return) * math.sqrt(252)
    switched = frame["code"].ne(frame["code"].shift(1)).fillna(False)
    if not frame.empty:
        switched.iloc[0] = False
    switch_count = int(switched.sum())
    same_code_streak_days = int((frame["code"] == frame["prev_code"]).fillna(False).sum())
    return {
        "num_days": int(frame.shape[0]),
        "mean_return": mean_return,
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "switch_count": switch_count,
        "same_code_streak_days": same_code_streak_days,
        "avg_entry_cost_bps": float(frame["entry_cost_bps"].mean()),
        "avg_exit_cost_bps": float(frame["exit_cost_bps"].mean()),
    }
