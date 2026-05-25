import math
import os
import tempfile
from contextlib import contextmanager

import fcntl
import numpy as np
import pandas as pd

from utils.timefeatures import time_features


BASE_FEATURE_COLUMNS = [
    "co",
    "ho",
    "lo",
    "cc",
    "oo",
    "log_volume",
    "log_volume_diff",
    "log_amount",
    "log_amount_diff",
    "avg_price",
    "turnover_rate",
    "amplitude",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "avg_turnover_5",
    "avg_turnover_20",
    "avg_amplitude_5",
    "avg_amplitude_20",
]

RANK_FEATURE_COLUMNS = [
    "co_rank",
    "cc_rank",
    "turnover_rate_rank",
    "amplitude_rank",
    "ret_20_rank",
    "vol_20_rank",
    "log_amount_rank",
]

MARKET_CONTEXT_COLUMNS = [
    "market_cc_mean",
    "market_cc_std",
    "market_turnover_mean",
    "market_amplitude_mean",
    "market_ret_20_mean",
    "market_vol_20_mean",
]

MARKET_STRUCTURE_COLUMNS = [
    "market_co_mean",
    "market_co_std",
    "market_cc_q25",
    "market_cc_median",
    "market_cc_q75",
    "market_turnover_q25",
    "market_turnover_median",
    "market_turnover_q75",
    "market_amplitude_q25",
    "market_amplitude_median",
    "market_amplitude_q75",
    "market_ret_20_q25",
    "market_ret_20_median",
    "market_ret_20_q75",
    "market_vol_20_q25",
    "market_vol_20_median",
    "market_vol_20_q75",
    "market_up_ratio",
    "market_strong_up_ratio",
    "market_amount_top10_share",
    "market_amount_hhi",
]

RELATIVE_MARKET_COLUMNS = [
    "co_vs_market",
    "cc_vs_market",
    "turnover_vs_market",
    "amplitude_vs_market",
    "ret_20_vs_market",
    "vol_20_vs_market",
    "log_amount_vs_market",
]

TRAIN_TARGET_COLUMN_MAP = {
    "1": "label",
    "3": "label_close_3d",
    "5": "label_close_5d",
}


def build_rolling_folds(start_year=2010, end_year=2019, train_years=5):
    folds = []
    for anchor_year in range(start_year, end_year + 1):
        val_year = anchor_year + train_years - 1
        test_year = anchor_year + train_years
        folds.append(
            {
                "fold": str(test_year),
                "train_start": f"{anchor_year:04d}-01-01",
                "train_end": f"{val_year - 1:04d}-12-31",
                "val_start": f"{val_year:04d}-01-01",
                "val_end": f"{val_year:04d}-12-31",
                "test_start": f"{test_year:04d}-01-01",
                "test_end": f"{test_year:04d}-12-31",
            }
        )
    return folds


def infer_price_limit_ratio(codes, dates):
    code_series = pd.Series(codes, copy=False).astype(str)
    date_series = pd.to_datetime(pd.Series(dates, copy=False))

    ratios = np.full(len(code_series), 0.10, dtype=np.float32)

    # STAR Market
    ratios[code_series.str.startswith("SH688").to_numpy()] = 0.20

    # ChiNext switched from 10% to 20% on 2020-08-24.
    chinext_mask = code_series.str.startswith("SZ300").to_numpy()
    chinext_20_mask = chinext_mask & (date_series.to_numpy() >= np.datetime64("2020-08-24"))
    ratios[chinext_20_mask] = 0.20

    # Beijing exchange and legacy NEEQ-style prefixes.
    beijing_mask = code_series.str.startswith("BJ").to_numpy()
    ratios[beijing_mask] = 0.30
    return ratios


def add_label_columns(df):
    frame = df.copy()
    frame = frame.sort_values(["code", "date"]).reset_index(drop=True)

    next_open = frame.groupby("code")["open"].shift(-1)
    next2_open = frame.groupby("code")["open"].shift(-2)
    next_close = frame.groupby("code")["close"].shift(-1)
    next3_close = frame.groupby("code")["close"].shift(-3)
    next5_close = frame.groupby("code")["close"].shift(-5)
    next_date = frame.groupby("code")["date"].shift(-1)
    next_limit_ratio = infer_price_limit_ratio(frame["code"], next_date)
    next_limit_up_price = frame["close"].to_numpy(dtype=np.float32) * (1.0 + next_limit_ratio)
    can_buy_on_next_open = next_open.to_numpy(dtype=np.float32) < (next_limit_up_price - 1e-6)

    frame["ret_1o"] = next_open / frame["open"] - 1.0
    frame["label"] = next2_open / next_open - 1.0
    frame["label_close_1d"] = next_close / frame["close"] - 1.0
    frame["label_close_3d"] = next3_close / frame["close"] - 1.0
    frame["label_close_5d"] = next5_close / frame["close"] - 1.0
    frame["label_cls"] = (frame["label"] > 0).astype("Int64")
    frame["label_prev_shift"] = frame.groupby("code")["label"].shift(1)
    frame["can_buy_on_next_open"] = pd.Series(can_buy_on_next_open, index=frame.index).fillna(False)
    return frame


def add_feature_columns(df):
    frame = df.copy()
    frame = frame.sort_values(["code", "date"]).reset_index(drop=True)

    grouped = frame.groupby("code", group_keys=False)
    prev_close = grouped["close"].shift(1)
    prev_open = grouped["open"].shift(1)

    frame["co"] = frame["close"] / frame["open"] - 1.0
    frame["ho"] = frame["high"] / frame["open"] - 1.0
    frame["lo"] = frame["low"] / frame["open"] - 1.0
    frame["cc"] = frame["close"] / prev_close - 1.0
    frame["oo"] = frame["open"] / prev_open - 1.0

    frame["log_volume"] = np.log1p(frame["volume"])
    frame["log_amount"] = np.log1p(frame["amount"])
    frame["log_volume_diff"] = grouped["log_volume"].diff()
    frame["log_amount_diff"] = grouped["log_amount"].diff()
    frame["avg_price"] = np.divide(
        frame["amount"].to_numpy(dtype=np.float32),
        frame["volume"].to_numpy(dtype=np.float32),
        out=np.full(len(frame), np.nan, dtype=np.float32),
        where=frame["volume"].to_numpy(dtype=np.float32) > 0,
    )

    for window in (5, 10, 20, 60):
        frame[f"ret_{window}"] = grouped["close"].pct_change(window)
        frame[f"vol_{window}"] = grouped["cc"].rolling(window).std().reset_index(level=0, drop=True)

    frame["avg_turnover_5"] = grouped["turnover_rate"].rolling(5).mean().reset_index(level=0, drop=True)
    frame["avg_turnover_20"] = grouped["turnover_rate"].rolling(20).mean().reset_index(level=0, drop=True)
    frame["avg_amplitude_5"] = grouped["amplitude"].rolling(5).mean().reset_index(level=0, drop=True)
    frame["avg_amplitude_20"] = grouped["amplitude"].rolling(20).mean().reset_index(level=0, drop=True)
    frame["listing_days"] = grouped.cumcount() + 1

    cross_section_bases = {
        "co_rank": "co",
        "cc_rank": "cc",
        "turnover_rate_rank": "turnover_rate",
        "amplitude_rank": "amplitude",
        "ret_20_rank": "ret_20",
        "vol_20_rank": "vol_20",
        "log_amount_rank": "log_amount",
    }
    for rank_name, source_name in cross_section_bases.items():
        frame[rank_name] = frame.groupby("date")[source_name].rank(pct=True)

    daily_groups = frame.groupby("date", sort=False)
    daily_summary = daily_groups.agg(
        market_co_mean=("co", "mean"),
        market_co_std=("co", "std"),
        market_cc_mean=("cc", "mean"),
        market_cc_std=("cc", "std"),
        market_turnover_mean=("turnover_rate", "mean"),
        market_amplitude_mean=("amplitude", "mean"),
        market_ret_20_mean=("ret_20", "mean"),
        market_vol_20_mean=("vol_20", "mean"),
        market_cc_q25=("cc", lambda s: s.quantile(0.25)),
        market_cc_median=("cc", "median"),
        market_cc_q75=("cc", lambda s: s.quantile(0.75)),
        market_turnover_q25=("turnover_rate", lambda s: s.quantile(0.25)),
        market_turnover_median=("turnover_rate", "median"),
        market_turnover_q75=("turnover_rate", lambda s: s.quantile(0.75)),
        market_amplitude_q25=("amplitude", lambda s: s.quantile(0.25)),
        market_amplitude_median=("amplitude", "median"),
        market_amplitude_q75=("amplitude", lambda s: s.quantile(0.75)),
        market_ret_20_q25=("ret_20", lambda s: s.quantile(0.25)),
        market_ret_20_median=("ret_20", "median"),
        market_ret_20_q75=("ret_20", lambda s: s.quantile(0.75)),
        market_vol_20_q25=("vol_20", lambda s: s.quantile(0.25)),
        market_vol_20_median=("vol_20", "median"),
        market_vol_20_q75=("vol_20", lambda s: s.quantile(0.75)),
        market_up_ratio=("cc", lambda s: (s > 0).mean()),
        market_strong_up_ratio=("cc", lambda s: (s > 0.02).mean()),
        market_amount_hhi=("amount", lambda s: np.square(s / s.sum()).sum() if s.sum() > 0 else 0.0),
        market_amount_top10_share=("amount", lambda s: s.nlargest(max(1, int(math.ceil(len(s) * 0.1)))).sum() / s.sum() if s.sum() > 0 else 0.0),
    ).reset_index()
    frame = frame.merge(daily_summary, on="date", how="left")

    for col in ("market_co_std", "market_cc_std"):
        frame[col] = frame[col].fillna(0.0)

    frame["co_vs_market"] = frame["co"] - frame["market_co_mean"]
    frame["cc_vs_market"] = frame["cc"] - frame["market_cc_mean"]
    frame["turnover_vs_market"] = frame["turnover_rate"] - frame["market_turnover_mean"]
    frame["amplitude_vs_market"] = frame["amplitude"] - frame["market_amplitude_mean"]
    frame["ret_20_vs_market"] = frame["ret_20"] - frame["market_ret_20_mean"]
    frame["vol_20_vs_market"] = frame["vol_20"] - frame["market_vol_20_mean"]
    frame["log_amount_vs_market"] = frame["log_amount"] - daily_groups["log_amount"].transform("mean")

    return frame


def prepare_market_dataframe(
    parquet_path,
    start_date="2010-01-01",
    min_history=120,
    min_avg_amount=2e7,
    cache_path=None,
):
    if cache_path:
        with _cache_file_lock(cache_path):
            cached = _read_cached_market_frame(cache_path, required_start_date=start_date)
            if cached is not None:
                return cached

            frame = _build_market_dataframe(
                parquet_path=parquet_path,
                start_date=start_date,
                min_history=min_history,
                min_avg_amount=min_avg_amount,
            )
            _write_parquet_atomic(frame, cache_path)
            return frame

    return _build_market_dataframe(
        parquet_path=parquet_path,
        start_date=start_date,
        min_history=min_history,
        min_avg_amount=min_avg_amount,
    )


def _build_market_dataframe(parquet_path, start_date, min_history, min_avg_amount):
    frame = pd.read_parquet(parquet_path)
    frame = frame.sort_values(["code", "date"]).reset_index(drop=True)
    frame = frame[frame["date"] >= start_date].copy()
    for col in ("open", "close", "high", "low"):
        frame = frame[frame[col] > 0]
    frame = frame.dropna(
        subset=[
            "code",
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "turnover_rate",
            "amplitude",
        ]
    )

    frame = add_label_columns(frame)
    frame = add_feature_columns(frame)
    frame["avg_amount_20"] = (
        frame.groupby("code")["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    )

    frame = frame[frame["listing_days"] >= min_history]
    frame = frame[frame["avg_amount_20"] >= min_avg_amount]
    frame["label_cs_rank"] = frame.groupby("date")["label"].rank(method="average", pct=True)
    frame = frame.dropna(
        subset=(
            BASE_FEATURE_COLUMNS
            + RANK_FEATURE_COLUMNS
            + MARKET_CONTEXT_COLUMNS
            + MARKET_STRUCTURE_COLUMNS
            + RELATIVE_MARKET_COLUMNS
            + ["label", "label_prev_shift", "label_close_1d", "label_close_3d", "label_close_5d"]
        )
    )
    frame = frame.reset_index(drop=True)
    return frame


def _read_cached_market_frame(cache_path, required_start_date=None):
    if not os.path.exists(cache_path):
        return None
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None
    required_columns = set(
        BASE_FEATURE_COLUMNS
        + RANK_FEATURE_COLUMNS
        + MARKET_CONTEXT_COLUMNS
        + MARKET_STRUCTURE_COLUMNS
        + RELATIVE_MARKET_COLUMNS
        + [
            "can_buy_on_next_open",
            "label",
            "label_cls",
            "label_cs_rank",
            "label_close_1d",
            "label_close_3d",
            "label_close_5d",
        ]
    )
    if not required_columns.issubset(set(cached.columns)):
        return None
    if required_start_date:
        cached_min_date = pd.to_datetime(cached["date"]).min()
        if pd.isna(cached_min_date) or cached_min_date > pd.Timestamp(required_start_date):
            return None
    return cached


def _write_parquet_atomic(frame, cache_path):
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_market_cache_", suffix=".parquet", dir=cache_dir)
    os.close(fd)
    try:
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@contextmanager
def _cache_file_lock(cache_path):
    lock_path = f"{cache_path}.lock"
    lock_dir = os.path.dirname(lock_path) or "."
    os.makedirs(lock_dir, exist_ok=True)
    with open(lock_path, "w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def get_feature_columns(feature_set="A"):
    if feature_set == "A":
        return BASE_FEATURE_COLUMNS
    if feature_set == "A_CTX":
        return BASE_FEATURE_COLUMNS + MARKET_CONTEXT_COLUMNS
    if feature_set == "B":
        return BASE_FEATURE_COLUMNS + RANK_FEATURE_COLUMNS
    if feature_set == "B_CTX":
        return BASE_FEATURE_COLUMNS + RANK_FEATURE_COLUMNS + MARKET_CONTEXT_COLUMNS
    if feature_set == "B_MKT":
        return (
            BASE_FEATURE_COLUMNS
            + RANK_FEATURE_COLUMNS
            + MARKET_CONTEXT_COLUMNS
            + MARKET_STRUCTURE_COLUMNS
            + RELATIVE_MARKET_COLUMNS
        )
    if feature_set == "C":
        return [
            "co",
            "cc",
            "oo",
            "log_amount",
            "log_amount_diff",
            "turnover_rate",
            "amplitude",
            "ret_10",
            "ret_20",
            "ret_60",
            "vol_10",
            "vol_20",
            "vol_60",
            "avg_turnover_20",
            "avg_amplitude_20",
        ] + RANK_FEATURE_COLUMNS
    if feature_set == "D":
        return [
            "co",
            "cc",
            "oo",
            "turnover_rate",
            "amplitude",
            "ret_5",
            "ret_20",
            "vol_5",
            "vol_20",
            "avg_turnover_20",
        ]
    raise ValueError(f"Unsupported feature set: {feature_set}")


def get_train_target_columns(horizon_spec="1,3,5"):
    horizons = [item.strip() for item in str(horizon_spec).split(",") if item.strip()]
    columns = []
    for horizon in horizons:
        if horizon not in TRAIN_TARGET_COLUMN_MAP:
            raise ValueError(f"Unsupported train horizon: {horizon}")
        columns.append(TRAIN_TARGET_COLUMN_MAP[horizon])
    if not columns:
        raise ValueError("At least one market train horizon must be specified")
    return columns


def make_time_features(dates, freq="d", embed_type="timeF"):
    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    if embed_type == "timeF":
        return time_features(date_index, freq=freq).transpose(1, 0).astype(np.float32)

    stamp = pd.DataFrame({"date": date_index})
    stamp["month"] = stamp["date"].dt.month
    stamp["day"] = stamp["date"].dt.day
    stamp["weekday"] = stamp["date"].dt.weekday
    stamp["hour"] = stamp["date"].dt.hour if freq in {"h", "t", "s"} else 0
    columns = ["month", "day", "weekday", "hour"]
    if freq in {"t", "s"}:
        stamp["minute"] = stamp["date"].dt.minute
        columns.append("minute")
    return stamp[columns].to_numpy(dtype=np.float32)


def apply_static_score_debias(prediction_frame, method="none", strength=1.0):
    frame = prediction_frame.copy()
    if frame.empty or method in {None, "", "none"}:
        return frame

    if method != "expanding_mean":
        raise ValueError(f"Unsupported static score debias method: {method}")

    frame = frame.sort_values(["date", "code"]).reset_index(drop=True)
    frame["pred_raw"] = frame["pred"].astype(np.float32)
    prior_mean = (
        frame.groupby("code")["pred_raw"]
        .transform(lambda s: s.expanding().mean().shift(1))
        .fillna(0.0)
        .astype(np.float32)
    )
    frame["pred_static_bias"] = prior_mean
    frame["pred"] = frame["pred_raw"] - float(strength) * frame["pred_static_bias"]
    return frame


def evaluate_topk_returns(prediction_frame, top_k=1):
    if prediction_frame.empty:
        return {
            "num_days": 0,
            "mean_return": 0.0,
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "top_picks": [],
        }

    if "tradable" in prediction_frame.columns:
        prediction_frame = prediction_frame[prediction_frame["tradable"]].copy()
        if prediction_frame.empty:
            return {
                "num_days": 0,
                "mean_return": 0.0,
                "cumulative_return": 0.0,
                "sharpe": 0.0,
                "top_picks": [],
            }

    ranked = prediction_frame.sort_values(["date", "pred"], ascending=[True, False])
    top = ranked.groupby("date").head(top_k).copy()
    daily = top.groupby("date")["true"].mean()
    mean_return = float(daily.mean())
    std_return = float(daily.std(ddof=0)) if len(daily) > 1 else 0.0
    sharpe = 0.0 if std_return == 0 else (mean_return / std_return) * math.sqrt(252)
    cumulative_return = float((1.0 + daily).prod() - 1.0)
    top_picks = list(top[["date", "code"]].itertuples(index=False, name=None))

    return {
        "num_days": int(daily.shape[0]),
        "mean_return": mean_return,
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "top_picks": top_picks,
    }


def evaluate_prediction_frame(prediction_frame, topk_list=(1,), score_debias="none", score_debias_strength=1.0):
    scored_frame = apply_static_score_debias(
        prediction_frame,
        method=score_debias,
        strength=score_debias_strength,
    )
    metrics = evaluate_topk_returns(scored_frame, top_k=1)
    if prediction_frame.empty:
        metrics.update({"rank_ic": 0.0, "ic": 0.0})
        return metrics

    ic_values = []
    rank_ic_values = []
    for _, daily in scored_frame.groupby("date"):
        if daily["pred"].nunique() <= 1 or daily["true"].nunique() <= 1:
            continue
        ic_values.append(daily["pred"].corr(daily["true"], method="pearson"))
        rank_ic_values.append(daily["pred"].corr(daily["true"], method="spearman"))

    metrics["ic"] = float(np.nanmean(ic_values)) if ic_values else 0.0
    metrics["rank_ic"] = float(np.nanmean(rank_ic_values)) if rank_ic_values else 0.0
    metrics["score_debias"] = score_debias

    for top_k in tuple(topk_list):
        basket_metrics = evaluate_topk_returns(scored_frame, top_k=top_k)
        metrics[f"top{top_k}_mean_return"] = basket_metrics["mean_return"]
        metrics[f"top{top_k}_cumulative_return"] = basket_metrics["cumulative_return"]
        metrics[f"top{top_k}_sharpe"] = basket_metrics["sharpe"]
        metrics[f"top{top_k}_num_days"] = basket_metrics["num_days"]

    return metrics


def _build_score_bucket_rows(scored_frame, bucket_count=10):
    if scored_frame.empty:
        return []

    bucket_frames = []
    for _, daily in scored_frame.groupby("date", sort=False):
        current = daily.copy()
        if current.empty:
            continue
        rank_pct = current["pred"].rank(method="first", pct=True)
        bucket_ids = np.ceil(rank_pct * int(bucket_count)).clip(1, int(bucket_count)).astype(int)
        current["score_bucket"] = bucket_ids
        bucket_frames.append(current)

    if not bucket_frames:
        return []

    combined = pd.concat(bucket_frames, axis=0, ignore_index=True)
    rows = []
    for bucket_id, group in combined.groupby("score_bucket", sort=True):
        rows.append(
            {
                "bucket": int(bucket_id),
                "count": int(group.shape[0]),
                "mean_pred": float(group["pred"].mean()),
                "mean_true": float(group["true"].mean()),
                "positive_share": float((group["true"] > 0).mean()),
            }
        )
    return rows


def _build_top_repeat_rows(metrics, prediction_frame, top_n=20):
    picks = metrics.get("top_picks", [])
    if not picks:
        return []

    pick_frame = pd.DataFrame(picks, columns=["date", "code"])
    counts = (
        pick_frame.groupby("code")
        .size()
        .reset_index(name="pick_count")
        .sort_values(["pick_count", "code"], ascending=[False, True])
        .head(int(top_n))
    )

    ranked = prediction_frame.sort_values(["date", "pred"], ascending=[True, False]).copy()
    top1 = ranked.groupby("date").head(1)[["date", "code", "pred", "true"]]
    merged = counts.merge(top1, on="code", how="left")
    rows = []
    for code, group in merged.groupby("code", sort=False):
        rows.append(
            {
                "code": str(code),
                "pick_count": int(group["pick_count"].iloc[0]),
                "mean_true": float(group["true"].mean()),
                "mean_pred": float(group["pred"].mean()),
            }
        )
    return rows


def _build_market_slice_summary(scored_frame, column_name):
    if column_name not in scored_frame.columns or scored_frame.empty:
        return {}

    daily_view = (
        scored_frame.groupby("date", sort=False)
        .agg(
            market_value=(column_name, "mean"),
            top1_return=("true", lambda s: np.nan),
        )
        .reset_index()
    )
    top1_frame = (
        scored_frame.sort_values(["date", "pred"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(1)[["date", "true"]]
        .rename(columns={"true": "top1_return"})
    )
    daily_view = daily_view.drop(columns=["top1_return"]).merge(top1_frame, on="date", how="left")
    if daily_view.empty:
        return {}

    median_value = float(daily_view["market_value"].median())
    low = daily_view[daily_view["market_value"] <= median_value]
    high = daily_view[daily_view["market_value"] > median_value]

    def _slice_stats(frame):
        if frame.empty:
            return {"num_days": 0, "top1_mean_return": 0.0}
        return {
            "num_days": int(frame.shape[0]),
            "top1_mean_return": float(frame["top1_return"].mean()),
        }

    return {
        "median": median_value,
        "low": _slice_stats(low),
        "high": _slice_stats(high),
    }


def build_market_diagnostics(
    prediction_frame,
    topk_list=(1, 3, 5, 10, 20),
    score_debias="none",
    score_debias_strength=1.0,
    score_bucket_count=10,
    high_repeat_top_n=20,
    market_slice_columns=("market_cc_mean", "market_amount_top10_share"),
):
    scored_frame = apply_static_score_debias(
        prediction_frame,
        method=score_debias,
        strength=score_debias_strength,
    )
    summary = evaluate_prediction_frame(
        scored_frame,
        topk_list=topk_list,
        score_debias="none",
        score_debias_strength=1.0,
    )
    top_codes = [code for _, code in summary.get("top_picks", [])]
    summary["top_pick_unique"] = len(set(top_codes))
    summary["top_pick_max_rep"] = max((top_codes.count(code) for code in set(top_codes)), default=0)

    market_slices = {}
    for column_name in market_slice_columns:
        slice_summary = _build_market_slice_summary(scored_frame, column_name)
        if slice_summary:
            market_slices[column_name] = slice_summary

    return {
        "summary": summary,
        "score_buckets": _build_score_bucket_rows(scored_frame, bucket_count=score_bucket_count),
        "top_repeated_picks": _build_top_repeat_rows(summary, scored_frame, top_n=high_repeat_top_n),
        "market_slices": market_slices,
    }


def build_head_candidate_diagnostics(
    prediction_frame,
    pred_topk_list=(10, 20, 50),
    true_topk_list=(10, 20, 50),
    score_debias="none",
    score_debias_strength=1.0,
):
    scored_frame = apply_static_score_debias(
        prediction_frame,
        method=score_debias,
        strength=score_debias_strength,
    )
    if scored_frame.empty:
        return {"summary": {}, "daily": []}

    pred_topk_list = tuple(int(k) for k in pred_topk_list)
    true_topk_list = tuple(int(k) for k in true_topk_list)

    daily_rows = []
    for date, daily in scored_frame.groupby("date", sort=True):
        pred_ranked = daily.sort_values("pred", ascending=False).reset_index(drop=True)
        tradable_ranked = pred_ranked[pred_ranked["tradable"]].reset_index(drop=True) if "tradable" in pred_ranked.columns else pred_ranked
        if tradable_ranked.empty:
            continue

        true_ranked = daily.sort_values("true", ascending=False).reset_index(drop=True)
        tradable_true_ranked = tradable_ranked.sort_values("true", ascending=False).reset_index(drop=True)

        top1 = tradable_ranked.iloc[0]
        top2 = tradable_ranked.iloc[1] if tradable_ranked.shape[0] > 1 else None
        x = pred_ranked["pred"].to_numpy(dtype=np.float64)
        x = x - np.nanmax(x)
        weights = np.exp(x)
        weights = weights / max(weights.sum(), 1e-12)
        row = {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "n": int(pred_ranked.shape[0]),
            "n_tradable": int(tradable_ranked.shape[0]),
            "top1_code": str(top1["code"]),
            "top1_true": float(top1["true"]),
            "top1_pred_gap": float(top1["pred"] - top2["pred"]) if top2 is not None else np.nan,
            "best_true_all": float(true_ranked.iloc[0]["true"]),
            "best_true_tradable": float(tradable_true_ranked.iloc[0]["true"]),
            "regret_vs_best_tradable": float(tradable_true_ranked.iloc[0]["true"] - top1["true"]),
            "top3_mean": float(tradable_ranked.head(3)["true"].mean()),
            "top5_mean": float(tradable_ranked.head(5)["true"].mean()),
            "score_std": float(pred_ranked["pred"].std(ddof=0)),
            "score_max_prob": float(weights.max()),
            "score_entropy": float(-(weights * np.log(weights + 1e-12)).sum()),
        }

        for true_k in true_topk_list:
            true_codes = set(true_ranked.head(true_k)["code"])
            row[f"hit_top1_in_true{true_k}"] = int(top1["code"] in true_codes)

        for pred_k in pred_topk_list:
            pred_codes = set(pred_ranked.head(pred_k)["code"])
            for true_k in true_topk_list:
                true_codes = set(true_ranked.head(true_k)["code"])
                row[f"overlap_pred{pred_k}_true{true_k}"] = int(len(pred_codes & true_codes))

        daily_rows.append(row)

    daily_frame = pd.DataFrame(daily_rows)
    raw_top1 = evaluate_prediction_frame(scored_frame, topk_list=(1,), score_debias="none", score_debias_strength=1.0)
    debias_top1 = evaluate_prediction_frame(
        prediction_frame,
        topk_list=(1,),
        score_debias="expanding_mean",
        score_debias_strength=0.15,
    )
    top_pick_counts = []
    if not daily_frame.empty:
        counts = (
            daily_frame.groupby("top1_code")
            .size()
            .reset_index(name="pick_count")
            .sort_values(["pick_count", "top1_code"], ascending=[False, True])
        )
        top_pick_counts = [
            {"code": str(row["top1_code"]), "pick_count": int(row["pick_count"])}
            for _, row in counts.head(20).iterrows()
        ]
        unique_top1 = int(counts.shape[0])
        max_rep_top1 = int(counts["pick_count"].max())
    else:
        unique_top1 = 0
        max_rep_top1 = 0

    summary = {
        "days": int(daily_frame.shape[0]),
        "top1_mean": float(daily_frame["top1_true"].mean()) if not daily_frame.empty else 0.0,
        "top1_std": float(daily_frame["top1_true"].std(ddof=0)) if daily_frame.shape[0] > 1 else 0.0,
        "top1_positive_rate": float((daily_frame["top1_true"] > 0).mean()) if not daily_frame.empty else 0.0,
        "top3_mean": float(daily_frame["top3_mean"].mean()) if not daily_frame.empty else 0.0,
        "top5_mean": float(daily_frame["top5_mean"].mean()) if not daily_frame.empty else 0.0,
        "avg_regret": float(daily_frame["regret_vs_best_tradable"].mean()) if not daily_frame.empty else 0.0,
        "median_regret": float(daily_frame["regret_vs_best_tradable"].median()) if not daily_frame.empty else 0.0,
        "avg_gap": float(daily_frame["top1_pred_gap"].dropna().mean()) if not daily_frame.empty else 0.0,
        "avg_score_std": float(daily_frame["score_std"].mean()) if not daily_frame.empty else 0.0,
        "avg_max_prob": float(daily_frame["score_max_prob"].mean()) if not daily_frame.empty else 0.0,
        "avg_entropy": float(daily_frame["score_entropy"].mean()) if not daily_frame.empty else 0.0,
        "unique_top1": unique_top1,
        "max_rep_top1": max_rep_top1,
        "top_pick_counts": top_pick_counts,
        "raw_top1_mean": float(raw_top1["top1_mean_return"]),
        "raw_top1_sharpe": float(raw_top1["top1_sharpe"]),
        "debias015_top1_mean": float(debias_top1["top1_mean_return"]),
        "debias015_top1_sharpe": float(debias_top1["top1_sharpe"]),
    }

    for true_k in true_topk_list:
        column = f"hit_top1_in_true{true_k}"
        summary[f"hit_true{true_k}_rate"] = float(daily_frame[column].mean()) if not daily_frame.empty else 0.0

    for pred_k in pred_topk_list:
        for true_k in true_topk_list:
            column = f"overlap_pred{pred_k}_true{true_k}"
            summary[f"avg_overlap_pred{pred_k}_true{true_k}"] = float(daily_frame[column].mean()) if not daily_frame.empty else 0.0

    return {
        "summary": summary,
        "daily": daily_rows,
    }


def evaluate_prediction_file(prediction_path, topk_list=(1,), score_debias="none", score_debias_strength=1.0):
    frame = pd.read_csv(prediction_path, parse_dates=["date"])
    return evaluate_prediction_frame(
        frame,
        topk_list=topk_list,
        score_debias=score_debias,
        score_debias_strength=score_debias_strength,
    )


def combine_prediction_frames(prediction_frames, method="rank_mean", weights=None):
    if not prediction_frames:
        raise ValueError("prediction_frames must not be empty")

    normalized = []
    for idx, frame in enumerate(prediction_frames):
        keep_columns = ["date", "code", "pred", "true"]
        if idx == 0 and "tradable" in frame.columns:
            keep_columns.append("tradable")
        current = frame[keep_columns].copy()
        current = current.rename(columns={"pred": f"pred_{idx}"})
        normalized.append(current)

    merged = normalized[0]
    for current in normalized[1:]:
        merged = merged.merge(current, on=["date", "code", "true"], how="inner")

    score_columns = [col for col in merged.columns if col.startswith("pred_")]
    if weights is None:
        weights = np.ones(len(score_columns), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != len(score_columns):
            raise ValueError("weights length must match number of prediction frames")

    if method == "rank_mean":
        transformed = []
        for column in score_columns:
            ranked = merged.groupby("date")[column].rank(method="average", pct=True)
            transformed.append(ranked.to_numpy(dtype=np.float64))
        score_matrix = np.vstack(transformed).T
    elif method == "zscore_mean":
        transformed = []
        for column in score_columns:
            standardized = merged.groupby("date")[column].transform(
                lambda x: ((x - x.mean()) / x.std(ddof=0)).fillna(0.0)
            )
            transformed.append(standardized.to_numpy(dtype=np.float64))
        score_matrix = np.vstack(transformed).T
    elif method == "mean":
        score_matrix = merged[score_columns].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")

    weights = weights / weights.sum()
    merged["pred"] = score_matrix.dot(weights)
    output_columns = ["date", "code", "pred", "true"]
    if "tradable" in merged.columns:
        output_columns.append("tradable")
    return merged[output_columns].copy()
