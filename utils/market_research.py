import math
import os
import tempfile
from contextlib import contextmanager

import fcntl
import numpy as np
import pandas as pd


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
    next_date = frame.groupby("code")["date"].shift(-1)
    next_limit_ratio = infer_price_limit_ratio(frame["code"], next_date)
    next_limit_up_price = frame["close"].to_numpy(dtype=np.float32) * (1.0 + next_limit_ratio)
    can_buy_on_next_open = next_open.to_numpy(dtype=np.float32) < (next_limit_up_price - 1e-6)

    frame["ret_1o"] = next_open / frame["open"] - 1.0
    frame["label"] = next2_open / next_open - 1.0
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
            cached = _read_cached_market_frame(cache_path)
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
    frame = frame.dropna(subset=BASE_FEATURE_COLUMNS + RANK_FEATURE_COLUMNS + ["label", "label_prev_shift"])
    frame = frame.reset_index(drop=True)
    return frame


def _read_cached_market_frame(cache_path):
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
    required_columns = set(BASE_FEATURE_COLUMNS + RANK_FEATURE_COLUMNS + ["can_buy_on_next_open", "label", "label_cls"])
    if not required_columns.issubset(set(cached.columns)):
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
    if feature_set == "B":
        return BASE_FEATURE_COLUMNS + RANK_FEATURE_COLUMNS
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


def make_time_features(dates):
    stamp = pd.DataFrame({"date": pd.to_datetime(dates)})
    stamp["month"] = stamp["date"].dt.month
    stamp["day"] = stamp["date"].dt.day
    stamp["weekday"] = stamp["date"].dt.weekday
    stamp["hour"] = 0
    return stamp[["month", "day", "weekday", "hour"]].to_numpy(dtype=np.float32)


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


def evaluate_prediction_frame(prediction_frame):
    metrics = evaluate_topk_returns(prediction_frame, top_k=1)
    if prediction_frame.empty:
        metrics.update({"rank_ic": 0.0, "ic": 0.0})
        return metrics

    ic_values = []
    rank_ic_values = []
    for _, daily in prediction_frame.groupby("date"):
        if daily["pred"].nunique() <= 1 or daily["true"].nunique() <= 1:
            continue
        ic_values.append(daily["pred"].corr(daily["true"], method="pearson"))
        rank_ic_values.append(daily["pred"].corr(daily["true"], method="spearman"))

    metrics["ic"] = float(np.nanmean(ic_values)) if ic_values else 0.0
    metrics["rank_ic"] = float(np.nanmean(rank_ic_values)) if rank_ic_values else 0.0
    return metrics


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
