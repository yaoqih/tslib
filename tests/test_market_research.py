import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from data_provider.data_loader import Dataset_MarketDaily
from models.TimeMixer import Model as TimeMixerModel
from utils.market_research import (
    add_label_columns,
    build_rolling_folds,
    combine_prediction_frames,
    evaluate_topk_returns,
    prepare_market_dataframe,
)
from utils.market_multitask import combine_market_multitask_losses, compute_pairwise_rank_loss
from utils.market_live_proxy import (
    apply_live_trading_proxy,
    build_daily_top1_strategy_frame,
    summarize_live_proxy,
)
from utils.market_selector_audit import (
    build_selector_audit_frame,
    build_threshold_gated_strategy_frame,
    summarize_selector_audit,
)
from scripts.market_daily.evaluate_ensembles import build_confidence_selector_frame


class TestMarketResearch(unittest.TestCase):
    def test_timemixer_multiscale_marks_match_downsampled_lengths(self):
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=20,
            label_len=0,
            pred_len=1,
            down_sampling_window=2,
            down_sampling_layers=3,
            channel_independence=0,
            e_layers=2,
            moving_avg=25,
            enc_in=24,
            c_out=24,
            d_model=16,
            d_ff=32,
            dropout=0.1,
            embed="fixed",
            freq="d",
            use_norm=1,
            decomp_method="moving_avg",
            top_k=5,
            down_sampling_method="avg",
        )
        model = TimeMixerModel(configs)
        x_enc = torch.randn(2, 20, 24)
        x_mark_enc = torch.randint(0, 3, (2, 20, 3))

        x_scales, x_mark_scales = model._Model__multi_scale_process_inputs(x_enc, x_mark_enc)

        self.assertEqual([x.shape[1] for x in x_scales], [20, 10, 5, 2])
        self.assertEqual([x_mark.shape[1] for x_mark in x_mark_scales], [20, 10, 5, 2])

    def test_compute_pairwise_rank_loss_rewards_correct_ordering(self):
        pred = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32)
        target = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float32)

        good_loss = compute_pairwise_rank_loss(pred, target, margin=0.0)
        bad_loss = compute_pairwise_rank_loss(-pred, target, margin=0.0)

        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertGreaterEqual(good_loss.item(), 0.0)

    def test_compute_pairwise_rank_loss_returns_zero_without_pairs(self):
        pred = torch.tensor([0.5, 0.5], dtype=torch.float32)
        target = torch.tensor([0.1, 0.1], dtype=torch.float32)

        loss = compute_pairwise_rank_loss(pred, target, margin=0.0)

        self.assertEqual(loss.item(), 0.0)

    def test_build_rolling_folds_uses_five_year_train_one_year_test(self):
        folds = build_rolling_folds(start_year=2010, end_year=2013, train_years=5)

        self.assertEqual(
            folds,
            [
                {
                    "fold": "2015",
                    "train_start": "2010-01-01",
                    "train_end": "2013-12-31",
                    "val_start": "2014-01-01",
                    "val_end": "2014-12-31",
                    "test_start": "2015-01-01",
                    "test_end": "2015-12-31",
                },
                {
                    "fold": "2016",
                    "train_start": "2011-01-01",
                    "train_end": "2014-12-31",
                    "val_start": "2015-01-01",
                    "val_end": "2015-12-31",
                    "test_start": "2016-01-01",
                    "test_end": "2016-12-31",
                },
                {
                    "fold": "2017",
                    "train_start": "2012-01-01",
                    "train_end": "2015-12-31",
                    "val_start": "2016-01-01",
                    "val_end": "2016-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2017-12-31",
                },
                {
                    "fold": "2018",
                    "train_start": "2013-01-01",
                    "train_end": "2016-12-31",
                    "val_start": "2017-01-01",
                    "val_end": "2017-12-31",
                    "test_start": "2018-01-01",
                    "test_end": "2018-12-31",
                },
            ],
        )

    def test_add_label_columns_uses_next_two_opens_per_stock(self):
        df = pd.DataFrame(
            {
                "code": ["AAA", "AAA", "AAA", "AAA"],
                "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
                "open": [10.0, 11.0, 12.1, 12.0],
                "close": [10.2, 11.2, 12.0, 11.8],
                "high": [10.3, 11.4, 12.4, 12.1],
                "low": [9.9, 10.8, 11.8, 11.7],
                "volume": [100, 120, 130, 90],
                "amount": [1000, 1200, 1300, 900],
                "amplitude": [1.0, 1.2, 1.1, 0.9],
                "pct_chg": [1.0, 1.1, 0.8, -0.2],
                "change": [0.1, 0.2, 0.1, -0.1],
                "turnover_rate": [0.5, 0.6, 0.7, 0.4],
            }
        )

        labeled = add_label_columns(df)

        self.assertLess(abs(labeled.loc[0, "label"] - 0.1), 1e-12)
        self.assertEqual(round(labeled.loc[1, "label"], 6), round(12.0 / 12.1 - 1.0, 6))
        self.assertTrue(pd.isna(labeled.loc[2, "label"]))
        self.assertEqual(labeled.loc[0, "label_cls"], 1)

    def test_evaluate_topk_returns_uses_daily_top1_prediction(self):
        prediction_frame = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.03, 0.01, -0.02, 0.05],
                "true": [0.02, -0.01, -0.03, 0.04],
            }
        )

        metrics = evaluate_topk_returns(prediction_frame, top_k=1)

        self.assertEqual(metrics["num_days"], 2)
        self.assertEqual(round(metrics["mean_return"], 6), 0.03)
        self.assertEqual(round(metrics["cumulative_return"], 6), round((1.02 * 1.04) - 1.0, 6))
        self.assertEqual(metrics["top_picks"], [("2020-01-02", "AAA"), ("2020-01-03", "BBB")])

    def test_evaluate_topk_returns_skips_untradable_limit_up_candidates(self):
        prediction_frame = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.90, 0.80, 0.70, 0.60],
                "true": [0.20, 0.03, 0.10, 0.04],
                "tradable": [False, True, False, True],
            }
        )

        metrics = evaluate_topk_returns(prediction_frame, top_k=1)

        self.assertEqual(metrics["top_picks"], [("2020-01-02", "BBB"), ("2020-01-03", "BBB")])
        self.assertEqual(round(metrics["mean_return"], 6), 0.035)

    def test_combine_prediction_frames_supports_rank_average_ensemble(self):
        first = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.8, 0.2, 0.4, 0.6],
                "true": [0.02, -0.01, -0.03, 0.04],
            }
        )
        second = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.7, 0.3, 0.8, 0.1],
                "true": [0.02, -0.01, -0.03, 0.04],
            }
        )

        combined = combine_prediction_frames([first, second], method="rank_mean")
        metrics = evaluate_topk_returns(combined, top_k=1)

        self.assertEqual(metrics["top_picks"], [("2020-01-02", "AAA"), ("2020-01-03", "AAA")])
        self.assertEqual(round(metrics["mean_return"], 6), round(-0.005, 6))

    def test_confidence_selector_switches_to_more_confident_model_by_day(self):
        first = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.9, 0.1, 0.51, 0.50],
                "true": [0.03, -0.01, -0.02, 0.05],
                "tradable": [True, True, True, True],
            }
        )
        second = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.54, 0.55, 0.8, 0.2],
                "true": [0.03, -0.01, -0.02, 0.05],
                "tradable": [True, True, True, True],
            }
        )

        selected = build_confidence_selector_frame(first, second, method="top1_gap")
        metrics = evaluate_topk_returns(selected, top_k=1)

        self.assertEqual(metrics["top_picks"], [("2020-01-02", "AAA"), ("2020-01-03", "AAA")])

    def test_confidence_selector_preserves_tradable_filter(self):
        first = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02"],
                "code": ["AAA", "BBB"],
                "pred": [0.9, 0.1],
                "true": [0.20, 0.03],
                "tradable": [False, True],
            }
        )
        second = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02"],
                "code": ["AAA", "BBB"],
                "pred": [0.8, 0.2],
                "true": [0.20, 0.03],
                "tradable": [False, True],
            }
        )

        selected = build_confidence_selector_frame(first, second, method="top1_gap")
        metrics = evaluate_topk_returns(selected, top_k=1)

        self.assertEqual(metrics["top_picks"], [("2020-01-02", "BBB")])

    def test_selector_audit_reports_switch_and_disagreement_rates(self):
        first = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.9, 0.1, 0.51, 0.50],
                "true": [0.03, -0.01, -0.02, 0.05],
                "tradable": [True, True, True, True],
            }
        )
        second = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.54, 0.55, 0.8, 0.2],
                "true": [0.03, -0.01, -0.02, 0.05],
                "tradable": [True, True, True, True],
            }
        )

        daily = build_selector_audit_frame(first, second)
        summary = summarize_selector_audit(daily)

        self.assertEqual(int(summary["num_days"]), 2)
        self.assertEqual(int(summary["switch_count"]), 1)
        self.assertEqual(round(summary["switch_rate"], 6), 1.0)
        self.assertEqual(round(summary["disagreement_rate"], 6), 0.5)
        self.assertEqual(round(summary["left_usage_rate"], 6), 0.5)
        self.assertEqual(round(summary["right_usage_rate"], 6), 0.5)

    def test_threshold_gated_strategy_falls_back_when_edge_is_below_threshold(self):
        first = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.90, 0.10, 0.51, 0.50],
                "true": [0.03, -0.01, -0.02, 0.05],
                "tradable": [True, True, True, True],
            }
        )
        second = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.54, 0.55, 0.80, 0.20],
                "true": [0.03, -0.01, -0.02, 0.05],
                "tradable": [True, True, True, True],
            }
        )

        daily = build_selector_audit_frame(first, second)
        strategy = build_threshold_gated_strategy_frame(
            daily_audit_frame=daily,
            fallback_source="left",
            min_abs_edge=0.60,
        )

        self.assertEqual(strategy["code"].tolist(), ["AAA", "AAA"])
        self.assertEqual(strategy["selected_source"].tolist(), ["left", "left"])
        self.assertEqual(strategy["gated_active_selector"].tolist(), [True, False])

    def test_live_proxy_avoids_duplicate_round_trip_cost_on_same_code_streak(self):
        strategy = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
                "code": ["AAA", "AAA", "BBB"],
                "pred": [0.9, 0.8, 0.7],
                "true": [0.10, 0.10, 0.10],
            }
        )

        proxy = apply_live_trading_proxy(strategy, buy_cost_bps=10.0, sell_cost_bps=10.0)

        self.assertEqual(proxy["entry_cost_bps"].tolist(), [10.0, 0.0, 10.0])
        self.assertEqual(proxy["exit_cost_bps"].tolist(), [0.0, 10.0, 10.0])
        self.assertAlmostEqual(proxy.loc[0, "net_return"], 0.0989, places=6)
        self.assertAlmostEqual(proxy.loc[1, "net_return"], 0.0989, places=6)
        self.assertAlmostEqual(proxy.loc[2, "net_return"], 0.097801, places=6)

        summary = summarize_live_proxy(proxy)
        self.assertEqual(summary["num_days"], 3)
        self.assertEqual(summary["switch_count"], 1)
        self.assertEqual(summary["same_code_streak_days"], 1)

    def test_build_daily_top1_strategy_frame_keeps_tradable_top_pick(self):
        prediction_frame = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "code": ["AAA", "BBB", "AAA", "BBB"],
                "pred": [0.90, 0.80, 0.20, 0.70],
                "true": [0.20, 0.03, 0.10, 0.04],
                "tradable": [False, True, True, True],
            }
        )

        strategy = build_daily_top1_strategy_frame(prediction_frame)

        self.assertEqual(strategy["code"].tolist(), ["BBB", "BBB"])
        self.assertEqual(strategy["true"].tolist(), [0.03, 0.04])

    def test_add_label_columns_marks_next_open_limit_up_as_untradable(self):
        df = pd.DataFrame(
            {
                "code": ["AAA", "AAA", "AAA", "AAA"],
                "date": ["2020-08-20", "2020-08-21", "2020-08-24", "2020-08-25"],
                "open": [10.0, 11.0, 12.1, 13.0],
                "close": [10.0, 11.0, 12.1, 13.0],
                "high": [10.1, 11.0, 12.2, 13.1],
                "low": [9.9, 10.8, 12.0, 12.9],
                "volume": [100, 120, 130, 90],
                "amount": [1000, 1200, 1300, 900],
                "amplitude": [1.0, 1.2, 1.1, 0.9],
                "pct_chg": [0.0, 10.0, 10.0, 7.4],
                "change": [0.0, 1.0, 1.1, 0.9],
                "turnover_rate": [0.5, 0.6, 0.7, 0.4],
            }
        )

        labeled = add_label_columns(df)

        self.assertFalse(bool(labeled.loc[0, "can_buy_on_next_open"]))
        self.assertFalse(bool(labeled.loc[1, "can_buy_on_next_open"]))
        self.assertTrue(bool(labeled.loc[2, "can_buy_on_next_open"]))

    def test_prepare_market_dataframe_rebuilds_corrupt_cache(self):
        rows = []
        for day_idx in range(140):
            rows.append(
                {
                    "code": "AAA",
                    "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                    "open": 10.0 + day_idx * 0.01,
                    "close": 10.1 + day_idx * 0.01,
                    "high": 10.2 + day_idx * 0.01,
                    "low": 9.9 + day_idx * 0.01,
                    "volume": 1000 + day_idx,
                    "amount": 3e7 + day_idx * 1000,
                    "amplitude": 1.0,
                    "pct_chg": 0.5,
                    "change": 0.05,
                    "turnover_rate": 0.8,
                }
            )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "market_daily.parquet")
            cache_path = os.path.join(tmpdir, "market_daily_features.parquet")
            frame.to_parquet(source_path, index=False)
            with open(cache_path, "wb") as handle:
                handle.write(b"not-a-valid-parquet")

            rebuilt = prepare_market_dataframe(
                parquet_path=source_path,
                start_date="2010-01-01",
                min_history=120,
                min_avg_amount=2e7,
                cache_path=cache_path,
            )

            self.assertFalse(rebuilt.empty)
            self.assertIn("can_buy_on_next_open", rebuilt.columns)
            reloaded = pd.read_parquet(cache_path)
            self.assertEqual(len(reloaded), len(rebuilt))
            self.assertIn("can_buy_on_next_open", reloaded.columns)

    def test_prepare_market_dataframe_exposes_only_past_or_same_day_features(self):
        rows = []
        for day_idx in range(140):
            rows.append(
                {
                    "code": "AAA",
                    "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                    "open": 10.0 + day_idx * 0.01,
                    "close": 10.1 + day_idx * 0.01,
                    "high": 10.2 + day_idx * 0.01,
                    "low": 9.9 + day_idx * 0.01,
                    "volume": 1000 + day_idx,
                    "amount": 3e7 + day_idx * 1000,
                    "amplitude": 1.0,
                    "pct_chg": 0.5,
                    "change": 0.05,
                    "turnover_rate": 0.8,
                }
            )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "market_daily.parquet")
            cache_path = os.path.join(tmpdir, "market_daily_features.parquet")
            frame.to_parquet(source_path, index=False)

            rebuilt = prepare_market_dataframe(
                parquet_path=source_path,
                start_date="2010-01-01",
                min_history=120,
                min_avg_amount=2e7,
                cache_path=cache_path,
            )

            self.assertIn("avg_price", rebuilt.columns)
            self.assertNotIn("target_shifted", rebuilt.columns)
            base_cols = [
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
            self.assertTrue(all(col in rebuilt.columns for col in base_cols))
            self.assertEqual(len(base_cols), 24)

    def test_dataset_market_daily_exposes_raw_direction_labels(self):
        rows = []
        for day_idx in range(160):
            base_open = 10.0 + day_idx * 0.02
            next_sign = 1.0 if day_idx % 2 == 0 else -1.0
            close_price = base_open * (1.01 if next_sign > 0 else 0.99)
            rows.append(
                {
                    "code": "AAA",
                    "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                    "open": base_open,
                    "close": close_price,
                    "high": max(base_open, close_price) * 1.01,
                    "low": min(base_open, close_price) * 0.99,
                    "volume": 1000 + day_idx,
                    "amount": 3e7 + day_idx * 1000,
                    "amplitude": 1.5,
                    "pct_chg": (close_price / base_open - 1.0) * 100.0,
                    "change": close_price - base_open,
                    "turnover_rate": 0.8,
                }
            )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "market_daily.parquet")
            cache_path = os.path.join(tmpdir, "market_daily_features.parquet")
            frame.to_parquet(source_path, index=False)

            args = SimpleNamespace(
                market_feature_set="A",
                market_cache_path=cache_path,
                market_fold_year=2015,
                market_start_year=2010,
                market_min_history=20,
                market_min_avg_amount=1e7,
            )
            dataset = Dataset_MarketDaily(
                args=args,
                root_path=tmpdir,
                flag="train",
                size=(20, 0, 1),
                features="MS",
                data_path="market_daily.parquet",
                target="label",
                scale=True,
                timeenc=0,
                freq="d",
            )

            self.assertIn("label_cls", dataset.sample_meta.columns)
            self.assertTrue(set(dataset.sample_meta["label_cls"].unique()).issubset({0.0, 1.0}))
            first_sample = dataset.samples[0]
            expected_cls = float(first_sample["label"] > 0)
            self.assertEqual(float(first_sample["label_cls"]), expected_cls)

    def test_combine_market_multitask_losses_adds_bce_term(self):
        losses = combine_market_multitask_losses(
            reg_loss=2.0,
            cls_loss=0.4,
            cls_weight=0.5,
        )

        self.assertEqual(losses["reg_loss"], 2.0)
        self.assertEqual(losses["cls_loss"], 0.4)
        self.assertEqual(losses["total_loss"], 2.2)


if __name__ == "__main__":
    unittest.main()
