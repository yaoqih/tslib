import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from data_provider.data_loader import Dataset_MarketDaily
from data_provider.data_factory import MarketDateBatchSampler
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast, MarketForecastMultiTaskWrapper
from models.TimeMixer import Model as TimeMixerModel
from models.Transformer import Model as TransformerModel
from utils.market_cross_section import MarketCrossSectionModel
from utils.market_research import (
    add_label_columns,
    apply_static_score_debias,
    build_head_candidate_diagnostics,
    build_market_diagnostics,
    build_rolling_folds,
    combine_prediction_frames,
    evaluate_prediction_file,
    evaluate_prediction_frame,
    evaluate_topk_returns,
    get_feature_columns,
    get_train_target_columns,
    make_time_features,
    prepare_market_dataframe,
)
from utils.market_multitask import (
    build_pred_topq_weights,
    combine_market_multitask_losses,
    compute_head_concentration_penalty,
    compute_head_gap_penalty,
    compute_masked_regression_loss,
    compute_grouped_pairwise_rank_loss,
    compute_masked_pairwise_rank_loss,
    compute_weighted_masked_regression_loss,
    compute_pairwise_rank_loss,
    compute_rank_ic,
    compute_static_bias_surrogate_penalty,
    compute_topk_mean_return_proxy,
    compute_masked_topk_mean_return_proxy,
    compute_topk_listwise_loss,
    compute_masked_topk_listwise_loss,
)
from utils.tools import TrainLossPlateauCheckpoint
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
from scripts.market_daily.run_backbone_stage2topheavy_topk_matrix import build_job_record


class TestMarketResearch(unittest.TestCase):
    def test_build_head_candidate_diagnostics_computes_daily_overlap_regret_and_score_stats(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-02", "2024-01-02", "2024-01-02",
                        "2024-01-03", "2024-01-03", "2024-01-03",
                    ]
                ),
                "code": ["A", "B", "C", "A", "B", "C"],
                "pred": [0.9, 0.8, 0.1, 0.7, 0.6, 0.5],
                "true": [0.20, 0.05, -0.10, -0.20, 0.30, 0.10],
                "tradable": [True, True, True, True, True, True],
            }
        )

        diagnostics = build_head_candidate_diagnostics(
            frame,
            pred_topk_list=(1, 2),
            true_topk_list=(1, 2),
        )

        self.assertIn("summary", diagnostics)
        self.assertIn("daily", diagnostics)
        self.assertEqual(len(diagnostics["daily"]), 2)

        day1 = diagnostics["daily"][0]
        self.assertEqual(day1["top1_code"], "A")
        self.assertAlmostEqual(day1["top1_true"], 0.20, places=6)
        self.assertAlmostEqual(day1["best_true_tradable"], 0.20, places=6)
        self.assertAlmostEqual(day1["regret_vs_best_tradable"], 0.0, places=6)
        self.assertEqual(day1["hit_top1_in_true1"], 1)
        self.assertEqual(day1["overlap_pred2_true2"], 2)

        day2 = diagnostics["daily"][1]
        self.assertEqual(day2["top1_code"], "A")
        self.assertAlmostEqual(day2["top1_true"], -0.20, places=6)
        self.assertAlmostEqual(day2["best_true_tradable"], 0.30, places=6)
        self.assertAlmostEqual(day2["regret_vs_best_tradable"], 0.50, places=6)
        self.assertEqual(day2["hit_top1_in_true1"], 0)
        self.assertEqual(day2["overlap_pred2_true2"], 1)

        summary = diagnostics["summary"]
        self.assertAlmostEqual(summary["top1_mean"], 0.0, places=6)
        self.assertAlmostEqual(summary["hit_true1_rate"], 0.5, places=6)
        self.assertAlmostEqual(summary["avg_overlap_pred2_true2"], 1.5, places=6)
        self.assertAlmostEqual(summary["avg_regret"], 0.25, places=6)

    def test_build_head_candidate_diagnostics_reports_top_pick_concentration_and_debias_delta(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
                "code": ["A", "B", "A", "B"],
                "pred": [0.9, 0.1, 0.8, 0.2],
                "true": [0.10, -0.05, 0.20, -0.10],
                "tradable": [True, True, True, True],
            }
        )

        diagnostics = build_head_candidate_diagnostics(frame, pred_topk_list=(1,), true_topk_list=(1,))

        summary = diagnostics["summary"]
        self.assertEqual(summary["unique_top1"], 1)
        self.assertEqual(summary["max_rep_top1"], 2)
        self.assertEqual(summary["top_pick_counts"][0]["code"], "A")
        self.assertEqual(summary["top_pick_counts"][0]["pick_count"], 2)
        self.assertLessEqual(summary["debias015_top1_mean"], summary["top1_mean"])

    def test_backbone_job_record_includes_des_and_model_specific_embedding(self):
        row = build_job_record(
            year=2021,
            model="PatchTST",
            feature_set="A",
            seq_len=20,
            batch_size=4096,
            cache_path="./cache/market_daily_features_full2010.parquet",
            learning_rate=0.0001,
            des="backbone_stage2topheavy_topk",
        )

        command = row["command"]
        self.assertIn("\"--des\"", command)
        self.assertIn("backbone_stage2topheavy_topk", command)
        self.assertIn("\"--embed\", \"timeF\"", command)

    def test_build_market_diagnostics_reports_head_buckets_repetition_and_market_slices(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2022-01-04", "2022-01-04", "2022-01-04",
                        "2022-01-05", "2022-01-05", "2022-01-05",
                    ]
                ),
                "code": ["A", "B", "C", "A", "B", "D"],
                "pred": [0.9, 0.5, 0.1, 0.8, 0.6, 0.2],
                "true": [0.03, 0.01, -0.02, -0.01, 0.02, 0.00],
                "tradable": [True, True, True, True, True, True],
                "market_cc_mean": [0.02, 0.02, 0.02, -0.01, -0.01, -0.01],
                "market_amount_top10_share": [0.45, 0.45, 0.45, 0.62, 0.62, 0.62],
            }
        )

        diagnostics = build_market_diagnostics(
            frame,
            topk_list=(1, 3),
            score_bucket_count=3,
            high_repeat_top_n=2,
        )

        self.assertIn("summary", diagnostics)
        self.assertIn("score_buckets", diagnostics)
        self.assertIn("top_repeated_picks", diagnostics)
        self.assertIn("market_slices", diagnostics)
        self.assertAlmostEqual(diagnostics["summary"]["top1_mean_return"], 0.01, places=6)
        self.assertEqual(diagnostics["summary"]["top_pick_unique"], 1)
        self.assertEqual(diagnostics["summary"]["top_pick_max_rep"], 2)
        self.assertEqual(len(diagnostics["score_buckets"]), 3)
        self.assertEqual(diagnostics["top_repeated_picks"][0]["code"], "A")
        self.assertIn("market_cc_mean", diagnostics["market_slices"])
        self.assertIn("market_amount_top10_share", diagnostics["market_slices"])

    def test_stage2_head_weights_override_stage1_weights(self):
        args = SimpleNamespace(
            market_rank_weight=0.1,
            market_topk_weight=0.0,
            market_topk_k=3,
            market_topk_temperature=1.0,
            market_topk_target_mode="soft",
            market_head_concentration_weight=0.02,
            market_head_concentration_temperature=1.0,
            market_head_gap_weight=0.01,
            market_static_bias_weight=0.015,
            market_static_bias_topk=3,
            stage2_rank_weight=0.3,
            stage2_topk_weight=0.05,
            stage2_topk_k=5,
            stage2_topk_temperature=0.7,
            stage2_topk_target_mode="hard",
            stage2_head_concentration_weight=0.06,
            stage2_head_gap_weight=0.03,
            stage2_head_concentration_temperature=0.5,
            stage2_static_bias_weight=0.04,
            stage2_static_bias_topk=5,
        )
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = args
        exp._training_phase = "stage1"

        self.assertAlmostEqual(exp._active_market_rank_weight(), 0.1, places=6)
        self.assertAlmostEqual(exp._active_market_topk_weight(), 0.0, places=6)
        self.assertEqual(exp._active_market_topk_k(), 3)
        self.assertAlmostEqual(exp._active_market_topk_temperature(), 1.0, places=6)
        self.assertEqual(exp._active_market_topk_target_mode(), "soft")
        self.assertAlmostEqual(exp._active_market_head_concentration_weight(), 0.02, places=6)
        self.assertAlmostEqual(exp._active_market_head_gap_weight(), 0.01, places=6)
        self.assertAlmostEqual(exp._active_market_head_concentration_temperature(), 1.0, places=6)
        self.assertAlmostEqual(exp._active_market_static_bias_weight(), 0.015, places=6)
        self.assertEqual(exp._active_market_static_bias_topk(), 3)

        exp._set_training_phase("stage2")
        self.assertAlmostEqual(exp._active_market_rank_weight(), 0.3, places=6)
        self.assertAlmostEqual(exp._active_market_topk_weight(), 0.05, places=6)
        self.assertEqual(exp._active_market_topk_k(), 5)
        self.assertAlmostEqual(exp._active_market_topk_temperature(), 0.7, places=6)
        self.assertEqual(exp._active_market_topk_target_mode(), "hard")
        self.assertAlmostEqual(exp._active_market_head_concentration_weight(), 0.06, places=6)
        self.assertAlmostEqual(exp._active_market_head_gap_weight(), 0.03, places=6)
        self.assertAlmostEqual(exp._active_market_head_concentration_temperature(), 0.5, places=6)
        self.assertAlmostEqual(exp._active_market_static_bias_weight(), 0.04, places=6)
        self.assertEqual(exp._active_market_static_bias_topk(), 5)

    def test_market_sample_cache_path_changes_with_split_args(self):
        args_a = SimpleNamespace(
            market_cache_path="./cache/market_daily_features.parquet",
            market_train_horizons="1,3,5",
            market_test_end="2022-02-28",
            market_start_year=2019,
            market_train_full_window=True,
            market_min_history=120,
            market_min_avg_amount=2e7,
        )
        args_b = SimpleNamespace(
            market_cache_path="./cache/market_daily_features.parquet",
            market_train_horizons="1,3,5",
            market_test_end="2022-03-31",
            market_start_year=2019,
            market_train_full_window=True,
            market_min_history=120,
            market_min_avg_amount=2e7,
        )
        dataset_a = Dataset_MarketDaily.__new__(Dataset_MarketDaily)
        dataset_a.args = args_a
        dataset_a.flag = "test"
        dataset_a.seq_len = 20
        dataset_a.target_mode = "raw"
        dataset_b = Dataset_MarketDaily.__new__(Dataset_MarketDaily)
        dataset_b.args = args_b
        dataset_b.flag = "test"
        dataset_b.seq_len = 20
        dataset_b.target_mode = "raw"

        path_a = dataset_a._sample_cache_path(2022)
        path_b = dataset_b._sample_cache_path(2022)

        self.assertNotEqual(path_a, path_b)

    def test_transformer_market_encode_uses_time_marks(self):
        torch.manual_seed(0)
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            pred_len=1,
            enc_in=24,
            dec_in=24,
            c_out=24,
            d_model=16,
            embed="timeF",
            freq="d",
            dropout=0.0,
            factor=3,
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=32,
            activation="gelu",
        )
        model = TransformerModel(configs)
        model.eval()
        x_enc = torch.randn(2, 20, 24)
        x_mark_a = torch.zeros(2, 20, 3)
        x_mark_b = torch.ones(2, 20, 3)

        with torch.no_grad():
            out_a = model.encode_market_sequence(x_enc, x_mark_a)
            out_b = model.encode_market_sequence(x_enc, x_mark_b)

        self.assertFalse(torch.allclose(out_a, out_b))

    def test_compute_masked_topk_mean_return_proxy_ignores_untradable_winners(self):
        target = torch.tensor([0.20, 0.05, -0.01], dtype=torch.float32)
        pred = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
        tradable_mask = torch.tensor([False, True, True], dtype=torch.bool)

        masked = compute_masked_topk_mean_return_proxy(pred, target, tradable_mask, top_k=1)

        self.assertAlmostEqual(masked.item(), 0.05, places=6)

    def test_compute_masked_topk_listwise_loss_only_uses_tradable_names(self):
        target = torch.tensor([0.50, 0.20, -0.30], dtype=torch.float32)
        pred = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
        tradable_mask = torch.tensor([False, True, True], dtype=torch.bool)

        masked_loss = compute_masked_topk_listwise_loss(
            pred=pred,
            target=target,
            tradable_mask=tradable_mask,
            top_k=1,
            temperature=1.0,
            target_mode="hard",
        )
        expected_loss = compute_topk_listwise_loss(
            pred=pred[1:],
            target=target[1:],
            top_k=1,
            temperature=1.0,
            target_mode="hard",
        )

        self.assertAlmostEqual(masked_loss.item(), expected_loss.item(), places=6)

    def test_market_sample_cache_path_contains_cross_section_rank_target_mode(self):
        args = SimpleNamespace(
            market_cache_path="./cache/market_daily_features.parquet",
            market_train_horizons="1,3,5",
            market_test_end="2022-12-31",
            market_start_year=2019,
            market_train_full_window=True,
            market_min_history=120,
            market_min_avg_amount=2e7,
        )
        dataset = Dataset_MarketDaily.__new__(Dataset_MarketDaily)
        dataset.args = args
        dataset.flag = "train"
        dataset.seq_len = 20
        dataset.target_mode = "cross_section_rank"

        path = dataset._sample_cache_path(2022)

        self.assertIn("cross_section_rank", path)

    def test_build_pred_topq_weights_upweights_predicted_head(self):
        pred = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3], dtype=torch.float32)
        weights = build_pred_topq_weights(
            pred=pred,
            tradable_mask=None,
            topq_ratio=0.4,
            topq_weight=3.0,
        )

        expected = torch.tensor([1.0, 3.0, 1.0, 3.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(weights, expected))

    def test_compute_weighted_masked_regression_loss_matches_manual_mse(self):
        pred = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        target = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
        weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        loss = compute_weighted_masked_regression_loss(
            pred=pred,
            target=target,
            tradable_mask=None,
            sample_weight=weights,
            loss_name="mse",
        )

        manual = ((weights * (pred - target).pow(2)).sum() / weights.sum()).item()
        self.assertAlmostEqual(loss.item(), manual, places=6)

    def test_compute_masked_pairwise_rank_loss_only_uses_tradable_pairs(self):
        pred = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
        target = torch.tensor([0.50, 0.20, -0.30], dtype=torch.float32)
        tradable_mask = torch.tensor([False, True, True], dtype=torch.bool)

        masked_loss = compute_masked_pairwise_rank_loss(
            pred=pred,
            target=target,
            tradable_mask=tradable_mask,
            margin=0.0,
        )
        expected_loss = compute_pairwise_rank_loss(
            pred=pred[1:],
            target=target[1:],
            margin=0.0,
        )

        self.assertAlmostEqual(masked_loss.item(), expected_loss.item(), places=6)

    def test_compute_masked_regression_loss_ignores_untradable_samples(self):
        pred = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float32)
        target = torch.tensor([10.0, 1.0, -1.0], dtype=torch.float32)
        tradable_mask = torch.tensor([False, True, True], dtype=torch.bool)
        criterion = torch.nn.MSELoss()

        masked_loss = compute_masked_regression_loss(criterion, pred, target, tradable_mask)

        self.assertAlmostEqual(masked_loss.item(), 0.0, places=6)

    def test_compute_head_concentration_penalty_penalizes_extreme_single_name(self):
        sharp_pred = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float32)
        flat_pred = torch.tensor([1.0, 0.9, 0.8], dtype=torch.float32)

        sharp_penalty = compute_head_concentration_penalty(sharp_pred, temperature=1.0)
        flat_penalty = compute_head_concentration_penalty(flat_pred, temperature=1.0)

        self.assertGreater(sharp_penalty.item(), flat_penalty.item())

    def test_compute_head_gap_penalty_penalizes_large_top1_gap(self):
        sharp_pred = torch.tensor([5.0, 0.5, 0.4, 0.3], dtype=torch.float32)
        flat_pred = torch.tensor([1.0, 0.9, 0.8, 0.7], dtype=torch.float32)

        sharp_penalty = compute_head_gap_penalty(sharp_pred, top_k=3)
        flat_penalty = compute_head_gap_penalty(flat_pred, top_k=3)

        self.assertGreater(sharp_penalty.item(), flat_penalty.item())

    def test_compute_head_penalties_respect_tradable_mask(self):
        pred = torch.tensor([5.0, 1.0, 0.9], dtype=torch.float32)
        tradable_mask = torch.tensor([False, True, True], dtype=torch.bool)

        concentration = compute_head_concentration_penalty(pred, tradable_mask=tradable_mask, temperature=1.0)
        gap = compute_head_gap_penalty(pred, tradable_mask=tradable_mask, top_k=2)

        expected_concentration = compute_head_concentration_penalty(torch.tensor([1.0, 0.9]), temperature=1.0)
        expected_gap = compute_head_gap_penalty(torch.tensor([1.0, 0.9]), top_k=2)

        self.assertAlmostEqual(concentration.item(), expected_concentration.item(), places=6)
        self.assertAlmostEqual(gap.item(), expected_gap.item(), places=6)

    def test_compute_static_bias_surrogate_penalty_matches_head_gap_penalty(self):
        pred = torch.tensor([3.0, 0.9, 0.7, 0.2], dtype=torch.float32)
        tradable_mask = torch.tensor([True, True, False, True], dtype=torch.bool)

        surrogate = compute_static_bias_surrogate_penalty(pred, tradable_mask=tradable_mask, top_k=2)
        expected = compute_head_gap_penalty(pred, tradable_mask=tradable_mask, top_k=2)

        self.assertAlmostEqual(surrogate.item(), expected.item(), places=6)

    def test_build_tradable_mask_from_batch_meta(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.device = torch.device("cpu")
        dataset = SimpleNamespace(
            sample_tradable_mask=torch.tensor([True, False, True], dtype=torch.bool)
        )
        batch_meta = torch.tensor([0, 1, 2], dtype=torch.long)

        tradable_mask = exp._build_market_tradable_mask(batch_meta, dataset)

        self.assertTrue(torch.equal(tradable_mask, torch.tensor([True, False, True], dtype=torch.bool)))

    def test_evaluate_prediction_file_matches_direct_frame_evaluation(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-04", "2022-01-04", "2022-01-05", "2022-01-05"]),
                "code": ["A", "B", "A", "B"],
                "pred": [0.5, 0.2, 0.1, 0.3],
                "true": [0.04, 0.01, -0.02, 0.06],
                "tradable": [True, True, True, True],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "preds.csv")
            frame.to_csv(csv_path, index=False)

            file_metrics = evaluate_prediction_file(csv_path, topk_list=(1, 3, 5))
            direct_metrics = evaluate_prediction_frame(frame, topk_list=(1, 3, 5))

        self.assertAlmostEqual(file_metrics["top1_mean_return"], direct_metrics["top1_mean_return"], places=8)
        self.assertAlmostEqual(file_metrics["top1_cumulative_return"], direct_metrics["top1_cumulative_return"], places=8)
        self.assertAlmostEqual(file_metrics["rank_ic"], direct_metrics["rank_ic"], places=8)

    def test_make_time_features_respects_frequency_shape(self):
        dates = pd.to_datetime(["2022-01-04", "2022-01-05"])

        daily = make_time_features(dates, freq="d")
        hourly = make_time_features(dates, freq="h")

        self.assertEqual(daily.shape, (2, 3))
        self.assertEqual(hourly.shape, (2, 4))

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

    def test_compute_grouped_pairwise_rank_loss_only_uses_same_day_pairs(self):
        pred = torch.tensor([0.9, 0.1, 0.8, 0.0], dtype=torch.float32)
        target = torch.tensor([0.2, -0.1, 0.3, -0.2], dtype=torch.float32)
        group_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

        grouped_loss = compute_grouped_pairwise_rank_loss(
            pred=pred,
            target=target,
            group_ids=group_ids,
            margin=0.0,
            min_target_gap=0.0,
        )

        cross_day_bad = compute_pairwise_rank_loss(pred=pred, target=target, margin=0.0)

        self.assertEqual(grouped_loss.item(), 0.0)
        self.assertGreater(cross_day_bad.item(), 0.0)

    def test_compute_topk_listwise_loss_rewards_better_top_order(self):
        target = torch.tensor([0.5, 0.2, -0.1, -0.3], dtype=torch.float32)
        pred_good = torch.tensor([3.0, 2.0, 0.0, -1.0], dtype=torch.float32)
        pred_bad = torch.tensor([-1.0, 0.0, 2.0, 3.0], dtype=torch.float32)

        good_loss = compute_topk_listwise_loss(pred_good, target, top_k=2, temperature=1.0, target_mode="soft")
        bad_loss = compute_topk_listwise_loss(pred_bad, target, top_k=2, temperature=1.0, target_mode="soft")

        self.assertLess(good_loss.item(), bad_loss.item())

    def test_compute_topk_listwise_loss_supports_hard_mode(self):
        target = torch.tensor([0.5, 0.2, -0.1], dtype=torch.float32)
        pred = torch.tensor([1.2, 0.3, -0.4], dtype=torch.float32)

        loss = compute_topk_listwise_loss(pred, target, top_k=1, temperature=1.0, target_mode="hard")

        self.assertGreaterEqual(loss.item(), 0.0)

    def test_compute_topk_mean_return_proxy_rewards_better_top_pick(self):
        target = torch.tensor([0.15, 0.04, -0.02], dtype=torch.float32)
        pred_good = torch.tensor([2.0, 0.5, -1.0], dtype=torch.float32)
        pred_bad = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float32)

        good_proxy = compute_topk_mean_return_proxy(pred_good, target, top_k=1)
        bad_proxy = compute_topk_mean_return_proxy(pred_bad, target, top_k=1)

        self.assertGreater(good_proxy.item(), bad_proxy.item())
        self.assertAlmostEqual(good_proxy.item(), 0.15, places=6)

    def test_compute_rank_ic_rewards_better_cross_section_order(self):
        target = torch.tensor([0.30, 0.10, -0.20], dtype=torch.float32)
        pred_good = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float32)
        pred_bad = torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float32)

        good_ic = compute_rank_ic(pred_good, target)
        bad_ic = compute_rank_ic(pred_bad, target)

        self.assertGreater(good_ic.item(), bad_ic.item())
        self.assertGreater(good_ic.item(), 0.9)

    def test_train_loss_plateau_checkpoint_stops_after_smoothed_plateau(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = TrainLossPlateauCheckpoint(
                patience=2,
                verbose=False,
                delta=0.01,
                ema_decay=0.5,
            )

            stopper(1.0, model, tmpdir)
            self.assertFalse(stopper.early_stop)
            self.assertAlmostEqual(stopper.best_smoothed_loss, 1.0)

            stopper(1.0, model, tmpdir)
            self.assertFalse(stopper.early_stop)
            self.assertAlmostEqual(stopper.smoothed_loss, 1.0)

            stopper(1.0, model, tmpdir)
            self.assertTrue(stopper.early_stop)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "checkpoint.pth")))

    def test_train_loss_plateau_checkpoint_supports_max_mode(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = TrainLossPlateauCheckpoint(
                patience=2,
                verbose=False,
                delta=0.01,
                ema_decay=0.5,
                mode="max",
                metric_name="top1_return",
            )

            stopper(0.10, model, tmpdir)
            self.assertFalse(stopper.early_stop)
            self.assertAlmostEqual(stopper.best_smoothed_loss, 0.10)

            stopper(0.20, model, tmpdir)
            self.assertFalse(stopper.early_stop)
            self.assertGreater(stopper.best_smoothed_loss, 0.10)

            stopper(0.10, model, tmpdir)
            self.assertFalse(stopper.early_stop)

            stopper(0.10, model, tmpdir)
            self.assertTrue(stopper.early_stop)

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
        self.assertEqual(round(labeled.loc[0, "label_close_3d"], 6), round(11.8 / 10.2 - 1.0, 6))
        self.assertTrue(pd.isna(labeled.loc[2, "label"]))
        self.assertEqual(labeled.loc[0, "label_cls"], 1)

    def test_get_train_target_columns_maps_multi_horizon_spec(self):
        self.assertEqual(get_train_target_columns("1,3,5"), ["label", "label_close_3d", "label_close_5d"])

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

    def test_apply_static_score_debias_reduces_constant_stock_bias(self):
        prediction_frame = pd.DataFrame(
            {
                "date": [
                    "2020-01-01", "2020-01-01",
                    "2020-01-02", "2020-01-02",
                    "2020-01-03", "2020-01-03",
                ],
                "code": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
                "pred": [0.90, 0.80, 0.91, 0.83, 0.92, 0.95],
                "true": [-0.02, 0.01, -0.01, 0.02, -0.03, 0.05],
            }
        )

        debiased = apply_static_score_debias(prediction_frame, method="expanding_mean", strength=1.0)

        raw_metrics = evaluate_topk_returns(prediction_frame, top_k=1)
        debiased_metrics = evaluate_topk_returns(debiased, top_k=1)

        self.assertEqual(raw_metrics["top_picks"], [("2020-01-01", "AAA"), ("2020-01-02", "AAA"), ("2020-01-03", "BBB")])
        self.assertEqual(debiased_metrics["top_picks"], [("2020-01-01", "AAA"), ("2020-01-02", "BBB"), ("2020-01-03", "BBB")])
        self.assertGreater(debiased_metrics["mean_return"], raw_metrics["mean_return"])

    def test_evaluate_prediction_frame_reports_head_basket_metrics(self):
        prediction_frame = pd.DataFrame(
            {
                "date": ["2020-01-02"] * 5 + ["2020-01-03"] * 5,
                "code": ["A", "B", "C", "D", "E"] * 2,
                "pred": [0.9, 0.7, 0.5, 0.2, 0.1, 0.8, 0.6, 0.4, 0.3, 0.0],
                "true": [0.10, 0.06, 0.02, -0.01, -0.03, 0.09, 0.05, 0.01, -0.02, -0.04],
            }
        )

        metrics = evaluate_prediction_frame(prediction_frame, topk_list=(1, 3, 5))

        self.assertIn("top1_mean_return", metrics)
        self.assertIn("top3_mean_return", metrics)
        self.assertIn("top5_mean_return", metrics)
        self.assertGreater(metrics["top1_mean_return"], metrics["top5_mean_return"])
        self.assertGreater(metrics["top3_mean_return"], 0.0)

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

    def test_prepare_market_dataframe_adds_market_context_columns(self):
        rows = []
        for code_idx, code in enumerate(["AAA", "BBB"]):
            for day_idx in range(140):
                rows.append(
                    {
                        "code": code,
                        "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                        "open": 10.0 + day_idx * 0.01 + code_idx * 0.1,
                        "close": 10.1 + day_idx * 0.01 + code_idx * 0.1,
                        "high": 10.2 + day_idx * 0.01 + code_idx * 0.1,
                        "low": 9.9 + day_idx * 0.01 + code_idx * 0.1,
                        "volume": 1000 + day_idx + code_idx * 10,
                        "amount": 3e7 + day_idx * 1000 + code_idx * 100,
                        "amplitude": 1.0 + code_idx * 0.1,
                        "pct_chg": 0.5 + code_idx * 0.1,
                        "change": 0.05 + code_idx * 0.01,
                        "turnover_rate": 0.8 + code_idx * 0.05,
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

            for col in [
                "market_cc_mean",
                "market_cc_std",
                "market_turnover_mean",
                "market_amplitude_mean",
                "market_ret_20_mean",
                "market_vol_20_mean",
            ]:
                self.assertIn(col, rebuilt.columns)

    def test_prepare_market_dataframe_adds_market_structure_columns(self):
        rows = []
        moves = [0.03, 0.0, -0.02]
        for code_idx, code in enumerate(["AAA", "BBB", "CCC"]):
            for day_idx in range(160):
                base_open = 10.0 + day_idx * 0.01 + code_idx * 0.1
                close_price = base_open * (1.0 + moves[code_idx])
                rows.append(
                    {
                        "code": code,
                        "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                        "open": base_open,
                        "close": close_price,
                        "high": max(base_open, close_price) * 1.01,
                        "low": min(base_open, close_price) * 0.99,
                        "volume": 1000 + day_idx + code_idx * 10,
                        "amount": 3e7 + day_idx * 1000 + code_idx * 1000,
                        "amplitude": 1.0 + code_idx * 0.2,
                        "pct_chg": moves[code_idx] * 100.0,
                        "change": close_price - base_open,
                        "turnover_rate": 0.8 + code_idx * 0.1,
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

            for col in [
                "cc_vs_market",
                "turnover_vs_market",
                "market_cc_q25",
                "market_cc_q75",
                "market_up_ratio",
                "market_amount_top10_share",
                "market_amount_hhi",
            ]:
                self.assertIn(col, rebuilt.columns)

            sample = rebuilt.iloc[0]
            self.assertAlmostEqual(sample["cc_vs_market"], sample["cc"] - sample["market_cc_mean"], places=6)
            self.assertGreaterEqual(sample["market_up_ratio"], 0.0)
            self.assertLessEqual(sample["market_up_ratio"], 1.0)

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

    def test_dataset_market_daily_can_use_cross_section_rank_targets(self):
        rows = []
        for code_idx, code in enumerate(["AAA", "BBB"]):
            for day_idx in range(160):
                base_open = 10.0 + day_idx * 0.02 + code_idx * 0.1
                move = 0.03 if code_idx == 0 else -0.01
                close_price = base_open * (1.0 + move)
                rows.append(
                    {
                        "code": code,
                        "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                        "open": base_open,
                        "close": close_price,
                        "high": max(base_open, close_price) * 1.01,
                        "low": min(base_open, close_price) * 0.99,
                        "volume": 1000 + day_idx + code_idx * 10,
                        "amount": 3e7 + day_idx * 1000 + code_idx * 100,
                        "amplitude": 1.5 + code_idx * 0.1,
                        "pct_chg": (close_price / base_open - 1.0) * 100.0,
                        "change": close_price - base_open,
                        "turnover_rate": 0.8 + code_idx * 0.05,
                    }
                )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "market_daily.parquet")
            cache_path = os.path.join(tmpdir, "market_daily_features.parquet")
            frame.to_parquet(source_path, index=False)

            args = SimpleNamespace(
                market_feature_set="A_CTX",
                market_cache_path=cache_path,
                market_fold_year=2015,
                market_start_year=2010,
                market_min_history=20,
                market_min_avg_amount=1e7,
                market_target_mode="cross_section_rank",
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

            seq_x, seq_y, _, _, sample_id = dataset[0]
            self.assertEqual(seq_x.shape[-1], len(get_feature_columns("A_CTX")))
            self.assertIn("train_label", dataset.sample_meta.columns)
            self.assertIn("raw_label", dataset.sample_meta.columns)
            self.assertGreaterEqual(float(seq_y[-1, -1]), 0.0)
            self.assertLessEqual(float(seq_y[-1, -1]), 1.0)
            self.assertNotAlmostEqual(
                float(seq_y[-1, -1]),
                float(dataset.sample_meta.iloc[sample_id]["raw_label"]),
            )

    def test_dataset_market_daily_can_use_market_heavy_feature_set(self):
        rows = []
        moves = [0.03, 0.01, -0.01]
        for code_idx, code in enumerate(["AAA", "BBB", "CCC"]):
            for day_idx in range(160):
                base_open = 10.0 + day_idx * 0.02 + code_idx * 0.1
                close_price = base_open * (1.0 + moves[code_idx])
                rows.append(
                    {
                        "code": code,
                        "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                        "open": base_open,
                        "close": close_price,
                        "high": max(base_open, close_price) * 1.01,
                        "low": min(base_open, close_price) * 0.99,
                        "volume": 1000 + day_idx + code_idx * 10,
                        "amount": 3e7 + day_idx * 1000 + code_idx * 100,
                        "amplitude": 1.5 + code_idx * 0.1,
                        "pct_chg": moves[code_idx] * 100.0,
                        "change": close_price - base_open,
                        "turnover_rate": 0.8 + code_idx * 0.05,
                    }
                )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "market_daily.parquet")
            cache_path = os.path.join(tmpdir, "market_daily_features.parquet")
            frame.to_parquet(source_path, index=False)

            args = SimpleNamespace(
                market_feature_set="B_MKT",
                market_cache_path=cache_path,
                market_fold_year=2015,
                market_start_year=2010,
                market_min_history=20,
                market_min_avg_amount=1e7,
                market_target_mode="cross_section_rank",
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

            seq_x, _, _, _, _ = dataset[0]
            self.assertEqual(seq_x.shape[-1], len(get_feature_columns("B_MKT")))

    def test_dataset_market_daily_can_return_separate_aux_market_features(self):
        rows = []
        moves = [0.03, 0.01, -0.01]
        for code_idx, code in enumerate(["AAA", "BBB", "CCC"]):
            for day_idx in range(160):
                base_open = 10.0 + day_idx * 0.02 + code_idx * 0.1
                close_price = base_open * (1.0 + moves[code_idx])
                rows.append(
                    {
                        "code": code,
                        "date": (pd.Timestamp("2010-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                        "open": base_open,
                        "close": close_price,
                        "high": max(base_open, close_price) * 1.01,
                        "low": min(base_open, close_price) * 0.99,
                        "volume": 1000 + day_idx + code_idx * 10,
                        "amount": 3e7 + day_idx * 1000 + code_idx * 100,
                        "amplitude": 1.5 + code_idx * 0.1,
                        "pct_chg": moves[code_idx] * 100.0,
                        "change": close_price - base_open,
                        "turnover_rate": 0.8 + code_idx * 0.05,
                    }
                )
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "market_daily.parquet")
            cache_path = os.path.join(tmpdir, "market_daily_features.parquet")
            frame.to_parquet(source_path, index=False)

            args = SimpleNamespace(
                market_feature_set="A_CTX",
                market_aux_feature_set="B_MKT",
                market_cache_path=cache_path,
                market_fold_year=2015,
                market_start_year=2010,
                market_min_history=20,
                market_min_avg_amount=1e7,
                market_target_mode="cross_section_rank",
                market_aux_cls=True,
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

            seq_x, _, _, _, aux_x, _ = dataset[0]
            self.assertEqual(seq_x.shape[-1], len(get_feature_columns("A_CTX")))
            self.assertEqual(aux_x.shape[-1], len(get_feature_columns("B_MKT")))

    def test_market_forecast_multitask_wrapper_uses_aux_market_input_for_cls(self):
        class DummyBase(torch.nn.Module):
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
                return torch.ones(x_enc.size(0), x_enc.size(1), 24)

        wrapper = MarketForecastMultiTaskWrapper(DummyBase(), feature_dim=24, aux_input_dim=6 * 4)
        x = torch.randn(2, 4, 24)
        aux = torch.randn(2, 6, 4)
        out = wrapper(
            x_enc=x,
            x_mark_enc=torch.zeros(2, 4, 4),
            x_dec=x,
            x_mark_dec=torch.zeros(2, 4, 4),
            aux_x_enc=aux,
        )

        self.assertIn("forecast", out)
        self.assertIn("cls_logits", out)
        self.assertEqual(out["forecast"].shape, (2, 4, 24))
        self.assertEqual(out["cls_logits"].shape, (2, 4, 1))

    def test_market_date_batch_sampler_groups_same_day_samples(self):
        sample_meta = pd.DataFrame(
            {
                "date": ["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-05", "2021-01-05"],
                "code": ["AAA", "BBB", "AAA", "BBB", "CCC"],
            }
        )
        dataset = SimpleNamespace(sample_meta=sample_meta)
        sampler = MarketDateBatchSampler(dataset, shuffle=False)

        batches = list(iter(sampler))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], [0, 1])
        self.assertEqual(batches[1], [2, 3, 4])

    def test_market_date_batch_sampler_groups_same_day_samples_when_dates_are_interleaved(self):
        sample_meta = pd.DataFrame(
            {
                "date": ["2021-01-04", "2021-01-05", "2021-01-04", "2021-01-05", "2021-01-04"],
                "code": ["AAA", "AAA", "BBB", "BBB", "CCC"],
            }
        )
        dataset = SimpleNamespace(sample_meta=sample_meta)
        sampler = MarketDateBatchSampler(dataset, shuffle=False)

        batches = list(iter(sampler))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], [0, 2, 4])
        self.assertEqual(batches[1], [1, 3])

    def test_market_cross_section_batches_disable_aux_head_path(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            market_aux_cls=True,
            market_cross_section_batches=True,
            data="market_daily",
            task_name="long_term_forecast",
        )

        self.assertTrue(exp._use_market_cross_section_batches())
        self.assertFalse(exp._use_market_aux_cls())

    def test_market_cross_section_model_outputs_single_score_per_stock(self):
        class DummyBase(torch.nn.Module):
            def encode_market_sequence(self, x_enc, x_mark_enc):
                return torch.ones(x_enc.size(0), x_enc.size(1), 8)

        configs = SimpleNamespace(
            seq_len=4,
            pred_len=1,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.1,
            market_cs_layers=1,
            market_cs_n_heads=2,
            market_cs_d_ff=16,
            market_cs_dropout=0.1,
        )
        model = MarketCrossSectionModel(DummyBase(), configs)
        x_enc = torch.randn(5, 4, 3)
        out = model(
            x_enc=x_enc,
            x_mark_enc=torch.zeros(5, 4, 3),
            x_dec=torch.zeros(5, 1, 3),
            x_mark_dec=torch.zeros(5, 1, 3),
        )

        self.assertEqual(out["forecast"].shape, (5, 1, 1))
        self.assertEqual(out["backbone_latent"].shape, (5, 4, 8))
        self.assertEqual(out["cross_section_tokens"].shape, (5, 4, 8))
        self.assertEqual(out["time_weights"].shape, (5, 4, 1))

    def test_market_cross_section_model_requires_true_sequence_latent_backbone(self):
        class DummyBase(torch.nn.Module):
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
                return x_enc[:, -1:, :1]

        configs = SimpleNamespace(
            seq_len=4,
            pred_len=1,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.1,
            market_cs_layers=1,
            market_cs_n_heads=2,
            market_cs_d_ff=16,
            market_cs_dropout=0.1,
        )

        with self.assertRaises(ValueError):
            MarketCrossSectionModel(DummyBase(), configs)

    def test_compute_market_loss_keeps_prediction_and_rank_terms_separate(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            features="MS",
            pred_len=1,
            market_rank_loss=True,
            market_rank_weight=0.5,
            market_topk_loss=False,
            market_topk_weight=0.0,
            market_rank_margin=0.0,
            market_cross_section_batches=True,
            market_aux_cls=False,
            data="market_daily",
            task_name="long_term_forecast",
        )
        exp.device = torch.device("cpu")

        forecast = torch.tensor(
            [
                [[0.0, 0.9]],
                [[0.0, 0.1]],
            ],
            dtype=torch.float32,
        )
        batch_y = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        losses = exp._compute_market_loss(
            outputs=forecast,
            batch_y=batch_y,
            batch_meta=None,
            dataset=SimpleNamespace(),
            reg_criterion=torch.nn.MSELoss(),
        )

        self.assertAlmostEqual(losses["reg_loss"].item(), 0.01, places=6)
        self.assertAlmostEqual(losses["rank_loss"].item(), 0.0, places=6)
        self.assertAlmostEqual(losses["topk_loss"].item(), 0.0, places=6)
        self.assertAlmostEqual(losses["total_loss"].item(), 0.01, places=6)

    def test_compute_market_loss_adds_topk_term_in_cross_section_mode(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            features="MS",
            pred_len=1,
            market_rank_loss=False,
            market_rank_weight=0.0,
            market_topk_loss=True,
            market_topk_weight=0.5,
            market_topk_k=2,
            market_topk_temperature=1.0,
            market_topk_target_mode="soft",
            market_rank_margin=0.0,
            market_cross_section_batches=True,
            market_aux_cls=False,
            data="market_daily",
            task_name="long_term_forecast",
        )
        exp.device = torch.device("cpu")

        forecast = torch.tensor(
            [
                [[0.0, 0.8]],
                [[0.0, 0.4]],
                [[0.0, -0.3]],
            ],
            dtype=torch.float32,
        )
        batch_y = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 0.5]],
                [[0.0, -0.2]],
            ],
            dtype=torch.float32,
        )

        losses = exp._compute_market_loss(
            outputs=forecast,
            batch_y=batch_y,
            batch_meta=None,
            dataset=SimpleNamespace(),
            reg_criterion=torch.nn.MSELoss(),
        )

        self.assertGreater(losses["topk_loss"].item(), 0.0)
        self.assertGreater(losses["total_loss"].item(), losses["reg_loss"].item())

    def test_compute_market_loss_ignores_tradable_mask_for_rank_and_topk_by_default(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            features="MS",
            pred_len=1,
            market_rank_loss=True,
            market_rank_weight=0.5,
            market_topk_loss=True,
            market_topk_weight=0.5,
            market_topk_k=2,
            market_topk_temperature=1.0,
            market_topk_target_mode="soft",
            market_rank_margin=0.0,
            market_cross_section_batches=True,
            market_aux_cls=False,
            market_train_horizon_weights="1.0,0.5,0.5",
            market_regression_use_tradable_mask=False,
            market_train_on_tradable_only=False,
            loss="MSE",
            huber_delta=1.0,
            data="market_daily",
            task_name="long_term_forecast",
        )
        exp.device = torch.device("cpu")

        forecast = torch.tensor(
            [
                [[0.0, 0.9]],
                [[0.0, -0.1]],
                [[0.0, 0.2]],
            ],
            dtype=torch.float32,
        )
        batch_y = torch.zeros_like(forecast)
        dataset = SimpleNamespace(
            train_target_columns=["label", "label_close_3d", "label_close_5d"],
            sample_train_targets_scaled=torch.tensor(
                [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [-0.2, -0.1, -0.05]],
                dtype=torch.float32,
            ),
            sample_train_targets_raw=torch.tensor(
                [[0.00, 0.00, 0.00], [0.03, 0.01, 0.0], [-0.04, -0.02, -0.01]],
                dtype=torch.float32,
            ),
            sample_tradable_mask=torch.tensor([False, True, True], dtype=torch.bool),
        )
        batch_meta = torch.tensor([0, 1, 2], dtype=torch.int64)

        losses = exp._compute_market_loss(
            outputs=forecast,
            batch_y=batch_y,
            batch_meta=batch_meta,
            dataset=dataset,
            reg_criterion=torch.nn.MSELoss(),
        )

        masked_rank_loss = compute_pairwise_rank_loss(
            pred=torch.tensor([-0.1, 0.2], dtype=torch.float32),
            target=torch.tensor([0.0175, -0.0275], dtype=torch.float32),
            margin=0.0,
        )

        self.assertNotAlmostEqual(losses["rank_loss"].item(), masked_rank_loss.item(), places=6)
        self.assertGreater(losses["topk_loss"].item(), 0.0)

    def test_compute_market_loss_can_mask_all_training_losses_when_enabled(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            features="MS",
            pred_len=1,
            market_rank_loss=True,
            market_rank_weight=0.5,
            market_topk_loss=True,
            market_topk_weight=0.5,
            market_topk_k=2,
            market_topk_temperature=1.0,
            market_topk_target_mode="soft",
            market_rank_margin=0.0,
            market_cross_section_batches=True,
            market_aux_cls=False,
            market_train_horizon_weights="1.0,0.5,0.5",
            market_regression_use_tradable_mask=False,
            market_train_on_tradable_only=True,
            loss="MSE",
            huber_delta=1.0,
            data="market_daily",
            task_name="long_term_forecast",
        )
        exp.device = torch.device("cpu")

        forecast = torch.tensor(
            [
                [[0.0, 0.9]],
                [[0.0, -0.1]],
                [[0.0, 0.2]],
            ],
            dtype=torch.float32,
        )
        batch_y = torch.zeros_like(forecast)
        dataset = SimpleNamespace(
            train_target_columns=["label", "label_close_3d", "label_close_5d"],
            sample_train_targets_scaled=torch.tensor(
                [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [-0.2, -0.1, -0.05]],
                dtype=torch.float32,
            ),
            sample_train_targets_raw=torch.tensor(
                [[0.00, 0.00, 0.00], [0.03, 0.01, 0.0], [-0.04, -0.02, -0.01]],
                dtype=torch.float32,
            ),
            sample_tradable_mask=torch.tensor([False, True, True], dtype=torch.bool),
        )
        batch_meta = torch.tensor([0, 1, 2], dtype=torch.int64)

        losses = exp._compute_market_loss(
            outputs=forecast,
            batch_y=batch_y,
            batch_meta=batch_meta,
            dataset=dataset,
            reg_criterion=torch.nn.MSELoss(),
        )

        masked_rank_loss = compute_pairwise_rank_loss(
            pred=torch.tensor([-0.1, 0.2], dtype=torch.float32),
            target=torch.tensor([0.0175, -0.0275], dtype=torch.float32),
            margin=0.0,
        )

        self.assertAlmostEqual(losses["rank_loss"].item(), masked_rank_loss.item(), places=6)

    def test_compute_market_loss_reports_training_monitor_proxies(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            features="MS",
            pred_len=1,
            market_rank_loss=False,
            market_rank_weight=0.0,
            market_topk_loss=False,
            market_topk_weight=0.0,
            market_rank_margin=0.0,
            market_cross_section_batches=True,
            market_aux_cls=False,
            market_train_horizon_weights="2.0,0.25,0.25",
            train_plateau_metric_k=3,
            data="market_daily",
            task_name="long_term_forecast",
        )
        exp.device = torch.device("cpu")

        forecast = torch.tensor(
            [
                [[0.0, 0.9]],
                [[0.0, 0.2]],
                [[0.0, -0.3]],
            ],
            dtype=torch.float32,
        )
        batch_y = torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        dataset = SimpleNamespace(
            train_target_columns=["label", "label_close_3d", "label_close_5d"],
            sample_train_targets_scaled=torch.tensor(
                [
                    [1.0, 0.6, 0.4],
                    [0.5, 0.3, 0.2],
                    [-0.2, -0.1, -0.05],
                ],
                dtype=torch.float32,
            ),
            sample_train_targets_raw=torch.tensor(
                [
                    [0.12, 0.08, 0.06],
                    [0.05, 0.04, 0.03],
                    [-0.03, -0.02, -0.01],
                ],
                dtype=torch.float32,
            ),
        )
        batch_meta = torch.tensor([0, 1, 2], dtype=torch.int64)

        losses = exp._compute_market_loss(
            outputs=forecast,
            batch_y=batch_y,
            batch_meta=batch_meta,
            dataset=dataset,
            reg_criterion=torch.nn.MSELoss(),
        )

        self.assertAlmostEqual(losses["monitor_top1_return"].item(), 0.12, places=6)
        self.assertAlmostEqual(losses["monitor_topk_return"].item(), (0.12 + 0.05 - 0.03) / 3.0, places=6)
        self.assertGreater(losses["monitor_rank_ic"].item(), 0.9)

    def test_compute_market_loss_adds_head_regularizers(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            features="MS",
            pred_len=1,
            market_rank_loss=False,
            market_rank_weight=0.0,
            market_topk_loss=False,
            market_topk_weight=0.0,
            market_rank_margin=0.0,
            market_cross_section_batches=True,
            market_aux_cls=False,
            market_train_horizon_weights="1.0,0.5,0.5",
            train_plateau_metric_k=3,
            market_head_concentration_weight=0.2,
            market_head_concentration_temperature=1.0,
            market_head_gap_weight=0.3,
            market_head_gap_topk=3,
            market_static_bias_weight=0.3,
            market_static_bias_topk=3,
            data="market_daily",
            task_name="long_term_forecast",
            market_regression_use_tradable_mask=False,
        )
        exp.device = torch.device("cpu")

        forecast = torch.tensor(
            [
                [[0.0, 3.0]],
                [[0.0, 0.3]],
                [[0.0, 0.2]],
            ],
            dtype=torch.float32,
        )
        batch_y = torch.zeros_like(forecast)
        dataset = SimpleNamespace(
            train_target_columns=["label", "label_close_3d", "label_close_5d"],
            sample_train_targets_scaled=torch.tensor(
                [[0.8, 0.6, 0.5], [0.2, 0.1, 0.05], [0.1, 0.05, 0.02]],
                dtype=torch.float32,
            ),
            sample_train_targets_raw=torch.tensor(
                [[0.10, 0.08, 0.07], [0.03, 0.02, 0.01], [0.02, 0.01, 0.005]],
                dtype=torch.float32,
            ),
            sample_tradable_mask=torch.tensor([True, True, True], dtype=torch.bool),
        )
        batch_meta = torch.tensor([0, 1, 2], dtype=torch.int64)

        losses = exp._compute_market_loss(
            outputs=forecast,
            batch_y=batch_y,
            batch_meta=batch_meta,
            dataset=dataset,
            reg_criterion=torch.nn.MSELoss(),
        )

        self.assertGreater(losses["head_concentration_loss"].item(), 0.0)
        self.assertGreater(losses["head_gap_loss"].item(), 0.0)
        self.assertAlmostEqual(losses["static_bias_loss"].item(), losses["head_gap_loss"].item(), places=6)
        self.assertGreater(losses["total_loss"].item(), losses["reg_loss"].item())

    def test_training_phase_config_uses_stage_specific_topk_settings(self):
        exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            market_rank_weight=0.1,
            market_topk_weight=0.0,
            market_topk_k=3,
            market_topk_temperature=1.0,
            market_topk_target_mode="soft",
            stage2_epochs=3,
            stage2_rank_weight=0.05,
            stage2_topk_weight=0.02,
            stage2_topk_k=5,
            stage2_topk_temperature=0.7,
            stage2_topk_target_mode="hard",
            train_plateau_metric="loss",
            stage2_train_plateau_metric="topk_mean_return",
        )

        exp._set_training_phase("stage1")
        self.assertEqual(exp._active_market_rank_weight(), 0.1)
        self.assertEqual(exp._active_market_topk_weight(), 0.0)
        self.assertEqual(exp._active_train_plateau_metric(), "loss")

        exp._set_training_phase("stage2")
        self.assertEqual(exp._active_market_rank_weight(), 0.05)
        self.assertEqual(exp._active_market_topk_weight(), 0.02)
        self.assertEqual(exp._active_market_topk_k(), 5)
        self.assertEqual(exp._active_market_topk_temperature(), 0.7)
        self.assertEqual(exp._active_market_topk_target_mode(), "hard")
        self.assertEqual(exp._active_train_plateau_metric(), "topk_mean_return")

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
