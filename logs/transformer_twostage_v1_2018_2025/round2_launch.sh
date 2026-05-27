#!/usr/bin/env bash
set -euo pipefail
PY=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs2/diffsynth2/bin/python
COMMON_ARGS=(
  run.py
  --task_name long_term_forecast
  --is_training 1
  --model Transformer
  --data market_daily
  --root_path .
  --data_path market_daily.parquet
  --features MS
  --target label
  --freq d
  --seq_len 20
  --label_len 0
  --pred_len 1
  --enc_in 24
  --dec_in 24
  --c_out 24
  --d_model 64
  --d_ff 128
  --e_layers 2
  --n_heads 4
  --factor 3
  --dropout 0.1
  --learning_rate 0.0001
  --train_epochs 20
  --patience 3
  --batch_size 4096
  --num_workers 8
  --loss Huber
  --huber_delta 1.0
  --market_feature_set A
  --market_cache_path ./cache/market_daily_features_full2010.parquet
  --market_start_year 2010
  --market_min_history 120
  --market_min_avg_amount 20000000.0
  --market_train_full_window
  --market_cross_section_batches
  --market_train_horizons 1,3,5
  --market_train_horizon_weights 2.0,0.25,0.25
  --market_target_mode cs_two_stage
  --market_rank_loss
  --market_rank_weight 1.0
  --market_rank_margin 0.0
  --market_mixed_rank_weight 1.0
  --market_mixed_reg_weight 1.0
  --market_winner_topk 20
  --train_mode train_loss_plateau
  --train_plateau_metric loss
  --train_plateau_patience 3
  --train_plateau_ema_decay 0.7
  --des stage2topheavy_topk_csrank_topq_twostage_v1
  --stage2_epochs 20
  --stage2_train_mode train_loss_plateau
  --stage2_train_plateau_metric loss
  --stage2_head_concentration_weight 0.01
  --stage2_head_concentration_temperature 0.7
  --stage2_static_bias_weight 0.01
  --stage2_static_bias_topk 3
  --market_cs_recent_k 5
  --embed timeF
  --gpu 0
)
for GPU_YEAR in "0 2022" "1 2023" "2 2024" "3 2025"; do
  set -- $GPU_YEAR
  GPU=$1
  YEAR=$2
  MID="market_${YEAR}_stage2topheavy_topk_csrank_topq_twostage_v1_Transformer"
  LOG="logs/transformer_twostage_v1_2018_2025/run_logs/${YEAR}_Transformer.log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PY" "${COMMON_ARGS[@]}" --model_id "$MID" --market_fold_year "$YEAR" > "$LOG" 2>&1 &
  echo "$YEAR $GPU $! $LOG"
done
