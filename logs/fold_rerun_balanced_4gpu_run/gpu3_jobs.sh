#!/usr/bin/env bash
set -u
ROOT=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2
PY=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python
LOG=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/fold_rerun_balanced_4gpu_run/gpu3_worker.log
cd "$ROOT"
echo "[launcher] gpu=3 start $(date -u +%F_%T)" >> "$LOG"
echo "[start] gpu=3 model=TimesNet fold=2018" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2018_20_fsA --model TimesNet --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2018 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --embed fixed >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=TimesNet fold=2018 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=TimesNet fold=2022" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2022_20_fsA --model TimesNet --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2022 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --embed fixed >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=TimesNet fold=2022 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=TimeMixer fold=2016" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2016_20_fsA --model TimeMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 3 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2016 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed fixed --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=TimeMixer fold=2016 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=TimeMixer fold=2018" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2018_20_fsA --model TimeMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 3 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2018 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed fixed --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=TimeMixer fold=2018 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=TimeMixer fold=2022" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2022_20_fsA --model TimeMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 3 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2022 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed fixed --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=TimeMixer fold=2022 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=PatchTST fold=2016" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2016_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2016 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=PatchTST fold=2016 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=PatchTST fold=2018" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2018_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2018 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=PatchTST fold=2018 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=PatchTST fold=2020" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2020_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2020 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=PatchTST fold=2020 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=PatchTST fold=2024" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2024_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2024 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=PatchTST fold=2024 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=DLinear fold=2022" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2022_20_fsA --model DLinear --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 64 --d_ff 128 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2022 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=DLinear fold=2022 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=iTransformer fold=2016" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2016_20_fsA --model iTransformer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2016 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=iTransformer fold=2016 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=iTransformer fold=2020" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2020_20_fsA --model iTransformer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2020 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=iTransformer fold=2020 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=iTransformer fold=2024" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2024_20_fsA --model iTransformer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2024 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=iTransformer fold=2024 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=TSMixer fold=2018" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2018_20_fsA --model TSMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2018 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=TSMixer fold=2018 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=3 model=TSMixer fold=2022" >> "$LOG"
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2022_20_fsA --model TSMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2022 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=3 model=TSMixer fold=2022 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[launcher] gpu=3 done $(date -u +%F_%T)" >> "$LOG"
