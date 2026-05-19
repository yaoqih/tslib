#!/usr/bin/env bash
set -u
ROOT=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2
PY=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python
LOG=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/fold_rerun_balanced_4gpu_run/gpu2_worker.log
cd "$ROOT"
echo "[launcher] gpu=2 start $(date -u +%F_%T)" >> "$LOG"
echo "[start] gpu=2 model=TimesNet fold=2017" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2017_20_fsA --model TimesNet --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2017 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --embed fixed >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=TimesNet fold=2017 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=TimesNet fold=2021" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2021_20_fsA --model TimesNet --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2021 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --embed fixed >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=TimesNet fold=2021 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=TimeMixer fold=2015" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2015_20_fsA --model TimeMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 3 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2015 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed fixed --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=TimeMixer fold=2015 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=TimeMixer fold=2017" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2017_20_fsA --model TimeMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 3 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2017 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed fixed --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=TimeMixer fold=2017 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=TimeMixer fold=2021" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2021_20_fsA --model TimeMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 3 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2021 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed fixed --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=TimeMixer fold=2021 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=PatchTST fold=2015" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2015_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2015 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=PatchTST fold=2015 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=PatchTST fold=2017" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2017_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2017 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=PatchTST fold=2017 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=PatchTST fold=2019" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2019_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2019 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=PatchTST fold=2019 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=PatchTST fold=2023" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2023_20_fsA --model PatchTST --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2023 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=PatchTST fold=2023 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=DLinear fold=2021" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2021_20_fsA --model DLinear --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 64 --d_ff 128 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2021 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=DLinear fold=2021 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=iTransformer fold=2015" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2015_20_fsA --model iTransformer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2015 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=iTransformer fold=2015 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=iTransformer fold=2019" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2019_20_fsA --model iTransformer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2019 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=iTransformer fold=2019 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=iTransformer fold=2023" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2023_20_fsA --model iTransformer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2023 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=iTransformer fold=2023 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=TSMixer fold=2017" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2017_20_fsA --model TSMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2017 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=TSMixer fold=2017 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[start] gpu=2 model=TSMixer fold=2021" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PY" run.py --task_name long_term_forecast --is_training 1 --model_id market_2021_20_fsA --model TSMixer --data market_daily --root_path . --data_path market_daily.parquet --features MS --target label --freq d --seq_len 20 --label_len 0 --pred_len 1 --enc_in 24 --dec_in 24 --c_out 24 --d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --factor 3 --dropout 0.1 --learning_rate 0.0003 --train_epochs 2 --patience 2 --batch_size 4096 --num_workers 8 --loss Huber --huber_delta 1.0 --gpu 0 --checkpoints ./checkpoints --market_fold_year 2021 --market_feature_set A --market_cache_path ./cache/market_daily_features.parquet --market_start_year 2010 --market_min_history 120 --market_min_avg_amount 20000000.0 --train_mode fixed_epoch --market_train_full_window --des market_round1_fsA --use_amp --embed timeF >> "$LOG" 2>&1
rc=$?
echo "[done] gpu=2 model=TSMixer fold=2021 rc=$rc $(date -u +%F_%T)" >> "$LOG"
if [ $rc -ne 0 ]; then exit $rc; fi
echo "[launcher] gpu=2 done $(date -u +%F_%T)" >> "$LOG"
