from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.tools import TrainLossPlateauCheckpoint
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.market_research import get_feature_columns
from utils.market_multitask import (
    build_head_union_mask,
    build_local_knn_mask,
    build_union_topq_weights,
    build_pred_topq_weights,
    build_true_rank_sample_weights,
    combine_market_multitask_losses,
    compute_head_concentration_penalty,
    compute_head_gap_penalty,
    compute_masked_regression_loss,
    compute_masked_pairwise_rank_loss,
    compute_weighted_masked_pairwise_rank_loss,
    compute_masked_topk_listwise_loss,
    compute_masked_topk_mean_return_proxy,
    compute_weighted_masked_regression_loss,
    compute_grouped_pairwise_rank_loss,
    compute_local_neighbor_pairwise_rank_loss,
    compute_pairwise_rank_loss,
    compute_rank_ic,
    compute_static_bias_surrogate_penalty,
    compute_topk_mean_return_proxy,
    compute_topk_listwise_loss,
    compute_winner_pairwise_rank_loss,
)
from utils.market_cross_section import MarketCrossSectionModel

warnings.filterwarnings('ignore')


class MarketForecastMultiTaskWrapper(nn.Module):
    def __init__(self, base_model, feature_dim, aux_input_dim=None):
        super().__init__()
        self.base_model = base_model
        self.cls_head = nn.Linear(feature_dim, 1)
        self.aux_encoder = None
        self.aux_cls_head = None
        if aux_input_dim is not None:
            hidden_dim = max(feature_dim, 64)
            self.aux_encoder = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(aux_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feature_dim),
                nn.GELU(),
            )
            self.aux_cls_head = nn.Linear(feature_dim, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, aux_x_enc=None):
        forecast = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        if aux_x_enc is not None and self.aux_encoder is not None and self.aux_cls_head is not None:
            aux_repr = self.aux_encoder(aux_x_enc)
            cls_logits = self.aux_cls_head(aux_repr).unsqueeze(1).expand(-1, forecast.size(1), -1)
        else:
            cls_logits = self.cls_head(forecast)
        return {
            'forecast': forecast,
            'cls_logits': cls_logits,
        }


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._training_phase = 'stage1'

    def _set_training_phase(self, phase_name):
        if phase_name not in {'stage1', 'stage2'}:
            raise ValueError(f'Unsupported training phase: {phase_name}')
        self._training_phase = phase_name

    def _get_training_phase(self):
        return getattr(self, '_training_phase', 'stage1')

    def _active_market_rank_weight(self):
        if self._get_training_phase() == 'stage2':
            return float(getattr(self.args, 'stage2_rank_weight', self.args.market_rank_weight))
        return float(self.args.market_rank_weight)

    def _active_market_topk_weight(self):
        if self._get_training_phase() == 'stage2':
            return float(getattr(self.args, 'stage2_topk_weight', getattr(self.args, 'market_topk_weight', 0.0)))
        return float(getattr(self.args, 'market_topk_weight', 0.0))

    def _active_market_topk_k(self):
        if self._get_training_phase() == 'stage2':
            return int(getattr(self.args, 'stage2_topk_k', getattr(self.args, 'market_topk_k', 3)))
        return int(getattr(self.args, 'market_topk_k', 3))

    def _active_market_topk_temperature(self):
        if self._get_training_phase() == 'stage2':
            return float(getattr(self.args, 'stage2_topk_temperature', getattr(self.args, 'market_topk_temperature', 1.0)))
        return float(getattr(self.args, 'market_topk_temperature', 1.0))

    def _active_market_topk_target_mode(self):
        if self._get_training_phase() == 'stage2':
            return getattr(self.args, 'stage2_topk_target_mode', getattr(self.args, 'market_topk_target_mode', 'soft'))
        return getattr(self.args, 'market_topk_target_mode', 'soft')

    def _active_market_head_concentration_weight(self):
        if self._get_training_phase() == 'stage2':
            return float(
                getattr(
                    self.args,
                    'stage2_head_concentration_weight',
                    getattr(self.args, 'market_head_concentration_weight', 0.0),
                )
            )
        return float(getattr(self.args, 'market_head_concentration_weight', 0.0))

    def _active_market_head_concentration_temperature(self):
        if self._get_training_phase() == 'stage2':
            return float(
                getattr(
                    self.args,
                    'stage2_head_concentration_temperature',
                    getattr(self.args, 'market_head_concentration_temperature', 1.0),
                )
            )
        return float(getattr(self.args, 'market_head_concentration_temperature', 1.0))

    def _active_market_head_gap_weight(self):
        if self._get_training_phase() == 'stage2':
            return float(
                getattr(
                    self.args,
                    'stage2_head_gap_weight',
                    getattr(self.args, 'market_head_gap_weight', 0.0),
                )
            )
        return float(getattr(self.args, 'market_head_gap_weight', 0.0))

    def _active_market_static_bias_weight(self):
        if self._get_training_phase() == 'stage2':
            return float(
                getattr(
                    self.args,
                    'stage2_static_bias_weight',
                    getattr(
                        self.args,
                        'stage2_head_gap_weight',
                        getattr(
                            self.args,
                            'market_static_bias_weight',
                            getattr(self.args, 'market_head_gap_weight', 0.0),
                        ),
                    ),
                )
            )
        return float(
            getattr(
                self.args,
                'market_static_bias_weight',
                getattr(self.args, 'market_head_gap_weight', 0.0),
            )
        )

    def _active_market_static_bias_topk(self):
        if self._get_training_phase() == 'stage2':
            return int(
                getattr(
                    self.args,
                    'stage2_static_bias_topk',
                    getattr(
                        self.args,
                        'stage2_head_gap_topk',
                        getattr(
                            self.args,
                            'market_static_bias_topk',
                            getattr(self.args, 'market_head_gap_topk', 3),
                        ),
                    ),
                )
            )
        return int(
            getattr(
                self.args,
                'market_static_bias_topk',
                getattr(self.args, 'market_head_gap_topk', 3),
            )
        )

    def _active_train_plateau_metric(self):
        if self._get_training_phase() == 'stage2':
            return getattr(self.args, 'stage2_train_plateau_metric', getattr(self.args, 'train_plateau_metric', 'loss'))
        return getattr(self.args, 'train_plateau_metric', 'loss')

    def _get_market_train_horizon_weights(self, dataset):
        weight_values = [
            float(item.strip())
            for item in str(getattr(self.args, "market_train_horizon_weights", "1.0")).split(",")
            if item.strip()
        ]
        if len(weight_values) != len(dataset.train_target_columns):
            raise ValueError(
                f"market_train_horizon_weights length {len(weight_values)} does not match "
                f"market_train_horizons length {len(dataset.train_target_columns)}"
            )
        weights = torch.tensor(weight_values, dtype=torch.float32, device=self.device)
        return weights / weights.sum().clamp_min(1e-6)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self._use_market_cross_section_batches():
            model = MarketCrossSectionModel(model, self.args).float()
        elif self._use_market_aux_cls():
            aux_feature_set = getattr(self.args, "market_aux_feature_set", "B_MKT")
            aux_input_dim = len(get_feature_columns(aux_feature_set)) * self.args.seq_len
            model = MarketForecastMultiTaskWrapper(model, self.args.c_out, aux_input_dim=aux_input_dim).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = self.args.loss.upper()
        if loss_name == 'MSE':
            return nn.MSELoss()
        if loss_name == 'MAE':
            return nn.L1Loss()
        if loss_name == 'HUBER':
            return nn.HuberLoss(delta=getattr(self.args, 'huber_delta', 1.0))
        raise ValueError(f'Unsupported loss: {self.args.loss}')

    def _use_market_aux_cls(self):
        return bool(
            getattr(self.args, 'market_aux_cls', False)
            and not self._use_market_cross_section_batches()
            and self.args.data == 'market_daily'
            and self.args.task_name == 'long_term_forecast'
        )

    def _use_market_cross_section_batches(self):
        return bool(
            getattr(self.args, 'market_cross_section_batches', False)
            and self.args.data == 'market_daily'
            and self.args.task_name == 'long_term_forecast'
        )

    def _use_market_rank_loss(self):
        return bool(
            getattr(self.args, 'market_rank_loss', False)
            and self.args.data == 'market_daily'
            and self.args.task_name == 'long_term_forecast'
        )

    def _use_market_topk_loss(self):
        return bool(
            (
                getattr(self.args, 'market_topk_loss', False)
                or (self._get_training_phase() == 'stage2' and getattr(self.args, 'stage2_epochs', 0) > 0)
            )
            and self._active_market_topk_weight() > 0.0
            and self.args.data == 'market_daily'
            and self.args.task_name == 'long_term_forecast'
        )

    def _split_model_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs['forecast'], outputs.get('cls_logits')
        return outputs, None

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_aux_x=None):
        if self._use_market_aux_cls():
            return self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                aux_x_enc=batch_aux_x,
            )
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _build_market_cls_targets(self, batch_meta, dataset):
        indices = batch_meta.reshape(-1).long()
        cls_values = dataset.sample_cls_labels.index_select(0, indices).reshape(-1, self.args.pred_len, 1)
        return cls_values.to(self.device, non_blocking=True)

    def _build_market_train_targets(self, batch_meta, dataset):
        indices = batch_meta.reshape(-1).long()
        scaled = dataset.sample_train_targets_scaled.index_select(0, indices).to(self.device, non_blocking=True)
        raw = dataset.sample_train_targets_raw.index_select(0, indices).to(self.device, non_blocking=True)
        return scaled, raw

    def _build_market_tradable_mask(self, batch_meta, dataset):
        if not hasattr(dataset, "sample_tradable_mask"):
            return torch.ones(batch_meta.numel(), dtype=torch.bool, device=self.device)
        indices = batch_meta.reshape(-1).long()
        tradable = dataset.sample_tradable_mask.index_select(0, indices)
        return tradable.to(self.device, non_blocking=True)

    def _build_composite_market_target(self, raw_targets, weights):
        ranked_targets = []
        for horizon_idx in range(raw_targets.size(1)):
            horizon_target = raw_targets[:, horizon_idx]
            order = torch.argsort(horizon_target)
            ranks = torch.empty_like(horizon_target, dtype=torch.float32)
            if horizon_target.numel() == 1:
                ranks.fill_(0.0)
            else:
                ranks[order] = torch.linspace(
                    0.0,
                    1.0,
                    steps=horizon_target.numel(),
                    device=horizon_target.device,
                )
            ranked_targets.append(ranks)
        ranked_stack = torch.stack(ranked_targets, dim=1)
        return torch.matmul(ranked_stack, weights)

    def _build_cross_section_rank_target(self, raw_targets, tradable_mask=None):
        rank_target = raw_targets[:, 0].reshape(-1)
        if tradable_mask is not None and bool(getattr(self.args, "market_train_on_tradable_only", False)):
            active_idx = torch.nonzero(tradable_mask.reshape(-1), as_tuple=False).reshape(-1)
            if active_idx.numel() > 0:
                active_values = rank_target.index_select(0, active_idx)
                order = torch.argsort(active_values)
                active_ranks = torch.empty_like(active_values, dtype=torch.float32)
                if active_values.numel() == 1:
                    active_ranks.fill_(0.0)
                else:
                    active_ranks[order] = torch.linspace(0.0, 1.0, steps=active_values.numel(), device=active_values.device)
                full_rank = torch.zeros_like(rank_target, dtype=torch.float32)
                full_rank.index_copy_(0, active_idx, active_ranks)
                return full_rank
        order = torch.argsort(rank_target)
        ranks = torch.empty_like(rank_target, dtype=torch.float32)
        if rank_target.numel() == 1:
            ranks.fill_(0.0)
        else:
            ranks[order] = torch.linspace(0.0, 1.0, steps=rank_target.numel(), device=rank_target.device)
        return ranks

    def _reshape_market_cross_section(self, reg_pred, reg_true):
        # In date-batch mode, one loader batch is one trading day across all stocks.
        if self._use_market_cross_section_batches():
            return reg_pred.unsqueeze(0), reg_true.unsqueeze(0)
        return reg_pred, reg_true

    def _get_train_plateau_metric_config(self):
        metric_name = self._active_train_plateau_metric()
        if metric_name == 'loss':
            return metric_name, 'min'
        if metric_name in {'top1_return', 'topk_mean_return', 'rank_ic'}:
            return metric_name, 'max'
        raise ValueError(f'Unsupported train_plateau_metric: {metric_name}')

    def _extract_train_plateau_metric_value(self, train_loss, epoch_metrics):
        metric_name, _ = self._get_train_plateau_metric_config()
        if metric_name == 'loss':
            return float(train_loss)
        if metric_name == 'top1_return':
            return float(epoch_metrics['monitor_top1_return'])
        if metric_name == 'topk_mean_return':
            return float(epoch_metrics['monitor_topk_return'])
        if metric_name == 'rank_ic':
            return float(epoch_metrics['monitor_rank_ic'])
        raise ValueError(f'Unsupported train_plateau_metric: {metric_name}')

    def _build_market_local_feature_tensor(self, batch_x, dataset):
        if batch_x is None or not hasattr(dataset, "feature_columns"):
            return None
        feature_names = str(
            getattr(
                self.args,
                "market_local_feature_names",
                "log_amount,turnover_rate,amplitude,ret_20,vol_20",
            )
        )
        selected_names = [item.strip() for item in feature_names.split(",") if item.strip()]
        if not selected_names:
            return None
        column_index = {name: idx for idx, name in enumerate(dataset.feature_columns)}
        feature_idx = [column_index[name] for name in selected_names if name in column_index]
        if not feature_idx:
            return None
        return batch_x[:, -1, feature_idx]

    def _compute_market_loss(self, outputs, batch_y, batch_meta, dataset, reg_criterion, batch_x=None):
        forecast, cls_logits = self._split_model_outputs(outputs)
        loss_name = getattr(self.args, "loss", "MSE")
        f_dim = -1 if self.args.features == 'MS' else 0
        reg_pred = forecast[:, -self.args.pred_len:, f_dim:]
        reg_true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        pred_score = reg_pred.reshape(-1)
        scaled_train_targets = None
        raw_train_targets = None
        horizon_weights = None
        tradable_mask = None
        effective_tradable_mask = None
        if batch_meta is not None and hasattr(dataset, "sample_train_targets_scaled"):
            scaled_train_targets, raw_train_targets = self._build_market_train_targets(batch_meta, dataset)
            horizon_weights = self._get_market_train_horizon_weights(dataset)
            tradable_mask = self._build_market_tradable_mask(batch_meta, dataset)
            effective_tradable_mask = tradable_mask if bool(getattr(self.args, "market_train_on_tradable_only", False)) else None
            pred_topq_weights = build_pred_topq_weights(
                pred=pred_score.detach(),
                tradable_mask=effective_tradable_mask,
                topq_ratio=float(getattr(self.args, "market_pred_topq_ratio", 0.0)),
                topq_weight=float(getattr(self.args, "market_pred_topq_weight", 1.0)),
            )
            candidate_head_weights = build_union_topq_weights(
                pred=pred_score.detach(),
                target=raw_train_targets[:, 0].detach(),
                tradable_mask=effective_tradable_mask,
                topq_ratio=float(getattr(self.args, "market_head_candidate_ratio", 0.0)),
                topq_weight=float(getattr(self.args, "market_head_candidate_weight", 1.0)),
            )
            pred_sample_weight = pred_topq_weights
            rank_sample_weight = torch.maximum(pred_topq_weights, candidate_head_weights)
            target_mode = getattr(dataset, "target_mode", "raw")
            if target_mode == "cross_section_rank_weighted":
                rank_target = self._build_cross_section_rank_target(raw_train_targets, tradable_mask=effective_tradable_mask)
                true_rank_weights = build_true_rank_sample_weights(
                    target=raw_train_targets[:, 0],
                    alpha=float(getattr(self.args, "market_true_rank_alpha", 3.0)),
                    power=float(getattr(self.args, "market_true_rank_power", 2.0)),
                )
                pred_loss = compute_weighted_masked_regression_loss(
                    pred=pred_score,
                    target=rank_target,
                    tradable_mask=effective_tradable_mask,
                    sample_weight=true_rank_weights,
                    loss_name=loss_name,
                    huber_delta=getattr(self.args, 'huber_delta', 1.0),
                )
                rank_sample_weight = true_rank_weights
            elif target_mode == "cross_section_rank":
                rank_target = self._build_cross_section_rank_target(raw_train_targets, tradable_mask=effective_tradable_mask)
                pred_loss = compute_weighted_masked_regression_loss(
                    pred=pred_score,
                    target=rank_target,
                    tradable_mask=effective_tradable_mask,
                    sample_weight=pred_sample_weight,
                    loss_name=loss_name,
                    huber_delta=getattr(self.args, 'huber_delta', 1.0),
                )
            else:
                pred_losses = []
                for horizon_idx in range(scaled_train_targets.size(1)):
                    target_for_horizon = scaled_train_targets[:, horizon_idx]
                    pred_losses.append(
                        compute_weighted_masked_regression_loss(
                            pred=pred_score,
                            target=target_for_horizon,
                            tradable_mask=effective_tradable_mask if getattr(self.args, "market_regression_use_tradable_mask", False) or bool(getattr(self.args, "market_train_on_tradable_only", False)) else None,
                            sample_weight=pred_sample_weight,
                            loss_name=loss_name,
                            huber_delta=getattr(self.args, 'huber_delta', 1.0),
                        )
                    )
                pred_loss = sum(
                    weight * loss
                    for weight, loss in zip(horizon_weights, pred_losses)
                )
        else:
            pred_loss = reg_criterion(reg_pred, reg_true)
        rank_loss = pred_loss.new_tensor(0.0)
        topk_loss = pred_loss.new_tensor(0.0)
        head_concentration_loss = pred_loss.new_tensor(0.0)
        head_gap_loss = pred_loss.new_tensor(0.0)
        static_bias_loss = pred_loss.new_tensor(0.0)
        winner_loss = pred_loss.new_tensor(0.0)
        local_loss = pred_loss.new_tensor(0.0)
        monitor_top1_return = pred_loss.new_tensor(0.0)
        monitor_topk_return = pred_loss.new_tensor(0.0)
        monitor_rank_ic = pred_loss.new_tensor(0.0)
        if self._use_market_rank_loss():
            if raw_train_targets is not None and horizon_weights is not None:
                if getattr(dataset, "target_mode", "raw") == "cross_section_rank_weighted":
                    rank_target = self._build_cross_section_rank_target(raw_train_targets, tradable_mask=effective_tradable_mask)
                else:
                    rank_target = self._build_composite_market_target(raw_train_targets, horizon_weights)
            else:
                rank_target = reg_true.reshape(-1)
            cs_pred, cs_true = self._reshape_market_cross_section(
                reg_pred,
                rank_target.reshape(-1, 1, 1) if rank_target.ndim == 1 else reg_true,
            )
            if self._use_market_cross_section_batches():
                rank_loss = compute_weighted_masked_pairwise_rank_loss(
                    pred=cs_pred.reshape(-1),
                    target=cs_true.reshape(-1),
                    tradable_mask=effective_tradable_mask,
                    sample_weight=rank_sample_weight if raw_train_targets is not None else None,
                    margin=self.args.market_rank_margin,
                )
            elif batch_meta is not None and hasattr(dataset, "sample_group_ids"):
                sample_ids = batch_meta.detach().cpu().numpy().astype(np.int64)
                group_ids = torch.tensor(
                    dataset.sample_group_ids[sample_ids],
                    dtype=torch.int64,
                    device=self.device,
                ).reshape(-1, 1, 1).expand(-1, self.args.pred_len, 1)
                rank_loss = compute_grouped_pairwise_rank_loss(
                    pred=reg_pred.reshape(-1),
                    target=reg_true.reshape(-1),
                    group_ids=group_ids.reshape(-1),
                    margin=self.args.market_rank_margin,
                    min_target_gap=getattr(self.args, "market_rank_min_target_gap", 0.0),
                )
            else:
                rank_loss = compute_weighted_masked_pairwise_rank_loss(
                    pred=reg_pred.reshape(-1),
                    target=reg_true.reshape(-1),
                    tradable_mask=effective_tradable_mask,
                    sample_weight=rank_sample_weight if raw_train_targets is not None else None,
                    margin=self.args.market_rank_margin,
                )
        if self._use_market_topk_loss():
            if not self._use_market_cross_section_batches():
                raise ValueError("market_topk_loss currently requires market_cross_section_batches")
            if raw_train_targets is not None and horizon_weights is not None:
                topk_target = self._build_composite_market_target(raw_train_targets, horizon_weights)
            else:
                topk_target = reg_true.reshape(-1)
            topk_loss = compute_masked_topk_listwise_loss(
                pred=pred_score,
                target=topk_target,
                tradable_mask=effective_tradable_mask,
                top_k=self._active_market_topk_k(),
                temperature=self._active_market_topk_temperature(),
                target_mode=self._active_market_topk_target_mode(),
            )
        if raw_train_targets is not None and self._use_market_cross_section_batches():
            tradable_target = raw_train_targets[:, 0]
            monitor_top1_return = compute_masked_topk_mean_return_proxy(
                pred=pred_score,
                target=tradable_target,
                tradable_mask=tradable_mask,
                top_k=1,
            )
            monitor_topk_return = compute_masked_topk_mean_return_proxy(
                pred=pred_score,
                target=tradable_target,
                tradable_mask=tradable_mask,
                top_k=getattr(self.args, 'train_plateau_metric_k', 3),
            )
            masked_pred = pred_score[tradable_mask] if tradable_mask is not None else pred_score
            masked_target = tradable_target[tradable_mask] if tradable_mask is not None else tradable_target
            monitor_rank_ic = compute_rank_ic(pred=masked_pred, target=masked_target)
            head_concentration_loss = compute_head_concentration_penalty(
                pred=pred_score,
                tradable_mask=effective_tradable_mask,
                temperature=self._active_market_head_concentration_temperature(),
            )
            static_bias_loss = compute_static_bias_surrogate_penalty(
                pred=pred_score,
                tradable_mask=effective_tradable_mask,
                top_k=self._active_market_static_bias_topk(),
            )
            head_gap_loss = static_bias_loss
            if bool(getattr(self.args, "market_winner_loss", False)):
                candidate_mask = build_head_union_mask(
                    pred=pred_score.detach(),
                    target=tradable_target.detach(),
                    tradable_mask=effective_tradable_mask,
                    topq_ratio=float(getattr(self.args, "market_winner_topq_ratio", 0.2)),
                )
                winner_loss = compute_winner_pairwise_rank_loss(
                    pred=pred_score,
                    target=tradable_target,
                    candidate_mask=candidate_mask,
                    margin=float(getattr(self.args, "market_winner_margin", 0.0)),
                    sample_weight=rank_sample_weight if raw_train_targets is not None else None,
                    min_target_gap=float(getattr(self.args, "market_winner_min_target_gap", 0.0)),
                )
            if bool(getattr(self.args, "market_local_loss", False)):
                local_features = self._build_market_local_feature_tensor(batch_x=batch_x, dataset=dataset)
                if local_features is not None:
                    neighbor_mask = build_local_knn_mask(
                        features=local_features,
                        tradable_mask=effective_tradable_mask,
                        neighbor_k=int(getattr(self.args, "market_local_neighbor_k", 20)),
                    )
                    local_loss = compute_local_neighbor_pairwise_rank_loss(
                        pred=pred_score,
                        target=tradable_target,
                        neighbor_mask=neighbor_mask,
                        margin=float(getattr(self.args, "market_local_margin", 0.0)),
                        sample_weight=rank_sample_weight if raw_train_targets is not None else None,
                        min_target_gap=float(getattr(self.args, "market_local_min_target_gap", 0.0)),
                    )
        total_loss = (
            pred_loss
            + self._active_market_rank_weight() * rank_loss
            + self._active_market_topk_weight() * topk_loss
            + self._active_market_head_concentration_weight() * head_concentration_loss
            + self._active_market_static_bias_weight() * static_bias_loss
            + float(getattr(self.args, "market_winner_weight", 0.0)) * winner_loss
            + float(getattr(self.args, "market_local_weight", 0.0)) * local_loss
        )
        if raw_train_targets is not None and getattr(dataset, "target_mode", "raw") == "cross_section_rank_weighted":
            aux_rank_reg_weight = float(getattr(self.args, "market_aux_rank_reg_weight", 0.1))
            total_loss = rank_loss + aux_rank_reg_weight * pred_loss

        if not self._use_market_aux_cls():
            return {
                'total_loss': total_loss,
                'reg_loss': pred_loss,
                'cls_loss': pred_loss.new_tensor(0.0),
                'rank_loss': rank_loss,
                'topk_loss': topk_loss,
                'head_concentration_loss': head_concentration_loss,
                'head_gap_loss': head_gap_loss,
                'static_bias_loss': static_bias_loss,
                'winner_loss': winner_loss,
                'local_loss': local_loss,
                'reg_pred': reg_pred,
                'reg_true': reg_true,
                'monitor_top1_return': monitor_top1_return,
                'monitor_topk_return': monitor_topk_return,
                'monitor_rank_ic': monitor_rank_ic,
            }

        cls_target = self._build_market_cls_targets(batch_meta, dataset)
        cls_loss = self.cls_criterion(
            cls_logits[:, -self.args.pred_len:, :].reshape(-1),
            cls_target.reshape(-1),
        )
        losses = combine_market_multitask_losses(
            reg_loss=total_loss,
            cls_loss=cls_loss,
            cls_weight=self.args.market_cls_weight,
        )
        return {
            'total_loss': losses['total_loss'],
            'reg_loss': losses['reg_loss'],
            'cls_loss': losses['cls_loss'],
            'rank_loss': rank_loss,
            'topk_loss': topk_loss,
            'head_concentration_loss': head_concentration_loss,
            'head_gap_loss': head_gap_loss,
            'static_bias_loss': static_bias_loss,
            'winner_loss': winner_loss,
            'local_loss': local_loss,
            'reg_pred': reg_pred,
            'reg_true': reg_true,
            'monitor_top1_return': monitor_top1_return,
            'monitor_topk_return': monitor_topk_return,
            'monitor_rank_ic': monitor_rank_ic,
        }

    def _unpack_batch(self, batch):
        if len(batch) == 6:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_aux_x, batch_meta = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_aux_x, batch_meta
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_meta = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, None, batch_meta
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return batch_x, batch_y, batch_x_mark, batch_y_mark, None, None
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_aux_x, batch_meta = self._unpack_batch(batch)
                batch_x = batch_x.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_aux_x = batch_aux_x.to(self.device, dtype=torch.float32, non_blocking=True) if batch_aux_x is not None else None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_aux_x)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_aux_x)
                losses = self._compute_market_loss(outputs, batch_y, batch_meta, vali_data, criterion, batch_x=batch_x)
                loss = losses['total_loss']

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _run_training_phase(
        self,
        phase_name,
        setting,
        train_data,
        train_loader,
        vali_data,
        vali_loader,
        test_data,
        test_loader,
        path,
        train_epochs,
        train_mode,
    ):
        self._set_training_phase(phase_name)
        train_steps = len(train_loader)
        use_early_stopping = train_mode == 'best_val'
        use_train_plateau = train_mode == 'train_loss_plateau'
        if train_mode not in {'best_val', 'fixed_epoch', 'train_loss_plateau'}:
            raise ValueError(f'Unsupported train_mode: {train_mode}')

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if use_early_stopping else None
        train_plateau = None
        if use_train_plateau:
            plateau_metric_name, plateau_mode = self._get_train_plateau_metric_config()
            train_plateau = TrainLossPlateauCheckpoint(
                patience=getattr(self.args, 'train_plateau_patience', self.args.patience),
                verbose=True,
                delta=getattr(self.args, 'train_plateau_delta', 0.0),
                ema_decay=getattr(self.args, 'train_plateau_ema_decay', 0.7),
                mode=plateau_mode,
                metric_name=plateau_metric_name,
            )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self._use_market_aux_cls():
            self.cls_criterion = nn.BCEWithLogitsLoss()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        time_now = time.time()

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            epoch_monitor_metrics = {
                'monitor_top1_return': [],
                'monitor_topk_return': [],
                'monitor_rank_ic': [],
                'monitor_head_concentration': [],
                'monitor_static_bias': [],
            }

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_aux_x, batch_meta = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_aux_x = batch_aux_x.to(self.device, dtype=torch.float32, non_blocking=True) if batch_aux_x is not None else None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_aux_x)
                        losses = self._compute_market_loss(outputs, batch_y, batch_meta, train_data, criterion, batch_x=batch_x)
                        loss = losses['total_loss']
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_aux_x)
                    losses = self._compute_market_loss(outputs, batch_y, batch_meta, train_data, criterion, batch_x=batch_x)
                    loss = losses['total_loss']

                train_loss.append(loss.item())
                for key in epoch_monitor_metrics:
                    source_key = key
                    if key == 'monitor_head_concentration':
                        source_key = 'head_concentration_loss'
                    elif key == 'monitor_static_bias':
                        source_key = 'static_bias_loss'
                    epoch_monitor_metrics[key].append(float(losses[source_key].item()))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"[{phase_name}] Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            epoch_monitor_metrics = {
                key: float(np.average(values)) if values else float('nan')
                for key, values in epoch_monitor_metrics.items()
            }
            vali_loss = self.vali(vali_data, vali_loader, criterion) if use_early_stopping else float('nan')
            test_loss = self.vali(test_data, test_loader, criterion) if use_early_stopping else float('nan')

            if use_train_plateau:
                print(
                    "[{0}] Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Top1Proxy: {4:.7f} TopKProxy: {5:.7f} RankICProxy: {6:.7f} HeadConc: {7:.7f} StaticBias: {8:.7f}".format(
                        phase_name,
                        epoch + 1,
                        train_steps,
                        train_loss,
                        epoch_monitor_metrics['monitor_top1_return'],
                        epoch_monitor_metrics['monitor_topk_return'],
                        epoch_monitor_metrics['monitor_rank_ic'],
                        epoch_monitor_metrics['monitor_head_concentration'],
                        epoch_monitor_metrics['monitor_static_bias'],
                    )
                )
            else:
                print(
                    "[{0}] Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                        phase_name, epoch + 1, train_steps, train_loss, vali_loss, test_loss
                    )
                )

            if use_early_stopping:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print(f"{phase_name} early stopping")
                    break
            elif use_train_plateau:
                plateau_metric_value = self._extract_train_plateau_metric_value(train_loss, epoch_monitor_metrics)
                train_plateau(plateau_metric_value, self.model, path)
                if train_plateau.early_stop:
                    print(f"{phase_name} train-loss plateau stopping")
                    break
            else:
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
                print(f"{phase_name} fixed-epoch mode: checkpoint overwritten by current epoch")

            adjust_learning_rate(model_optim, epoch + 1, self.args)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        stage1_train_mode = getattr(self.args, 'train_mode', 'best_val')
        self._run_training_phase(
            phase_name='stage1',
            setting=setting,
            train_data=train_data,
            train_loader=train_loader,
            vali_data=vali_data,
            vali_loader=vali_loader,
            test_data=test_data,
            test_loader=test_loader,
            path=path,
            train_epochs=self.args.train_epochs,
            train_mode=stage1_train_mode,
        )

        stage2_epochs = int(getattr(self.args, 'stage2_epochs', 0))
        if stage2_epochs > 0:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
            stage2_train_mode = getattr(self.args, 'stage2_train_mode', 'fixed_epoch')
            self._run_training_phase(
                phase_name='stage2',
                setting=setting,
                train_data=train_data,
                train_loader=train_loader,
                vali_data=vali_data,
                vali_loader=vali_loader,
                test_data=test_data,
                test_loader=test_loader,
                path=path,
                train_epochs=stage2_epochs,
                train_mode=stage2_train_mode,
            )

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(
                    os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                    map_location="cpu",
                )
            )

        preds = []
        trues = []
        sample_ids = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        enable_visual = not hasattr(test_data, 'build_prediction_frame')
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_aux_x, batch_meta = self._unpack_batch(batch)
                batch_x = batch_x.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_x_mark = batch_x_mark.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y_mark = batch_y_mark.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_aux_x = batch_aux_x.to(self.device, dtype=torch.float32, non_blocking=True) if batch_aux_x is not None else None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_aux_x)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_aux_x)

                forecast, _ = self._split_model_outputs(outputs)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = forecast[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if batch_meta is not None:
                    sample_ids.append(batch_meta.detach().cpu().numpy())
                if enable_visual and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if hasattr(test_data, 'build_prediction_frame') and sample_ids:
            pred_frame = test_data.build_prediction_frame(
                np.concatenate(sample_ids, axis=0),
                preds[:, :, -1:],
                trues[:, :, -1:],
            )
            market_metrics = test_data.evaluate_predictions(pred_frame)
            pred_frame.to_csv(os.path.join(folder_path, 'top1_predictions.csv'), index=False)
            with open(os.path.join(folder_path, 'market_metrics.txt'), 'w') as mf:
                for key, value in market_metrics.items():
                    mf.write(f'{key}: {value}\n')
            print('market metrics:', market_metrics)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
