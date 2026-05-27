import torch
import torch.nn.functional as F


def build_head_union_mask(pred, target, tradable_mask=None, topq_ratio=0.2):
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(device=pred.device, dtype=torch.float32)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    if pred.numel() == 0 or topq_ratio <= 0.0:
        return mask

    if tradable_mask is not None:
        tradable_mask = tradable_mask.reshape(-1).to(dtype=torch.bool, device=pred.device)
        active_idx = torch.nonzero(tradable_mask, as_tuple=False).reshape(-1)
    else:
        active_idx = torch.arange(pred.numel(), device=pred.device)
    if active_idx.numel() == 0:
        return mask

    k = max(1, min(int(torch.ceil(torch.tensor(active_idx.numel() * float(topq_ratio), device=pred.device)).item()), int(active_idx.numel())))
    active_pred = pred.index_select(0, active_idx)
    active_target = target.index_select(0, active_idx)
    pred_top_idx = active_idx.index_select(0, torch.topk(active_pred, k=k, largest=True).indices)
    target_top_idx = active_idx.index_select(0, torch.topk(active_target, k=k, largest=True).indices)
    union_idx = torch.unique(torch.cat([pred_top_idx, target_top_idx], dim=0), sorted=False)
    mask.index_fill_(0, union_idx, True)
    return mask


def build_local_knn_mask(features, tradable_mask=None, neighbor_k=20):
    features = features.reshape(features.shape[0], -1).to(dtype=torch.float32)
    device = features.device
    num_items = int(features.shape[0])
    mask = torch.zeros((num_items, num_items), dtype=torch.bool, device=device)
    if num_items <= 1 or neighbor_k <= 0:
        return mask

    if tradable_mask is not None:
        tradable_mask = tradable_mask.reshape(-1).to(dtype=torch.bool, device=device)
        active_idx = torch.nonzero(tradable_mask, as_tuple=False).reshape(-1)
    else:
        active_idx = torch.arange(num_items, device=device)
    if active_idx.numel() <= 1:
        return mask

    active_features = features.index_select(0, active_idx)
    distances = torch.cdist(active_features, active_features, p=2)
    eye = torch.eye(active_idx.numel(), dtype=torch.bool, device=device)
    distances = distances.masked_fill(eye, float("inf"))
    k = max(1, min(int(neighbor_k), int(active_idx.numel() - 1)))
    neighbor_local = torch.topk(distances, k=k, largest=False).indices
    row_idx = active_idx.unsqueeze(1).expand(-1, k).reshape(-1)
    col_idx = active_idx.index_select(0, neighbor_local.reshape(-1))
    mask[row_idx, col_idx] = True
    mask[col_idx, row_idx] = True
    return mask


def combine_market_multitask_losses(reg_loss, cls_loss, cls_weight):
    total_loss = reg_loss + cls_weight * cls_loss
    return {
        "reg_loss": reg_loss,
        "cls_loss": cls_loss,
        "total_loss": total_loss,
    }


def _apply_mask(values, mask):
    values = values.reshape(-1)
    if mask is None:
        return values
    mask = mask.reshape(-1).to(dtype=torch.bool, device=values.device)
    if mask.numel() != values.numel():
        raise ValueError("Mask size must match values size")
    return values[mask]


def _apply_mask_pair(values, mask, sample_weight=None):
    values = values.reshape(-1)
    if sample_weight is not None:
        sample_weight = sample_weight.reshape(-1).to(device=values.device, dtype=torch.float32)
    if mask is None:
        return values, sample_weight
    mask = mask.reshape(-1).to(dtype=torch.bool, device=values.device)
    if mask.numel() != values.numel():
        raise ValueError("Mask size must match values size")
    masked_values = values[mask]
    masked_weight = sample_weight[mask] if sample_weight is not None else None
    return masked_values, masked_weight


def build_pred_topq_weights(pred, tradable_mask=None, topq_ratio=0.1, topq_weight=1.0):
    pred = pred.reshape(-1)
    weights = torch.ones_like(pred, dtype=torch.float32)
    if pred.numel() == 0 or topq_weight <= 1.0 or topq_ratio <= 0.0:
        return weights

    if tradable_mask is not None:
        tradable_mask = tradable_mask.reshape(-1).to(dtype=torch.bool, device=pred.device)
        active_idx = torch.nonzero(tradable_mask, as_tuple=False).reshape(-1)
    else:
        active_idx = torch.arange(pred.numel(), device=pred.device)

    if active_idx.numel() == 0:
        return weights

    k = max(1, min(int(torch.ceil(torch.tensor(active_idx.numel() * float(topq_ratio))).item()), int(active_idx.numel())))
    active_pred = pred.index_select(0, active_idx)
    top_local_idx = torch.topk(active_pred, k=k, largest=True).indices
    top_global_idx = active_idx.index_select(0, top_local_idx)
    weights.index_fill_(0, top_global_idx, float(topq_weight))
    return weights


def build_union_topq_weights(pred, target, tradable_mask=None, topq_ratio=0.2, topq_weight=1.0):
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(device=pred.device, dtype=torch.float32)
    weights = torch.ones_like(pred, dtype=torch.float32)
    if pred.numel() == 0 or topq_weight <= 1.0 or topq_ratio <= 0.0:
        return weights

    if tradable_mask is not None:
        tradable_mask = tradable_mask.reshape(-1).to(dtype=torch.bool, device=pred.device)
        active_idx = torch.nonzero(tradable_mask, as_tuple=False).reshape(-1)
    else:
        active_idx = torch.arange(pred.numel(), device=pred.device)

    if active_idx.numel() == 0:
        return weights

    k = max(1, min(int(torch.ceil(torch.tensor(active_idx.numel() * float(topq_ratio))).item()), int(active_idx.numel())))
    active_pred = pred.index_select(0, active_idx)
    active_target = target.index_select(0, active_idx)
    pred_top_idx = active_idx.index_select(0, torch.topk(active_pred, k=k, largest=True).indices)
    target_top_idx = active_idx.index_select(0, torch.topk(active_target, k=k, largest=True).indices)
    union_idx = torch.unique(torch.cat([pred_top_idx, target_top_idx], dim=0), sorted=False)
    weights.index_fill_(0, union_idx, float(topq_weight))
    return weights


def build_true_rank_sample_weights(target, alpha=3.0, power=2.0):
    target = target.reshape(-1).to(dtype=torch.float32)
    weights = torch.ones_like(target, dtype=torch.float32)
    if target.numel() == 0 or float(alpha) <= 0.0:
        return weights

    order = torch.argsort(target)
    ranks = torch.empty_like(target, dtype=torch.float32)
    if target.numel() == 1:
        ranks.fill_(0.0)
    else:
        ranks[order] = torch.linspace(0.0, 1.0, steps=target.numel(), device=target.device)
    return weights + float(alpha) * ranks.pow(float(power))


def compute_masked_regression_loss(criterion, pred, target, tradable_mask=None):
    pred = _apply_mask(pred, tradable_mask)
    target = _apply_mask(target, tradable_mask)
    if pred.numel() == 0:
        flat_pred = pred.reshape(-1)
        return flat_pred.new_tensor(0.0)
    return criterion(pred, target)


def compute_weighted_masked_regression_loss(pred, target, tradable_mask=None, sample_weight=None, loss_name="mse", huber_delta=1.0):
    pred, sample_weight = _apply_mask_pair(pred, tradable_mask, sample_weight=sample_weight)
    target, _ = _apply_mask_pair(target, tradable_mask, sample_weight=None)
    if pred.numel() == 0:
        return pred.new_tensor(0.0)

    if loss_name.lower() == "mse":
        per_sample = (pred - target).pow(2)
    elif loss_name.lower() == "mae":
        per_sample = (pred - target).abs()
    elif loss_name.lower() == "huber":
        per_sample = F.huber_loss(pred, target, reduction="none", delta=float(huber_delta))
    else:
        raise ValueError(f"Unsupported weighted regression loss: {loss_name}")

    if sample_weight is None:
        return per_sample.mean()
    denom = sample_weight.sum().clamp_min(1e-12)
    return (per_sample * sample_weight).sum() / denom


def compute_pairwise_rank_loss(pred, target, margin=0.0):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pair_mask = target_diff > 0
    if not torch.any(pair_mask):
        return pred.new_tensor(0.0)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    losses = F.relu(margin - pred_diff[pair_mask])
    if losses.numel() == 0:
        return pred.new_tensor(0.0)
    return losses.mean()


def compute_masked_pairwise_rank_loss(pred, target, tradable_mask=None, margin=0.0):
    pred = _apply_mask(pred, tradable_mask)
    target = _apply_mask(target, tradable_mask)
    if pred.numel() <= 1:
        return pred.new_tensor(0.0)
    return compute_pairwise_rank_loss(pred=pred, target=target, margin=margin)


def compute_weighted_masked_pairwise_rank_loss(pred, target, tradable_mask=None, sample_weight=None, margin=0.0):
    pred, sample_weight = _apply_mask_pair(pred, tradable_mask, sample_weight=sample_weight)
    target, _ = _apply_mask_pair(target, tradable_mask, sample_weight=None)
    if pred.numel() <= 1:
        return pred.new_tensor(0.0)

    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pair_mask = target_diff > 0
    if not torch.any(pair_mask):
        return pred.new_tensor(0.0)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    losses = F.relu(margin - pred_diff)
    if sample_weight is None:
        selected = losses[pair_mask]
        return selected.mean() if selected.numel() > 0 else pred.new_tensor(0.0)

    pair_weight = 0.5 * (sample_weight.unsqueeze(1) + sample_weight.unsqueeze(0))
    selected_losses = losses[pair_mask]
    selected_weights = pair_weight[pair_mask]
    denom = selected_weights.sum().clamp_min(1e-12)
    return (selected_losses * selected_weights).sum() / denom


def compute_grouped_pairwise_rank_loss(pred, target, group_ids, margin=0.0, min_target_gap=0.0):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    group_ids = group_ids.reshape(-1)

    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    same_group = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
    pair_mask = same_group & (target_diff > min_target_gap)
    if not torch.any(pair_mask):
        return pred.new_tensor(0.0)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    losses = F.relu(margin - pred_diff[pair_mask])
    if losses.numel() == 0:
        return pred.new_tensor(0.0)
    return losses.mean()


def compute_winner_pairwise_rank_loss(pred, target, candidate_mask=None, margin=0.0, sample_weight=None, min_target_gap=0.0):
    pred, sample_weight = _apply_mask_pair(pred, candidate_mask, sample_weight=sample_weight)
    target, _ = _apply_mask_pair(target, candidate_mask, sample_weight=None)
    if pred.numel() <= 1:
        return pred.new_tensor(0.0)

    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pair_mask = target_diff > float(min_target_gap)
    if not torch.any(pair_mask):
        return pred.new_tensor(0.0)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    losses = F.relu(float(margin) - pred_diff)
    if sample_weight is None:
        selected = losses[pair_mask]
        return selected.mean() if selected.numel() > 0 else pred.new_tensor(0.0)

    pair_weight = 0.5 * (sample_weight.unsqueeze(1) + sample_weight.unsqueeze(0))
    selected_losses = losses[pair_mask]
    selected_weights = pair_weight[pair_mask]
    denom = selected_weights.sum().clamp_min(1e-12)
    return (selected_losses * selected_weights).sum() / denom


def compute_local_neighbor_pairwise_rank_loss(pred, target, neighbor_mask, margin=0.0, sample_weight=None, min_target_gap=0.0):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    neighbor_mask = neighbor_mask.to(dtype=torch.bool, device=pred.device)
    if pred.numel() <= 1 or neighbor_mask.numel() == 0:
        return pred.new_tensor(0.0)

    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pair_mask = neighbor_mask & (target_diff > float(min_target_gap))
    if not torch.any(pair_mask):
        return pred.new_tensor(0.0)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    losses = F.relu(float(margin) - pred_diff)
    if sample_weight is None:
        selected = losses[pair_mask]
        return selected.mean() if selected.numel() > 0 else pred.new_tensor(0.0)

    sample_weight = sample_weight.reshape(-1).to(device=pred.device, dtype=torch.float32)
    pair_weight = 0.5 * (sample_weight.unsqueeze(1) + sample_weight.unsqueeze(0))
    selected_losses = losses[pair_mask]
    selected_weights = pair_weight[pair_mask]
    denom = selected_weights.sum().clamp_min(1e-12)
    return (selected_losses * selected_weights).sum() / denom


def _compute_unit_interval_ranks(values):
    values = values.reshape(-1)
    ranks = torch.empty_like(values, dtype=torch.float32)
    if values.numel() <= 1:
        ranks.fill_(0.0)
        return ranks
    order = torch.argsort(values)
    ranks[order] = torch.linspace(0.0, 1.0, steps=values.numel(), device=values.device)
    return ranks


def compute_topk_mean_return_proxy(pred, target, top_k=1):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.numel() == 0:
        return pred.new_tensor(0.0)

    k = max(1, min(int(top_k), int(pred.numel())))
    top_idx = torch.topk(pred, k=k, largest=True).indices
    return target[top_idx].mean()


def compute_masked_topk_mean_return_proxy(pred, target, tradable_mask=None, top_k=1):
    pred = _apply_mask(pred, tradable_mask)
    target = _apply_mask(target, tradable_mask)
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    return compute_topk_mean_return_proxy(pred=pred, target=target, top_k=top_k)


def compute_head_concentration_penalty(pred, tradable_mask=None, temperature=1.0):
    pred = _apply_mask(pred, tradable_mask)
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    probs = F.softmax(pred / max(float(temperature), 1e-6), dim=0)
    return probs.max()


def compute_head_gap_penalty(pred, tradable_mask=None, top_k=3):
    pred = _apply_mask(pred, tradable_mask)
    if pred.numel() <= 1:
        return pred.new_tensor(0.0)

    k = max(2, min(int(top_k), int(pred.numel())))
    top_values = torch.topk(pred, k=k, largest=True).values
    top1 = top_values[0]
    head_mean = top_values[1:].mean() if top_values.numel() > 1 else top1
    return F.relu(top1 - head_mean)


def compute_static_bias_surrogate_penalty(pred, tradable_mask=None, top_k=3):
    return compute_head_gap_penalty(
        pred=pred,
        tradable_mask=tradable_mask,
        top_k=top_k,
    )


def compute_rank_ic(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.numel() <= 1:
        return pred.new_tensor(0.0)

    pred_rank = _compute_unit_interval_ranks(pred)
    target_rank = _compute_unit_interval_ranks(target)
    pred_centered = pred_rank - pred_rank.mean()
    target_centered = target_rank - target_rank.mean()
    denom = torch.sqrt(
        pred_centered.pow(2).sum().clamp_min(1e-12)
        * target_centered.pow(2).sum().clamp_min(1e-12)
    )
    if denom.item() <= 0.0:
        return pred.new_tensor(0.0)
    return (pred_centered * target_centered).sum() / denom


def compute_topk_listwise_loss(pred, target, top_k=1, temperature=1.0, target_mode="soft"):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.numel() == 0:
        return pred.new_tensor(0.0)

    k = max(1, min(int(top_k), int(pred.numel())))
    top_idx = torch.topk(target, k=k, largest=True).indices

    log_probs = F.log_softmax(pred / max(float(temperature), 1e-6), dim=0)
    if target_mode == "hard":
        return -log_probs[top_idx[0]]
    if target_mode != "soft":
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    top_target = target[top_idx]
    target_weights = F.softmax(top_target / max(float(temperature), 1e-6), dim=0)
    return -(target_weights * log_probs[top_idx]).sum()


def compute_masked_topk_listwise_loss(pred, target, tradable_mask=None, top_k=1, temperature=1.0, target_mode="soft"):
    pred = _apply_mask(pred, tradable_mask)
    target = _apply_mask(target, tradable_mask)
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    return compute_topk_listwise_loss(
        pred=pred,
        target=target,
        top_k=top_k,
        temperature=temperature,
        target_mode=target_mode,
    )
