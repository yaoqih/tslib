import torch
import torch.nn.functional as F


def combine_market_multitask_losses(reg_loss, cls_loss, cls_weight):
    total_loss = reg_loss + cls_weight * cls_loss
    return {
        "reg_loss": reg_loss,
        "cls_loss": cls_loss,
        "total_loss": total_loss,
    }


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
