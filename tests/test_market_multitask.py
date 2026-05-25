import torch

from utils.market_multitask import (
    build_pred_topq_weights,
    compute_weighted_masked_regression_loss,
)


def test_build_pred_topq_weights_upweights_predicted_head():
    pred = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3], dtype=torch.float32)
    weights = build_pred_topq_weights(
        pred=pred,
        tradable_mask=None,
        topq_ratio=0.4,
        topq_weight=3.0,
    )

    expected = torch.tensor([1.0, 3.0, 1.0, 3.0, 1.0], dtype=torch.float32)
    assert torch.allclose(weights, expected)


def test_build_pred_topq_weights_respects_tradable_mask():
    pred = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3], dtype=torch.float32)
    tradable_mask = torch.tensor([True, False, True, True, False])
    weights = build_pred_topq_weights(
        pred=pred,
        tradable_mask=tradable_mask,
        topq_ratio=0.5,
        topq_weight=4.0,
    )

    expected = torch.tensor([1.0, 1.0, 1.0, 4.0, 1.0], dtype=torch.float32)
    assert torch.allclose(weights, expected)


def test_weighted_masked_regression_loss_matches_manual_weighted_mse():
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
    assert abs(loss.item() - manual) < 1e-6
