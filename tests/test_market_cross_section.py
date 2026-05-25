from types import SimpleNamespace

import torch

from utils.market_cross_section import MarketCrossSectionModel


class DummyBaseModel(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def encode_market_sequence(self, x_enc, x_mark_enc):
        batch, seq_len, _ = x_enc.shape
        base = torch.arange(batch * seq_len * self.d_model, dtype=torch.float32)
        return base.reshape(batch, seq_len, self.d_model)


class ShortSeqDummyBaseModel(torch.nn.Module):
    def __init__(self, d_model, latent_seq_len):
        super().__init__()
        self.d_model = d_model
        self.latent_seq_len = latent_seq_len

    def encode_market_sequence(self, x_enc, x_mark_enc):
        batch = x_enc.shape[0]
        base = torch.arange(batch * self.latent_seq_len * self.d_model, dtype=torch.float32)
        return base.reshape(batch, self.latent_seq_len, self.d_model)


def test_market_cross_section_model_keeps_recent_token_views():
    configs = SimpleNamespace(
        seq_len=20,
        pred_len=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        dropout=0.0,
        market_cs_layers=1,
        market_cs_dropout=0.0,
        market_cs_recent_k=5,
    )
    model = MarketCrossSectionModel(DummyBaseModel(d_model=configs.d_model), configs)
    x_enc = torch.randn(7, configs.seq_len, 24)

    outputs = model(x_enc, None, None, None)

    assert outputs["forecast"].shape == (7, configs.pred_len, 1)
    assert outputs["recent_raw_tokens"].shape == (7, configs.market_cs_recent_k, configs.d_model)
    assert outputs["recent_cs_tokens"].shape == (7, configs.market_cs_recent_k, configs.d_model)


def test_market_cross_section_model_pads_short_backbone_sequences():
    configs = SimpleNamespace(
        seq_len=20,
        pred_len=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        dropout=0.0,
        market_cs_layers=1,
        market_cs_dropout=0.0,
        market_cs_recent_k=5,
    )
    model = MarketCrossSectionModel(
        ShortSeqDummyBaseModel(d_model=configs.d_model, latent_seq_len=2),
        configs,
    )
    x_enc = torch.randn(7, configs.seq_len, 24)

    outputs = model(x_enc, None, None, None)

    assert outputs["forecast"].shape == (7, configs.pred_len, 1)
    assert outputs["recent_raw_tokens"].shape == (7, configs.market_cs_recent_k, configs.d_model)
    assert outputs["recent_cs_tokens"].shape == (7, configs.market_cs_recent_k, configs.d_model)
    assert torch.allclose(outputs["recent_raw_tokens"][:, :3, :], torch.zeros(7, 3, configs.d_model))
