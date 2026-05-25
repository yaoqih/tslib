import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSectionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MarketCrossSectionModel(nn.Module):
    def __init__(self, base_model, configs):
        super().__init__()
        self.base_model = base_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.recent_k = max(1, int(getattr(configs, "market_cs_recent_k", 5)))
        if not hasattr(base_model, "encode_market_sequence"):
            raise ValueError(
                f"{base_model.__class__.__name__} does not expose encode_market_sequence(x_enc, x_mark_enc); "
                "market cross-section mode now requires true [stock, seq, latent] backbone features."
            )

        self.pre_norm = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.cross_section_layers = nn.ModuleList(
            [
                CrossSectionBlock(
                    d_model=self.d_model,
                    n_heads=getattr(configs, "market_cs_n_heads", configs.n_heads),
                    d_ff=getattr(configs, "market_cs_d_ff", configs.d_ff),
                    dropout=getattr(configs, "market_cs_dropout", configs.dropout),
                )
                for _ in range(getattr(configs, "market_cs_layers", 1))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(self.recent_k * self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(getattr(configs, "market_cs_dropout", configs.dropout)),
            nn.Linear(self.d_model, self.pred_len),
        )

    def _take_recent_tokens(self, tokens):
        recent = tokens[:, -self.recent_k :, :]
        pad_len = self.recent_k - recent.size(1)
        if pad_len > 0:
            recent = F.pad(recent, (0, 0, pad_len, 0))
        return recent

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        latent_tokens = self.base_model.encode_market_sequence(x_enc, x_mark_enc)
        latent_tokens = self.pre_norm(latent_tokens)

        cross_section_tokens = latent_tokens.permute(1, 0, 2)
        for layer in self.cross_section_layers:
            cross_section_tokens = layer(cross_section_tokens)
        cross_section_tokens = cross_section_tokens.permute(1, 0, 2)

        recent_raw_tokens = self._take_recent_tokens(latent_tokens)
        recent_cs_tokens = self._take_recent_tokens(cross_section_tokens)
        final_repr = torch.cat([recent_raw_tokens, recent_cs_tokens], dim=-1).reshape(latent_tokens.size(0), -1)

        score = self.head(final_repr).unsqueeze(-1)
        return {
            "forecast": score,
            "backbone_latent": latent_tokens,
            "cross_section_tokens": cross_section_tokens,
            "recent_raw_tokens": recent_raw_tokens,
            "recent_cs_tokens": recent_cs_tokens,
        }
