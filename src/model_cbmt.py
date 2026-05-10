from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x)).squeeze(-1)


class CBMTTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 4, n_encoder_layers: int = 2, dropout: float = 0.1, head_hidden_dim: int = 128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers=n_encoder_layers)
        head_in = d_model + 6 * 4
        self.visit_head = MLPHead(head_in, head_hidden_dim)
        self.txn_head = MLPHead(head_in, head_hidden_dim)
        self.payment_head = MLPHead(head_in, head_hidden_dim)
        self.aggregate_heads = nn.ModuleDict({
            "agg_visits": MLPHead(head_in, head_hidden_dim),
            "agg_txns": MLPHead(head_in, head_hidden_dim),
            "agg_revenue": MLPHead(head_in, head_hidden_dim),
        })

    def _lag_summary(self, x: torch.Tensor) -> torch.Tensor:
        behavior = x[:, :, :6]
        last = behavior[:, -1, :]
        mean = behavior.mean(dim=1)
        first = behavior[:, 0, :]
        slope = (last - first) / max(behavior.size(1) - 1, 1)
        std = behavior.std(dim=1, unbiased=False)
        return torch.cat([last, mean, slope, std], dim=1)

    def forward(self, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_projection(x_seq)
        h = self.position(h)
        h = self.backbone(h)[:, -1, :]
        z = torch.cat([h, self._lag_summary(x_seq)], dim=1)
        return {
            "visits_pc": self.visit_head(z),
            "txns_pc": self.txn_head(z),
            "avg_payment": self.payment_head(z),
            "agg_visits": self.aggregate_heads["agg_visits"](z),
            "agg_txns": self.aggregate_heads["agg_txns"](z),
            "agg_revenue": self.aggregate_heads["agg_revenue"](z),
        }
