"""corrector.py — ResidualTransformer: 学习并修正物理模型残差.

架构:
  Input:  (phys_pred, actual, hour_sin, hour_cos, dow_sin, dow_cos) × 6
  Embedding: Linear(6, d_model)
  Positional: Sinusoidal (additive)
  Encoder: N layers × H heads × d_model
  Output: Linear(d_model, 1) → 预测残差

输入序列: [past_window(phys, actual, time)] + [future_horizon(phys, time)]
输出: future_horizon 步残差修正
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch 未安装, Transformer 残差修正不可用")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1), 0, :]
        return self.dropout(x)


class ResidualTransformer(nn.Module):
    """TransformerEncoder 学习物理模型残差.

    Input:  (batch, seq_len, 6) — (phys_pred, actual, h_sin, h_cos, dow_sin, dow_cos)
    Output: (batch, horizon, 1) — predicted residuals
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 seq_len: int = 96, horizon: int = 96, dropout: float = 0.1):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for ResidualTransformer")

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.horizon = horizon

        self.input_proj = nn.Linear(6, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + horizon, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, past_phys, past_actual, past_time_feat,
                future_phys, future_time_feat):
        """前向传播.

        Args:
            past_phys:    (batch, seq_len)    过去物理预测
            past_actual:  (batch, seq_len)    过去实际值
            past_time_feat: (batch, seq_len, 4) 过去时间特征 (h_sin, h_cos, dow_sin, dow_cos)
            future_phys:  (batch, horizon)    未来物理预测
            future_time_feat: (batch, horizon, 4) 未来时间特征

        Returns:
            residuals: (batch, horizon) 预测残差修正
        """
        batch_size = past_phys.size(0)

        # 构造过去段: (batch, seq_len, 6)
        past = torch.stack([
            past_phys,
            past_actual,
            past_time_feat[:, :, 0],
            past_time_feat[:, :, 1],
            past_time_feat[:, :, 2],
            past_time_feat[:, :, 3],
        ], dim=-1)

        # 构造未来段: (batch, horizon, 6) — actual 列填 0
        future = torch.stack([
            future_phys,
            torch.zeros_like(future_phys),
            future_time_feat[:, :, 0],
            future_time_feat[:, :, 1],
            future_time_feat[:, :, 2],
            future_time_feat[:, :, 3],
        ], dim=-1)

        # 拼接: (batch, seq_len + horizon, 6)
        combined = torch.cat([past, future], dim=1)

        # Embed + positional
        x = self.input_proj(combined)  # (batch, total_len, d_model)
        x = self.pos_encoding(x)

        # Causal mask (optional — allows past to attend to future for encoding)
        # 这里不用 causal mask, 让 encoder 自由 attend
        x = self.encoder(x)

        # 只取未来部分: (batch, horizon, d_model)
        future_x = x[:, self.seq_len:, :]
        residuals = self.output_proj(future_x).squeeze(-1)  # (batch, horizon)

        return residuals
