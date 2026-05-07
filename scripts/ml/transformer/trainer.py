"""trainer.py — Transformer 训练器: 从物理模型残差学习修正.

训练数据: 历史 (phys_pred, actual) 序列 → 残差 = actual − phys_pred
优化器: Adam, MSE loss
"""
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ResidualDataset(Dataset):
    """残差序列数据集.

    每样本:
      - past_seq:  (seq_len,) phys_pred + actual + time features
      - future_residuals: (horizon,) actual − phys_pred
    """

    def __init__(self, phys_pred: np.ndarray, actuals: np.ndarray,
                 dts, seq_len: int = 96, horizon: int = 96):
        self.phys_pred = phys_pred.astype(np.float32)
        self.actuals = actuals.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_samples = len(phys_pred) - seq_len - horizon + 1

        # 时间特征
        import pandas as pd
        dts = pd.to_datetime(dts)
        if isinstance(dts, pd.DatetimeIndex):
            hours = dts.hour.astype(float).values + dts.minute.astype(float).values / 60.0
            dow = dts.dayofweek.astype(float).values
        else:
            hours = dts.dt.hour.values.astype(float) + dts.dt.minute.values.astype(float) / 60.0
            dow = dts.dt.dayofweek.values.astype(float)

        self.h_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        self.h_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
        self.dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
        self.dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)

        self.time_feat = np.stack([self.h_sin, self.h_cos,
                                    self.dow_sin, self.dow_cos], axis=-1)

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        s, h = self.seq_len, self.horizon

        past_phys = self.phys_pred[idx:idx + s]
        past_actual = self.actuals[idx:idx + s]
        past_time = self.time_feat[idx:idx + s]

        future_phys = self.phys_pred[idx + s:idx + s + h]
        future_actual = self.actuals[idx + s:idx + s + h]
        future_time = self.time_feat[idx + s:idx + s + h]
        future_residuals = future_actual - future_phys

        return {
            "past_phys": torch.tensor(past_phys),
            "past_actual": torch.tensor(past_actual),
            "past_time": torch.tensor(past_time),
            "future_phys": torch.tensor(future_phys),
            "future_time": torch.tensor(future_time),
            "target": torch.tensor(future_residuals),
        }


class TransformerTrainer:
    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 seq_len: int = 96, horizon: int = 96, dropout: float = 0.1,
                 lr: float = 0.001, batch_size: int = 32, epochs: int = 50,
                 patience: int = 10, device: str = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TransformerTrainer")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.train_losses = []
        self.val_losses = []

    def build_model(self):
        from scripts.ml.transformer.corrector import ResidualTransformer
        self.model = ResidualTransformer(
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            seq_len=self.seq_len, horizon=self.horizon, dropout=self.dropout)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, phys_pred: np.ndarray, actuals: np.ndarray, dts,
            val_split: float = 0.15):
        self.build_model()

        dataset = ResidualDataset(phys_pred, actuals, dts,
                                  seq_len=self.seq_len, horizon=self.horizon)
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val

        if n_train <= 0 or n_val <= 0:
            logger.warning("数据不足 (n=%d), 跳过 Transformer 训练", len(dataset))
            return

        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.patience // 2, factor=0.5)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                past_phys = batch["past_phys"].to(self.device)
                past_actual = batch["past_actual"].to(self.device)
                past_time = batch["past_time"].to(self.device)
                future_phys = batch["future_phys"].to(self.device)
                future_time = batch["future_time"].to(self.device)
                target = batch["target"].to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(past_phys, past_actual, past_time,
                                  future_phys, future_time)
                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    past_phys = batch["past_phys"].to(self.device)
                    past_actual = batch["past_actual"].to(self.device)
                    past_time = batch["past_time"].to(self.device)
                    future_phys = batch["future_phys"].to(self.device)
                    future_time = batch["future_time"].to(self.device)
                    target = batch["target"].to(self.device)
                    pred = self.model(past_phys, past_actual, past_time,
                                      future_phys, future_time)
                    val_loss += criterion(pred, target).item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d: train_loss=%.4f, val_loss=%.4f",
                             epoch + 1, train_loss, val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.cpu().clone()
                                    for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
        logger.info("Transformer 训练完成: best_val_loss=%.4f", best_val_loss)

    def predict_residuals(self, past_phys: np.ndarray, past_actual: np.ndarray,
                          future_phys: np.ndarray, past_time: np.ndarray = None,
                          future_time: np.ndarray = None) -> np.ndarray:
        """预测未来残差修正.

        Args:
            past_phys: (seq_len,) 或 (batch, seq_len)
            past_actual: (seq_len,) 或 (batch, seq_len)
            future_phys: (horizon,) 或 (batch, horizon)
            past_time: (seq_len, 4) or None → auto-generate from 0-index
            future_time: (horizon, 4) or None → auto-generate

        Returns:
            residuals: (horizon,) or (batch, horizon)
        """
        if self.model is None:
            return np.zeros_like(future_phys)

        single = (past_phys.ndim == 1)
        if single:
            past_phys = past_phys[np.newaxis, :]
            past_actual = past_actual[np.newaxis, :]
            future_phys = future_phys[np.newaxis, :]

        # Time features fallback
        if past_time is None:
            past_time = self._default_time_feat(past_phys.shape[-1])
        if future_time is None:
            future_time = self._default_time_feat(future_phys.shape[-1], offset=past_phys.shape[-1])

        if past_time.ndim == 2:
            past_time = past_time[np.newaxis, :, :]
        if future_time.ndim == 2:
            future_time = future_time[np.newaxis, :, :]

        self.model.eval()
        with torch.no_grad():
            past_phys_t = torch.tensor(past_phys.astype(np.float32)).to(self.device)
            past_actual_t = torch.tensor(past_actual.astype(np.float32)).to(self.device)
            past_time_t = torch.tensor(past_time.astype(np.float32)).to(self.device)
            future_phys_t = torch.tensor(future_phys.astype(np.float32)).to(self.device)
            future_time_t = torch.tensor(future_time.astype(np.float32)).to(self.device)
            residuals = self.model(past_phys_t, past_actual_t, past_time_t,
                                    future_phys_t, future_time_t)
            residuals = residuals.cpu().numpy()

        if single:
            residuals = residuals[0]
        return residuals

    @staticmethod
    def _default_time_feat(n: int, offset: int = 0) -> np.ndarray:
        idx = np.arange(n) + offset
        hours = (idx % 96) / 4.0  # assumes 15min steps
        dow = (idx // 96) % 7
        return np.stack([
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ], axis=-1).astype(np.float32)

    def save(self, path: str) -> None:
        if self.model is None:
            return
        torch.save({
            "model_state": self.model.state_dict(),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "seq_len": self.seq_len,
            "horizon": self.horizon,
            "dropout": self.dropout,
        }, path)

    def load(self, path: str) -> None:
        self.build_model()
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()


# PyTorch 不存在时的 fallback
if not HAS_TORCH:
    import torch.nn as nn  # noqa — won't execute
