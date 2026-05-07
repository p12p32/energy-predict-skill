"""delta.py — Delta 层: 短期相对误差线性修正. Ridge 保方向信号, 无树模型压缩."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.ml.layers.base import BaseLayer
import logging

logger = logging.getLogger(__name__)


class DeltaLayer(BaseLayer):
    def __init__(self, clip: float = 0.5, linear: bool = True, alpha: float = 1.0):
        super().__init__("delta")
        self.clip = clip
        self.eps = 0.01
        self.linear = linear
        self.alpha = alpha
        self.scaler = None

    def load(self, path: str) -> None:
        super().load(path)
        self.linear = self.metadata.get("linear", False)  # 旧模型默认 LGBM
        self.clip = self.metadata.get("clip", 0.5)
        self.eps = self.metadata.get("eps", 0.01)

    def train(self, df: pd.DataFrame, level_oof: np.ndarray = None,
              feature_names: list = None, **kwargs) -> dict:
        self.feature_names = feature_names or []
        if level_oof is None:
            raise ValueError("Delta 层需要 level_oof")

        y = df["value"].values
        self.eps = max(np.quantile(np.abs(level_oof[level_oof > 0]), 0.01)
                       if (level_oof > 0).any() else 0.01, 0.01)

        denom = np.maximum(np.abs(level_oof), self.eps)
        y_delta = (y - level_oof) / denom
        y_delta = np.clip(y_delta, -self.clip * 2, self.clip * 2)

        valid = np.isfinite(y_delta) & np.isfinite(level_oof)
        if not valid.all():
            logger.warning("Delta 训练: 过滤 %d/%d NaN/Inf 行", (~valid).sum(), len(y))
            y_delta = y_delta[valid]
            df = df.loc[valid].reset_index(drop=True)
            level_oof = level_oof[valid]

        X = self._build_X(df, level_oof)

        if self.linear:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)
            self.model.fit(X_scaled, y_delta)

            weights = self.model.coef_
            top_idx = np.argsort(np.abs(weights))[-5:][::-1]
            top_feats = [(self._col_names[i] if i < len(self._col_names) else f"f{i}",
                         round(float(weights[i]), 4)) for i in top_idx]
            logger.info("Delta(Ridge) 训练完成: eps=%.4f, intercept=%.4f, top=%s",
                         self.eps, float(self.model.intercept_), top_feats)
        else:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                objective="regression", n_estimators=500, num_leaves=63,
                learning_rate=0.02, random_state=42, verbose=-1,
            )
            self.model.fit(X, y_delta)
            logger.info("Delta(LGBM) 训练完成: eps=%.4f", self.eps)

        self._trained = True
        self.metadata["linear"] = self.linear
        self.metadata["clip"] = self.clip
        self.metadata["eps"] = self.eps

        return {"eps": self.eps, "y_mean": float(y_delta.mean()), "y_std": float(y_delta.std()),
                "linear": self.linear}

    def predict(self, df: pd.DataFrame, level_pred: np.ndarray = None, **kwargs) -> np.ndarray:
        if level_pred is None:
            raise ValueError("Delta 层需要 level_pred")
        X = self._build_X(df, level_pred)
        if self.linear and self.scaler is not None:
            X = self.scaler.transform(X)
        delta = self.model.predict(X)
        delta = np.clip(delta, -self.clip, self.clip)
        return delta

    def _build_X(self, df: pd.DataFrame, level_vals: np.ndarray) -> np.ndarray:
        df = df.copy()
        level_col = "level_oof" if "level_oof" in self.feature_names else "level_pred"
        df[level_col] = level_vals

        cols = list(self.feature_names)
        if level_col not in cols:
            cols.append(level_col)

        self._col_names = list(cols)
        X = np.zeros((len(df), len(cols)), dtype=np.float64)
        for i, fn in enumerate(cols):
            if fn in df.columns:
                col = pd.to_numeric(df[fn], errors="coerce").fillna(0).values
                X[:, i] = col
        return X
