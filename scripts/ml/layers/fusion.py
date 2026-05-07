"""fusion.py — 组件融合层: Ridge 线性融合 base/ts_pred/trend + 时间特征.

TrendClassify 方向信号在外面做非线性门控 (pipeline._apply_trend_gating),
不进入 Ridge — 避免方向信息被线性平均掉.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.ml.layers.base import BaseLayer
import logging

logger = logging.getLogger(__name__)

META_COLS = [
    "base", "ts_pred", "trend",
    "step", "hour", "dow", "is_weekend", "season",
    "extreme_weather", "layer_agreement",
]


class FusionLayer(BaseLayer):
    def __init__(self, linear: bool = True, alpha: float = 1.0):
        super().__init__("fusion")
        self._meta_cols = list(META_COLS)
        self.linear = linear
        self.alpha = alpha  # Ridge 正则化强度
        self.scaler = None

    def load(self, path: str) -> None:
        super().load(path)
        stored = self.metadata.get("_meta_cols")
        if stored:
            self._meta_cols = stored
        self.linear = self.metadata.get("linear", False)  # 旧模型默认 LGBM

    def train(self, meta_df: pd.DataFrame, y: np.ndarray = None, **kwargs) -> dict:
        self._meta_cols = [c for c in META_COLS if c in meta_df.columns]
        X = meta_df[self._meta_cols].fillna(0).values
        if y is None:
            y = meta_df["actual"].values

        if self.linear:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model = Ridge(alpha=self.alpha, fit_intercept=True)
            self.model.fit(X_scaled, y)

            # 报告权重
            weights = self.model.coef_
            named_w = [(self._meta_cols[i], round(float(weights[i]), 4))
                       for i in np.argsort(np.abs(weights))[-6:][::-1]]
            intercept = float(self.model.intercept_)
            logger.info("融合层(Ridge)训练完成: features=%d, samples=%d, intercept=%.1f",
                         len(self._meta_cols), len(y), intercept)
            logger.info("  权重: %s", named_w)
        else:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                objective="regression", n_estimators=300, num_leaves=31,
                learning_rate=0.02, random_state=42, verbose=-1,
            )
            self.model.fit(X, y)
            importances = self.model.feature_importances_
            top_idx = np.argsort(importances)[-5:][::-1]
            top_feats = [(self._meta_cols[i], round(float(importances[i]), 4)) for i in top_idx]
            logger.info("融合层(LGBM)训练完成: features=%d, samples=%d, top=%s",
                         len(self._meta_cols), len(y), top_feats)

        self._trained = True
        self.metadata["_meta_cols"] = self._meta_cols
        self.metadata["linear"] = self.linear

        return {"n_features": len(self._meta_cols), "n_samples": len(y), "linear": self.linear}

    def predict(self, meta_df: pd.DataFrame, **kwargs) -> np.ndarray:
        cols = [c for c in self._meta_cols if c in meta_df.columns]
        X = meta_df[cols].fillna(0).values
        if self.linear and self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def build_meta_features(self, base: np.ndarray, ts_pred: np.ndarray,
                            trend: np.ndarray, future_df: pd.DataFrame) -> pd.DataFrame:
        """构建融合层 meta-features (纯组件 + 时间特征, 不含趋势方向).

        TrendClassify 方向信号在外部做非线性门控, 不进入 Ridge.
        """
        n = len(base)
        eps = np.maximum(np.abs(base), 1.0)
        layer_agreement = 1.0 - np.clip(np.abs(base - ts_pred) / eps, 0, 1)

        meta = pd.DataFrame({
            "base": base,
            "ts_pred": ts_pred,
            "trend": trend[:n],
            "step": np.arange(n) / max(n, 1),
            "layer_agreement": layer_agreement,
        })

        for col, default in [("hour", 0), ("day_of_week", 0),
                              ("is_weekend", 0), ("season", 0),
                              ("extreme_weather_flag", 0)]:
            if col in future_df.columns:
                meta["dow" if col == "day_of_week" else
                     "extreme_weather" if col == "extreme_weather_flag" else
                     col] = future_df[col].values[:n]
            else:
                name = ("dow" if col == "day_of_week" else
                        "extreme_weather" if col == "extreme_weather_flag" else col)
                if name not in meta.columns:
                    meta[name] = default

        return meta


def apply_trend_gating(ensemble: np.ndarray, base: np.ndarray,
                       ts_pred: np.ndarray, trend: np.ndarray,
                       trend_probs, target_type: str = "",
                       future_df=None) -> np.ndarray:
    """TrendClassify 方向门控 + 太阳能时段趋势偏置.

    两层机制:
      1. 方向门控: TrendClassify 预测 up/down → 方向对齐组件主导
      2. 太阳能趋势偏置: H9-H16 太阳能>30%峰值 → 趋势组件有基线权重
         解决谷底时机滞后问题 (趋势的 7d 加权日模式相位更准确).

    风电/光伏/联络线/非市场禁用门控.
    """
    if trend_probs is None or trend_probs.shape[1] < 3:
        return ensemble

    skip_types = {"风电", "光伏", "联络线", "非市场"}
    if any(kw in target_type for kw in skip_types):
        return ensemble

    n = len(ensemble)
    p_down = trend_probs[:n, 0]
    p_flat = trend_probs[:n, 1]
    p_up = trend_probs[:n, 2]

    direction = p_up - p_down              # [-1, 1]
    confidence = p_up + p_down             # [0, 1], 方向性概率
    gate = np.abs(direction) * confidence  # [0, 1]

    # 方向对齐组件: UP→取 max(base,ts,trend), DOWN→取 min
    components = np.column_stack([base[:n], ts_pred[:n], trend[:n]])
    up_mask = direction > 0.05
    down_mask = direction < -0.05

    trend_aligned = ensemble.copy()
    trend_aligned[up_mask] = np.max(components[up_mask], axis=1)
    trend_aligned[down_mask] = np.min(components[down_mask], axis=1)

    gate = np.clip(gate, 0.0, 0.85)
    result = gate * trend_aligned + (1.0 - gate) * ensemble

    # 太阳能时段趋势偏置: 修正谷底时机
    if future_df is not None and "solar_radiation" in future_df.columns:
        solar = future_df["solar_radiation"].values[:n]
        solar_max = np.max(solar) + 1e-6
        if solar_max > 10:
            solar_norm = solar / solar_max           # [0, 1]
            solar_mask = solar_norm > 0.3            # 有效日照时段
            solar_bias = solar_norm * solar_mask * 0.35  # 最多 35% 趋势
            # 仅电价/负荷启用 (出力类走 skip_types 上面已返回)
            result = (1.0 - solar_bias) * result + solar_bias * trend[:n]

    return result
