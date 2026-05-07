"""price_classify.py — 价格区间分类: 5类(负/低/中/高/极端) LGB 多分类.

负电价独立分类 (value<0), 避免被低正价平均.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scripts.ml.layers.base import BaseLayer
import logging

logger = logging.getLogger(__name__)

PRICE_CLASSES = ["negative", "low", "normal", "high", "extreme"]
N_PRICE_CLASSES = 5


class PriceClassifyLayer(BaseLayer):
    """5 分类: 负价 / 低 / 中 / 高 / 极端.

    负价阈值: value < 0 (绝对)
    分位数阈值 (数据驱动, 仅对 value≥0 计算):
      low:     0 ~ Q33
      normal:  Q33 ~ Q67
      high:    Q67 ~ Q95
      extreme: > max(Q95, extreme_floor)
    """

    def __init__(self, low_q: float = 0.33, high_q: float = 0.67,
                 extreme_q: float = 0.95, extreme_floor: float = 800.0):
        super().__init__("price_classify")
        self.low_q = low_q
        self.high_q = high_q
        self.extreme_q = extreme_q
        self.extreme_floor = extreme_floor
        self.thresholds = {}
        self.class_names = list(PRICE_CLASSES)

    def _build_labels(self, values: np.ndarray) -> np.ndarray:
        """构造 5 分类标签: 0=negative, 1=low, 2=normal, 3=high, 4=extreme."""
        # 分位数仅对非负值计算
        nonneg = values[values >= 0]
        if len(nonneg) < 10:
            nonneg = values

        low_t = float(np.quantile(nonneg, self.low_q))
        high_t = float(np.quantile(nonneg, self.high_q))
        extreme_t_raw = float(np.quantile(nonneg, self.extreme_q))
        extreme_t = max(extreme_t_raw, self.extreme_floor)

        self.thresholds = {
            "negative_max": 0.0,
            "low": round(float(low_t), 2),
            "high": round(float(high_t), 2),
            "extreme": round(float(extreme_t), 2),
        }

        labels = np.zeros(len(values), dtype=int)             # default: negative (0)
        labels[(values >= 0) & (values < low_t)] = 1          # low
        labels[(values >= low_t) & (values < high_t)] = 2     # normal
        labels[(values >= high_t) & (values < extreme_t)] = 3  # high
        labels[values >= extreme_t] = 4                       # extreme

        return labels

    def train(self, df: pd.DataFrame, target_type: str = "",
              feature_names: list = None, target_col: str = "value",
              extreme_floor: float = None, **kwargs) -> dict:
        self.feature_names = feature_names or []
        if extreme_floor is not None:
            self.extreme_floor = extreme_floor

        y_vals = df[target_col].values
        labels = self._build_labels(y_vals)

        counts = np.bincount(labels, minlength=N_PRICE_CLASSES)
        ratios = counts / counts.sum()
        logger.info("PriceClassify 标签分布: neg=%.1f%%, low=%.1f%%, normal=%.1f%%, "
                     "high=%.1f%%, extreme=%.1f%%",
                     ratios[0]*100, ratios[1]*100, ratios[2]*100,
                     ratios[3]*100, ratios[4]*100)
        logger.info("  阈值: neg<0, low<%.1f, high<%.1f, extreme<%.1f",
                     self.thresholds["low"], self.thresholds["high"],
                     self.thresholds["extreme"])

        # 极端样本不足时放宽阈值
        if ratios[4] < 0.02 and self.extreme_q > 0.90:
            self.extreme_q -= 0.03
            logger.info("  极端样本不足, 放宽 extreme_q → %.2f", self.extreme_q)
            labels = self._build_labels(y_vals)
            counts = np.bincount(labels, minlength=N_PRICE_CLASSES)
            ratios = counts / counts.sum()

        X = self._extract_features(df)
        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_PRICE_CLASSES,
            n_estimators=300, num_leaves=31, learning_rate=0.03,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        self.model.fit(X, labels)
        self._trained = True

        return {
            "active": True,
            "n_classes": N_PRICE_CLASSES,
            "class_distribution": {PRICE_CLASSES[i]: round(float(r), 4)
                                   for i, r in enumerate(ratios)},
            "thresholds": self.thresholds,
            "extreme_floor": self.extreme_floor,
            "n_samples": len(df),
            "n_features": len(self.feature_names),
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """返回 (n, 5) 概率矩阵."""
        if not self._trained or self.model is None:
            default = np.zeros(N_PRICE_CLASSES)
            default[2] = 1.0  # fallback to normal
            return np.tile(default, (len(df), 1))

        X = self._extract_features(df)
        booster = getattr(self.model, "_Booster", None)
        if booster is not None:
            probs = booster.predict(X)
            if probs.ndim == 1:
                probs = probs.reshape(-1, N_PRICE_CLASSES)
        else:
            probs = self.model.predict_proba(X)

        if probs.shape[1] != N_PRICE_CLASSES:
            default = np.zeros(N_PRICE_CLASSES)
            default[2] = 1.0
            full = np.tile(default, (len(df), 1))
            n_copy = min(probs.shape[1], N_PRICE_CLASSES)
            full[:, :n_copy] = probs[:, :n_copy]
            return full

        return probs

    def load(self, path: str) -> None:
        """加载模型并恢复 extreme_floor."""
        super().load(path)
        if "extreme_floor" in self.metadata:
            self.extreme_floor = self.metadata["extreme_floor"]
        # 恢复 thresholds
        if "thresholds" in self.metadata:
            self.thresholds = self.metadata["thresholds"]

    def predict_class(self, df: pd.DataFrame) -> np.ndarray:
        """返回硬分类 (0/1/2/3/4)."""
        probs = self.predict(df)
        return np.argmax(probs, axis=1)
