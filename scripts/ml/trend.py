"""trend.py — 趋势外推模型: 近期加权日内模式 (Holt 基线退化为常量)."""
import numpy as np


class TrendModel:
    """近期加权日内模式: 用最近 7 天指数加权平均作为基线模式.

    不再使用 Holt 线性外推 — 线性趋势会污染谷底时机(斜率导致相位偏移).
    对电价等强周期型, 日内模式的相位准确性远比趋势斜率重要.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None

    def fit(self, series: np.ndarray):
        n = len(series)
        if n < 2:
            self.level = float(series[-1]) if n > 0 else 0.0
            self.trend = 0.0
            return

        # Holt 拟合保留用于 level 估计 (仅 predict 使用常量)
        self.level = float(series[0])
        self.trend = float(series[1] - series[0])

        for i in range(1, n):
            prev_level = self.level
            self.level = (self.alpha * float(series[i]) +
                          (1 - self.alpha) * (prev_level + self.trend))
            self.trend = (self.beta * (self.level - prev_level) +
                          (1 - self.beta) * self.trend)

    def predict(self, steps: int) -> np.ndarray:
        if self.level is None:
            return np.zeros(steps)
        # 常量基线: 退化为 level, 不做线性外推
        return np.full(steps, self.level)

    def predict_with_daily_pattern(self, steps: int,
                                    history: np.ndarray = None) -> np.ndarray:
        """近期加权日内模式 + 常量基线.

        history: 历史 value 数组 (最近在最右).
        取最近 7 天做指数加权日模式, 越近权重越大.
        不叠加 Holt 趋势 — 避免相位偏移.
        """
        base = self.predict(steps)

        if history is not None and len(history) >= 96:
            pts_per_day = 96
            total_days = len(history) // pts_per_day
            use_days = min(total_days, 7)

            if use_days > 0:
                reshaped = history[-use_days * pts_per_day:].reshape(use_days, pts_per_day)
                weights = np.exp(-np.arange(use_days)[::-1] * 0.4)
                weights /= weights.sum()
                daily_pattern = np.average(reshaped, axis=0, weights=weights)
                pattern = daily_pattern - np.mean(daily_pattern)

                for i in range(steps):
                    base[i] += pattern[i % pts_per_day]

        return base
