"""trend.py — 趋势外推模型: 双指数平滑 + 日内模式"""
import numpy as np


class TrendModel:
    """与 LightGBM 互补:
    - LightGBM 擅长多维特征交互(气象/时间/滞后)
    - TrendModel 擅长纯时序趋势和平移(结构性变化)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None

    def fit(self, series: np.ndarray):
        n = len(series)
        if n < 2:
            self.level = series[-1] if n > 0 else 0
            self.trend = 0
            return

        self.level = series[0]
        self.trend = series[1] - series[0]

        for i in range(1, n):
            prev_level = self.level
            prev_trend = self.trend
            self.level = (self.alpha * series[i] +
                          (1 - self.alpha) * (prev_level + prev_trend))
            self.trend = (self.beta * (self.level - prev_level) +
                          (1 - self.beta) * prev_trend)

    def predict(self, steps: int) -> np.ndarray:
        if self.level is None:
            return np.zeros(steps)
        return np.array([self.level + self.trend * (i + 1)
                         for i in range(steps)])

    def predict_with_daily_pattern(self, steps: int,
                                    history: np.ndarray = None) -> np.ndarray:
        base = self.predict(steps)

        if history is not None and len(history) >= 96:
            pattern = np.zeros(96)
            days = len(history) // 96
            if days > 0:
                reshaped = history[-days * 96:].reshape(days, 96)
                daily_mean = np.mean(reshaped, axis=0)
                pattern = daily_mean - np.mean(daily_mean)

            for i in range(min(steps, 96 * 7)):
                base[i] += pattern[i % 96]

        return base
