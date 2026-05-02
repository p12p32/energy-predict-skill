"""trend.py — FFT 多重季节性分解 + 趋势外推

基于 FFT 自动检测日内(96步)、周(672步)等周期，
提取主要频率分量进行外推，保留双指数平滑作为 fallback。
"""
import numpy as np
from typing import Dict, Optional, Tuple


class TrendModel:
    """FFT 季节性分解 + Holt-Winters 风格趋势外推."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
        self.daily_pattern = None    # 日内模式 (96)
        self.weekly_pattern = None   # 周模式 (672)
        self.fft_freqs = None
        self.fft_amps = None
        self.fft_phases = None
        self.has_fft = False
        self.period = 96             # 默认周期

    def fit(self, series: np.ndarray):
        n = len(series)
        if n < 96:
            self._fit_holt(series)
            return

        self.period = self._detect_period(series)
        if self.period and n >= 2 * self.period:
            self._fit_fft(series, n)
        else:
            self._fit_holt(series)

    def _detect_period(self, series: np.ndarray) -> Optional[int]:
        """自相关法检测主周期."""
        n = len(series)
        if n < 96:
            return None

        x = series - np.mean(series)
        var = np.dot(x, x)
        if var < 1e-10:
            return 96  # fallback

        best_period = 96
        best_corr = -1.0

        for period in [96, 672, 96 * 2, 48]:
            if n < 2 * period:
                continue
            corr = np.dot(x[:n - period], x[period:]) / var
            if corr > best_corr:
                best_corr = corr
                best_period = period

        return best_period if best_corr > 0.1 else 96

    def _fit_fft(self, series: np.ndarray, n: int):
        """FFT 拟合: 提取主要频率分量."""
        self.level = float(np.mean(series[-self.period:]))
        detrended = series - np.linspace(
            series[0], series[-1], n
        )

        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n)
        amps = np.abs(fft_vals)
        phases = np.angle(fft_vals)

        # 保留能量占前 95% 的频率分量
        total_energy = np.sum(amps ** 2)
        if total_energy < 1e-10:
            self._fit_holt(series)
            return

        sorted_idx = np.argsort(amps)[::-1]
        cumsum = 0.0
        keep = set()
        for idx in sorted_idx:
            cumsum += amps[idx] ** 2
            keep.add(idx)
            if cumsum / total_energy > 0.95:
                break

        self.fft_freqs = freqs[list(keep)]
        self.fft_amps = amps[list(keep)]
        self.fft_phases = phases[list(keep)]
        self.has_fft = True

        # 提取日内模式
        days = n // 96
        if days >= 3:
            reshaped = series[-days * 96:].reshape(days, 96)
            self.daily_pattern = np.mean(reshaped, axis=0) - self.level
        else:
            self.daily_pattern = np.zeros(96)

        # 提取周模式
        weeks = n // 672
        if weeks >= 2:
            reshaped = series[-weeks * 672:].reshape(weeks, 672)
            self.weekly_pattern = np.mean(reshaped, axis=0) - np.mean(series[-weeks * 672:])

        # Holt 趋势
        self._fit_holt(series)

    def _fit_holt(self, series: np.ndarray):
        n = len(series)
        if n < 2:
            self.level = series[-1] if n > 0 else 0
            self.trend = 0
            return

        self.level = float(series[0])
        self.trend = float(series[1] - series[0])

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
        """预测: Holt 趋势 + 日内均值模式.

        不用 FFT 外推 (频域泄漏导致发散), 而是用 Holt 线性趋势
        叠加历史的日内均值模式.
        """
        base = self.predict(steps)

        if history is not None and len(history) >= 96:
            pattern = np.zeros(96)
            days = len(history) // 96
            if days > 0:
                reshaped = history[-days * 96:].reshape(days, 96)
                daily_mean = np.mean(reshaped, axis=0)
                pattern = daily_mean - np.mean(daily_mean)

            # 叠加日内模式 (带衰减)
            for i in range(min(steps, 96 * 7)):
                day_idx = i // 96
                decay = np.exp(-0.02 * day_idx)  # 每天衰减 2%
                base[i] += pattern[i % 96] * decay

        return base

    def _predict_fft(self, steps: int) -> np.ndarray:
        """FFT 外推预测 (带振幅衰减防止发散)."""
        n_hist = 0
        if self.fft_freqs is not None:
            n_hist = len(self.fft_freqs)

        # Holt 趋势基线
        base_trend = self.predict(steps)

        # 叠加 FFT 季节分量，远期振幅衰减
        result = base_trend.copy()
        if self.fft_freqs is not None:
            for i in range(steps):
                t = n_hist + i
                seasonal = 0.0
                # 衰减因子: 远期预测振幅逐渐减小
                decay = np.exp(-0.01 * i)
                for freq, amp, phase in zip(
                    self.fft_freqs, self.fft_amps, self.fft_phases
                ):
                    seasonal += decay * amp * np.cos(2 * np.pi * freq * t + phase)
                result[i] += seasonal

        # 限幅: 不超过基线趋势的 ±50%
        if len(base_trend) > 0 and abs(base_trend[0]) > 1e-8:
            baseline = np.maximum(np.abs(base_trend), 1.0)
            max_dev = baseline * 0.5
            result = np.clip(result, base_trend - max_dev, base_trend + max_dev)

        return result

    def decompose(self, series: np.ndarray) -> Dict[str, np.ndarray]:
        """分解为 trend + seasonal + residual."""
        n = len(series)
        trend = np.zeros(n)
        seasonal = np.zeros(n)
        residual = np.zeros(n)

        if self.has_fft and self.fft_freqs is not None:
            for i in range(n):
                trend[i] = self.level + self.trend * i
                s = 0.0
                for freq, amp, phase in zip(
                    self.fft_freqs, self.fft_amps, self.fft_phases
                ):
                    s += amp * np.cos(2 * np.pi * freq * i + phase)
                seasonal[i] = s
            residual = series - trend - seasonal
        else:
            for i in range(n):
                trend[i] = self.level + self.trend * i
            residual = series - trend

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }
