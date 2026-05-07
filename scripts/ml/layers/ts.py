"""ts.py — TS 层: Holt 双指数平滑外推. 纯时序, 无树模型压缩."""
import numpy as np
import pandas as pd
from scripts.ml.layers.base import BaseLayer
import logging

logger = logging.getLogger(__name__)

TS_FEATURE_PATTERNS = [
    "value_lag", "value_diff", "value_rolling", "value_sign",
    "value_range", "value_zscore", "value_percentile",
    "pred_error", "hour_x_lag", "weekend_x_lag",
]


class TSLayer(BaseLayer):
    def __init__(self):
        super().__init__("ts")
        self._holt_level = None
        self._holt_trend = None
        self._holt_alpha = 0.3
        self._holt_beta = 0.1
        self._daily_pattern = None
        self._pts_per_day = 96
        self._use_linear = True
        self._recent_mean = 0.0

    def load(self, path: str) -> None:
        super().load(path)
        self._use_linear = self.metadata.get("use_linear", False)
        self._holt_level = self.metadata.get("holt_level")
        self._holt_trend = self.metadata.get("holt_trend")
        self._holt_alpha = self.metadata.get("holt_alpha", 0.3)
        self._holt_beta = self.metadata.get("holt_beta", 0.1)
        self._recent_mean = self.metadata.get("recent_mean", 0.0)
        self._pts_per_day = self.metadata.get("pts_per_day", 96)
        dp = self.metadata.get("daily_pattern")
        self._daily_pattern = np.array(dp) if dp is not None else None

    def train(self, df: pd.DataFrame, feature_names: list = None,
              target_col: str = "value", **kwargs) -> dict:
        y = df[target_col].values

        # 检测频率
        if "dt" in df.columns:
            try:
                diffs = df["dt"].diff().dropna()
                if len(diffs) > 0:
                    freq_minutes = diffs.dt.total_seconds().median() / 60
                    pts_per_hour = max(1, int(round(60 / freq_minutes)))
                    self._pts_per_day = pts_per_hour * 24
                else:
                    self._pts_per_day = 96
            except Exception:
                self._pts_per_day = 96
        else:
            self._pts_per_day = 96

        self._recent_mean = float(np.mean(y[-min(len(y), self._pts_per_day):]))

        # Grid search 最优 alpha/beta
        if len(y) >= max(self._pts_per_day, 2):
            self._holt_alpha, self._holt_beta = self._grid_search_holt(y)
            self._holt_level, self._holt_trend = self._holt_final_state(y)

            # 提取日内模式
            days = len(y) // self._pts_per_day
            if days >= 2:
                reshaped = y[-days * self._pts_per_day:].reshape(days, self._pts_per_day)
                daily_mean = np.mean(reshaped, axis=0)
                self._daily_pattern = daily_mean - np.mean(daily_mean)
        else:
            self._holt_level = float(np.mean(y))
            self._holt_trend = 0.0

        self._trained = True
        self._use_linear = True
        self._save_params()

        logger.info("TS(Holt) 训练完成: pts_per_day=%d, alpha=%.2f, beta=%.2f, level=%.1f, trend=%.2f",
                     self._pts_per_day, self._holt_alpha, self._holt_beta,
                     self._holt_level, self._holt_trend)
        return {"method": "holt", "n_values": len(y), "pts_per_day": self._pts_per_day}

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        horizon = len(df)

        if self._use_linear:
            return self._holt_forecast(horizon)
        else:
            # 旧 LGB 路径 (backward compat)
            feature_names = [c for c in df.columns
                             if any(p in c for p in TS_FEATURE_PATTERNS)]
            if not feature_names:
                feature_names = [c for c in df.columns
                                 if any(k in c for k in ("lag", "rolling", "diff", "sign"))
                                 and c not in ("dt", "type", "province")]
            X = np.zeros((len(df), len(feature_names)), dtype=np.float64)
            for i, fn in enumerate(feature_names):
                if fn in df.columns:
                    X[:, i] = pd.to_numeric(df[fn], errors="coerce").fillna(0).values
            return self.model.predict(X)

    def _holt_forecast(self, steps: int) -> np.ndarray:
        level = self._holt_level or 0.0
        trend = self._holt_trend or 0.0
        forecasts = np.array([level + trend * (i + 1) for i in range(steps)], dtype=np.float64)

        if self._daily_pattern is not None:
            pattern = np.array(self._daily_pattern, dtype=np.float64)
            period = len(pattern)
            for i in range(min(steps, period * 7)):
                forecasts[i] += pattern[i % period]

        return forecasts

    def _grid_search_holt(self, y: np.ndarray) -> tuple:
        best_sse = float("inf")
        best_alpha, best_beta = 0.3, 0.1

        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for beta in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
                sse = self._holt_sse(y, alpha, beta)
                if sse < best_sse:
                    best_sse = sse
                    best_alpha, best_beta = alpha, beta

        return best_alpha, best_beta

    def _holt_sse(self, y: np.ndarray, alpha: float, beta: float) -> float:
        n = len(y)
        level = y[0]
        trend = y[1] - y[0] if n > 1 else 0.0
        sse = 0.0

        for t in range(1, n):
            prev_level = level
            prev_trend = trend
            forecast = prev_level + prev_trend
            sse += (y[t] - forecast) ** 2
            level = alpha * y[t] + (1 - alpha) * (prev_level + prev_trend)
            trend = beta * (level - prev_level) + (1 - beta) * prev_trend

        return sse

    def _holt_final_state(self, y: np.ndarray) -> tuple:
        level = float(y[0])
        trend = float(y[1] - y[0]) if len(y) > 1 else 0.0

        for t in range(1, len(y)):
            prev_level = level
            prev_trend = trend
            level = self._holt_alpha * y[t] + (1 - self._holt_alpha) * (prev_level + prev_trend)
            trend = self._holt_beta * (level - prev_level) + (1 - self._holt_beta) * prev_trend

        return level, trend

    def _save_params(self) -> None:
        self.metadata["use_linear"] = True
        self.metadata["holt_level"] = float(self._holt_level or 0)
        self.metadata["holt_trend"] = float(self._holt_trend or 0)
        self.metadata["holt_alpha"] = float(self._holt_alpha)
        self.metadata["holt_beta"] = float(self._holt_beta)
        self.metadata["recent_mean"] = float(self._recent_mean)
        self.metadata["pts_per_day"] = self._pts_per_day
        if self._daily_pattern is not None:
            self.metadata["daily_pattern"] = self._daily_pattern.tolist()
