"""predictor.py — 预测执行器: 未来特征外推 + 残差概率区间 + 趋势集成"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from src.trainer import Trainer
from src.feature_store import FeatureStore
from src.db import DorisDB
from src.data_fetcher import DataFetcher
from src.config_loader import load_config


class TrendModel:
    """简单趋势外推: 双指数平滑 + 日内模式.

    与 LightGBM 互补:
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
        """趋势 + 日内模式叠加."""
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


class Predictor:
    def __init__(self, trainer: Trainer = None,
                 store: FeatureStore = None):
        self.trainer = trainer or Trainer()
        self.store = store or FeatureStore(DorisDB())
        self.fetcher = DataFetcher()

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24,
                model_version: str = None) -> pd.DataFrame:
        model, feature_names = self.trainer.load_model(province, target_type)
        horizon_steps = min(horizon_hours * 4, 96)

        lookback_days = 14
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        history = self.store.load_features(
            province, target_type,
            start_date.strftime("%Y-%m-%d"),
            (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if history.empty:
            raise ValueError(f"没有可用的特征数据: {province}/{target_type}")

        future_features = self._build_future_features(
            history, province, target_type, horizon_steps, end_date
        )

        lgb_result = self._predict_with_model(
            model, future_features, feature_names,
            province=province, target_type=target_type,
            history=history,
        )

        trend_pred = self._predict_trend(history, horizon_steps)

        ensemble = self._ensemble(lgb_result, trend_pred, history)

        self.store.insert_predictions(ensemble)

        return ensemble

    def _predict_trend(self, history: pd.DataFrame,
                       horizon_steps: int) -> np.ndarray:
        if "value" not in history.columns or len(history) < 96:
            return np.zeros(horizon_steps)

        values = history["value"].dropna().values
        if len(values) == 0:
            return np.zeros(horizon_steps)

        trend_model = TrendModel()
        trend_model.fit(values)
        return trend_model.predict_with_daily_pattern(horizon_steps, values)

    def _ensemble(self, lgb_result: pd.DataFrame,
                  trend_preds: np.ndarray,
                  history: pd.DataFrame) -> pd.DataFrame:
        """集成: 短期偏 LightGBM, 长期偏趋势模型."""
        n = len(lgb_result)
        trend_preds = trend_preds[:n]

        lgb_weight = np.array([
            max(0.3, 0.75 - 0.005 * i) for i in range(n)
        ])
        trend_weight = 1.0 - lgb_weight

        lgb_p50 = lgb_result["p50"].values
        ensemble_p50 = lgb_weight * lgb_p50 + trend_weight * trend_preds

        result = lgb_result.copy()
        result["p50"] = np.maximum(ensemble_p50, 0)

        width = result["p90"].values - result["p10"].values
        result["p10"] = np.maximum(ensemble_p50 - width / 2, 0)
        result["p90"] = np.maximum(ensemble_p50 + width / 2, 0)

        result["trend_adjusted"] = True
        return result