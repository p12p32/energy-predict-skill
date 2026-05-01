"""predictor.py — 预测执行器: 未来特征外推 + 分位数预测 + 趋势集成 + ECM"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

from scripts.ml.trainer import Trainer
from scripts.ml.trend import TrendModel
from scripts.ml.error_correction import ErrorCorrectionModel
from scripts.data.features import FeatureStore
from scripts.data.fetcher import DataFetcher
from scripts.core.config import load_config


class Predictor:
    def __init__(self, trainer: Trainer = None,
                 store: FeatureStore = None):
        self.trainer = trainer or Trainer()
        self.store = store or FeatureStore()
        self.fetcher = DataFetcher()
        self.ecm = ErrorCorrectionModel()

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24,
                model_version: str = None) -> pd.DataFrame:
        horizon_steps = horizon_hours * 4

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

        # ── 尝试分位数模型 ──
        try:
            quantile_models = self.trainer.load_quantile_models(province, target_type)
            predictions = self._predict_quantile(quantile_models, future_features,
                                                  province, target_type)
        except (FileNotFoundError, KeyError):
            # 回退: 单模型 + 残差概率区间
            model, feature_names, _ = self.trainer.load_model(province, target_type)
            predictions = self._predict_with_model(
                model, future_features, feature_names,
                province=province, target_type=target_type, history=history,
            )

        trend_pred = self._predict_trend(history, horizon_steps)
        ensemble = self._ensemble(predictions, trend_pred, history)

        # ECM 残差修正
        if len(history) > 200 and "value" in history.columns:
            try:
                recent = history["value"].tail(500).values
                self.ecm.fit(recent)
                residual_fix = self.ecm.predict(np.zeros(self.ecm.order), horizon_steps)
                ensemble["p50"] = np.maximum(ensemble["p50"].values + residual_fix, 0)
                half_fix = residual_fix / 2
                ensemble["p10"] = np.maximum(ensemble["p10"].values + half_fix, 0)
                ensemble["p90"] = np.maximum(ensemble["p90"].values + half_fix, 0)
            except Exception:
                pass

        self.store.insert_predictions(ensemble)

        return ensemble

    def _predict_quantile(self, models: Dict, features_df: pd.DataFrame,
                           province: str, target_type: str) -> pd.DataFrame:
        """用分位数模型做预测."""
        p10_model, feature_names = models.get("p10", (None, None))
        p50_model, _ = models.get("p50", (None, None))
        p90_model, _ = models.get("p90", (None, None))

        if p50_model is None:
            raise ValueError("无 P50 模型")

        for fn in feature_names:
            if fn not in features_df.columns:
                features_df[fn] = 0.0
        X = features_df[feature_names].values

        result = pd.DataFrame({
            "dt": features_df["dt"].values,
            "province": province, "type": target_type,
            "p10": np.maximum(p10_model.predict(X), 0) if p10_model else np.zeros(len(X)),
            "p50": np.maximum(p50_model.predict(X), 0),
            "p90": np.maximum(p90_model.predict(X), 0) if p90_model else np.zeros(len(X)),
            "model_version": "quantile_v1",
        })
        return result

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
        n = len(lgb_result)
        trend_preds = trend_preds[:n]
        lgb_weight = np.array([max(0.3, 0.75 - 0.005 * i) for i in range(n)])
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

    def _build_future_features(self, history: pd.DataFrame,
                                province: str, target_type: str,
                                horizon_steps: int,
                                base_dt: datetime) -> pd.DataFrame:
        last_row = history.iloc[-1].copy()
        last_value = last_row.get("value", 0)
        last_price = last_row.get("price", 0)
        future_times = pd.date_range(
            start=base_dt + timedelta(minutes=15),
            periods=horizon_steps,
            freq="15min",
        )
        rows = []
        for i, ft in enumerate(future_times):
            row = {
                "dt": ft, "province": province, "type": target_type,
                "value": None, "price": last_price,
                "hour": ft.hour, "day_of_week": ft.dayofweek,
                "day_of_month": ft.day, "month": ft.month,
                "is_weekend": ft.dayofweek in [5, 6],
                "season": (1 if ft.month in [3, 4, 5]
                           else 2 if ft.month in [6, 7, 8]
                           else 3 if ft.month in [9, 10, 11] else 4),
            }
            lag_1d_step, lag_7d_step = 96, 672
            if i < lag_1d_step and len(history) > lag_1d_step:
                row["value_lag_1d"] = history.iloc[-(lag_1d_step - i)].get("value", last_value)
            else:
                row["value_lag_1d"] = last_value
            if i < lag_7d_step and len(history) > lag_7d_step:
                row["value_lag_7d"] = history.iloc[-(lag_7d_step - i)].get("value", last_value)
            else:
                row["value_lag_7d"] = last_value
            recent_96 = history["value"].tail(96)
            row["value_rolling_mean_24h"] = recent_96.mean() if not recent_96.empty else last_value
            row["value_diff_1d"] = 0.0
            row["value_diff_7d"] = 0.0
            rows.append(row)
        future_df = pd.DataFrame(rows)

        from scripts.data.holidays import add_holiday_features, add_cyclical_features
        future_df = add_holiday_features(future_df)
        future_df = add_cyclical_features(future_df)
        future_df["quality_flag"] = 0
        for col in ["temperature", "humidity", "wind_speed", "wind_direction",
                     "solar_radiation", "precipitation", "pressure"]:
            if col not in future_df.columns:
                future_df[col] = history[col].mean() if col in history.columns else 0.0

        try:
            forecast_end = (base_dt + timedelta(days=8)).strftime("%Y-%m-%d")
            weather = self.fetcher.fetch_weather(
                province, base_dt.strftime("%Y-%m-%d"), forecast_end, mode="forecast")
            if not weather.empty:
                weather["dt_merge"] = weather["dt"].dt.floor("15min")
                future_df["dt_merge"] = future_df["dt"].dt.floor("15min")
                weather_cols = ["temperature", "humidity", "wind_speed",
                                "wind_direction", "solar_radiation", "precipitation", "pressure"]
                for col in weather_cols:
                    if col in weather.columns:
                        merged = weather[["dt_merge"] + [col]].copy()
                        future_df = future_df.merge(merged, on="dt_merge", how="left", suffixes=("", "_w"))
                        if col in future_df.columns:
                            future_df[col] = future_df[col].fillna(
                                history[col].mean() if col in history.columns else 0)
                future_df.drop(columns=["dt_merge"], inplace=True, errors="ignore")
        except Exception:
            pass
        for col in future_df.columns:
            if col not in ("dt", "province", "type") and future_df[col].dtype == np.float64:
                future_df[col] = future_df[col].fillna(
                    history[col].mean() if col in history.columns else 0)
        return future_df

    def _predict_with_model(self, model, features_df: pd.DataFrame,
                            feature_names: List[str], province: str,
                            target_type: str,
                            history: pd.DataFrame = None) -> pd.DataFrame:
        for fn in feature_names:
            if fn not in features_df.columns:
                features_df[fn] = 0.0
        predict_features = features_df[feature_names].copy()
        predicted = model.predict(predict_features)
        residual_std = 0.05
        if history is not None and len(history) > 96 and "value" in history.columns:
            try:
                recent = history.tail(96)
                if all(fn in recent.columns for fn in feature_names):
                    hist_pred = model.predict(recent[feature_names].values)
                    hist_actual = recent["value"].values
                    mask = hist_actual != 0
                    if mask.sum() > 10:
                        residuals = (hist_actual[mask] - hist_pred[mask]) / hist_actual[mask]
                        residual_std = float(np.std(residuals))
            except Exception:
                pass
        p10 = predicted * (1 - 1.28 * residual_std)
        p90 = predicted * (1 + 1.28 * residual_std)
        result = pd.DataFrame({
            "dt": features_df["dt"].values, "province": province, "type": target_type,
            "p50": predicted,
            "p10": np.maximum(p10, 0), "p90": np.maximum(p90, 0),
            "model_version": "v1",
        })
        return result
