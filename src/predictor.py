"""predictor.py — 预测执行器（修复版：未来特征外推 + 残差概率区间）"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from src.trainer import Trainer
from src.feature_store import FeatureStore
from src.db import DorisDB
from src.data_fetcher import DataFetcher
from src.config_loader import load_config


class Predictor:
    def __init__(self, trainer: Trainer = None,
                 store: FeatureStore = None):
        self.trainer = trainer or Trainer()
        self.store = store or FeatureStore(DorisDB())
        self.fetcher = DataFetcher()

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24,
                model_version: str = None) -> pd.DataFrame:
        """对指定省份/类型做未来 N 小时预测.

        修复要点:
        1. 生成未来的时间特征（而非复用过去的时间特征）
        2. 尝试拉取天气预报作为输入
        3. 用残差分布计算概率区间
        """
        model, feature_names = self.trainer.load_model(province, target_type)

        horizon_steps = min(horizon_hours * 4, 96)

        # ── 1. 加载近期特征作为"最后已知状态" ──
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

        # ── 2. 生成未来时间步的特征 ──
        future_features = self._build_future_features(
            history, province, target_type, horizon_steps, end_date
        )

        # ── 3. 预测 ──
        predictions = self._predict_with_model(
            model, future_features, feature_names,
            province=province, target_type=target_type,
            history=history,
        )

        self.store.insert_predictions(predictions)

        return predictions

    def _build_future_features(self, history: pd.DataFrame,
                                province: str, target_type: str,
                                horizon_steps: int,
                                base_dt: datetime) -> pd.DataFrame:
        """从历史数据外推未来 N 步的特征.

        策略:
        - 时间特征: 正确生成未来的 hour/day_of_week/month 等
        - 滞后特征: 用最后已知值的滚动外推
        - 气象特征: 尝试从 Open-Meteo 拉预报，失败则用历史同时段均值
        """
        last_row = history.iloc[-1].copy()
        last_value = last_row.get("value", 0)
        last_price = last_row.get("price", 0)

        future_times = [
            base_dt + timedelta(minutes=15 * i)
            for i in range(1, horizon_steps + 1)
        ]

        rows = []
        for i, ft in enumerate(future_times):
            row = {
                "dt": ft,
                "province": province,
                "type": target_type,
                "value": None,
                "price": last_price,
                "hour": ft.hour,
                "day_of_week": ft.dayofweek,
                "day_of_month": ft.day,
                "month": ft.month,
                "is_weekend": ft.dayofweek in [5, 6],
                "season": (1 if ft.month in [3, 4, 5]
                           else 2 if ft.month in [6, 7, 8]
                           else 3 if ft.month in [9, 10, 11]
                           else 4),
            }

            # 滞后特征: 用最后已知值的投射
            lag_1d_step = 96
            lag_7d_step = 672
            if i < lag_1d_step and len(history) > lag_1d_step:
                # 可以用历史值
                idx = -(lag_1d_step - i)
                row["value_lag_1d"] = history.iloc[idx].get("value", last_value)
            else:
                row["value_lag_1d"] = last_value

            if i < lag_7d_step and len(history) > lag_7d_step:
                idx = -(lag_7d_step - i)
                row["value_lag_7d"] = history.iloc[idx].get("value", last_value)
            else:
                row["value_lag_7d"] = last_value

            # 滑动均值: 用历史最后 96 步均值
            recent_96 = history["value"].tail(96)
            row["value_rolling_mean_24h"] = (
                recent_96.mean() if not recent_96.empty else last_value
            )

            # 差分: 用 last_value 替代
            row["value_diff_1d"] = 0.0
            row["value_diff_7d"] = 0.0

            rows.append(row)

        future_df = pd.DataFrame(rows)

        # ── 尝试合并气象预报 ──
        try:
            forecast_end = (base_dt + timedelta(days=8)).strftime("%Y-%m-%d")
            weather = self.fetcher.fetch_weather(
                province,
                base_dt.strftime("%Y-%m-%d"),
                forecast_end,
                mode="forecast",
            )
            if not weather.empty:
                weather["dt_merge"] = weather["dt"].dt.floor("15min")
                future_df["dt_merge"] = future_df["dt"].dt.floor("15min")
                weather_cols = ["temperature", "humidity", "wind_speed",
                                "wind_direction", "solar_radiation",
                                "precipitation", "pressure"]
                for col in weather_cols:
                    if col in weather.columns:
                        merged = weather[["dt_merge"] + [col]].copy()
                        future_df = future_df.merge(
                            merged, on="dt_merge", how="left", suffixes=("", "_w")
                        )
                        if col in future_df.columns:
                            future_df[col] = future_df[col].fillna(history[col].mean() if col in history.columns else 0)
                future_df.drop(columns=["dt_merge"], inplace=True, errors="ignore")
        except Exception:
            pass

        # 填充缺失的数值特征
        for col in future_df.columns:
            if col not in ("dt", "province", "type") and future_df[col].dtype == np.float64:
                future_df[col] = future_df[col].fillna(
                    history[col].mean() if col in history.columns else 0
                )

        return future_df

    def _predict_with_model(self, model, features_df: pd.DataFrame,
                            feature_names: List[str], province: str,
                            target_type: str,
                            history: pd.DataFrame = None) -> pd.DataFrame:
        """用模型做预测，残差法计算概率区间."""
        predict_features = features_df[feature_names].copy()
        predicted = model.predict(predict_features)

        # ── 概率区间: 基于残差分布 ──
        residual_std = 0.05  # 默认 5%
        if history is not None and len(history) > 96 and "value" in history.columns:
            try:
                # 用最近 96 步的模型预测残差估算标准差
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
            "dt": features_df["dt"].values,
            "province": province,
            "type": target_type,
            "p50": predicted,
            "p10": np.maximum(p10, 0),
            "p90": np.maximum(p90, 0),
            "model_version": "v1",
        })
        return result
