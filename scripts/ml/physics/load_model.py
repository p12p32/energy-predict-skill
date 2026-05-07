"""load_model.py — 负荷分解模型: 基线 + 气象弹性 + 节假日 + 偏移.

Load(t) = STL_baseline(t) + weather_elasticity(t) + holiday_adj(t) + offset(t)

STL 提取日/周周期结构, ML 只学气象驱动的弹性部分.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import defaultdict

from scripts.ml.physics.base import PhysicalModel, PhysicalModelConfig

logger = logging.getLogger(__name__)


class LoadDecompositionModel(PhysicalModel):
    def __init__(self, config: Optional[PhysicalModelConfig] = None):
        if config is None:
            config = PhysicalModelConfig()
        super().__init__("load", config)
        self._stl_trend: Optional[np.ndarray] = None
        self._stl_seasonal_weekly: Optional[np.ndarray] = None   # 完整周模式 (96×7点)
        self._stl_last_dt: Optional[pd.Timestamp] = None          # STL 末尾时间戳，用于周模式对齐
        self._stl_resid_std: float = 1.0
        self._holiday_factors: Dict[str, float] = {}
        self._offset_ema: float = 0.0
        self._offset_alpha: float = 0.3
        self._points_per_day: int = 96
        self._elasticity_features: list = []

    # ── STL 分解 ──

    def _fit_stl(self, series: np.ndarray):
        from statsmodels.tsa.seasonal import STL

        period = self._points_per_day
        max_len = 50000
        if len(series) > max_len:
            series = series[-max_len:]

        try:
            stl = STL(series, period=period, seasonal=7, robust=True)
            res = stl.fit()
            self._stl_trend = res.trend
            seasonal_full = res.seasonal

            # 提取完整周模式: 最后 7 天
            week_len = period * 7
            n_days = len(seasonal_full) // period
            if n_days >= 7:
                self._stl_seasonal_weekly = seasonal_full[-week_len:].copy()
            elif n_days >= 1:
                # 不足7天则重复填充
                base = seasonal_full[-(n_days * period):]
                repeats = (week_len // len(base)) + 1
                self._stl_seasonal_weekly = np.tile(base, repeats)[:week_len]
            else:
                self._stl_seasonal_weekly = np.zeros(week_len)

            self._stl_resid_std = float(np.std(res.resid[~np.isnan(res.resid)])) + 1.0
            logger.info("STL 分解完成: period=%d, trend_len=%d, seasonal_len=%d",
                         period, len(self._stl_trend), len(seasonal_full))

        except Exception as e:
            logger.warning("STL 分解失败 (%s), 退化为移动平均", e)
            self._fallback_baseline(series)

    def _fallback_baseline(self, series: np.ndarray):
        window = min(self._points_per_day * 2, len(series) // 4, 192)
        if window < 4:
            window = 4
        kernel = np.ones(window) / window
        self._stl_trend = np.convolve(series, kernel, mode="same")
        resid = series - self._stl_trend
        pts = self._points_per_day
        week_len = pts * 7
        self._stl_seasonal_weekly = np.zeros(week_len)
        # 按 (星期几, 时刻) 取均值
        for i in range(week_len):
            vals = resid[i::week_len]
            if len(vals) > 0:
                self._stl_seasonal_weekly[i] = np.mean(vals)
        self._stl_resid_std = float(np.std(resid)) + 1.0
        logger.info("退化为 MA 基线: window=%d", window)

    def _extrapolate_stl_baseline(self, n: int, start_offset: int = 0) -> np.ndarray:
        """外推 STL 基线到未来 n 步. start_offset 为从 STL 末尾的偏移 (用于 fit 时对齐)."""
        if self._stl_trend is None:
            return np.zeros(n)

        # 趋势线性外推
        trend_tail = self._stl_trend[-self._points_per_day * 2:]
        if len(trend_tail) >= 4:
            x = np.arange(len(trend_tail))
            slope = np.polyfit(x, trend_tail, 1)[0]
        else:
            slope = 0.0

        last_trend = self._stl_trend[-1] + slope * start_offset
        future_trend = last_trend + slope * np.arange(1, n + 1)

        # 完整周模式重复
        weekly = (self._stl_seasonal_weekly if self._stl_seasonal_weekly is not None
                  else np.zeros(self._points_per_day * 7))
        week_len = len(weekly)

        baseline = future_trend.copy()
        for i in range(n):
            baseline[i] += weekly[(start_offset + i) % week_len]

        return baseline

    # ── 气象弹性 ──

    def _build_elasticity_features(self, weather: pd.DataFrame) -> pd.DataFrame:
        features = weather.copy()
        dts = pd.to_datetime(features["dt"])
        features["hour"] = dts.dt.hour
        features["day_of_week"] = dts.dt.dayofweek
        features["month"] = dts.dt.month

        feat_cols = ["CDD", "HDD", "THI", "temp_extremity", "temp_zscore",
                     "is_heat_wave", "is_cold_snap", "extreme_weather_flag",
                     "hour", "day_of_week", "month"]

        for col in feat_cols:
            if col not in features.columns:
                features[col] = 0.0

        available = [c for c in feat_cols if c in features.columns]
        self._elasticity_features = available
        return features[available]

    def _fit_elasticity(self, weather: pd.DataFrame, residuals: np.ndarray):
        import lightgbm as lgb

        X_df = self._build_elasticity_features(weather)
        X = X_df.values.astype(float)
        y = residuals

        n_val = max(self._points_per_day, int(len(y) * 0.15))
        if n_val >= len(y):
            n_val = 0

        self._ml_fallback = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.03,
            random_state=42, verbose=-1,
        )
        if n_val > 0:
            self._ml_fallback.fit(X[:-n_val], y[:-n_val],
                                  eval_set=[(X[-n_val:], y[-n_val:])],
                                  eval_metric="l1")
        else:
            self._ml_fallback.fit(X, y)

    def _predict_elasticity(self, weather: pd.DataFrame) -> np.ndarray:
        if self._ml_fallback is not None:
            X = self._build_elasticity_features(weather).values.astype(float)
            return self._ml_fallback.predict(X)
        return np.zeros(len(weather))

    # ── 节假日调整 ──

    def _fit_holiday_factors(self, weather: pd.DataFrame,
                             residuals: np.ndarray):
        if "is_holiday" not in weather.columns:
            return

        dts = pd.to_datetime(weather["dt"])
        is_holiday = weather["is_holiday"].values.astype(bool)
        is_workday = ~is_holiday & (dts.dt.dayofweek < 5).values

        if is_holiday.sum() < self._points_per_day:
            return

        # 节假日平均残差 vs 工作日
        holiday_resid = np.mean(residuals[is_holiday]) if is_holiday.sum() > 0 else 0.0
        workday_resid = np.mean(residuals[is_workday]) if is_workday.sum() > 0 else 0.0

        if abs(workday_resid) > 1.0:
            factor = holiday_resid / workday_resid
            self._holiday_factors["default"] = float(np.clip(factor, -3.0, 3.0))

        # 区分节假日类型
        for holiday_type in ["spring_festival", "national_day", "labor_day"]:
            # 按月份粗略区分
            pass

        logger.info("节假日因子: %s", self._holiday_factors)

    def _predict_holiday_adj(self, weather: pd.DataFrame) -> np.ndarray:
        adj = np.zeros(len(weather))
        if not self._holiday_factors or "is_holiday" not in weather.columns:
            return adj

        is_holiday = weather["is_holiday"].values.astype(bool)
        if is_holiday.sum() > 0:
            adj[is_holiday] = self._holiday_factors.get("default", 0.0)
        return adj

    # ── 历史偏移 ──

    def _update_offset(self, actual: np.ndarray, predicted: np.ndarray):
        errors = actual - predicted
        recent_errors = errors[-min(len(errors), self._points_per_day * 2):]
        self._offset_ema = (self._offset_alpha * np.mean(recent_errors) +
                            (1 - self._offset_alpha) * self._offset_ema)

    def _predict_offset(self, n: int) -> np.ndarray:
        # EMA 衰减
        decay = np.exp(-np.arange(n) / (self._points_per_day * 3))
        return np.full(n, self._offset_ema) * decay

    # ── 主接口 ──

    def predict_physics(self, weather: pd.DataFrame) -> np.ndarray:
        """物理预测: STL基线 + 气象弹性 + 节假日."""
        n = len(weather)

        # 计算周模式对齐偏移
        start_offset = 0
        if self._stl_last_dt is not None and "dt" in weather.columns:
            first_dt = pd.to_datetime(weather["dt"].iloc[0])
            gap_steps = int((first_dt - self._stl_last_dt).total_seconds() / 900)  # 15min steps
            start_offset = max(0, gap_steps)

        baseline = self._extrapolate_stl_baseline(n, start_offset=start_offset)
        elasticity = self._predict_elasticity(weather)
        holiday = self._predict_holiday_adj(weather)
        return baseline + elasticity + holiday

    def get_equation(self) -> str:
        lines = ["Load(t) = STL_baseline(t) + weather_elasticity(t) + holiday_adj(t) + offset(t)"]
        lines.append(f"  STL: period={self._points_per_day}, seasonal=7")
        if self._elasticity_features:
            lines.append(f"  Elasticity features: {self._elasticity_features[:6]}")
        if self._holiday_factors:
            lines.append(f"  Holiday factors: {self._holiday_factors}")
        return "\n".join(lines)

    def fit(self, weather: pd.DataFrame, actuals: pd.Series,
            **kwargs) -> Dict:
        logger.info("LoadModel 拟合: %d 样本", len(weather))

        # 频率检测
        if "dt" in weather.columns and len(weather) >= 2:
            delta = pd.to_datetime(weather["dt"]).diff().mode().iloc[0]
            freq_min = delta.total_seconds() / 60
            self._points_per_day = int(round(1440 / freq_min))
            self.config.points_per_hour = int(round(60 / freq_min))
            logger.info("频率: %.0fmin/点, %d点/天", freq_min, self._points_per_day)

        y = actuals.values.astype(float)

        # 1. STL 分解
        self._fit_stl(y)

        # 记录 STL 末尾时间戳用于预测对齐
        if "dt" in weather.columns and len(weather) > 0:
            self._stl_last_dt = pd.to_datetime(weather["dt"].iloc[-1])

        # 2. STL 残差 = actual - trend - seasonal
        n_stl = len(self._stl_trend)
        weekly = self._stl_seasonal_weekly if self._stl_seasonal_weekly is not None else np.zeros(self._points_per_day * 7)
        week_len = len(weekly)
        stl_baseline = self._stl_trend[:n_stl].copy()
        for i in range(n_stl):
            stl_baseline[i] += weekly[i % week_len]

        residuals = y[-n_stl:] - stl_baseline

        # 对齐天气数据
        weather_aligned = weather.iloc[-n_stl:].copy() if len(weather) > n_stl else weather.copy()

        # 3. 气象弹性 ML
        self._fit_elasticity(weather_aligned, residuals)

        # 4. 节假日因子
        self._fit_holiday_factors(weather_aligned, residuals)

        # 5. 评估: 用 STL 实际分解值 (非外推) 评估历史拟合质量
        pred_hist = stl_baseline + self._predict_elasticity(weather_aligned) + self._predict_holiday_adj(weather_aligned)
        err_hist = y[-n_stl:] - pred_hist
        mae = float(np.mean(np.abs(err_hist)))
        rmse = float(np.sqrt(np.mean(err_hist ** 2)))
        r2 = float(1.0 - np.sum(err_hist ** 2) /
                   max(np.sum((y[-n_stl:] - np.mean(y[-n_stl:])) ** 2), 1e-10))

        self._update_offset(y[-n_stl:], pred_hist)
        self.metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        self._trained = True

        logger.info("Load 拟合完成: MAE=%.1f MW, RMSE=%.1f MW, R²=%.3f, offset=%.1f",
                     mae, rmse, r2, self._offset_ema)
        return dict(self.metrics)

    def predict(self, weather: pd.DataFrame, solar_pred: np.ndarray = None,
                wind_pred: np.ndarray = None) -> np.ndarray:
        pred = self.predict_physics(weather)
        offset = self._predict_offset(len(weather))
        return pred + offset

    def save(self, path: str) -> None:
        import pickle, os, json
        base = os.path.splitext(path)[0]

        # 父类保存
        super().save(path)

        # STL 组件保存
        stl_data = {
            "trend": self._stl_trend.tolist() if self._stl_trend is not None else None,
            "seasonal_weekly": self._stl_seasonal_weekly.tolist() if self._stl_seasonal_weekly is not None else None,
            "last_dt": str(self._stl_last_dt) if self._stl_last_dt is not None else None,
            "resid_std": self._stl_resid_std,
            "points_per_day": self._points_per_day,
            "offset_ema": self._offset_ema,
            "offset_alpha": self._offset_alpha,
            "holiday_factors": self._holiday_factors,
        }
        with open(f"{base}_stl.json", "w", encoding="utf-8") as f:
            json.dump(stl_data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        import json, os
        super().load(path)
        base = os.path.splitext(path)[0]
        stl_path = f"{base}_stl.json"
        if os.path.exists(stl_path):
            with open(stl_path, encoding="utf-8") as f:
                stl_data = json.load(f)
            self._stl_trend = np.array(stl_data["trend"]) if stl_data["trend"] else None
            self._stl_seasonal_weekly = np.array(stl_data["seasonal_weekly"]) if stl_data["seasonal_weekly"] else None
            self._stl_last_dt = pd.Timestamp(stl_data["last_dt"]) if stl_data.get("last_dt") else None
            self._stl_resid_std = stl_data.get("resid_std", 1.0)
            self._points_per_day = stl_data.get("points_per_day", 96)
            self._offset_ema = stl_data.get("offset_ema", 0.0)
            self._offset_alpha = stl_data.get("offset_alpha", 0.3)
            self._holiday_factors = stl_data.get("holiday_factors", {})
