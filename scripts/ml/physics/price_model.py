"""price_model.py — 电价结构化模型: Price = f(净负荷, DA价格, 时间特征).

净负荷 (net_load) 为主导特征 — 替代当前管线中散落的
supply_demand_ratio / renewable_share / residual_load.

方向分类器: LGBMClassifier 直接预测涨跌方向, 校正回归模型的方向盲区.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional

from scripts.ml.physics.base import PhysicalModel, PhysicalModelConfig

logger = logging.getLogger(__name__)


class PriceStructuralModel(PhysicalModel):
    def __init__(self, config: Optional[PhysicalModelConfig] = None):
        if config is None:
            config = PhysicalModelConfig()
        super().__init__("price", config)
        self._feature_cols: list = []
        self._use_da_price: bool = False
        self._offset_ema: float = 0.0
        self._offset_alpha: float = 0.3
        self._delta_mode: bool = False  # True=预测昨日偏差, False=预测绝对水平
        self._dir_classifier = None  # LGBMClassifier
        self._dir_feature_cols: list = []
        self._dir_acc: float = 0.0

        # 目标预处理: 离群剔除 + 降噪 (HP滤波复用之)
        self._clean_target: bool = True  # 训练前清洗目标
        self._outlier_sigma: float = 4.0  # MAD 离群阈值 (0=禁用)
        self._smooth_window: int = 3  # 滚动平均窗口

        # HP 滤波 + 日内形态 (实验性)
        self._use_hp_filter: bool = False
        self._hp_lambda: float = 1e8
        self._daily_shape: np.ndarray = None

        # Prophet 日趋势特征 (实验性)
        self._use_prophet: bool = False
        self._prophet_model = None
        self._prophet_train_dates = None

    def predict_physics(self, weather: pd.DataFrame) -> np.ndarray:
        """纯物理无意义 (电价没有物理方程), 返回统计基线."""
        n = len(weather)
        # 返回气候均值作为 fallback
        if "price_lag_1d" in weather.columns:
            return weather["price_lag_1d"].values
        return np.full(n, 350.0)

    def get_equation(self) -> str:
        return ("Price(t) = LGBM(net_load, net_load_ramp, DA_price, "
                "hour, dow, month, is_weekend, season, extreme_weather)")

    def predict(self, net_load: np.ndarray, weather: pd.DataFrame = None,
                da_price: np.ndarray = None,
                lag_1d: np.ndarray = None, lag_7d: np.ndarray = None) -> np.ndarray:
        if self._ml_fallback is None:
            base = self.predict_physics(weather) if weather is not None else np.full(len(net_load), 350.0)
        else:
            # 同步平滑 lag 特征 (与训练时一致)
            if self._clean_target:
                if lag_1d is not None:
                    lag_1d = self._smooth_array(lag_1d)
                if lag_7d is not None:
                    lag_7d = self._smooth_array(lag_7d)

            X = self._build_price_features(net_load, weather, da_price, lag_1d, lag_7d)
            raw = self._ml_fallback.predict(X)

            if self._delta_mode and lag_1d is not None and len(lag_1d) >= len(net_load):
                lag_vals = lag_1d[:len(net_load)]
                if not np.any(np.isnan(lag_vals)) and not np.all(lag_vals == 0):
                    n = len(raw)
                    delta_smoothed = np.convolve(raw, np.ones(3) / 3, mode="same")
                    delta_smoothed[:1] = raw[:1]
                    delta_smoothed[-1:] = raw[-1:]
                    base = lag_vals + delta_smoothed
                else:
                    base = raw
            else:
                base = raw

        # HP 滤波模式: 叠加日内形态 (趋势预测 + 日内固定形态)
        if self._use_hp_filter and self._daily_shape is not None:
            n = len(base)
            tiled_shape = np.tile(self._daily_shape, max(1, n // 96 + 1))[:n]
            base = base + tiled_shape

        return base + self._predict_offset(len(net_load))

    def _build_price_features(self, net_load: np.ndarray,
                               weather: pd.DataFrame = None,
                               da_price: np.ndarray = None,
                               lag_1d: np.ndarray = None,
                               lag_7d: np.ndarray = None) -> np.ndarray:
        n = len(net_load)
        features = {}

        # ═══════════════════════════════════════════════════════
        # 1. 净负荷核心特征 (~10)
        # ═══════════════════════════════════════════════════════
        nl_ma_1h = np.convolve(net_load, np.ones(4) / 4, mode="same")
        nl_ma_1h[:2] = net_load[:2]; nl_ma_1h[-2:] = net_load[-2:]
        nl_ma_2h = np.convolve(net_load, np.ones(8) / 8, mode="same")
        nl_ma_2h[:4] = nl_ma_1h[:4]; nl_ma_2h[-4:] = nl_ma_1h[-4:]
        nl_ma_4h = np.convolve(net_load, np.ones(16) / 16, mode="same")
        nl_ma_4h[:8] = nl_ma_2h[:8]; nl_ma_4h[-8:] = nl_ma_2h[-8:]

        features["net_load"] = nl_ma_1h
        features["net_load_raw"] = net_load
        features["net_load_ma_2h"] = nl_ma_2h
        features["net_load_ma_4h"] = nl_ma_4h

        # 多尺度 ramp
        features["net_load_ramp_15m"] = np.zeros(n)
        features["net_load_ramp_15m"][1:] = np.diff(nl_ma_1h)
        features["net_load_ramp_1h"] = np.zeros(n)
        features["net_load_ramp_1h"][4:] = nl_ma_1h[4:] - nl_ma_1h[:-4]
        features["net_load_ramp_2h"] = np.zeros(n)
        features["net_load_ramp_2h"][8:] = nl_ma_1h[8:] - nl_ma_1h[:-8]
        features["net_load_ramp_4h"] = np.zeros(n)
        features["net_load_ramp_4h"][16:] = nl_ma_1h[16:] - nl_ma_1h[:-16]

        # 加速度
        features["net_load_accel"] = np.zeros(n)
        features["net_load_accel"][2:] = (nl_ma_1h[2:] - nl_ma_1h[1:-1]) - \
                                         (nl_ma_1h[1:-1] - nl_ma_1h[:-2])
        # 趋势: 短期 vs 中期 MA 差异
        features["net_load_trend_2h"] = nl_ma_1h - nl_ma_4h

        # 分位数归一化
        q10, q50, q90 = np.percentile(net_load, [10, 50, 90])
        scale = max(q90 - q10, 1.0)
        features["net_load_norm"] = (nl_ma_1h - q50) / scale
        features["net_load_deviation"] = nl_ma_1h - q50
        features["net_load_percentile"] = np.clip((nl_ma_1h - q10) / max(q90 - q10, 1), 0, 1)

        # ═══════════════════════════════════════════════════════
        # 2. 时间特征 (~15)
        # ═══════════════════════════════════════════════════════
        has_dt = weather is not None and "dt" in weather.columns
        if has_dt:
            dts = pd.to_datetime(weather["dt"])
            hours = dts.dt.hour.values + dts.dt.minute.values / 60.0
            days = dts.dt.dayofweek.values
            months = dts.dt.month.values
            dom = dts.dt.day.values
            doy = dts.dt.dayofyear.values

            features["hour"] = hours
            features["day_of_week"] = days
            features["month"] = months
            features["is_weekend"] = (days >= 5).astype(float)

            features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
            features["dow_sin"] = np.sin(2 * np.pi * days / 7)
            features["dow_cos"] = np.cos(2 * np.pi * days / 7)

            # 峰谷时段标识
            is_peak = ((hours >= 8) & (hours <= 11)) | ((hours >= 17) & (hours <= 20))
            features["is_peak_hour"] = is_peak.astype(float)
            is_valley = (hours >= 23) | (hours <= 6)
            features["is_valley_hour"] = is_valley.astype(float)
        else:
            for col in ["hour", "day_of_week", "month", "is_weekend",
                        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                        "is_peak_hour", "is_valley_hour"]:
                features[col] = np.zeros(n)

        # ═══════════════════════════════════════════════════════
        # 3. 天气衍生特征 (~15)
        # ═══════════════════════════════════════════════════════
        def _wcol(name):
            if weather is not None and name in weather.columns:
                return weather[name].values[:n].astype(float)
            return np.zeros(n)

        temp = _wcol("temperature")
        humidity = _wcol("humidity")
        wind_speed = _wcol("wind_speed")
        cloud = _wcol("cloud_factor")
        solar_rad = _wcol("solar_radiation")

        features["temperature"] = temp
        features["humidity"] = humidity
        features["wind_speed"] = wind_speed

        # 度日
        features["CDD"] = np.clip(temp - 26, 0, None)   # 制冷度日
        features["HDD"] = np.clip(18 - temp, 0, None)   # 采暖度日
        features["THI"] = 0.8 * temp + humidity / 100 * temp + 46.4  # 温湿指数

        # 温度极值
        temp_mean = np.mean(temp) if n > 0 else 20
        temp_std = max(np.std(temp), 1.0)
        features["temp_extremity"] = np.abs(temp - temp_mean) / temp_std

        # 温度变化
        features["temp_change_1h"] = np.zeros(n)
        features["temp_change_1h"][4:] = temp[4:] - temp[:-4]
        features["temp_change_6h"] = np.zeros(n)
        features["temp_change_6h"][24:] = temp[24:] - temp[:-24]

        # 风光潜力
        features["wind_power_potential"] = np.clip(wind_speed, 0, 25) ** 3 / 100
        features["solar_potential"] = solar_rad / 1000.0
        features["cloud_factor"] = cloud

        # 交互
        features["temp_x_humidity"] = temp * humidity / 100

        # 极端标志
        for col in ["extreme_weather_flag", "is_heat_wave", "is_cold_snap"]:
            features[col] = _wcol(col)

        # ═══════════════════════════════════════════════════════
        # 4. 日前电价信号 (~4)
        # ═══════════════════════════════════════════════════════
        if da_price is not None and len(da_price) >= n:
            features["da_price"] = da_price[:n]
            features["da_price_diff"] = np.zeros(n)
            features["da_price_diff"][1:] = np.diff(da_price[:n])
            features["da_price_dir"] = np.sign(features["da_price_diff"])
            features["da_price_ramp_1h"] = np.zeros(n)
            features["da_price_ramp_1h"][4:] = da_price[:n][4:] - da_price[:n][:-4]
            self._use_da_price = True
        else:
            for col in ["da_price", "da_price_diff", "da_price_dir", "da_price_ramp_1h"]:
                features[col] = np.zeros(n)

        # ═══════════════════════════════════════════════════════
        # 5. 价格滞后 + 衍生特征 (~18)
        # ═══════════════════════════════════════════════════════
        l1 = lag_1d[:n] if lag_1d is not None and len(lag_1d) >= n else np.zeros(n)
        l7 = lag_7d[:n] if lag_7d is not None and len(lag_7d) >= n else np.zeros(n)

        features["price_lag_1d"] = l1
        features["price_lag_7d"] = l7

        # 价差
        features["price_spread_1d_7d"] = l1 - l7
        features["price_ratio_1d_7d"] = np.where(np.abs(l7) > 1, l1 / l7, 1.0)

        # ramp + 方向
        l1_ramp = np.zeros(n); l1_ramp[1:] = np.diff(l1)
        features["price_lag_1d_dir"] = np.sign(l1_ramp)
        features["price_lag_1d_vol"] = np.abs(l1_ramp)
        l1_ramp_1h = np.zeros(n); l1_ramp_1h[4:] = l1[4:] - l1[:-4]
        features["price_lag_1d_ramp_1h"] = l1_ramp_1h

        l7_ramp = np.zeros(n); l7_ramp[1:] = np.diff(l7)
        features["price_lag_7d_dir"] = np.sign(l7_ramp)
        features["price_lag_7d_vol"] = np.abs(l7_ramp)
        l7_ramp_1h = np.zeros(n); l7_ramp_1h[4:] = l7[4:] - l7[:-4]
        features["price_lag_7d_ramp_1h"] = l7_ramp_1h

        # 动量 (多尺度)
        features["price_momentum_1h"] = np.zeros(n); features["price_momentum_1h"][4:] = l1[4:] - l1[:-4]
        features["price_momentum_4h"] = np.zeros(n); features["price_momentum_4h"][16:] = l1[16:] - l1[:-16]
        features["price_momentum_1d"] = np.zeros(n); features["price_momentum_1d"][96:] = l1[96:] - l1[:-96]

        # 价格相对位置 (24h 分位数)
        prank = np.full(n, 0.5)
        for i in range(24, n):
            w = l1[i-24:i]
            mn, mx = w.min(), w.max()
            if mx - mn > 1:
                prank[i] = (l1[i] - mn) / (mx - mn)
        features["price_position_24h"] = prank

        # 日间加速度
        price_accel = np.zeros(n)
        price_accel[8:] = (l1[8:] - l1[4:-4]) - (l1[4:-4] - l1[:-8])
        features["price_accel_2h"] = price_accel

        # ═══════════════════════════════════════════════════════
        # 6. 供需弹性交互特征 (~20)
        # ═══════════════════════════════════════════════════════
        hrs = features["hour_sin"] if has_dt else np.zeros(n)
        hrc = features["hour_cos"] if has_dt else np.zeros(n)
        wknd = features["is_weekend"] if has_dt else np.zeros(n)
        peak = features["is_peak_hour"] if has_dt else np.zeros(n)
        valley = features["is_valley_hour"] if has_dt else np.zeros(n)

        features["nl_x_hour_sin"] = nl_ma_1h * hrs
        features["nl_x_hour_cos"] = nl_ma_1h * hrc
        features["nl_x_weekend"] = nl_ma_1h * wknd
        features["nl_x_peak"] = nl_ma_1h * peak
        features["nl_x_valley"] = nl_ma_1h * valley

        # 净负荷 × 天气
        features["nl_x_temp"] = nl_ma_1h * temp / 30.0
        features["nl_x_cdd"] = nl_ma_1h * features["CDD"]
        features["nl_x_hdd"] = nl_ma_1h * features["HDD"]
        features["nl_x_thi"] = nl_ma_1h * features["THI"] / 100.0
        features["nl_x_cloud"] = nl_ma_1h * cloud

        # 天气 × 时段
        features["temp_x_hour_sin"] = temp * hrs
        features["temp_x_hour_cos"] = temp * hrc
        features["temp_x_weekend"] = temp * wknd
        features["cdd_x_peak"] = features["CDD"] * peak
        features["hdd_x_valley"] = features["HDD"] * valley

        # ramp × 时段
        features["nl_ramp_x_peak"] = features["net_load_ramp_1h"] * peak
        features["nl_ramp_x_valley"] = features["net_load_ramp_1h"] * valley

        # lag × 时段
        features["lag1d_x_weekend"] = l1 * wknd
        features["lag1d_x_peak"] = l1 * peak
        features["lag7d_x_weekend"] = l7 * wknd

        # ═══════════════════════════════════════════════════════
        # 7. Prophet 日趋势 (可选, ~3)
        # ═══════════════════════════════════════════════════════
        if self._use_prophet and has_dt:
            prophet_feats = self._get_prophet_features(weather["dt"].values, n)
            features.update(prophet_feats)

        self._feature_cols = list(features.keys())
        return np.column_stack([features[k] for k in self._feature_cols])

    # ── Prophet 日趋势特征 ──

    def _fit_prophet(self, actuals: pd.Series):
        """训练 Prophet 日趋势模型."""
        try:
            from prophet import Prophet
        except ImportError:
            logger.info("Prophet 未安装, 跳过日趋势特征")
            self._use_prophet = False
            return

        n = len(actuals)
        n_days = n // 96
        if n_days < 14:
            logger.info("训练数据不足 (%d 天), 跳过 Prophet", n_days)
            self._use_prophet = False
            return

        # 日平均价格
        trimmed = actuals.values[:n_days * 96]
        daily_mean = trimmed.reshape(n_days, 96).mean(axis=1)

        # 日期索引: 从 actuals 的 index 提取日期
        if hasattr(actuals, 'index'):
            start_date = pd.Timestamp(actuals.index[0]).date()
        else:
            start_date = pd.Timestamp.today().date() - pd.Timedelta(days=n_days)
        daily_dates = pd.date_range(start_date, periods=n_days, freq="D")

        prophet_df = pd.DataFrame({"ds": daily_dates, "y": daily_mean})
        self._prophet_train_dates = daily_dates

        self._prophet_model = Prophet(
            daily_seasonality=False, weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
        )
        self._prophet_model.fit(prophet_df)
        logger.info("Prophet 已训练: %d 天, trend=%.1f~%.1f",
                    n_days, daily_mean.min(), daily_mean.max())

    def _get_prophet_features(self, dts, n: int) -> dict:
        """为 n 个时间点生成 Prophet 日趋势特征.

        dts: array-like of datetime strings or Timestamps
        返回 dict with prophet_trend, prophet_weekly, prophet_yhat
        """
        feats = {"prophet_trend": np.zeros(n), "prophet_weekly": np.zeros(n),
                 "prophet_yhat": np.zeros(n)}

        if not self._use_prophet or self._prophet_model is None:
            return feats

        try:
            # 解析日期
            dates = pd.to_datetime(dts)
            unique_dates = pd.date_range(dates.min().date(), dates.max().date(), freq="D")

            # 判断是训练期(in-sample)还是预测期
            if self._prophet_train_dates is not None:
                train_max = self._prophet_train_dates.max()
                if unique_dates.max() <= train_max:
                    # 训练期: 用 in-sample fitted values
                    forecast = self._prophet_model.predict(
                        pd.DataFrame({"ds": unique_dates}))
                else:
                    # 预测期: 用 Prophet forecast
                    future = self._prophet_model.make_future_dataframe(
                        periods=len(unique_dates))
                    # 需要包含历史日期
                    all_dates = pd.date_range(
                        self._prophet_train_dates.min(), unique_dates.max(), freq="D")
                    forecast = self._prophet_model.predict(
                        pd.DataFrame({"ds": all_dates}))
                    # 只取需要的日期
                    forecast = forecast[forecast["ds"].isin(unique_dates)]
            else:
                forecast = self._prophet_model.predict(
                    pd.DataFrame({"ds": unique_dates}))

            # 映射到 15min
            date_to_trend = dict(zip(forecast["ds"], forecast["trend"]))
            date_to_weekly = dict(zip(forecast["ds"], forecast["weekly"]))
            date_to_yhat = dict(zip(forecast["ds"], forecast["yhat"]))

            for i in range(n):
                d = dates[i].date()
                d_ts = pd.Timestamp(d)
                if d_ts in date_to_trend:
                    feats["prophet_trend"][i] = date_to_trend[d_ts]
                    feats["prophet_weekly"][i] = date_to_weekly[d_ts]
                    feats["prophet_yhat"][i] = date_to_yhat[d_ts]

        except Exception as e:
            logger.debug("Prophet 特征生成失败: %s", e)

        return feats

    def _clean_price_target(self, y: np.ndarray) -> np.ndarray:
        """目标预处理: 离群点剔除 → 滚动平均降噪.

        电价尖峰异常远大于负荷, 需要先清洗再建模, 否则模型在学毛刺.
        """
        n = len(y)
        if n < 10:
            return y

        # Step 1: MAD 离群点剔除
        if self._outlier_sigma > 0:
            cleaned = y.copy()
            window = 96
            half = window // 2
            n_out = 0
            for i in range(n):
                lo, hi = max(0, i - half), min(n, i + half + 1)
                seg = cleaned[lo:hi]
                med = np.median(seg)
                mad = np.median(np.abs(seg - med)) * 1.4826
                if mad < 1e-6:
                    continue
                if abs(cleaned[i] - med) > self._outlier_sigma * mad:
                    cleaned[i] = med
                    n_out += 1
            if n_out > 0:
                logger.info("  离群点剔除: %d 个 (%.2f%%)", n_out, n_out / n * 100)
            y = cleaned

        # Step 2: 滚动平均降噪
        if self._smooth_window > 1:
            w = self._smooth_window
            y_smooth = np.convolve(y, np.ones(w) / w, mode="same")
            y_smooth[:w//2] = y[:w//2]
            y_smooth[-(w//2):] = y[-(w//2):]
            y = y_smooth
            logger.info("  目标降噪: rolling(window=%d), noise_reduction=%.0f%%",
                        w, (1 - np.std(y - y.mean()) / max(np.std(y), 1)) * 100)

        return y

    def _smooth_array(self, arr: np.ndarray) -> np.ndarray:
        """对数组做滚动平均降噪 (与目标清洗一致, 无 MAD)."""
        if arr is None or self._smooth_window <= 1:
            return arr
        w = self._smooth_window
        out = np.convolve(arr, np.ones(w) / w, mode="same")
        out[:w//2] = arr[:w//2]
        out[-(w//2):] = arr[-(w//2):]
        return out

    def fit(self, weather: pd.DataFrame = None, actuals: pd.Series = None,
            net_load: np.ndarray = None, da_price: np.ndarray = None,
            lag_1d: np.ndarray = None, lag_7d: np.ndarray = None,
            **kwargs) -> Dict:
        import lightgbm as lgb

        if net_load is None or actuals is None:
            return {"error": "net_load and actuals required"}

        logger.info("PriceModel 拟合: %d 样本 (delta_mode=%s, clean=%s, smooth=%d)",
                    len(actuals), self._delta_mode, self._clean_target, self._smooth_window)

        # 训练 Prophet (在特征构造之前, 因为特征需要 Prophet)
        if self._use_prophet:
            self._fit_prophet(actuals)

        y_raw = actuals.values.astype(float)

        # delta 模式: 截断 NaN 前缀 + 转为 delta 目标
        # 必须在特征构造之前, 保证 X 和 y 对齐
        if self._delta_mode and lag_1d is not None and len(lag_1d) >= len(y_raw):
            lag_vals = lag_1d[:len(y_raw)]
            valid_start = 0
            while valid_start < len(lag_vals) and (np.isnan(lag_vals[valid_start]) or lag_vals[valid_start] == 0):
                valid_start += 1
            if valid_start > len(lag_vals) - 96:
                logger.info("  lag_1d 有效数据不足 (%d/%d), 回退 level 模式", valid_start, len(lag_vals))
                self._delta_mode = False
            else:
                if valid_start > 0:
                    logger.info("  lag_1d 前 %d 点截断, 剩余 %d 点", valid_start, len(y_raw) - valid_start)
                    y_raw = y_raw[valid_start:]
                    lag_1d = lag_1d[valid_start:]
                    if lag_7d is not None:
                        lag_7d = lag_7d[valid_start:]
                    net_load = net_load[valid_start:]
                    if weather is not None:
                        weather = weather.iloc[valid_start:]

        # 目标预处理: 离群剔除 + 降噪 → 减少毛刺被模型学习
        if self._clean_target:
            y_raw = self._clean_price_target(y_raw)
            if lag_1d is not None:
                lag_1d = self._smooth_array(lag_1d)
            if lag_7d is not None:
                lag_7d = self._smooth_array(lag_7d)

        X = self._build_price_features(net_load, weather, da_price, lag_1d, lag_7d)

        if self._delta_mode and lag_1d is not None:
            y = y_raw - lag_1d[:len(y_raw)]
            logger.info("  目标: delta (price − lag_1d), mean=%.1f, std=%.1f", np.mean(y), np.std(y))
        else:
            y = y_raw

        # HP 滤波模式: 平滑 → HP滤波 → 提取日内形态 → 只对趋势建模
        if self._use_hp_filter:
            n = len(y)
            from statsmodels.tsa.filters.hp_filter import hpfilter

            # 3~5点滚动平均降噪
            y_smooth = pd.Series(y).rolling(
                window=self._smooth_window, center=True, min_periods=1).mean().values

            # HP 滤波
            y_trend, y_cycle = hpfilter(y_smooth, lamb=self._hp_lambda)
            logger.info("  HP滤波 (λ=%.0e): trend_std=%.1f, cycle_std=%.1f",
                        self._hp_lambda, y_trend.std(), y_cycle.std())

            # 提取日内形态 (从 cycle 中取历史同时段均值)
            n_days = n // 96
            if n_days >= 3:
                cycle_trimmed = y_cycle[:n_days * 96]
                daily_profiles = cycle_trimmed.reshape(n_days, 96)
                self._daily_shape = daily_profiles.mean(axis=0)
                self._daily_shape -= self._daily_shape.mean()  # 零均值
                logger.info("  日内形态: min=%.1f, max=%.1f, std=%.1f",
                            self._daily_shape.min(), self._daily_shape.max(), self._daily_shape.std())
            else:
                self._daily_shape = np.zeros(96)

            # 用 HP 趋势替代原始目标
            y = y_trend
            logger.info("  目标替换: raw→HP_trend, MAE(raw,trend)=%.1f",
                        np.mean(np.abs(y_smooth - y_trend)))

        n_val = max(96, int(len(y) * 0.15))

        # 时序衰减权重 — 近期样本权重更大, 半衰期=全量数据长度
        sample_weight = np.exp(-np.arange(len(y))[::-1] / len(y))
        sample_weight = sample_weight / sample_weight.mean()

        n_features = X.shape[1]
        # 特征多时加强正则化: 防止噪声特征稀释核心信号
        if n_features > 50:
            self._ml_fallback = lgb.LGBMRegressor(
                n_estimators=500, num_leaves=15, learning_rate=0.02,
                min_child_samples=100, feature_fraction=0.6, subsample=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbose=-1,
            )
        else:
            self._ml_fallback = lgb.LGBMRegressor(
                n_estimators=300, num_leaves=31, learning_rate=0.02,
                min_child_samples=30,
                random_state=42, verbose=-1,
            )
        if n_val > 0:
            self._ml_fallback.fit(X[:-n_val], y[:-n_val],
                                  sample_weight=sample_weight[:-n_val],
                                  eval_set=[(X[-n_val:], y[-n_val:])],
                                  eval_metric="l1")
        else:
            self._ml_fallback.fit(X, y, sample_weight=sample_weight)

        pred = self._ml_fallback.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        r2 = float(1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-10))

        # 特征重要性
        importances = self._ml_fallback.feature_importances_
        top_idx = np.argsort(importances)[-5:][::-1]
        top_feats = [(self._feature_cols[i], round(float(importances[i]), 4))
                     for i in top_idx]
        logger.info("Price 拟合完成: MAE=%.1f, RMSE=%.1f, R²=%.3f", mae, rmse, r2)
        logger.info("  Top features: %s", top_feats)

        # 更新 EMA 偏移 (在 delta 空间或 level 空间)
        errors = y_raw - self.predict(net_load, weather, da_price, lag_1d, lag_7d)
        n_bias = min(len(errors), 96 * 3)
        self._offset_ema = self._offset_alpha * np.mean(errors[-n_bias:]) + (1 - self._offset_alpha) * self._offset_ema
        logger.info("  Offset EMA: %.1f", self._offset_ema)

        # 训练方向分类器 — 用回归预测值 (非实际价格), 与预测时分布一致
        if lag_1d is not None and lag_7d is not None:
            try:
                reg_pred = self.predict(net_load, weather, da_price, lag_1d, lag_7d)
                dir_result = self.fit_dir(y_raw, lag_1d, lag_7d, reg_predictions=reg_pred)
                self.metrics.update(dir_result)
            except Exception as e:
                logger.warning("方向分类器训练失败: %s", e)

        self.metrics = {"mae": mae, "rmse": rmse, "r2": r2, **self.metrics}
        self._trained = True
        return dict(self.metrics)

    # ── 方向分类器 ──

    def _build_dir_features(self, prices: np.ndarray, lag_1d: np.ndarray,
                            lag_7d: np.ndarray) -> np.ndarray:
        """构造方向特征: 严格只用 t 时刻已知的历史数据."""
        n = len(prices)
        feats = {}

        # 短期动量 (基于最近已知价格)
        for w in [1, 4, 12, 24]:
            roc = np.zeros(n)
            for t in range(w + 1, n):
                roc[t] = (prices[t - 1] - prices[t - 1 - w]) / (abs(prices[t - 1 - w]) + 1)
            feats[f"dir_price_roc_{w}"] = roc

        # 日间动量 (基于 lag_1d)
        for w in [1, 4, 12, 24, 48, 96]:
            roc_lag = np.zeros(n)
            for t in range(w + 1, n):
                if t - w < 0:
                    continue
                roc_lag[t] = (lag_1d[t] - lag_1d[t - w]) / (abs(lag_1d[t - w]) + 1)
            feats[f"dir_lag1d_roc_{w}"] = roc_lag

        # lag_1d 自身方向
        lag1d_dir = np.zeros(n)
        lag1d_dir[1:] = np.sign(np.diff(lag_1d))
        feats["dir_lag1d_dir"] = lag1d_dir

        # RSI (基于最近已知价格)
        for w in [4, 12, 24]:
            rsi_vals = np.full(n, 50.0)
            for t in range(w + 2, n):
                segment = prices[t - w:t]
                diffs = np.diff(segment)
                gains = max(diffs.sum(), 0)
                losses = max((-diffs).sum(), 0)
                if losses > 0:
                    rsi_vals[t] = 100 - 100 / (1 + gains / losses)
                elif gains > 0:
                    rsi_vals[t] = 100
            feats[f"dir_rsi_{w}"] = rsi_vals

        # 价格相对位置 (lag_1d 历史分位数)
        for w in [24, 48, 96]:
            rank_lag = np.full(n, 0.5)
            for t in range(w + 1, n):
                window = lag_1d[t - w:t]
                mn, mx = window.min(), window.max()
                if mx - mn > 1:
                    rank_lag[t] = (lag_1d[t] - mn) / (mx - mn)
            feats[f"dir_lag1d_rank_{w}"] = rank_lag

        # 波动率
        for w in [4, 12, 24, 48]:
            vol_vals = np.zeros(n)
            for t in range(w + 1, n):
                segment = prices[t - w:t]
                vol_vals[t] = np.std(segment) / (np.mean(np.abs(segment)) + 1)
            feats[f"dir_volatility_{w}"] = vol_vals

        # 连续涨跌 (到 t-1)
        streak = np.zeros(n)
        for t in range(2, n):
            diff = prices[t - 1] - prices[t - 2]
            if diff > 0:
                streak[t] = max(0, streak[t - 1]) + 1
            elif diff < 0:
                streak[t] = min(0, streak[t - 1]) - 1
        feats["dir_streak"] = streak
        feats["dir_streak_abs"] = np.abs(streak)

        # 日内趋势方向 (lag_1d 序列的15分钟变化)
        intraday_dir = np.zeros(n)
        intraday_dir[1:] = np.sign(lag_1d[1:] - lag_1d[:-1])
        feats["dir_intraday_lag"] = intraday_dir

        # 周对比
        feats["dir_week_spread"] = lag_1d - lag_7d
        feats["dir_week_spread_sign"] = np.sign(lag_1d - lag_7d)

        # 价格加速度
        accel = np.zeros(n)
        for t in range(9, n):
            mom1 = prices[t - 1] - prices[t - 5]
            mom2 = prices[t - 5] - prices[t - 9]
            accel[t] = mom1 - mom2
        feats["dir_price_accel"] = accel

        # 价格交叉 (短期 vs 中期趋势)
        for t in range(24, n):
            short_ma = np.mean(prices[t - 4:t])
            mid_ma = np.mean(prices[t - 24:t])
            feats.setdefault("dir_ma_cross", np.zeros(n))[t] = (
                (short_ma - mid_ma) / (abs(mid_ma) + 1))

        # 上次方向翻转距今时间
        time_since_flip = np.zeros(n)
        last_flip = 0
        for t in range(2, n):
            curr_dir = np.sign(prices[t - 1] - prices[t - 2])
            prev_dir = np.sign(prices[t - 2] - prices[t - 3]) if t >= 3 else 0
            if curr_dir != prev_dir and prev_dir != 0:
                last_flip = t
            time_since_flip[t] = t - last_flip
        feats["dir_time_since_flip"] = time_since_flip

        self._dir_feature_cols = list(feats.keys())
        return np.column_stack([feats[k] for k in self._dir_feature_cols])

    def fit_dir(self, prices: np.ndarray, lag_1d: np.ndarray,
                lag_7d: np.ndarray, reg_predictions: np.ndarray = None) -> Dict:
        """训练方向分类器 (LGBMClassifier).

        Args:
            prices: 实际价格 (用于构造目标标签)
            lag_1d, lag_7d: 滞后特征
            reg_predictions: 回归模型的预测值. 如果提供, 用于构造方向特征
                             (与预测时分布一致); 否则用 prices.
        """
        import lightgbm as lgb

        n = len(prices)
        if n < 10:
            return {"dir_acc": 0.0}

        # 方向特征用回归预测值 (训练/预测分布一致)
        feature_prices = reg_predictions if reg_predictions is not None else prices
        X = self._build_dir_features(feature_prices, lag_1d, lag_7d)

        # 目标: price[t] > price[t-1] (1-step ahead direction)
        y = np.zeros(n, dtype=int)
        y[1:] = (prices[1:] > prices[:-1]).astype(int)

        # 去掉特征中的 NaN
        valid = np.ones(n, dtype=bool)
        for i in range(X.shape[1]):
            valid &= ~np.isnan(X[:, i])
        valid &= ~np.isnan(lag_1d)

        idx = np.where(valid)[0]
        if len(idx) < 100:
            logger.warning("方向分类器数据不足 (%d)", len(idx))
            return {"dir_acc": 0.0}

        X_v, y_v = X[idx], y[idx]

        n_val = max(96 * 3, int(len(idx) * 0.15))

        self._dir_classifier = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.03,
            min_child_samples=30, random_state=42, verbose=-1, n_jobs=1,
        )

        if n_val > 0 and n_val < len(idx):
            self._dir_classifier.fit(
                X_v[:-n_val], y_v[:-n_val],
                eval_set=[(X_v[-n_val:], y_v[-n_val:])],
                eval_metric="binary_logloss",
            )
        else:
            self._dir_classifier.fit(X_v, y_v)

        pred = self._dir_classifier.predict(X_v)
        self._dir_acc = float(np.mean(pred == y_v))
        up_ratio = float(np.mean(y_v))
        baseline = max(up_ratio, 1 - up_ratio)

        logger.info("方向分类器: acc=%.1f%% (baseline=%.1f%%, +%.1f%%)",
                     self._dir_acc * 100, baseline * 100, (self._dir_acc - baseline) * 100)

        # 特征重要性
        importances = self._dir_classifier.feature_importances_
        top_idx = np.argsort(importances)[-8:][::-1]
        top_feats = [(self._dir_feature_cols[i], round(float(importances[i]), 4))
                     for i in top_idx]
        logger.info("  Dir top features: %s", top_feats)

        return {"dir_acc": self._dir_acc, "dir_baseline": baseline}

    def predict_dir(self, prices: np.ndarray, lag_1d: np.ndarray,
                    lag_7d: np.ndarray) -> np.ndarray:
        """预测每步涨跌概率 (1-step ahead)."""
        if self._dir_classifier is None:
            return np.full(len(prices), 0.5)

        X = self._build_dir_features(prices, lag_1d, lag_7d)
        # 填充 NaN
        X = np.nan_to_num(X, nan=0.0)
        return self._dir_classifier.predict_proba(X)[:, 1]

    def predict_with_dir(self, net_load: np.ndarray, weather: pd.DataFrame = None,
                         da_price: np.ndarray = None,
                         lag_1d: np.ndarray = None, lag_7d: np.ndarray = None,
                         past_actual: np.ndarray = None, dir_weight: float = 0.15) -> np.ndarray:
        """回归预测 + 方向分类器趋势倾斜 (保守模式).

        用分类器的聚合方向信号做平滑趋势偏置. 强度很低 (dir_weight=0.15),
        作为方向信号的软融合而非硬校正.

        Returns:
            校正后的价格预测 (与 base 的差异通常 < 2%)
        """
        base = self.predict(net_load, weather, da_price, lag_1d, lag_7d)

        if self._dir_classifier is None or lag_1d is None:
            return base

        n = len(base)
        if n < 2:
            return base

        probs = self.predict_dir(base, lag_1d[:n], lag_7d[:n])

        # 聚合方向信号
        K = min(n, 24)
        weights = np.exp(-np.arange(K) / 8)
        weights = weights / weights.sum()
        dir_signals = (probs[:K] - 0.5) * 2
        agg_signal = float(np.sum(dir_signals * weights))

        if abs(agg_signal) < 0.05:
            return base

        # 平滑趋势偏置 (增强版: 方向分类器 acc=81%, 值得更信任)
        base_std = max(float(np.std(base)), 1.0)
        step = base_std * 0.03 * agg_signal * dir_weight

        corrected = base.copy()
        for i in range(n):
            decay_i = 1.0 if i < K else np.exp(-(i - K) / 12)
            corrected[i] += step * min(i + 1, K) * decay_i

        return corrected

    def compute_trend(self, predictions: np.ndarray) -> dict:
        """从预测序列计算多周期趋势信号.

        Returns:
            dict with keys trend_1h, trend_4h, trend_6h:
            每个元素 ∈ [-1, 1], 正值=看涨, 负值=看跌
        """
        n = len(predictions)
        result = {}
        for label, steps in [("trend_1h", 4), ("trend_4h", 16), ("trend_6h", 24)]:
            signal = np.zeros(n)
            for t in range(n):
                if t + steps < n:
                    diff = predictions[t + steps] - predictions[t]
                    signal[t] = np.tanh(diff / (abs(predictions[t]) + 1) * 10)
                elif t > 0:
                    signal[t] = signal[t - 1]  # 尾部延续
            result[label] = signal
        return result

    def _predict_offset(self, n: int) -> np.ndarray:
        decay = np.exp(-np.arange(n) / (96 * 5))
        return np.full(n, self._offset_ema) * decay

    def _update_offset(self, actual: np.ndarray, predicted: np.ndarray):
        errors = actual - predicted
        recent_errors = errors[-min(len(errors), 96 * 2):]
        self._offset_ema = (self._offset_alpha * np.mean(recent_errors) +
                            (1 - self._offset_alpha) * self._offset_ema)

    def _get_extra_save(self) -> dict:
        extra = {"offset_ema": self._offset_ema, "offset_alpha": self._offset_alpha,
                 "delta_mode": self._delta_mode, "dir_acc": self._dir_acc,
                 "has_dir_classifier": self._dir_classifier is not None,
                 "use_hp_filter": self._use_hp_filter,
                 "hp_lambda": self._hp_lambda,
                 "clean_target": self._clean_target,
                 "outlier_sigma": self._outlier_sigma,
                 "smooth_window": self._smooth_window}
        if self._daily_shape is not None:
            extra["daily_shape"] = self._daily_shape.tolist()
        return extra

    def _set_extra_load(self, meta: dict):
        self._offset_ema = meta.get("offset_ema", 0.0)
        self._offset_alpha = meta.get("offset_alpha", 0.3)
        self._delta_mode = meta.get("delta_mode", True)
        self._dir_acc = meta.get("dir_acc", 0.0)
        self._use_hp_filter = meta.get("use_hp_filter", False)
        self._hp_lambda = meta.get("hp_lambda", 1e8)
        self._clean_target = meta.get("clean_target", True)
        self._outlier_sigma = meta.get("outlier_sigma", 4.0)
        self._smooth_window = meta.get("smooth_window", 3)
        if "daily_shape" in meta:
            self._daily_shape = np.array(meta["daily_shape"])

    def save(self, path: str) -> None:
        """保存模型 + 方向分类器."""
        base = os.path.splitext(path)[0]
        super().save(path)

        if self._dir_classifier is not None:
            import pickle
            dir_path = f"{base}_dir_classifier.pkl"
            with open(dir_path, "wb") as f:
                pickle.dump(self._dir_classifier, f)
            logger.info("方向分类器已保存: %s", dir_path)

    def load(self, path: str) -> None:
        """加载模型 + 方向分类器."""
        super().load(path)

        base = os.path.splitext(path)[0]
        dir_path = f"{base}_dir_classifier.pkl"
        if os.path.exists(dir_path):
            import pickle
            with open(dir_path, "rb") as f:
                self._dir_classifier = pickle.load(f)
            logger.info("方向分类器已加载: %s (acc=%.1f%%)", dir_path, self._dir_acc * 100)
