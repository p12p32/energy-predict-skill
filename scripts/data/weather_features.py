"""weather_features.py — 深度天气特征工程

从基础气象变量派生:
  CDD / HDD:  制冷/采暖度日
  THI:       温湿指数(体感)
  wind_power: 风功率理论值
  solar_potential: 光伏潜力
  temp_change: 温度变化率
  consecutive_hot: 连续高温天数
"""
from typing import Optional
import numpy as np
import pandas as pd


class WeatherFeatureEngineer:
    COOLING_BASE = 26.0   # 制冷基准温度(°C)
    HEATING_BASE = 18.0   # 采暖基准温度(°C)

    @staticmethod
    def _clear_sky_irradiance(lat: float, day_of_year: int, hour: float) -> float:
        """理论晴空太阳辐照度 (W/m²).

        基于太阳几何: 赤纬 + 时角 + 纬度 → 太阳高度角 → 晴空辐照度.
        """
        import numpy as np
        # 太阳赤纬 (Spencer, 1971)
        b = 2 * np.pi * (day_of_year - 1) / 365
        decl = (0.006918 - 0.399912 * np.cos(b) + 0.070257 * np.sin(b)
                - 0.006758 * np.cos(2 * b) + 0.000907 * np.sin(2 * b)
                - 0.002697 * np.cos(3 * b) + 0.001480 * np.sin(3 * b))
        # 时角 (度)
        ha = 15.0 * (hour - 12.0)
        # 太阳高度角正弦
        lat_rad = np.radians(lat)
        sin_elev = (np.sin(lat_rad) * np.sin(decl)
                    + np.cos(lat_rad) * np.cos(decl) * np.cos(np.radians(ha)))
        if sin_elev <= 0:
            return 0.0
        # 大气质量修正
        S0 = 1361.0  # 太阳常数 W/m²
        tau = 0.75   # 大气透射率 (晴空典型值)
        return float(S0 * tau * sin_elev)

    def transform(self, df: pd.DataFrame,
                  lat: Optional[float] = None) -> pd.DataFrame:
        """输入含基础气象列的 DataFrame, 添加派生特征."""
        result = df.copy()

        # ── 制冷/采暖度日 ──
        if "temperature" in result.columns:
            result["CDD"] = np.maximum(result["temperature"] - self.COOLING_BASE, 0)
            result["HDD"] = np.maximum(self.HEATING_BASE - result["temperature"], 0)

        # ── 温湿指数 ──
        if "temperature" in result.columns and "humidity" in result.columns:
            T = result["temperature"]
            RH = result["humidity"]
            # THI = 0.8T + RH×T/500 (体感温度近似)
            result["THI"] = 0.8 * T + RH * T / 500

        # ── 风功率理论值 ──
        if "wind_speed" in result.columns:
            # P ∝ v³ (功率与风速立方成正比)
            ws = result["wind_speed"]
            result["wind_power_potential"] = ws ** 3

        # ── 光伏潜力 ──
        if "solar_radiation" in result.columns:
            result["solar_potential"] = result["solar_radiation"]
            if "temperature" in result.columns:
                # 高温降低光伏效率
                result["solar_efficiency"] = np.where(
                    result["temperature"] > 25,
                    1.0 - 0.004 * (result["temperature"] - 25),
                    1.0
                )

            # cloud_factor: 实际/晴空辐照度, 剥离昼夜循环, 只保留云量信号
            if "dt" in result.columns and lat is not None:
                dts = pd.to_datetime(result["dt"])
                doy = dts.dt.dayofyear.values
                hrs = dts.dt.hour.values + dts.dt.minute.values / 60.0
                clear_sky = np.array([
                    self._clear_sky_irradiance(lat, int(d), float(h))
                    for d, h in zip(doy, hrs)
                ])
                mask = clear_sky > 10  # 白天
                cf = np.zeros(len(clear_sky))
                cf[mask] = np.clip(
                    result["solar_radiation"].values[mask] / clear_sky[mask],
                    0.0, 1.5
                )
                result["cloud_factor"] = cf
            else:
                result["cloud_factor"] = 0.0

        # ── 温度变化率 ──
        if "temperature" in result.columns:
            result["temp_change_1h"] = result["temperature"].diff(4).fillna(0)   # 1小时间隔
            result["temp_change_6h"] = result["temperature"].diff(24).fillna(0)  # 6小时

        # ── 连续高温天数 ──
        if "temperature" in result.columns:
            hot = (result["temperature"] > 35).astype(int)
            result["consecutive_hot_days"] = self._consecutive_count(hot.values, steps_per_day=96)

        # ── 温度极端度 (偏离舒适区程度, 负荷核心驱动) ──
        if "temperature" in result.columns:
            result["temp_extremity"] = np.abs(result["temperature"] - 22) / 15
            if "humidity" in result.columns:
                hum_factor = 1.0 + np.clip((result["humidity"] - 50) / 100, -0.2, 0.3)
                result["temp_extremity"] = result["temp_extremity"] * hum_factor

        # ── 极端天气检测 ──
        if "temperature" in result.columns:
            # 温度异常度 (7日滚动 z-score)
            t_roll = result["temperature"].rolling(672, min_periods=96)
            t_mean = t_roll.mean()
            t_std = t_roll.std().replace(0, 1.0)
            result["temp_zscore"] = ((result["temperature"] - t_mean) / t_std).fillna(0)

            # 热浪: T > 35°C 持续 > 2 天
            hot = (result["temperature"] > 35).astype(int)
            hot_days = self._consecutive_count(hot.values)
            result["is_heat_wave"] = ((result["temperature"] > 35) & (hot_days > 2)).astype(int)

            # 寒潮: T < -5°C 持续 > 2 天
            cold = (result["temperature"] < -5).astype(int)
            cold_days = self._consecutive_count(cold.values)
            result["is_cold_snap"] = ((result["temperature"] < -5) & (cold_days > 2)).astype(int)

        # 极端天气综合标志
        extreme_conditions = []
        if "is_heat_wave" in result.columns:
            extreme_conditions.append(result["is_heat_wave"])
        if "is_cold_snap" in result.columns:
            extreme_conditions.append(result["is_cold_snap"])
        if "wind_speed" in result.columns:
            # 大风: > 15 m/s (7级)
            extreme_conditions.append((result["wind_speed"] > 15).astype(int))
        if "precipitation" in result.columns:
            # 暴雨: > 25 mm/h
            extreme_conditions.append((result["precipitation"] > 25).astype(int))
        if extreme_conditions:
            result["extreme_weather_flag"] = (
                sum(extreme_conditions) > 0
            ).astype(int)
            result["extreme_weather_count"] = (
                sum(extreme_conditions)
            ).astype(int)

        return result

    @staticmethod
    def _consecutive_count(hot_mask: np.ndarray, steps_per_day: int = 96) -> np.ndarray:
        """计算每个时间步的连续高温天数."""
        n = len(hot_mask)
        result = np.zeros(n)
        count = 0
        for i in range(n):
            if hot_mask[i]:
                count += 1
            else:
                count = 0
            result[i] = count / steps_per_day
        return result
