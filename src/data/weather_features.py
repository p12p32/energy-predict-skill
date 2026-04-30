"""weather_features.py — 深度天气特征工程

从基础气象变量派生:
  CDD / HDD:  制冷/采暖度日
  THI:       温湿指数(体感)
  wind_power: 风功率理论值
  solar_potential: 光伏潜力
  temp_change: 温度变化率
  consecutive_hot: 连续高温天数
"""
import numpy as np
import pandas as pd


class WeatherFeatureEngineer:
    COOLING_BASE = 26.0   # 制冷基准温度(°C)
    HEATING_BASE = 18.0   # 采暖基准温度(°C)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

        # ── 温度变化率 ──
        if "temperature" in result.columns:
            result["temp_change_1h"] = result["temperature"].diff(4).fillna(0)   # 1小时间隔
            result["temp_change_6h"] = result["temperature"].diff(24).fillna(0)  # 6小时

        # ── 连续高温天数 ──
        if "temperature" in result.columns:
            hot = (result["temperature"] > 35).astype(int)
            result["consecutive_hot_days"] = self._consecutive_count(hot.values, steps_per_day=96)

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
