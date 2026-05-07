"""constraints.py — 物理约束层: 规则后处理, 无训练."""
import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PhysicalConstraints:
    def apply(self, p50: np.ndarray, p10: np.ndarray, p90: np.ndarray,
              future_df: pd.DataFrame, target_type: str,
              province: str = "", config: Dict = None) -> tuple:
        p50, p10, p90 = p50.copy(), p10.copy(), p90.copy()

        # 1. 非负裁剪
        if "电价" not in target_type:
            p50 = np.maximum(p50, 0)
            p10 = np.maximum(p10, 0)
            p90 = np.maximum(p90, 0)

        # 2. 光伏夜间归零
        if "光伏" in target_type and "dt" in future_df.columns:
            self._solar_night_zero(p50, p10, p90, future_df, province, config)

        # 3. 风电功率曲线上限
        if "风电" in target_type and "wind_power_curve" in future_df.columns:
            self._wind_power_cap(p50, p10, p90, future_df, config)

        # 4. 联络线容量
        if "联络线" in target_type and config:
            self._tie_line_cap(p50, p10, p90, config, province)

        return p50, p10, p90

    def _solar_night_zero(self, p50, p10, p90, future_df, province, config):
        from scripts.data.weather_features import WeatherFeatureEngineer as WFE
        coords = (config or {}).get("province_coords", {})
        lat = coords.get(province, {}).get("lat") if coords else None
        dts = pd.to_datetime(future_df["dt"])
        doy = dts.dt.dayofyear.values
        hrs = dts.dt.hour.values + dts.dt.minute.values / 60.0
        if lat is not None:
            clear_sky = np.array([
                WFE._clear_sky_irradiance(lat, int(d), float(h))
                for d, h in zip(doy, hrs)
            ])
            night = clear_sky <= 10
        else:
            hours = dts.dt.hour
            mins = dts.dt.minute
            night = (hours >= 20) | (hours < 6) | ((hours == 6) & (mins < 15))
        p50[night] = 0
        p10[night] = 0
        p90[night] = 0

    def _wind_power_cap(self, p50, p10, p90, future_df, config):
        curve = future_df["wind_power_curve"].values
        cap_mw = (config or {}).get("wind_capacity_mw", None)
        if cap_mw:
            max_power = curve * cap_mw
            p50 = np.minimum(p50, max_power)
            p10 = np.minimum(p10, max_power)
            p90 = np.minimum(p90, max_power)

    def _tie_line_cap(self, p50, p10, p90, config, province):
        caps = (config or {}).get("tie_line_caps", {})
        cap = caps.get(province, 3000)
        p50 = np.clip(p50, -cap, cap)
        p10 = np.clip(p10, -cap, cap)
        p90 = np.clip(p90, -cap, cap)
