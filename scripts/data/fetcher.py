"""data_fetcher.py — Open-Meteo 气象数据采集 + 本地 CSV 缓存

优化:
- 使用 forecast API 的 past_days 参数，一次请求同时获取历史+预报
- 增加气象变量: 云量、阵风、DNI、DHI、短波辐射
- 小时数据→15分钟插值
- 批量预取 + 速率限制
"""
import time
import os
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
from scripts.core.config import get_province_coords, get_cache_config

logger = logging.getLogger(__name__)

# Open-Meteo API endpoints
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# 请求的气象变量 (全量)
FORECAST_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
]

# 历史数据额外变量 (archive API 支持更多)
ARCHIVE_HOURLY_VARS = FORECAST_HOURLY_VARS + [
    "surface_pressure",
    "wind_speed_100m",
    "wind_direction_100m",
    "et0_fao_evapotranspiration",
]

# API 响应列名 → 内部列名
WEATHER_COLUMN_MAP = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "precipitation": "precipitation",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed_10m",
    "wind_direction_10m": "wind_direction_10m",
    "wind_gusts_10m": "wind_gusts",
    "shortwave_radiation": "shortwave_radiation",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "surface_pressure": "pressure",
    "wind_speed_100m": "wind_speed_100m",
    "wind_direction_100m": "wind_direction_100m",
    "et0_fao_evapotranspiration": "evapotranspiration",
}

# 所有内部列名
WEATHER_ALL_COLS = list(set(WEATHER_COLUMN_MAP.values()))


class WeatherFetcher:
    """Open-Meteo 气象数据采集器.

    使用策略:
    - 近期历史 (< 7 天): 用 forecast API 的 past_days 参数 (更快更可靠)
    - 远期历史 (> 7 天): 用 archive API (覆盖更久)
    - 未来预报: 用 forecast API
    """

    def __init__(self, timeout: float = 15.0, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time = 0.0
        self._min_interval = 1.0  # 请求间隔 1s (Open-Meteo 免费限制)

    @property
    def cache_dir(self) -> str:
        cache_cfg = get_cache_config()
        skill_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(skill_home, cache_cfg.get("weather_dir", ".energy_data/weather"))

    # ── 缓存管理 ──

    def _cache_path(self, province: str, start: str, end: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        safe = lambda d: d.replace("-", "")
        return os.path.join(self.cache_dir, f"w_{province}_{safe(start)}_{safe(end)}.csv")

    def _save_cache(self, path: str, df: pd.DataFrame):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def _load_cache(self, path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=["dt"])
            return df
        except Exception as e:
            logger.warning("缓存读取失败 %s: %s", path, e)
            return None

    # ── 速率限制 ──

    def _rate_limit(self):
        """确保请求间隔 >= _min_interval."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    # ── HTTP 请求 ──

    def _request(self, url: str, params: Dict) -> Dict:
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning("Open-Meteo 限速 429, 等待 %ds", wait)
                    time.sleep(wait)
                else:
                    logger.warning("Open-Meteo 错误 %d: %s", resp.status_code, resp.text[:200])
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
            except requests.Timeout:
                logger.warning("Open-Meteo 超时 (attempt %d/%d)", attempt + 1, self.max_retries)
                if attempt < self.max_retries - 1:
                    time.sleep(3)
            except requests.RequestException as e:
                logger.warning("Open-Meteo 请求异常: %s", e)
                if attempt < self.max_retries - 1:
                    time.sleep(3)

        raise RuntimeError(f"气象 API 请求失败: {url}")

    # ── 数据解析 ──

    def _parse_hourly(self, data: Dict, province: str) -> pd.DataFrame:
        """解析 hourly 响应为 DataFrame (小时级)."""
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df["dt"] = pd.to_datetime(df.pop("time"))
        df["province"] = province

        # 重命名
        for src, dst in WEATHER_COLUMN_MAP.items():
            if src in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        want_cols = ["dt", "province"] + [c for c in WEATHER_ALL_COLS if c in df.columns]
        return df[want_cols]

    def _hourly_to_15min(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """将小时数据插值为 15 分钟数据.

        策略:
        - 温度/湿度/气压/风速风向: 线性插值
        - 降水: 向后填充 (小时总量均分到 4 个 15 分钟)
        - 辐照度/云量: 线性插值, 裁剪非负
        """
        if df_hourly.empty:
            return df_hourly

        df_hourly = df_hourly.sort_values("dt").copy()
        df_hourly = df_hourly.set_index("dt")

        # 生成 15 分钟时间索引
        freq_15min = pd.date_range(
            start=df_hourly.index.min(),
            end=df_hourly.index.max() + timedelta(hours=1) - timedelta(minutes=15),
            freq="15min"
        )

        df_15min = pd.DataFrame(index=freq_15min)
        df_15min.index.name = "dt"

        # 线性插值的列
        interp_cols = [c for c in df_hourly.columns if c != "province"
                       and c not in ("precipitation",)]

        for col in interp_cols:
            if col in df_hourly.columns:
                series = df_hourly[col].astype(float)
                interpolated = series.reindex(df_15min.index).interpolate(method="time")
                # 辐照度/云量等不能为负
                if col in ("shortwave_radiation", "dni", "dhi", "cloud_cover",
                            "wind_gusts", "wind_speed_10m", "wind_speed_100m"):
                    interpolated = interpolated.clip(lower=0)
                df_15min[col] = interpolated

        # 降水: 小时总量 → 4 份
        if "precipitation" in df_hourly.columns:
            precip_hourly = df_hourly["precipitation"].astype(float) / 4.0
            precip_15 = precip_hourly.reindex(df_15min.index).ffill()
            df_15min["precipitation"] = precip_15

        # 恢复 province
        if "province" in df_hourly.columns:
            df_15min["province"] = df_hourly["province"].iloc[0]

        df_15min = df_15min.reset_index()
        return df_15min

    # ── 对外接口 ──

    def fetch_recent(self, province: str, lat: float, lon: float,
                     past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
        """获取近期历史 + 未来预报 (单次请求).

        适用于预测场景: 过去 N 天 + 未来 M 天.
        返回 15 分钟粒度数据.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(FORECAST_HOURLY_VARS),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Asia/Shanghai",
        }

        data = self._request(OPEN_METEO_FORECAST, params)
        df_hourly = self._parse_hourly(data, province)

        if not df_hourly.empty:
            # 缓存
            start_str = df_hourly["dt"].min().strftime("%Y-%m-%d")
            end_str = df_hourly["dt"].max().strftime("%Y-%m-%d")
            cache_path = self._cache_path(province, start_str, end_str)
            self._save_cache(cache_path, df_hourly)

        return self._hourly_to_15min(df_hourly)

    def fetch_historical(self, province: str, lat: float, lon: float,
                         start_date: str, end_date: str) -> pd.DataFrame:
        """获取远期历史气象 (archive API).

        返回小时数据 (archive API 不支持 15 分钟).
        """
        # 检查缓存
        cache_path = self._cache_path(province, start_date, end_date)
        cached = self._load_cache(cache_path)
        if cached is not None and not cached.empty:
            logger.info("气象缓存命中: %s [%s~%s]", province, start_date, end_date)
            return cached

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(ARCHIVE_HOURLY_VARS),
            "timezone": "Asia/Shanghai",
        }

        data = self._request(OPEN_METEO_ARCHIVE, params)
        df_hourly = self._parse_hourly(data, province)

        if not df_hourly.empty:
            self._save_cache(cache_path, df_hourly)
            logger.info("气象下载: %s [%s~%s] %d 行", province, start_date, end_date, len(df_hourly))

        return self._hourly_to_15min(df_hourly)

    def fetch_for_range(self, province: str, lat: float, lon: float,
                        start_date: str, end_date: str) -> pd.DataFrame:
        """智能获取指定时间段的气象数据.

        近 7 天用 forecast API (含更多变量), 更早的用 archive API.
        自动拼接.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        now = pd.Timestamp.now()

        cutoff = now - timedelta(days=7)
        cutoff = pd.Timestamp(cutoff.year, cutoff.month, cutoff.day)

        frames = []

        # 远期: archive
        if start_dt < cutoff:
            archive_end = min(cutoff, end_dt).strftime("%Y-%m-%d")
            try:
                df_hist = self.fetch_historical(province, lat, lon,
                                                start_date, archive_end)
                if not df_hist.empty:
                    frames.append(df_hist)
            except Exception as e:
                logger.warning("archive 气象获取失败: %s", e)

        # 近期 + 预报: forecast API
        if end_dt >= cutoff:
            past_days = max(0, (now.date() - max(start_dt.date(), cutoff.date())).days)
            forecast_days = max(1, (end_dt.date() - now.date()).days + 1)
            try:
                df_recent = self.fetch_recent(province, lat, lon,
                                              past_days=min(past_days, 7),
                                              forecast_days=forecast_days)
                if not df_recent.empty:
                    frames.append(df_recent)
            except Exception as e:
                logger.warning("forecast 气象获取失败: %s", e)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["dt"], keep="last")
        result = result.sort_values("dt").reset_index(drop=True)

        # 裁剪到请求范围
        result = result[(result["dt"] >= start_dt) & (result["dt"] <= end_dt)]

        return result


class DataFetcher:
    """数据采集入口, 封装气象/市场数据."""

    def __init__(self):
        self.weather = WeatherFetcher()
        self._coords = get_province_coords()

    def fetch_weather(self, province: str, start_date: str,
                      end_date: str = None) -> pd.DataFrame:
        """获取气象数据 (自动选择 archive/forecast).

        Args:
            province: 省份名
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 结束日期, 默认到明天

        Returns:
            15 分钟粒度的气象 DataFrame
        """
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        coords = self._coords.get(province)
        if not coords:
            raise ValueError(f"未找到省份坐标: {province}")

        return self.weather.fetch_for_range(
            province, coords["lat"], coords["lon"],
            start_date, end_date
        )

    def preload_historical_weather(self, provinces: List[str],
                                   start_date: str,
                                   end_date: str) -> Dict[str, pd.DataFrame]:
        """批量预取多个省份气象数据.

        Returns:
            {province: DataFrame} 字典
        """
        cache: Dict[str, pd.DataFrame] = {}
        for province in provinces:
            coords = self._coords.get(province)
            if not coords:
                continue
            try:
                df = self.weather.fetch_for_range(
                    province, coords["lat"], coords["lon"],
                    start_date, end_date
                )
                if not df.empty:
                    cache[province] = df
                    logger.info("气象预取: %s (%d 行)", province, len(df))
            except Exception as e:
                logger.warning("气象预取失败 %s: %s", province, e)
        return cache

    def fetch_weather_for_all_provinces(self, start_date: str,
                                         end_date: str = None) -> pd.DataFrame:
        """所有省份气象数据合并."""
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        frames = []
        for province in self._coords:
            try:
                df = self.fetch_weather(province, start_date, end_date)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning("气象获取失败 %s: %s", province, e)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
