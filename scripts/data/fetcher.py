"""data_fetcher.py — 外部数据采集（气象、煤价、碳价）"""
import time
import atexit
from typing import Optional, Dict
from datetime import datetime, timedelta

import requests
import pandas as pd
from scripts.core.config import get_province_coords
from scripts.core.cleanup import is_shutting_down

OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

WEATHER_HOURLY_PARAMS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_100m",
    "wind_direction_100m", "direct_radiation", "precipitation",
    "surface_pressure",
]

WEATHER_COLUMN_MAP = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_100m": "wind_speed",
    "wind_direction_100m": "wind_direction",
    "direct_radiation": "solar_radiation",
    "precipitation": "precipitation",
    "surface_pressure": "pressure",
}


class WeatherFetcher:
    def __init__(self, source: str = "open-meteo",
                 max_retries: int = 2, retry_delay: float = 2.0):
        self.source = source
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = 15  # 历史API较慢
        self._session = None
        atexit.register(self.close)

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def close(self):
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def fetch_historical(self, province: str, lat: float, lon: float,
                         start_date: str, end_date: str) -> pd.DataFrame:
        url = OPEN_METEO_HISTORICAL
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_HOURLY_PARAMS),
            "timezone": "Asia/Shanghai",
        }
        data = self._request(url, params)
        return self._parse_response(data, province, is_forecast=False)

    def fetch_forecast(self, province: str, lat: float, lon: float,
                       forecast_days: int = 7) -> pd.DataFrame:
        url = OPEN_METEO_FORECAST
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_HOURLY_PARAMS),
            "forecast_days": forecast_days,
            "timezone": "Asia/Shanghai",
        }
        data = self._request(url, params)
        return self._parse_response(data, province, is_forecast=True)

    def _request(self, url: str, params: Dict) -> Dict:
        session = self._get_session()
        for attempt in range(self.max_retries):
            if is_shutting_down():
                raise RuntimeError("进程正在关闭，取消网络请求")
            try:
                resp = session.get(url, params=params, timeout=self.request_timeout)
                if resp.status_code == 200:
                    return resp.json()
            except requests.exceptions.Timeout:
                if attempt >= self.max_retries - 1:
                    raise RuntimeError(f"气象 API 超时 ({self.request_timeout}s): {url}")
            except requests.exceptions.ConnectionError:
                if attempt >= self.max_retries - 1:
                    raise RuntimeError(f"气象 API 连接失败: {url}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"气象 API 请求失败: {url}")

    def _parse_response(self, data: Dict, province: str,
                        is_forecast: bool) -> pd.DataFrame:
        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["dt"] = pd.to_datetime(df.pop("time"))

        for src, dst in WEATHER_COLUMN_MAP.items():
            if src in df.columns:
                df.rename(columns={src: dst}, inplace=True)
            else:
                df[dst] = None

        df["province"] = province

        want_cols = ["dt", "province", "temperature", "humidity",
                     "wind_speed", "wind_direction", "solar_radiation",
                     "precipitation", "pressure"]

        if is_forecast:
            df["forecast_hour"] = range(len(df))

        result = df[[c for c in want_cols if c in df.columns]]
        return result


class DataFetcher:
    def __init__(self):
        self.weather = WeatherFetcher(source="open-meteo")
        self._coords = get_province_coords()

    def fetch_weather(self, province: str, start_date: str,
                      end_date: str = None, mode: str = "historical",
                      forecast_days: int = 7) -> pd.DataFrame:
        coords = self._coords.get(province)
        if not coords:
            raise ValueError(f"未找到省份坐标: {province}")

        if mode == "historical":
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            return self.weather.fetch_historical(
                province, coords["lat"], coords["lon"],
                start_date, end_date
            )
        elif mode == "forecast":
            return self.weather.fetch_forecast(
                province, coords["lat"], coords["lon"],
                forecast_days=forecast_days
            )
        else:
            raise ValueError(f"mode must be 'historical' or 'forecast', got: {mode}")

    def fetch_weather_for_all_provinces(self, start_date: str,
                                         end_date: str = None,
                                         mode: str = "historical") -> pd.DataFrame:
        frames = []
        for province in self._coords:
            df = self.fetch_weather(province, start_date, end_date, mode)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
