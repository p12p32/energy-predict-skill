"""tests/test_feature_engineer.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestFeatureEngineer:
    def setup_method(self):
        from scripts.data.features import FeatureEngineer
        self.engineer = FeatureEngineer()
        dates = pd.date_range("2025-01-01", periods=500, freq="15min")
        raw = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 500,
            "type": ["load"] * 500,
            "value": 100 + np.sin(np.arange(500) * 0.03) * 30 + np.random.normal(0, 5, 500),
            "price": [0.35] * 500,
        })
        self.raw = raw

    def test_build_features_adds_all_columns(self):
        df = self.engineer.build_features_from_raw(self.raw)
        expected = ["hour", "day_of_week", "month", "is_weekend", "season",
                    "is_holiday", "value_lag_1d", "value_lag_7d",
                    "value_rolling_mean_24h", "bridge_day", "school_holiday",
                    "working_day_type", "quality_flag"]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_lag_features_computed(self):
        df = self.engineer.build_features_from_raw(self.raw)
        assert pd.isna(df["value_lag_1d"].iloc[0])
        assert not pd.isna(df["value_lag_1d"].iloc[100])

    def test_merge_weather_adds_columns(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="h")
        features = pd.DataFrame({
            "dt": dates, "province": ["广东"] * 10,
            "type": ["load"] * 10, "value": range(10),
        })
        weather = pd.DataFrame({
            "dt": dates, "province": ["广东"] * 10,
            "temperature": [12, 13, 14, 13, 12, 11, 10, 11, 13, 14],
            "humidity": [70] * 10, "wind_speed": [3] * 10,
            "wind_direction": [180] * 10, "solar_radiation": [200] * 10,
            "precipitation": [0] * 10, "pressure": [1013] * 10,
        })
        merged = self.engineer.merge_weather(features, weather)
        assert "temperature" in merged.columns


class TestWeatherFeatures:
    def test_cdd_hdd(self):
        from scripts.data.weather_features import WeatherFeatureEngineer
        wfe = WeatherFeatureEngineer()
        df = pd.DataFrame({"temperature": [30, 10, 20, 35]})
        result = wfe.transform(df)
        assert "CDD" in result.columns
        assert "HDD" in result.columns
        assert result["CDD"].iloc[0] == 4.0  # 30 - 26
        assert result["HDD"].iloc[1] == 8.0  # 18 - 10

    def test_thi(self):
        from scripts.data.weather_features import WeatherFeatureEngineer
        wfe = WeatherFeatureEngineer()
        df = pd.DataFrame({"temperature": [30, 20], "humidity": [80, 50]})
        result = wfe.transform(df)
        assert "THI" in result.columns
        assert result["THI"].iloc[0] > result["THI"].iloc[1]  # hot+humid > cool+dry
