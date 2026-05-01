"""tests/test_executor.py — StrategyExecutor 单元测试"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-04-01", periods=n, freq="15min")
    return pd.DataFrame({
        "dt": dates,
        "province": "广东",
        "type": "load",
        "value": 100 + np.random.normal(0, 10, n),
        "temperature": 25 + np.random.normal(0, 3, n),
        "humidity": 60 + np.random.normal(0, 5, n),
        "wind_speed": 3 + np.random.normal(0, 1, n),
        "hour": dates.hour,
        "day_of_week": dates.dayofweek,
        "is_holiday": np.zeros(n, dtype=bool),
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "season": np.full(n, 2),
        "value_lag_1d": 100 + np.random.normal(0, 8, n),
        "value_lag_7d": 100 + np.random.normal(0, 5, n),
        "value_rolling_mean_24h": 100 + np.random.normal(0, 3, n),
        "solar_radiation": np.maximum(0, 300 * np.sin(dates.hour * np.pi / 12)),
    })


@pytest.fixture
def executor():
    from scripts.ml.executor import StrategyExecutor
    return StrategyExecutor()


# ============================================================
# TestExecute
# ============================================================

class TestExecute:
    def test_execute_dispatches_to_correct_method(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "polynomial_features", "params": {"power": 2}})
        assert "temperature_p2" in result.columns
        assert (result["temperature_p2"] == sample_df["temperature"] ** 2).all()

    def test_unknown_strategy_returns_unchanged(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "nonexistent_strategy"})
        pd.testing.assert_frame_equal(result, sample_df)

    def test_execute_all_chains_strategies(self, executor, sample_df):
        strategies = [
            {"name": "polynomial_features", "params": {"power": 2}},
            {"name": "dayofweek_interaction", "params": {"interact_with": ["hour"]}},
        ]
        result = executor.execute_all(sample_df, strategies)
        assert "temperature_p2" in result.columns
        assert "dow_hour" in result.columns


# ============================================================
# TestPolynomialFeatures
# ============================================================

class TestPolynomialFeatures:
    def test_adds_power_column(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "polynomial_features", "params": {"power": 2}})
        assert "temperature_p2" in result.columns
        np.testing.assert_array_almost_equal(
            result["temperature_p2"].values, sample_df["temperature"].values ** 2
        )

    def test_respects_power_param(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "polynomial_features", "params": {"power": 3}})
        col = [c for c in result.columns if c.endswith("_p3")]
        assert len(col) > 0

    def test_skips_existing_columns(self, executor, sample_df):
        df = sample_df.copy()
        df["temperature_p2"] = 999.0
        result = executor.execute(df, {"name": "polynomial_features", "params": {"power": 2}})
        assert (result["temperature_p2"] == 999.0).all()


# ============================================================
# TestDayofweekInteraction
# ============================================================

class TestDayofweekInteraction:
    def test_adds_dow_interactions(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "dayofweek_interaction",
                                               "params": {"interact_with": ["hour"]}})
        assert "dow_hour" in result.columns
        expected = sample_df["day_of_week"] * sample_df["hour"]
        assert (result["dow_hour"] == expected).all()

    def test_handles_missing_dayofweek(self, executor, sample_df):
        df = sample_df.drop(columns=["day_of_week"])
        result = executor.execute(df, {"name": "dayofweek_interaction"})
        pd.testing.assert_frame_equal(result, df)


# ============================================================
# TestRollingWindowFeatures
# ============================================================

class TestRollingWindowFeatures:
    def test_adds_rolling_mean(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "rolling_window_features",
                                               "params": {"window_hours": 4}})
        assert "value_roll4h" in result.columns
        expected = sample_df["value"].rolling(window=16, min_periods=1).mean()
        assert (result["value_roll4h"] == expected).all()


# ============================================================
# TestDataLevelTransforms
# ============================================================

class TestDataLevelTransforms:
    def test_recent_upsample_duplicates(self, executor):
        n = 20
        dates = pd.date_range("2026-04-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "dt": dates,
            "value": np.arange(n, dtype=float),
        })
        result = executor.execute(df, {"name": "recent_upsample",
                                        "params": {"weight": 3, "days": 7}})
        # 最近的 7 天数据应被复制
        assert len(result) > len(df)

    def test_extreme_oversample(self, executor):
        n = 100
        dates = pd.date_range("2026-04-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "dt": dates,
            "value": np.concatenate([np.full(95, 10.0), np.full(5, 100.0)]),
        })
        result = executor.execute(df, {"name": "extreme_oversample",
                                        "params": {"factor": 3, "target_col": "value"}})
        assert len(result) > len(df)

    def test_extreme_oversample_no_extremes(self, executor):
        n = 50
        dates = pd.date_range("2026-04-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "dt": dates,
            "value": np.full(n, 50.0),
        })
        result = executor.execute(df, {"name": "extreme_oversample",
                                        "params": {"factor": 3}})
        pd.testing.assert_frame_equal(result, df)

    def test_shorter_window_filters(self, executor):
        n = 200
        dates = pd.date_range("2026-03-01", periods=n, freq="15min")
        df = pd.DataFrame({
            "dt": dates,
            "value": np.arange(n, dtype=float),
        })
        result = executor.execute(df, {"name": "shorter_window",
                                        "params": {"n": 7}})
        assert len(result) <= len(df)
        window_span = result["dt"].max() - result["dt"].min()
        assert window_span <= timedelta(days=7)


# ============================================================
# TestPassthrough
# ============================================================

class TestPassthrough:
    def test_province_independent_unchanged(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "province_independent_model"})
        pd.testing.assert_frame_equal(result, sample_df)

    def test_bias_correction_unchanged(self, executor, sample_df):
        result = executor.execute(sample_df, {"name": "bias_correction"})
        pd.testing.assert_frame_equal(result, sample_df)
