"""tests/test_predictor.py — 预测执行器单元测试"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock


# ============================================================
# 合成数据 fixture
# ============================================================

def make_history(n_steps: int = 500, province: str = "广东",
                 target_type: str = "load", seed: int = 42) -> pd.DataFrame:
    """生成可用于 predictor 测试的历史特征数据."""
    np.random.seed(seed)
    dates = pd.date_range(
        datetime(2026, 4, 1), periods=n_steps, freq="15min"
    )
    hour = dates.hour.values.astype(float)
    dow = dates.dayofweek.values.astype(float)
    month = dates.month.values.astype(float)

    # 日周期 + 噪声
    value = (
        100
        + 30 * np.sin((hour - 6) * np.pi / 12)
        + 15 * np.sin((hour - 14) * np.pi / 8)
        + np.random.normal(0, 3, n_steps)
    )
    value = np.maximum(value, 10)

    df = pd.DataFrame({
        "dt": dates, "province": province, "type": target_type,
        "value": value,
        "price": np.full(n_steps, 0.35),
        "hour": hour.astype(int),
        "day_of_week": dow.astype(int),
        "day_of_month": dates.day.values,
        "month": month.astype(int),
        "is_weekend": (dow >= 5).astype(int),
        "season": np.select(
            [month == 4, month == 5],
            [1, 1],
            default=2,
        ),
        "temperature": 22 + 5 * np.sin((hour - 10) * np.pi / 12),
        "humidity": 60 + np.random.normal(0, 5, n_steps),
        "wind_speed": 3 + np.random.normal(0, 1, n_steps),
        "wind_direction": 180 + np.random.normal(0, 20, n_steps),
        "solar_radiation": np.maximum(0, 300 * np.sin(hour * np.pi / 12)),
        "precipitation": np.maximum(0, np.random.exponential(1, n_steps)),
        "pressure": 1013 + np.random.normal(0, 3, n_steps),
        "value_lag_1d": value,
        "value_lag_7d": value,
        "value_rolling_mean_24h": value,
        "value_diff_1d": np.zeros(n_steps),
        "value_diff_7d": np.zeros(n_steps),
        "quality_flag": 0,
        "peak_valley": np.zeros(n_steps),
        "weekend_hour": np.zeros(n_steps),
        "dow_hour": np.zeros(n_steps),
        "weekend_x_lag7d": np.zeros(n_steps),
        "hour_x_lag1d": np.zeros(n_steps),
    })

    # 添加滞后真实值
    for i in range(n_steps):
        if i >= 96:
            df.loc[i, "value_lag_1d"] = df.loc[i - 96, "value"]
        if i >= 672:
            df.loc[i, "value_lag_7d"] = df.loc[i - 672, "value"]

    return df


# ============================================================
# 测试 _build_future_features
# ============================================================

class TestBuildFutureFeatures:
    def test_output_shape(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(500)
        horizon = 12  # 3 hours
        base_dt = history["dt"].iloc[-1]

        future = p._build_future_features(
            history, "广东", "load", horizon, base_dt
        )

        assert len(future) == horizon
        assert "dt" in future.columns
        assert future["province"].iloc[0] == "广东"
        assert future["type"].iloc[0] == "load"

    def test_time_features(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(500)
        base_dt = datetime(2026, 4, 15, 12, 0)

        future = p._build_future_features(
            history, "广东", "load", 96, base_dt
        )

        assert future["hour"].iloc[0] == 12  # first step = 12:15
        assert future["hour"].iloc[4] == 13  # 4 x 15min = 1hr later
        assert all(future["month"] == 4)

    def test_lag_fill_forward(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(500)
        last_val = history["value"].iloc[-1]

        future = p._build_future_features(
            history, "广东", "load", 96, history["dt"].iloc[-1]
        )

        # 第一行应该取 history.iloc[-(96-0)] = history.iloc[-96]
        # lag_1d 步是 96，第一行 i=0 < 96，从 history 尾部取
        assert "value_lag_1d" in future.columns
        assert "value_lag_7d" in future.columns
        # 基本检查: lag 值不是 NaN
        assert not future["value_lag_1d"].isna().any()

    def test_no_history_weather_cols_filled(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(200)
        history_with_weather = history.copy()
        history_with_weather["temperature"] = 25.0
        history_with_weather["humidity"] = 60.0

        with patch.object(p.fetcher, "fetch_weather", return_value=pd.DataFrame()):
            future = p._build_future_features(
                history_with_weather, "广东", "load", 24,
                history_with_weather["dt"].iloc[-1],
            )

        assert "temperature" in future.columns
        assert not future["temperature"].isna().any()


# ============================================================
# 测试 _predict_with_model
# ============================================================

class TestPredictWithModel:
    def test_output_columns(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(100)
        base_dt = history["dt"].iloc[-1]
        future = p._build_future_features(
            history, "广东", "load", 10, base_dt
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = np.full(10, 100.0)
        feature_names = ["hour", "day_of_week", "month", "temperature"]

        result = p._predict_with_model(
            mock_model, future, feature_names,
            province="广东", target_type="load",
        )

        assert list(result.columns) == ["dt", "province", "type", "p50", "p10", "p90", "model_version"]
        assert len(result) == 10
        assert all(result["p50"] == 100.0)
        assert all(result["p10"] <= result["p50"])
        assert all(result["p90"] >= result["p50"])

    def test_missing_features_filled_with_zero(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(50)
        base_dt = history["dt"].iloc[-1]
        future = p._build_future_features(
            history, "广东", "load", 5, base_dt
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = np.full(5, 200.0)
        feature_names = ["hour", "nonexistent_feature"]

        # Should not crash — missing features get filled with 0
        result = p._predict_with_model(
            mock_model, future, feature_names,
            province="广东", target_type="load",
        )
        assert len(result) == 5

    def test_residual_std_from_history(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(200)
        base_dt = history["dt"].iloc[-1]
        future = p._build_future_features(
            history, "广东", "load", 5, base_dt
        )

        feature_names = ["hour", "day_of_week"]

        def mock_predict_side_effect(X):
            if len(X) == 96:
                # 历史预测加入随机噪声, 使 residual_std > 0
                return history["value"].tail(96).values * (1 + np.random.normal(0, 0.03, 96))
            else:
                return np.full(len(X), 110.0)

        mock_model = MagicMock()
        mock_model.predict.side_effect = mock_predict_side_effect

        result = p._predict_with_model(
            mock_model, future, feature_names,
            province="广东", target_type="load",
            history=history,
        )

        # P10 < P50 < P90 (residual_std > 0)
        assert all(result["p10"] < result["p50"])
        assert all(result["p90"] > result["p50"])


# ============================================================
# 测试 _ensemble
# ============================================================

class TestEnsemble:
    def test_shape_preserved(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(200)
        n = 24

        lgb_result = pd.DataFrame({
            "dt": pd.date_range("2026-04-15 12:15", periods=n, freq="15min"),
            "province": "广东",
            "type": "load",
            "p10": np.full(n, 80.0),
            "p50": np.full(n, 100.0),
            "p90": np.full(n, 120.0),
            "model_version": "v1",
        })
        trend_preds = np.full(n, 105.0)

        result = p._ensemble(lgb_result, trend_preds, history)

        assert len(result) == n
        assert "p50" in result.columns
        assert "p10" in result.columns
        assert "p90" in result.columns

    def test_blending_weights_decay(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(200)
        n = 96

        lgb_result = pd.DataFrame({
            "dt": pd.date_range("2026-04-15 12:15", periods=n, freq="15min"),
            "province": "广东",
            "type": "load",
            "p10": np.full(n, 50.0),
            "p50": np.full(n, 100.0),
            "p90": np.full(n, 150.0),
            "model_version": "v1",
        })
        trend_preds = np.full(n, 200.0)

        result = p._ensemble(lgb_result, trend_preds, history)

        # 早期: LGB 权重大 → P50 接近 100
        # 晚期: Trend 权重大 → P50 接近 200
        early_p50 = result["p50"].iloc[0]
        late_p50 = result["p50"].iloc[-1]
        assert early_p50 < late_p50, f"early_p50={early_p50:.1f}, late_p50={late_p50:.1f}"

    def test_all_values_non_negative(self):
        """load 类型应钳制所有负值到 0."""
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(200)

        lgb_result = pd.DataFrame({
            "dt": pd.date_range("2026-04-15", periods=24, freq="15min"),
            "province": "广东",
            "type": "load",
            "p10": np.full(24, -10.0),  # 负值应被截断
            "p50": np.full(24, 50.0),
            "p90": np.full(24, 100.0),
            "model_version": "v1",
        })
        trend_preds = np.full(24, 30.0)

        result = p._ensemble(lgb_result, trend_preds, history,
                            target_type="load")

        assert (result["p10"] >= 0).all()
        assert (result["p50"] >= 0).all()
        assert (result["p90"] >= 0).all()

    def test_price_can_be_negative(self):
        """price 类型允许负值通过，不被钳制."""
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(200)

        lgb_result = pd.DataFrame({
            "dt": pd.date_range("2026-04-15", periods=24, freq="15min"),
            "province": "山东",
            "type": "price",
            "p10": np.full(24, -50.0),
            "p50": np.full(24, -20.0),
            "p90": np.full(24, 10.0),
            "model_version": "v1",
        })
        trend_preds = np.full(24, -15.0)

        result = p._ensemble(lgb_result, trend_preds, history,
                            target_type="price")

        # price 类型不钳制负值
        assert (result["p10"] < 0).any()
        assert (result["p50"] < 0).any()

    def test_asymmetry_preserved(self):
        """P50 偏移后, 分位数模型的非对称间隔应保持."""
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(200)

        lgb_result = pd.DataFrame({
            "dt": pd.date_range("2026-04-15", periods=24, freq="15min"),
            "province": "广东",
            "type": "load",
            "p10": np.full(24, 80.0),
            "p50": np.full(24, 100.0),
            "p90": np.full(24, 130.0),  # 上宽下窄: 20 下, 30 上
            "model_version": "v1",
        })
        trend_preds = np.full(24, 105.0)

        result = p._ensemble(lgb_result, trend_preds, history)

        lo_width = result["p50"].values - result["p10"].values
        hi_width = result["p90"].values - result["p50"].values
        # 原始非对称性应保持: hi_width ≈ 30, lo_width ≈ 20
        assert np.allclose(hi_width - lo_width, 10.0, atol=2.0)


# ============================================================
# 测试 _calibrate_intervals
# ============================================================

class TestCalibrateIntervals:
    def test_no_history_returns_unchanged(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()

        ensemble = pd.DataFrame({
            "p10": [80.0], "p50": [100.0], "p90": [120.0],
        })
        empty_history = pd.DataFrame({"value": []})

        result = p._calibrate_intervals(ensemble, empty_history, "广东", "load")
        pd.testing.assert_frame_equal(result, ensemble)

    def test_short_history_returns_unchanged(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(50)  # < 96

        ensemble = pd.DataFrame({
            "p10": [80.0], "p50": [100.0], "p90": [120.0],
        })

        result = p._calibrate_intervals(ensemble, history, "广东", "load")
        pd.testing.assert_frame_equal(result, ensemble)

    def test_no_quantile_models_returns_unchanged(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        p.trainer.load_quantile_models = MagicMock(
            side_effect=FileNotFoundError("no models")
        )
        history = make_history(200)

        ensemble = pd.DataFrame({
            "p10": [80.0], "p50": [100.0], "p90": [120.0],
        })

        result = p._calibrate_intervals(ensemble, history, "广东", "load")
        pd.testing.assert_frame_equal(result, ensemble)

    def test_calibrated_intervals_stay_non_negative(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()
        history = make_history(300)

        # 构造假的分位数模型
        mock_p50 = MagicMock()
        mock_p50.predict.return_value = history["value"].values[-672:] * 0.95
        mock_p10 = MagicMock()
        mock_p10.predict.return_value = history["value"].values[-672:] * 0.80

        p.trainer.load_quantile_models = MagicMock(return_value={
            "p50": (mock_p50, ["hour", "day_of_week", "month", "temperature"]),
            "p10": (mock_p10, ["hour", "day_of_week", "month", "temperature"]),
        })

        ensemble = pd.DataFrame({
            "p10": np.full(24, 80.0),
            "p50": np.full(24, 100.0),
            "p90": np.full(24, 130.0),
        })

        result = p._calibrate_intervals(ensemble, history, "广东", "load")

        assert (result["p10"] >= 0).all()
        assert (result["p90"] >= 0).all()
        assert (result["p10"] <= result["p50"]).all()
        assert (result["p90"] >= result["p50"]).all()

    def test_price_intervals_can_be_negative(self):
        """price 类型校准后允许负值区间."""
        from scripts.ml.predictor import Predictor
        p = Predictor()
        # history 里也包含负值
        history = make_history(300, target_type="price")
        history["value"] = history["value"] - 120  # 制造负值

        mock_p50 = MagicMock()
        mock_p50.predict.return_value = history["value"].values[-672:] * 0.95
        mock_p10 = MagicMock()
        mock_p10.predict.return_value = history["value"].values[-672:] * 0.80

        p.trainer.load_quantile_models = MagicMock(return_value={
            "p50": (mock_p50, ["hour", "day_of_week", "month", "temperature"]),
            "p10": (mock_p10, ["hour", "day_of_week", "month", "temperature"]),
        })

        ensemble = pd.DataFrame({
            "p10": np.full(24, -50.0),
            "p50": np.full(24, -20.0),
            "p90": np.full(24, -5.0),
        })

        result = p._calibrate_intervals(ensemble, history, "山东", "price")

        # price 类型不钳制负值
        assert (result["p10"] < 0).any()
        assert (result["p50"] < 0).any()


# ============================================================
# 测试 predict (集成)
# ============================================================

class TestPredict:
    def test_predict_returns_dataframe(self):
        from scripts.ml.predictor import Predictor
        from scripts.core.data_source import MemorySource
        from scripts.data.features import FeatureStore

        source = MemorySource()
        store = FeatureStore(source)
        # 使用最近 14 天内的日期，确保 predict() 能查到
        end = datetime.now()
        history = make_history(500)
        # 调整日期到最近
        new_dates = pd.date_range(
            end - timedelta(days=6), periods=500, freq="15min"
        )
        history["dt"] = new_dates

        store.insert_features(history)

        p = Predictor(store=store)

        mock_p50 = MagicMock()
        mock_p50.predict.return_value = np.full(96, 100.0)
        mock_p10 = MagicMock()
        mock_p10.predict.return_value = np.full(96, 80.0)
        mock_p90 = MagicMock()
        mock_p90.predict.return_value = np.full(96, 120.0)

        p.trainer.load_quantile_models = MagicMock(return_value={
            "p10": (mock_p10, ["hour", "day_of_week", "month", "temperature"]),
            "p50": (mock_p50, ["hour", "day_of_week", "month", "temperature"]),
            "p90": (mock_p90, ["hour", "day_of_week", "month", "temperature"]),
        })

        with patch.object(p.fetcher, "fetch_weather", return_value=pd.DataFrame()):
            result = p.predict("广东", "load", horizon_hours=24)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 96
        assert all(c in result.columns for c in ["dt", "p10", "p50", "p90"])

    def test_predict_validates_input(self):
        from scripts.ml.predictor import Predictor
        p = Predictor()

        with pytest.raises(ValueError, match="未知省份"):
            p.predict("INVALID_PROVINCE", "load")

        with pytest.raises(ValueError, match="未知类型"):
            p.predict("广东", "INVALID_TYPE")

    def test_predict_empty_history_raises(self):
        from scripts.ml.predictor import Predictor
        from scripts.core.data_source import MemorySource
        from scripts.data.features import FeatureStore

        source = MemorySource()
        store = FeatureStore(source)
        p = Predictor(store=store)

        with pytest.raises(ValueError, match="没有可用的特征数据"):
            p.predict("广东", "load")

    def test_predict_fallback_to_single_model(self):
        """分位数模型不存在时回退到单模型."""
        from scripts.ml.predictor import Predictor
        from scripts.core.data_source import MemorySource
        from scripts.data.features import FeatureStore

        source = MemorySource()
        store = FeatureStore(source)
        end = datetime.now()
        history = make_history(500)
        new_dates = pd.date_range(
            end - timedelta(days=6), periods=500, freq="15min"
        )
        history["dt"] = new_dates
        store.insert_features(history)

        p = Predictor(store=store)

        p.trainer.load_quantile_models = MagicMock(
            side_effect=FileNotFoundError("no quantile models")
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = np.full(96, 100.0)
        p.trainer.load_model = MagicMock(
            return_value=(mock_model, ["hour", "day_of_week"], "test.lgb")
        )

        with patch.object(p.fetcher, "fetch_weather", return_value=pd.DataFrame()):
            result = p.predict("广东", "load", horizon_hours=24)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 96
