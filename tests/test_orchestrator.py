"""tests/test_orchestrator.py — Orchestrator 关键方法单元测试"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# ============================================================
# Fixtures
# ============================================================

def make_raw_data(n_steps: int = 2000, province: str = "广东",
                  target_type: str = "load") -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n_steps, freq="15min")
    return pd.DataFrame({
        "dt": dates,
        "province": province,
        "type": target_type,
        "value": 100 + 20 * np.sin(np.arange(n_steps) * np.pi / 48) + np.random.normal(0, 5, n_steps),
        "price": np.full(n_steps, 0.35),
    })


def make_features_df(n_steps: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2026-04-01", periods=n_steps, freq="15min")
    return pd.DataFrame({
        "dt": dates,
        "province": "广东",
        "type": "load",
        "value": 100 + 20 * np.sin(np.arange(n_steps) * np.pi / 48),
        "hour": dates.hour,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "temperature": 25 + np.random.normal(0, 3, n_steps),
    })


@pytest.fixture
def orch():
    from scripts.core.data_source import MemorySource
    from scripts.data.features import FeatureStore
    from scripts.orchestrator import Orchestrator

    source = MemorySource()
    orch = Orchestrator.__new__(Orchestrator)
    orch.source = source
    orch.store = FeatureStore(source)
    orch.trainer = MagicMock()
    orch.predictor = MagicMock()
    orch.backtester = MagicMock()
    orch.validator = MagicMock()
    orch.analyzer = MagicMock()
    orch.improver = MagicMock()
    orch.engineer = MagicMock()
    orch._validator_history = []
    return orch


# ============================================================
# TestScanAvailable
# ============================================================

class TestScanAvailable:
    def test_returns_provinces_with_data(self, orch):
        df = make_features_df(200)
        orch.store.insert_features(df)
        available = orch._scan_available()
        assert ("广东", "load") in available

    def test_excludes_insufficient_data(self, orch):
        df = make_features_df(50)  # < 96 rows
        orch.store.insert_features(df)
        available = orch._scan_available()
        assert ("广东", "load") not in available


# ============================================================
# TestValidateData
# ============================================================

class TestValidateData:
    def test_detects_null_rate(self, orch):
        df = make_raw_data(500)
        df.loc[df.sample(50).index, "value"] = np.nan
        orch.source.set_raw(df, "广东", "load")
        result = orch.validate_data("广东", "load")
        assert result["null_rate"] > 0
        assert any("缺失率" in i for i in result.get("issues", []))

    def test_detects_time_gaps(self, orch):
        df = make_raw_data(500)
        # 插入 2 小时间隙
        df.loc[300:, "dt"] = df.loc[300:, "dt"] + timedelta(hours=2)
        orch.source.set_raw(df, "广东", "load")
        result = orch.validate_data("广东", "load")
        assert any("时间间隙" in i for i in result.get("issues", []))

    def test_no_data_returns_error(self, orch):
        result = orch.validate_data("广东", "load")
        assert "error" in result
        assert result["error"] == "no_data"


# ============================================================
# TestExplain
# ============================================================

class TestExplain:
    def test_returns_top_features(self, orch):
        orch.trainer.feature_importance = MagicMock(return_value=[
            {"rank": i + 1, "feature": f"f{i}", "importance": 0.1, "pct": 10.0}
            for i in range(20)
        ])
        orch.trainer.list_versions = MagicMock(return_value=[])
        result = orch.explain("广东", "load")
        assert len(result["feature_importance"]) <= 15
        assert "model_versions" in result

    def test_returns_version_list(self, orch):
        orch.trainer.feature_importance = MagicMock(return_value=[])
        orch.trainer.list_versions = MagicMock(return_value=[
            {"version": 0, "filename": "v1.lgbm"},
            {"version": 1, "filename": "v2.lgbm"},
        ])
        result = orch.explain("广东", "load")
        assert len(result["model_versions"]) == 2


# ============================================================
# TestExport
# ============================================================

class TestExport:
    def test_exports_json(self, orch):
        df = make_features_df(50).head(5)
        orch.store.insert_predictions(df)
        path = os.path.join(tempfile.gettempdir(), "test_export.json")
        try:
            result = orch.export("广东", "load", fmt="json", output=path)
            assert os.path.exists(path)
            assert "已导出" in result
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_exports_csv(self, orch):
        df = make_features_df(50).head(5)
        orch.store.insert_predictions(df)
        path = os.path.join(tempfile.gettempdir(), "test_export.csv")
        try:
            result = orch.export("广东", "load", fmt="csv", output=path)
            assert os.path.exists(path)
            loaded = pd.read_csv(path)
            assert len(loaded) == 5
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_empty_returns_message(self, orch):
        result = orch.export("广东", "load")
        assert result == "无预测数据"


# ============================================================
# TestRollback
# ============================================================

class TestRollback:
    def test_delegates_to_trainer(self, orch):
        orch.trainer.rollback = MagicMock(return_value={
            "status": "rolled_back",
            "removed_version": "old.lgbm",
            "current_version": "new.lgbm",
        })
        result = orch.rollback_model("广东", "load")
        orch.trainer.rollback.assert_called_once_with("广东", "load")
        assert result["status"] == "rolled_back"


# ============================================================
# TestChart
# ============================================================

class TestChart:
    def test_returns_ascii_chart(self, orch):
        samples = [
            {"dt": str(datetime.now() + timedelta(hours=i)), "p50": 100 + i * 5}
            for i in range(24)
        ]
        orch.predictor.predict = MagicMock()
        # 需要 mock 整个 predict 方法
        with patch.object(orch, "predict", return_value={
            "province": "广东", "type": "load",
            "horizon_hours": 24, "n_predictions": 96,
            "sample": samples,
        }):
            result = orch.chart("广东", "load")
        assert "广东" in result
        assert "load" in result
        assert "█" in result or "│" in result or "─" in result

    def test_empty_returns_message(self, orch):
        with patch.object(orch, "predict", return_value={"sample": []}):
            result = orch.chart("广东", "load")
        assert result == "无预测数据"
