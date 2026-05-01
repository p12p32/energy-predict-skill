"""tests/test_improver.py — 自优化引擎单元测试"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock


# ============================================================
# 合成数据
# ============================================================

def make_experiment_data(n_steps: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2026-04-01", periods=n_steps, freq="15min")
    hour = dates.hour.values.astype(float)
    dow = dates.dayofweek.values.astype(float)
    month = dates.month.values.astype(float)

    value = (
        100
        + 30 * np.sin((hour - 6) * np.pi / 12)
        + 15 * np.sin((hour - 14) * np.pi / 8)
        + np.random.normal(0, 3, n_steps)
    )
    value = np.maximum(value, 10)

    return pd.DataFrame({
        "dt": dates, "province": "广东", "type": "load",
        "value": value,
        "hour": hour.astype(int),
        "day_of_week": dow.astype(int),
        "day_of_month": dates.day.values,
        "month": month.astype(int),
        "is_weekend": (dow >= 5).astype(int),
        "season": np.full(n_steps, 2),
        "temperature": 25 + np.random.normal(0, 3, n_steps),
        "humidity": 60 + np.random.normal(0, 5, n_steps),
        "wind_speed": 3 + np.random.normal(0, 1, n_steps),
        "solar_radiation": np.maximum(0, 300 * np.sin(hour * np.pi / 12)),
        "precipitation": np.zeros(n_steps),
        "pressure": np.full(n_steps, 1013.0),
        "wind_direction": np.full(n_steps, 180.0),
        "is_holiday": np.zeros(n_steps, dtype=bool),
        "value_lag_1d": value,
        "value_lag_7d": value,
        "value_rolling_mean_24h": value,
        "value_diff_1d": np.zeros(n_steps),
        "value_diff_7d": np.zeros(n_steps),
        "quality_flag": np.zeros(n_steps, dtype=int),
        "peak_valley": np.zeros(n_steps),
        "weekend_hour": np.zeros(n_steps),
        "dow_hour": np.zeros(n_steps),
        "weekend_x_lag7d": np.zeros(n_steps),
        "hour_x_lag1d": np.zeros(n_steps),
    })


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def improver_with_mocks():
    from scripts.evolve.improver import Improver
    from scripts.core.data_source import MemorySource

    source = MemorySource()
    imp = Improver(data_source=source)

    # Mock trainer.quick_train
    mock_model = MagicMock()
    mock_model.predict.return_value = np.full(96, 100.0)

    imp.trainer.quick_train = MagicMock(return_value={
        "model": mock_model,
        "feature_names": ["hour", "day_of_week", "temperature"],
        "n_samples": 500,
        "province": "广东",
        "target_type": "load",
    })

    imp.trainer.train = MagicMock(return_value={
        "province": "广东", "target_type": "load",
        "model_path": "models/test.lgb",
        "n_samples": 500, "n_features": 5,
    })

    return imp


@pytest.fixture
def summer_diagnosis():
    return [
        {
            "scenario": "summer",
            "root_cause": "nonlinear_heat_effect",
            "description": "夏季高温导致负荷非线性增长",
            "severity": "high",
            "suggested_features": ["temperature²", "temperature×is_weekend"],
        },
    ]


@pytest.fixture
def overall_drift_diagnosis():
    return [
        {
            "scenario": "overall",
            "root_cause": "distribution_shift",
            "description": "整体预测大幅度退化",
            "severity": "critical",
            "suggested_features": ["shorter_training_window", "full_retrain"],
        },
    ]


# ============================================================
# 测试 generate_hypotheses
# ============================================================

class TestGenerateHypotheses:
    def test_returns_hypotheses_for_diagnosis(self, improver_with_mocks, summer_diagnosis):
        imp = improver_with_mocks
        hypotheses = imp.generate_hypotheses(summer_diagnosis)

        assert len(hypotheses) > 0
        assert all("name" in h for h in hypotheses)
        assert all("params" in h for h in hypotheses)
        assert all("knowledge_score" in h for h in hypotheses)

    def test_critical_returns_all_strategies(self, improver_with_mocks, overall_drift_diagnosis):
        imp = improver_with_mocks
        hypotheses = imp.generate_hypotheses(overall_drift_diagnosis)

        assert len(hypotheses) > 0

    def test_empty_diagnosis_returns_empty(self, improver_with_mocks):
        imp = improver_with_mocks
        hypotheses = imp.generate_hypotheses([])

        assert hypotheses == []

    def test_no_duplicate_strategies(self, improver_with_mocks, summer_diagnosis):
        imp = improver_with_mocks
        hypotheses = imp.generate_hypotheses(summer_diagnosis)

        names = [h["name"] for h in hypotheses]
        assert len(names) == len(set(names))

    def test_retired_strategy_excluded(self, improver_with_mocks, summer_diagnosis):
        imp = improver_with_mocks
        import json
        # generate_hypotheses 用 name + json.dumps(params) 做知识库键
        sig = "polynomial_features" + json.dumps({"power": 2}, sort_keys=True)
        imp.source.save_knowledge({
            "strategy_hash": sig,
            "strategy_desc": "polynomial_features",
            "applied_count": 5,
            "success_count": 0,
            "avg_improvement": -0.03,
            "retired": True,
        })

        hypotheses = imp.generate_hypotheses(summer_diagnosis)
        names = [h["name"] for h in hypotheses]

        assert "polynomial_features" not in names

    def test_high_failure_count_excluded(self, improver_with_mocks, summer_diagnosis):
        imp = improver_with_mocks
        # 连续5次失败 → 应被跳过
        imp.source.save_knowledge({
            "strategy_hash": imp._hash("shorter_window"),
            "strategy_desc": "shorter_window",
            "applied_count": 5,
            "success_count": 0,
            "avg_improvement": -0.02,
            "retired": False,
        })

        hypotheses = imp.generate_hypotheses(summer_diagnosis)
        names = [h["name"] for h in hypotheses]

        assert "shorter_window" not in names


# ============================================================
# 测试 _calc_knowledge_score
# ============================================================

class TestKnowledgeScore:
    def test_new_strategy_neutral_score(self, improver_with_mocks):
        imp = improver_with_mocks
        score = imp._calc_knowledge_score({})
        assert score == 0.5

    def test_successful_strategy_high_score(self, improver_with_mocks):
        imp = improver_with_mocks
        k_info = {
            "applied_count": 10,
            "success_count": 8,
            "avg_improvement": 0.05,
            "retired": False,
        }
        score = imp._calc_knowledge_score(k_info)
        assert score > 0.6  # 应高于基准

    def test_failing_strategy_low_score(self, improver_with_mocks):
        imp = improver_with_mocks
        k_info = {
            "applied_count": 5,
            "success_count": 1,
            "avg_improvement": -0.02,
            "retired": False,
        }
        score = imp._calc_knowledge_score(k_info)
        assert score < 0.5  # 应低于新策略


# ============================================================
# 测试 run_arena
# ============================================================

class TestRunArena:
    def test_returns_sorted_results(self, improver_with_mocks, summer_diagnosis):
        imp = improver_with_mocks
        df = make_experiment_data(500)
        hypotheses = imp.generate_hypotheses(summer_diagnosis)

        results = imp.run_arena(hypotheses, df, "广东", "load")

        assert len(results) > 0
        # 按 MAPE 升序排列
        mapes = [r.get("mape", float("inf")) for r in results]
        assert mapes == sorted(mapes)

    def test_error_results_included(self, improver_with_mocks, summer_diagnosis):
        imp = improver_with_mocks
        df = make_experiment_data(200)

        # 强制 trainer 失败
        imp.trainer.quick_train = MagicMock(side_effect=RuntimeError("训练失败"))

        hypotheses = imp.generate_hypotheses(summer_diagnosis)
        results = imp.run_arena(hypotheses, df, "广东", "load")

        assert all("error" in r for r in results)


# ============================================================
# 测试 select_best
# ============================================================

class TestSelectBest:
    def test_selects_lowest_mape(self, improver_with_mocks):
        imp = improver_with_mocks

        arena = [
            {"hypothesis_id": "a", "hypothesis": {"name": "a"}, "mape": 0.10},
            {"hypothesis_id": "b", "hypothesis": {"name": "b"}, "mape": 0.05},
            {"hypothesis_id": "c", "hypothesis": {"name": "c"}, "mape": 0.15},
        ]
        baseline = {"overall_mape": 0.12}
        diagnosis = [{"scenario": "overall", "severity": "critical"}]

        best = imp.select_best(arena, diagnosis, baseline)

        assert best["selected"]["name"] == "b"
        assert best["improvement"] > 0

    def test_all_errors_returns_error(self, improver_with_mocks):
        imp = improver_with_mocks

        arena = [
            {"hypothesis": {"name": "a"}, "error": "failed"},
            {"hypothesis": {"name": "b"}, "error": "failed"},
        ]
        baseline = {"overall_mape": 0.12}

        best = imp.select_best(arena, [], baseline)
        assert "error" in best

    def test_empty_arena_returns_error(self, improver_with_mocks):
        imp = improver_with_mocks
        best = imp.select_best([], [], {"overall_mape": 0.10})
        assert "error" in best


# ============================================================
# 测试 record_strategy
# ============================================================

class TestRecordStrategy:
    def test_new_strategy_recorded(self, improver_with_mocks):
        imp = improver_with_mocks
        strategy = {
            "name": "test_strategy",
            "desc": "test description",
            "improvement": {"before": 0.10, "after": 0.07},
        }

        imp.record_strategy(strategy, scenario="summer", success=True)

        kb = imp.source.load_knowledge()
        assert not kb.empty

    def test_existing_strategy_updated(self, improver_with_mocks):
        imp = improver_with_mocks

        # 先创建
        imp.source.save_knowledge({
            "strategy_hash": imp._hash("update_test"),
            "strategy_desc": "update test",
            "applied_count": 3,
            "success_count": 2,
            "avg_improvement": 0.02,
            "retired": False,
        })

        strategy = {
            "name": "update_test",
            "improvement": {"before": 0.12, "after": 0.08},
        }
        imp.record_strategy(strategy, success=True)

        kb = imp.source.load_knowledge()
        row = kb[kb["strategy_hash"] == imp._hash("update_test")]
        assert len(row) > 0


# ============================================================
# 测试 improve 完整流程
# ============================================================

class TestImprove:
    def test_full_cycle(self, improver_with_mocks, summer_diagnosis):
        imp = improver_with_mocks
        df = make_experiment_data(500)
        baseline = {"overall_mape": 0.12}

        result = imp.improve(summer_diagnosis, df, "广东", "load",
                            baseline=baseline)

        assert "selected_strategy" in result
        assert "mape_before" in result
        assert "mape_after" in result
        assert result["knowledge_updated"] is True

    def test_no_diagnosis_returns_empty(self, improver_with_mocks):
        imp = improver_with_mocks
        df = make_experiment_data(100)
        baseline = {"overall_mape": 0.05}

        result = imp.improve([], df, "广东", "load", baseline=baseline)

        assert result["selected_strategy"] == ""
        assert result["improvement"] == 0
