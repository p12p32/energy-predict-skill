"""tests/test_validator.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestValidator:
    def setup_method(self):
        try:
            from src.evolve.validator import Validator
            self.validator = Validator()
        except (FileNotFoundError, KeyError) as e:
            pytest.skip(f"Config not available: {e}")

    def test_compute_metrics(self):
        dates = pd.date_range("2025-01-01", periods=96, freq="15min")
        actual = 100 + np.sin(np.arange(96) * 0.05) * 50
        preds = pd.DataFrame({
            "dt": dates, "province": ["广东"] * 96, "type": ["load"] * 96,
            "p50": actual + np.random.normal(0, 5, 96),
        })
        actuals = pd.DataFrame({
            "dt": dates, "province": ["广东"] * 96, "type": ["load"] * 96,
            "value": actual,
        })
        metrics = self.validator.compute_metrics(preds, actuals)
        assert "mape" in metrics
        assert "rmse" in metrics
        assert metrics["mape"] < 0.15

    def test_should_trigger_high_mape(self):
        metrics = {"mape": 0.12, "rmse": 10.0, "bias_direction": "high"}
        assert self.validator.should_trigger(metrics)

    def test_should_not_trigger_low_mape(self):
        metrics = {"mape": 0.03, "rmse": 5.0, "bias_direction": "ok"}
        assert not self.validator.should_trigger(metrics)

    def test_should_trigger_consecutive_bias(self):
        metrics = {"mape": 0.04, "rmse": 5.0, "bias_direction": "high"}
        history = [
            {"bias_direction": "high"},
            {"bias_direction": "high"},
            {"bias_direction": "high"},  # need 3 consecutive
        ]
        assert self.validator.should_trigger(metrics, history)

    def test_validate_returns_report(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="h")
        preds = pd.DataFrame({"dt": dates, "province": "广东", "type": "load", "p50": range(10)})
        actuals = pd.DataFrame({"dt": dates, "province": "广东", "type": "load", "value": range(10)})
        report = self.validator.validate(preds, actuals)
        assert "triggered" in report
        assert "metrics" in report
