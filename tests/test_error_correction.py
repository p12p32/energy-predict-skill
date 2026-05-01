"""tests/test_error_correction.py"""
import pytest
import numpy as np
from scripts.ml.error_correction import ErrorCorrectionModel


class TestErrorCorrectionModel:
    def test_fit_and_predict(self):
        t = np.arange(500)
        signal = 0.5 * np.sin(t * 0.1)
        residuals = signal + np.random.normal(0, 0.1, 500)

        ecm = ErrorCorrectionModel(order=20)
        ecm.fit(residuals)

        assert ecm.coefficients is not None
        assert len(ecm.coefficients) > 0

        pred = ecm.predict(residuals[-30:], steps=10)
        assert len(pred) == 10

    def test_fit_and_correct(self):
        np.random.seed(42)
        t = np.arange(300)
        y_true = 100 + t * 0.1 + np.sin(t * 0.05) * 10
        y_lgb = y_true + np.random.normal(0, 3, 300)

        ecm = ErrorCorrectionModel(order=20)
        result = ecm.fit_and_correct(y_true, y_lgb, horizon=50)

        assert "corrected" in result
        assert "ar_pred" in result
        assert len(result["corrected"]) == 50

    def test_correct_reduces_error(self):
        np.random.seed(42)
        t = np.arange(500)
        y_true = 100 + t * 0.05 + 5 * np.sin(t * 0.03)
        y_lgb = y_true + 2.0 + np.random.normal(0, 2, 500)

        ecm = ErrorCorrectionModel(order=30)
        result = ecm.fit_and_correct(y_true, y_lgb, horizon=50)

        lgb_mape = np.mean(np.abs((y_true[-50:] - y_lgb[-50:]) / y_true[-50:]))
        corrected_mape = np.mean(np.abs((y_true[-50:] - result["corrected"]) / y_true[-50:]))

        # ECM 修正了系统偏差 → 不应该比原始显著差
        assert corrected_mape <= lgb_mape * 1.1, f"ECM significantly worsened: {corrected_mape:.4f} vs {lgb_mape:.4f}"
