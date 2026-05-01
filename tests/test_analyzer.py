"""tests/test_analyzer.py"""
import pytest
from scripts.evolve.analyzer import Analyzer


class TestAnalyzer:
    def test_diagnose_summer_high_mape(self):
        analyzer = Analyzer()
        report = {"by_season": {"summer": {"mape": 0.18, "samples": 2000}}}
        diag = analyzer.diagnose(report)
        assert any("summer" in d.get("scenario", "") for d in diag)

    def test_diagnose_weekend_pattern(self):
        analyzer = Analyzer()
        report = {"by_time_type": {"weekend": {"mape": 0.15}}}
        diag = analyzer.diagnose(report)
        assert any("weekend" in d.get("scenario", "") for d in diag)

    def test_diagnose_overall_drift(self):
        analyzer = Analyzer()
        report = {"overall_mape": 0.20}
        diag = analyzer.diagnose(report)
        assert any("overall" in d.get("scenario", "") for d in diag)

    def test_diagnose_persistent_bias(self):
        analyzer = Analyzer()
        history = [
            {"bias_direction": "high"},
            {"bias_direction": "high"},
            {"bias_direction": "high"},
        ]
        diag = analyzer.diagnose({}, validator_history=history)
        assert any("persistent_bias" in d.get("scenario", "") for d in diag)
