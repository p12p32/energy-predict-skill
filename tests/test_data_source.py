"""tests/test_data_source.py"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta


class TestFileSource:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        from scripts.core.data_source import FileSource
        self.source = FileSource(base_dir=self.tmpdir)

        dates = pd.date_range("2025-01-01", periods=200, freq="15min")
        base = 100 + np.arange(200) * 0.5 + np.random.normal(0, 3, 200)
        self.test_df = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 200,
            "type": ["load"] * 200,
            "value": base,
            "price": [0.35] * 200,
        })

    def test_import_and_load_raw(self):
        csv_path = os.path.join(self.tmpdir, "test.csv")
        self.test_df.to_csv(csv_path, index=False)

        self.source.import_csv(csv_path)
        result = self.source.load_raw("广东", "load", "2025-01-01", "2025-01-02")
        assert len(result) > 0
        assert "value" in result.columns

    def test_save_and_load_features(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="h")
        features = pd.DataFrame({
            "dt": dates, "province": ["广东"] * 4, "type": ["load"] * 4,
            "value": [100, 102, 105, 103], "price": [0.35] * 4,
            "hour": [0, 1, 2, 3], "day_of_week": [2] * 4,
        })
        count = self.source.save_features(features)
        assert count == 4

        loaded = self.source.load_features("广东", "load", "2025-01-01", "2025-01-02")
        assert len(loaded) == 4

    def test_save_and_load_predictions(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="h")
        preds = pd.DataFrame({
            "dt": dates, "province": ["广东"] * 4, "type": ["load"] * 4,
            "p10": [95, 97, 100, 98], "p50": [100, 102, 105, 103],
            "p90": [105, 107, 110, 108],
        })
        self.source.save_predictions(preds)
        loaded = self.source.load_predictions("广东", "load", "2025-01-01", "2025-01-02")
        assert len(loaded) == 4

    def test_knowledge_crud(self):
        self.source.save_knowledge({"strategy_hash": "abc123", "name": "test", "applied_count": 5})
        df = self.source.load_knowledge()
        assert not df.empty


class TestMemorySource:
    def setup_method(self):
        from scripts.core.data_source import MemorySource
        self.source = MemorySource()

    def test_set_and_load_raw(self):
        dates = pd.date_range("2025-01-01", periods=100, freq="15min")
        df = pd.DataFrame({
            "dt": dates, "province": ["广东"] * 100, "type": ["load"] * 100,
            "value": range(100), "price": [0.35] * 100,
        })
        self.source.set_raw(df, "广东", "load")
        result = self.source.load_raw("广东", "load", "2025-01-01", "2025-01-02")
        assert not result.empty
        assert len(result) <= 100
