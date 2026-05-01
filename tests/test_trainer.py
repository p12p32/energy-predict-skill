"""tests/test_trainer.py"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta


class TestTrainer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        from src.ml.trainer import Trainer
        self.trainer = Trainer(model_dir=self.tmpdir)

        dates = pd.date_range("2025-01-01", periods=2000, freq="15min")
        base = 100 + np.sin(np.arange(2000) * 0.005) * 30 + np.random.normal(0, 3, 2000)
        months = dates.month
        season_map = [1 if m in [3,4,5] else 2 if m in [6,7,8] else 3 if m in [9,10,11] else 4 for m in months]
        self.train_df = pd.DataFrame({
            "dt": dates, "province": "广东", "type": "load",
            "hour": dates.hour, "day_of_week": dates.dayofweek,
            "month": months, "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
            "season": season_map,
            "value_lag_1d": np.roll(base, 96), "value_lag_7d": np.roll(base, 672),
            "value_rolling_mean_24h": pd.Series(base).rolling(96, min_periods=1).mean(),
            "value_diff_1d": np.diff(base, prepend=base[0]),
            "value": base,
        }).dropna()

    def test_train_and_save(self):
        result = self.trainer.train(self.train_df, "广东", "load")
        assert result["n_samples"] == len(self.train_df)
        assert os.path.exists(result["model_path"])

    def test_quick_train_returns_model(self):
        result = self.trainer.quick_train(self.train_df.head(500), "广东", "load")
        assert "model" in result
        assert "feature_names" in result

    def test_quantile_train(self):
        result = self.trainer.quantile_train(self.train_df.head(1000), "广东", "load")
        assert "paths" in result
        assert "p50" in result["paths"]
        assert os.path.exists(result["paths"]["p50"])

    def test_load_model(self):
        self.trainer.train(self.train_df, "广东", "load")
        model, features, fname = self.trainer.load_model("广东", "load")
        assert model is not None
        assert len(features) > 0

    def test_feature_importance(self):
        self.trainer.train(self.train_df, "广东", "load")
        importance = self.trainer.feature_importance("广东", "load")
        assert len(importance) > 0
        assert "feature" in importance[0]
        assert "pct" in importance[0]

    def test_version_rollback(self):
        self.trainer.train(self.train_df, "广东", "load")
        self.trainer.train(self.train_df, "广东", "load")
        versions = self.trainer.list_versions("广东", "load")
        assert len(versions) >= 2

        result = self.trainer.rollback("广东", "load")
        assert result["status"] == "rolled_back"

    def test_max_versions_cleanup(self):
        for _ in range(5):
            self.trainer.train(self.train_df, "广东", "load")
        versions = self.trainer.list_versions("广东", "load")
        assert len(versions) <= 3
