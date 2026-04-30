"""predictor.py — 预测执行器"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

from src.trainer import Trainer
from src.feature_store import FeatureStore
from src.db import DorisDB
from src.config_loader import load_config


class Predictor:
    def __init__(self, trainer: Trainer = None,
                 store: FeatureStore = None):
        self.trainer = trainer or Trainer()
        self.store = store or FeatureStore(DorisDB())

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24,
                model_version: str = None) -> pd.DataFrame:
        model, feature_names = self.trainer.load_model(province, target_type)

        cfg = load_config()
        lookback_days = 14
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        features = self.store.load_features(
            province, target_type,
            start_date.strftime("%Y-%m-%d"),
            (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if features.empty:
            raise ValueError(
                f"没有可用的特征数据: {province}/{target_type}"
            )

        horizon_steps = min(horizon_hours * 4, 96)

        recent_features = features.tail(horizon_steps).copy()
        if len(recent_features) < horizon_steps:
            recent_features = features.tail(horizon_steps)

        predictions = self._predict_with_model(
            model, recent_features, feature_names,
            province=province, target_type=target_type,
        )

        self.store.insert_predictions(predictions)

        return predictions

    def _predict_with_model(self, model, features_df: pd.DataFrame,
                            feature_names: List[str], province: str,
                            target_type: str) -> pd.DataFrame:
        predict_features = features_df[feature_names].copy()
        predicted = model.predict(predict_features)

        result = pd.DataFrame({
            "dt": features_df["dt"].values[:len(predicted)],
            "province": province,
            "type": target_type,
            "p50": predicted,
            "p10": predicted * 0.97,
            "p90": predicted * 1.03,
            "model_version": "v1",
        })
        return result
