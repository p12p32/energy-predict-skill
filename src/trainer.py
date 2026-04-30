"""trainer.py — 模型训练器（LightGBM 短期预测）"""
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src.config_loader import get_model_config

EXCLUDE_COLS = {"dt", "province", "type", "price"}


class Trainer:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or get_model_config()["storage_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_training_data(self, df: pd.DataFrame,
                               target_col: str = "value") -> Tuple[pd.DataFrame, pd.Series]:
        df = df.dropna(subset=[target_col]).copy()

        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS and c != target_col
        ]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        return X, y

    def train(self, df: pd.DataFrame, province: str,
              target_type: str, target_col: str = "value",
              params: Dict = None, model_filename: str = None) -> Dict:
        X, y = self.prepare_training_data(df, target_col)

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 200,
            "early_stopping_rounds": 20,
        }
        if params:
            lgb_params.update(params)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
        )

        if model_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{province}_{target_type}_{timestamp}.lgb"

        province_dir = os.path.join(self.model_dir, province)
        os.makedirs(province_dir, exist_ok=True)
        model_path = os.path.join(province_dir, model_filename)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        self._update_registry(province, target_type, model_filename,
                              list(X.columns))

        return {
            "province": province,
            "target_type": target_type,
            "model_path": model_path,
            "n_samples": len(df),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
        }

    def quick_train(self, df: pd.DataFrame, province: str,
                    target_type: str, target_col: str = "value",
                    params: Dict = None) -> Dict:
        X, y = self.prepare_training_data(df, target_col)

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "verbose": -1,
            "n_estimators": 100,
        }
        if params:
            lgb_params.update(params)

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, y)

        return {
            "model": model,
            "feature_names": list(X.columns),
            "n_samples": len(df),
            "province": province,
            "target_type": target_type,
        }

    def load_model(self, province: str,
                   target_type: str) -> Tuple[lgb.LGBMRegressor, List[str]]:
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            raise FileNotFoundError(f"未找到模型: {key}")

        model_filename = registry[key]["latest"]
        model_path = os.path.join(self.model_dir, province, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model, registry[key]["feature_names"]

    def _update_registry(self, province: str, target_type: str,
                         filename: str, feature_names: List[str]):
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        registry[key] = {
            "latest": filename,
            "feature_names": feature_names,
            "updated_at": datetime.now().isoformat(),
        }
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def _read_registry(self) -> Dict:
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                return json.load(f)
        return {}
