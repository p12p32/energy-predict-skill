"""trainer.py — 模型训练器（LightGBM + 分位数回归 + 版本回滚 + 特征重要性）"""
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.core.config import get_model_config

EXCLUDE_COLS = {"dt", "province", "type", "price"}
MAX_VERSIONS = 3
QUANTILES = [0.1, 0.5, 0.9]  # P10, P50, P90


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
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
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

        tscv = TimeSeriesSplit(n_splits=3)
        _, val_idx = list(tscv.split(X))[-1]
        train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

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
                              list(X.columns), model_path)

        self._cleanup_old_versions(province, target_type)

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
            "objective": "regression", "metric": "rmse",
            "num_leaves": 31, "learning_rate": 0.1,
            "verbose": -1, "n_estimators": 100,
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

    def quantile_train(self, df: pd.DataFrame, province: str,
                       target_type: str, target_col: str = "value") -> Dict:
        """训练 P10/P50/P90 三个分位数模型.
        返回: {"models": {"p10": model, "p50": model, "p90": model}, ...}
        """
        X, y = self.prepare_training_data(df, target_col)
        tscv = TimeSeriesSplit(n_splits=3)
        _, val_idx = list(tscv.split(X))[-1]
        train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        province_dir = os.path.join(self.model_dir, province)
        os.makedirs(province_dir, exist_ok=True)

        models = {}
        for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=alpha, metric="quantile",
                boosting_type="gbdt", num_leaves=31, learning_rate=0.05,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                verbose=-1, n_estimators=200, early_stopping_rounds=20,
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="quantile")

            fname = f"{province}_{target_type}_{label}_{timestamp}.lgb"
            path = os.path.join(province_dir, fname)
            with open(path, "wb") as f:
                pickle.dump(m, f)
            models[label] = path

        # 注册表用 P50 作为主模型
        self._update_registry(province, target_type,
                              f"{province}_{target_type}_p50_{timestamp}.lgb",
                              list(X.columns),
                              os.path.join(province_dir, f"{province}_{target_type}_p50_{timestamp}.lgb"))

        return {
            "province": province, "target_type": target_type,
            "paths": models, "n_samples": len(df), "n_features": X.shape[1],
            "feature_names": list(X.columns),
        }

    def load_quantile_models(self, province: str, target_type: str,
                              version: int = -1) -> Dict[str, Tuple]:
        """加载 P10/P50/P90 三个模型."""
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            raise FileNotFoundError(f"未找到模型: {key}")

        entry = registry[key]
        versions = entry.get("versions", [])
        if not versions:
            raise FileNotFoundError(f"未找到模型版本: {key}")

        if version == -1:
            p50_fname = versions[-1]
        else:
            p50_fname = versions[max(0, min(version, len(versions) - 1))]
        p10_fname = p50_fname.replace("p50_", "p10_")
        p90_fname = p50_fname.replace("p50_", "p90_")

        province_dir = os.path.join(self.model_dir, province)
        feature_names = entry["feature_names"]

        models = {}
        for label, fname in [("p10", p10_fname), ("p50", p50_fname), ("p90", p90_fname)]:
            path = os.path.join(province_dir, fname)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models[label] = (pickle.load(f), feature_names)

        return models

    def load_model(self, province: str,
                   target_type: str,
                   version: int = -1) -> Tuple[lgb.LGBMRegressor, List[str], str]:
        """加载模型. version: -1=最新, 0=上一个, 1=上上个..."""
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            raise FileNotFoundError(f"未找到模型: {key}")

        entry = registry[key]
        versions = entry.get("versions", [entry.get("latest", "")])
        if not versions:
            raise FileNotFoundError(f"未找到模型版本: {key}")

        idx = version if version == -1 else min(version, len(versions) - 1)
        model_filename = versions[idx]
        model_path = os.path.join(self.model_dir, province, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model, entry["feature_names"], model_filename

    # ── 版本回滚 ──

    def rollback(self, province: str, target_type: str) -> Dict:
        """回滚到上一个版本."""
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            return {"error": f"未找到模型: {key}"}

        versions = registry[key].get("versions", [])
        if len(versions) < 2:
            return {"error": "只有一个版本，无法回滚"}

        old = versions.pop()
        registry[key]["versions"] = versions

        registry_path = os.path.join(self.model_dir, "model_registry.json")
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        # 删除废弃的模型文件
        old_path = os.path.join(self.model_dir, province, old)
        if os.path.exists(old_path):
            os.remove(old_path)

        return {
            "status": "rolled_back",
            "removed_version": old,
            "current_version": versions[-1],
            "remaining_versions": len(versions),
        }

    def list_versions(self, province: str, target_type: str) -> List[Dict]:
        """列出所有版本."""
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            return []

        entry = registry[key]
        versions = entry.get("versions", [])
        result = []
        for i, v in enumerate(reversed(versions)):
            info = {"version": i, "filename": v}
            vpath = os.path.join(self.model_dir, province, v)
            if os.path.exists(vpath):
                info["size_kb"] = round(os.path.getsize(vpath) / 1024, 1)
            result.append(info)

        return result

    # ── 特征重要性 ──

    def feature_importance(self, province: str,
                           target_type: str) -> List[Dict]:
        """返回 Top 特征重要性.

        Returns: [{rank, feature, importance, pct}, ...]
        """
        model, feature_names, _ = self.load_model(province, target_type)
        importances = model.feature_importances_

        pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True
        )

        total = sum(importances) or 1
        result = []
        for i, (name, imp) in enumerate(pairs):
            result.append({
                "rank": i + 1,
                "feature": name,
                "importance": round(float(imp), 6),
                "pct": round(float(imp / total * 100), 1),
            })

        return result

    # ── 注册表管理 ──

    def _update_registry(self, province: str, target_type: str,
                         filename: str, feature_names: List[str],
                         model_path: str):
        registry = self._read_registry()
        key = f"{province}_{target_type}"

        old_versions = registry.get(key, {}).get("versions", [])

        registry[key] = {
            "versions": (old_versions + [filename])[-MAX_VERSIONS:],
            "feature_names": feature_names,
            "updated_at": datetime.now().isoformat(),
        }
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def _cleanup_old_versions(self, province: str, target_type: str):
        """删除超出 MAX_VERSIONS 的旧模型文件."""
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        versions = registry.get(key, {}).get("versions", [])

        province_dir = os.path.join(self.model_dir, province)
        if not os.path.exists(province_dir):
            return

        for fname in os.listdir(province_dir):
            if fname.startswith(f"{province}_{target_type}") and fname.endswith(".lgb"):
                if fname not in versions:
                    old_path = os.path.join(province_dir, fname)
                    os.remove(old_path)

    def _read_registry(self) -> Dict:
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                return json.load(f)
        return {}
