"""trainer.py — 模型训练器（LightGBM + XGBoost + 分位数回归 + 版本回滚 + 特征重要性）

模型存储格式:
  - .lgbm       → LightGBM 原生文本格式 (booster_.save_model)
  - .lgbm.meta   → JSON 元数据 (feature_names 等)
  - .lgb          → 旧版 pickle 格式 (向后兼容，仅读取)
  - .xgb.json    → XGBoost 原生 JSON 格式 (model.save_model)
"""
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from scripts.core.config import get_model_config, load_config

logger = logging.getLogger(__name__)

EXCLUDE_COLS = {"dt", "province", "type", "price"}
MAX_VERSIONS = 3
QUANTILES = [0.1, 0.5, 0.9]  # P10, P50, P90


class Trainer:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or get_model_config()["storage_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    # ── 安全模型 I/O (LightGBM 原生文本, 不用 pickle) ──

    @staticmethod
    def _save_model_native(model: lgb.LGBMRegressor, path: str):
        """以 LightGBM 原生文本格式保存模型."""
        model.booster_.save_model(path)

    @staticmethod
    def _save_meta(path: str, feature_names: List[str]):
        """保存模型元数据到 JSON."""
        meta_path = path + ".meta"
        with open(meta_path, "w") as f:
            json.dump({"feature_names": feature_names}, f)

    @staticmethod
    def _load_meta(path: str) -> List[str]:
        """从 .meta JSON 加载特征名."""
        meta_path = path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f).get("feature_names", [])
        return []

    @staticmethod
    def _load_model_native(path: str) -> lgb.LGBMRegressor:
        """从 LightGBM 原生文本文件加载模型."""
        model = lgb.LGBMRegressor()
        booster = lgb.Booster(model_file=path)
        n_features = booster.num_feature()
        model._Booster = booster
        model._n_features = n_features
        model.n_features_in_ = n_features
        model.fitted_ = True
        return model

    @staticmethod
    def _load_model_file(path: str) -> Tuple[lgb.LGBMRegressor, List[str]]:
        """加载单个模型文件 (优先原生格式, 回退 pickle)."""
        # 优先尝试 .lgbm 原生格式
        lgbm_path = path if path.endswith(".lgbm") else path.replace(".lgb", ".lgbm")
        if os.path.exists(lgbm_path):
            model = Trainer._load_model_native(lgbm_path)
            feature_names = Trainer._load_meta(lgbm_path)
            if feature_names:
                return model, feature_names

        # 回退: .lgb pickle 格式 (向后兼容)
        if os.path.exists(path) and path.endswith(".lgb"):
            with open(path, "rb") as f:
                model = pickle.load(f)
            # pickle 格式没有独立元数据, 需要外部提供 feature_names
            return model, []

        # 尝试直接用给定路径
        if os.path.exists(path):
            try:
                return Trainer._load_model_native(path), Trainer._load_meta(path)
            except Exception as e:
                logger.warning("原生格式加载失败，回退 pickle: %s", e)
                with open(path, "rb") as f:
                    return pickle.load(f), []

        raise FileNotFoundError(f"模型文件不存在: {path} or {lgbm_path}")

    def prepare_training_data(self, df: pd.DataFrame,
                               target_col: str = "value",
                               target_type: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据，自动过滤非实际值.

        光伏类型: 目标自动切换为 value - value_lag_1d (残差建模),
        让天气特征直接解释日间变化量, 而非与自回归特征竞争.
        """
        df = df.dropna(subset=[target_col]).copy()

        # value_type 过滤: 只用实际值训练
        if "value_type" in df.columns:
            vt_col = df["value_type"]
            actual_mask = vt_col.isna() | (vt_col == "实际")
            if not actual_mask.all():
                skipped = (~actual_mask).sum()
                logger.info("训练过滤: 排除 %d 条非实际值 (预测/其他)", skipped)
                df = df[actual_mask]

        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS and c != target_col
        ]
        X = df[feature_cols].select_dtypes(include=[np.number])

        # 光伏残差建模: 训练目标 = 日间变化量
        if target_type and "光伏" in target_type and "value_lag_1d" in df.columns:
            lag1d = df["value_lag_1d"].fillna(0)
            y = df[target_col] - lag1d
            logger.info("光伏残差建模: target = value - value_lag_1d (均值=%.2f, std=%.2f)",
                        y.mean(), y.std())
        else:
            y = df[target_col]
        return X, y

    def train(self, df: pd.DataFrame, province: str,
              target_type: str, target_col: str = "value",
              params: Dict = None, model_filename: str = None) -> Dict:
        X, y = self.prepare_training_data(df, target_col, target_type)

        lgb_params = {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
            "min_child_samples": 20,
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
            model_filename = f"{province}_{target_type}_{timestamp}.lgbm"

        province_dir = os.path.join(self.model_dir, province)
        os.makedirs(province_dir, exist_ok=True)
        model_path = os.path.join(province_dir, model_filename)

        self._save_model_native(model, model_path)
        self._save_meta(model_path, list(X.columns))

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
        X, y = self.prepare_training_data(df, target_col, target_type)
        cfg = load_config()
        tc = cfg.get("trainer", {})
        lgb_params = {
            "objective": "quantile", "alpha": 0.5, "metric": "quantile",
            "num_leaves": tc.get("quick_num_leaves", 63),
            "learning_rate": tc.get("quick_learning_rate", 0.03),
            "verbose": -1,
            "n_estimators": tc.get("quick_n_estimators", 500),
            "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
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

    def xgboost_train(self, df: pd.DataFrame, province: str,
                      target_type: str, target_col: str = "value",
                      params: Dict = None) -> Dict:
        """训练 XGBoost 分位数回归模型 (P50)."""
        X, y = self.prepare_training_data(df, target_col, target_type)
        cfg = load_config()
        tc = cfg.get("trainer", {})
        xgb_params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.5,
            "max_depth": tc.get("xgb_max_depth", 7),
            "learning_rate": tc.get("xgb_learning_rate", 0.03),
            "n_estimators": tc.get("xgb_n_estimators", 500),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": 0,
        }
        if params:
            xgb_params.update(params)
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X, y)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{province}_{target_type}_xgb_{timestamp}.json"
        province_dir = os.path.join(self.model_dir, province)
        os.makedirs(province_dir, exist_ok=True)
        model_path = os.path.join(province_dir, model_filename)
        model.save_model(model_path)

        self._update_registry(province, target_type,
                              model_filename, list(X.columns), model_path,
                              model_type="xgb")

        return {
            "model": model,
            "feature_names": list(X.columns),
            "n_samples": len(df), "province": province,
            "target_type": target_type, "model_path": model_path,
        }

    def load_xgboost_model(self, province: str, target_type: str,
                           version: int = -1) -> Tuple[xgb.XGBRegressor, List[str]]:
        """加载 XGBoost 模型."""
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            raise FileNotFoundError(f"未找到模型: {key}")

        entry = registry[key]
        xgb_versions = entry.get("xgb_versions", [])
        if not xgb_versions:
            raise FileNotFoundError(f"未找到 XGBoost 模型: {key}")

        idx = version if version == -1 else min(version, len(xgb_versions) - 1)
        model_filename = xgb_versions[idx]
        model_path = os.path.join(self.model_dir, province, model_filename)

        model = xgb.XGBRegressor()
        model.load_model(model_path)
        feature_names = entry.get("feature_names", [])

        return model, feature_names

    def quantile_train(self, df: pd.DataFrame, province: str,
                       target_type: str, target_col: str = "value") -> Dict:
        """训练 P10/P50/P90 三个分位数模型.
        返回: {"models": {"p10": model, "p50": model, "p90": model}, ...}
        """
        X, y = self.prepare_training_data(df, target_col, target_type)
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
                boosting_type="gbdt", num_leaves=31, learning_rate=0.03,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                verbose=-1, n_estimators=500, early_stopping_rounds=50,
                min_child_samples=20,
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="quantile")

            fname = f"{province}_{target_type}_{label}_{timestamp}.lgbm"
            path = os.path.join(province_dir, fname)
            self._save_model_native(m, path)
            self._save_meta(path, list(X.columns))
            models[label] = path

        # 注册表用 P50 作为主模型
        self._update_registry(province, target_type,
                              f"{province}_{target_type}_p50_{timestamp}.lgbm",
                              list(X.columns),
                              os.path.join(province_dir, f"{province}_{target_type}_p50_{timestamp}.lgbm"))

        return {
            "province": province, "target_type": target_type,
            "paths": models, "n_samples": len(df), "n_features": X.shape[1],
            "feature_names": list(X.columns),
        }

    def load_quantile_models(self, province: str, target_type: str,
                              version: int = -1) -> Dict[str, Tuple]:
        """加载 P10/P50/P90 三个模型.

        向后兼容: 优先查找 .lgbm 格式, 回退 .lgb 格式.
        """
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
        # 仅当文件名包含 "p50_" 时说明是 quantile_train 训练的, 才有独立 P10/P90 文件
        if "p50_" in p50_fname:
            p10_fname = p50_fname.replace("p50_", "p10_")
            p90_fname = p50_fname.replace("p50_", "p90_")
        else:
            # 单模型 (train()) → 无独立 P10/P90, 设空让下游回退
            p10_fname = "__nonexistent__"
            p90_fname = "__nonexistent__"

        province_dir = os.path.join(self.model_dir, province)
        feature_names = entry["feature_names"]

        models = {}
        for label, fname in [("p10", p10_fname), ("p50", p50_fname), ("p90", p90_fname)]:
            path = os.path.join(province_dir, fname)
            try:
                m, fn = self._load_model_file(path)
                models[label] = (m, fn if fn else feature_names)
            except FileNotFoundError:
                logger.warning("分位数模型文件缺失: %s", path)

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

        models, feature_names = self._load_model_file(model_path)
        if not feature_names:
            feature_names = entry.get("feature_names", [])

        return models, feature_names, model_filename

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

        # 删除废弃的模型文件 (含 .lgbm + .meta 和 .lgb)
        for ext in [".lgbm", ".lgb"]:
            old_path = os.path.join(self.model_dir, province,
                                     old.replace(".lgbm", ext).replace(".lgb", ext))
            if os.path.exists(old_path):
                os.remove(old_path)
            meta_path = old_path + ".meta"
            if os.path.exists(meta_path):
                os.remove(meta_path)

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
            else:
                # 尝试 .lgb 回退
                alt = v.replace(".lgbm", ".lgb")
                alt_path = os.path.join(self.model_dir, province, alt)
                if os.path.exists(alt_path):
                    info["size_kb"] = round(os.path.getsize(alt_path) / 1024, 1)
            result.append(info)

        return result

    # ── 特征重要性 ──

    def feature_importance(self, province: str,
                           target_type: str) -> List[Dict]:
        """返回 Top 特征重要性.

        Returns: [{rank, feature, importance, pct}, ...]
        """
        model, feature_names, _ = self.load_model(province, target_type)
        # 兼容: 原生格式从 booster 取, pickle 格式从 feature_importances_ 取
        booster = getattr(model, "_Booster", None)
        if booster is not None:
            importances = booster.feature_importance(importance_type="gain")
        else:
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
                         model_path: str, model_type: str = "lgb"):
        registry = self._read_registry()
        key = f"{province}_{target_type}"

        entry = registry.get(key, {})
        if model_type == "xgb":
            old_xgb = entry.get("xgb_versions", [])
            entry["xgb_versions"] = (old_xgb + [filename])[-MAX_VERSIONS:]
        else:
            old_versions = entry.get("versions", [])
            entry["versions"] = (old_versions + [filename])[-MAX_VERSIONS:]

        entry["feature_names"] = feature_names
        entry["updated_at"] = datetime.now().isoformat()
        registry[key] = entry

        registry_path = os.path.join(self.model_dir, "model_registry.json")
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def _cleanup_old_versions(self, province: str, target_type: str):
        """删除超出 MAX_VERSIONS 的旧模型文件 (含 .lgbm 和 .lgb)."""
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        versions = registry.get(key, {}).get("versions", [])

        province_dir = os.path.join(self.model_dir, province)
        if not os.path.exists(province_dir):
            return

        # 构建保留集合 (版本文件 + 其元数据)
        keep = set(versions)
        for v in versions:
            keep.add(v + ".meta")

        for fname in os.listdir(province_dir):
            prefix = f"{province}_{target_type}"
            if fname.startswith(prefix) and (fname.endswith(".lgbm") or fname.endswith(".lgb")):
                if fname not in keep:
                    old_path = os.path.join(province_dir, fname)
                    os.remove(old_path)
                    # 同时清理 meta 文件
                    meta_path = old_path + ".meta"
                    if os.path.exists(meta_path):
                        os.remove(meta_path)

    def _read_registry(self) -> Dict:
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                return json.load(f)
        return {}
