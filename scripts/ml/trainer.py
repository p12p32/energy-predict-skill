"""trainer.py — 模型训练器（LightGBM + 分位数回归 + 超参搜索 + 版本回滚）

修复:
- 加入简单超参搜索 (grid search + expanding window CV)
- 验证改为 expanding window cross validation
- quick_train 加超时保护
- 保存最优超参到 model_registry.json
"""
import os
import json
import pickle
import logging
import time
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from scripts.core.config import get_model_config, get_training_config

logger = logging.getLogger(__name__)

EXCLUDE_COLS = {"dt", "province", "type", "price"}
MAX_VERSIONS = 3
QUANTILES = [0.1, 0.5, 0.9]

# 光伏白天有效时段 (6:00~19:45, 对应 hour 6~19)
SOLAR_ACTIVE_HOURS = set(range(6, 20))

# 风电高活跃时段 (通常风速较大)
WIND_ACTIVE_HOURS = set(range(0, 24))  # 全天都有风


class _TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutException("训练超时")


class Trainer:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or get_model_config()["storage_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    # ── 安全模型 I/O ──

    @staticmethod
    def _save_model_native(model: lgb.LGBMRegressor, path: str):
        model.booster_.save_model(path)

    @staticmethod
    def _save_meta(path: str, feature_names: List[str], params: Dict = None):
        meta = {"feature_names": feature_names}
        if params:
            meta["best_params"] = params
        with open(path + ".meta", "w") as f:
            json.dump(meta, f)

    @staticmethod
    def _load_meta(path: str) -> Dict:
        meta_path = path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _load_model_native(path: str) -> lgb.LGBMRegressor:
        model = lgb.LGBMRegressor()
        booster = lgb.Booster(model_file=path)
        n_features = booster.num_feature()
        model._Booster = booster
        model._n_features = n_features
        model.n_features_in_ = n_features
        model.fitted_ = True
        return model

    @staticmethod
    def _load_model_file(path: str) -> Tuple[lgb.LGBMRegressor, List[str], Dict]:
        lgbm_path = path if path.endswith(".lgbm") else path.replace(".lgb", ".lgbm")
        if os.path.exists(lgbm_path):
            model = Trainer._load_model_native(lgbm_path)
            meta = Trainer._load_meta(lgbm_path)
            return model, meta.get("feature_names", []), meta

        if os.path.exists(path) and path.endswith(".lgb"):
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model, [], {}

        if os.path.exists(path):
            try:
                return Trainer._load_model_native(path), Trainer._load_meta(path)
            except Exception:
                with open(path, "rb") as f:
                    return pickle.load(f), [], {}

        raise FileNotFoundError(f"模型文件不存在: {path}")

    def prepare_training_data(self, df: pd.DataFrame,
                               target_col: str = "value",
                               target_type: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.dropna(subset=[target_col]).copy()

        if "value_type" in df.columns:
            vt_col = df["value_type"]
            actual_mask = vt_col.isna() | (vt_col == "实际")
            if not actual_mask.all():
                skipped = (~actual_mask).sum()
                logger.info("训练过滤: 排除 %d 条非实际值", skipped)
                df = df[actual_mask]

        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS and c != target_col
        ]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        return X, y

    # ── 超参搜索 ──

    def _search_params(self, X: pd.DataFrame, y: pd.Series,
                       timeout: int = 120) -> Tuple[Dict, float]:
        """简单 grid search: expanding window CV + 精度优先.

        返回 (best_params, best_mape).
        """
        train_cfg = get_training_config()
        if not train_cfg.get("hyperparam_search", True):
            return train_cfg.get("default_params", {}), 0.0

        search_space = train_cfg.get("search_space", {})
        default_params = train_cfg.get("default_params", {})
        cv_folds = train_cfg.get("cv_folds", 3)

        param_grid = [
            {"num_leaves": nl, "learning_rate": lr, "n_estimators": ne,
             "min_child_samples": mc,
             "feature_fraction": default_params.get("feature_fraction", 0.8),
             "bagging_fraction": default_params.get("bagging_fraction", 0.8),
             "bagging_freq": default_params.get("bagging_freq", 5)}
            for nl in search_space.get("num_leaves", [31])
            for lr in search_space.get("learning_rate", [0.03])
            for ne in search_space.get("n_estimators", [500])
            for mc in search_space.get("min_child_samples", [20])
        ]

        n = len(X)
        fold_size = n // (cv_folds + 1)

        best_mape = float('inf')
        best_params = default_params.copy()
        start_time = time.time()

        for combo in param_grid:
            if time.time() - start_time > timeout:
                logger.info("超参搜索超时 (%ds), 使用当前最优", timeout)
                break

            fold_mapes = []
            for fold in range(cv_folds):
                # expanding window
                val_end = n - (cv_folds - 1 - fold) * fold_size
                val_start = max(0, val_end - fold_size)
                train_end = val_start

                X_train = X.iloc[:train_end]
                y_train = y.iloc[:train_end]
                X_val = X.iloc[val_start:val_end]
                y_val = y.iloc[val_start:val_end]

                if len(X_train) < 200 or len(X_val) < 50:
                    continue

                try:
                    m = lgb.LGBMRegressor(
                        objective="quantile", alpha=0.5,
                        metric="quantile", boosting_type="gbdt",
                        verbose=-1, **combo,
                        early_stopping_rounds=30,
                    )
                    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          eval_metric="rmse")
                    pred = m.predict(X_val)
                    mask = y_val != 0
                    if mask.sum() > 10:
                        mape = float(np.mean(np.abs((y_val[mask] - pred[mask]) / y_val[mask])))
                    else:
                        mape = float('inf')
                    fold_mapes.append(mape)
                except Exception as e:
                    logger.debug("超参组合训练失败: %s", e)
                    continue

            if fold_mapes:
                avg_mape = float(np.mean(fold_mapes))
                if avg_mape < best_mape:
                    best_mape = avg_mape
                    best_params = combo.copy()
                    logger.info("超参更新: MAPE=%.4f, params=%s", avg_mape, combo)

        logger.info("超参搜索完成: best MAPE=%.4f", best_mape)
        return best_params, best_mape

    # ── 训练接口 ──

    def train(self, df: pd.DataFrame, province: str,
              target_type: str, target_col: str = "value",
              params: Dict = None, model_filename: str = None,
              search_params: bool = True) -> Dict:
        X, y = self.prepare_training_data(df, target_col, target_type=target_type)

        train_cfg = get_training_config()
        default_params = train_cfg.get("default_params", {})

        # 超参搜索
        best_params = default_params.copy()
        if search_params and params is None:
            timeout = train_cfg.get("max_search_time_seconds", 120)
            best_params, _ = self._search_params(X, y, timeout)
        elif params:
            best_params.update(params)

        # Expanding window CV 选验证集
        n = len(X)
        cv_folds = train_cfg.get("cv_folds", 3)
        fold_size = n // (cv_folds + 1)
        val_end = n - fold_size
        val_start = max(0, val_end - fold_size)
        train_end = val_start

        X_train, X_val = X.iloc[:train_end], X.iloc[val_start:val_end]
        y_train, y_val = y.iloc[:train_end], y.iloc[val_start:val_end]

        if len(X_train) < 200:
            X_train, X_val = X, pd.DataFrame()
            y_train = y

        lgb_params = {
            "objective": "quantile", "alpha": 0.5,
            "metric": "quantile", "boosting_type": "gbdt",
            "verbose": -1, **best_params,
            "early_stopping_rounds": 50,
        }

        model = lgb.LGBMRegressor(**lgb_params)
        fit_kwargs = {"X": X_train, "y": y_train}
        if not X_val.empty:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = "rmse"
        model.fit(**fit_kwargs)

        if model_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{province}_{target_type}_{timestamp}.lgbm"

        province_dir = os.path.join(self.model_dir, province)
        os.makedirs(province_dir, exist_ok=True)
        model_path = os.path.join(province_dir, model_filename)

        self._save_model_native(model, model_path)
        self._save_meta(model_path, list(X.columns), best_params)
        self._update_registry(province, target_type, model_filename,
                              list(X.columns), model_path, best_params)
        self._cleanup_old_versions(province, target_type)

        return {
            "province": province,
            "target_type": target_type,
            "model_path": model_path,
            "n_samples": len(df),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
            "best_params": best_params,
        }

    def quick_train(self, df: pd.DataFrame, province: str,
                    target_type: str, target_col: str = "value",
                    params: Dict = None,
                    timeout: int = 60) -> Dict:
        """快速训练 (带超时保护).

        返回带 model 对象的 dict，用于 improver 的快速实验。
        """
        X, y = self.prepare_training_data(df, target_col, target_type=target_type)
        lgb_params = {
            "objective": "regression", "metric": "rmse",
            "num_leaves": 31, "learning_rate": 0.05,
            "verbose": -1, "n_estimators": 300,
            "min_child_samples": 20,
        }
        if params:
            lgb_params.update(params)

        model = None
        try:
            if timeout and timeout > 0:
                def _train():
                    nonlocal model
                    model = lgb.LGBMRegressor(**lgb_params)
                    model.fit(X, y)
                # 非阻塞超时 (Unix)
                try:
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout)
                    _train()
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except (ValueError, _TimeoutException):
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)
                    logger.warning("quick_train 超时 (%ds), 使用部分结果", timeout)
            else:
                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(X, y)
        except _TimeoutException:
            logger.warning("quick_train 超时")
        except Exception as e:
            logger.warning("quick_train 失败: %s", e)

        if model is None:
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X.iloc[:min(len(X), 5000)], y.iloc[:min(len(y), 5000)])

        return {
            "model": model,
            "feature_names": list(X.columns),
            "n_samples": len(df),
            "province": province,
            "target_type": target_type,
        }

    def quantile_train(self, df: pd.DataFrame, province: str,
                       target_type: str, target_col: str = "value") -> Dict:
        X, y = self.prepare_training_data(df, target_col, target_type=target_type)

        train_cfg = get_training_config()
        n = len(X)
        cv_folds = train_cfg.get("cv_folds", 3)
        fold_size = n // (cv_folds + 1)
        val_end = n - fold_size
        val_start = max(0, val_end - fold_size)
        train_end = val_start

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]

        default_params = train_cfg.get("default_params", {})
        base_params = {
            "objective": "quantile", "metric": "quantile",
            "boosting_type": "gbdt", "verbose": -1,
            "early_stopping_rounds": 50,
            "feature_fraction": default_params.get("feature_fraction", 0.8),
            "bagging_fraction": default_params.get("bagging_fraction", 0.8),
            "bagging_freq": default_params.get("bagging_freq", 5),
            "n_estimators": default_params.get("n_estimators", 500),
            "num_leaves": default_params.get("num_leaves", 31),
            "learning_rate": default_params.get("learning_rate", 0.03),
            "min_child_samples": default_params.get("min_child_samples", 20),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        province_dir = os.path.join(self.model_dir, province)
        os.makedirs(province_dir, exist_ok=True)

        models = {}
        for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
            m = lgb.LGBMRegressor(alpha=alpha, **base_params)
            fit_kw = {"X": X_train, "y": y_train}
            if not X_val.empty:
                fit_kw["eval_set"] = [(X_val, y_val)]
                fit_kw["eval_metric"] = "quantile"
            m.fit(**fit_kw)

            fname = f"{province}_{target_type}_{label}_{timestamp}.lgbm"
            path = os.path.join(province_dir, fname)
            self._save_model_native(m, path)
            self._save_meta(path, list(X.columns))
            models[label] = path

        p50_fname = f"{province}_{target_type}_p50_{timestamp}.lgbm"
        self._update_registry(
            province, target_type, p50_fname,
            list(X.columns),
            os.path.join(province_dir, p50_fname),
            base_params,
        )

        return {
            "province": province, "target_type": target_type,
            "paths": models, "n_samples": len(df), "n_features": X.shape[1],
            "feature_names": list(X.columns), "params": base_params,
        }

    # ── 模型加载 ──

    def load_quantile_models(self, province: str, target_type: str,
                              version: int = -1) -> Dict[str, Tuple]:
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            raise FileNotFoundError(f"未找到模型: {key}")

        entry = registry[key]
        versions = entry.get("versions", [])
        if not versions:
            raise FileNotFoundError(f"未找到模型版本: {key}")

        p50_fname = versions[version] if version >= 0 else versions[-1]

        if "p50_" in p50_fname:
            p10_fname = p50_fname.replace("p50_", "p10_")
            p90_fname = p50_fname.replace("p50_", "p90_")
        else:
            p10_fname = "__nonexistent__"
            p90_fname = "__nonexistent__"

        province_dir = os.path.join(self.model_dir, province)
        feature_names = entry.get("feature_names", [])
        models = {}

        for label, fname in [("p10", p10_fname), ("p50", p50_fname), ("p90", p90_fname)]:
            path = os.path.join(province_dir, fname)
            try:
                m, fn, meta = self._load_model_file(path)
                models[label] = (m, fn if fn else feature_names)
            except FileNotFoundError:
                logger.warning("分位数模型文件缺失: %s", path)

        return models

    def load_model(self, province: str,
                   target_type: str,
                   version: int = -1) -> Tuple[lgb.LGBMRegressor, List[str], str]:
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

        models, feature_names, meta = self._load_model_file(model_path)
        if not feature_names:
            feature_names = entry.get("feature_names", [])

        return models, feature_names, model_filename

    # ── 版本回滚 ──

    def rollback(self, province: str, target_type: str) -> Dict:
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            return {"error": f"未找到模型: {key}"}

        versions = registry[key].get("versions", [])
        if len(versions) < 2:
            return {"error": "只有一个版本，无法回滚"}

        # 删除最新版本
        latest = versions.pop()
        registry[key]["versions"] = versions

        province_dir = os.path.join(self.model_dir, province)
        for ext in ["", ".meta"]:
            path = os.path.join(province_dir, latest + ext) if ext == ".meta" else os.path.join(province_dir, latest)
            # 清理 p10/p90 配套文件
            for f in [path, path.replace(".lgbm", ".lgb")]:
                if os.path.exists(f):
                    os.remove(f)
                    if f.endswith(".lgbm"):
                        meta_f = f + ".meta"
                        if os.path.exists(meta_f):
                            os.remove(meta_f)
            # p10/p90
            for prefix in ["p10", "p90"]:
                derived = latest.replace("p50_", f"{prefix}_")
                derived_path = os.path.join(province_dir, derived)
                if os.path.exists(derived_path):
                    os.remove(derived_path)
                    meta_d = derived_path + ".meta"
                    if os.path.exists(meta_d):
                        os.remove(meta_d)

        self._save_registry(registry)
        return {"status": "ok", "rolled_back": latest, "current": versions[-1]}

    def list_versions(self, province: str, target_type: str) -> List[Dict]:
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
                alt = v.replace(".lgbm", ".lgb")
                alt_path = os.path.join(self.model_dir, province, alt)
                if os.path.exists(alt_path):
                    info["size_kb"] = round(os.path.getsize(alt_path) / 1024, 1)
            result.append(info)
        return result

    # ── 特征重要性 ──

    def feature_importance(self, province: str,
                           target_type: str) -> List[Dict]:
        model, feature_names, _ = self.load_model(province, target_type)
        booster = getattr(model, "_Booster", None)
        if booster is not None:
            importances = booster.feature_importance(importance_type="gain")
        else:
            importances = model.feature_importances_

        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        total = sum(importances) or 1
        return [
            {"rank": i + 1, "feature": name, "importance": round(float(imp), 6),
             "pct": round(float(imp / total * 100), 1)}
            for i, (name, imp) in enumerate(pairs)
        ]

    # ── 注册表 ──

    def _update_registry(self, province: str, target_type: str,
                         filename: str, feature_names: List[str],
                         model_path: str, best_params: Dict = None):
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        old_versions = registry.get(key, {}).get("versions", [])

        entry = {
            "versions": (old_versions + [filename])[-MAX_VERSIONS:],
            "feature_names": feature_names,
            "updated_at": datetime.now().isoformat(),
        }
        if best_params:
            entry["best_params"] = best_params
        registry[key] = entry

        self._save_registry(registry)

    def _cleanup_old_versions(self, province: str, target_type: str):
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        versions = registry.get(key, {}).get("versions", [])

        province_dir = os.path.join(self.model_dir, province)
        if not os.path.exists(province_dir):
            return

        keep = set(versions)
        for v in versions:
            keep.add(v + ".meta")

        for fname in os.listdir(province_dir):
            prefix = f"{province}_{target_type}"
            if fname.startswith(prefix) and (fname.endswith(".lgbm") or fname.endswith(".lgb")):
                if fname not in keep:
                    old_path = os.path.join(province_dir, fname)
                    os.remove(old_path)
                    meta_path = old_path + ".meta"
                    if os.path.exists(meta_path):
                        os.remove(meta_path)

    def _read_registry(self) -> Dict:
        path = os.path.join(self.model_dir, "model_registry.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self, registry: Dict):
        path = os.path.join(self.model_dir, "model_registry.json")
        with open(path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
