"""price_regime.py — 价格区间回归: 4 独立模型, 按价格体制分别训练.

Negative / LowNormal / High: LGB (树模型适合正常范围)
Extreme: Ridge 线性 (可外推, 不压缩极端值)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.ml.layers.price_classify import PriceClassifyLayer, N_PRICE_CLASSES, PRICE_CLASSES
import logging

logger = logging.getLogger(__name__)

# 4 回归区间 (负价独立建模, 解决树模型外推缺陷)
# Regime 0: negative  → 负价 (单独训练, 让模型学会负值预测)
# Regime 1: low_normal → 稳价 (供需平衡, 量驱动)
# Regime 2: high      → 高价 (供给偏紧)
# Regime 3: extreme   → 极端 (稀缺/博弈, 机制完全不同)
REGIME_MAP = {
    0: "negative",      # negative → regime 0
    1: "low_normal",    # low → regime 1
    2: "low_normal",    # normal → regime 1
    3: "high",          # high → regime 2
    4: "extreme",       # extreme → regime 3
}
REGIME_LABELS = ["negative", "low_normal", "high", "extreme"]


def _get_regime_idx(class_label: int) -> int:
    if class_label == 0:
        return 0  # negative
    elif class_label <= 2:
        return 1  # low_normal
    elif class_label == 3:
        return 2  # high
    else:
        return 3  # extreme


class PriceRegimeRegressor:
    """4 独立回归器: negative/low_normal/high → LGB, extreme → Ridge 线性.

    训练: S1 → 分类打标签 → 拆分为 4 个训练集 → 独立训练
    预测: 分类概率为权重 → 加权混合 4 个回归器输出 (软边界)
    """

    def __init__(self):
        self.price_classify = PriceClassifyLayer()
        self.models = [None, None, None, None]  # [negative, low_normal, high, extreme]
        self.extreme_scaler = None  # Ridge 用 StandardScaler
        self.feature_names = []
        self.thresholds = {}
        self.neg_train_mean = 0.0
        self.neg_train_std = 1.0

    def train(self, df: pd.DataFrame, target_type: str,
              feature_names: list, extreme_floor: float = None, **kwargs) -> dict:
        """完整训练: 分类 + 4 回归."""
        if feature_names is None:
            feature_names = [c for c in df.columns
                             if c not in {"dt", "province", "type", "price", "model_version",
                                          "p10", "p50", "p90", "trend_adjusted", "value"}
                             and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
        self.feature_names = feature_names or []

        # Phase 1: 训练分类器
        logger.info("[PriceRegime] 训练价格分类器")
        cls_result = self.price_classify.train(df, target_type=target_type,
                                                feature_names=feature_names,
                                                extreme_floor=extreme_floor)
        self.thresholds = cls_result.get("thresholds", {})

        # Phase 2: 拆分训练集
        y = df["value"].values
        labels = self.price_classify._build_labels(y)

        results = {"classify": cls_result, "regimes": {}}

        for regime_idx, regime_name in enumerate(REGIME_LABELS):
            mask = np.array([_get_regime_idx(l) == regime_idx for l in labels])
            n_regime = mask.sum()

            if n_regime < 50:
                logger.warning("[PriceRegime] %s 样本不足 (%d), 使用全部数据", regime_name, n_regime)
                mask = np.ones(len(df), dtype=bool)

            df_regime = df[mask]
            X = self._extract(df_regime)
            y_regime = df_regime["value"].values

            # Extreme regime → Ridge 线性 (可外推, 不压缩极端值)
            if regime_idx == 3:
                # 过滤零方差和 NaN 特征
                X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
                valid_cols = np.std(X_clean, axis=0) > 1e-8
                if valid_cols.sum() < 3:
                    logger.warning("[PriceRegime] extreme 有效特征不足(%d), 回退 LGB", valid_cols.sum())
                    model = lgb.LGBMRegressor(
                        objective="regression", n_estimators=400, num_leaves=63,
                        learning_rate=0.02, random_state=42, verbose=-1,
                    )
                    model.fit(X, y_regime)
                else:
                    X_clean = X_clean[:, valid_cols]
                    self.extreme_scaler = StandardScaler()
                    X_scaled = self.extreme_scaler.fit_transform(X_clean)
                    self.extreme_valid_cols = valid_cols
                    model = Ridge(alpha=10.0, fit_intercept=True)
                    model.fit(X_scaled, y_regime)
                self.models[regime_idx] = model
            else:
                model = lgb.LGBMRegressor(
                    objective="regression", n_estimators=400, num_leaves=63,
                    learning_rate=0.02, random_state=42, verbose=-1,
                )
                model.fit(X, y_regime)
                self.models[regime_idx] = model

            # 简单评估
            if regime_idx == 3 and self.extreme_scaler is not None:
                X_eval = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
                if hasattr(self, "extreme_valid_cols"):
                    X_eval = X_eval[:, self.extreme_valid_cols]
                X_eval = self.extreme_scaler.transform(X_eval)
                pred = model.predict(X_eval)
            else:
                pred = model.predict(X)
            mape = float(np.mean(np.abs(pred - y_regime) / (np.abs(y_regime) + 1)) * 100)
            rmse = float(np.sqrt(np.mean((pred - y_regime) ** 2)))
            logger.info("[PriceRegime] %s: n=%d, MAPE=%.1f%%, RMSE=%.1f",
                         regime_name, n_regime, mape, rmse)

            results["regimes"][regime_name] = {
                "n_samples": int(n_regime), "mape": round(mape, 2), "rmse": round(rmse, 2),
            }

            # 存储负价 regime 统计 (用于分布外预测时的 fallback)
            if regime_name == "negative":
                self.neg_train_mean = float(np.mean(y_regime))
                self.neg_train_std = float(np.std(y_regime)) if len(y_regime) > 1 else 1.0

        return results

    def predict(self, df: pd.DataFrame, min_regime_weight: float = 0.10,
                negative_boost: float = 0.0) -> np.ndarray:
        """软边界加权预测: Σ prob_class_i × pred_regime(i).

        min_regime_weight: 每个 regime 的最小混合权重, 防止分类器外推失效.
        negative_boost: 负价 regime 额外权重加成 (当近期出现负价时使用).
        """
        probs = self.price_classify.predict(df)  # (n, 5)

        all_preds = []
        for regime_idx in range(4):
            if self.models[regime_idx] is not None:
                X = self._extract(df)
                # Extreme regime → Ridge 线性 (需要 scale + clean)
                if regime_idx == 3 and self.extreme_scaler is not None:
                    X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
                    if hasattr(self, "extreme_valid_cols"):
                        X_clean = X_clean[:, self.extreme_valid_cols]
                    X_scaled = self.extreme_scaler.transform(X_clean)
                    raw = self.models[regime_idx].predict(X_scaled)
                else:
                    raw = self.models[regime_idx].predict(X)
                # 负价 regime: 树模型在分布外特征上预测偏浅,
                # 当 negative_boost 激活时向训练均值收缩
                if regime_idx == 0 and negative_boost > 0:
                    alpha = min(1.0, negative_boost * 2.0)
                    raw = (1.0 - alpha) * raw + alpha * self.neg_train_mean
                all_preds.append(raw)
            else:
                all_preds.append(np.zeros(len(df)))

        # 加权: prob[0] → negative, prob[1]+prob[2] → low_normal,
        #       prob[3] → high, prob[4] → extreme
        mw = min_regime_weight
        # 负价加成: 当 negative_boost 激活时, negative regime 获得主导权重,
        # 解决树模型在分布外特征上全部预测 high 类的问题
        if negative_boost > 0.15:
            # 负价信号强 → negative regime 主导 (≥50%)
            weight_0 = probs[:, 0] + 0.50 + negative_boost
            weight_1 = probs[:, 1] + probs[:, 2] + 0.05
            weight_2 = probs[:, 3] + 0.05
            weight_3 = probs[:, 4] + 0.05
        else:
            weight_0 = probs[:, 0] + mw + negative_boost
            weight_1 = probs[:, 1] + probs[:, 2] + mw
            weight_2 = probs[:, 3] + mw
            weight_3 = probs[:, 4] + mw

        total = weight_0 + weight_1 + weight_2 + weight_3 + 1e-10
        final = (weight_0 * all_preds[0] + weight_1 * all_preds[1]
                 + weight_2 * all_preds[2] + weight_3 * all_preds[3]) / total

        return final

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        """安全提取特征矩阵."""
        X = np.zeros((len(df), len(self.feature_names)), dtype=np.float64)
        for i, fn in enumerate(self.feature_names):
            if fn in df.columns:
                col = pd.to_numeric(df[fn], errors="coerce").fillna(0).values
                X[:, i] = col
        return X

    def load(self, base_dir: str, prefix: str) -> bool:
        """加载分类器 + 4 回归器."""
        import os, pickle
        cls_path = os.path.join(base_dir, f"{prefix}_price_cls.lgbm")
        self.price_classify.load(cls_path)
        self.thresholds = self.price_classify.metadata.get("thresholds", {})
        self.feature_names = self.price_classify.feature_names

        for i, name in enumerate(REGIME_LABELS):
            # Extreme regime → Ridge (pickle)
            if i == 3:
                ridge_path = os.path.join(base_dir, f"{prefix}_price_reg_{name}.pkl")
                if os.path.exists(ridge_path):
                    with open(ridge_path, "rb") as f:
                        saved = pickle.load(f)
                    self.models[i] = saved["model"]
                    self.extreme_scaler = saved.get("scaler")
                    self.extreme_valid_cols = saved.get("valid_cols")
                else:
                    self.models[i] = None
                continue

            reg_path = os.path.join(base_dir, f"{prefix}_price_reg_{name}.lgbm")
            if os.path.exists(reg_path):
                booster = lgb.Booster(model_file=reg_path)
                model = lgb.LGBMRegressor()
                model._Booster = booster
                model._n_features = booster.num_feature()
                model.n_features_in_ = model._n_features
                model.fitted_ = True
                self.models[i] = model
            else:
                self.models[i] = None

        return True

    def save(self, base_dir: str, prefix: str) -> None:
        """保存分类器 + 4 回归器."""
        import os, pickle
        os.makedirs(base_dir, exist_ok=True)

        cls_path = os.path.join(base_dir, f"{prefix}_price_cls.lgbm")
        self.price_classify.save(cls_path)

        for i, name in enumerate(REGIME_LABELS):
            if self.models[i] is None:
                continue
            # Extreme regime → Ridge (pickle)
            if i == 3:
                ridge_path = os.path.join(base_dir, f"{prefix}_price_reg_{name}.pkl")
                with open(ridge_path, "wb") as f:
                    pickle.dump({"model": self.models[i], "scaler": self.extreme_scaler,
                                 "valid_cols": getattr(self, "extreme_valid_cols", None)}, f)
                continue
            # Other regimes → LGB
            reg_path = os.path.join(base_dir, f"{prefix}_price_reg_{name}.lgbm")
            if hasattr(self.models[i], "booster_"):
                self.models[i].booster_.save_model(reg_path)
            elif hasattr(self.models[i], "_Booster"):
                self.models[i]._Booster.save_model(reg_path)
