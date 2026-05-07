"""trend_classify.py — 趋势分类层: LGBM 三分类预测短期方向, 为融合层提供门控信号."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scripts.ml.layers.base import BaseLayer
import logging

logger = logging.getLogger(__name__)

# 三分类: 下行 / 平稳 / 上行
TREND_CLASSES = ["down", "flat", "up"]
N_CLASSES = 3
# 相对变化阈值 (6h 均值 vs 当前值)
DOWN_THRESHOLD = -0.02
UP_THRESHOLD = 0.02


class TrendClassifyLayer(BaseLayer):
    """预测未来短期趋势方向 (下行/平稳/上行), 输出各类概率.

    y 构造: 对每条样本, 取未来 24 步 (6h) 的均值 vs 当前值,
    按相对变化量分为三类.
    """

    def __init__(self, lookahead_steps: int = 24,
                 down_threshold: float = -0.02,
                 up_threshold: float = 0.02):
        super().__init__("trend_classify")
        self.lookahead_steps = lookahead_steps
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.n_classes = N_CLASSES
        self.class_names = list(TREND_CLASSES)
        self._class_weights = None

    def _build_labels(self, values: np.ndarray) -> np.ndarray:
        """从时序值构造三分类标签.

        对每个位置 t, 计算:
          change = (mean(values[t+1:t+1+lookahead]) - values[t]) / (|values[t]| + eps)
        分类:
          0 (down):  change < down_threshold
          1 (flat):  down_threshold <= change <= up_threshold
          2 (up):    change > up_threshold
        末尾 lookahead 步无法计算标签, 填充为 1 (flat).
        """
        n = len(values)
        labels = np.full(n, 1, dtype=int)  # 默认 flat
        eps = np.mean(np.abs(values)) * 0.01 + 1.0

        for t in range(n - self.lookahead_steps):
            future = values[t + 1 : t + 1 + self.lookahead_steps]
            current = values[t]
            change = (np.mean(future) - current) / (abs(current) + eps)
            if change < self.down_threshold:
                labels[t] = 0
            elif change > self.up_threshold:
                labels[t] = 2
            else:
                labels[t] = 1

        return labels

    def train(self, df: pd.DataFrame, target_type: str = "",
              feature_names: list = None, target_col: str = "value",
              **kwargs) -> dict:
        self.feature_names = feature_names or []

        y_vals = df[target_col].values
        labels = self._build_labels(y_vals)

        # 过滤尾部的 flat 填充 (末尾 lookahead 步不是真实标签)
        valid_len = len(labels) - self.lookahead_steps
        if valid_len < 100:
            logger.warning("TrendClassify: 有效样本不足 (%d), 跳过", valid_len)
            self._trained = True
            return {"active": False, "reason": f"有效样本不足: {valid_len}"}

        train_labels = labels[:valid_len]
        counts = np.bincount(train_labels, minlength=N_CLASSES)
        class_ratio = counts / counts.sum()
        logger.info("TrendClassify 标签分布: down=%.1f%%, flat=%.1f%%, up=%.1f%%",
                     class_ratio[0] * 100, class_ratio[1] * 100, class_ratio[2] * 100)

        if class_ratio.min() < 0.05:
            logger.warning("TrendClassify: 类别不平衡, 跳过")
            self._trained = True
            return {"active": False, "reason": f"类别不平衡: {class_ratio}"}

        X = self._extract_features(df.iloc[:valid_len])

        self._class_weights = {i: 1.0 / max(r, 0.01) for i, r in enumerate(class_ratio)}
        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES,
            n_estimators=300, num_leaves=31, learning_rate=0.03,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        self.model.fit(X, train_labels)
        self._trained = True

        return {
            "active": True,
            "n_classes": N_CLASSES,
            "class_distribution": {TREND_CLASSES[i]: round(float(r), 4) for i, r in enumerate(class_ratio)},
            "n_samples": valid_len,
            "n_features": len(self.feature_names),
        }

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """返回 (n, 3) 趋势概率矩阵."""
        if not self._trained or self.model is None:
            return np.full((len(df), N_CLASSES), [0.0, 1.0, 0.0])

        X = self._extract_features(df)

        # 直接调用 booster.predict (避开 LGBMClassifier sklearn 包装器的兼容问题)
        booster = getattr(self.model, "_Booster", None)
        if booster is not None:
            probs = booster.predict(X)
            if probs.ndim == 1:
                probs = probs.reshape(-1, N_CLASSES)
        else:
            probs = self.model.predict_proba(X)

        if probs.shape[1] != N_CLASSES:
            full = np.full((len(df), N_CLASSES), [0.0, 1.0, 0.0])
            n_copy = min(probs.shape[1], N_CLASSES)
            full[:, :n_copy] = probs[:, :n_copy]
            return full

        return probs

    def evaluate(self, df: pd.DataFrame, actual_values: np.ndarray) -> dict:
        """评估趋势方向准确率.

        对比预测趋势方向 vs 真实方向标签.
        返回: accuracy, per-class precision/recall, confusion matrix.
        """
        labels = self._build_labels(actual_values)
        valid_len = len(labels) - self.lookahead_steps
        if valid_len < 10:
            return {"accuracy": None, "error": "样本不足"}

        probs = self.predict(df.iloc[:valid_len])
        preds = np.argmax(probs, axis=1)
        trues = labels[:valid_len]

        # 总体准确率
        accuracy = float(np.mean(preds == trues))

        # 每类 precision / recall
        per_class = {}
        for i, name in enumerate(self.class_names):
            tp = int(np.sum((preds == i) & (trues == i)))
            fp = int(np.sum((preds == i) & (trues != i)))
            fn = int(np.sum((preds != i) & (trues == i)))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            per_class[name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "support": int(np.sum(trues == i)),
            }

        # 混淆矩阵 (3×3)
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1

        # 二分类方向准确率 (忽略 flat, 只看 up vs down)
        directional_mask = trues != 1  # 非 flat 样本
        if directional_mask.sum() >= 10:
            dir_acc = float(np.mean(
                preds[directional_mask] == trues[directional_mask]
            ))
        else:
            dir_acc = None

        logger.info(
            "趋势方向评估: acc=%.2f%%, dir_acc=%s, n=%d",
            accuracy * 100,
            f"{dir_acc*100:.1f}%" if dir_acc is not None else "N/A",
            valid_len,
        )

        return {
            "accuracy": round(accuracy, 4),
            "directional_accuracy": round(dir_acc, 4) if dir_acc is not None else None,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "n_valid": valid_len,
        }
