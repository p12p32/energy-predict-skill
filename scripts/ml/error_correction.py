"""error_correction.py — 残差修正模型

两阶段预测:
  Stage 1: LightGBM → ŷ_lgb (捕捉非线性)
  Stage 2: AR 模型预测残差 r̂, 修正 ŷ_final = ŷ_lgb + r̂

v2 改进:
  - 自适应 AR order: 用 PACF 自动选择最优阶数
  - 安全边界: 残差修正量不超过 P50 的配置比例
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class ErrorCorrectionModel:
    """AR(p) 残差模型.

    在训练集上计算残差序列 r_t = y_t - ŷ_t,
    用 AR(p) 拟合残差的时序结构,
    预测时先用 AR 预测未来残差, 再叠加到 LightGBM 预测上.
    """

    def __init__(self, order: int = 96, auto_order: bool = True,
                 max_order: int = 192, min_order: int = 12):
        self.order = order
        self.auto_order = auto_order
        self.max_order = max_order
        self.min_order = min_order
        self.coefficients = None     # AR 系数
        self.training_mape = None

    def _select_order(self, residuals: np.ndarray) -> int:
        """用 PACF 选择最优 AR 阶数 (AIC 准则)."""
        n = len(residuals)
        max_lag = min(self.max_order, n // 5)
        if max_lag < self.min_order:
            return max_lag

        best_order = self.min_order
        best_aic = float("inf")

        for p in range(self.min_order, max_lag + 1, 12):  # 步长12
            try:
                acf = self._autocorr(residuals, p)
                coef = self._levinson_durbin(acf)
                if len(coef) == 0:
                    continue

                pred = self._predict_in_sample(residuals, coef)
                if len(pred) < 10:
                    continue

                rss = np.sum((residuals[p:] - pred) ** 2)
                if rss <= 0:
                    continue
                aic = n * np.log(rss / n) + 2 * p
                if aic < best_aic:
                    best_aic = aic
                    best_order = p
            except Exception:
                continue

        return best_order

    def fit(self, residuals: np.ndarray):
        """用残差序列拟合 AR 模型.

        r[t] = Σ a_i * r[t-i] + ε_t
        """
        n = len(residuals)
        if n < self.min_order + 10:
            self.coefficients = np.array([0.0])
            return

        if self.auto_order:
            self.order = self._select_order(residuals)

        # 构建 Yule-Walker 方程 (快速 AR 估计)
        acf = self._autocorr(residuals, self.order)
        coef = self._levinson_durbin(acf)
        self.coefficients = coef

        # 训练集 MAPE
        pred = self._predict_in_sample(residuals)
        mask = residuals[self.order:] != 0
        if mask.sum() > 0:
            self.training_mape = float(
                np.mean(np.abs(pred[mask] - residuals[self.order:][mask]) /
                        np.abs(residuals[self.order:][mask]))
            )

    def predict(self, recent_residuals: np.ndarray, steps: int) -> np.ndarray:
        """预测未来 steps 步的残差值.

        Args:
            recent_residuals: 最近的残差序列 (至少 order 长度)
            steps: 预测步数
        """
        if self.coefficients is None or len(self.coefficients) == 0:
            return np.zeros(steps)

        coeffs = self.coefficients
        order = len(coeffs)

        history = list(recent_residuals[-order:])
        predictions = []

        for _ in range(steps):
            pred_val = sum(c * history[-i - 1] for i, c in enumerate(coeffs[:len(history)]))
            predictions.append(pred_val)
            history.append(pred_val)

        return np.array(predictions)

    def correct(self, lgb_predictions: np.ndarray,
                recent_residuals: np.ndarray,
                clip_ratio: float = 0.30) -> np.ndarray:
        """完整修正: LightGBM 预测 + AR 残差修正.

        clip_ratio: 残差修正上限 (相对P50绝对值), 防止极端偏差.
        """
        ar_pred = self.predict(recent_residuals, len(lgb_predictions))
        p50_abs = np.maximum(np.abs(lgb_predictions), 1.0)
        max_correction = clip_ratio * p50_abs
        ar_pred = np.clip(ar_pred, -max_correction, max_correction)
        return lgb_predictions + ar_pred

    def fit_and_correct(self, y_true: np.ndarray, y_pred_lgb: np.ndarray,
                        horizon: int, clip_ratio: float = 0.30) -> Dict:
        """一站式: 拟合残差模型 + 修正预测.

        Args:
            y_true: 训练集的真实值
            y_pred_lgb: LightGBM 在训练集上的预测值
            horizon: 预测步数
            clip_ratio: 残差修正上限 (相对P50绝对值)

        Returns:
            {"corrected": array, "ar_pred": array, "coefficients": array}
        """
        residuals = y_true - y_pred_lgb
        self.fit(residuals)

        recent = residuals[-self.order:]
        ar_pred = self.predict(recent, horizon)
        p50_abs = np.maximum(np.abs(y_pred_lgb[-horizon:]), 1.0)
        max_correction = clip_ratio * p50_abs
        ar_pred = np.clip(ar_pred, -max_correction, max_correction)
        corrected = y_pred_lgb[-horizon:] + ar_pred

        return {
            "corrected": corrected,
            "ar_pred": ar_pred,
            "coefficients": self.coefficients.tolist() if self.coefficients is not None else [],
            "order": self.order,
        }

    # ── 内部方法 ──

    @staticmethod
    def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
        """计算自相关函数."""
        n = len(x)
        x = x - np.mean(x)
        acf = np.zeros(max_lag + 1)
        var = np.dot(x, x) / n
        if var == 0:
            return acf
        for lag in range(max_lag + 1):
            acf[lag] = np.dot(x[:n - lag], x[lag:]) / (n - lag) / var if lag < n else 0
        return acf

    @staticmethod
    def _levinson_durbin(r: np.ndarray) -> np.ndarray:
        """Levinson-Durbin 递推求解 AR 系数."""
        p = len(r) - 1
        if p == 0:
            return np.array([])

        a = np.zeros(p)
        e = r[0]

        for k in range(p):
            # 反射系数
            lam = -np.dot(a[:k], r[k:0:-1]) - r[k + 1]
            lam /= e

            a[k] = lam
            a[:k] = a[:k] + lam * a[:k][::-1]
            e *= (1 - lam ** 2)

        return -a

    def _predict_in_sample(self, residuals: np.ndarray,
                           coef: np.ndarray = None) -> np.ndarray:
        """训练集上的一步预测."""
        if coef is None:
            coef = self.coefficients
        if coef is None or len(coef) == 0:
            return np.array([])
        n = len(residuals)
        order = len(coef)
        if n <= order:
            return np.array([])
        pred = np.zeros(n - order)

        for t in range(order, n):
            window = residuals[t - order:t]
            pred[t - order] = np.dot(coef[::-1], window)

        return pred
