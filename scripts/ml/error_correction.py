"""error_correction.py — 季节性 AR 残差修正模型

修复:
- 确保 predict 传入真实残差而非全零
- 按小时分组的季节性 AR（不同 hour 用不同系数）
- AR 阶数自适应（AIC 选择）
"""
import numpy as np
from typing import Dict, List, Optional


class ErrorCorrectionModel:
    """分时段 AR 残差修正.

    在训练集上计算残差序列 r_t = y_t - ŷ_t,
    按小时分组拟合 AR(p) 模型，
    预测时叠加残差修正到 LightGBM 预测上.
    """

    def __init__(self, max_order: int = 96, min_order: int = 12,
                 per_hour: bool = True):
        self.max_order = max_order
        self.min_order = min_order
        self.per_hour = per_hour

        # 全局 AR 系数（fallback）
        self.coefficients: Optional[np.ndarray] = None
        self.order: int = max_order
        self.training_mape: Optional[float] = None

        # 分小时 AR 系数
        self.hourly_models: Dict[int, Dict] = {}
        self._fitted = False

    def fit(self, residuals: np.ndarray,
            hours: Optional[np.ndarray] = None):
        """用残差序列拟合 AR 模型.

        Args:
            residuals: y_true - y_predicted
            hours: 对应的小时 (0-23)，用于分时段拟合
        """
        n = len(residuals)
        if n < self.min_order + 10:
            self.coefficients = np.array([0.0])
            self._fitted = False
            return

        if self.per_hour and hours is not None and n >= 96 * 3:
            self._fit_hourly(residuals, hours)
        else:
            self._fit_global(residuals)

        self._fitted = True

    def _fit_global(self, residuals: np.ndarray):
        """全局 AR 拟合 + AIC 阶数选择."""
        best_aic = float('inf')
        best_coef = np.array([0.0])
        best_order = self.min_order

        for order in range(self.min_order, min(self.max_order + 1, len(residuals) // 3)):
            if len(residuals) < order + 10:
                break
            acf = self._autocorr(residuals, order)
            coef = self._levinson_durbin(acf)
            if len(coef) == 0:
                continue

            # AIC 计算
            pred = self._predict_in_sample(residuals, coef, order)
            n_eff = len(pred)
            if n_eff < 10:
                continue
            mse = float(np.mean((residuals[order:order + n_eff] - pred) ** 2))
            if mse < 1e-20:
                mse = 1e-20
            k = len(coef[coef != 0])
            aic = n_eff * np.log(mse) + 2 * k

            if aic < best_aic:
                best_aic = aic
                best_coef = coef
                best_order = order

        self.coefficients = best_coef
        self.order = best_order

        # 训练 MAPE
        if len(residuals) > self.order:
            pred = self._predict_in_sample(residuals, self.coefficients, self.order)
            actuals = residuals[self.order:self.order + len(pred)]
            mask = np.abs(actuals) > 1e-8
            if mask.sum() > 10:
                self.training_mape = float(
                    np.mean(np.abs(pred[mask] - actuals[mask]) / np.abs(actuals[mask]))
                )

    def _fit_hourly(self, residuals: np.ndarray, hours: np.ndarray):
        """按小时分组的 AR 拟合."""
        for h in range(24):
            mask = hours == h
            h_residuals = residuals[mask]
            if len(h_residuals) < self.min_order:
                continue

            acf = self._autocorr(h_residuals, min(self.max_order, len(h_residuals) // 3))
            coef = self._levinson_durbin(acf)
            if len(coef) == 0:
                continue

            self.hourly_models[h] = {
                "coefficients": coef,
                "order": len(coef),
            }

        # 同时拟合全局作为 fallback
        self._fit_global(residuals)

    def predict(self, recent_residuals: np.ndarray,
                steps: int,
                recent_hours: Optional[np.ndarray] = None,
                future_hours: Optional[np.ndarray] = None) -> np.ndarray:
        """预测未来残差.

        Args:
            recent_residuals: 真实的历史残差序列 (y_true - y_predicted)
            steps: 预测步数
            recent_hours: 最近残差对应的小时
            future_hours: 未来预测对应的小时
        """
        if not self._fitted or self.coefficients is None:
            return np.zeros(steps)

        # 确保残差非零（核心修复：不再接受全零输入）
        if len(recent_residuals) == 0 or np.all(np.abs(recent_residuals) < 1e-15):
            return np.zeros(steps)

        # 分时段预测
        if self.per_hour and future_hours is not None and self.hourly_models:
            return self._predict_hourly(recent_residuals, steps, future_hours)

        # 全局预测
        return self._ar_forecast(recent_residuals, self.coefficients, steps)

    def _predict_hourly(self, recent_residuals: np.ndarray,
                        steps: int, future_hours: np.ndarray) -> np.ndarray:
        result = np.zeros(steps)
        # 按连续小时段分组
        segments = self._segment_by_hour(future_hours, steps)
        for seg_start, seg_end, hour in segments:
            if hour not in self.hourly_models:
                model = self.hourly_models.get(hour, None)
                if model is None:
                    continue
            else:
                model = self.hourly_models[hour]

            seg_steps = seg_end - seg_start
            coef = model["coefficients"]
            seg_pred = self._ar_forecast(recent_residuals, coef, seg_steps)
            result[seg_start:seg_end] = seg_pred

        # 没有分时段结果的部分用全局
        zero_mask = result == 0
        if zero_mask.any():
            global_pred = self._ar_forecast(recent_residuals, self.coefficients, steps)
            result = np.where(zero_mask, global_pred, result)

        return result

    @staticmethod
    def _segment_by_hour(hours: np.ndarray, steps: int) -> List:
        """将预测序列按连续相同小时分段."""
        segments = []
        if len(hours) == 0:
            return segments
        seg_start = 0
        current_hour = int(hours[0]) if len(hours) > 0 else 0
        for i in range(1, steps):
            h = int(hours[i]) if i < len(hours) else current_hour
            if h != current_hour:
                segments.append((seg_start, i, current_hour))
                seg_start = i
                current_hour = h
        segments.append((seg_start, steps, current_hour))
        return segments

    def _ar_forecast(self, recent_residuals: np.ndarray,
                     coefficients: np.ndarray,
                     steps: int) -> np.ndarray:
        """标准 AR 递推预测."""
        order = len(coefficients)
        if order == 0:
            return np.zeros(steps)

        # 只取最近 order 个残差
        history = list(recent_residuals[-order:])
        predictions = []

        for _ in range(steps):
            available = min(len(history), order)
            pred_val = sum(
                coefficients[order - 1 - i] * history[-(i + 1)]
                for i in range(available)
            )
            predictions.append(pred_val)
            history.append(pred_val)

        return np.array(predictions)

    def correct(self, lgb_predictions: np.ndarray,
                recent_residuals: np.ndarray,
                recent_hours: Optional[np.ndarray] = None,
                future_hours: Optional[np.ndarray] = None) -> np.ndarray:
        """完整修正: LightGBM + AR 残差."""
        ar_pred = self.predict(recent_residuals, len(lgb_predictions),
                               recent_hours, future_hours)
        return lgb_predictions + ar_pred

    def fit_and_correct(self, y_true: np.ndarray, y_pred_lgb: np.ndarray,
                        horizon: int,
                        true_hours: Optional[np.ndarray] = None,
                        future_hours: Optional[np.ndarray] = None) -> Dict:
        """一站式: 拟合 + 修正."""
        residuals = y_true - y_pred_lgb
        self.fit(residuals, true_hours)

        recent = residuals[-self.order:]
        recent_h = true_hours[-self.order:] if true_hours is not None else None
        ar_pred = self.predict(recent, horizon, recent_h, future_hours)
        corrected = y_pred_lgb[-horizon:] + ar_pred

        return {
            "corrected": corrected,
            "ar_pred": ar_pred,
            "coefficients": self.coefficients.tolist() if self.coefficients is not None else [],
            "order": self.order,
            "hourly_models": list(self.hourly_models.keys()),
        }

    # ── 内部方法 ──

    @staticmethod
    def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
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
        p = len(r) - 1
        if p == 0:
            return np.array([])

        a = np.zeros(p)
        e = r[0]

        for k in range(p):
            lam = -np.dot(a[:k], r[k:0:-1]) - r[k + 1]
            if abs(e) < 1e-20:
                break
            lam /= e

            a[k] = lam
            a[:k] = a[:k] + lam * a[:k][::-1]
            e *= (1 - lam ** 2)

        return -a

    @staticmethod
    def _predict_in_sample(residuals: np.ndarray,
                           coefficients: np.ndarray,
                           order: int) -> np.ndarray:
        n = len(residuals)
        pred = np.zeros(n - order)

        for t in range(order, n):
            window = residuals[t - order:t]
            pred[t - order] = np.dot(coefficients[::-1], window)

        return pred
