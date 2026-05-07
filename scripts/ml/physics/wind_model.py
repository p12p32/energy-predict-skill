"""wind_model.py — 风电出力物理模型: 风速→功率.

Wind(t) = C × power_curve(v_eff)
  v_eff = v × directional_factor(θ)
  power_curve:
    0           if v < v_cut_in
    0           if v > v_cut_out
    1.0         if v >= v_rated
    cubic       otherwise

5-7个可解释参数: C(容量MW), v_cut_in, v_rated, v_cut_out + 方向因子
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.optimize import minimize

from scripts.ml.physics.base import PhysicalModel, PhysicalModelConfig

logger = logging.getLogger(__name__)


def _power_curve(v: np.ndarray, v_cut_in: float, v_rated: float,
                 v_cut_out: float) -> np.ndarray:
    """S型风功率曲线 (归一化 0-1)."""
    result = np.zeros(len(v))
    # cut-in < v < rated: cubic
    mid = (v >= v_cut_in) & (v < v_rated)
    result[mid] = ((v[mid] - v_cut_in) / (v_rated - v_cut_in)) ** 3
    # rated <= v <= cut_out: full power
    full = (v >= v_rated) & (v <= v_cut_out)
    result[full] = 1.0
    # v > cut_out: zero (already 0)
    return result


class WindParametricModel(PhysicalModel):
    def __init__(self, config: Optional[PhysicalModelConfig] = None):
        if config is None:
            config = PhysicalModelConfig()
        super().__init__("wind", config)
        self._init_defaults()

    def _init_defaults(self):
        self.params = {
            "C": self.config.capacity_mw or 500.0,
            "v_cut_in": 3.0,
            "v_rated": 12.0,
            "v_cut_out": 25.0,
        }
        self._dir_factors: Dict[str, float] = {}
        self._use_directional = False

    # ── 方程评估 ──

    def predict_physics(self, weather: pd.DataFrame) -> np.ndarray:
        v = weather["wind_speed"].values if "wind_speed" in weather.columns else np.zeros(len(weather))

        if self._use_directional and "wind_direction" in weather.columns:
            # 方向修正: 按扇区乘因子
            v_eff = self._apply_directional(v, weather["wind_direction"].values)
        else:
            v_eff = v

        curve = _power_curve(v_eff, self.params["v_cut_in"],
                             self.params["v_rated"], self.params["v_cut_out"])
        return self.params["C"] * curve

    def _apply_directional(self, speed: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """对每个风向扇区应用方向修正因子."""
        result = speed.copy()
        for sector, factor in self._dir_factors.items():
            mask = self._sector_mask(direction, sector)
            result[mask] *= factor
        return result

    @staticmethod
    def _sector_mask(direction: np.ndarray, sector: str) -> np.ndarray:
        """风向扇区掩码. sector: N/NE/E/SE/S/SW/W/NW."""
        sectors = {"N": (-22.5, 22.5), "NE": (22.5, 67.5),
                   "E": (67.5, 112.5), "SE": (112.5, 157.5),
                   "S": (157.5, 202.5), "SW": (202.5, 247.5),
                   "W": (247.5, 292.5), "NW": (292.5, 337.5)}
        lo, hi = sectors.get(sector, (-22.5, 22.5))
        if lo < 0:
            return (direction >= lo + 360) | (direction < hi)
        if hi > 360:
            return (direction >= lo) | (direction < hi - 360)
        return (direction >= lo) & (direction < hi)

    def get_equation(self) -> str:
        p = self.params
        eq = (f"Wind = {p['C']:.0f}MW × power_curve(v, "
              f"cut_in={p['v_cut_in']:.1f}, rated={p['v_rated']:.1f}, "
              f"cut_out={p['v_cut_out']:.1f})")
        if self._use_directional and self._dir_factors:
            eq += f"\n  directional: {self._dir_factors}"
        return eq

    # ── 拟合 ──

    def fit(self, weather: pd.DataFrame, actuals: pd.Series, **kwargs) -> Dict:
        logger.info("WindModel 拟合: %d 样本", len(weather))

        if self.config.model_type == "lightgbm":
            return self._fit_lgbm(weather, actuals)

        try:
            return self._fit_parametric(weather, actuals)
        except Exception as e:
            logger.warning("参数拟合失败 (%s), fallback LGBM", e)
            self.config.model_type = "lightgbm"
            return self._fit_lgbm(weather, actuals)

    def _fit_parametric(self, weather: pd.DataFrame, actuals: pd.Series) -> Dict:
        y = actuals.values.astype(float)

        # C 初始估计: 99分位数 × 1.1
        if self.config.capacity_mw:
            C0 = self.config.capacity_mw
        else:
            C0 = float(np.percentile(y, 99)) * 1.1 if len(y) > 0 else 500.0
            C0 = max(C0, 50.0)

        # 网格粗搜
        v_ci_candidates = [2.5, 3.0, 3.5, 4.0]
        v_r_candidates = [10.0, 11.0, 12.0, 13.0, 14.0]
        v_co_candidates = [22.0, 25.0, 28.0, 30.0]

        v = weather["wind_speed"].values if "wind_speed" in weather.columns else np.zeros(len(y))

        best_mse = float("inf")
        best = (3.0, 12.0, 25.0)
        for ci in v_ci_candidates:
            for vr in v_r_candidates:
                if vr <= ci:
                    continue
                for co in v_co_candidates:
                    if co <= vr:
                        continue
                    curve = _power_curve(v, ci, vr, co)
                    # 拟合 C
                    denom = np.sum(curve ** 2) + 1e-10
                    C_opt = np.sum(curve * y) / denom
                    pred = C_opt * curve
                    mse = float(np.mean((pred - y) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best = (ci, vr, co)
                        best_C = C_opt

        self.params["v_cut_in"] = best[0]
        self.params["v_rated"] = best[1]
        self.params["v_cut_out"] = best[2]
        self.params["C"] = max(best_C, 10.0)

        # L-BFGS-B 精调 (带物理约束)
        def objective(params):
            C, ci, vr, co = params
            p = {"C": C, "v_cut_in": ci, "v_rated": vr, "v_cut_out": co}
            self.params = p
            pred = self.predict_physics(weather)
            mse = float(np.mean((pred - y) ** 2))
            # 物理约束惩罚: rated 必须 > cut_in, cut_out 必须 > rated
            if vr <= ci + 0.5:
                mse += 1e6 * (ci + 0.5 - vr) ** 2
            if co <= vr + 1.0:
                mse += 1e6 * (vr + 1.0 - co) ** 2
            return mse

        # 物理合理边界
        v_max = float(np.percentile(v, 99)) if len(v) > 0 else 20.0
        bounds = [
            (best_C * 0.5, best_C * 2.0),          # C
            (0.5, min(6.0, v_max * 0.4)),          # v_cut_in: 0.5~6 m/s
            (max(5.0, best[1] * 0.6), min(18.0, v_max * 0.8)),  # v_rated: 5~18 m/s
            (max(15.0, best[2] * 0.6), min(35.0, v_max * 1.2)),  # v_cut_out: 15~35 m/s
        ]

        r = minimize(objective,
                     x0=[self.params["C"], best[0], best[1], best[2]],
                     method="L-BFGS-B",
                     bounds=bounds)
        self.params["C"] = float(r.x[0])
        self.params["v_cut_in"] = float(r.x[1])
        self.params["v_rated"] = float(r.x[2])
        self.params["v_cut_out"] = float(r.x[3])

        # 可选: 方向因子拟合
        if "wind_direction" in weather.columns and len(y) >= 96 * 7:
            self._fit_directional(weather, y)

        # 评估
        pred = self.predict_physics(weather)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        r2 = float(1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-10))

        # 参数模型质量检查: R²<0 → 自动回退 LGBM
        if r2 < 0:
            logger.info("参数模型 R²=%.3f < 0, 回退 LGBM", r2)
            self.config.model_type = "lightgbm"
            return self._fit_lgbm(weather, actuals)

        self.metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        self._trained = True

        logger.info("Wind 拟合完成: %s", self.get_equation())
        logger.info("  MAE=%.1f MW, RMSE=%.1f MW, R²=%.3f", mae, rmse, r2)
        return dict(self.metrics)

    def _fit_directional(self, weather: pd.DataFrame, y: np.ndarray):
        """按风向扇区拟合修正因子."""
        direction = weather["wind_direction"].values
        pred_base = self.predict_physics(weather)
        eps = 1e-6

        sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for sector in sectors:
            mask = self._sector_mask(direction, sector)
            if mask.sum() < 50:
                continue
            ratio = np.median(y[mask]) / max(np.median(pred_base[mask]), eps)
            ratio = np.clip(ratio, 0.5, 1.5)
            if abs(ratio - 1.0) > 0.05:
                self._dir_factors[sector] = float(ratio)

        if self._dir_factors:
            self._use_directional = True
            logger.info("  方向因子: %s", self._dir_factors)

    def _fit_lgbm(self, weather: pd.DataFrame, actuals: pd.Series) -> Dict:
        import lightgbm as lgb

        y = actuals.values.astype(float)
        feat_cols = ["wind_speed", "wind_direction", "temperature",
                     "pressure", "hour", "day_of_week", "month"]

        available = [c for c in feat_cols if c in weather.columns]
        if not available:
            dts = pd.to_datetime(weather["dt"])
            weather = weather.copy()
            weather["hour"] = dts.dt.hour
            weather["day_of_week"] = dts.dt.dayofweek
            weather["month"] = dts.dt.month
            available = [c for c in feat_cols if c in weather.columns]

        X = weather[available].fillna(0).values.astype(float)
        n_val = max(96, int(len(y) * 0.15))

        self._ml_fallback = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=0.03,
            random_state=42, verbose=-1,
        )
        self._ml_fallback.fit(X[:-n_val], y[:-n_val],
                              eval_set=[(X[-n_val:], y[-n_val:])],
                              eval_metric="l1")

        pred = self._ml_fallback.predict(weather[available].fillna(0).values.astype(float))
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

        self.metrics = {"mae": mae, "rmse": rmse, "r2": None}
        self._trained = True
        logger.info("Wind(LGBM) 拟合完成: MAE=%.1f, RMSE=%.1f", mae, rmse)
        return dict(self.metrics)

    def predict(self, weather: pd.DataFrame) -> np.ndarray:
        if self.config.model_type == "lightgbm" and self._ml_fallback is not None:
            return self._predict_with_fallback(weather)
        return self.predict_physics(weather)

    def _predict_with_fallback(self, weather: pd.DataFrame) -> np.ndarray:
        feat_cols = ["wind_speed", "wind_direction", "temperature",
                     "pressure", "hour", "day_of_week", "month"]
        available = [c for c in feat_cols if c in weather.columns]
        if "hour" not in weather.columns:
            weather = weather.copy()
            dts = pd.to_datetime(weather["dt"])
            weather["hour"] = dts.dt.hour
            weather["day_of_week"] = dts.dt.dayofweek
            weather["month"] = dts.dt.month
        available = [c for c in feat_cols if c in weather.columns]
        return self._ml_fallback.predict(weather[available].fillna(0).values.astype(float))
