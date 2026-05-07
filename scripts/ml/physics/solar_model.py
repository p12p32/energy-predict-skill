"""solar_model.py — 光伏出力物理模型: 辐照→功率.

Solar(t) = C × η(T) × GHI_norm(t) × cloud_trans(t)
  GHI_norm = clear_sky(doy, hour, lat) / 1361
  η(T)      = η₀ × (1 + β × (T − 25))
  cloud_trans = clip(α × cloud_factor + γ, 0, 1)

5个可解释参数: C(有效容量MW), η₀(基础效率), β(温度系数),
               α(云量斜率), γ(云量截距)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.optimize import minimize

from scripts.ml.physics.base import PhysicalModel, PhysicalModelConfig
from scripts.data.weather_features import WeatherFeatureEngineer

logger = logging.getLogger(__name__)

S0 = 1000.0  # STC 辐照度 W/m² (光伏板额定条件), 使 C ≈ 装机容量


class SolarParametricModel(PhysicalModel):
    def __init__(self, config: Optional[PhysicalModelConfig] = None):
        if config is None:
            config = PhysicalModelConfig()
        super().__init__("solar", config)
        self._init_defaults()

    def _init_defaults(self):
        self.params = {
            "C": self.config.capacity_mw or 1000.0,
            "eta0": 0.18,
            "beta": -0.004,
            "alpha": 1.0,
            "gamma": 0.0,
        }

    # ── 方程评估 ──

    def _clear_sky_array(self, weather: pd.DataFrame) -> np.ndarray:
        """计算每个时间点的晴空辐照度."""
        lat = self.config.lat
        dts = pd.to_datetime(weather["dt"])
        doy = dts.dt.dayofyear.values
        hrs = dts.dt.hour.values + dts.dt.minute.values / 60.0
        result = np.zeros(len(weather))
        for i in range(len(weather)):
            result[i] = WeatherFeatureEngineer._clear_sky_irradiance(
                lat, int(doy[i]), float(hrs[i]))
        return result

    def _compute_ghi_norm(self, weather: pd.DataFrame) -> np.ndarray:
        """归一化晴空辐照度: GHI_norm = clear_sky / S0."""
        clear = self._clear_sky_array(weather)
        return clear / S0

    def _compute_eta(self, weather: pd.DataFrame) -> np.ndarray:
        """温度修正效率: η(T) = η₀ × (1 + β × (T − 25))."""
        T = weather["temperature"].values if "temperature" in weather.columns else np.full(len(weather), 25.0)
        return self.params["eta0"] * (1.0 + self.params["beta"] * (T - 25.0))

    def _compute_cloud_trans(self, weather: pd.DataFrame) -> np.ndarray:
        """云量透过率: clip(α × cf + γ, 0, 1)."""
        cf = weather["cloud_factor"].values if "cloud_factor" in weather.columns else np.zeros(len(weather))
        return np.clip(self.params["alpha"] * cf + self.params["gamma"], 0.0, 1.0)

    def predict_physics(self, weather: pd.DataFrame) -> np.ndarray:
        """纯物理预测."""
        ghi_norm = self._compute_ghi_norm(weather)
        eta = self._compute_eta(weather)
        cloud = self._compute_cloud_trans(weather)
        return self.params["C"] * eta * ghi_norm * cloud

    def get_equation(self) -> str:
        p = self.params
        return (f"Solar = {p['C']:.0f}MW × {p['eta0']:.3f} "
                f"× (1 {p['beta']:+.4f}×(T−25)) "
                f"× GHI_norm "
                f"× clip({p['alpha']:.3f}×cf + {p['gamma']:.3f}, 0, 1)")

    # ── 拟合 ──

    def fit(self, weather: pd.DataFrame, actuals: pd.Series, **kwargs) -> Dict:
        logger.info("SolarModel 拟合: %d 样本, lat=%.1f", len(weather), self.config.lat)

        if self.config.model_type == "lightgbm":
            return self._fit_lgbm(weather, actuals)

        try:
            return self._fit_parametric(weather, actuals)
        except Exception as e:
            logger.warning("参数拟合失败 (%s), fallback LGBM", e)
            self.config.model_type = "lightgbm"
            return self._fit_lgbm(weather, actuals)

    def _fit_parametric(self, weather: pd.DataFrame, actuals: pd.Series) -> Dict:
        """两阶段参数拟合."""
        y = actuals.values.astype(float)
        n = len(y)

        # 晴天识别: 日照时段 cloud_factor > 0.7
        cf = weather["cloud_factor"].values if "cloud_factor" in weather.columns else np.zeros(n)
        is_day = self._compute_ghi_norm(weather) > 0.05
        is_clear = is_day & (cf > 0.7)
        clear_frac = is_clear.sum() / max(is_day.sum(), 1)

        if clear_frac < 0.05:
            # 晴天太少, 降级为全量拟合
            logger.info("晴天比例仅 %.1f%%, 全量拟合", clear_frac * 100)
            clear_mask = is_day  # all daytime
        else:
            clear_mask = is_clear

        # Stage 1: 固定 α=1, γ=0, 拟合 C, η₀, β
        def objective_stage1(params):
            C, eta0, beta = params
            self.params = {"C": C, "eta0": eta0, "beta": beta,
                           "alpha": 1.0, "gamma": 0.0}
            pred = self.predict_physics(weather)
            err = pred[clear_mask] - y[clear_mask]
            return np.sum(err ** 2)

        # 初始猜测: C 需要补偿 GHI_norm(0~0.7) × eta(0.1~0.2) ≈ 0.05-0.14
        if self.config.capacity_mw:
            C0 = self.config.capacity_mw
        else:
            daytime_actuals = y[is_day] if is_day.sum() > 0 else y
            p99 = float(np.percentile(daytime_actuals, 99)) if len(daytime_actuals) > 0 else 1000.0
            # GHI_norm≈0.9 (peak clear_sky/1000), eta≈0.18, cloud≈1.0 → 0.162
            C0 = p99 / 0.16

        r1 = minimize(objective_stage1, x0=[C0, 0.18, -0.004],
                      bounds=[(C0 * 0.1, C0 * 10.0), (0.02, 0.50), (-0.05, 0.01)],
                      method="L-BFGS-B")
        self.params["C"] = float(r1.x[0])
        self.params["eta0"] = float(r1.x[1])
        self.params["beta"] = float(r1.x[2])

        # Stage 2: 固定 C, η₀, β, 拟合 α, γ
        def objective_stage2(params):
            alpha, gamma = params
            self.params["alpha"] = alpha
            self.params["gamma"] = gamma
            pred = self.predict_physics(weather)
            err = pred - y
            return np.sum(err ** 2)

        r2 = minimize(objective_stage2, x0=[1.0, 0.0],
                      bounds=[(0.01, 5.0), (-0.5, 0.5)],
                      method="L-BFGS-B")
        self.params["alpha"] = float(r2.x[0])
        self.params["gamma"] = float(r2.x[1])

        # 全量评估
        pred = self.predict_physics(weather)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        r2 = float(1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-10))

        self.metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        self._trained = True

        logger.info("Solar 拟合完成: %s", self.get_equation())
        logger.info("  MAE=%.1f MW, RMSE=%.1f MW, R²=%.3f", mae, rmse, r2)
        return dict(self.metrics)

    def _fit_lgbm(self, weather: pd.DataFrame, actuals: pd.Series) -> Dict:
        """LGBM fallback: 只用气象特征, 不用交叉类型滞后."""
        import lightgbm as lgb

        y = actuals.values.astype(float)
        feat_cols = ["solar_radiation", "temperature", "cloud_factor",
                     "humidity", "hour", "day_of_week", "month"]

        available = [c for c in feat_cols if c in weather.columns]
        if not available:
            # 补充时间列
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
        logger.info("Solar(LGBM) 拟合完成: MAE=%.1f, RMSE=%.1f", mae, rmse)
        return dict(self.metrics)

    def predict(self, weather: pd.DataFrame) -> np.ndarray:
        if self.config.model_type == "lightgbm" and self._ml_fallback is not None:
            return self._predict_with_fallback(weather)
        return self.predict_physics(weather)

    def _predict_with_fallback(self, weather: pd.DataFrame) -> np.ndarray:
        feat_cols = ["solar_radiation", "temperature", "cloud_factor",
                     "humidity", "hour", "day_of_week", "month"]
        available = [c for c in feat_cols if c in weather.columns]
        if "hour" not in weather.columns:
            weather = weather.copy()
            dts = pd.to_datetime(weather["dt"])
            weather["hour"] = dts.dt.hour
            weather["day_of_week"] = dts.dt.dayofweek
            weather["month"] = dts.dt.month
        available = [c for c in feat_cols if c in weather.columns]
        return self._ml_fallback.predict(weather[available].fillna(0).values.astype(float))
