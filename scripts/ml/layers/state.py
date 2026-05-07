"""state.py — 二分类层: 物理规则 + LGB 混合, 仅风电/光伏生效, 其他类型返回 1.0."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scripts.ml.layers.base import BaseLayer
import logging

logger = logging.getLogger(__name__)

VOLATILE_STATE_TYPES = {"风电", "光伏"}

# 物理规则参数
WIND_CUT_IN_CURVE = 0.05     # 风功率曲线 > 此值 = 已切入
WIND_RATED_CURVE = 0.8       # 风功率曲线 > 此值 = 近满发
SOLAR_WEAK_SR = 50           # W/m², 弱辐照阈值
SOLAR_STRONG_SR = 300        # W/m², 强辐照阈值
SOLAR_CLOUDY_CF = 0.1        # cloud_factor > 此值才算有日照
SOLAR_CLEAR_CF = 0.6         # cloud_factor > 此值算晴空


class StateLayer(BaseLayer):
    def __init__(self, noise_threshold_pct: float = 5.0):
        super().__init__("state")
        self.noise_threshold_pct = noise_threshold_pct
        self.noise_threshold = 0.0
        self.active = False
        self._is_wind = False
        self._is_solar = False

    def train(self, df: pd.DataFrame, target_type: str = "",
              feature_names: list = None, **kwargs) -> dict:
        self.active = any(kw in target_type for kw in VOLATILE_STATE_TYPES)
        self._is_wind = "风电" in target_type
        self._is_solar = "光伏" in target_type
        self.feature_names = feature_names or []

        if not self.active:
            self._trained = True
            return {"active": False, "reason": f"{target_type} 不需要 State 层"}

        y = df["value"].values
        self.noise_threshold = max(
            np.quantile(y[y > 0], 0.05) if (y > 0).any() else 1.0,
            np.quantile(y[y > 0], 0.01) if (y > 0).any() else 0.1
        ) if (y > 0).sum() > 10 else (self.noise_threshold_pct / 100.0)

        label = (y > self.noise_threshold).astype(int)
        pos_ratio = label.mean()
        if pos_ratio < 0.05 or pos_ratio > 0.95:
            self.active = False
            self._trained = True
            return {"active": False, "reason": f"类别不平衡 ({pos_ratio:.2%}), 跳过 State"}

        X = self._build_X(df)
        self.model = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, num_leaves=31,
            learning_rate=0.03, class_weight="balanced",
            random_state=42, verbose=-1,
        )
        self.model.fit(X, label)
        self._trained = True

        logger.info("State 训练完成: %s, pos_ratio=%.1f%%, threshold=%.3f, wind=%s, solar=%s",
                     target_type, pos_ratio * 100, self.noise_threshold,
                     self._is_wind, self._is_solar)
        return {"active": True, "pos_ratio": pos_ratio, "threshold": self.noise_threshold,
                "is_wind": self._is_wind, "is_solar": self._is_solar}

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.active:
            return np.ones(len(df))

        # LGB 模型概率
        if self.model is not None:
            model_prob = self._model_predict(df)
        else:
            model_prob = np.full(len(df), 0.5)

        # 物理规则地板
        physics_prob = self._physics_floor(df)
        prob = np.maximum(model_prob, physics_prob)

        # 光伏夜间强制归零 (物理确定性)
        if self._is_solar:
            night_mask = self._night_mask(df)
            prob[night_mask] = 0.0

        return np.clip(prob, 0, 1)

    def _model_predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_X(df)
        booster = getattr(self.model, "_Booster", None)
        if booster is not None:
            probs = booster.predict(X)
            if probs.ndim == 2:
                return probs[:, 1]
            return probs
        return self.model.predict_proba(X)[:, 1]

    def _physics_floor(self, df: pd.DataFrame) -> np.ndarray:
        """物理规则: 基于风速/辐照度给出最低发电概率.

        风电: wind_power_curve > CUT_IN → ≥0.3, > RATED → ≥0.8
        光伏: 白天 + solar_radiation > WEAK + cloud_factor > CLOUDY → ≥0.3
              白天 + solar_radiation > STRONG + cloud_factor > CLEAR → ≥0.7
              夜间 → 0
        """
        n = len(df)
        floor = np.zeros(n)

        # ── 风电规则 ──
        if self._is_wind and "wind_power_curve" in df.columns:
            wpc = df["wind_power_curve"].values
            # 切入风速以上 (风功率曲线 > 0.05)
            floor = np.maximum(floor, np.where(wpc > WIND_CUT_IN_CURVE, 0.30, 0))
            # 额定风速附近 (风功率曲线 > 0.8)
            floor = np.maximum(floor, np.where(wpc > WIND_RATED_CURVE, 0.80, 0))
            # 风速 lag 辅助: 如果1日前同时刻风功率曲线很高, 今天也有较大概率
            if "wind_speed_lag_1d" in df.columns and "wind_speed" in df.columns:
                ws = df["wind_speed"].fillna(0).values
                ws_lag = df["wind_speed_lag_1d"].fillna(0).values
                # 昨天+今天风速都 > 4m/s → 持续大风, 概率不应太低
                sustained = (ws > 4) & (ws_lag > 4)
                floor = np.maximum(floor, np.where(sustained, 0.40, 0))

        # ── 光伏规则 ──
        if self._is_solar:
            # 夜间强制归零
            if "is_daylight" in df.columns:
                is_day = df["is_daylight"].values > 0
            else:
                # fallback: 用 hour 判断
                hour_vals = df["hour"].values if "hour" in df.columns else np.full(n, 12)
                is_day = (hour_vals >= 6) & (hour_vals <= 18)

            if "solar_radiation" in df.columns and "cloud_factor" in df.columns:
                sr = df["solar_radiation"].fillna(0).values
                cf = df["cloud_factor"].fillna(0).values
                # 白天 + 有辐照 → 大概率发电
                floor = np.maximum(floor, np.where(
                    is_day & (sr > SOLAR_WEAK_SR) & (cf > SOLAR_CLOUDY_CF), 0.30, 0))
                # 白天 + 强辐照 + 晴空 → 几乎肯定发电
                floor = np.maximum(floor, np.where(
                    is_day & (sr > SOLAR_STRONG_SR) & (cf > SOLAR_CLEAR_CF), 0.70, 0))
            elif "solar_radiation" in df.columns:
                sr = df["solar_radiation"].fillna(0).values
                floor = np.maximum(floor, np.where(
                    is_day & (sr > SOLAR_WEAK_SR), 0.25, 0))
                floor = np.maximum(floor, np.where(
                    is_day & (sr > SOLAR_STRONG_SR), 0.60, 0))

        return floor

    def _night_mask(self, df: pd.DataFrame) -> np.ndarray:
        """光伏夜间 mask."""
        if "is_daylight" in df.columns:
            return df["is_daylight"].values <= 0
        hour_vals = df["hour"].values if "hour" in df.columns else np.full(len(df), 12)
        return (hour_vals < 6) | (hour_vals > 18)

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_names:
            return self._extract_features(df)
        return df.select_dtypes(include=[np.number]).fillna(0).values
