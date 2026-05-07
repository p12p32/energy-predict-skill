"""level.py — Level 层: LGB 回归 + 变换 + 物理分档, 风电/光伏仅用 value>0 训练."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scripts.ml.layers.base import BaseLayer
from scripts.ml.layers.transform import TransformSelector, TransformConfig
import logging

logger = logging.getLogger(__name__)

# 风电物理分档 (基于风功率曲线)
WIND_BINS = [0, 0.10, 0.30, 0.70, 1.01]  # 4档: 切入/低风/中风/额定
WIND_LABELS = [0, 1, 2, 3]
# 光伏物理分档 (基于 cloud_factor)
SOLAR_BINS = [0, 0.30, 0.70, 1.01]  # 3档: 阴天/多云/晴空
SOLAR_LABELS = [0, 1, 2]


class LevelLayer(BaseLayer):
    def __init__(self, transform_config: TransformConfig = None):
        super().__init__("level")
        self.transform_config = transform_config or TransformConfig("identity")
        self.selector = TransformSelector()
        self._is_wind = False
        self._is_solar = False
        self._augmented_cols = []

    def load(self, path: str) -> None:
        super().load(path)
        self._is_wind = self.metadata.get("is_wind", False)
        self._is_solar = self.metadata.get("is_solar", False)
        self._augmented_cols = self.metadata.get("_augmented_cols", [])

    def train(self, df: pd.DataFrame, target_type: str = "",
              feature_names: list = None, oof_mode: bool = False, **kwargs) -> dict:
        self.feature_names = feature_names or []
        self._is_wind = "风电" in target_type
        self._is_solar = "光伏" in target_type

        if not self.feature_names:
            X_df = df.select_dtypes(include=[np.number])
            exclude = {"dt", "value", "type", "province", "price"}
            self.feature_names = [c for c in X_df.columns if c not in exclude]

        y_raw = df["value"].dropna().values

        # 变换选择 (仅首次训练)
        if not oof_mode and self.transform_config.name == "identity":
            self.transform_config = self.selector.select(y_raw, target_type)
            logger.info("Level 变换: %s → %s", target_type, self.transform_config)

        # 风电/光伏: 仅用正值样本训练
        is_volatile = any(kw in target_type for kw in ("风电", "光伏"))
        train_mask = np.ones(len(df), dtype=bool)
        if is_volatile:
            train_mask = df["value"].values > 0
            if train_mask.sum() < 50:
                train_mask = np.ones(len(df), dtype=bool)

        df_train = df[train_mask]
        X, self._augmented_cols = self._build_training_matrix(df_train)
        y_t = self.selector.apply(df_train["value"].values, self.transform_config)

        # 样本权重: 强发电样本更高权重 (物理可信度更高)
        sample_weight = self._compute_sample_weights(df_train)

        self.model = lgb.LGBMRegressor(
            objective="regression", n_estimators=500, num_leaves=63,
            learning_rate=0.02, random_state=42, verbose=-1,
        )
        self.model.fit(X, y_t, sample_weight=sample_weight)
        self._trained = True

        self.metadata["is_volatile_train"] = is_volatile
        self.metadata["n_train"] = len(df_train)
        self.metadata["is_wind"] = self._is_wind
        self.metadata["is_solar"] = self._is_solar
        self.metadata["_augmented_cols"] = self._augmented_cols

        logger.info("Level 训练完成: n=%d, transform=%s, wind=%s, solar=%s, aug_cols=%d",
                     len(df_train), self.transform_config, self._is_wind, self._is_solar,
                     len(self._augmented_cols))
        return {"n_train": len(df_train), "transform": str(self.transform_config),
                "is_wind": self._is_wind, "is_solar": self._is_solar}

    def predict(self, df: pd.DataFrame, prob_on: np.ndarray = None, **kwargs) -> np.ndarray:
        X = self._build_prediction_matrix(df)
        gen_pred = self._predict_raw(X)
        if prob_on is not None:
            gen_pred = prob_on * gen_pred
        return gen_pred

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        t_pred = self.model.predict(X)
        return self.selector.inverse(t_pred, self.transform_config)

    # ─── 物理增强特征 ────────────────────────────────────────

    def _augment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为风电/光伏添加物理分档和交互特征."""
        df_aug = df.copy()
        added = []

        if self._is_wind:
            added += self._add_wind_features(df_aug)
        if self._is_solar:
            added += self._add_solar_features(df_aug)

        self._augmented_cols = added
        return df_aug

    def _add_wind_features(self, df: pd.DataFrame) -> list:
        added = []
        wpc = df["wind_power_curve"].values if "wind_power_curve" in df.columns else np.zeros(len(df))

        # 物理分档
        regime = pd.cut(wpc, bins=WIND_BINS, labels=WIND_LABELS, include_lowest=True)
        df["wind_regime"] = regime.astype(int)
        added.append("wind_regime")

        # 交互: 风功率曲线 × 历史出力
        if "value_lag_1d" in df.columns:
            lag = df["value_lag_1d"].fillna(0).values
            df["wind_curve_x_lag1d"] = wpc * lag
            added.append("wind_curve_x_lag1d")
        if "value_lag_7d" in df.columns:
            lag7 = df["value_lag_7d"].fillna(0).values
            df["wind_curve_x_lag7d"] = wpc * lag7
            added.append("wind_curve_x_lag7d")

        # 交互: 风向 × 风速 (地形效应)
        if "wind_speed" in df.columns and "wind_direction" in df.columns:
            ws = df["wind_speed"].fillna(0).values
            wd = df["wind_direction"].fillna(180).values
            df["wind_speed_x_dir_sin"] = ws * np.sin(np.radians(wd))
            df["wind_speed_x_dir_cos"] = ws * np.cos(np.radians(wd))
            added += ["wind_speed_x_dir_sin", "wind_speed_x_dir_cos"]

        # 风速变化趋势
        if "wind_speed_diff_1d" in df.columns:
            ws_diff = df["wind_speed_diff_1d"].fillna(0).values
            df["wind_ramping"] = np.abs(ws_diff)
            df["wind_ramp_direction"] = np.sign(ws_diff)
            added += ["wind_ramping", "wind_ramp_direction"]

        return added

    def _add_solar_features(self, df: pd.DataFrame) -> list:
        added = []
        cf = df["cloud_factor"].values if "cloud_factor" in df.columns else np.zeros(len(df))

        # 物理分档
        regime = pd.cut(cf, bins=SOLAR_BINS, labels=SOLAR_LABELS, include_lowest=True)
        df["solar_regime"] = regime.astype(int)
        added.append("solar_regime")

        # 交互: cloud_factor × solar_potential (有效辐照)
        if "solar_potential" in df.columns:
            sp = df["solar_potential"].fillna(0).values
            df["effective_solar"] = cf * sp
            added.append("effective_solar")

        # 交互: 晴空因子 × 昨日出力
        if "value_lag_1d" in df.columns:
            lag = df["value_lag_1d"].fillna(0).values
            df["solar_cloud_x_lag1d"] = cf * lag
            added.append("solar_cloud_x_lag1d")

        # 交互: 日照 × 温度折损 (高温降低光伏效率)
        if "solar_potential" in df.columns and "solar_efficiency" in df.columns:
            se = df["solar_efficiency"].fillna(1.0).values
            sp = df["solar_potential"].fillna(0).values
            df["solar_temp_derated"] = sp * se
            added.append("solar_temp_derated")

        # 辐照度变化趋势
        if "solar_radiation_diff_1d" in df.columns:
            sr_diff = df["solar_radiation_diff_1d"].fillna(0).values
            df["solar_ramping"] = np.abs(sr_diff)
            df["solar_ramp_direction"] = np.sign(sr_diff)
            added += ["solar_ramping", "solar_ramp_direction"]

        return added

    def _compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """强发电样本更高权重 (物理可信度高, 信号噪声比高)."""
        weights = np.ones(len(df))

        if self._is_wind and "wind_power_curve" in df.columns:
            wpc = df["wind_power_curve"].values
            # 权重: 1.0 + 2.0 × wpc, 范围 [1, 3]
            weights = 1.0 + 2.0 * np.clip(wpc, 0, 1)

        if self._is_solar and "cloud_factor" in df.columns:
            cf = df["cloud_factor"].values
            dl = df["is_daylight"].values if "is_daylight" in df.columns else np.ones(len(df))
            # 白天 + 晴空: 权重高; 阴天/夜间: 权重低
            weights = 1.0 + 2.0 * np.clip(cf * dl, 0, 1)

        return weights

    # ─── 矩阵构建 ──────────────────────────────────────────

    def _build_training_matrix(self, df: pd.DataFrame) -> tuple:
        """训练时: 增广特征 → 提取矩阵."""
        df_aug = self._augment_features(df)
        all_features = list(self.feature_names) + self._augmented_cols
        return self._extract_with_names(df_aug, all_features), self._augmented_cols

    def _build_prediction_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """预测时: 增广特征 → 提取矩阵."""
        df_aug = self._augment_features(df)
        all_features = list(self.feature_names) + self._augmented_cols
        return self._extract_with_names(df_aug, all_features)

    def _extract_with_names(self, df: pd.DataFrame, names: list) -> np.ndarray:
        X = np.zeros((len(df), len(names)), dtype=np.float64)
        for i, fn in enumerate(names):
            if fn in df.columns:
                col = pd.to_numeric(df[fn], errors="coerce").fillna(0).values
                X[:, i] = col
        return X
