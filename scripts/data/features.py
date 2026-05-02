"""feature_store.py — 特征工程 (type 三段式 + pred_error 特征 + 三层交叉)"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from scripts.core.config import parse_type, TypeInfo, get_cross_type_rules
from scripts.data.holidays import add_holiday_features, add_cyclical_features, add_deep_calendar_features
from scripts.data.quality import DataQuality
from scripts.data.weather_features import WeatherFeatureEngineer
from scripts.data.fetcher import WEATHER_ALL_COLS

PREDICTION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS energy_predictions (
    dt              DATETIME      NOT NULL,
    province        VARCHAR(32)   NOT NULL,
    type            VARCHAR(64)   NOT NULL,
    p10             DOUBLE        COMMENT 'P10分位数',
    p50             DOUBLE        COMMENT 'P50分位数（中位数）',
    p90             DOUBLE        COMMENT 'P90分位数',
    model_version   VARCHAR(64)   COMMENT '模型版本',
    created_at      DATETIME      DEFAULT CURRENT_TIMESTAMP
)
DUPLICATE KEY (dt, province, type)
DISTRIBUTED BY HASH (province) BUCKETS 8;
"""

FEATURE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS energy_feature_store (
    dt              DATETIME      NOT NULL,
    province        VARCHAR(32)   NOT NULL,
    type            VARCHAR(64)   NOT NULL,
    sub_type        VARCHAR(32)   COMMENT '子类型: 风电|水电|光伏|火电|...',
    value_type      VARCHAR(8)    COMMENT '实际|预测',
    value           DOUBLE        COMMENT '原始值',
    price           DOUBLE        COMMENT '电价',
    hour            TINYINT       COMMENT '小时(0-23)',
    day_of_week     TINYINT       COMMENT '星期(0=周一)',
    day_of_month    TINYINT       COMMENT '日(1-31)',
    month           TINYINT       COMMENT '月(1-12)',
    is_weekend      BOOLEAN       COMMENT '是否周末',
    season          TINYINT       COMMENT '季节(1=春,2=夏,3=秋,4=冬)',
    temperature     DOUBLE        COMMENT '温度(°C)',
    humidity        DOUBLE        COMMENT '相对湿度(%)',
    wind_speed      DOUBLE        COMMENT '风速(m/s)',
    wind_direction  DOUBLE        COMMENT '风向(°)',
    solar_radiation DOUBLE        COMMENT '太阳辐射(W/m²)',
    precipitation   DOUBLE        COMMENT '降水量(mm)',
    pressure        DOUBLE        COMMENT '气压(hPa)',
    value_lag_1d    DOUBLE        COMMENT '一天前同期值',
    value_lag_7d    DOUBLE        COMMENT '七天前同期值',
    value_rolling_mean_24h DOUBLE  COMMENT '24小时滑动均值',
    value_diff_1d   DOUBLE        COMMENT '日环比变化',
    value_diff_7d   DOUBLE        COMMENT '周环比变化',
    is_holiday      BOOLEAN       COMMENT '是否节假日',
    is_work_weekend BOOLEAN       COMMENT '是否调休工作日',
    days_to_holiday TINYINT       COMMENT '距下一节假日天数',
    days_from_holiday TINYINT     COMMENT '距上一节假日天数',
    hour_sin        DOUBLE        COMMENT '小时sin编码',
    hour_cos        DOUBLE        COMMENT '小时cos编码',
    dow_sin         DOUBLE        COMMENT '星期sin编码',
    dow_cos         DOUBLE        COMMENT '星期cos编码',
    month_sin       DOUBLE        COMMENT '月sin编码',
    month_cos       DOUBLE        COMMENT '月cos编码',
    quality_flag    TINYINT       COMMENT '数据质量标记(0=正常,1=疑似,2=异常)',
    pred_error            DOUBLE        COMMENT '历史预测误差 (actual - p50)',
    pred_error_lag_1d     DOUBLE        COMMENT '一天前同期预测误差',
    pred_error_lag_7d     DOUBLE        COMMENT '七天前同期预测误差',
    pred_error_bias_24h   DOUBLE        COMMENT '24h系统性偏差',
    pred_error_std_24h    DOUBLE        COMMENT '24h误差波动',
    pred_error_trend      DOUBLE        COMMENT '误差趋势变化',
    interval_coverage     DOUBLE        COMMENT '区间覆盖率',
    coverage_rate_24h     DOUBLE        COMMENT '24h平均覆盖',
	    pred_error_hour_bias  DOUBLE        COMMENT '该时点历史平均误差 (多点位)',
	    pred_error_weekend    DOUBLE        COMMENT '周末vs工作日误差差',
	    pred_error_holiday    DOUBLE        COMMENT '节假日误差放大系数',
	    pred_error_x_temp     DOUBLE        COMMENT '误差×温度交互',
	    pred_error_x_wind     DOUBLE        COMMENT '误差×风速交互',
	    pred_error_autocorr   DOUBLE        COMMENT '误差1天自相关',
	    pred_error_regime     DOUBLE        COMMENT '持续偏差方向 (+1高估/-1低估/0正常)',
    INDEX idx_province_dt (province, dt) USING BITMAP
)
DUPLICATE KEY (dt, province, type)
DISTRIBUTED BY HASH (province) BUCKETS 16;
"""

STRATEGY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS strategy_knowledge (
    strategy_hash   VARCHAR(64)   PRIMARY KEY,
    strategy_desc   STRING,
    applied_count   INT           DEFAULT 0,
    success_count   INT           DEFAULT 0,
    avg_improvement DOUBLE        DEFAULT 0,
    best_scenario   VARCHAR(256)  DEFAULT '',
    worst_scenario  VARCHAR(256)  DEFAULT '',
    last_applied    DATETIME,
    last_effect     DOUBLE        DEFAULT 0,
    retired         BOOLEAN       DEFAULT FALSE
)
UNIQUE KEY (strategy_hash)
DISTRIBUTED BY HASH (strategy_hash) BUCKETS 4;
"""


class FeatureEngineer:
    LAG_PERIODS = {
        # 1 天 (96 步)
        "value_lag_1d": 96,
        "value_diff_1d": 96,
        # 2 天
        "value_lag_2d": 192,
        # 3 天
        "value_lag_3d": 288,
        # 4 天
        "value_lag_4d": 384,
        # 5 天
        "value_lag_5d": 480,
        # 6 天
        "value_lag_6d": 576,
        # 7 天 (672 步)
        "value_lag_7d": 672,
        "value_diff_7d": 672,
        # 14 天
        "value_lag_14d": 1344,
        # 21 天
        "value_lag_21d": 2016,
        # 28 天 (月周期)
        "value_lag_28d": 2688,
    }

    # 多尺度滚动窗口 (步数)
    ROLLING_WINDOWS = {
        "value_rolling_mean_6h": (24, "mean"),
        "value_rolling_mean_12h": (48, "mean"),
        "value_rolling_mean_24h": (96, "mean"),
        "value_rolling_mean_48h": (192, "mean"),
        "value_rolling_mean_7d": (672, "mean"),
        "value_rolling_std_24h": (96, "std"),
        "value_rolling_std_7d": (672, "std"),
        "value_rolling_max_24h": (96, "max"),
        "value_rolling_min_24h": (96, "min"),
        "value_rolling_range_24h": (96, "range"),
    }

    PRED_ERROR_LAGS = {
        "pred_error_lag_1d": 96,
        "pred_error_lag_7d": 672,
    }

    def build_features_from_raw(self, raw: pd.DataFrame,
                                value_type_filter: str = "实际",
                                weather_df: pd.DataFrame = None) -> pd.DataFrame:
        """从原始数据构建特征.

        Args:
            raw: 原始数据 DataFrame
            value_type_filter: 只对此 value_type 的数据计算特征
            weather_df: 外部气象数据 (15min 粒度, 含 dt/province 列),
                        若提供则合并到 raw, 覆盖原有气象列
        """
        df = raw.copy()

        # ── 合并外部气象数据 ──
        if weather_df is not None and not weather_df.empty:
            weather_cols = [c for c in weather_df.columns
                           if c not in ("dt", "province") and c in WEATHER_ALL_COLS]
            if weather_cols:
                w_subset = weather_df[["dt", "province"] + weather_cols].copy()
                if "dt" in df.columns and "province" in df.columns:
                    df = df.merge(w_subset, on=["dt", "province"], how="left",
                                  suffixes=("", "_w"))
                    # 覆盖原有气象列
                    for col in weather_cols:
                        w_col = col + "_w" if col + "_w" in df.columns else col
                        if w_col in df.columns and w_col != col:
                            df[col] = df[w_col]
                            df.drop(columns=[w_col], inplace=True, errors="ignore")

        df = df.sort_values(["province", "type", "dt"]).reset_index(drop=True)

        # ── type 三段式解析 ──
        type_infos = df["type"].apply(lambda t: parse_type(str(t)))
        df["sub_type"] = type_infos.apply(lambda ti: ti.sub or "")
        df["value_type"] = type_infos.apply(lambda ti: ti.value_type)

        # ── 数据质量检测 ──
        dq = DataQuality()
        df = dq.detect(df, "value")

        # ── 时间特征 ──
        df["hour"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek
        df["day_of_month"] = df["dt"].dt.day
        df["month"] = df["dt"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        df["season"] = df["month"].apply(
            lambda m: 1 if m in [3, 4, 5] else
                      2 if m in [6, 7, 8] else
                      3 if m in [9, 10, 11] else 4
        )

        # ── 气象缺失填充 (季节+小时统计均值, 而非 0) ──
        weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction",
                       "solar_radiation", "precipitation", "pressure"]
        for col in weather_cols:
            if col in df.columns and df[col].isna().any():
                grp = df.groupby(["season", "hour"])[col].transform("mean")
                df[col] = df[col].fillna(grp)
                df[col] = df[col].fillna(df[col].mean())

        # ── 深度天气特征 ──
        wfe = WeatherFeatureEngineer()
        df = wfe.transform(df)

        # ── 节假日 + 深度日历 ──
        df = add_holiday_features(df)
        df = add_deep_calendar_features(df)

        # ── 周期编码 ──
        df = add_cyclical_features(df)

        # ── 交互特征 ──
        df["peak_valley"] = df["hour"].apply(
            lambda h: 2 if h in [8, 9, 10, 11, 17, 18, 19, 20] else
                      1 if h in [12, 13, 14, 21, 22] else 0
        )
        df["weekend_hour"] = df["is_weekend"].astype(int) * df["hour"]
        df["dow_hour"] = df["day_of_week"] * 24 + df["hour"]

        # ── 滞后特征 (只在 实际 数据上计算) ──
        actual_mask = df["value_type"] == value_type_filter
        for group_key, group_df in df[actual_mask].groupby(["province", "type"]):
            idx = group_df.index
            for col_name, shift_n in self.LAG_PERIODS.items():
                if "diff" in col_name:
                    df.loc[idx, col_name] = group_df["value"].diff(shift_n)
                else:
                    df.loc[idx, col_name] = group_df["value"].shift(shift_n)

            # 多尺度滚动统计
            for feat_name, (win_size, win_func) in self.ROLLING_WINDOWS.items():
                if feat_name == "value_rolling_range_24h":
                    continue  # 后面单独算
                if win_func == "std":
                    df.loc[idx, feat_name] = (
                        group_df["value"].rolling(window=win_size, min_periods=1).std().values
                    )
                elif win_func == "mean":
                    df.loc[idx, feat_name] = (
                        group_df["value"].rolling(window=win_size, min_periods=1).mean().values
                    )
                elif win_func == "max":
                    df.loc[idx, feat_name] = (
                        group_df["value"].rolling(window=win_size, min_periods=1).max().values
                    )
                elif win_func == "min":
                    df.loc[idx, feat_name] = (
                        group_df["value"].rolling(window=win_size, min_periods=1).min().values
                    )
                elif win_func == "range":
                    roll = group_df["value"].rolling(window=win_size, min_periods=1)
                    df.loc[idx, feat_name] = (roll.max() - roll.min()).values

        # 滞后交互特征 ──
        if "value_lag_7d" in df.columns:
            df["weekend_x_lag7d"] = df["is_weekend"].astype(int) * df["value_lag_7d"]
        if "hour" in df.columns and "value_lag_1d" in df.columns:
            df["hour_x_lag1d"] = df["hour"] * df["value_lag_1d"]

        # ── 波动率特征 (只在 实际 数据上计算) ──
        for group_key, group_df in df[actual_mask].groupby(["province", "type"]):
            idx = group_df.index
            df.loc[idx, "value_rolling_std_24h"] = (
                group_df["value"].rolling(window=96, min_periods=1).std().values
            )
            df.loc[idx, "value_rolling_max_24h"] = (
                group_df["value"].rolling(window=96, min_periods=1).max().values
            )
            df.loc[idx, "value_rolling_min_24h"] = (
                group_df["value"].rolling(window=96, min_periods=1).min().values
            )
            df.loc[idx, "value_range_24h"] = (
                df.loc[idx, "value_rolling_max_24h"] - df.loc[idx, "value_rolling_min_24h"]
            )

        # ── 滞后交互特征 ──
        if "value_lag_7d" in df.columns:
            df["weekend_x_lag7d"] = df["is_weekend"].astype(int) * df["value_lag_7d"]
        if "hour" in df.columns and "value_lag_1d" in df.columns:
            df["hour_x_lag1d"] = df["hour"] * df["value_lag_1d"]
        # 周/月周期差异
        if "value_lag_7d" in df.columns and "value_lag_28d" in df.columns:
            df["weekly_vs_monthly"] = df["value_lag_7d"] - df["value_lag_28d"]

        # ── 天气×时间交互 (增强版) ──
        if "temperature" in df.columns and "hour" in df.columns:
            df["temp_x_hour"] = df["temperature"] * df["hour"]
            if "peak_valley" in df.columns:
                df["temp_x_peak"] = df["temperature"] * df["peak_valley"]
            if "is_weekend" in df.columns:
                df["temp_x_weekend"] = df["temperature"] * df["is_weekend"].astype(int)
        if "wind_speed" in df.columns and "season" in df.columns:
            df["wind_x_season"] = df["wind_speed"] * df["season"]
        if "temperature" in df.columns and "humidity" in df.columns:
            df["temp_x_humidity"] = df["temperature"] * df["humidity"] / 100
        if "solar_radiation" in df.columns and "season" in df.columns:
            df["solar_x_season"] = df["solar_radiation"] * df["season"]
        if "precipitation" in df.columns and "temperature" in df.columns:
            df["rain_x_temp"] = df["precipitation"] * df["temperature"]

        # ── 风电专项特征 ──
        if "wind_speed_10m" in df.columns:
            ws = df["wind_speed_10m"]
            # 风切变指数 (10m vs 100m)
            if "wind_speed_100m" in df.columns:
                ws100 = df["wind_speed_100m"]
                df["wind_shear"] = np.where(
                    ws > 0.1,
                    np.log(np.maximum(ws100, 0.01) / np.maximum(ws, 0.01)) / np.log(10),
                    0.0
                )
            # 阵风系数
            if "wind_gusts" in df.columns:
                df["gust_factor"] = np.where(
                    ws > 0.5,
                    df["wind_gusts"] / ws,
                    0.0
                )
            # 风速立方 (风能 ∝ v³)
            df["wind_power"] = ws ** 3
            # 有效风速标记 (切出风速以上, 切入风速以下)
            df["wind_effective"] = ((ws >= 3) & (ws <= 25)).astype(float) * ws
            # 风向稳定性 (24h 内风向标准差)
            if "wind_direction_10m" in df.columns:
                wd = df["wind_direction_10m"]
                # 风向环形标准差
                sin_wd = np.sin(np.radians(wd))
                cos_wd = np.cos(np.radians(wd))
                df["wind_dir_var"] = 1.0 - np.sqrt(sin_wd ** 2 + cos_wd ** 2)

        # ── 光伏专项特征 ──
        if "shortwave_radiation" in df.columns:
            ghi = df["shortwave_radiation"]
            # 日间活跃标记 (核心特征: 让模型学会夜间出力=0)
            df["solar_active"] = (df["hour"].isin(range(6, 20)) & (ghi > 10)).astype(float)
            # 太阳高度角近似 (用 hour 余弦)
            if "hour" in df.columns:
                # 简化太阳高度角: 正午最高(1), 夜间最低(0)
                df["solar_elevation"] = np.maximum(np.cos(np.abs(df["hour"] - 12) / 12 * np.pi), 0) ** 2
            # ── 光伏物理基准线 (GHI → 出力的非线性映射) ──
            # 理论: 出力 = 装机容量 × GHI/1000 × PR × 温度系数
            # 这里用历史 GHI 和出力拟合一个简单映射, 作为基准特征
            # 实际值/GHI 比 = "单位辐照出力效率"
            if actual_mask.any() and "value" in df.columns:
                for group_key, group_df in df[actual_mask].groupby(["province", "type"]):
                    idx = group_df.index
                    ghi_vals = group_df["shortwave_radiation"].values
                    val_vals = group_df["value"].values
                    # 单位辐照出力效率 (MW per W/m²)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        eff = np.where(ghi_vals > 10, val_vals / ghi_vals, np.nan)
                    # 用滚动中位数作为效率基线 (减少异常值影响)
                    eff_series = pd.Series(eff, index=idx)
                    eff_baseline = eff_series.rolling(window=672, min_periods=96).median()
                    # 效率基线外推
                    eff_median = eff_baseline.dropna().median()
                    eff_filled = eff_baseline.fillna(eff_median)
                    df.loc[idx, "pv_efficiency_baseline"] = eff_filled.values
                    # 物理基准出力 = GHI × 效率基线
                    df.loc[idx, "pv_physical_baseline"] = ghi_vals * eff_filled.values
            # 温度系数 (温度越高, 光伏效率越低, 约 -0.4%/°C)
            if "temperature" in df.columns:
                df["pv_temp_coeff"] = ghi * (1 - 0.004 * np.maximum(df["temperature"] - 25, 0))
            # DNI/DHI 比值 (晴空指数)
            if "dni" in df.columns and "dhi" in df.columns:
                total = df["dni"] + df["dhi"]
                df["clear_sky_index"] = np.where(total > 1, df["dni"] / total, 0)
            # 云量对辐照的衰减
            if "cloud_cover" in df.columns:
                df["cloud_attenuation"] = ghi * (1 - df["cloud_cover"] / 100.0)
            # 日照强度分级
            df["solar_bin"] = pd.cut(ghi, bins=[-1, 0, 100, 400, 800, 2000],
                                       labels=[0, 1, 2, 3, 4]).astype(float)
            # ── 新增: 光伏出力物理模型特征 ──
            # GHI 平方根 (出力 ∝ sqrt(辐照), 非线性)
            df["ghi_sqrt"] = np.sqrt(np.maximum(ghi, 0))
            # DHI 主导度 (DHI/GHI, 高值=多云天气, 出力更平滑)
            if "dhi" in df.columns:
                df["dhi_dominance"] = np.where(ghi > 1, df["dhi"] / ghi, 0)
            # 云量二次衰减 (云量对出力的非线性影响)
            if "cloud_cover" in df.columns:
                cc = df["cloud_cover"] / 100.0
                df["cloud_nonlinear"] = ghi * (1 - cc ** 1.5)
            # 辐照 × cos(天顶角近似: 用 hour_cos 近似)
            if "hour_cos" in df.columns:
                df["ghi_x_cos"] = ghi * np.maximum(df["hour_cos"], 0)
            # 预报时间窗口内累计辐照 (6h 窗口)
            if actual_mask.any():
                for group_key, group_df in df[actual_mask].groupby(["province", "type"]):
                    idx = group_df.index
                    df.loc[idx, "solar_cumsum_6h"] = (
                        group_df["shortwave_radiation"].rolling(window=24, min_periods=1).sum().values
                    )

        # ── 初始化 pred_error 列为 0 (后续由 add_prediction_error_features 填充) ──
        for col in ["pred_error", "pred_error_lag_1d", "pred_error_lag_7d",
                     "pred_error_bias_24h", "pred_error_std_24h", "pred_error_trend",
                     "interval_coverage", "coverage_rate_24h",
                     "pred_error_hour_bias", "pred_error_weekend", "pred_error_holiday",
                     "pred_error_x_temp", "pred_error_x_wind",
                     "pred_error_autocorr", "pred_error_regime"]:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def add_prediction_error_features(self, features: pd.DataFrame,
                                       predictions: pd.DataFrame) -> pd.DataFrame:
        """注入预测误差特征 (多维度版).

        对齐条件: (dt, province, base_type+sub_type).

        新增列:
          基础误差: pred_error, pred_error_lag_1d/7d, bias_24h, std_24h, trend
          区间覆盖: interval_coverage, coverage_rate_24h
          时点模式: pred_error_hour_bias (每个时点的历史平均偏差)
          周模式:   pred_error_weekend (周末vs工作日误差差)
          节假日:   pred_error_holiday (节假日误差放大系数)
          气象交互: pred_error_x_temp, pred_error_x_wind (极端条件下的误差)
          自相关:   pred_error_autocorr (误差1天自相关)
          持续偏差: pred_error_regime (方向性系统性偏差)
        """
        if predictions.empty or features.empty:
            return features

        features = features.copy()
        features["_ti"] = features["type"].apply(lambda t: parse_type(str(t)))
        features["_base_sub"] = features["_ti"].apply(
            lambda ti: f"{ti.base}_{ti.sub}" if ti.sub else ti.base
        )

        preds = predictions.copy()
        if "type" in preds.columns:
            preds["_ti"] = preds["type"].apply(lambda t: parse_type(str(t)))
            preds["_base_sub"] = preds["_ti"].apply(
                lambda ti: f"{ti.base}_{ti.sub}" if ti.sub else ti.base
            )
        else:
            features.drop(columns=["_ti", "_base_sub"], inplace=True, errors="ignore")
            return features

        pred_subset = preds[["dt", "province", "_base_sub", "p50", "p10", "p90"]].copy()
        pred_subset = pred_subset.rename(columns={
            "p50": "_pred_p50", "p10": "_pred_p10", "p90": "_pred_p90"
        })

        merged = features.merge(pred_subset, on=["dt", "province", "_base_sub"], how="left")

        if "_pred_p50" not in merged.columns or merged["_pred_p50"].isna().all():
            features.drop(columns=["_ti", "_base_sub"], inplace=True, errors="ignore")
            return features

        # ── 基础误差 ──
        merged["pred_error"] = merged["value"] - merged["_pred_p50"]
        merged["pred_error_pct"] = np.where(
            merged["value"] != 0,
            merged["pred_error"] / merged["value"].abs(),
            0.0,
        )
        merged["interval_coverage"] = (
            (merged["value"] >= merged["_pred_p10"]) &
            (merged["value"] <= merged["_pred_p90"])
        ).astype(float)

        merged = merged.sort_values(["province", "_base_sub", "dt"]).reset_index(drop=True)

        # 提取时间维度 (用于多维误差分解)
        merged["_hour"] = merged["dt"].dt.hour
        merged["_dow"] = merged["dt"].dt.dayofweek
        merged["_is_weekend"] = merged["_dow"].isin([5, 6]).astype(int)

        for group_key, group_df in merged.groupby(["province", "_base_sub"]):
            idx = group_df.index
            n = len(group_df)

            # ── 基础滞后 + 滑动特征 ──
            for col_name, shift_n in self.PRED_ERROR_LAGS.items():
                merged.loc[idx, col_name] = group_df["pred_error"].shift(shift_n)

            merged.loc[idx, "pred_error_bias_24h"] = (
                group_df["pred_error"].rolling(window=96, min_periods=1).mean().values
            )
            merged.loc[idx, "pred_error_std_24h"] = (
                group_df["pred_error"].rolling(window=96, min_periods=1).std().values
            )
            merged.loc[idx, "pred_error_trend"] = merged.loc[idx, "pred_error_bias_24h"].diff(96)

            merged.loc[idx, "coverage_rate_24h"] = (
                group_df["interval_coverage"].rolling(window=96, min_periods=1).mean().values
            )

            # ── 时点误差模式 (每个小时的历史平均误差) ──
            hour_error_map = group_df.groupby("_hour")["pred_error"].mean()
            merged.loc[idx, "pred_error_hour_bias"] = (
                group_df["_hour"].map(hour_error_map).values
            )

            # ── 周末效应 (周末 vs 工作日误差幅度差) ──
            weekend_mask = group_df["_is_weekend"] == 1
            if weekend_mask.any() and (~weekend_mask).any():
                weekend_mean = group_df.loc[weekend_mask, "pred_error"].mean()
                weekday_mean = group_df.loc[~weekend_mask, "pred_error"].mean()
                merged.loc[idx, "pred_error_weekend"] = weekend_mean - weekday_mean
            else:
                merged.loc[idx, "pred_error_weekend"] = 0.0

            # ── 节假日效应 (误差放大系数) ──
            if "is_holiday" in group_df.columns:
                holiday_mask = group_df["is_holiday"] == True
                if holiday_mask.any() and (~holiday_mask).any():
                    holiday_mae = group_df.loc[holiday_mask, "pred_error"].abs().mean()
                    normal_mae = group_df.loc[~holiday_mask, "pred_error"].abs().mean()
                    if normal_mae > 0:
                        merged.loc[idx, "pred_error_holiday"] = (holiday_mae / normal_mae) - 1.0
                    else:
                        merged.loc[idx, "pred_error_holiday"] = 0.0
                else:
                    merged.loc[idx, "pred_error_holiday"] = 0.0
            else:
                merged.loc[idx, "pred_error_holiday"] = 0.0

            # ── 气象×误差交互 ──
            if "temperature" in group_df.columns:
                merged.loc[idx, "pred_error_x_temp"] = (
                    group_df["pred_error"].abs() * group_df["temperature"].fillna(0)
                )
            else:
                merged.loc[idx, "pred_error_x_temp"] = 0.0

            if "wind_speed" in group_df.columns:
                merged.loc[idx, "pred_error_x_wind"] = (
                    group_df["pred_error"].abs() * group_df["wind_speed"].fillna(0)
                )
            else:
                merged.loc[idx, "pred_error_x_wind"] = 0.0

            # ── 误差自相关 (1天间隔) ──
            if n >= 192:
                err_series = group_df["pred_error"].fillna(0)
                ac = err_series.autocorr(lag=96)
                merged.loc[idx, "pred_error_autocorr"] = ac if not np.isnan(ac) else 0.0
            else:
                merged.loc[idx, "pred_error_autocorr"] = 0.0

            # ── 持续偏差方向 (3天滑动偏差的符号) ──
            rolling_bias = group_df["pred_error"].rolling(window=288, min_periods=96).mean()
            merged.loc[idx, "pred_error_regime"] = (
                np.sign(rolling_bias.fillna(0).values).astype(int)
            )

        # 清理临时列
        drop_cols = ["_ti", "_base_sub", "_pred_p50", "_pred_p10", "_pred_p90",
                     "pred_error_pct", "_hour", "_dow", "_is_weekend"]
        merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)

        return merged

    def add_cross_type_features(self, features: pd.DataFrame,
                                 cross_df: pd.DataFrame,
                                 other_type: str) -> pd.DataFrame:
        """从其他类型数据注入交叉特征（v2: 三层交叉）."""
        cross_val_col = "value"
        if cross_val_col not in cross_df.columns:
            return features

        # 对齐时间粒度
        features = features.copy()
        features["_dt_hour"] = features["dt"].dt.floor("h")
        cross_subset = cross_df[["dt", cross_val_col]].copy()
        cross_subset["_dt_hour"] = cross_subset["dt"].dt.floor("h")
        cross_agg = cross_subset.groupby("_dt_hour")[cross_val_col].mean().reset_index()
        cross_agg.columns = ["_dt_hour", f"{other_type}_value"]

        result = features.merge(cross_agg, on="_dt_hour", how="left")
        result.drop(columns=["_dt_hour"], inplace=True)

        # 为 cross value 创建滞后特征
        col = f"{other_type}_value"
        if col in result.columns:
            result = result.sort_values("dt")
            result[f"{other_type}_lag_1d"] = result[col].shift(96)
            result[f"{other_type}_lag_7d"] = result[col].shift(672)

        # 价格专项: 出力/负荷 → 供需比
        if "output_value" in result.columns and "load_value" in result.columns:
            result["supply_demand_ratio"] = (
                result["output_value"] / result["load_value"].replace(0, None)
            )

        return result

    def add_cross_type_features_v2(self, features: pd.DataFrame,
                                    all_type_features: Dict[str, pd.DataFrame],
                                    province: str) -> pd.DataFrame:
        """三层交叉特征 (v2).

        层1: 同基类不同子类型交叉
        层2: 不同基类交叉 (去重，每种 pair 只注入一次)
        层3: 物理/经济约束特征
        """
        result = features.copy()
        rules = get_cross_type_rules()

        # 收集所有 sibling types
        all_siblings: set = set()
        for sibling_group in ["output_siblings", "load_siblings", "price_siblings"]:
            for s in rules.get(sibling_group, []):
                all_siblings.add(s)

        # ── 层1: 同基类子类型交叉 ──
        current_type = result["type"].iloc[0] if "type" in result.columns and len(result) > 0 else ""
        for other_stype in all_type_features:
            if other_stype == current_type or other_stype not in all_type_features:
                continue
            # 判断是否同基类 sibling
            cur_base = current_type.split("_")[0] if "_" in current_type else current_type
            oth_base = other_stype.split("_")[0] if "_" in other_stype else other_stype
            if cur_base == oth_base and current_type in all_siblings and other_stype in all_siblings:
                result = self.add_cross_type_features(
                    result, all_type_features[other_stype], other_stype
                )

        # ── 层2: 跨基类交叉 (每种 type 只注入一次) ──
        injected: set = set()
        cur_base = current_type.split("_")[0] if "_" in current_type else current_type
        for other_stype, other_df in all_type_features.items():
            if other_stype == current_type:
                continue
            oth_base = other_stype.split("_")[0] if "_" in other_stype else other_stype
            if cur_base != oth_base and other_stype not in injected:
                result = self.add_cross_type_features(result, other_df, other_stype)
                injected.add(other_stype)

        # ── 层3: 物理/经济约束 ──
        derived = rules.get("derived", [])
        for rule in derived:
            name = rule.get("name", "")
            formula = rule.get("formula", "")

            if name == "renewable_penetration":
                # (风电 + 光伏) / 总出力
                wind_col = None
                solar_col = None
                total_col = None
                for col in result.columns:
                    if "风电" in col and "value" in col and "lag" not in col:
                        wind_col = col
                    if "光伏" in col and "value" in col and "lag" not in col:
                        solar_col = col
                    if "总" in col and "出力" in col and "value" == col.split("_")[-1]:
                        total_col = col
                # 简化: 查 output_风电_value, output_光伏_value, output_总_value
                for col in result.columns:
                    if col.endswith("出力_风电_value"):
                        wind_col = col
                    if col.endswith("出力_光伏_value"):
                        solar_col = col
                    if col.endswith("出力_总_value"):
                        total_col = col

                if wind_col and solar_col and total_col:
                    numerator = result[wind_col].fillna(0) + result[solar_col].fillna(0)
                    denom = result[total_col].replace(0, None).fillna(1)
                    result[name] = numerator / denom

            elif name == "residual_demand":
                # 总负荷 - 风电 - 光伏 → 火电必须补的缺口
                load_col = None
                wind_col = None
                solar_col = None
                for col in result.columns:
                    if col.endswith("负荷_总_value") or col == "load_value":
                        load_col = col
                for col in result.columns:
                    if "出力_风电_value" in col:
                        wind_col = col
                    if "出力_光伏_value" in col:
                        solar_col = col
                if load_col:
                    residual = result[load_col].fillna(0)
                    if wind_col:
                        residual = residual - result[wind_col].fillna(0)
                    if solar_col:
                        residual = residual - result[solar_col].fillna(0)
                    result[name] = residual

            elif name == "thermal_marginal_cost":
                if "coal_price" in result.columns:
                    result[name] = result["coal_price"].fillna(0) * 0.35

            elif name == "price_pressure_index":
                if "supply_demand_ratio" in result.columns and "coal_price" in result.columns:
                    result[name] = (
                        result["supply_demand_ratio"].fillna(1) *
                        result["coal_price"].fillna(0)
                    )

        return result

    def merge_weather(self, features: pd.DataFrame,
                      weather: pd.DataFrame) -> pd.DataFrame:
        _weather_cols = ["temperature", "humidity", "wind_speed",
                         "wind_direction", "solar_radiation",
                         "precipitation", "pressure"]
        available = [c for c in _weather_cols if c in weather.columns]
        if not available:
            return features
        weather_subset = weather[["dt", "province"] + available].copy()
        features["_dt_hour"] = features["dt"].dt.floor("h")
        weather_subset = weather_subset.rename(columns={"dt": "_dt_hour"})
        result = features.merge(
            weather_subset, on=["_dt_hour", "province"], how="left"
        )
        result.drop(columns=["_dt_hour"], inplace=True)
        wfe = WeatherFeatureEngineer()
        result = wfe.transform(result)
        return result

    def add_price_features(self, features: pd.DataFrame,
                           all_siblings: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """电价专属特征: 供需紧密度、可再生能源冲击、价差、波动率."""
        import numpy as np
        result = features.copy()

        if "value" not in result.columns:
            return result

        # ── 1. 负荷交互: price / load 捕捉市场紧密度 ──
        load_cols = [c for c in result.columns
                     if "负荷" in c and c.endswith("_value") and "lag" not in c]
        if not load_cols:
            load_cols = [c for c in result.columns
                         if "负荷" in c and "value" in c and "lag" not in c]
        for lc in load_cols[:1]:
            load_vals = result[lc].replace(0, np.nan)
            result["price_per_load"] = result["value"] / load_vals.fillna(1)
            result["price_x_load"] = result["value"] * load_vals.fillna(0)

        # ── 2. 可再生能源渗透率 × 价格 (风光出力 ↑ → 电价 ↓) ──
        wind_cols = [c for c in result.columns
                     if "风电" in c and c.endswith("_value") and "lag" not in c]
        solar_cols = [c for c in result.columns
                      if "光伏" in c and c.endswith("_value") and "lag" not in c]
        renewable_cols = wind_cols[:1] + solar_cols[:1]
        if renewable_cols:
            re_total = sum(result[c].fillna(0) for c in renewable_cols)
            total_output_cols = [c for c in result.columns
                                 if "总" in c and "出力" in c and c.endswith("_value") and "lag" not in c]
            if total_output_cols:
                re_share = re_total / result[total_output_cols[0]].replace(0, np.nan).fillna(1)
                result["renewable_share"] = re_share.clip(0, 1)
                result["price_x_re_share"] = result["value"] * (1 - result["renewable_share"])

        # ── 3. 供需平衡: 总出力 / 总负荷 ──
        output_total_cols = [c for c in result.columns
                             if "总" in c and "出力" in c and c.endswith("_value") and "lag" not in c]
        if output_total_cols and load_cols:
            supply = result[output_total_cols[0]]
            demand = result[load_cols[0]].replace(0, np.nan)
            result["supply_demand_ratio"] = supply / demand.fillna(1)
            result["supply_surplus"] = supply - demand.fillna(0)

        # ── 4. 电价波动率 (日内、周内) ──
        if len(result) >= 96:
            result["price_vol_24h"] = (
                result["value"].rolling(96, min_periods=24).std()
                / result["value"].rolling(96, min_periods=24).mean().replace(0, np.nan)
            ).fillna(0)
        if len(result) >= 672:
            result["price_vol_7d"] = (
                result["value"].rolling(672, min_periods=96).std()
                / result["value"].rolling(672, min_periods=96).mean().replace(0, np.nan)
            ).fillna(0)

        # ── 5. 电价动量: 1h/6h/24h 变化 ──
        result["price_momentum_1h"] = result["value"].diff(4).fillna(0)
        result["price_momentum_6h"] = result["value"].diff(24).fillna(0)
        result["price_momentum_24h"] = result["value"].diff(96).fillna(0)

        # ── 6. 价格区间 (相对于日内波动) ──
        if len(result) >= 96:
            roll_max = result["value"].rolling(96, min_periods=24).max()
            roll_min = result["value"].rolling(96, min_periods=24).min()
            roll_range = (roll_max - roll_min).replace(0, np.nan)
            result["price_position"] = (
                (result["value"] - roll_min) / roll_range.fillna(1)
            ).fillna(0.5).clip(0, 1)

        # ── 7. 峰谷价差 (peak vs off-peak) ──
        hour = result.get("hour", pd.Series(0, index=result.index))
        peak_mask = hour.isin([8, 9, 10, 11, 17, 18, 19, 20])
        if peak_mask.any() and (~peak_mask).any():
            peak_avg = result.loc[peak_mask, "value"].mean()
            off_avg = result.loc[~peak_mask, "value"].mean() if (~peak_mask).sum() > 0 else peak_avg
            result["peak_off_peak_spread"] = peak_avg - off_avg

        return result


class FeatureStore:
    """特征存储管理. 通过 DataSource 抽象, 不依赖特定数据库."""

    def __init__(self, source=None):
        if source is None:
            from scripts.core.data_source import FileSource
            source = FileSource()
        self.source = source

    def setup(self):
        self.source.setup()

    def insert_features(self, df: pd.DataFrame) -> int:
        return self.source.save_features(df)

    def insert_predictions(self, df: pd.DataFrame) -> int:
        cols = ["dt", "province", "type", "p10", "p50", "p90",
                "model_version"]
        available = [c for c in cols if c in df.columns]
        if "model_version" not in df.columns:
            df["model_version"] = "v1"
        return self.source.save_predictions(df[available])

    def load_features(self, province: str, data_type: str,
                      start_date: str = None, end_date: str = None,
                      value_type_filter: Optional[str] = None) -> pd.DataFrame:
        return self.source.load_features(province, data_type, start_date, end_date,
                                         value_type_filter=value_type_filter)

    def load_raw_data(self, province: str, data_type: str,
                      start_date: str, end_date: str,
                      value_type_filter: Optional[str] = None) -> pd.DataFrame:
        return self.source.load_raw(province, data_type, start_date, end_date,
                                    value_type_filter=value_type_filter)

    def load_predictions(self, province: str, data_type: str,
                         start_date: str, end_date: str,
                         value_type_filter: Optional[str] = None) -> pd.DataFrame:
        return self.source.load_predictions(province, data_type, start_date, end_date,
                                            value_type_filter=value_type_filter)
