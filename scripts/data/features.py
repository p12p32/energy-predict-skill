"""feature_store.py — 特征工程与数据源抽象管理"""
import pandas as pd
import numpy as np
from typing import Optional, List
from scripts.data.holidays import add_holiday_features, add_cyclical_features, add_deep_calendar_features
from scripts.data.quality import DataQuality
from scripts.data.weather_features import WeatherFeatureEngineer

PREDICTION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS energy_predictions (
    dt              DATETIME      NOT NULL,
    province        VARCHAR(32)   NOT NULL,
    type            VARCHAR(16)   NOT NULL,
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
    type            VARCHAR(16)   NOT NULL,
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
        "value_lag_1d": 96,
        "value_lag_7d": 672,
        "value_diff_1d": 96,
        "value_diff_7d": 672,
    }

    def build_features_from_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df = df.sort_values(["province", "type", "dt"]).reset_index(drop=True)

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

        # ── 深度天气特征 ──
        wfe = WeatherFeatureEngineer()
        df = wfe.transform(df)

        # ── 节假日 + 深度日历 ──
        df = add_holiday_features(df)
        df = add_deep_calendar_features(df)

        # ── 周期编码 ──
        df = add_cyclical_features(df)

        # ── 滞后特征 ──
        for group_key, group_df in df.groupby(["province", "type"]):
            idx = group_df.index

            for col_name, shift_n in self.LAG_PERIODS.items():
                if "diff" in col_name:
                    df.loc[idx, col_name] = group_df["value"].diff(shift_n)
                else:
                    df.loc[idx, col_name] = group_df["value"].shift(shift_n)

            df.loc[idx, "value_rolling_mean_24h"] = (
                group_df["value"].rolling(window=96, min_periods=1).mean().values
            )

        return df

    def merge_weather(self, features: pd.DataFrame,
                      weather: pd.DataFrame) -> pd.DataFrame:
        _weather_cols = ["temperature", "humidity", "wind_speed",
                         "wind_direction", "solar_radiation",
                         "precipitation", "pressure"]
        available = [c for c in _weather_cols if c in weather.columns]
        if not available:
            return features
        weather_subset = weather[["dt", "province"] + available].copy()
        # 截断到小时以匹配电力数据的15分钟粒度
        features["_dt_hour"] = features["dt"].dt.floor("h")
        weather_subset = weather_subset.rename(columns={"dt": "_dt_hour"})
        result = features.merge(
            weather_subset, on=["_dt_hour", "province"], how="left"
        )
        result.drop(columns=["_dt_hour"], inplace=True)
        # 合并后重新计算派生天气特征 (CDD/HDD/THI 等)
        wfe = WeatherFeatureEngineer()
        result = wfe.transform(result)
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
        cols = [
            "dt", "province", "type", "value", "price",
            "hour", "day_of_week", "day_of_month", "month",
            "is_weekend", "season",
            "temperature", "humidity", "wind_speed", "wind_direction",
            "solar_radiation", "precipitation", "pressure",
            "CDD", "HDD", "THI", "wind_power_potential",
            "solar_potential", "solar_efficiency",
            "temp_change_1h", "temp_change_6h", "consecutive_hot_days",
            "value_lag_1d", "value_lag_7d",
            "value_rolling_mean_24h",
            "value_diff_1d", "value_diff_7d",
            "is_holiday", "is_work_weekend",
            "days_to_holiday", "days_from_holiday",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos",
            "quality_flag",
        ]
        available = [c for c in cols if c in df.columns]
        return self.source.save_features(df[available])

    def insert_predictions(self, df: pd.DataFrame) -> int:
        cols = ["dt", "province", "type", "p10", "p50", "p90",
                "model_version"]
        available = [c for c in cols if c in df.columns]
        if "model_version" not in df.columns:
            df["model_version"] = "v1"
        return self.source.save_predictions(df[available])

    def load_features(self, province: str, data_type: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        return self.source.load_features(province, data_type, start_date, end_date)

    def load_raw_data(self, province: str, data_type: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        return self.source.load_raw(province, data_type, start_date, end_date)

    def load_predictions(self, province: str, data_type: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        return self.source.load_predictions(province, data_type, start_date, end_date)
