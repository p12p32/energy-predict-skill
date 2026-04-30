"""data_quality.py — 异常值检测与数据质量标记

不删除异常值，而是标记 + 提供质量报告。
让模型自己学习哪些数据点可靠，哪些是噪声。
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class DataQuality:
    def __init__(self, iqr_multiplier: float = 3.0,
                 spike_window: int = 16, spike_threshold: float = 5.0):
        self.iqr_multiplier = iqr_multiplier
        self.spike_window = spike_window     # 跳变检测窗口(步)
        self.spike_threshold = spike_threshold  # 跳变倍数

    def detect(self, df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
        """对数据做三种异常检测，输出标记列.

        输出新增列:
        - quality_flag: 0=正常, 1=疑似异常, 2=确认异常
        - outlier_IQR: 是否 IQR 异常
        - outlier_spike: 是否跳变异常
        - outlier_zero: 是否零值异常
        """
        result = df.copy()
        result["quality_flag"] = 0
        result["outlier_IQR"] = False
        result["outlier_spike"] = False
        result["outlier_zero"] = False

        values = result[value_col]

        # ── IQR 异常检测 ──
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - self.iqr_multiplier * IQR
        upper = Q3 + self.iqr_multiplier * IQR

        iqr_mask = (values < lower) | (values > upper)
        result.loc[iqr_mask, "outlier_IQR"] = True
        result.loc[iqr_mask, "quality_flag"] = result.loc[iqr_mask, "quality_flag"].clip(lower=1)

        # ── 跳变异常检测 ──
        # 相邻值突变超过 n 倍标准差
        diffs = np.abs(values.diff())
        recent_std = values.rolling(window=self.spike_window * 4, min_periods=1).std()
        spike_mask = diffs > self.spike_threshold * recent_std
        result.loc[spike_mask, "outlier_spike"] = True
        result.loc[spike_mask, "quality_flag"] = result.loc[spike_mask, "quality_flag"].clip(lower=1)

        # ── 零值/负值检测 ──
        zero_mask = values <= 0
        result.loc[zero_mask, "outlier_zero"] = True
        result.loc[zero_mask, "quality_flag"] = 2  # 确认异常

        # ── 同时触发多个 → 确认异常 ──
        multi = (result["outlier_IQR"] & result["outlier_spike"])
        result.loc[multi, "quality_flag"] = 2

        return result

    def report(self, df: pd.DataFrame) -> Dict:
        """生成数据质量报告."""
        if "quality_flag" not in df.columns:
            df = self.detect(df)

        total = len(df)
        flagged = (df["quality_flag"] > 0).sum()
        confirmed = (df["quality_flag"] == 2).sum()

        return {
            "total_rows": total,
            "flagged_count": int(flagged),
            "flagged_pct": round(float(flagged / total * 100), 2) if total > 0 else 0,
            "confirmed_anomaly_count": int(confirmed),
            "confirmed_pct": round(float(confirmed / total * 100), 2) if total > 0 else 0,
            "iqr_outliers": int(df["outlier_IQR"].sum()),
            "spike_outliers": int(df["outlier_spike"].sum()),
            "zero_outliers": int(df["outlier_zero"].sum()),
        }

    def clean(self, df: pd.DataFrame, value_col: str = "value",
              strategy: str = "interpolate") -> pd.DataFrame:
        """清洗异常值.

        Args:
            strategy: 'interpolate' 插值, 'clip' 截断, 'remove' 删除
        """
        if "quality_flag" not in df.columns:
            df = self.detect(df, value_col)

        result = df.copy()
        bad = result["quality_flag"] >= 2

        if strategy == "interpolate":
            result.loc[bad, value_col] = np.nan
            result[value_col] = result[value_col].interpolate(
                method="linear", limit_direction="both"
            )
        elif strategy == "clip":
            Q1 = result[value_col].quantile(0.05)
            Q3 = result[value_col].quantile(0.95)
            result[value_col] = result[value_col].clip(Q1, Q3)
        elif strategy == "remove":
            result = result[~bad].copy()

        return result
