"""strategy_executor.py — 将 improver 的策略翻译为实际数据/特征操作"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class StrategyExecutor:
    """把 JSON 策略变成真实的数据变换。

    improver 说"加 temperature² 特征"→ executor 在 DataFrame 上真的加这列。
    """

    def execute(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """根据策略对数据执行变换，返回新 DataFrame."""
        name = strategy.get("name", "")
        params = strategy.get("params", {})

        method = getattr(self, f"_exec_{name}", None)
        if method is None:
            return df

        return method(df, params)

    def execute_all(self, df: pd.DataFrame,
                    strategies: List[Dict]) -> pd.DataFrame:
        """链式执行多个策略."""
        result = df.copy()
        for s in strategies:
            result = self.execute(result, s)
        return result

    # ── 特征级变换 ──

    def _exec_polynomial_features(self, df: pd.DataFrame,
                                   params: Dict) -> pd.DataFrame:
        """加入多项式特征: 对数值列做 x²、x³."""
        power = params.get("power", 2)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 优先对关键特征做多项式
        priority = ["temperature", "humidity", "wind_speed",
                     "value_lag_1d", "value_lag_7d",
                     "value_rolling_mean_24h"]
        targets = [c for c in priority if c in numeric_cols]

        if not targets:
            targets = list(numeric_cols)[-5:]

        result = df.copy()
        for col in targets[:4]:  # 最多加 4 个多项式列
            col_name = f"{col}_p{power}"
            if col_name not in result.columns:
                result[col_name] = result[col] ** power

        return result

    def _exec_dayofweek_interaction(self, df: pd.DataFrame,
                                     params: Dict) -> pd.DataFrame:
        """加入 day_of_week 交互特征."""
        interact_with = params.get("interact_with", ["hour", "value_lag_7d"])

        if "day_of_week" not in df.columns:
            return df

        result = df.copy()
        for col in interact_with:
            if col in df.columns:
                int_name = f"dow_{col}"
                if int_name not in result.columns:
                    result[int_name] = df["day_of_week"] * df[col]

        return result

    def _exec_rolling_window_features(self, df: pd.DataFrame,
                                       params: Dict) -> pd.DataFrame:
        """加入滑动窗口特征."""
        window_hours = params.get("window_hours", 4)
        window_steps = window_hours * 4  # 每小时 4 步

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        priority = ["value", "temperature", "wind_speed",
                     "solar_radiation", "humidity"]
        targets = [c for c in priority if c in numeric_cols]

        result = df.copy()
        for col in targets[:3]:
            roll_name = f"{col}_roll{window_hours}h"
            if roll_name not in result.columns:
                result[roll_name] = (
                    result[col].rolling(window=window_steps, min_periods=1).mean()
                )

        return result

    # ── 数据级变换 ──

    def _exec_recent_upsample(self, df: pd.DataFrame,
                               params: Dict) -> pd.DataFrame:
        """近期样本上采样：复制最近 N 天的样本 weight 次."""
        weight = params.get("weight", 3)
        days = params.get("days", 7)

        if "dt" not in df.columns or len(df) == 0:
            return df

        recent_cutoff = df["dt"].max() - pd.Timedelta(days=days)
        recent_mask = df["dt"] >= recent_cutoff

        if recent_mask.sum() == 0:
            return df

        recent = df[recent_mask]
        # 每行复制 weight-1 次
        copies = [recent] * (weight - 1)
        upsampled = [df] + copies

        result = pd.concat(upsampled, ignore_index=True)
        return result.sort_values("dt", ascending=True).reset_index(drop=True)

    def _exec_extreme_oversample(self, df: pd.DataFrame,
                                  params: Dict) -> pd.DataFrame:
        """极端条件样本过采样：对高温/强风/极端负荷样本加倍."""
        factor = params.get("factor", 3)
        condition = params.get("condition", "percentile>95")
        target_col = params.get("target_col", "value")

        if target_col not in df.columns or len(df) == 0:
            return df

        vals = df[target_col].dropna()
        if len(vals) == 0:
            return df

        threshold = np.percentile(vals, 95)
        extreme_mask = df[target_col] > threshold

        if extreme_mask.sum() == 0:
            return df

        extreme = df[extreme_mask]
        copies = [extreme] * (factor - 1)

        result = pd.concat([df] + copies, ignore_index=True)
        return result.sort_values("dt", ascending=True).reset_index(drop=True)

    def _exec_holiday_oversample(self, df: pd.DataFrame,
                                  params: Dict) -> pd.DataFrame:
        """节假日样本过采样."""
        weight = params.get("weight", 5)

        if "is_holiday" not in df.columns or "is_weekend" not in df.columns:
            return df
        if df["is_holiday"].sum() == 0 and df["is_weekend"].sum() == 0:
            return df

        holiday_mask = df["is_holiday"] | df["is_weekend"]
        if holiday_mask.sum() == 0:
            return df

        holiday_samples = df[holiday_mask]
        copies = [holiday_samples] * (weight - 1)

        result = pd.concat([df] + copies, ignore_index=True)
        return result.sort_values("dt", ascending=True).reset_index(drop=True)

    def _exec_shorter_window(self, df: pd.DataFrame,
                              params: Dict) -> pd.DataFrame:
        """减小训练窗口：只保留最近 N 天."""
        n_days = params.get("n", 30)

        if "dt" not in df.columns or len(df) == 0:
            return df

        cutoff = df["dt"].max() - pd.Timedelta(days=n_days)
        return df[df["dt"] >= cutoff].copy()

    # ── 参数级变换（透传，由 trainer 处理）──

    def _exec_switch_to_catboost(self, df: pd.DataFrame,
                                  params: Dict) -> pd.DataFrame:
        """模型切换：数据本身不变，由 trainer 根据 params 切换模型."""
        return df

    def _exec_province_independent_model(self, df: pd.DataFrame,
                                          params: Dict) -> pd.DataFrame:
        """省份独立建模：数据不变，由 orchestrator 分拆调用."""
        return df

    def _exec_bias_correction(self, df: pd.DataFrame,
                               params: Dict) -> pd.DataFrame:
        """偏差补偿：数据不变，后处理阶段应用."""
        return df
