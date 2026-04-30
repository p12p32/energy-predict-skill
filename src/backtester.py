"""backtester.py — 回塑验证 + 多维度精细打分（循环 B）"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.trainer import Trainer


class Backtester:
    def __init__(self, trainer: Trainer = None):
        self.trainer = trainer or Trainer()

    def rolling_window_backtest(self, df: pd.DataFrame,
                                 train_window_days: int,
                                 test_window_hours: int,
                                 n_windows: int,
                                 province: str,
                                 target_type: str,
                                 target_col: str = "value") -> Dict:
        total_steps = 96 * train_window_days
        test_steps = 4 * test_window_hours
        max_start = len(df) - total_steps - test_steps - 1

        if max_start <= 0:
            return {"error": "数据不足以做滚动窗口回测"}

        windows = []
        step_size = max(1, max_start // n_windows)

        for i, start_idx in enumerate(range(0, max_start, step_size)[:n_windows]):
            train_df = df.iloc[start_idx : start_idx + total_steps]
            test_df = df.iloc[
                start_idx + total_steps : start_idx + total_steps + test_steps
            ]

            result = self.trainer.quick_train(
                train_df, province, target_type, target_col
            )
            model = result["model"]
            feature_names = result["feature_names"]

            test_features = test_df[feature_names].values
            predictions = model.predict(test_features)
            actuals = test_df[target_col].values

            mape = self._calc_mape(actuals, predictions)
            rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

            windows.append({
                "window": i + 1,
                "train_start": train_df["dt"].iloc[0].isoformat(),
                "test_start": test_df["dt"].iloc[0].isoformat(),
                "mape": round(float(mape), 4),
                "rmse": round(float(rmse), 2),
                "n_train": len(train_df),
                "n_test": len(test_df),
            })

        summary = {
            "overall_mape": round(float(np.mean([w["mape"] for w in windows])), 4),
            "overall_rmse": round(float(np.mean([w["rmse"] for w in windows])), 2),
            "best_mape": round(float(min(w["mape"] for w in windows)), 4),
            "worst_mape": round(float(max(w["mape"] for w in windows)), 4),
            "n_windows": len(windows),
        }

        return {"windows": windows, "summary": summary}

    def multi_dimension_score(self, actuals: np.ndarray,
                               predictions: np.ndarray,
                               metadata: pd.DataFrame) -> Dict:
        errors = np.abs(actuals - predictions)

        def _mape_in_mask(mask):
            if mask.sum() == 0:
                return None
            a = actuals[mask]
            p = predictions[mask]
            m = a != 0
            return round(float(np.mean(np.abs((a[m] - p[m]) / a[m]))), 4) if m.sum() > 0 else None

        meta = metadata.reset_index(drop=True)
        if len(meta) != len(actuals):
            meta = metadata.iloc[:len(actuals)].reset_index(drop=True)

        by_season = {}
        season_map = {1: "spring", 2: "summer", 3: "autumn", 4: "winter"}
        if "season" in meta.columns:
            for s_val, s_name in season_map.items():
                mask = meta["season"].values == s_val
                by_season[s_name] = {
                    "mape": _mape_in_mask(mask),
                    "samples": int(mask.sum()),
                }

        by_time = {}
        if "is_weekend" in meta.columns:
            is_weekend = meta["is_weekend"].astype(bool).values
            by_time["workday"] = {"mape": _mape_in_mask(~is_weekend)}
            by_time["weekend"] = {"mape": _mape_in_mask(is_weekend)}

        by_hour = {}
        if "hour" in meta.columns:
            hours = meta["hour"].values
            peak_mask = (hours >= 8) & (hours < 12) | (hours >= 17) & (hours < 21)
            valley_mask = (hours >= 0) & (hours < 8) | (hours >= 22)
            flat_mask = (hours >= 12) & (hours < 17)
            by_hour["peak"] = {"mape": _mape_in_mask(peak_mask)}
            by_hour["valley"] = {"mape": _mape_in_mask(valley_mask)}
            by_hour["flat"] = {"mape": _mape_in_mask(flat_mask)}

        return {
            "by_season": by_season,
            "by_time_type": by_time,
            "by_hour_bucket": by_hour,
        }

    def evaluate_model(self, df: pd.DataFrame,
                        train_window_days: int,
                        test_window_hours: int,
                        province: str,
                        target_type: str,
                        target_col: str = "value") -> Dict:
        rb = self.rolling_window_backtest(
            df, train_window_days, test_window_hours,
            n_windows=3, province=province,
            target_type=target_type, target_col=target_col,
        )

        if "error" in rb:
            return rb

        train_steps = 96 * train_window_days
        test_steps = 4 * test_window_hours
        total = train_steps + test_steps
        last_train = df.iloc[-total : -test_steps]
        last_test = df.iloc[-test_steps:]

        bt_result = self.trainer.quick_train(
            last_train, province, target_type, target_col
        )
        pred = bt_result["model"].predict(last_test[bt_result["feature_names"]].values)
        multi_scores = self.multi_dimension_score(
            last_test[target_col].values, pred, last_test
        )

        return {
            "overall_mape": rb["summary"]["overall_mape"],
            "overall_rmse": rb["summary"]["overall_rmse"],
            "rolling_windows": rb["windows"],
            **multi_scores,
        }

    @staticmethod
    def _calc_mape(actuals: np.ndarray, predictions: np.ndarray) -> float:
        mask = actuals != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])))
