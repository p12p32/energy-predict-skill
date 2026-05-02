"""validator.py — 实时验证器 (type 三段式感知)"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from scripts.core.config import get_validator_config, parse_type


class Validator:
    def __init__(self):
        cfg = get_validator_config()
        self.short_threshold = cfg.get("short_term_mape_threshold", 0.05)
        self.mid_threshold = cfg.get("mid_term_mape_threshold", 0.10)
        self.consecutive_trigger = cfg.get("consecutive_bias_trigger", 3)

    @staticmethod
    def _add_alignment_keys(df: pd.DataFrame) -> pd.DataFrame:
        """为三段式 type 创建对齐 key: (dt, province, base, sub)."""
        df = df.copy()
        if "type" in df.columns:
            tis = df["type"].apply(lambda t: parse_type(str(t)))
            df["_base"] = tis.apply(lambda ti: ti.base)
            df["_sub"] = tis.apply(lambda ti: ti.sub or "")
        return df

    def compute_metrics(self, predictions: pd.DataFrame,
                        actuals: pd.DataFrame,
                        value_col: str = "p50") -> Dict:
        preds = self._add_alignment_keys(predictions)
        acts = self._add_alignment_keys(actuals[["dt", "province", "type", "value"]])

        # 用 (dt, province, _base, _sub) 对齐，而非简单的 type 字符串
        on_keys = ["dt", "province"]
        if "_base" in preds.columns and "_base" in acts.columns:
            on_keys += ["_base", "_sub"]

        merged = preds.merge(acts, on=on_keys, how="inner",
                            suffixes=("_pred", "_actual"))

        if merged.empty:
            return {"error": "no_overlap", "mape": None, "rmse": None}

        actual = merged["value"].values
        predicted = merged[value_col].values

        # sMAPE — 对接近0/负值更鲁棒
        denom = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denom > 1e-8
        if mask.sum() >= 10:
            mape = np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask])
        else:
            mask2 = actual != 0
            mape = np.mean(np.abs((actual[mask2] - predicted[mask2]) / actual[mask2])) if mask2.sum() > 0 else 0.0

        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        mae = np.mean(np.abs(actual - predicted))

        mean_bias = np.mean(predicted - actual)
        if abs(mean_bias) < 0.01 * np.mean(actual):
            bias_dir = "ok"
        elif mean_bias > 0:
            bias_dir = "high"
        else:
            bias_dir = "low"

        return {
            "mape": round(float(mape), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "bias_direction": bias_dir,
            "bias_magnitude": round(float(mean_bias), 2),
            "n_samples": len(merged),
        }

    def should_trigger(self, metrics: Dict,
                       history: List[Dict] = None) -> bool:
        if history is None:
            history = []

        mape = metrics.get("mape")
        if mape is None:
            return False

        if mape > self.short_threshold:
            return True

        if len(history) >= self.consecutive_trigger:
            recent_biases = [
                h.get("bias_direction") for h in history[-self.consecutive_trigger:]
            ]
            current_bias = metrics.get("bias_direction")
            if (current_bias in ("high", "low") and
                all(b == current_bias for b in recent_biases)):
                return True

        return False

    def validate(self, predictions: pd.DataFrame,
                 actuals: pd.DataFrame,
                 value_col: str = "p50") -> Dict:
        metrics = self.compute_metrics(predictions, actuals, value_col)
        triggered = self.should_trigger(metrics)

        return {
            "timestamp": datetime.now().isoformat(),
            "triggered": triggered,
            "metrics": metrics,
        }
