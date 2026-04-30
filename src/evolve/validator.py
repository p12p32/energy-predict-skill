"""validator.py — 实时验证器（循环 A）"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from src.core.config import get_validator_config


class Validator:
    def __init__(self):
        cfg = get_validator_config()
        self.short_threshold = cfg.get("short_term_mape_threshold", 0.05)
        self.mid_threshold = cfg.get("mid_term_mape_threshold", 0.10)
        self.consecutive_trigger = cfg.get("consecutive_bias_trigger", 3)

    def compute_metrics(self, predictions: pd.DataFrame,
                        actuals: pd.DataFrame,
                        value_col: str = "p50") -> Dict:
        merged = predictions.merge(
            actuals[["dt", "province", "type", "value"]],
            on=["dt", "province", "type"], how="inner",
            suffixes=("_pred", "_actual"),
        )

        if merged.empty:
            return {"error": "no_overlap", "mape": None, "rmse": None}

        actual = merged["value"].values
        predicted = merged[value_col].values

        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))

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
