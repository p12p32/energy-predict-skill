"""analyzer.py — 误差根因诊断器"""
from typing import Dict, List, Optional


class Analyzer:
    DIAGNOSIS_RULES = [
        {
            "condition": lambda r: r.get("by_season", {}).get("summer", {}).get("mape", 0) is not None
                                   and r.get("by_season", {}).get("summer", {}).get("mape", 0) > 0.12,
            "diagnosis": {
                "scenario": "summer",
                "root_cause": "nonlinear_heat_effect",
                "description": "夏季高温导致负荷非线性增长，线性模型捕捉不足",
                "severity": "high",
                "suggested_features": ["temperature²", "temperature×is_weekend"],
            },
        },
        {
            "condition": lambda r: r.get("by_time_type", {}).get("weekend", {}).get("mape", 0) is not None
                                   and r.get("by_time_type", {}).get("weekend", {}).get("mape", 0) > 0.10,
            "diagnosis": {
                "scenario": "weekend",
                "root_cause": "weekend_pattern_mismatch",
                "description": "周末用电模式与工作日差异大，缺乏交互特征",
                "severity": "medium",
                "suggested_features": ["day_of_week×hour", "is_weekend×value_lag_7d"],
            },
        },
        {
            "condition": lambda r: r.get("by_time_type", {}).get("holiday", {}).get("mape", 0) is not None
                                   and r.get("by_time_type", {}).get("holiday", {}).get("mape", 0) > 0.15,
            "diagnosis": {
                "scenario": "holiday",
                "root_cause": "holiday_pattern_break",
                "description": "节假日用电模式与平时显著不同",
                "severity": "high",
                "suggested_features": ["is_holiday", "days_from_holiday"],
            },
        },
        {
            "condition": lambda r: max(
                r.get("by_season", {}).get(s, {}).get("mape", 0) or 0
                for s in ["spring", "summer", "autumn", "winter"]
                if s in r.get("by_season", {})
            ) > 0.12 if any(s in r.get("by_season", {}) for s in ["spring", "summer", "autumn", "winter"]) else False,
            "diagnosis": {
                "scenario": "seasonal",
                "root_cause": "seasonal_concept_drift",
                "description": "季节更替导致数据分布偏移",
                "severity": "medium",
                "suggested_features": ["season_onehot", "month_sin_cos"],
            },
        },
        {
            "condition": lambda r: r.get("overall_mape", 0) > 0.15,
            "diagnosis": {
                "scenario": "overall",
                "root_cause": "distribution_shift",
                "description": "整体预测大幅度退化，可能有概念漂移或数据结构变化",
                "severity": "critical",
                "suggested_features": ["shorter_training_window", "full_retrain"],
            },
        },
        {
            "condition": lambda r: r.get("overall_mape", 0) > 0.08
                                   and r.get("by_hour_bucket", {}).get("peak", {}).get("mape", 0) is not None
                                   and r.get("by_hour_bucket", {}).get("peak", {}).get("mape", 0) > 0.10,
            "diagnosis": {
                "scenario": "peak_hours",
                "root_cause": "peak_load_variance",
                "description": "高峰时段负荷波动大，需要更细粒度的特征",
                "severity": "medium",
                "suggested_features": ["hour_onehot", "peak_valley_indicator"],
            },
        },
    ]

    def diagnose(self, backtest_report: Dict,
                 baseline_mape: float = 0.10,
                 validator_history: List[Dict] = None) -> List[Dict]:
        results = []
        for rule in self.DIAGNOSIS_RULES:
            try:
                if rule["condition"](backtest_report):
                    diag = dict(rule["diagnosis"])
                    diag["baseline_mape"] = baseline_mape
                    diag["current_mape"] = backtest_report.get("overall_mape")
                    results.append(diag)
            except (KeyError, TypeError):
                continue

        if validator_history and len(validator_history) >= 3:
            recent_biases = [
                h.get("bias_direction") for h in validator_history[-3:]
                if "bias_direction" in h
            ]
            if (len(recent_biases) >= 3 and
                all(b == recent_biases[0] for b in recent_biases) and
                recent_biases[0] in ("high", "low")):
                direction = "偏高" if recent_biases[0] == "high" else "偏低"
                results.append({
                    "scenario": "persistent_bias",
                    "root_cause": "systematic_bias",
                    "description": f"连续多轮同向偏差（{direction}），存在系统性误差",
                    "severity": "high",
                    "suggested_features": ["bias_correction_factor"],
                })

        return results
