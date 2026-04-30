"""improver.py — 学习型优化引擎 + 策略知识库"""
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.db import DorisDB
from src.trainer import Trainer
from src.backtester import Backtester


class Improver:
    SEED_STRATEGIES = [
        {
            "name": "polynomial_features",
            "description": "加入多项式特征 {col}²",
            "params": {"power": 2},
            "applicable": ["nonlinear"],
        },
        {
            "name": "recent_upsample",
            "description": "近期样本上采样 ×{weight}",
            "params": {"weight": 3, "days": 7},
            "applicable": ["drift", "decay"],
        },
        {
            "name": "shorter_window",
            "description": "减小训练窗口至 {n} 天",
            "params": {"n": 30},
            "applicable": ["drift", "shift"],
        },
        {
            "name": "switch_to_catboost",
            "description": "切换模型至 CatBoost",
            "params": {"model": "catboost"},
            "applicable": ["variance", "price"],
        },
        {
            "name": "province_independent_model",
            "description": "{province} 独立建模",
            "params": {},
            "applicable": ["province_bias"],
        },
        {
            "name": "extreme_oversample",
            "description": "极端条件样本过采样 ×{factor}",
            "params": {"factor": 3, "condition": "percentile>95"},
            "applicable": ["extreme", "tail"],
        },
        {
            "name": "dayofweek_interaction",
            "description": "加 day_of_week 交互特征",
            "params": {"interact_with": ["hour", "value_lag_7d"]},
            "applicable": ["weekend", "pattern"],
        },
        {
            "name": "holiday_oversample",
            "description": "节假日样本过采样 ×{weight}",
            "params": {"weight": 5},
            "applicable": ["holiday"],
        },
        {
            "name": "rolling_window_features",
            "description": "{col} 滑动窗口均值({n}h) 作为新特征",
            "params": {"window_hours": 4},
            "applicable": ["smoothness"],
        },
        {
            "name": "bias_correction",
            "description": "偏差自动补偿: 预测值 + 近期平均偏差",
            "params": {},
            "applicable": ["systematic_bias"],
        },
    ]

    def __init__(self, db: DorisDB = None,
                 trainer: Trainer = None,
                 backtester: Backtester = None):
        self.db = db or DorisDB()
        self.trainer = trainer or Trainer()
        self.backtester = backtester or Backtester(self.trainer)

    def generate_hypotheses(self, diagnosis: List[Dict]) -> List[Dict]:
        keywords = set()
        for d in diagnosis:
            root = d.get("root_cause", "")
            scenario = d.get("scenario", "")
            for word in root.replace("_", " ").split():
                keywords.add(word.lower())
            keywords.add(scenario.lower())

        suggested = set()
        for d in diagnosis:
            for sf in d.get("suggested_features", []):
                suggested.add(sf.lower())

        hypotheses = []

        for seed in self.SEED_STRATEGIES:
            applicable = seed.get("applicable", [])
            if any(kw in applicable for kw in keywords):
                hypotheses.append({
                    "name": seed["name"],
                    "description": seed["description"],
                    "params": dict(seed.get("params", {})),
                    "source": "seed_match",
                })
            elif suggested & set(applicable):
                hypotheses.append({
                    "name": seed["name"],
                    "description": seed["description"],
                    "params": dict(seed.get("params", {})),
                    "source": "suggested_match",
                })
            elif "overall" in keywords or "critical" in keywords:
                hypotheses.append({
                    "name": seed["name"],
                    "description": seed["description"],
                    "params": dict(seed.get("params", {})),
                    "source": "critical_explore",
                })

        seen = set()
        unique = []
        for h in hypotheses:
            sig = h["name"] + json.dumps(h["params"], sort_keys=True)
            if sig not in seen:
                seen.add(sig)
                unique.append(h)

        return unique[:12]

    def run_experiment(self, hypothesis: Dict, df: pd.DataFrame,
                       province: str, target_type: str,
                       target_col: str = "value") -> Dict:
        try:
            bt_result = self.trainer.quick_train(
                df, province, target_type, target_col,
                params=hypothesis.get("model_params"),
            )

            total = len(df)
            test_steps = min(96, total // 5)
            train_df = df.iloc[:-test_steps]
            test_df = df.iloc[-test_steps:]

            model = bt_result["model"]
            pred = model.predict(
                test_df[bt_result["feature_names"]].values
            )
            actual = test_df[target_col].values

            mask = actual != 0
            mape = float(
                np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask]))
                if mask.sum() > 0 else 1.0
            )

            return {
                "hypothesis_id": self._hash(hypothesis["name"]),
                "hypothesis": hypothesis,
                "mape": round(mape, 4),
                "n_samples": len(df),
            }
        except Exception as e:
            return {
                "hypothesis_id": self._hash(hypothesis["name"]),
                "hypothesis": hypothesis,
                "error": str(e),
            }

    def run_arena(self, hypotheses: List[Dict], df: pd.DataFrame,
                  province: str, target_type: str,
                  target_col: str = "value") -> List[Dict]:
        results = []
        for h in hypotheses:
            result = self.run_experiment(h, df, province, target_type, target_col)
            results.append(result)

        return sorted(results, key=lambda r: r.get("mape", float("inf")))

    def select_best(self, arena_results: List[Dict],
                     diagnosis: List[Dict],
                     baseline: Dict) -> Dict:
        if not arena_results:
            return {"error": "no_valid_experiments"}

        pain_points = {}
        for d in diagnosis:
            scenario = d.get("scenario", "")
            pain_points[scenario] = d.get("severity", "medium")

        severity_weight = {"critical": 5.0, "high": 3.0, "medium": 1.5, "low": 1.0}

        scored = []
        for r in arena_results:
            if "error" in r:
                continue

            mape = r.get("mape", 1.0)
            baseline_mape = baseline.get("overall_mape", 0.10)

            improvement = baseline_mape - mape

            complexity_penalty = 0.0
            if "polynomial" in r.get("hypothesis", {}).get("name", ""):
                complexity_penalty = 0.002
            if "catboost" in r.get("hypothesis", {}).get("name", "").lower():
                complexity_penalty = 0.004

            weight = 1.0
            for scenario, severity in pain_points.items():
                weight = max(weight, severity_weight.get(severity, 1.0))

            score = improvement * weight - complexity_penalty

            scored.append({
                **r,
                "improvement": round(float(improvement), 4),
                "score": round(float(score), 4),
            })

        scored.sort(key=lambda r: r["score"], reverse=True)

        if not scored:
            return {"error": "all_experiments_failed"}

        winner = scored[0]
        return {
            "selected": winner["hypothesis"],
            "hypothesis_id": winner["hypothesis_id"],
            "mape_after": winner["mape"],
            "mape_before": baseline.get("overall_mape"),
            "improvement": winner["improvement"],
            "all_results": [s["hypothesis_id"] for s in scored[:5]],
        }

    def record_strategy(self, strategy: Dict, scenario: str = "",
                        success: bool = True):
        strategy_hash = self._hash(strategy.get("name", ""))
        desc = strategy.get("desc", strategy.get("name", ""))

        improvement = strategy.get("improvement", {})
        if isinstance(improvement, dict):
            effect = improvement.get("after", 0) - improvement.get("before", 0)
        else:
            effect = float(improvement) if improvement else 0

        sql = f"""
            INSERT INTO strategy_knowledge
                (strategy_hash, strategy_desc, applied_count, success_count,
                 avg_improvement, best_scenario, worst_scenario,
                 last_applied, last_effect)
            VALUES
                ('{strategy_hash}', '{desc}', 1, {1 if success else 0},
                 {abs(effect)}, '{scenario}', '',
                 '{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', {effect})
            ON DUPLICATE KEY UPDATE
                applied_count = applied_count + 1,
                success_count = success_count + {1 if success else 0},
                avg_improvement = (avg_improvement * applied_count + {abs(effect)}) 
                                  / (applied_count + 1),
                last_applied = '{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                last_effect = {effect}
        """
        self.db.execute(sql)

    def query_knowledge(self, scenario: str) -> List[Dict]:
        sql = f"""
            SELECT strategy_hash, strategy_desc, applied_count,
                   success_count, avg_improvement, best_scenario,
                   last_applied, last_effect
            FROM strategy_knowledge
            WHERE NOT retired
              AND (best_scenario LIKE '%{scenario}%' OR best_scenario = '')
            ORDER BY avg_improvement DESC
            LIMIT 20
        """
        return self.db.query(sql).to_dict("records") if self.db.table_exists("strategy_knowledge") else []

    def improve(self, diagnosis: List[Dict], df: pd.DataFrame,
                province: str, target_type: str,
                target_col: str = "value",
                baseline: Dict = None) -> Dict:
        if baseline is None:
            baseline = {"overall_mape": 0.10}

        hypotheses = self.generate_hypotheses(diagnosis)

        arena_results = self.run_arena(
            hypotheses, df, province, target_type, target_col
        )

        best = self.select_best(arena_results, diagnosis, baseline)

        if "error" not in best:
            scenario = diagnosis[0].get("scenario", "") if diagnosis else ""
            success = best.get("improvement", 0) > 0
            self.record_strategy(
                {
                    "name": best.get("selected", {}).get("name", ""),
                    "desc": best.get("selected", {}).get("description", ""),
                    "improvement": {
                        "before": best.get("mape_before", 0),
                        "after": best.get("mape_after", 0),
                    },
                },
                scenario=scenario,
                success=success,
            )

        return {
            "selected_strategy": best.get("selected", {}).get("description", ""),
            "mape_before": best.get("mape_before"),
            "mape_after": best.get("mape_after"),
            "improvement": best.get("improvement", 0),
            "knowledge_updated": True,
            "hypotheses_tested": len([r for r in arena_results if "error" not in r]),
        }

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
