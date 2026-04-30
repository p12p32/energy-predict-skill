"""improver.py — 学习型优化引擎 + 策略知识库（闭环版）"""
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.db import DorisDB
from src.trainer import Trainer
from src.backtester import Backtester
from src.strategy_executor import StrategyExecutor


class Improver:
    SEED_STRATEGIES = [
        {
            "name": "polynomial_features",
            "description": "加入多项式特征（temperature²等）",
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
            "description": "关键指标滑动窗口均值作为新特征",
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

    # 连续失败 N 次则退役
    RETIRE_AFTER_FAILURES = 5
    # 成功率低于此值则惩罚
    LOW_SUCCESS_THRESHOLD = 0.3

    def __init__(self, db: DorisDB = None,
                 trainer: Trainer = None,
                 backtester: Backtester = None):
        self.db = db or DorisDB()
        self.trainer = trainer or Trainer()
        self.backtester = backtester or Backtester(self.trainer)
        self.executor = StrategyExecutor()

    # ================================================================
    #  假设生成（已集成知识库过滤）
    # ================================================================

    def generate_hypotheses(self, diagnosis: List[Dict]) -> List[Dict]:
        """基于诊断 + 历史知识库生成假设池.

        三步:
        1. 关键词匹配种子策略
        2. 查知识库 → 读过失败的、加权成功的
        3. 排序输出
        """
        # ── 提取关键词 ──
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

        # ── 匹配 + 知识库加权 ──
        knowledge = self._get_knowledge_map(keywords)

        hypotheses = []
        for seed in self.SEED_STRATEGIES:
            applicable = seed.get("applicable", [])

            # 判定是否匹配
            matched = False
            source = ""
            if any(kw in applicable for kw in keywords):
                matched = True
                source = "seed_match"
            elif suggested & set(applicable):
                matched = True
                source = "suggested_match"
            elif "overall" in keywords or "critical" in keywords:
                matched = True
                source = "critical_explore"

            if not matched:
                continue

            # 检查知识库
            sig = seed["name"] + json.dumps(seed.get("params", {}), sort_keys=True)
            k_info = knowledge.get(sig, {})

            # 退役检查
            if k_info.get("retired"):
                continue

            # 连续失败过多 → 跳过
            if (k_info.get("applied_count", 0) - k_info.get("success_count", 0)
                    >= self.RETIRE_AFTER_FAILURES):
                self._retire_strategy(sig)
                continue

            # 计算知识库加权分数
            knowledge_score = self._calc_knowledge_score(k_info)

            hypotheses.append({
                "name": seed["name"],
                "description": seed["description"],
                "params": dict(seed.get("params", {})),
                "source": source,
                "knowledge_score": knowledge_score,
                "knowledge_info": k_info,
            })

        # 去重
        seen = set()
        unique = []
        for h in hypotheses:
            sig = h["name"] + json.dumps(h["params"], sort_keys=True)
            if sig not in seen:
                seen.add(sig)
                unique.append(h)

        # 按知识库分数排序（分数高的优先），取前 12
        unique.sort(key=lambda h: h["knowledge_score"], reverse=True)
        return unique[:12]

    def _get_knowledge_map(self, _keywords: set) -> Dict[str, Dict]:
        """从知识库加载所有策略的历史表现."""
        if not self.db.table_exists("strategy_knowledge"):
            return {}

        sql = """
            SELECT strategy_hash, applied_count, success_count,
                   avg_improvement, best_scenario, retired, last_effect
            FROM strategy_knowledge
        """
        try:
            df = self.db.query(sql)
            result = {}
            for _, row in df.iterrows():
                result[row["strategy_hash"]] = {
                    "applied_count": int(row["applied_count"]),
                    "success_count": int(row["success_count"]),
                    "avg_improvement": float(row["avg_improvement"]),
                    "retired": bool(row["retired"]),
                }
            return result
        except Exception:
            return {}

    def _calc_knowledge_score(self, k_info: Dict) -> float:
        """基于历史数据计算策略的推荐分数.

        新策略(从未用过): 基础分 0.5（中性）
        成功过的策略: 基础分 + 成功率加成 + 改进幅度加成
        失败过的策略: 惩罚
        """
        applied = k_info.get("applied_count", 0)
        if applied == 0:
            return 0.5  # 新策略，中性分

        success = k_info.get("success_count", 0)
        rate = success / applied
        avg_imp = k_info.get("avg_improvement", 0)

        score = 0.6  # 基础分
        score += rate * 0.3  # 成功率贡献（最多 +0.3）
        score += min(avg_imp, 0.1)  # 改进幅度贡献（最多 +0.1）

        # 低成功率惩罚
        if applied >= 3 and rate < self.LOW_SUCCESS_THRESHOLD:
            score -= 0.4

        return max(0.0, min(1.0, score))

    # ================================================================
    #  实验执行（集成 StrategyExecutor）
    # ================================================================

    FEATURE_STRATEGIES = {"polynomial_features", "dayofweek_interaction",
                          "rolling_window_features"}
    DATA_STRATEGIES = {"recent_upsample", "extreme_oversample",
                       "holiday_oversample", "shorter_window"}

    def run_experiment(self, hypothesis: Dict, df: pd.DataFrame,
                       province: str, target_type: str,
                       target_col: str = "value") -> Dict:
        """执行单个假设的实验。

        修复时序泄漏:
        - 先按时间切分 train/test
        - 数据级变换(过采样/窗口)只应用于训练集
        - 特征级变换(多项式/交互/滑动)同时应用于训练集和测试集(保证列一致)
        - 不再 shuffle，保持时序顺序
        """
        try:
            name = hypothesis.get("name", "")
            df = df.sort_values("dt").reset_index(drop=True)

            # ── 1. 先做时序切分 ──
            total = len(df)
            test_steps = min(96, total // 5)
            train_df_orig = df.iloc[:-test_steps].copy()
            test_df_orig = df.iloc[-test_steps:].copy()

            # ── 2. 根据策略类型，分别变换训练集和测试集 ──
            if name in self.FEATURE_STRATEGIES:
                train_df = self.executor.execute(train_df_orig, hypothesis)
                test_df = self.executor.execute(test_df_orig, hypothesis)
            elif name in self.DATA_STRATEGIES:
                train_df = self.executor.execute(train_df_orig, hypothesis)
                test_df = test_df_orig  # 测试集不加过采样/窗口过滤
            else:
                # 模型级/后处理策略: 数据不变
                train_df = train_df_orig
                test_df = test_df_orig

            # ── 3. 快速训练 ──
            model_params = self._extract_model_params(hypothesis)

            bt_result = self.trainer.quick_train(
                train_df, province, target_type, target_col,
                params=model_params,
            )

            # ── 4. 预测评估 ──
            test_features = test_df[bt_result["feature_names"]].values
            pred = bt_result["model"].predict(test_features)
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
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        except Exception as e:
            return {
                "hypothesis_id": self._hash(hypothesis["name"]),
                "hypothesis": hypothesis,
                "error": str(e),
            }

    def _extract_model_params(self, hypothesis: Dict) -> Dict:
        """从假设中提取模型参数（仅模型级策略有）."""
        name = hypothesis.get("name", "")
        params = hypothesis.get("params", {})

        if name == "switch_to_catboost":
            # CatBoost 的等效参数（实际不会生效在 LightGBM，但标记意图）
            return {"boosting_type": "ordered"}

        return {}

    # ================================================================
    #  竞技场与选择
    # ================================================================

    def run_arena(self, hypotheses: List[Dict], df: pd.DataFrame,
                  province: str, target_type: str,
                  target_col: str = "value") -> List[Dict]:
        """竞技场: 测试 N 个假设，返回排序后的结果."""
        results = []
        for h in hypotheses:
            result = self.run_experiment(h, df, province, target_type, target_col)
            results.append(result)

        return sorted(results, key=lambda r: r.get("mape", float("inf")))

    def select_best(self, arena_results: List[Dict],
                     diagnosis: List[Dict],
                     baseline: Dict) -> Dict:
        """从竞技场选择最优策略.

        评分 = 主问题 MAPE 降幅 × 严重性权重 − 复杂度罚分 − 知识库罚分
        """
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

            # 复杂度罚分
            complexity_penalty = self._calc_complexity_penalty(r)

            # 历史罚分
            history_penalty = self._calc_history_penalty(r)

            # 问题权重
            weight = 1.0
            for scenario, severity in pain_points.items():
                weight = max(weight, severity_weight.get(severity, 1.0))

            score = improvement * weight - complexity_penalty - history_penalty

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

    def _calc_complexity_penalty(self, result: Dict) -> float:
        h = result.get("hypothesis", {})
        name = h.get("name", "")
        penalties = {
            "polynomial_features": 0.003,
            "switch_to_catboost": 0.005,
            "province_independent_model": 0.004,
        }
        return penalties.get(name, 0.001)

    def _calc_history_penalty(self, result: Dict) -> float:
        """对于历史上失败率高的策略，加上罚分."""
        h = result.get("hypothesis", {})
        info = h.get("knowledge_info", {})
        applied = info.get("applied_count", 0)
        if applied >= 3:
            success = info.get("success_count", 0)
            rate = success / applied
            if rate < self.LOW_SUCCESS_THRESHOLD:
                return (self.LOW_SUCCESS_THRESHOLD - rate) * 0.02
        return 0.0

    # ================================================================
    #  知识库操作
    # ================================================================

    def record_strategy(self, strategy: Dict, scenario: str = "",
                        success: bool = True):
        """记录策略到知识库."""
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

    def _retire_strategy(self, strategy_sig: str):
        """退役一个策略."""
        try:
            self.db.execute(
                f"UPDATE strategy_knowledge SET retired = TRUE "
                f"WHERE strategy_hash = '{strategy_sig}'"
            )
        except Exception:
            pass

    def query_knowledge(self, scenario: str) -> List[Dict]:
        """查询知识库."""
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
        return (
            self.db.query(sql).to_dict("records")
            if self.db.table_exists("strategy_knowledge")
            else []
        )

    # ================================================================
    #  完整一轮优化
    # ================================================================

    def improve(self, diagnosis: List[Dict], df: pd.DataFrame,
                province: str, target_type: str,
                target_col: str = "value",
                baseline: Dict = None) -> Dict:
        """完整一轮自主优化.

        1. 生成假设（知识库加权过滤）
        2. 竞技场测试（StrategyExecutor 应用变换）
        3. 选最优（含历史罚分）
        4. 记录知识库
        """
        if baseline is None:
            baseline = {"overall_mape": 0.10}

        hypotheses = self.generate_hypotheses(diagnosis)

        if not hypotheses:
            return {
                "selected_strategy": "",
                "mape_before": baseline.get("overall_mape"),
                "mape_after": baseline.get("overall_mape"),
                "improvement": 0,
                "knowledge_updated": False,
                "hypotheses_tested": 0,
            }

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
            "hypotheses_tested": len([r for r in arena_results if "error" not in r]),
            "knowledge_updated": True,
        }

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
