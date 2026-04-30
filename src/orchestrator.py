"""orchestrator.py — 总调度器：管理循环 A/B/C"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.core.config import get_provinces, get_types
from src.core.data_source import FileSource
from src.data.features import FeatureStore, FeatureEngineer
from src.data.fetcher import DataFetcher
from src.ml.trainer import Trainer
from src.ml.predictor import Predictor
from src.evolve.validator import Validator
from src.evolve.backtester import Backtester
from src.evolve.analyzer import Analyzer
from src.evolve.improver import Improver

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.source = FileSource()
        self.store = FeatureStore(self.source)
        self.trainer = Trainer()
        self.backtester = Backtester(self.trainer)
        self.predictor = Predictor(self.trainer, self.store)
        self.validator = Validator()
        self.analyzer = Analyzer()
        self.improver = Improver(self.source, self.trainer, self.backtester)
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()

        self._validator_history: List[Dict] = []

    def setup(self):
        logger.info("初始化系统...")
        self.source.setup()
        logger.info("数据存储已就绪")

    def build_features(self, province: str = None,
                       start_date: str = None,
                       end_date: str = None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        provinces = [province] if province else get_provinces()

        for p in provinces:
            for t in get_types():
                logger.info(f"构建特征: {p}/{t} ...")
                raw = self.store.load_raw_data(p, t, start_date, end_date)
                if raw.empty:
                    logger.warning(f"  无数据: {p}/{t}")
                    continue

                features = self.engineer.build_features_from_raw(raw)

                try:
                    weather = self.fetcher.fetch_weather(
                        p, start_date, end_date, mode="historical"
                    )
                    if not weather.empty:
                        features = self.engineer.merge_weather(features, weather)
                except Exception as e:
                    logger.warning(f"  气象数据合并失败: {e}")

                count = self.store.insert_features(features)
                logger.info(f"  {p}/{t}: 写入 {count} 行")

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24) -> Dict:
        logger.info(f"预测: {province}/{target_type}, {horizon_hours}h")
        df = self.predictor.predict(province, target_type, horizon_hours)
        return {
            "province": province,
            "type": target_type,
            "horizon_hours": horizon_hours,
            "n_predictions": len(df),
            "sample": df.head(5)[["dt", "p50"]].to_dict("records") if not df.empty else [],
        }

    def run_validation_cycle(self, province: str,
                              target_type: str) -> Dict:
        end = datetime.now()
        start = end - timedelta(hours=48)

        predictions = self.store.load_predictions(
            province, target_type,
            start.strftime("%Y-%m-%d"),
            (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        actuals = self.store.load_raw_data(
            province, target_type,
            start.strftime("%Y-%m-%d"),
            (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if predictions.empty or actuals.empty:
            return {"status": "no_data"}

        report = self.validator.validate(predictions, actuals, "p50")
        self._validator_history.append(report)

        logger.info(
            f"验证 {province}/{target_type}: "
            f"MAPE={report['metrics'].get('mape')}, "
            f"triggered={report['triggered']}"
        )

        if report["triggered"]:
            return self._run_improvement_cycle(
                province, target_type, report
            )

        return {"status": "ok", "report": report}

    def run_backtest_cycle(self, province: str,
                            target_type: str) -> Dict:
        end = datetime.now()
        start = end - timedelta(days=120)

        df = self.store.load_features(
            province, target_type,
            start.strftime("%Y-%m-%d"),
            (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if df.empty:
            return {"status": "no_data"}

        result = self.backtester.evaluate_model(
            df, train_window_days=14, test_window_hours=24,
            province=province, target_type=target_type,
        )

        diagnosis = self.analyzer.diagnose(
            result, validator_history=self._validator_history
        )

        logger.info(
            f"回测 {province}/{target_type}: "
            f"MAPE={result.get('overall_mape')}, "
            f"diagnoses={len(diagnosis)}"
        )

        return {
            "status": "ok",
            "mape": result.get("overall_mape"),
            "by_season": result.get("by_season", {}),
            "diagnoses": diagnosis,
        }

    def _run_improvement_cycle(self, province: str,
                                target_type: str,
                                validation_report: Dict) -> Dict:
        """循环 C: improver → trainer 实验 → backtester 裁决.

        修复数据泄漏: 回测和实验使用独立时间窗口。
        - 回测窗口 (60~20 天前): 建立基线
        - 实验窗口 (20~0 天前): improver 实验
        """
        end = datetime.now()
        bt_start = end - timedelta(days=60)
        bt_end = end - timedelta(days=20)
        exp_start = bt_end
        exp_end = end

        # ── 回测窗口：建立基线 ──
        bt_df = self.store.load_features(
            province, target_type,
            bt_start.strftime("%Y-%m-%d"),
            (bt_end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if bt_df.empty:
            return {"status": "no_data"}

        bt_result = self.backtester.evaluate_model(
            bt_df, train_window_days=14, test_window_hours=24,
            province=province, target_type=target_type,
        )

        diagnosis = self.analyzer.diagnose(bt_result)

        # ── 实验窗口：improver 在独立数据上实验 ──
        exp_df = self.store.load_features(
            province, target_type,
            exp_start.strftime("%Y-%m-%d"),
            (exp_end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if exp_df.empty:
            return {
                "status": "backtest_only",
                "mape": bt_result.get("overall_mape"),
                "diagnoses": diagnosis,
            }

        baseline_mape = bt_result.get("overall_mape", 0.10)
        improvement = self.improver.improve(
            diagnosis, exp_df, province, target_type,
            baseline={"overall_mape": baseline_mape},
        )

        logger.info(
            f"优化 {province}/{target_type}: "
            f"策略={improvement.get('selected_strategy')}, "
            f"基线MAPE={baseline_mape}, "
            f"实验后MAPE={improvement.get('mape_after')}"
        )

        # ── 改善显著 → 全量重训练 ──
        if improvement.get("improvement", 0) > 0.03:
            logger.info("触发全量重训练...")
            full_df = self.store.load_features(
                province, target_type,
                (end - timedelta(days=90)).strftime("%Y-%m-%d"),
                (end + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if not full_df.empty:
                self.trainer.train(
                    full_df, province, target_type,
                    target_col="value",
                )

        return {
            "status": "improved",
            "validation": validation_report,
            "baseline_mape": baseline_mape,
            "improvement": improvement,
            "diagnoses": diagnosis,
        }

    def train_all(self):
        end = datetime.now()
        start = end - timedelta(days=90)

        for province in get_provinces():
            for target_type in get_types():
                logger.info(f"训练: {province}/{target_type}")

                df = self.store.load_features(
                    province, target_type,
                    start.strftime("%Y-%m-%d"),
                    (end + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                if df.empty:
                    logger.warning(f"  无数据: {province}/{target_type}")
                    continue

                result = self.trainer.train(df, province, target_type)
                logger.info(
                    f"  完成: n={result['n_samples']}, "
                    f"features={result['n_features']}"
                )

    # ── Tier 2 新增方法 ──

    def validate_data(self, province: str, target_type: str) -> Dict:
        """数据校验: 检查缺失率、时间连续性、类型正确性."""
        end = datetime.now()
        start = end - timedelta(days=365)
        df = self.store.load_raw_data(province, target_type,
                                       start.strftime("%Y-%m-%d"),
                                       (end + timedelta(days=1)).strftime("%Y-%m-%d"))
        if df.empty:
            return {"error": "no_data"}

        issues = []
        total = len(df)

        # 缺失率
        null_rate = df["value"].isna().mean() if "value" in df.columns else 0
        if null_rate > 0:
            issues.append(f"缺失率: {null_rate:.2%}")

        # 时间连续性 (15分钟粒度，最大间隔应<30分钟)
        if "dt" in df.columns:
            gaps = df["dt"].diff().dropna()
            large_gaps = gaps[gaps > pd.Timedelta(minutes=30)]
            if len(large_gaps) > 0:
                issues.append(f"时间间隙>30min: {len(large_gaps)}处, 最大{gaps.max()}")

        # 零值/负值
        if "value" in df.columns:
            neg_rate = (df["value"] <= 0).mean()
            if neg_rate > 0:
                issues.append(f"零值/负值比例: {neg_rate:.2%}")

        return {
            "province": province,
            "type": target_type,
            "total_rows": total,
            "date_range": f"{df['dt'].min()} ~ {df['dt'].max()}",
            "null_rate": round(null_rate, 4),
            "issues": issues,
            "status": "ok" if not issues else "has_issues",
        }

    def explain(self, province: str, target_type: str) -> Dict:
        """特征重要性分析."""
        importance = self.trainer.feature_importance(province, target_type)
        versions = self.trainer.list_versions(province, target_type)
        return {
            "province": province,
            "type": target_type,
            "feature_importance": importance[:15],
            "model_versions": versions,
        }

    def export(self, province: str, target_type: str,
               fmt: str = "json", output: str = None) -> str:
        """导出预测结果为 JSON 或 CSV."""
        end = datetime.now()
        start = end - timedelta(hours=48)
        df = self.store.load_predictions(province, target_type,
                                          start.strftime("%Y-%m-%d"),
                                          (end + timedelta(days=1)).strftime("%Y-%m-%d"))
        if df.empty:
            return "无预测数据"

        if output is None:
            output = f"{province}_{target_type}_predictions.{fmt}"

        if fmt == "csv":
            df.to_csv(output, index=False)
        else:
            df.to_json(output, orient="records", force_ascii=False, indent=2)

        return f"已导出 {len(df)} 条记录到 {output}"

    def rollback_model(self, province: str, target_type: str) -> Dict:
        """回滚模型到上一个版本."""
        return self.trainer.rollback(province, target_type)

    def chart(self, province: str, target_type: str,
              hours: int = 24) -> str:
        """生成终端 ASCII 折线图."""
        import math

        result = self.predict(province, target_type, hours)
        samples = result.get("sample", [])
        if not samples:
            return "无预测数据"

        values = [s["p50"] for s in samples]
        vmin, vmax = min(values), max(values)
        height, width = 12, 60

        if vmax == vmin:
            return "所有预测值相同，无法绘图"

        chart = f"\n  {province} {target_type} 预测 ({hours}h)\n"
        chart += f"  {'─' * (width + 8)}\n"

        for row in range(height - 1, -1, -1):
            line = ""
            for col in range(min(len(values), width)):
                val = values[col]
                y = int((val - vmin) / (vmax - vmin) * (height - 1))
                line += "█" if y >= row else " "
            val_label = vmin + (vmax - vmin) * row / (height - 1)
            chart += f"  {val_label:>10,.0f} │{line}\n"

        chart += f"  {'─' * (width + 12)}\n"
        return chart
