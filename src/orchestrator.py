"""orchestrator.py — 总调度器：管理循环 A/B/C"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.db import DorisDB
from src.config_loader import get_provinces, get_types
from src.feature_store import FeatureStore, FeatureEngineer
from src.data_fetcher import DataFetcher
from src.trainer import Trainer
from src.predictor import Predictor
from src.validator import Validator
from src.backtester import Backtester
from src.analyzer import Analyzer
from src.improver import Improver

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.db = DorisDB()
        self.store = FeatureStore(self.db)
        self.trainer = Trainer()
        self.backtester = Backtester(self.trainer)
        self.predictor = Predictor(self.trainer, self.store)
        self.validator = Validator()
        self.analyzer = Analyzer()
        self.improver = Improver(self.db, self.trainer, self.backtester)
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()

        self._validator_history: List[Dict] = []

    def setup(self):
        logger.info("初始化系统...")
        self.store.ensure_tables()
        logger.info("表结构已就绪")

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

        predictions = self.db.query(
            f"SELECT * FROM energy_predictions "
            f"WHERE province='{province}' AND type='{target_type}' "
            f"AND dt >= '{start.strftime('%Y-%m-%d %H:%M:%S')}'"
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
