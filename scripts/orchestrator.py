"""orchestrator.py — 总调度器 (type 三段式 + value_type 感知)"""
import glob
import hashlib
import logging
import os

import scripts.core.cleanup  # noqa — signal/atexit 注册，防止僵尸进程
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from scripts.core.config import (
    get_provinces, get_base_types, load_config,
    validate_province_and_type, get_available_types,
    get_available_actual_types, parse_type, TypeInfo,
)
from scripts.core.data_source import FileSource, MemorySource
from scripts.data.features import FeatureStore, FeatureEngineer
from scripts.ml.trainer import Trainer
from scripts.ml.predictor import Predictor
from scripts.evolve.validator import Validator
from scripts.evolve.backtester import Backtester
from scripts.evolve.analyzer import Analyzer
from scripts.evolve.improver import Improver

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        cfg = load_config()
        ds_type = cfg.get("data_source", "file")
        if ds_type == "doris":
            from scripts.core.db import DorisDB
            self.source = DorisDB()
        elif ds_type == "memory":
            self.source = MemorySource()
        else:
            self.source = FileSource()
        self.store = FeatureStore(self.source)
        self.trainer = Trainer()
        self.backtester = Backtester(self.trainer)
        pred_cfg = cfg.get("predictor", {})
        self._lookback_days = pred_cfg.get("lookback_days", 60)
        self.predictor = Predictor(self.trainer, self.store)
        self.validator = Validator()
        self.analyzer = Analyzer()
        self.improver = Improver(self.source, self.trainer, self.backtester)
        self.engineer = FeatureEngineer()
        self._validator_history: List[Dict] = []

    def setup(self):
        logger.info("初始化系统...")
        self.source.setup()
        logger.info("数据存储已就绪")

    def _find_latest_date(self, province: str, target_type: str):
        """扫描特征库，返回该 province/type 的最新数据日期."""
        import glob, os
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                ".energy_data", "features")
        pattern = os.path.join(base_dir, f"{province}_{target_type}_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        try:
            df = pd.read_parquet(files[-1], columns=["dt"])
            if not df.empty:
                return pd.to_datetime(df["dt"].max())
        except Exception:
            pass
        return None

    def _load_latest_features(self, province: str, target_type: str,
                               days: int = 30):
        """加载该 province/type 最近 N 天的特征数据 (自动探测最新日期)."""
        import glob, os
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                ".energy_data", "features")
        pattern = os.path.join(base_dir, f"{province}_{target_type}_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            return pd.DataFrame()
        try:
            df = pd.read_parquet(files[-1])
            if "dt" in df.columns:
                df["dt"] = pd.to_datetime(df["dt"])
                latest = df["dt"].max()
                start = latest - timedelta(days=days)
                return df[(df["dt"] >= start) & (df["dt"] <= latest)]
        except Exception:
            pass
        return pd.DataFrame()

    # ================================================================
    # 特征版本追踪 (增量 vs 全量重建)
    # ================================================================

    def _get_features_dir(self) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            ".energy_data", "features")

    def _hash_feature_code(self) -> str:
        """对特征工程相关源文件做 hash，代码变更时自动触发全量重建."""
        src_dir = os.path.dirname(__file__)
        files_to_hash = [
            os.path.join(src_dir, "data", "features.py"),
            os.path.join(src_dir, "data", "db_importer.py"),
        ]
        h = hashlib.sha256()
        for fp in files_to_hash:
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    h.update(f.read())
        return h.hexdigest()

    def _get_version_path(self) -> str:
        return os.path.join(self._get_features_dir(), ".feature_version")

    def _check_version(self) -> bool:
        """返回 True 若特征代码未变更."""
        vp = self._get_version_path()
        if not os.path.exists(vp):
            return False
        try:
            with open(vp, "r") as f:
                stored = f.read().strip()
            return stored == self._hash_feature_code()
        except Exception:
            return False

    def _save_version(self):
        vp = self._get_version_path()
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        with open(vp, "w") as f:
            f.write(self._hash_feature_code())

    def _get_existing_max_date(self, provinces: List[str]) -> Optional[datetime]:
        """扫描所有已有特征文件，返回最早的 max(dt)（按省份聚合）."""
        earliest = None
        features_dir = self._get_features_dir()
        for p in provinces:
            files = sorted(glob.glob(os.path.join(features_dir, f"{p}_*.parquet")))
            if not files:
                return None  # 该省无任何特征 → 需全量
            for f in files:
                try:
                    df = pd.read_parquet(f, columns=["dt"])
                    if not df.empty:
                        m = pd.to_datetime(df["dt"].max())
                        if earliest is None or m < earliest:
                            earliest = m
                except Exception:
                    pass
        return earliest

    def _resolve_types(self, province: str, target_type: str) -> List[str]:
        """将用户输入的 type 解析为实际的完整 type 列表.

        "出力" → 扫描数据，展开为 ["出力_风电_实际", "出力_光伏_实际", ...]
        "出力_风电" → ["出力_风电_实际"]
        "出力_风电_实际" → ["出力_风电_实际"]
        """
        ti = parse_type(target_type)

        # 完整三段式: 原始字符串含两个下划线 → 直接返回
        parts = target_type.split("_")
        if len(parts) >= 3:
            return [target_type]

        # 需要展开
        available = get_available_actual_types(province)
        if ti.sub:
            # 有子类型但无 value_type → 匹配 base_sub
            prefix = f"{ti.base}_{ti.sub}"
            candidates = [t for t in available if t.startswith(prefix)]
        else:
            # 只有基类 → 匹配所有
            candidates = [t for t in available if t.startswith(ti.base)]

        return candidates if candidates else [target_type]

    def build_features(self, province: str = None,
                       start_date: str = None,
                       end_date: str = None,
                       force_rebuild: bool = False):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        provinces = [province] if province else get_provinces()

        # ── 决定全量重建 vs 增量追加 ──
        version_ok = self._check_version()
        existing_max = self._get_existing_max_date(provinces) if not force_rebuild else None

        if force_rebuild or not version_ok or existing_max is None:
            # 全量重建
            if force_rebuild:
                logger.info("强制全量重建 (--force-rebuild)")
            elif not version_ok:
                logger.info("特征代码已变更，触发全量重建")
            else:
                logger.info("无已有特征，首次全量构建")
            self.store.source.clear_features(provinces)
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            # 增量追加: 从已有最新日期 - 30 天开始 (覆盖 rolling window 特征)
            inc_start = existing_max - timedelta(days=30)
            if start_date is not None:
                # 用户显式指定 start_date → 取更早的
                user_start = datetime.strptime(start_date, "%Y-%m-%d")
                inc_start = min(inc_start, user_start)
            start_date = inc_start.strftime("%Y-%m-%d")
            logger.info("增量特征构建: %s ~ %s (已有数据截止 %s)",
                        start_date, end_date,
                        existing_max.strftime("%Y-%m-%d"))

        # 加载气象数据
        weather_cache: Dict[str, pd.DataFrame] = {}
        for p in provinces:
            weather_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                ".energy_data", "weather", f"{p}_weather.csv",
            )
            if os.path.exists(weather_path):
                w = pd.read_csv(weather_path)
                w["dt"] = pd.to_datetime(w["dt"])
                weather_cache[p] = w
                logger.info(f"加载气象数据: {p} ({len(w)} 行)")

        # 第一遍: 按 (province, type) 构建独立特征 (只用实际值)
        built: Dict[str, pd.DataFrame] = {}
        for p in provinces:
            available_types = get_available_actual_types(p)
            if not available_types:
                available_types = get_available_types(p)

            for t in available_types:
                logger.info(f"构建特征: {p}/{t} ...")
                raw = self.store.load_raw_data(p, t, start_date, end_date,
                                               value_type_filter="实际")
                if raw.empty:
                    logger.warning(f"  无数据: {p}/{t}")
                    continue

                features = self.engineer.build_features_from_raw(raw, value_type_filter="实际")

                # 合并气象数据 (含省份纬度用于 cloud_factor)
                if p in weather_cache:
                    coords = load_config().get("province_coords", {})
                    lat = coords.get(p, {}).get("lat") if coords else None
                    features = self.engineer.merge_weather(features, weather_cache[p], lat=lat)

                # 注入预测误差特征 (如果有历史预测)
                try:
                    preds = self.store.load_predictions(p, t, start_date, end_date)
                    if not preds.empty:
                        features = self.engineer.add_prediction_error_features(features, preds)
                except Exception as e:
                    logger.debug("预测误差特征注入跳过 (%s/%s): %s", p, t, e)

                built[f"{p}/{t}"] = features

        # 第二遍: 交叉特征 v2
        for p in provinces:
            p_built = {t: df for key, df in built.items()
                      if key.startswith(f"{p}/")
                      for t in [key.split("/", 1)[1]]}
            if len(p_built) < 2:
                continue
            logger.info(f"交叉特征 v2: {p} ({', '.join(p_built.keys())})")
            for t, features in p_built.items():
                built[f"{p}/{t}"] = self.engineer.add_cross_type_features_v2(
                    features, p_built, p,
                )

        # 第二遍半: 电价专属特征 (供需紧密度、可再生能源冲击、波动率)
        for key, features in built.items():
            p, t = key.split("/", 1)
            if t.startswith("电价"):
                # 找同省份的所有 built 作为 siblings
                siblings = {tk: df for k, df in built.items()
                           if k.startswith(f"{p}/") and k != key
                           for tk in [k.split("/", 1)[1]]}
                built[key] = self.engineer.add_price_features(features, siblings)

        # 第三遍: 写入
        for key, features in built.items():
            p, t = key.split("/", 1)
            count = self.store.insert_features(features)
            logger.info(f"  {p}/{t}: 写入 {count} 行")

        # 保存特征代码版本 hash
        self._save_version()
        logger.info("特征版本已记录: %s", self._hash_feature_code()[:12])

    def predict_engineered(self, province: str, target_type: str,
                           horizon_hours: int = 24) -> Dict:
        """工程化预测入口: 一步完成 加载→健康检查→预测→校准→结构化结果.

        返回包含 MAPE 历史、特征覆盖度、区间覆盖率等诊断信息.
        """
        result = self.predict(province, target_type, horizon_hours)

        # 附加诊断信息
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            for rt in result.get("resolved_types", [target_type]):
                features_df = self.store.load_features(
                    province, rt,
                    start.strftime("%Y-%m-%d"),
                    (end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    value_type_filter="实际",
                )
                if features_df.empty:
                    features_df = self._load_latest_features(province, rt, days=30)

                if not features_df.empty and len(features_df) >= 96 * 7:
                    # 特征覆盖度
                    weather_cols = ["temperature", "humidity", "wind_speed",
                                    "solar_radiation", "precipitation", "pressure"]
                    present = [c for c in weather_cols if c in features_df.columns and features_df[c].notna().sum() > 0]
                    result["feature_coverage"] = {
                        "total_features": len(features_df.columns),
                        "weather_present": len(present),
                        "weather_missing": len(weather_cols) - len(present),
                        "has_economic": "coal_price" in features_df.columns,
                    }
                    # 数据质量
                    if "quality_flag" in features_df.columns:
                        bad_rate = (features_df["quality_flag"] >= 2).mean()
                        result["data_quality"] = {
                            "anomaly_rate": round(float(bad_rate), 4),
                            "status": "ok" if bad_rate < 0.05 else "degraded",
                        }
                    break
        except Exception as e:
            logger.warning("诊断信息收集失败: %s", e)

        # 区间覆盖率估计 (基于最近历史)
        try:
            bt = self.backtester.evaluate_model(
                self._load_latest_features(province, target_type, days=30),
                train_window_days=7, test_window_hours=24,
                province=province, target_type=target_type,
            )
            if "overall_mape" in bt:
                result["recent_mape"] = bt["overall_mape"]
        except Exception:
            pass

        return result

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24) -> Dict:
        validate_province_and_type(province, target_type)
        logger.info(f"预测: {province}/{target_type}, {horizon_hours}h")

        # 展开 type (base → full type list)
        resolved_types = self._resolve_types(province, target_type)

        # 预测前轻量回测 + 健康检查
        health = None
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            for rt in resolved_types:
                features_df = self.store.load_features(
                    province, rt,
                    start.strftime("%Y-%m-%d"),
                    (end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    value_type_filter="实际",
                )
                # 如果最近30天没数据，尝试用特征库中最新数据回退
                if features_df.empty:
                    features_df = self._load_latest_features(province, rt, days=30)
                if not features_df.empty and len(features_df) >= 96 * 7:
                    bt = self.backtester.evaluate_model(
                        features_df, train_window_days=7, test_window_hours=24,
                        province=province, target_type=rt,
                    )
                    mape = bt.get("overall_mape")
                    if mape is not None:
                        health = {"mape": round(mape, 4),
                                  "status": "ok" if mape < 0.05 else "degraded"}
                        if health["status"] == "degraded":
                            logger.warning(f"模型精度退化 {rt} MAPE={mape:.2%}，建议 /improve")
        except Exception as e:
            logger.warning("健康检查失败 (%s/%s): %s", province, target_type, e)

        # 预测每个展开的类型
        all_samples = []
        total = 0
        for rt in resolved_types:
            try:
                df = self.predictor.predict(province, rt, horizon_hours)
                all_samples.extend(df.head(5)[["dt", "p50"]].to_dict("records"))
                total += len(df)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("预测跳过 %s/%s: %s", province, rt, e)

        result = {
            "province": province,
            "type": target_type,
            "resolved_types": resolved_types,
            "horizon_hours": horizon_hours,
            "n_predictions": total,
            "sample": all_samples[:10],
        }
        if health is not None:
            result["health"] = health
        return result

    def run_validation_cycle(self, province: str,
                              target_type: str) -> Dict:
        end = self._find_latest_date(province, target_type) or datetime.now()
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
            value_type_filter="实际",
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
        end = self._find_latest_date(province, target_type) or datetime.now()
        start = end - timedelta(days=120)

        df = self.store.load_features(
            province, target_type,
            start.strftime("%Y-%m-%d"),
            (end + timedelta(days=1)).strftime("%Y-%m-%d"),
            value_type_filter="实际",
        )

        if df.empty:
            return {"status": "no_data"}

        result = self.backtester.evaluate_model(
            df, train_window_days=self._lookback_days, test_window_hours=24,
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
        end = self._find_latest_date(province, target_type) or datetime.now()
        bt_start = end - timedelta(days=60)
        bt_end = end - timedelta(days=20)
        exp_start = bt_end
        exp_end = end

        # 回测窗口
        bt_df = self.store.load_features(
            province, target_type,
            bt_start.strftime("%Y-%m-%d"),
            (bt_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            value_type_filter="实际",
        )

        if bt_df.empty:
            return {"status": "no_data"}

        bt_result = self.backtester.evaluate_model(
            bt_df, train_window_days=self._lookback_days, test_window_hours=24,
            province=province, target_type=target_type,
        )

        diagnosis = self.analyzer.diagnose(bt_result)

        # 实验窗口
        exp_df = self.store.load_features(
            province, target_type,
            exp_start.strftime("%Y-%m-%d"),
            (exp_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            value_type_filter="实际",
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

        if improvement.get("improvement", 0) > 0.03:
            logger.info("触发全量重训练...")
            full_df = self.store.load_features(
                province, target_type,
                (end - timedelta(days=90)).strftime("%Y-%m-%d"),
                (end + timedelta(days=1)).strftime("%Y-%m-%d"),
                value_type_filter="实际",
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

    def _scan_available(self) -> list:
        """返回有数据的 (province, type) 列表."""
        available = []
        for province in get_provinces():
            for target_type in get_available_actual_types(province):
                if not target_type:
                    continue
                df = self.store.load_features(
                    province, target_type, None, None,
                    value_type_filter="实际",
                )
                if not df.empty and len(df) >= 96:
                    available.append((province, target_type))
        return available

    def train_all(self):
        available = self._scan_available()
        if not available:
            logger.warning("无可用数据，跳过训练")
            return

        logger.info(f"扫描到 {len(available)} 组有数据的 (province, type)，开始全量训练...")

        for province, target_type in available:
            logger.info(f"训练: {province}/{target_type}")
            df = self.store.load_features(
                province, target_type, None, None,
                value_type_filter="实际",
            )
            result = self.trainer.train(df, province, target_type)
            logger.info(
                f"  LGB 完成: n={result['n_samples']}, "
                f"features={result['n_features']}"
            )
            # XGBoost 联合训练
            try:
                xgb_result = self.trainer.xgboost_train(df, province, target_type)
                logger.info(f"  XGB 完成: n={xgb_result['n_samples']}")
            except Exception as e:
                logger.warning(f"  XGB 训练跳过 ({province}/{target_type}): {e}")

    def validate_data(self, province: str, target_type: str) -> Dict:
        """数据校验."""
        validate_province_and_type(province, target_type)
        end = self._find_latest_date(province, target_type) or datetime.now()
        start = end - timedelta(days=365)
        df = self.store.load_raw_data(province, target_type,
                                       start.strftime("%Y-%m-%d"),
                                       (end + timedelta(days=1)).strftime("%Y-%m-%d"),
                                       value_type_filter="实际")
        if df.empty:
            return {"error": "no_data"}

        issues = []
        total = len(df)
        null_rate = df["value"].isna().mean() if "value" in df.columns else 0
        if null_rate > 0:
            issues.append(f"缺失率: {null_rate:.2%}")

        if "dt" in df.columns:
            gaps = df["dt"].diff().dropna()
            large_gaps = gaps[gaps > pd.Timedelta(minutes=30)]
            if len(large_gaps) > 0:
                issues.append(f"时间间隙>30min: {len(large_gaps)}处, 最大{gaps.max()}")

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
        """导出预测结果."""
        end = self._find_latest_date(province, target_type) or datetime.now()
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
