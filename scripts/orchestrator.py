"""orchestrator.py — 总调度器 (type 三段式 + 跨类型 + 并行训练 + 自动进化)

修复:
- build_features 自动获取气象数据
- predict 集成跨类型特征
- train_all 并行化
- 新增 auto_improve 完整闭环
- 全局超时保护
"""
import glob
import hashlib
import logging
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from scripts.core.config import (
    get_provinces, get_base_types, load_config,
    validate_province_and_type, get_available_types,
    get_available_actual_types, parse_type, TypeInfo,
    get_daemon_config,
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
        self.predictor = Predictor(self.trainer, self.store)
        self.validator = Validator()
        self.analyzer = Analyzer()
        self.improver = Improver(self.source, self.trainer, self.backtester)
        self.engineer = FeatureEngineer()
        self._validator_history: List[Dict] = []
        self._start_time = 0

    def setup(self):
        logger.info("初始化系统...")
        self.source.setup()
        logger.info("数据存储已就绪")

    def _check_timeout(self):
        cfg = get_daemon_config()
        timeout = cfg.get("timeout_seconds", 1800)
        elapsed = time.time() - self._start_time
        if elapsed > timeout:
            raise TimeoutError(f"调度超时 ({elapsed:.0f}s > {timeout}s)")

    def _find_latest_date(self, province: str, target_type: str):
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

    def _get_features_dir(self) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            ".energy_data", "features")

    def _hash_feature_code(self) -> str:
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
        vp = self._get_version_path()
        if not os.path.exists(vp):
            return False
        try:
            with open(vp, "r") as f:
                return f.read().strip() == self._hash_feature_code()
        except Exception:
            return False

    def _save_version(self):
        vp = self._get_version_path()
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        with open(vp, "w") as f:
            f.write(self._hash_feature_code())

    def _get_existing_max_date(self, provinces: List[str]) -> Optional[datetime]:
        earliest = None
        features_dir = self._get_features_dir()
        for p in provinces:
            files = sorted(glob.glob(os.path.join(features_dir, f"{p}_*.parquet")))
            if not files:
                return None
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
        ti = parse_type(target_type)
        parts = target_type.split("_")
        if len(parts) >= 3:
            return [target_type]

        available = get_available_actual_types(province)
        if ti.sub:
            prefix = f"{ti.base}_{ti.sub}"
            candidates = [t for t in available if t.startswith(prefix)]
        else:
            candidates = [t for t in available if t.startswith(ti.base)]
        return candidates if candidates else [target_type]

    # ================================================================
    #  特征构建 (自动获取气象)
    # ================================================================

    def build_features(self, province: str = None,
                       start_date: str = None,
                       end_date: str = None,
                       force_rebuild: bool = False):
        self._start_time = time.time()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        provinces = [province] if province else get_provinces()

        version_ok = self._check_version()
        existing_max = self._get_existing_max_date(provinces) if not force_rebuild else None

        if force_rebuild or not version_ok or existing_max is None:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            inc_start = existing_max - timedelta(days=30)
            if start_date is not None:
                user_start = datetime.strptime(start_date, "%Y-%m-%d")
                inc_start = min(inc_start, user_start)
            start_date = inc_start.strftime("%Y-%m-%d")
            logger.info("增量特征构建: %s ~ %s", start_date, end_date)

        # ── 自动批量获取气象数据 (核心修复) ──
        logger.info("批量获取气象数据...")
        from scripts.data.fetcher import DataFetcher
        fetcher = DataFetcher()
        weather_cache: Dict[str, pd.DataFrame] = {}
        try:
            weather_cache = fetcher.preload_historical_weather(
                provinces, start_date, end_date
            )
        except Exception as e:
            logger.warning("批量气象获取失败: %s", e)

        # 第一遍: 按 (province, type) 构建特征
        built: Dict[str, pd.DataFrame] = {}
        for p in provinces:
            available_types = get_available_actual_types(p)
            if not available_types:
                available_types = get_available_types(p)

            for t in available_types:
                logger.info(f"构建特征: {p}/{t} ...")
                self._check_timeout()

                raw = self.store.load_raw_data(p, t, start_date, end_date,
                                               value_type_filter="实际")
                if raw.empty:
                    logger.warning(f"  无数据: {p}/{t}")
                    continue

                features = self.engineer.build_features_from_raw(
                    raw, value_type_filter="实际",
                    weather_df=weather_cache.get(p)
                )

                # 注入预测误差特征
                try:
                    preds = self.store.load_predictions(p, t, start_date, end_date)
                    if not preds.empty:
                        features = self.engineer.add_prediction_error_features(features, preds)
                except Exception:
                    pass

                built[f"{p}/{t}"] = features

        # 第二遍: 交叉特征 v2
        for p in provinces:
            p_built = {t: df for key, df in built.items()
                      if key.startswith(f"{p}/")
                      for t in [key.split("/", 1)[1]]}
            if len(p_built) < 2:
                continue
            logger.info(f"交叉特征 v2: {p}")
            for t, features in p_built.items():
                self._check_timeout()
                built[f"{p}/{t}"] = self.engineer.add_cross_type_features_v2(
                    features, p_built, p,
                )

        # 第二遍半: 电价专属特征
        for key, features in built.items():
            p, t = key.split("/", 1)
            if t.startswith("电价"):
                siblings = {tk: df for k, df in built.items()
                           if k.startswith(f"{p}/") and k != key
                           for tk in [k.split("/", 1)[1]]}
                built[key] = self.engineer.add_price_features(features, siblings)

        # 第三遍: 写入
        for key, features in built.items():
            p, t = key.split("/", 1)
            count = self.store.insert_features(features)
            logger.info(f"  {p}/{t}: 写入 {count} 行")

        self._save_version()

    # ================================================================
    #  训练 (并行化)
    # ================================================================

    def _train_single(self, province: str, target_type: str) -> Dict:
        """训练单个 province/type."""
        try:
            features = self.store.load_features(
                province, target_type,
                (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            )
            if features.empty:
                return {"error": f"无特征数据: {province}/{target_type}"}

            return self.trainer.quantile_train(features, province, target_type)
        except Exception as e:
            logger.error("训练失败 %s/%s: %s", province, target_type, e)
            return {"error": str(e)}

    def train_all(self):
        """并行训练所有 province/type."""
        cfg = get_daemon_config()
        parallel = cfg.get("parallel_provinces", True)

        available = self._scan_available()
        if not available:
            logger.warning("无可用数据")
            return

        tasks = []
        for province in get_provinces():
            for t in available.get(province, []):
                tasks.append((province, t))

        if not tasks:
            return

        if parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
                futures = {
                    executor.submit(self._train_single, p, t): (p, t)
                    for p, t in tasks
                }
                for future in as_completed(futures, timeout=600):
                    p, t = futures[future]
                    try:
                        result = future.result(timeout=300)
                        if "error" in result:
                            logger.warning("训练失败: %s/%s - %s", p, t, result["error"])
                        else:
                            logger.info("训练完成: %s/%s (%d features)", p, t,
                                        result.get("n_features", 0))
                    except Exception as e:
                        logger.error("训练超时/异常: %s/%s - %s", p, t, e)
        else:
            for p, t in tasks:
                self._train_single(p, t)

    def _scan_available(self) -> Dict[str, List[str]]:
        result = {}
        for province in get_provinces():
            types = get_available_actual_types(province)
            if types:
                result[province] = types
        return result

    # ================================================================
    #  预测
    # ================================================================

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24) -> Dict:
        validate_province_and_type(province, target_type)
        resolved_types = self._resolve_types(province, target_type)

        health = None
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            for rt in resolved_types:
                features_df = self.store.load_features(
                    province, rt,
                    start.strftime("%Y-%m-%d"),
                    (end + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                if features_df.empty:
                    continue

                bt = self.backtester.rolling_window_backtest(
                    features_df, train_window_days=14, test_window_hours=24,
                    n_windows=2, province=province,
                    target_type=rt,
                )
                if "summary" in bt:
                    mape = bt["summary"]["overall_mape"]
                    health = {
                        "mape": mape,
                        "status": "ok" if mape < 0.10 else ("warning" if mape < 0.15 else "degraded"),
                    }
                    break
        except Exception:
            pass

        all_samples = []
        all_results = []
        for rt in resolved_types:
            try:
                pred_df = self.predictor.predict(province, rt, horizon_hours)
                samples = [
                    {"dt": r["dt"].strftime("%Y-%m-%d %H:%M") if hasattr(r["dt"], "strftime") else str(r["dt"]),
                     "p10": round(r["p10"], 1), "p50": round(r["p50"], 1),
                     "p90": round(r["p90"], 1)}
                    for _, r in pred_df.iterrows()
                ]
                all_samples.extend(samples[:96])  # 前 24h
                all_results.append({"type": rt, "n": len(pred_df)})
            except Exception as e:
                logger.warning("预测失败 %s/%s: %s", province, rt, e)

        return {
            "province": province,
            "target_type": target_type,
            "resolved_types": resolved_types,
            "health": health,
            "n_predictions": len(all_samples),
            "sample": all_samples,
            "results": all_results,
        }

    # ================================================================
    #  回测 + 验证 + 优化
    # ================================================================

    def run_backtest_cycle(self, province: str, target_type: str) -> Dict:
        features = self.store.load_features(
            province, target_type,
            (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d"),
        )
        if features.empty:
            return {"error": "no_data"}

        return self.backtester.evaluate_model(
            features, train_window_days=30, test_window_hours=24,
            province=province, target_type=target_type,
        )

    def run_validation_cycle(self, province: str, target_type: str) -> Dict:
        features = self.store.load_features(
            province, target_type,
            (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d"),
        )
        if features.empty:
            return {"error": "no_data"}

        # 回测
        report = self.backtester.evaluate_model(
            features, train_window_days=30, test_window_hours=24,
            province=province, target_type=target_type,
        )
        if "error" in report:
            return {"status": "error", "report": report}

        metrics = {
            "overall_mape": report.get("overall_mape", 0),
            "by_season": report.get("by_season", {}),
            "by_time_type": report.get("by_time_type", {}),
            "by_hour_bucket": report.get("by_hour_bucket", {}),
        }

        # 诊断
        diagnoses = self.analyzer.diagnose(metrics)
        if not diagnoses:
            return {"status": "ok", "report": report}

        # 触发优化
        should_improve = self.validator.should_trigger(
            {"mape": metrics["overall_mape"]},
            self._validator_history,
        )

        if should_improve:
            logger.info("退化检测触发: %s/%s, MAPE=%.4f, 诊断: %s",
                        province, target_type, metrics["overall_mape"],
                        [d["root_cause"] for d in diagnoses])
            try:
                improvement = self.improver.improve(
                    diagnoses, features, province, target_type,
                    baseline=metrics,
                )
                self._validator_history.append({"bias_direction": "ok"})

                if improvement.get("improvement", 0) > 0:
                    # 重训练
                    self.trainer.quantile_train(features, province, target_type)
                    return {"status": "improved", "improvement": improvement, "report": report}
            except Exception as e:
                logger.warning("优化失败: %s", e)

        self._validator_history.append(
            {"bias_direction": metrics.get("bias_direction", "ok")}
        )
        return {"status": "degraded", "report": report, "diagnoses": diagnoses}

    def auto_improve(self, province: str, target_type: str) -> Dict:
        """完整自动优化闭环: 回测 → 诊断 → 策略实验 → 重训练."""
        features = self.store.load_features(
            province, target_type,
            (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d"),
        )
        if features.empty:
            return {"error": "no_data"}

        # 1. 基线回测
        baseline = self.backtester.evaluate_model(
            features, train_window_days=30, test_window_hours=24,
            province=province, target_type=target_type,
        )
        if "error" in baseline:
            return baseline

        baseline_metrics = {
            "overall_mape": baseline.get("overall_mape", 0),
            "by_season": baseline.get("by_season", {}),
            "by_time_type": baseline.get("by_time_type", {}),
            "by_hour_bucket": baseline.get("by_hour_bucket", {}),
        }
        logger.info("基线 MAPE=%.4f: %s/%s", baseline_metrics["overall_mape"],
                    province, target_type)

        # 2. 诊断
        diagnoses = self.analyzer.diagnose(baseline_metrics)
        if not diagnoses:
            logger.info("无显著问题，跳过优化")
            return {"status": "ok", "mape": baseline_metrics["overall_mape"]}

        logger.info("诊断: %s", [d["root_cause"] for d in diagnoses])

        # 3. 策略实验
        improvement = self.improver.improve(
            diagnoses, features, province, target_type,
            baseline=baseline_metrics,
        )

        # 4. 重训练
        if improvement.get("improvement", 0) > 0:
            new_result = self.trainer.quantile_train(features, province, target_type)
            logger.info("重训练完成: %s/%s", province, target_type)

            # 验证新模型
            new_backtest = self.backtester.evaluate_model(
                features, train_window_days=30, test_window_hours=24,
                province=province, target_type=target_type,
            )
            new_mape = new_backtest.get("overall_mape", baseline_metrics["overall_mape"])

            return {
                "status": "improved",
                "mape_before": baseline_metrics["overall_mape"],
                "mape_after": new_mape,
                "improvement": improvement,
            }
        else:
            return {
                "status": "no_improvement",
                "mape": baseline_metrics["overall_mape"],
                "improvement": improvement,
            }

    # ================================================================
    #  其他命令
    # ================================================================

    def validate_data(self, province: str, target_type: str):
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
                issues.append(f"时间间隙>30min: {len(large_gaps)}处")

        if "value" in df.columns:
            neg_rate = (df["value"] <= 0).mean()
            if neg_rate > 0:
                issues.append(f"零值/负值比例: {neg_rate:.2%}")

        return {
            "province": province, "type": target_type,
            "total_rows": total,
            "date_range": f"{df['dt'].min()} ~ {df['dt'].max()}",
            "null_rate": round(null_rate, 4),
            "issues": issues,
            "status": "ok" if not issues else "has_issues",
        }

    def explain(self, province: str, target_type: str) -> Dict:
        importance = self.trainer.feature_importance(province, target_type)
        versions = self.trainer.list_versions(province, target_type)
        return {
            "province": province, "type": target_type,
            "feature_importance": importance[:15],
            "model_versions": versions,
        }

    def export(self, province: str, target_type: str,
               fmt: str = "json", output: str = None) -> str:
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
        return self.trainer.rollback(province, target_type)

    def chart(self, province: str, target_type: str,
              hours: int = 24) -> str:
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
