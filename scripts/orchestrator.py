"""orchestrator.py — 总调度器 (type 三段式 + value_type 感知)"""
import glob
import hashlib
import logging
import os

import scripts.core.cleanup  # noqa — signal/atexit 注册，防止僵尸进程
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from scripts.core.config import (
    get_provinces, get_base_types, load_config,
    validate_province_and_type, get_available_types,
    get_available_actual_types, parse_type, TypeInfo,
    get_data_delay, get_available_date,
)
from scripts.core.data_source import FileSource, MemorySource
from scripts.data.features import FeatureStore, FeatureEngineer

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
        self.model_dir = "models"
        self.engineer = FeatureEngineer()
        pred_cfg = cfg.get("predictor", {})
        self._lookback_days = pred_cfg.get("lookback_days", 60)
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
        """构建特征。每种 (province, type) 按 data_availability 延迟独立确定可用日期."""
        run_date = datetime.now()
        if end_date is None:
            end_date = run_date.strftime("%Y-%m-%d")

        provinces = [province] if province else get_provinces()

        # ── 决定全量重建 vs 增量追加 ──
        version_ok = self._check_version()
        existing_max = self._get_existing_max_date(provinces) if not force_rebuild else None

        if force_rebuild or not version_ok or existing_max is None:
            if force_rebuild:
                logger.info("强制全量重建 (--force-rebuild)")
            elif not version_ok:
                logger.info("特征代码已变更，触发全量重建")
            else:
                logger.info("无已有特征，首次全量构建")
            self.store.source.clear_features(provinces)
            if start_date is None:
                start_date = (run_date - timedelta(days=365)).strftime("%Y-%m-%d")
            force_rebuild = True
        else:
            # 增量追加: 从已有最新日期 - 30 天开始
            inc_start = existing_max - timedelta(days=30)
            if start_date is not None:
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

        # 第一遍: 按 (province, type) 构建独立特征
        # 每种类型独立计算 data_availability 截断日期
        built: Dict[str, pd.DataFrame] = {}
        type_avail_dates: Dict[str, datetime] = {}  # key="p/t" → 该类型数据可用截止日期

        for p in provinces:
            available_types = get_available_actual_types(p)
            if not available_types:
                available_types = get_available_types(p)

            for t in available_types:
                # 该类型实际可用日期 (考虑延迟)
                type_avail = get_available_date(p, t, run_date)
                type_avail_dates[f"{p}/{t}"] = type_avail

                # 加载原始数据时截止到可用日期
                load_end = (type_avail + timedelta(days=1)).strftime("%Y-%m-%d")
                delay = get_data_delay(p, t)
                logger.info(f"构建特征: {p}/{t} (可用截止={type_avail.strftime('%Y-%m-%d')}, "
                           f"延迟={delay:+d}d) ...")

                if force_rebuild:
                    load_start = start_date
                else:
                    # 增量: 从该类型已有特征的最新日期 - 30 天开始
                    type_existing = self._find_latest_date(p, t)
                    if type_existing:
                        load_start = (type_existing - timedelta(days=30)).strftime("%Y-%m-%d")
                    else:
                        load_start = start_date

                raw = self.store.load_raw_data(p, t, load_start, load_end,
                                               value_type_filter="实际")
                if raw.empty:
                    logger.warning(f"  无数据: {p}/{t} (range={load_start}~{load_end})")
                    continue

                features = self.engineer.build_features_from_raw(raw, value_type_filter="实际")

                # 合并气象数据
                if p in weather_cache:
                    coords = load_config().get("province_coords", {})
                    lat = coords.get(p, {}).get("lat") if coords else None
                    features = self.engineer.merge_weather(features, weather_cache[p], lat=lat)

                # 注入预测误差特征 (如果有历史预测)
                try:
                    preds = self.store.load_predictions(p, t, load_start, load_end)
                    if not preds.empty:
                        features = self.engineer.add_prediction_error_features(features, preds)
                except Exception as e:
                    logger.debug("预测误差特征注入跳过 (%s/%s): %s", p, t, e)

                built[f"{p}/{t}"] = features

        # 第二遍: 交叉特征 v2 (只在同省份、同时段可用数据之间做交叉)
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
            delay = get_data_delay(p, t) if p in type_avail_dates.get("_") else "?"
            logger.info(f"  {p}/{t}: 写入 {count} 行")

        # 保存特征代码版本 hash
        self._save_version()
        logger.info("特征版本已记录: %s", self._hash_feature_code()[:12])

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

    def train_all_layered(self) -> Dict:
        """分层架构全量训练: State→Level→Delta→TS→Fusion 五层串联."""
        from scripts.ml.layers.training import LayeredTrainer

        available = self._scan_available()
        if not available:
            logger.warning("无可用数据，跳过分层训练")
            return {"status": "no_data"}

        layered_trainer = LayeredTrainer(model_dir=self.model_dir)
        results = {}

        for province, target_type in available:
            logger.info(f"[分层训练] {province}/{target_type}")
            df = self.store.load_features(
                province, target_type, None, None,
                value_type_filter="实际",
            )
            result = layered_trainer.train(df, province, target_type)
            results[f"{province}/{target_type}"] = {
                "transform": result.get("transform"),
                "n_s1": result.get("n_s1"),
                "n_s2": result.get("n_s2"),
                "n_s3": result.get("n_s3"),
                "validation": result.get("validation"),
                "error": result.get("error"),
            }
            if "error" in result:
                logger.warning(f"  分层训练失败: {result['error']}")
            else:
                val = result.get("validation", {})
                mape = val.get('mape', 'N/A')
                nrmse = val.get('nrmse', 'N/A')
                nrmse_str = f"{nrmse:.3f}" if isinstance(nrmse, float) else str(nrmse)
                logger.info(f"  S3 MAPE={mape}%, NRMSE={nrmse_str}, n={val.get('n', 0)}")

        return {"status": "ok", "results": results}

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24,
                reference_date: str = None) -> pd.DataFrame:
        """统一预测入口, 按 config pipeline_mode 分发."""
        cfg = load_config()
        mode = cfg.get("hybrid", {}).get("pipeline_mode", "layered")

        if mode == "hybrid":
            return self.predict_hybrid(province, target_type, horizon_hours, reference_date)
        elif mode == "both":
            h_result = self.predict_hybrid(province, target_type, horizon_hours, reference_date)
            l_result = self.predict_layered(province, target_type, horizon_hours, reference_date)
            self._log_comparison(province, target_type, h_result, l_result)
            return h_result
        else:
            return self.predict_layered(province, target_type, horizon_hours, reference_date)

    def predict_hybrid(self, province: str, target_type: str,
                       horizon_hours: int = 24,
                       reference_date: str = None) -> pd.DataFrame:
        """混合架构预测: 物理链 + ML残差."""
        from scripts.ml.pipeline_hybrid import HybridPipeline
        pipeline = HybridPipeline(model_dir=self.model_dir, store=self.store)
        return pipeline.predict(province, target_type,
                                horizon_hours=horizon_hours,
                                reference_date=reference_date)

    def predict_layered(self, province: str, target_type: str,
                       horizon_hours: int = 24,
                       reference_date: str = None) -> pd.DataFrame:
        """分层架构预测: 11步串联 State→Level→Delta→TS→Fusion→Constraints."""
        from scripts.ml.pipeline import LayeredPipeline

        pipeline = LayeredPipeline(model_dir=self.model_dir,
                                   store=self.store)
        return pipeline.predict(province, target_type,
                               horizon_hours=horizon_hours,
                               reference_date=reference_date)

    def train_all_hybrid(self) -> Dict:
        """混合架构训练: 依赖顺序的分层训练."""
        from scripts.ml.trainer_hybrid import HybridTrainer
        trainer = HybridTrainer(model_dir=self.model_dir)
        return trainer.train_all()

    def _log_comparison(self, province, target_type, hybrid_result, layered_result):
        """双跑对比日志."""
        try:
            h_p50 = hybrid_result["p50"].values if "p50" in hybrid_result.columns else []
            l_p50 = layered_result["p50"].values if "p50" in layered_result.columns else []
            if len(h_p50) > 0 and len(l_p50) > 0:
                diff = np.mean(np.abs(h_p50 - l_p50))
                logger.info("[%s/%s] hybrid vs layered mean diff: %.1f", province, target_type, diff)
        except Exception:
            pass

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
