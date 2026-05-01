"""data_watcher.py — 数据接管器

持续从外部拉取新数据 → 构建特征 → 触发预测流水线。
让"自我进化"从概念变成现实: 新数据不断流入, 真实值验证预测, 退化自动改进。

支持三种数据源:
  - file:   监控 CSV/Parquet 目录, 新文件自动导入
  - doris:  轮询 Doris 表, 检测新记录
  - api:    定时拉取 HTTP API 数据
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

import pandas as pd

from scripts.core.data_source import FileSource, DataSource
from scripts.core.config import load_config, get_provinces, get_types
from scripts.data.features import FeatureStore, FeatureEngineer
from scripts.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class DataWatcher:
    """持续监控数据源, 有新数据→构建特征→触发回调"""

    def __init__(self, source: DataSource = None):
        self.source = source or FileSource()
        self.store = FeatureStore(self.source)
        self.engineer = FeatureEngineer()
        self.fetcher = DataFetcher()

        # 状态追踪: 记住上次处理到哪了
        self._state_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".energy_data", "watcher_state.json"
        )
        self._state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self._state_file):
            with open(self._state_file) as f:
                return json.load(f)
        return {"last_processed": {}, "last_doris_max_dt": {}}

    def _save_state(self):
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        self._state["updated"] = datetime.now().isoformat()
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    # ================================================================
    # File: 监控 CSV 目录
    # ================================================================

    def watch_file(self, directory: str, callback: Callable[[str, str, str], None] = None) -> int:
        """扫描目录中的新 CSV/Parquet 文件, 导入并构建特征.

        Args:
            directory: CSV 文件目录
            callback: 每处理完一组 (province, type) 后调用 callback(province, type, msg)

        Returns: 导入的批次数量
        """
        if not os.path.isdir(directory):
            logger.warning(f"目录不存在: {directory}")
            return 0

        imported = 0
        processed_key = f"file:{directory}"
        processed_files = self._state.get(processed_key, [])

        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(('.csv', '.parquet')):
                continue
            fpath = os.path.join(directory, fname)
            if fname in processed_files:
                continue

            try:
                if fname.endswith('.csv'):
                    result = self.source.import_csv(fpath)
                else:
                    df = pd.read_parquet(fpath)
                    if "province" in df.columns and "type" in df.columns:
                        for (province, dtype), group in df.groupby(["province", "type"]):
                            self._ingest(group, province, dtype, callback)
                    result = {"rows": len(df)}
            except Exception as e:
                logger.error(f"导入失败 {fname}: {e}")
                continue

            imported += 1
            processed_files.append(fname)
            logger.info(f"导入: {fname} ({result.get('rows', '?')} 行)")

        self._state[processed_key] = processed_files[-100:]  # 只保留最近 100 个文件名
        self._save_state()
        return imported

    # ================================================================
    # Doris: 轮询新记录
    # ================================================================

    def watch_doris(self, callback: Callable[[str, str, str], None] = None) -> int:
        """轮询 Doris, 拉取上次检查后新增的记录, 构建特征.

        Returns: 处理的数据批次数量
        """
        try:
            from scripts.core.db import DorisDB
            db = DorisDB()
        except Exception as e:
            logger.warning(f"Doris 连接失败: {e}")
            return 0

        cfg = load_config()
        src_table = cfg["data"].get("source_table", "energy_raw")
        processed = 0

        for province in get_provinces():
            for dtype in get_types():
                key = f"doris:{province}:{dtype}"
                last_max_dt = self._state.get("last_doris_max_dt", {}).get(key)

                sql = (
                    f"SELECT dt, province, type, value, price "
                    f"FROM {src_table} "
                    f"WHERE province='{province}' AND type='{dtype}'"
                )
                if last_max_dt:
                    sql += f" AND dt > '{last_max_dt}'"
                sql += " ORDER BY dt"

                try:
                    raw = db.query(sql)
                    if raw.empty:
                        continue

                    self._ingest(raw, province, dtype, callback)

                    new_max = raw["dt"].max()
                    self._state.setdefault("last_doris_max_dt", {})[key] = str(new_max)
                    processed += 1
                except Exception as e:
                    logger.error(f"Doris 查询失败 {province}/{dtype}: {e}")

        self._save_state()
        return processed

    # ================================================================
    # API: 定时拉取
    # ================================================================

    def watch_api(self, url: str, interval_seconds: int = 900,
                  callback: Callable[[str, str, str], None] = None):
        """持续轮询 HTTP API, 拉取新数据."""
        import requests

        last_fetch = self._state.get("last_api_fetch")

        while True:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict) and "data" in data:
                        df = pd.DataFrame(data["data"])
                    else:
                        df = pd.DataFrame([data])

                    if not df.empty and "province" in df.columns and "type" in df.columns:
                        for (province, dtype), group in df.groupby(["province", "type"]):
                            self._ingest(group, province, dtype, callback)

                    last_fetch = datetime.now().isoformat()
                    self._state["last_api_fetch"] = last_fetch
                    self._save_state()
                    logger.info(f"API 拉取: {len(df)} 行")

            except Exception as e:
                logger.error(f"API 拉取失败: {e}")

            time.sleep(interval_seconds)

    # ================================================================
    # 内部: 数据采集 → 构建特征
    # ================================================================

    def _ingest(self, raw: pd.DataFrame, province: str, dtype: str,
                callback: Callable = None):
        """原始数据 → 特征构建 → 存储 → 回调."""
        if "dt" not in raw.columns:
            raw["dt"] = pd.to_datetime(raw.get("dt", datetime.now()))

        # 尝试拉取气象数据补充
        try:
            start = raw["dt"].min().strftime("%Y-%m-%d")
            end = (raw["dt"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
            weather = self.fetcher.fetch_weather(province, start, end, mode="historical")
        except Exception:
            weather = pd.DataFrame()

        features = self.engineer.build_features_from_raw(raw)
        if not weather.empty:
            features = self.engineer.merge_weather(features, weather)

        count = self.store.insert_features(features)
        msg = f"{province}/{dtype}: 新数据 {len(raw)} 行 → {count} 特征"
        logger.info(msg)

        if callback:
            callback(province, dtype, msg)
