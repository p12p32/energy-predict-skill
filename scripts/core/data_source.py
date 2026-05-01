"""data_source.py — 数据源抽象层

去掉"必须有数据库"的限制.
支持三种模式:
  - FileSource: 本地文件系统 (零依赖, 默认)
  - DorisSource: Apache Doris (生产环境)
  - MemorySource: 直接传 DataFrame (临时测试)
"""
import os
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


class DataSource(ABC):
    """数据源抽象接口."""

    @abstractmethod
    def load_raw(self, province: str, data_type: str,
                 start_date: str, end_date: str,
                 table: str = "energy_raw") -> pd.DataFrame:
        ...

    @abstractmethod
    def save_features(self, df: pd.DataFrame) -> int:
        ...

    @abstractmethod
    def load_features(self, province: str, data_type: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def save_predictions(self, df: pd.DataFrame) -> int:
        ...

    @abstractmethod
    def load_predictions(self, province: str, data_type: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def save_knowledge(self, data: Dict) -> None:
        ...

    @abstractmethod
    def load_knowledge(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def setup(self) -> None:
        """初始化数据存储结构."""
        ...


# ============================================================
# FileSource — 本地文件系统
# ============================================================

class FileSource(DataSource):
    """数据存在本地文件系统 .energy_data/ 目录下.

    文件布局:
      .energy_data/
        raw/              用户导入的原始数据
          {province}_{type}.csv
        features/         计算后的特征
          {province}_{type}_{YYYYMMDD}.parquet
        predictions/      预测结果
          {province}_{type}_{YYYYMMDD}_pred.parquet
        knowledge.json    策略知识库
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".energy_data")
        self.base_dir = base_dir
        self._ensure_dirs()

    def _ensure_dirs(self):
        for sub in ["raw", "features", "predictions"]:
            os.makedirs(os.path.join(self.base_dir, sub), exist_ok=True)

    def setup(self):
        self._ensure_dirs()

    # ── Raw Data ──

    def load_raw(self, province: str, data_type: str,
                 start_date: str, end_date: str,
                 table: str = "energy_raw") -> pd.DataFrame:
        path = os.path.join(self.base_dir, "raw", f"{province}_{data_type}.csv")
        if not os.path.exists(path):
            path = os.path.join(self.base_dir, "raw", f"{province}.csv")

        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])

        if "province" not in df.columns:
            df["province"] = province
        if "type" not in df.columns:
            df["type"] = data_type

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (df["dt"] >= start_dt) & (df["dt"] < end_dt)
        return df[mask].sort_values("dt").reset_index(drop=True)

    def import_csv(self, filepath: str) -> Dict:
        """导入 CSV 文件到 raw 目录，完成后诊断维度覆盖."""
        df = pd.read_csv(filepath)

        required = ["dt"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少必要列: {missing}")

        if "province" not in df.columns:
            raise ValueError("CSV 缺少 province 列")

        df["dt"] = pd.to_datetime(df["dt"])

        for (province, dtype), group in df.groupby(["province", "type"]) if "type" in df.columns else df.groupby("province"):
            if "type" not in df.columns:
                output_path = os.path.join(self.base_dir, "raw", f"{province}.csv")
            else:
                output_path = os.path.join(self.base_dir, "raw", f"{province}_{dtype}.csv")
            group.to_csv(output_path, index=False)

        return {
            "status": "ok",
            "rows": len(df),
            "file": filepath,
            "diagnosis": self._diagnose_columns(df),
        }

    def _diagnose_columns(self, df: pd.DataFrame) -> Dict:
        """诊断 CSV 列的维度覆盖，返回缺失维度及获取建议."""
        cols = set(df.columns)

        weather_cols = ["temperature", "humidity", "wind_speed",
                        "solar_radiation", "precipitation", "pressure"]
        economic_cols = ["coal_price", "carbon_price", "price"]

        present_weather = [c for c in weather_cols if c in cols]
        missing_weather = [c for c in weather_cols if c not in cols]
        present_economic = [c for c in economic_cols if c in cols]
        missing_economic = [c for c in economic_cols if c not in cols]

        # 自动维度: 系统从 dt 列自动计算，无需外部数据
        auto_dimensions = [
            "lag (滞后特征 — 从 value 列自动推导)",
            "calendar (小时/星期/季节/节假日 — 从 dt 列自动推导)",
            "cyclical (sin/cos 周期编码 — 自动计算)",
        ]

        suggestions = []
        if missing_weather:
            suggestions.append({
                "dimension": "气象",
                "missing": missing_weather,
                "present": present_weather if present_weather else ["(无)"],
                "how_to_get": (
                    "可用 tools/python 执行: "
                    "python3 -c \"from scripts.data.fetcher import DataFetcher; "
                    "df = DataFetcher().fetch_weather_for_all_provinces('YYYY-MM-DD', 'YYYY-MM-DD'); "
                    "df.to_csv('data/weather.csv', index=False)\"；"
                    "或使用自己的气象 API / 数据源，写入 CSV 后重新 /import"
                ),
            })
        if missing_economic:
            suggestions.append({
                "dimension": "经济",
                "missing": missing_economic,
                "present": present_economic if present_economic else ["(无)"],
                "how_to_get": (
                    "煤价 → 搜索 '秦皇岛动力煤价格' + 爬取/手动录入；"
                    "碳价 → 搜索 '全国碳排放权交易市场价格'；"
                    "数据写入 CSV 后重新 /import"
                ),
            })

        all_covered = not missing_weather and not missing_economic

        return {
            "auto_dimensions": auto_dimensions,
            "missing_dimensions": suggestions,
            "all_covered": all_covered,
            "summary": "数据维度完整，可直接训练" if all_covered
                       else f"缺失 {len(missing_weather) + len(missing_economic)} 个外部数据列，建议 AI 补全后重新导入",
        }

    # ── Features ──

    def save_features(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        date_str = datetime.now().strftime("%Y%m%d")
        for (province, dtype), group in df.groupby(["province", "type"]):
            path = os.path.join(
                self.base_dir, "features",
                f"{province}_{dtype}_{date_str}.parquet"
            )
            if os.path.exists(path):
                existing = pd.read_parquet(path)
                group = pd.concat([existing, group]).drop_duplicates(subset=["dt"]).sort_values("dt")
            group.to_parquet(path, index=False)
        return len(df)

    def load_features(self, province: str, data_type: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        import glob
        pattern = os.path.join(self.base_dir, "features", f"{province}_{data_type}_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            return pd.DataFrame()

        frames = []
        for f in files:
            df = pd.read_parquet(f)
            if "dt" in df.columns:
                df["dt"] = pd.to_datetime(df["dt"])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            mask = (df["dt"] >= start_dt) & (df["dt"] < end_dt)
            frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return result.drop_duplicates(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    # ── Predictions ──

    def save_predictions(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        for (province, dtype), group in df.groupby(["province", "type"]):
            path = os.path.join(
                self.base_dir, "predictions",
                f"{province}_{dtype}_{date_str}_pred.parquet"
            )
            group.to_parquet(path, index=False)
        return len(df)

    def load_predictions(self, province: str, data_type: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        import glob
        pattern = os.path.join(self.base_dir, "predictions", f"{province}_{data_type}_*_pred.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(f) for f in files]
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        if "dt" in result.columns:
            result["dt"] = pd.to_datetime(result["dt"])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (result["dt"] >= start_dt) & (result["dt"] < end_dt)
        return result[mask].sort_values("dt").reset_index(drop=True)

    # ── Knowledge ──

    def save_knowledge(self, data: Dict) -> None:
        path = os.path.join(self.base_dir, "knowledge.json")
        existing = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                existing = json.load(f)
        key = data.get("strategy_hash", data.get("name", ""))
        existing[key] = {**existing.get(key, {}), **data, "updated": datetime.now().isoformat()}
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def load_knowledge(self) -> pd.DataFrame:
        path = os.path.join(self.base_dir, "knowledge.json")
        if not os.path.exists(path):
            return pd.DataFrame()
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(list(data.values()))


# ============================================================
# MemorySource — 直接传 DataFrame
# ============================================================

class MemorySource(DataSource):
    """内存数据源，适合临时测试或通过代码直接传入 DataFrame."""

    def __init__(self):
        self._raw: Dict[str, pd.DataFrame] = {}
        self._features: Dict[str, pd.DataFrame] = {}
        self._predictions: Dict[str, pd.DataFrame] = {}
        self._knowledge: List[Dict] = []

    def setup(self):
        pass

    def set_raw(self, df: pd.DataFrame, province: str, data_type: str):
        self._raw[f"{province}_{data_type}"] = df.copy()

    def load_raw(self, province: str, data_type: str,
                 start_date: str, end_date: str,
                 table: str = "energy_raw") -> pd.DataFrame:
        key = f"{province}_{data_type}"
        df = self._raw.get(key, pd.DataFrame())
        if df.empty:
            return df
        df["dt"] = pd.to_datetime(df["dt"])
        mask = (df["dt"] >= pd.to_datetime(start_date)) & (df["dt"] < pd.to_datetime(end_date))
        return df[mask].sort_values("dt").reset_index(drop=True)

    def save_features(self, df: pd.DataFrame) -> int:
        for (province, dtype), group in df.groupby(["province", "type"]):
            key = f"{province}_{dtype}"
            existing = self._features.get(key, pd.DataFrame())
            self._features[key] = pd.concat([existing, group]).drop_duplicates(subset=["dt"])
        return len(df)

    def load_features(self, province: str, data_type: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        key = f"{province}_{data_type}"
        df = self._features.get(key, pd.DataFrame())
        if df.empty:
            return df
        df["dt"] = pd.to_datetime(df["dt"])
        mask = (df["dt"] >= pd.to_datetime(start_date)) & (df["dt"] < pd.to_datetime(end_date))
        return df[mask].sort_values("dt").reset_index(drop=True)

    def save_predictions(self, df: pd.DataFrame) -> int:
        date_str = datetime.now().strftime("%Y%m%d")
        for (province, dtype), group in df.groupby(["province", "type"]):
            self._predictions[f"{province}_{dtype}_{date_str}"] = group.copy()
        return len(df)

    def load_predictions(self, province: str, data_type: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        dfs = []
        for key, df in self._predictions.items():
            if key.startswith(f"{province}_{data_type}"):
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def save_knowledge(self, data: Dict) -> None:
        self._knowledge.append(data)

    def load_knowledge(self) -> pd.DataFrame:
        return pd.DataFrame(self._knowledge)
