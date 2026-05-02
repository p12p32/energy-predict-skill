"""data_source.py — 数据源抽象层

去掉"必须有数据库"的限制.
支持三种模式:
  - FileSource: 本地文件系统 (零依赖, 默认)
  - DorisSource: Apache Doris (生产环境)
  - MemorySource: 直接传 DataFrame (临时测试)
"""
import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from scripts.core.config import parse_type


def _filter_by_value_type(df: pd.DataFrame, value_type: Optional[str] = None) -> pd.DataFrame:
    """按 value_type 过滤 DataFrame. 解析 type 列的三段式结构."""
    if value_type is None or df.empty or "type" not in df.columns:
        return df
    mask = df["type"].apply(
        lambda t: parse_type(str(t)).value_type == value_type
    )
    return df[mask].reset_index(drop=True)


class DataSource(ABC):
    """数据源抽象接口."""

    @abstractmethod
    def load_raw(self, province: str, data_type: str,
                 start_date: str, end_date: str,
                 table: str = "energy_raw",
                 value_type_filter: Optional[str] = None) -> pd.DataFrame:
        ...

    @abstractmethod
    def save_features(self, df: pd.DataFrame) -> int:
        ...

    @abstractmethod
    def load_features(self, province: str, data_type: str,
                      start_date: str, end_date: str,
                      value_type_filter: Optional[str] = None) -> pd.DataFrame:
        ...

    @abstractmethod
    def save_predictions(self, df: pd.DataFrame) -> int:
        ...

    @abstractmethod
    def load_predictions(self, province: str, data_type: str,
                         start_date: str, end_date: str,
                         value_type_filter: Optional[str] = None) -> pd.DataFrame:
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
                 table: str = "energy_raw",
                 value_type_filter: Optional[str] = None) -> pd.DataFrame:
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
        df = df[mask].sort_values("dt").reset_index(drop=True)

        return _filter_by_value_type(df, value_type_filter)

    def import_csv(self, filepath: str) -> Dict:
        """导入 CSV 到 raw 目录。返回列信息供 AI 自行判断维度覆盖."""
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

        # 哑报告：只管列名，不做任何维度假定
        auto_cols = {"dt", "province", "type", "value"}
        extra_cols = [c for c in df.columns if c not in auto_cols]

        return {
            "status": "ok",
            "rows": len(df),
            "columns": list(df.columns),
            "extra_columns": extra_cols,
        }

    # ── Features ──

    def save_features(self, df: pd.DataFrame, replace: bool = False) -> int:
        if df.empty:
            return 0
        date_str = datetime.now().strftime("%Y%m%d")
        for (province, dtype), group in df.groupby(["province", "type"]):
            path = os.path.join(
                self.base_dir, "features",
                f"{province}_{dtype}_{date_str}.parquet"
            )
            if not replace and os.path.exists(path):
                existing = pd.read_parquet(path)
                # 新数据优先——新列覆盖旧 NaN
                group = pd.concat([group, existing]).drop_duplicates(subset=["dt"], keep="first").sort_values("dt")
            group.to_parquet(path, index=False)
        return len(df)

    def clear_features(self, provinces: list = None):
        import glob
        if provinces is None:
            pattern = os.path.join(self.base_dir, "features", "*.parquet")
            files = glob.glob(pattern)
        else:
            files = []
            for p in provinces:
                files.extend(glob.glob(os.path.join(
                    self.base_dir, "features", f"{p}_*.parquet"
                )))
        for f in files:
            os.remove(f)

    def load_features(self, province: str, data_type: str,
                      start_date: str = None, end_date: str = None,
                      value_type_filter: Optional[str] = None) -> pd.DataFrame:
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
            if start_date is not None or end_date is not None:
                start_dt = pd.to_datetime(start_date or "2000-01-01")
                end_dt = pd.to_datetime(end_date or "2100-01-01")
                df = df[(df["dt"] >= start_dt) & (df["dt"] < end_dt)]
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["dt"]).sort_values("dt").reset_index(drop=True)
        return _filter_by_value_type(result, value_type_filter)

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
                         start_date: str, end_date: str,
                         value_type_filter: Optional[str] = None) -> pd.DataFrame:
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
        result = result[mask].sort_values("dt").reset_index(drop=True)
        return _filter_by_value_type(result, value_type_filter)

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

    def clear_features(self, provinces: list = None):
        pass

    def set_raw(self, df: pd.DataFrame, province: str, data_type: str):
        self._raw[f"{province}_{data_type}"] = df.copy()

    def load_raw(self, province: str, data_type: str,
                 start_date: str, end_date: str,
                 table: str = "energy_raw",
                 value_type_filter: Optional[str] = None) -> pd.DataFrame:
        key = f"{province}_{data_type}"
        df = self._raw.get(key, pd.DataFrame())
        if df.empty:
            return df
        df["dt"] = pd.to_datetime(df["dt"])
        mask = (df["dt"] >= pd.to_datetime(start_date)) & (df["dt"] < pd.to_datetime(end_date))
        df = df[mask].sort_values("dt").reset_index(drop=True)
        return _filter_by_value_type(df, value_type_filter)

    def save_features(self, df: pd.DataFrame) -> int:
        for (province, dtype), group in df.groupby(["province", "type"]):
            key = f"{province}_{dtype}"
            existing = self._features.get(key, pd.DataFrame())
            self._features[key] = pd.concat([existing, group]).drop_duplicates(subset=["dt"])
        return len(df)

    def load_features(self, province: str, data_type: str,
                      start_date: str = None, end_date: str = None,
                      value_type_filter: Optional[str] = None) -> pd.DataFrame:
        key = f"{province}_{data_type}"
        df = self._features.get(key, pd.DataFrame())
        if df.empty:
            return df
        df["dt"] = pd.to_datetime(df["dt"])
        if start_date is not None or end_date is not None:
            start_dt = pd.to_datetime(start_date or "2000-01-01")
            end_dt = pd.to_datetime(end_date or "2100-01-01")
            df = df[(df["dt"] >= start_dt) & (df["dt"] < end_dt)]
        df = df.sort_values("dt").reset_index(drop=True)
        return _filter_by_value_type(df, value_type_filter)

    def save_predictions(self, df: pd.DataFrame) -> int:
        date_str = datetime.now().strftime("%Y%m%d")
        for (province, dtype), group in df.groupby(["province", "type"]):
            self._predictions[f"{province}_{dtype}_{date_str}"] = group.copy()
        return len(df)

    def load_predictions(self, province: str, data_type: str,
                         start_date: str, end_date: str,
                         value_type_filter: Optional[str] = None) -> pd.DataFrame:
        dfs = []
        for key, df in self._predictions.items():
            if key.startswith(f"{province}_{data_type}"):
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, ignore_index=True)
        return _filter_by_value_type(result, value_type_filter)

    def save_knowledge(self, data: Dict) -> None:
        self._knowledge.append(data)

    def load_knowledge(self) -> pd.DataFrame:
        return pd.DataFrame(self._knowledge)
