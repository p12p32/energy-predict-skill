"""base.py — 物理模型抽象基类.

每个物理模型编码领域知识为显式可检查方程,
参数具有物理意义, 可通过 get_equation() / get_params() 直接检查.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional
import json, logging, os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PhysicalModelConfig:
    model_type: str = "parametric"
    points_per_hour: int = 4
    capacity_mw: Optional[float] = None
    province: str = ""
    lat: float = 30.0
    lon: float = 104.0


class PhysicalModel(ABC):
    """物理/结构化模型基类.

    Unlike ML layers, these models encode domain physics as explicit,
    inspectable equations. Fitted parameters have physical meaning.
    predict_physics() produces a pure physics prediction with no ML.
    predict() may combine physics + optional ML correction.
    """

    def __init__(self, name: str, config: PhysicalModelConfig):
        self.name = name
        self.config = config
        self.params: Dict[str, float] = {}
        self._ml_fallback = None
        self._trained = False
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def fit(self, weather: pd.DataFrame, actuals: pd.Series, **kwargs) -> Dict:
        """拟合物理参数. 返回 metrics dict."""

    @abstractmethod
    def predict_physics(self, weather: pd.DataFrame) -> np.ndarray:
        """纯物理预测 — 不使用 ML."""

    def predict(self, weather: pd.DataFrame) -> np.ndarray:
        """默认: 物理预测. 子类可覆盖加入 ML 修正."""
        return self.predict_physics(weather)

    @abstractmethod
    def get_equation(self) -> str:
        """人类可读方程字符串."""

    def get_params(self) -> dict:
        return dict(self.params)

    def save(self, path: str) -> None:
        base = os.path.splitext(path)[0]
        extra = self._get_extra_save() if hasattr(self, "_get_extra_save") else {}
        meta = {
            "name": self.name,
            "config": {
                "model_type": self.config.model_type,
                "points_per_hour": self.config.points_per_hour,
                "capacity_mw": self.config.capacity_mw,
                "province": self.config.province,
                "lat": self.config.lat,
                "lon": self.config.lon,
            },
            "params": self.params,
            "metrics": self.metrics,
            "trained": self._trained,
            **extra,
        }
        json_path = f"{base}_params.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if self._ml_fallback is not None:
            import pickle
            with open(f"{base}_fallback.pkl", "wb") as f:
                pickle.dump(self._ml_fallback, f)

        logger.info("%s 参数已保存: %s", self.name, json_path)

    def load(self, path: str) -> None:
        base = os.path.splitext(path)[0]
        json_path = f"{base}_params.json"
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.name = meta["name"]
        cfg = meta["config"]
        self.config = PhysicalModelConfig(**cfg)
        self.params = meta.get("params", {})
        self.metrics = meta.get("metrics", {})
        self._trained = meta.get("trained", False)

        if hasattr(self, "_set_extra_load"):
            self._set_extra_load(meta)

        fallback_path = f"{base}_fallback.pkl"
        if os.path.exists(fallback_path):
            import pickle
            with open(fallback_path, "rb") as f:
                self._ml_fallback = pickle.load(f)

        logger.info("%s 参数已加载: %s", self.name, json_path)

    @property
    def is_trained(self) -> bool:
        return self._trained
