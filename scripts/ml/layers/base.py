"""base.py — 各层抽象基类, 统一 train/predict/save/load 接口."""
import os, json, logging
from abc import ABC, abstractmethod
import numpy as np
import lightgbm as lgb
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseLayer(ABC):
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.model = None
        self.feature_names: List[str] = []
        self.metadata: dict = {}
        self._trained = False

    @abstractmethod
    def train(self, df, **kwargs) -> dict:
        """训练本层. 返回 metadata."""

    @abstractmethod
    def predict(self, X, **kwargs) -> np.ndarray:
        """预测. 返回 numpy array."""

    def save(self, path: str) -> None:
        model = self.model
        if model is not None:
            if hasattr(model, "booster_"):
                model.booster_.save_model(path)
            elif hasattr(model, "_Booster"):
                model._Booster.save_model(path)
        meta_path = path + ".layer_meta"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "layer_name": self.layer_name,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            }, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        meta_path = path + ".layer_meta"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
                self.metadata = meta.get("metadata", {})
        if not os.path.exists(path):
            self._trained = True
            return
        booster = lgb.Booster(model_file=path)
        obj = booster.params.get("objective", "")
        if obj in ("binary", "multiclass", "multiclassova"):
            num_class = booster.params.get("num_class", 2)
            model = lgb.LGBMClassifier()
            model._Booster = booster
            model._n_classes = num_class
            model._n_features = booster.num_feature()
        else:
            model = lgb.LGBMRegressor()
            model._Booster = booster
            model._n_features = booster.num_feature()
        model.n_features_in_ = model._n_features
        model.fitted_ = True
        self.model = model
        self._trained = True

    def _extract_features(self, df: "pd.DataFrame") -> np.ndarray:
        """安全提取特征矩阵, 缺失列填 0.0."""
        import pandas as pd
        X = np.zeros((len(df), len(self.feature_names)), dtype=np.float64)
        for i, fn in enumerate(self.feature_names):
            if fn in df.columns:
                col = pd.to_numeric(df[fn], errors="coerce").fillna(0).values
                X[:, i] = col
        return X

    @property
    def is_trained(self) -> bool:
        return self._trained
