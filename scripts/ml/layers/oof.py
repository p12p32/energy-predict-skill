"""oof.py — KFold 时间序列 OOF 生成器, 为 Delta 层提供无偏 Level 预测."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import List
import logging

logger = logging.getLogger(__name__)


class OOFGenerator:
    def __init__(self, n_splits: int = 5):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits

    def generate(self, level_layer, df: pd.DataFrame,
                 feature_names: List[str],
                 target_col: str = "value") -> np.ndarray:
        n = len(df)
        oof = np.full(n, np.nan, dtype=float)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        y_full = df[target_col].values
        first_train_idx = None

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

            fold_layer = type(level_layer)(level_layer.transform_config)
            fold_layer.selector = level_layer.selector
            fold_layer.train(df_train, feature_names=feature_names,
                             target_col=target_col, oof_mode=False)

            X_val = df_val[feature_names].fillna(0).values
            pred = fold_layer._predict_raw(X_val)

            nan_mask = ~np.isfinite(pred)
            if nan_mask.any():
                fallback = np.nanmean(pred) if np.isfinite(pred).any() else np.mean(y_full)
                pred = np.where(nan_mask, fallback, pred)

            oof[val_idx] = pred

            if fold_idx == 0:
                first_train_idx = train_idx

            logger.debug("Fold %d/%d: train=%d, val=%d",
                         fold_idx + 1, self.n_splits, len(train_idx), len(val_idx))

        # TimeSeriesSplit 不覆盖最初 train 块, 用 fold-1 模型做 in-sample 填充
        if first_train_idx is not None:
            gap_mask = ~np.isfinite(oof)
            if gap_mask.any():
                gap_layer = type(level_layer)(level_layer.transform_config)
                gap_layer.selector = level_layer.selector
                gap_layer.train(df.iloc[first_train_idx], feature_names=feature_names,
                                 target_col=target_col, oof_mode=False)
                X_gap = df.iloc[gap_mask][feature_names].fillna(0).values
                gap_pred = gap_layer._predict_raw(X_gap)
                nan_mask = ~np.isfinite(gap_pred)
                if nan_mask.any():
                    fallback = np.nanmean(gap_pred) if np.isfinite(gap_pred).any() else np.mean(y_full)
                    gap_pred = np.where(nan_mask, fallback, gap_pred)
                oof[gap_mask] = gap_pred
                logger.info("OOF 初始块填充: %d 行 (in-sample)", gap_mask.sum())

        return oof
