"""training.py — 分层训练编排器: 按阶段串联 State→Level→Delta→TS→Fusion."""
import os, json, logging
from datetime import datetime
import numpy as np
import pandas as pd

from scripts.ml.layers.transform import TransformSelector
from scripts.ml.layers.state import StateLayer
from scripts.ml.layers.level import LevelLayer
from scripts.ml.layers.delta import DeltaLayer
from scripts.ml.layers.ts import TSLayer
from scripts.ml.layers.fusion import FusionLayer, apply_trend_gating
from scripts.ml.layers.oof import OOFGenerator
from scripts.ml.layers.trend_classify import TrendClassifyLayer
from scripts.ml.layers.price_regime import PriceRegimeRegressor
from scripts.ml.trend import TrendModel

logger = logging.getLogger(__name__)

EXCLUDE_TRAIN_COLS = {"dt", "province", "type", "price", "model_version",
                       "p10", "p50", "p90", "trend_adjusted"}


class LayeredTrainer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.selector = TransformSelector()
        self.oof_gen = OOFGenerator(n_splits=5)
        self.train_ratio = 0.70
        self.holdout_ratio = 0.20

    def train(self, df: pd.DataFrame, province: str, target_type: str,
              feature_names: list = None) -> dict:
        """完整训练: 10阶段 (非电价) 或电价专用管线."""
        df = df.sort_values("dt").reset_index(drop=True)
        if len(df) < 500:
            return {"error": f"数据不足: {len(df)} 行"}

        # 电价类型走新架构: PriceClassify → PriceRegime ×3 → TS
        if "电价" in target_type:
            return self._train_price(df, province, target_type, feature_names)

        if feature_names is None:
            feature_names = [c for c in df.columns
                             if c not in EXCLUDE_TRAIN_COLS
                             and c != "value"
                             and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]

        # Phase 0: 变换选择
        logger.info("[Phase 0] 变换选择: %s/%s", province, target_type)
        transform_cfg = self.selector.select(df["value"].dropna().values, target_type)

        # Phase 2: 时间切分
        n = len(df)
        s1_end = int(n * self.train_ratio)
        s2_end = int(n * (self.train_ratio + self.holdout_ratio))
        s1 = df.iloc[:s1_end].copy()
        s2 = df.iloc[s1_end:s2_end].copy()
        s3 = df.iloc[s2_end:].copy()
        logger.info("[Phase 2] 切分: S1=%d, S2=%d, S3=%d", len(s1), len(s2), len(s3))

        # Phase 3: State 训练
        logger.info("[Phase 3] State 训练: %s/%s", province, target_type)
        state = StateLayer()
        state.train(s1, target_type=target_type, feature_names=feature_names)

        # Phase 4: Level 训练 + KFold OOF
        logger.info("[Phase 4] Level 训练 + OOF: %s/%s", province, target_type)
        level = LevelLayer(transform_config=transform_cfg)
        level.train(s1, target_type=target_type, feature_names=feature_names, oof_mode=False)
        level_oof = self.oof_gen.generate(level, s1, feature_names)
        s1["level_oof"] = level_oof

        # Phase 5: Delta 训练
        logger.info("[Phase 5] Delta 训练: %s/%s", province, target_type)
        delta = DeltaLayer()
        delta.train(s1, level_oof=level_oof, feature_names=feature_names)
        # 需要 level_oof 作为特征列
        delta.feature_names = list(feature_names) + ["level_oof"]

        # Phase 6: TS 训练
        logger.info("[Phase 6] TS 训练: %s/%s", province, target_type)
        ts = TSLayer()
        ts.train(s1, feature_names=feature_names)

        # Phase 6b: Trend 拟合
        logger.info("[Phase 6b] Trend 拟合: %s/%s", province, target_type)
        trend = TrendModel()
        trend_vals = s1["value"].dropna().values
        if len(trend_vals) >= 96:
            trend.fit(trend_vals)
        else:
            trend.level = float(np.mean(trend_vals)) if len(trend_vals) > 0 else 0.0
            trend.trend = 0.0

        # Phase 6c: TrendClassify 训练
        logger.info("[Phase 6c] TrendClassify 训练: %s/%s", province, target_type)
        trend_cls = TrendClassifyLayer()
        trend_cls_result = trend_cls.train(s1, target_type=target_type,
                                            feature_names=feature_names)

        # Phase 7: 前向预测 S2
        logger.info("[Phase 7] 前向预测 S2: %d 行", len(s2))
        s2_preds = self._forward_predict(
            s1, s2, state, level, delta, ts, trend, trend_cls,
            province, target_type, feature_names
        )

        # Phase 8: Fusion 训练
        logger.info("[Phase 8] 融合训练: %s/%s", province, target_type)
        fusion = FusionLayer()
        meta = fusion.build_meta_features(
            s2_preds["base"], s2_preds["ts_pred"],
            s2_preds["trend"], s2,
        )
        fusion.train(meta, y=s2["value"].values)

        # Phase 9: S3 验证
        logger.info("[Phase 9] S3 验证: %d 行", len(s3))
        val_result = self._validate(
            s1, s2, s3, state, level, delta, ts, trend, trend_cls, fusion,
            province, target_type, feature_names
        )

        # 保存模型
        self._save_all(province, target_type, state, level, delta, ts, fusion,
                       trend, trend_cls, transform_cfg, feature_names)

        return {
            "province": province, "target_type": target_type,
            "transform": str(transform_cfg),
            "n_s1": len(s1), "n_s2": len(s2), "n_s3": len(s3),
            "ts_features": len(ts.feature_names),
            "state_active": state.active,
            "validation": val_result,
        }

    def _forward_predict(self, s1, s2, state, level, delta, ts, trend,
                         trend_cls, province, target_type, feature_names):
        """批量前向预测 S2 (比逐行快 100x+)."""
        from scripts.ml.features_future import build_future_features

        n = len(s2)
        end_date = s2["dt"].iloc[-1]
        ff = build_future_features(s1, province, target_type, n, end_date)

        prob_on = state.predict(ff)
        lev = level.predict(ff, prob_on=prob_on)
        d = delta.predict(ff, level_pred=lev)
        base = lev * (1.0 + d)
        ts_pred = ts.predict(ff)
        trend_probs = trend_cls.predict(ff)

        trend_vals = s1["value"].dropna().values
        if len(trend_vals) >= 96:
            t = TrendModel()
            t.fit(trend_vals)
            tr = t.predict_with_daily_pattern(n, trend_vals)
        else:
            tr = np.full(n, np.mean(trend_vals)) if len(trend_vals) > 0 else np.zeros(n)

        return {
            "base": base,
            "ts_pred": ts_pred,
            "trend": tr[:n],
            "level": lev,
            "trend_probs": trend_probs,
        }

    def _validate(self, s1, s2, s3, state, level, delta, ts, trend, trend_cls, fusion,
                  province, target_type, feature_names):
        if len(s3) == 0:
            return {"mape": None, "rmse": None, "n": 0}

        from scripts.ml.features_future import build_future_features
        history = pd.concat([s1, s2], ignore_index=True)
        history["dt"] = pd.to_datetime(history["dt"])
        n = len(s3)
        end_date = s3["dt"].iloc[-1]
        ff = build_future_features(history, province, target_type, n, end_date)

        prob_on = state.predict(ff)
        lev = level.predict(ff, prob_on=prob_on)
        d = delta.predict(ff, level_pred=lev)
        base = lev * (1.0 + d)
        ts_pred = ts.predict(ff)
        trend_probs = trend_cls.predict(ff)

        trend_vals = history["value"].dropna().values
        if len(trend_vals) >= 96:
            t = TrendModel()
            t.fit(trend_vals)
            tr = t.predict_with_daily_pattern(n, trend_vals)
        else:
            tr = np.full(n, np.mean(trend_vals)) if len(trend_vals) > 0 else np.zeros(n)

        meta = fusion.build_meta_features(base, ts_pred, tr, ff)
        ensemble = fusion.predict(meta)
        preds = apply_trend_gating(ensemble, base, ts_pred, tr[:n],
                                    trend_probs, target_type, future_df=ff)
        actuals = s3["value"].values

        mape = float(np.mean(np.abs(preds - actuals) / (np.abs(actuals) + 1)) * 100)
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        nrmse = float(rmse / np.std(actuals)) if np.std(actuals) > 0 else float("inf")

        # 趋势方向准确率
        trend_eval = trend_cls.evaluate(ff, s3["value"].values)

        logger.info("S3 验证: MAPE=%.2f%%, RMSE=%.1f, NRMSE=%.3f, TrendDirAcc=%s, n=%d",
                     mape, rmse, nrmse,
                     f"{trend_eval.get('accuracy', 0)*100:.1f}%" if trend_eval.get('accuracy') else "N/A",
                     n)
        return {"mape": mape, "rmse": rmse, "nrmse": nrmse, "n": n, "trend_direction": trend_eval}

    # ─── 电价专用训练: PriceClassify → PriceRegime ×3 → TS ──────

    def _train_price(self, df: pd.DataFrame, province: str, target_type: str,
                     feature_names: list) -> dict:
        logger.info("[电价管线] %s/%s — PriceClassify + PriceRegime ×3", province, target_type)

        if feature_names is None:
            feature_names = [c for c in df.columns
                             if c not in EXCLUDE_TRAIN_COLS
                             and c != "value"
                             and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]

        # 频率检测 — 只用最近7天 (避免历史频率变化)
        recent_cutoff = df["dt"].max() - pd.Timedelta(days=7)
        recent = df[df["dt"] >= recent_cutoff]
        if len(recent) >= 24:
            freq_minutes = recent["dt"].diff().mode().iloc[0].total_seconds() / 60
        else:
            freq_minutes = df["dt"].diff().mode().iloc[0].total_seconds() / 60
        points_per_hour = int(round(60 / freq_minutes))
        logger.info("[频率] %s: %.0fmin/点, %d点/小时", province, freq_minutes, points_per_hour)

        # 山东电价: 数据驱动极端阈值 (无业务floor)
        extreme_floor = 0.0 if "山东" in province else 800.0
        logger.info("[极端阈值] %s: extreme_floor=%.0f", province, extreme_floor)

        # Phase 0: 变换选择
        transform_cfg = self.selector.select(df["value"].dropna().values, target_type)

        # Phase 2: 时间切分
        n = len(df)
        s1_end = int(n * self.train_ratio)
        s2_end = int(n * (self.train_ratio + self.holdout_ratio))
        s1 = df.iloc[:s1_end].copy()
        s2 = df.iloc[s1_end:s2_end].copy()
        s3 = df.iloc[s2_end:].copy()
        logger.info("[Phase 2] 切分: S1=%d, S2=%d, S3=%d", len(s1), len(s2), len(s3))

        # Phase 3-5: PriceRegime 训练 (分类 + 3 回归)
        logger.info("[Phase 3-5] PriceRegime 训练")
        price_regime = PriceRegimeRegressor()
        regime_result = price_regime.train(s1, target_type=target_type,
                                            feature_names=feature_names,
                                            extreme_floor=extreme_floor)

        # Phase 6: TS 训练
        logger.info("[Phase 6] TS 训练")
        ts = TSLayer()
        ts.train(s1, feature_names=feature_names)

        # Phase 6b: Trend 拟合
        trend = TrendModel()
        trend_vals = s1["value"].dropna().values
        if len(trend_vals) >= 96:
            trend.fit(trend_vals)
        else:
            trend.level = float(np.mean(trend_vals)) if len(trend_vals) > 0 else 0.0
            trend.trend = 0.0

        # Phase 9: S3 验证
        logger.info("[Phase 9] S3 验证: %d 行", len(s3))
        val_result = self._validate_price(
            s1, s2, s3, price_regime, ts, trend,
            province, target_type, feature_names
        )

        # 保存模型
        base_dir = os.path.join(self.model_dir, province)
        os.makedirs(base_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{target_type}"
        price_regime.save(base_dir, f"{prefix}_{ts_str}")
        ts.save(os.path.join(base_dir, f"{prefix}_{ts_str}_ts.lgbm"))

        # Registry
        reg_path = os.path.join(self.model_dir, "model_registry.json")
        reg = {}
        if os.path.exists(reg_path):
            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)
        key = f"{province}_{target_type}"
        if "layered" not in reg:
            reg["layered"] = {}
        reg["layered"][key] = {
            "pipeline": "price",
            "transform": str(transform_cfg),
            "transform_name": transform_cfg.name,
            "transform_C": transform_cfg.C,
            "transform_lmbda": transform_cfg.lmbda,
            "ts_features": ts.feature_names,
            "feature_names": feature_names,
            "price_thresholds": price_regime.thresholds,
            "points_per_hour": points_per_hour,
            "extreme_floor": extreme_floor,
            "model_prefix": f"{prefix}_{ts_str}",
            "updated_at": datetime.now().isoformat(),
        }
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)

        logger.info("电价模型已保存: %s/%s → %s/", province, target_type, base_dir)

        return {
            "province": province, "target_type": target_type,
            "pipeline": "price",
            "transform": str(transform_cfg),
            "n_s1": len(s1), "n_s2": len(s2), "n_s3": len(s3),
            "price_thresholds": price_regime.thresholds,
            "regime_result": regime_result,
            "validation": val_result,
        }

    def _validate_price(self, s1, s2, s3, price_regime, ts, trend,
                         province, target_type, feature_names):
        if len(s3) == 0:
            return {"mape": None, "rmse": None, "nrmse": None, "n": 0}

        from scripts.ml.features_future import build_future_features
        history = pd.concat([s1, s2], ignore_index=True)
        history["dt"] = pd.to_datetime(history["dt"])
        n = len(s3)
        end_date = s3["dt"].iloc[-1]
        ff = build_future_features(history, province, target_type, n, end_date)

        # 价格区间预测
        preds = price_regime.predict(ff)
        actuals = s3["value"].values

        mape = float(np.mean(np.abs(preds - actuals) / (np.abs(actuals) + 1)) * 100)
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        nrmse = float(rmse / np.std(actuals)) if np.std(actuals) > 0 else float("inf")

        # TS 修正效果评估
        ts_pred = ts.predict(ff)
        ts_mape = float(np.mean(np.abs(ts_pred - actuals) / (np.abs(actuals) + 1)) * 100)

        logger.info("S3 验证(电价): MAPE=%.2f%%, RMSE=%.1f, NRMSE=%.3f, TS_MAPE=%.2f%%, n=%d",
                     mape, rmse, nrmse, ts_mape, n)
        return {"mape": mape, "rmse": rmse, "nrmse": nrmse, "ts_mape": ts_mape, "n": n}

    def _save_all(self, province, target_type, state, level, delta, ts, fusion,
                  trend, trend_cls, transform_cfg, feature_names):
        base_dir = os.path.join(self.model_dir, province)
        os.makedirs(base_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{target_type}"

        state.save(os.path.join(base_dir, f"{prefix}_{ts_str}_state.lgbm"))
        level.save(os.path.join(base_dir, f"{prefix}_{ts_str}_level.lgbm"))
        delta.save(os.path.join(base_dir, f"{prefix}_{ts_str}_delta.lgbm"))
        ts.save(os.path.join(base_dir, f"{prefix}_{ts_str}_ts.lgbm"))
        trend_cls.save(os.path.join(base_dir, f"{prefix}_{ts_str}_trend_cls.lgbm"))
        fusion.save(os.path.join(base_dir, f"{prefix}_{ts_str}_fusion.lgbm"))

        # Registry
        reg_path = os.path.join(self.model_dir, "model_registry.json")
        reg = {}
        if os.path.exists(reg_path):
            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)

        key = f"{province}_{target_type}"
        if "layered" not in reg:
            reg["layered"] = {}
        reg["layered"][key] = {
            "transform": str(transform_cfg),
            "transform_name": transform_cfg.name,
            "transform_C": transform_cfg.C,
            "transform_lmbda": transform_cfg.lmbda,
            "ts_features": ts.feature_names,
            "feature_names": feature_names,
            "state_active": state.active,
            "model_prefix": f"{prefix}_{ts_str}",
            "updated_at": datetime.now().isoformat(),
        }

        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)

        logger.info("模型已保存: %s/%s → %s/", province, target_type, base_dir)
