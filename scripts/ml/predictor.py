"""predictor.py — 预测执行器: 未来特征外推 + 分位数预测 + 趋势集成 + ECM"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

from scripts.ml.trainer import Trainer
from scripts.ml.trend import TrendModel
from scripts.ml.error_correction import ErrorCorrectionModel
from scripts.data.features import FeatureStore
from scripts.data.fetcher import DataFetcher
from scripts.core.config import load_config, validate_province_and_type, parse_type


class Predictor:
    def __init__(self, trainer: Trainer = None,
                 store: FeatureStore = None):
        self.trainer = trainer or Trainer()
        self.store = store or FeatureStore()
        self.fetcher = DataFetcher()
        self.ecm = ErrorCorrectionModel()

    @staticmethod
    def _allow_negative(target_type: str) -> bool:
        """price 类型允许负值预测 (山东电力市场可能出现负电价)."""
        return target_type == "price"

    def _find_latest_date(self, province: str, target_type: str):
        """扫描特征库，返回该 province/type 的最新数据日期."""
        import glob, os
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            ".energy_data", "features",
        )
        pattern = os.path.join(base_dir, f"{province}_{target_type}_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        # 读取最后一个文件的最新日期
        try:
            df = pd.read_parquet(files[-1], columns=["dt"])
            if not df.empty:
                return pd.to_datetime(df["dt"].max())
        except Exception:
            pass
        return None

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24,
                model_version: str = None,
                reference_date: str = None) -> pd.DataFrame:
        validate_province_and_type(province, target_type)
        horizon_steps = horizon_hours * 4

        lookback_days = 14
        if reference_date:
            end_date = pd.to_datetime(reference_date)
        else:
            # 自动探测: 用特征库中最新日期
            end_date = self._find_latest_date(province, target_type)
            if end_date is None:
                end_date = datetime.now()

        start_date = end_date - timedelta(days=lookback_days)

        history = self.store.load_features(
            province, target_type,
            start_date.strftime("%Y-%m-%d"),
            (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if history.empty:
            raise ValueError(f"没有可用的特征数据: {province}/{target_type}")

        future_features = self._build_future_features(
            history, province, target_type, horizon_steps, end_date
        )

        # ── 尝试分位数模型 ──
        try:
            quantile_models = self.trainer.load_quantile_models(province, target_type)
            predictions = self._predict_quantile(quantile_models, future_features,
                                                  province, target_type)
        except (FileNotFoundError, KeyError):
            # 回退: 单模型 + 残差概率区间
            model, feature_names, _ = self.trainer.load_model(province, target_type)
            predictions = self._predict_with_model(
                model, future_features, feature_names,
                province=province, target_type=target_type, history=history,
            )

        trend_pred = self._predict_trend(history, horizon_steps)
        ensemble = self._ensemble(predictions, trend_pred, history, target_type)

        # ECM 残差修正
        if len(history) > 200 and "value" in history.columns:
            try:
                recent = history["value"].tail(500).values
                self.ecm.fit(recent)
                residual_fix = self.ecm.predict(np.zeros(self.ecm.order), horizon_steps)
                allow_neg = self._allow_negative(target_type)
                ensemble["p50"] = ensemble["p50"].values + residual_fix
                if not allow_neg:
                    ensemble["p50"] = np.maximum(ensemble["p50"], 0)
                half_fix = residual_fix / 2
                ensemble["p10"] = ensemble["p10"].values + half_fix
                ensemble["p90"] = ensemble["p90"].values + half_fix
                if not allow_neg:
                    ensemble["p10"] = np.maximum(ensemble["p10"], 0)
                    ensemble["p90"] = np.maximum(ensemble["p90"], 0)
            except Exception as e:
                logger.warning("ECM 残差修正失败 (%s/%s): %s", province, target_type, e)

        # 自适应区间校准
        ensemble = self._calibrate_intervals(ensemble, history, province, target_type)

        self.store.insert_predictions(ensemble)

        return ensemble

    def _predict_quantile(self, models: Dict, features_df: pd.DataFrame,
                           province: str, target_type: str) -> pd.DataFrame:
        """用分位数模型做预测."""
        p10_model, feature_names = models.get("p10", (None, None))
        p50_model, p50_feature_names = models.get("p50", (None, None))
        p90_model, _ = models.get("p90", (None, None))

        if p50_model is None:
            raise ValueError("无 P50 模型")

        if feature_names is None:
            feature_names = p50_feature_names or []
        for fn in feature_names:
            if fn not in features_df.columns:
                features_df[fn] = 0.0
        X = features_df[feature_names].values

        raw_p10 = p10_model.predict(X) if p10_model else np.zeros(len(X))
        raw_p50 = p50_model.predict(X)
        raw_p90 = p90_model.predict(X) if p90_model else np.zeros(len(X))
        allow_neg = self._allow_negative(target_type)
        result = pd.DataFrame({
            "dt": features_df["dt"].values,
            "province": province, "type": target_type,
            "p10": raw_p10 if allow_neg else np.maximum(raw_p10, 0),
            "p50": raw_p50 if allow_neg else np.maximum(raw_p50, 0),
            "p90": raw_p90 if allow_neg else np.maximum(raw_p90, 0),
            "model_version": "quantile_v1",
        })
        return result

    def _predict_trend(self, history: pd.DataFrame,
                       horizon_steps: int) -> np.ndarray:
        if "value" not in history.columns or len(history) < 96:
            return np.zeros(horizon_steps)
        values = history["value"].dropna().values
        if len(values) == 0:
            return np.zeros(horizon_steps)
        trend_model = TrendModel()
        trend_model.fit(values)
        return trend_model.predict_with_daily_pattern(horizon_steps, values)

    def _ensemble(self, lgb_result: pd.DataFrame,
                  trend_preds: np.ndarray,
                  history: pd.DataFrame,
                  target_type: str = "load") -> pd.DataFrame:
        n = len(lgb_result)
        trend_preds = trend_preds[:n]
        lgb_weight = np.array([max(0.3, 0.75 - 0.005 * i) for i in range(n)])
        trend_weight = 1.0 - lgb_weight
        lgb_p50 = lgb_result["p50"].values
        ensemble_p50 = lgb_weight * lgb_p50 + trend_weight * trend_preds
        result = lgb_result.copy()
        allow_neg = self._allow_negative(target_type)
        result["p50"] = ensemble_p50 if allow_neg else np.maximum(ensemble_p50, 0)
        p50_shift = result["p50"].values - lgb_p50
        result["p10"] = lgb_result["p10"].values + p50_shift
        result["p90"] = lgb_result["p90"].values + p50_shift
        if not allow_neg:
            result["p10"] = np.maximum(result["p10"], 0)
            result["p90"] = np.maximum(result["p90"], 0)
        result["trend_adjusted"] = True
        return result

    def _build_future_features(self, history: pd.DataFrame,
                                province: str, target_type: str,
                                horizon_steps: int,
                                base_dt: datetime) -> pd.DataFrame:
        last_row = history.iloc[-1].copy()
        last_value = last_row.get("value", 0)
        last_price = last_row.get("price", 0)
        ti = parse_type(target_type)
        future_times = pd.date_range(
            start=base_dt + timedelta(minutes=15),
            periods=horizon_steps,
            freq="15min",
        )
        rows = []
        for i, ft in enumerate(future_times):
            row = {
                "dt": ft, "province": province, "type": target_type,
                "sub_type": ti.sub or "",
                "value_type": ti.value_type,
                "value": None, "price": last_price,
                "hour": ft.hour, "day_of_week": ft.dayofweek,
                "day_of_month": ft.day, "month": ft.month,
                "is_weekend": ft.dayofweek in [5, 6],
                "season": (1 if ft.month in [3, 4, 5]
                           else 2 if ft.month in [6, 7, 8]
                           else 3 if ft.month in [9, 10, 11] else 4),
            }
            lag_1d_step, lag_7d_step = 96, 672
            if i < lag_1d_step and len(history) > lag_1d_step:
                row["value_lag_1d"] = history.iloc[-(lag_1d_step - i)].get("value", last_value)
            else:
                row["value_lag_1d"] = last_value
            if i < lag_7d_step and len(history) > lag_7d_step:
                row["value_lag_7d"] = history.iloc[-(lag_7d_step - i)].get("value", last_value)
            else:
                row["value_lag_7d"] = last_value
            recent_96 = history["value"].tail(96)
            row["value_rolling_mean_24h"] = recent_96.mean() if not recent_96.empty else last_value
            row["value_diff_1d"] = 0.0
            row["value_diff_7d"] = 0.0
            rows.append(row)
        future_df = pd.DataFrame(rows)

        from scripts.data.holidays import add_holiday_features, add_cyclical_features
        future_df = add_holiday_features(future_df)
        future_df = add_cyclical_features(future_df)
        future_df["quality_flag"] = 0

        # 交互特征
        future_df["peak_valley"] = future_df["hour"].apply(
            lambda h: 2 if h in [8, 9, 10, 11, 17, 18, 19, 20] else
                      1 if h in [12, 13, 14, 21, 22] else 0
        )
        future_df["weekend_hour"] = future_df["is_weekend"].astype(int) * future_df["hour"]
        future_df["dow_hour"] = future_df["day_of_week"] * 24 + future_df["hour"]
        future_df["weekend_x_lag7d"] = future_df["is_weekend"].astype(int) * future_df["value_lag_7d"]
        future_df["hour_x_lag1d"] = future_df["hour"] * future_df["value_lag_1d"]
        for col in ["temperature", "humidity", "wind_speed", "wind_direction",
                     "solar_radiation", "precipitation", "pressure"]:
            if col not in future_df.columns:
                future_df[col] = history[col].mean() if col in history.columns else 0.0

        try:
            forecast_end = (base_dt + timedelta(days=8)).strftime("%Y-%m-%d")
            weather = self.fetcher.fetch_weather(
                province, base_dt.strftime("%Y-%m-%d"), forecast_end, mode="forecast")
            if not weather.empty:
                weather["dt_merge"] = weather["dt"].dt.floor("15min")
                future_df["dt_merge"] = future_df["dt"].dt.floor("15min")
                weather_cols = ["temperature", "humidity", "wind_speed",
                                "wind_direction", "solar_radiation", "precipitation", "pressure"]
                for col in weather_cols:
                    if col in weather.columns:
                        merged = weather[["dt_merge"] + [col]].copy()
                        future_df = future_df.merge(merged, on="dt_merge", how="left", suffixes=("", "_w"))
                        if col in future_df.columns:
                            future_df[col] = future_df[col].fillna(
                                history[col].mean() if col in history.columns else 0)
                future_df.drop(columns=["dt_merge"], inplace=True, errors="ignore")
        except Exception as e:
            logger.warning("气象数据获取失败 (%s/%s): %s", province, target_type, e)
        for col in future_df.columns:
            if col not in ("dt", "province", "type") and future_df[col].dtype == np.float64:
                future_df[col] = future_df[col].fillna(
                    history[col].mean() if col in history.columns else 0)
        return future_df

    def _predict_with_model(self, model, features_df: pd.DataFrame,
                            feature_names: List[str], province: str,
                            target_type: str,
                            history: pd.DataFrame = None) -> pd.DataFrame:
        for fn in feature_names:
            if fn not in features_df.columns:
                features_df[fn] = 0.0
        predict_features = features_df[feature_names].copy()
        predicted = model.predict(predict_features)
        residual_std = 0.05
        if history is not None and len(history) > 96 and "value" in history.columns:
            try:
                recent = history.tail(96)
                if all(fn in recent.columns for fn in feature_names):
                    hist_pred = model.predict(recent[feature_names].values)
                    hist_actual = recent["value"].values
                    mask = hist_actual != 0
                    if mask.sum() > 10:
                        residuals = (hist_actual[mask] - hist_pred[mask]) / hist_actual[mask]
                        residual_std = float(np.std(residuals))
            except Exception as e:
                logger.warning("残差标准差计算失败 (%s/%s): %s", province, target_type, e)
        p10 = predicted * (1 - 1.28 * residual_std)
        p90 = predicted * (1 + 1.28 * residual_std)
        result = pd.DataFrame({
            "dt": features_df["dt"].values, "province": province, "type": target_type,
            "p50": predicted,
            "p10": p10 if self._allow_negative(target_type) else np.maximum(p10, 0),
            "p90": p90 if self._allow_negative(target_type) else np.maximum(p90, 0),
            "model_version": "v1",
        })
        return result

    def _calibrate_intervals(self, ensemble: pd.DataFrame,
                             history: pd.DataFrame,
                             province: str, target_type: str) -> pd.DataFrame:
        """自适应区间校准: 历史残差分位数 + 分布偏移缓冲."""
        try:
            if "value" not in history.columns or len(history) < 96:
                return ensemble

            try:
                models = self.trainer.load_quantile_models(province, target_type)
                feature_names = list(models.values())[0][1] if models else []
            except Exception as e:
                logger.warning("校准阶段加载分位数模型失败 (%s/%s): %s", province, target_type, e)
                return ensemble

            if not feature_names:
                return ensemble

            recent = history.tail(672).copy()
            if "value" not in recent.columns or len(recent) < 96:
                return ensemble

            for fn in feature_names:
                if fn not in recent.columns:
                    recent[fn] = 0.0

            X_hist = recent[feature_names].values
            actual = recent["value"].values

            p50_m, _ = models.get("p50", (None, None))
            if p50_m is None:
                return ensemble

            hist_p50 = p50_m.predict(X_hist)

            # ── 相对残差 (避免绝对值受量级影响) ──
            denom = np.maximum(np.abs(hist_p50), 1.0)
            rel_residuals = np.abs(actual - hist_p50) / denom

            target_quantile = 0.80
            sigma_rel = float(np.quantile(rel_residuals, target_quantile))
            sigma_rel = max(sigma_rel, 0.02)

            # 分布偏移缓冲
            sigma_rel *= 1.3

            # ── 应用: 相对于 P50 缩放的宽度 ──
            p50 = np.maximum(ensemble["p50"].values, 1.0)
            model_lo = ensemble["p50"].values - ensemble["p10"].values
            model_hi = ensemble["p90"].values - ensemble["p50"].values

            # 残差驱动的宽度
            cal_lo = sigma_rel * p50
            cal_hi = sigma_rel * p50

            # 混合: 模型不对称性 40% + 残差宽度 60%
            final_lo = 0.4 * model_lo + 0.6 * cal_lo
            final_hi = 0.4 * model_hi + 0.6 * cal_hi

            # 上限: 宽度不超过 P50 的 2x
            cap = p50 * 2.0
            final_lo = np.minimum(final_lo, cap)
            final_hi = np.minimum(final_hi, cap)

            allow_neg = self._allow_negative(target_type)
            raw_p10 = ensemble["p50"].values - final_lo
            raw_p90 = ensemble["p50"].values + final_hi
            ensemble["p10"] = raw_p10 if allow_neg else np.maximum(raw_p10, 0)
            ensemble["p90"] = raw_p90 if allow_neg else np.maximum(raw_p90, 0)
        except Exception as e:
            logger.warning("区间校准失败 (%s/%s): %s", province, target_type, e)

        return ensemble
