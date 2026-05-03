"""predictor.py — 预测执行器: 未来特征外推 + 分位数预测 + 趋势集成 + ECM

v2 改进:
  - 未来特征迭代构建: 用预测值回填 lag 特征，不再退化到 last_value
  - ECM 传入正确残差序列 (y_true - y_pred_lgb)
  - 集成权重调优: LGB 保持更高权重
  - 回退路径 P10/P90 用经验分位数替代 1.28σ 假设
"""
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
        cfg = load_config()
        pred_cfg = cfg.get("predictor", {})
        self._lgb_weight_min = pred_cfg.get("lgb_weight_min", 0.45)
        self._lgb_weight_decay = pred_cfg.get("lgb_weight_decay", 0.002)
        self._ecm_clip_ratio = pred_cfg.get("ecm_clip_ratio", 0.30)
        self._lookback_days = pred_cfg.get("lookback_days", 60)

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

        if reference_date:
            end_date = pd.to_datetime(reference_date)
        else:
            end_date = self._find_latest_date(province, target_type)
            if end_date is None:
                end_date = datetime.now()

        # 自适应 lookback: 最少14天，优先用配置值，数据不足自动降级
        min_lookback = 14
        for try_days in [self._lookback_days, 30, min_lookback]:
            start_date = end_date - timedelta(days=try_days)
            history = self.store.load_features(
                province, target_type,
                start_date.strftime("%Y-%m-%d"),
                (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if len(history) >= 96 * min_lookback:
                break
            logger.info("lookback=%d天 数据不足(%d行), 降级重试", try_days, len(history))

        if history.empty or len(history) < 96:
            raise ValueError(f"没有可用的特征数据: {province}/{target_type}")

        future_features = self._build_future_features(
            history, province, target_type, horizon_steps, end_date
        )

        # ── LGB 分位数模型 ──
        quantile_models = None
        feature_names = None
        try:
            quantile_models = self.trainer.load_quantile_models(province, target_type)
            predictions = self._predict_quantile(quantile_models, future_features,
                                                  province, target_type)
            _, feature_names = quantile_models.get("p50", (None, None))
        except (FileNotFoundError, KeyError):
            model, feature_names, _ = self.trainer.load_model(province, target_type)
            predictions = self._predict_with_model(
                model, future_features, feature_names,
                province=province, target_type=target_type, history=history,
            )

        # ── XGBoost 模型 (P50) ──
        xgb_p50 = None
        try:
            xgb_model, _ = self.trainer.load_xgboost_model(province, target_type)
            for fn in feature_names:
                if fn not in future_features.columns:
                    future_features[fn] = 0.0
            xgb_p50 = xgb_model.predict(future_features[feature_names].values)
        except (FileNotFoundError, KeyError):
            pass

        # ── 迭代 lag 回填 ──
        if horizon_steps > 96:
            predictions = self._iterative_lag_backfill(
                predictions, future_features, history,
                quantile_models, feature_names, province, target_type,
                horizon_steps
            )

        trend_pred = self._predict_trend(history, horizon_steps)
        ensemble = self._ensemble(predictions, trend_pred, history, target_type,
                                  xgb_p50=xgb_p50)

        # 光伏残差还原: 模型预测的是 value - value_lag_1d, 加回基线
        if "光伏" in target_type and "value_lag_1d" in future_features.columns:
            n_ens = len(ensemble)
            lag1d = future_features["value_lag_1d"].values[:n_ens]
            for col in ["p10", "p50", "p90"]:
                if col in ensemble.columns:
                    ensemble[col] = lag1d + ensemble[col].values

        # ECM 残差修正
        if len(history) > 200 and "value" in history.columns and feature_names:
            try:
                ensemble = self._apply_ecm_correction(
                    ensemble, history, quantile_models, feature_names,
                    province, target_type, horizon_steps
                )
            except Exception as e:
                logger.warning("ECM 残差修正失败 (%s/%s): %s", province, target_type, e)

        # 自适应区间校准
        ensemble = self._calibrate_intervals(ensemble, history, province, target_type)

        # 光伏夜间归零 — 树模型无法硬归零，后处理修复
        if "光伏" in target_type and "dt" in ensemble.columns:
            hours = pd.to_datetime(ensemble["dt"]).dt.hour
            night_mask = (hours < 6) | (hours >= 20)
            for col in ["p10", "p50", "p90"]:
                if col in ensemble.columns:
                    ensemble.loc[night_mask, col] = 0.0

        # 抽水蓄能方向感知 — 值接近零时用近期符号
        if "水电含抽蓄" in target_type and "dt" in ensemble.columns:
            # 对 P50 接近零的步，倾向于维持最近已知方向
            p50_vals = ensemble["p50"].values
            near_zero = np.abs(p50_vals) < 50.0
            if near_zero.any() and "value" in history.columns:
                recent_sign = np.sign(history["value"].tail(96).mean())
                if abs(recent_sign) > 0.1:
                    # 保留符号信息：接近零的预测值向最近趋势方向偏移
                    ensemble.loc[near_zero, "p50"] = (
                        ensemble.loc[near_zero, "p50"] + recent_sign * 20
                    )

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
                  target_type: str = "load",
                  xgb_p50: np.ndarray = None) -> pd.DataFrame:
        """集成预测: LGB + (XGB) + Trend 加权平均.

        有 XGBoost 时: LGB 0.45 + XGB 0.25 + Trend 0.30 (初始, 按时序衰减)
        无 XGBoost 时: LGB 0.85 + Trend 0.15 → LGB 0.45 + Trend 0.55 (96步后)
        """
        n = len(lgb_result)
        trend_preds = trend_preds[:n]
        lgb_p50 = lgb_result["p50"].values
        allow_neg = self._allow_negative(target_type)

        if xgb_p50 is not None and len(xgb_p50) >= n:
            xgb_p50 = xgb_p50[:n]
            if not allow_neg:
                xgb_p50 = np.maximum(xgb_p50, 0)
            # 3-way: LGB + XGB + Trend
            lgb_w = np.array([max(self._lgb_weight_min, 0.50 - self._lgb_weight_decay * i) for i in range(n)])
            xgb_w = np.full(n, 0.25)
            trend_w = 1.0 - lgb_w - xgb_w
            ensemble_p50 = lgb_w * lgb_p50 + xgb_w * xgb_p50 + trend_w * trend_preds
        else:
            # 2-way: LGB + Trend (回退)
            lgb_w = np.array([max(self._lgb_weight_min,
                                  0.85 - self._lgb_weight_decay * i) for i in range(n)])
            trend_w = 1.0 - lgb_w
            ensemble_p50 = lgb_w * lgb_p50 + trend_w * trend_preds

        result = lgb_result.copy()
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
            hist_vals = history["value"].values
            n_hist = len(hist_vals)

            # value_lag_1d: 96 步前 (24h)
            pos_1d = n_hist - lag_1d_step + i
            if pos_1d >= 0 and pos_1d < n_hist:
                row["value_lag_1d"] = float(hist_vals[pos_1d])
            else:
                # 回退: 用 value_lag_7d 替代 (上周同时刻)
                pos_7d = n_hist - lag_7d_step + i
                if pos_7d >= 0 and pos_7d < n_hist:
                    row["value_lag_1d"] = float(hist_vals[pos_7d])
                else:
                    row["value_lag_1d"] = last_value

            # value_lag_7d: 672 步前 (7天)
            pos_7d = n_hist - lag_7d_step + i
            if pos_7d >= 0 and pos_7d < n_hist:
                row["value_lag_7d"] = float(hist_vals[pos_7d])
            else:
                row["value_lag_7d"] = last_value

            # rolling mean: 最近 96 步均值 (始终从 history 尾部取)
            recent_96 = hist_vals[-96:] if n_hist >= 96 else hist_vals
            row["value_rolling_mean_24h"] = float(np.mean(recent_96))

            # diff 特征: 用已知 lag 计算
            lag1_val = row.get("value_lag_1d", last_value)
            lag7_val = row.get("value_lag_7d", last_value)
            row["value_diff_1d"] = float(last_value - lag1_val) if last_value and lag1_val else 0.0
            row["value_diff_7d"] = float(last_value - lag7_val) if last_value and lag7_val else 0.0

            # 方向特征 (抽水蓄能等双向运行类型)
            row["value_sign"] = float(np.sign(last_value)) if n_hist > 0 else 0.0
            # value_sign_lag_1d: 从历史查找对应96步前的符号
            sign1d_pos = n_hist - 96 + i
            if sign1d_pos >= 0 and sign1d_pos < n_hist:
                row["value_sign_lag_1d"] = float(np.sign(hist_vals[sign1d_pos]))
            else:
                row["value_sign_lag_1d"] = row["value_sign"]
            row["value_sign_change"] = 1.0 if abs(row["value_sign"] - row["value_sign_lag_1d"]) > 0.5 else 0.0

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

        # ── 白天/黑夜 + 时段细分 ──
        future_df["is_daylight"] = future_df["hour"].apply(lambda h: 1 if 6 <= h <= 18 else 0)
        future_df["time_of_day"] = future_df["hour"].apply(
            lambda h: 0 if h >= 22 or h < 6 else
                      1 if 6 <= h < 9 else
                      2 if 9 <= h < 12 else
                      3 if 12 <= h < 15 else
                      4 if 15 <= h < 18 else 5
        )
        future_df["season_x_tod"] = future_df["season"] * 6 + future_df["time_of_day"]
        if "temperature" in future_df.columns:
            _tb = pd.cut(future_df["temperature"].fillna(20),
                bins=[-100, 0, 10, 20, 30, 40, 100],
                labels=[0, 1, 2, 3, 4, 5], include_lowest=True).astype(int)
            future_df["daylight_x_temp"] = future_df["is_daylight"].astype(int) * 6 + _tb.astype(int)
        future_df["weekend_x_tod"] = future_df["is_weekend"].astype(int) * 6 + future_df["time_of_day"]
        # 气象列 fallback: 最近7天同时刻均值 (比全量均值更准)
        weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction",
                        "solar_radiation", "precipitation", "pressure"]
        for col in weather_cols:
            if col not in future_df.columns:
                if col in history.columns and len(history) >= 96:
                    # 按日内时刻分组取最近7天均值
                    history_hour = history.copy()
                    history_hour["_minute_of_day"] = history_hour["dt"].dt.hour * 4 + history_hour["dt"].dt.minute // 15
                    future_df["_minute_of_day"] = future_df["dt"].dt.hour * 4 + future_df["dt"].dt.minute // 15
                    hourly_mean = history_hour.tail(672).groupby("_minute_of_day")[col].mean()
                    future_df[col] = future_df["_minute_of_day"].map(hourly_mean).fillna(history[col].mean())
                    future_df.drop(columns=["_minute_of_day"], inplace=True, errors="ignore")
                elif col in history.columns:
                    future_df[col] = history[col].mean()
                else:
                    future_df[col] = 0.0

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

        # ── 派生天气特征: 温度极端度 + 时段×极端度交互 ──
        if "temperature" in future_df.columns:
            future_df["temp_extremity"] = np.abs(future_df["temperature"] - 22) / 15
            if "humidity" in future_df.columns:
                hum_factor = 1.0 + np.clip((future_df["humidity"] - 50) / 100, -0.2, 0.3)
                future_df["temp_extremity"] = future_df["temp_extremity"] * hum_factor
            _te = pd.cut(future_df["temp_extremity"].fillna(0.5),
                bins=[0, 0.2, 0.5, 1.0, 100],
                labels=[0, 1, 2, 3], include_lowest=True).astype(int)
            future_df["tod_x_temp_extreme"] = future_df["time_of_day"] * 4 + _te.astype(int)

        # ── cloud_factor: 实际/晴空辐照度, 剥离昼夜循环 ──
        if "solar_radiation" in future_df.columns:
            from scripts.core.config import load_config
            from scripts.data.weather_features import WeatherFeatureEngineer
            coords = load_config().get("province_coords", {})
            lat = coords.get(province, {}).get("lat") if coords else None
            if lat is not None:
                dts = pd.to_datetime(future_df["dt"])
                doy = dts.dt.dayofyear.values
                hrs = dts.dt.hour.values + dts.dt.minute.values / 60.0
                clear_sky = np.array([
                    WeatherFeatureEngineer._clear_sky_irradiance(lat, int(d), float(h))
                    for d, h in zip(doy, hrs)
                ])
                mask = clear_sky > 10
                cf = np.zeros(len(clear_sky))
                cf[mask] = np.clip(
                    future_df["solar_radiation"].values[mask] / clear_sky[mask],
                    0.0, 1.5
                )
                future_df["cloud_factor"] = cf
            else:
                future_df["cloud_factor"] = 0.0
        else:
            future_df["cloud_factor"] = 0.0

        # ── 极端天气标志 (未来行) ──
        if "temperature" in future_df.columns:
            t_mean = history["temperature"].mean() if "temperature" in history.columns else 22
            t_std = history["temperature"].std() if "temperature" in history.columns else 5
            t_std = max(t_std, 0.1)
            future_df["temp_zscore"] = (future_df["temperature"] - t_mean) / t_std
            future_df["is_heat_wave"] = ((future_df["temperature"] > 35) & (future_df["temp_zscore"] > 2)).astype(int)
            future_df["is_cold_snap"] = ((future_df["temperature"] < -5) & (future_df["temp_zscore"] < -2)).astype(int)

        # 极端天气综合标志
        extreme_fut = pd.DataFrame(index=future_df.index)
        extreme_fut["hw"] = future_df.get("is_heat_wave", 0)
        extreme_fut["cs"] = future_df.get("is_cold_snap", 0)
        extreme_fut["sw"] = (future_df.get("wind_speed", 0) > 15).astype(int)
        extreme_fut["hp"] = (future_df.get("precipitation", 0) > 25).astype(int)
        future_df["extreme_weather_flag"] = (extreme_fut.sum(axis=1) > 0).astype(int)
        future_df["extreme_weather_count"] = extreme_fut.sum(axis=1).astype(int)

        # ── 极端值统计特征 (未来行从 history 推导) ──
        if "value" in history.columns:
            hist_vals = history["value"].dropna()
            if len(hist_vals) >= 96:
                rmean = hist_vals.iloc[-96:].mean()
                rstd = max(hist_vals.iloc[-96:].std(), 0.01)
                future_df["value_zscore_24h"] = 0.0  # 未来值未知, 设为0
                future_df["value_percentile_7d"] = 0.5
            else:
                future_df["value_zscore_24h"] = 0.0
                future_df["value_percentile_7d"] = 0.5
        else:
            future_df["value_zscore_24h"] = 0.0
            future_df["value_percentile_7d"] = 0.5

        # ── 极端×时间交互 ──
        future_df["extreme_x_tod"] = future_df["extreme_weather_flag"] * future_df["time_of_day"]
        future_df["heat_wave_x_daylight"] = future_df["is_heat_wave"] * future_df["is_daylight"]
        if "temp_zscore" in future_df.columns:
            future_df["tzscore_x_tod"] = future_df["temp_zscore"] * future_df["time_of_day"]
        future_df["val_extreme_x_weather"] = (
            np.abs(future_df["value_zscore_24h"]) * future_df["extreme_weather_flag"]
        )

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

        # 光伏残差还原: 模型预测的是 value - value_lag_1d, 加回基线
        if "光伏" in target_type and "value_lag_1d" in features_df.columns:
            predicted = predicted + features_df["value_lag_1d"].values

        # 历史残差分布估算 P10/P90 (P05/P95 残差分位数 + 地板 + 时序扩容)
        n = len(predicted)
        p50_abs = np.maximum(np.abs(predicted), 1.0)
        lo_adj = np.full(n, -0.03 * p50_abs)  # 默认 3% 地板
        hi_adj = np.full(n, 0.03 * p50_abs)
        if history is not None and len(history) > 96 and "value" in history.columns:
            try:
                recent = history.tail(672).copy()
                if all(fn in recent.columns for fn in feature_names):
                    hist_pred = model.predict(recent[feature_names].values)
                    residuals = recent["value"].values - hist_pred
                    r_std = np.std(residuals)
                    r_mean = np.mean(residuals)
                    clean = residuals[(residuals > r_mean - 3*r_std) & (residuals < r_mean + 3*r_std)]
                    if len(clean) > 20:
                        base_lo = float(np.quantile(clean, 0.05)) * 1.3
                        base_hi = float(np.quantile(clean, 0.95)) * 1.3
                        max_adj = 0.60 * p50_abs
                        floor = 0.03 * p50_abs
                        lo_adj = np.clip(np.full(n, base_lo), -max_adj, -floor)
                        hi_adj = np.clip(np.full(n, base_hi), floor, max_adj)
                        # 时序扩容: 越远越宽 (最多 50%)
                        hfactor = 1.0 + 0.5 * np.arange(n) / max(n, 1)
                        lo_adj *= hfactor
                        hi_adj *= hfactor
            except Exception as e:
                logger.warning("经验分位数计算失败 (%s/%s): %s", province, target_type, e)

        p10 = predicted + lo_adj
        p90 = predicted + hi_adj
        result = pd.DataFrame({
            "dt": features_df["dt"].values, "province": province, "type": target_type,
            "p50": predicted,
            "p10": p10 if self._allow_negative(target_type) else np.maximum(p10, 0),
            "p90": p90 if self._allow_negative(target_type) else np.maximum(p90, 0),
            "model_version": "v2",
        })
        return result

    def _apply_ecm_correction(self, ensemble: pd.DataFrame,
                               history: pd.DataFrame,
                               quantile_models: Dict,
                               feature_names: List[str],
                               province: str, target_type: str,
                               horizon_steps: int) -> pd.DataFrame:
        """ECM 残差修正 v2: 用 LGB 在历史上的残差拟合 AR，而非原始 value.

        残差 = y_true - y_pred_lgb
        用 AR 预测未来残差，叠加到 ensemble p50 上。
        """
        recent = history.tail(672).copy()
        if len(recent) < 96:
            return ensemble

        for fn in feature_names:
            if fn not in recent.columns:
                recent[fn] = 0.0

        # 用 LGB 在历史上预测，计算残差
        if quantile_models:
            p50_m, _ = quantile_models.get("p50", (None, None))
        else:
            p50_m, _, _ = self.trainer.load_model(province, target_type)

        if p50_m is None:
            return ensemble

        X_hist = recent[feature_names].values
        lgb_hist_pred = p50_m.predict(X_hist)
        residuals = recent["value"].values - lgb_hist_pred

        # 剔除异常残差 (超过 3σ)
        res_std = np.std(residuals)
        if res_std > 0:
            clean_mask = np.abs(residuals - np.mean(residuals)) < 3 * res_std
            residuals[~clean_mask] = 0.0

        self.ecm.fit(residuals)
        residual_fix = self.ecm.predict(residuals[-self.ecm.order:], horizon_steps)

        # 安全边界: 修正量不超过 P50 的 30%
        p50_vals = ensemble["p50"].values
        clip_limit = np.maximum(np.abs(p50_vals), 1.0) * self._ecm_clip_ratio
        residual_fix = np.clip(residual_fix, -clip_limit, clip_limit)

        allow_neg = self._allow_negative(target_type)
        ensemble["p50"] = ensemble["p50"].values + residual_fix
        if not allow_neg:
            ensemble["p50"] = np.maximum(ensemble["p50"], 0)

        # P10/P90 应用一半修正 (保留区间宽度)
        half_fix = residual_fix / 2
        ensemble["p10"] = ensemble["p10"].values + half_fix
        ensemble["p90"] = ensemble["p90"].values + half_fix
        if not allow_neg:
            ensemble["p10"] = np.maximum(ensemble["p10"], 0)
            ensemble["p90"] = np.maximum(ensemble["p90"], 0)

        return ensemble

    def _iterative_lag_backfill(self, predictions: pd.DataFrame,
                                 future_features: pd.DataFrame,
                                 history: pd.DataFrame,
                                 quantile_models: Dict,
                                 feature_names: List[str],
                                 province: str, target_type: str,
                                 horizon_steps: int) -> pd.DataFrame:
        """迭代 lag 回填: 每 96 步用预测值更新后续 lag 特征，逐批重预测.

        解决 _build_future_features 中 lag 特征退化到 last_value 的问题。
        """
        batch_size = 96
        n_batches = (horizon_steps + batch_size - 1) // batch_size
        if n_batches <= 1:
            return predictions

        # 取关键特征名
        lag_cols = [c for c in future_features.columns if c.startswith("value_lag") or c.startswith("value_rolling") or c.startswith("value_diff")]
        if not lag_cols:
            return predictions

        result_rows = []
        last_values = history["value"].tail(672).tolist()

        for batch_idx in range(n_batches):
            start_i = batch_idx * batch_size
            end_i = min(start_i + batch_size, horizon_steps)
            batch_ff = future_features.iloc[start_i:end_i].copy()

            # 用当前 last_values 更新 lag 特征
            for i_rel, global_i in enumerate(range(start_i, end_i)):
                row_idx = batch_ff.index[i_rel]
                # value_lag_1d: 96步前
                lag1_pos = len(last_values) - 96
                if lag1_pos >= 0:
                    batch_ff.at[row_idx, "value_lag_1d"] = last_values[lag1_pos]
                # value_lag_7d: 672步前
                lag7_pos = len(last_values) - 672
                if lag7_pos >= 0:
                    batch_ff.at[row_idx, "value_lag_7d"] = last_values[lag7_pos]
                # rolling mean
                recent96 = last_values[-96:]
                batch_ff.at[row_idx, "value_rolling_mean_24h"] = np.mean(recent96) if recent96 else 0

            # 预测当前 batch
            if quantile_models:
                batch_pred = self._predict_quantile(quantile_models, batch_ff, province, target_type)
            else:
                model, fn, _ = self.trainer.load_model(province, target_type)
                batch_pred = self._predict_with_model(model, batch_ff, fn or feature_names,
                                                       province, target_type, history)

            result_rows.append(batch_pred)

            # 更新 last_values 缓冲：追加本批预测的 p50
            batch_p50 = batch_pred["p50"].tolist()
            last_values.extend(batch_p50)

        return pd.concat(result_rows, ignore_index=True)

    def _calibrate_intervals(self, ensemble: pd.DataFrame,
                             history: pd.DataFrame,
                             province: str, target_type: str) -> pd.DataFrame:
        """自适应区间校准: 基于历史残差分布 + 时序衰减扩容."""
        try:
            if "value" not in history.columns or len(history) < 96:
                return ensemble

            recent = history.tail(672).copy()
            if "value" not in recent.columns or len(recent) < 96:
                return ensemble

            p50_vals = ensemble["p50"].values
            n = len(p50_vals)

            # 尝试加载模型计算历史残差
            cal_lo = None
            cal_hi = None
            try:
                models = self.trainer.load_quantile_models(province, target_type)
                _, feature_names = models.get("p50", (None, None))
                if feature_names is not None:
                    p50_m = models.get("p50")[0]
                    for fn in feature_names:
                        if fn not in recent.columns:
                            recent[fn] = 0.0
                    X_hist = recent[feature_names].values
                    hist_p50 = p50_m.predict(X_hist)
                    actual = recent["value"].values
                    denom = np.maximum(np.abs(hist_p50), 1.0)
                    rel_residuals = np.abs(actual - hist_p50) / denom
                    sigma_rel = float(np.quantile(rel_residuals, 0.80))
                    sigma_rel = max(sigma_rel, 0.02)
                    sigma_rel *= 1.3
                    cal_lo = sigma_rel * np.abs(p50_vals)
                    cal_hi = sigma_rel * np.abs(p50_vals)
            except Exception:
                pass

            # 回退: 直接用历史 residual 分布
            if cal_lo is None:
                try:
                    model, fnames, _ = self.trainer.load_model(province, target_type)
                    for fn in fnames:
                        if fn not in recent.columns:
                            recent[fn] = 0.0
                    hist_pred = model.predict(recent[fnames].values)
                    residuals = recent["value"].values - hist_pred
                    r_std = np.std(residuals)
                    r_mean = np.mean(residuals)
                    clean = residuals[(residuals > r_mean - 3*r_std) & (residuals < r_mean + 3*r_std)]
                    if len(clean) > 20:
                        q05 = float(np.quantile(clean, 0.05))
                        q95 = float(np.quantile(clean, 0.95))
                        # 扩张系数 1.3
                        q05 *= 1.3
                        q95 *= 1.3
                        cal_lo = np.full(n, -q05)
                        cal_hi = np.full(n, q95)
                except Exception:
                    pass

            if cal_lo is not None:
                # 混合模型区间 + 残差区间
                model_lo = ensemble["p50"].values - ensemble["p10"].values
                model_hi = ensemble["p90"].values - ensemble["p50"].values
                final_lo = 0.3 * model_lo + 0.7 * cal_lo
                final_hi = 0.3 * model_hi + 0.7 * cal_hi

                # 时序衰减扩容: 越远的预测越不确定
                horizon_factor = 1.0 + 0.4 * np.arange(n) / max(n, 1)
                final_lo *= horizon_factor
                final_hi *= horizon_factor

                # 上限: 不超过 P50 的 3x
                cap = np.abs(p50_vals) * 3.0
                final_lo = np.minimum(np.abs(final_lo), cap)
                final_hi = np.minimum(np.abs(final_hi), cap)

                allow_neg = self._allow_negative(target_type)
                raw_p10 = ensemble["p50"].values - final_lo
                raw_p90 = ensemble["p50"].values + final_hi
                ensemble["p10"] = raw_p10 if allow_neg else np.maximum(raw_p10, 0)
                ensemble["p90"] = raw_p90 if allow_neg else np.maximum(raw_p90, 0)
        except Exception as e:
            logger.warning("区间校准失败 (%s/%s): %s", province, target_type, e)

        return ensemble
