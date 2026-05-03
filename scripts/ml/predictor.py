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
        except (FileNotFoundError, KeyError, ValueError):
            pass

        # ── 迭代 lag 回填 ──
        if horizon_steps > 96:
            predictions = self._iterative_lag_backfill(
                predictions, future_features, history,
                quantile_models, feature_names, province, target_type,
                horizon_steps
            )

        trend_pred = self._predict_trend(history, horizon_steps, target_type)
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
            dts = pd.to_datetime(ensemble["dt"])
            hours = dts.dt.hour
            mins = dts.dt.minute
            night_mask = (hours >= 20) | (hours < 6) | ((hours == 6) & (mins < 15))
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
        missing = [fn for fn in feature_names if fn not in features_df.columns]
        if missing:
            missing_df = pd.DataFrame(0.0, index=features_df.index, columns=missing)
            features_df = pd.concat([features_df, missing_df], axis=1)
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
                       horizon_steps: int, target_type: str = "") -> np.ndarray:
        if "value" not in history.columns or len(history) < 96:
            return np.zeros(horizon_steps)
        values = history["value"].dropna().values
        if len(values) == 0:
            return np.zeros(horizon_steps)

        # 波动型: Holt线性外推不可靠, 用多日平均日内模式作为趋势基线
        VOLATILE_TREND_TYPES = ["风电", "联络线", "非市场"]
        if any(kw in target_type for kw in VOLATILE_TREND_TYPES) and len(values) >= 96 * 3:
            days = min(len(values) // 96, 7)
            reshaped = values[-days * 96:].reshape(days, 96)
            pattern = np.mean(reshaped, axis=0)
            repeats = horizon_steps // 96 + 1
            return np.tile(pattern, repeats)[:horizon_steps]

        trend_model = TrendModel()
        trend_model.fit(values)
        return trend_model.predict_with_daily_pattern(horizon_steps, values)

    def _ensemble(self, lgb_result: pd.DataFrame,
                  trend_preds: np.ndarray,
                  history: pd.DataFrame,
                  target_type: str = "load",
                  xgb_p50: np.ndarray = None) -> pd.DataFrame:
        """集成预测: LGB + (XGB) + Trend 加权平均.

        波动型(风电/联络线/光伏/非市场): LGB 高权重, 趋势基线为昨日值(非Holt外推).
        稳定型(总/负荷/地方/自备): 趋势模型可靠, 允许更多趋势权重.
        """
        n = len(lgb_result)
        trend_preds = trend_preds[:n]
        lgb_p50 = lgb_result["p50"].values
        allow_neg = self._allow_negative(target_type)

        VOLATILE_TYPES = ["风电", "联络线", "光伏", "非市场"]
        is_volatile = any(kw in target_type for kw in VOLATILE_TYPES)

        if is_volatile:
            lgb_start, lgb_min = 0.90, 0.75
        else:
            lgb_start, lgb_min = 0.85, self._lgb_weight_min

        if xgb_p50 is not None and len(xgb_p50) >= n:
            xgb_p50 = xgb_p50[:n]
            if not allow_neg:
                xgb_p50 = np.maximum(xgb_p50, 0)
            if is_volatile:
                lgb_w = np.array([max(lgb_min, lgb_start - self._lgb_weight_decay * i) for i in range(n)])
                xgb_w = np.full(n, 0.08)
            else:
                lgb_w = np.array([max(lgb_min, 0.50 - self._lgb_weight_decay * i) for i in range(n)])
                xgb_w = np.full(n, 0.25)
            trend_w = 1.0 - lgb_w - xgb_w
            ensemble_p50 = lgb_w * lgb_p50 + xgb_w * xgb_p50 + trend_w * trend_preds
        else:
            lgb_w = np.array([max(lgb_min,
                                  lgb_start - self._lgb_weight_decay * i) for i in range(n)])
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
        hist_vals = history["value"].values
        n_hist = len(hist_vals)

        # 构建 datetime → index 查找表, 解决预测日期与数据截止日之间有缺口时 lag 位置错位
        hist_dt_index = pd.Series(np.arange(n_hist), index=history["dt"].values)

        def _find_pos(target_dt, offset_hours):
            """在 history 中查找 target_dt - offset_hours 时刻的 index 位置.
            找不到精确匹配时取最近的同时刻 (允许 ±30min), 都没有返回 None."""
            lag_dt = target_dt - timedelta(hours=offset_hours)
            if lag_dt in hist_dt_index.index:
                return hist_dt_index[lag_dt]
            for delta_m in [0, 15, -15, 30, -30, 45, -45, 60, -60]:
                candidate = lag_dt + timedelta(minutes=delta_m)
                if candidate in hist_dt_index.index:
                    return hist_dt_index[candidate]
            return None

        def _lookup_lag(target_dt, offset_hours):
            """在 history 中查找 target_dt - offset_hours 时刻的 value."""
            pos = _find_pos(target_dt, offset_hours)
            return float(hist_vals[pos]) if pos is not None else last_value

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

            # value_lag_1d / value_lag_7d: 基于 datetime 查找, 消除缺口偏移
            lag1 = _lookup_lag(ft, 24)
            lag7 = _lookup_lag(ft, 168)
            row["value_lag_1d"] = lag1
            row["value_lag_7d"] = lag7

            # rolling mean: 最近 96 步均值 (history 尾部)
            recent_96 = hist_vals[-96:] if n_hist >= 96 else hist_vals
            row["value_rolling_mean_24h"] = float(np.mean(recent_96))

            # diff 特征
            row["value_diff_1d"] = float(last_value - lag1) if last_value and lag1 else 0.0
            row["value_diff_7d"] = float(last_value - lag7) if last_value and lag7 else 0.0

            # 方向特征
            row["value_sign"] = float(np.sign(last_value)) if n_hist > 0 else 0.0
            sign_lag1 = np.sign(lag1) if lag1 else 0.0
            row["value_sign_lag_1d"] = float(sign_lag1)
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
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(_tz.utc).replace(tzinfo=None)
            weather_start = base_dt.strftime("%Y-%m-%d")
            forecast_end = (base_dt + timedelta(days=8)).strftime("%Y-%m-%d")
            if base_dt < now - timedelta(days=1):
                # 历史模式：end_date 不能超过昨天（历史 API 有延迟）
                hist_end = min(base_dt + timedelta(days=8), now - timedelta(days=1))
                forecast_end = hist_end.strftime("%Y-%m-%d")
                weather = self.fetcher.fetch_weather(
                    province, weather_start, forecast_end, mode="historical")
            else:
                try:
                    weather = self.fetcher.fetch_weather(
                        province, weather_start, forecast_end, mode="forecast")
                except Exception:
                    weather = self.fetcher.fetch_weather(
                        province, weather_start, forecast_end, mode="historical")
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

        # ── 补齐高级滚动统计 (与 features.py training 对齐: rolling window 标量) ──
        hist_vals = history["value"].values
        n_hist = len(hist_vals)
        if n_hist >= 96:
            recent_96_vals = hist_vals[-96:]
            future_df["value_rolling_std_24h"] = float(np.std(recent_96_vals))
            future_df["value_rolling_max_24h"] = float(np.max(recent_96_vals))
            future_df["value_rolling_min_24h"] = float(np.min(recent_96_vals))
            future_df["value_range_24h"] = float(np.max(recent_96_vals) - np.min(recent_96_vals))
        else:
            future_df["value_rolling_std_24h"] = float(np.std(hist_vals))
            future_df["value_rolling_max_24h"] = float(np.max(hist_vals))
            future_df["value_rolling_min_24h"] = float(np.min(hist_vals))
            future_df["value_range_24h"] = float(np.max(hist_vals) - np.min(hist_vals))

        # ── 补齐高级天气衍生特征 (与 WeatherFeatureEngineer 对齐) ──
        temp = future_df.get("temperature", pd.Series(20, index=future_df.index))
        hum = future_df.get("humidity", pd.Series(50, index=future_df.index))
        wind = future_df.get("wind_speed", pd.Series(3, index=future_df.index))
        solar = future_df.get("solar_radiation", pd.Series(0, index=future_df.index))
        press = future_df.get("pressure", pd.Series(1013, index=future_df.index))

        future_df["CDD"] = np.maximum(temp.values - 26, 0)
        future_df["HDD"] = np.maximum(18 - temp.values, 0)
        future_df["THI"] = 0.8 * temp.values + 0.2 * hum.values * temp.values / 100.0
        future_df["wind_power_potential"] = 0.5 * 1.225 * np.maximum(wind.values, 0) ** 3
        future_df["solar_potential"] = solar.values / 1000.0
        future_df["solar_efficiency"] = np.clip(1.0 - 0.005 * (temp.values - 25), 0.5, 1.0)

        # 天气 lag/diff (基于 datetime 查找, 消除缺口偏移)
        if "solar_radiation" in history.columns:
            hist_solar = history["solar_radiation"].values
            for i in range(len(future_df)):
                ft = future_df.loc[future_df.index[i], "dt"]
                pos_1d = _find_pos(ft, 24)
                future_df.loc[future_df.index[i], "solar_radiation_lag_1d"] = (
                    float(hist_solar[pos_1d]) if pos_1d is not None else float(solar.iloc[i]))
                future_df.loc[future_df.index[i], "solar_radiation_diff_1d"] = (
                    float(solar.iloc[i]) - future_df.loc[future_df.index[i], "solar_radiation_lag_1d"])

        if "wind_speed" in history.columns:
            hist_wind = history["wind_speed"].values
            for i in range(len(future_df)):
                ft = future_df.loc[future_df.index[i], "dt"]
                pos_1d = _find_pos(ft, 24)
                future_df.loc[future_df.index[i], "wind_speed_lag_1d"] = (
                    float(hist_wind[pos_1d]) if pos_1d is not None else float(wind.iloc[i]))
                future_df.loc[future_df.index[i], "wind_speed_diff_1d"] = (
                    float(wind.iloc[i]) - future_df.loc[future_df.index[i], "wind_speed_lag_1d"])

        if "temperature" in history.columns:
            hist_temp = history["temperature"].values
            for i in range(len(future_df)):
                ft = future_df.loc[future_df.index[i], "dt"]
                pos_1d = _find_pos(ft, 24)
                pos_1h = _find_pos(ft, 1)
                pos_6h = _find_pos(ft, 6)
                future_df.loc[future_df.index[i], "temperature_lag_1d"] = (
                    float(hist_temp[pos_1d]) if pos_1d is not None else float(temp.iloc[i]))
                future_df.loc[future_df.index[i], "temperature_diff_1d"] = (
                    float(temp.iloc[i]) - future_df.loc[future_df.index[i], "temperature_lag_1d"])
                future_df.loc[future_df.index[i], "temp_change_1h"] = (
                    float(temp.iloc[i]) - float(hist_temp[pos_1h])
                    if pos_1h is not None else 0.0)
                future_df.loc[future_df.index[i], "temp_change_6h"] = (
                    float(temp.iloc[i]) - float(hist_temp[pos_6h])
                    if pos_6h is not None else 0.0)

        # 连续高温天数 (简单近似: 最近 history 中的连续高温)
        if "temperature" in history.columns:
            hist_temp_vals = history["temperature"].values[-96:]
            hot_streak = 0
            for t_val in reversed(hist_temp_vals):
                if t_val > 30:
                    hot_streak += 1
                else:
                    break
            future_df["consecutive_hot_days"] = float(hot_streak // 96)
        else:
            future_df["consecutive_hot_days"] = 0.0

        # ── 补齐 working_day_type (从节假日模块) ──
        if "is_work_weekend" in future_df.columns:
            future_df["working_day_type"] = future_df["is_work_weekend"].astype(int) * 2 + (~future_df["is_weekend"]).astype(int)
        else:
            future_df["working_day_type"] = (~future_df["is_weekend"]).astype(int)

        # ── 补齐交叉类型特征 (从 store 加载其他类型最近数据) ──
        cross_types = [
            "出力_光伏_实际", "出力_总_实际", "出力_水电含抽蓄_实际",
            "出力_联络线_实际", "出力_非市场_实际", "出力_风电_实际", "负荷_系统_实际",
        ]
        for ct in cross_types:
            try:
                ct_data = self.store.load_features(
                    province, ct,
                    (base_dt - timedelta(days=14)).strftime("%Y-%m-%d"),
                    (base_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                if ct_data is not None and not ct_data.empty and "value" in ct_data.columns:
                    ct_vals = ct_data["value"].values
                    ct_dt_index = pd.Series(np.arange(len(ct_vals)), index=ct_data["dt"].values)
                    ct_mean_96 = float(np.mean(ct_vals[-96:]) if len(ct_vals) >= 96 else np.mean(ct_vals))
                    for i in range(len(future_df)):
                        ft = future_df.loc[future_df.index[i], "dt"]

                        def _ct_pos(offset_hours):
                            """在交叉类型数据中查找 target_dt - offset_hours 的位置."""
                            lag_dt = ft - timedelta(hours=offset_hours)
                            if lag_dt in ct_dt_index.index:
                                return ct_dt_index[lag_dt]
                            for dm in [0, 15, -15, 30, -30, 45, -45, 60, -60]:
                                c = lag_dt + timedelta(minutes=dm)
                                if c in ct_dt_index.index:
                                    return ct_dt_index[c]
                            return None

                        v_pos = _ct_pos(0)
                        future_df.loc[future_df.index[i], f"{ct}_value"] = (
                            float(ct_vals[v_pos]) if v_pos is not None else ct_mean_96)

                        l1_pos = _ct_pos(24)
                        future_df.loc[future_df.index[i], f"{ct}_lag_1d"] = (
                            float(ct_vals[l1_pos]) if l1_pos is not None else ct_mean_96)

                        l7_pos = _ct_pos(168)
                        future_df.loc[future_df.index[i], f"{ct}_lag_7d"] = (
                            float(ct_vals[l7_pos]) if l7_pos is not None else ct_mean_96)
            except Exception:
                # 如果某类型数据不可用，填0
                for suffix in ["_value", "_lag_1d", "_lag_7d"]:
                    col_name = f"{ct}{suffix}"
                    if col_name not in future_df.columns:
                        future_df[col_name] = 0.0

        # ── 补齐价格衍生特征 ──
        # 用 value_lag_1d 作为当前 value 的近似（因为未来值未知）
        proxy_value = future_df["value_lag_1d"].values

        # price_per_load / price_x_load
        load_col = "负荷_系统_实际_value"
        if load_col in future_df.columns:
            load_vals = future_df[load_col].replace(0, np.nan).values
            future_df["price_per_load"] = proxy_value / np.where(np.isnan(load_vals), 1.0, np.maximum(load_vals, 1.0))
            future_df["price_x_load"] = proxy_value * np.nan_to_num(load_vals, nan=0)
        else:
            # fallback: 从 history 推断
            if load_col in future_df.columns:
                load_vals = future_df[load_col].fillna(0).values
            else:
                # 从 history 的 value_rolling_mean 近似
                load_vals = np.full(len(future_df), float(np.mean(hist_vals[-96:]) if n_hist >= 96 else last_value))
            future_df["price_per_load"] = proxy_value / np.maximum(np.abs(load_vals), 1.0)
            future_df["price_x_load"] = proxy_value * load_vals

        # renewable_share / price_x_re_share
        wind_col = "出力_风电_实际_value"
        solar_col = "出力_光伏_实际_value"
        total_out_col = "出力_总_实际_value"
        if all(c in future_df.columns for c in [wind_col, solar_col, total_out_col]):
            re_total = future_df[wind_col].fillna(0) + future_df[solar_col].fillna(0)
            total_out = future_df[total_out_col].replace(0, np.nan).fillna(1)
            future_df["renewable_share"] = (re_total / total_out).clip(0, 1)
            future_df["price_x_re_share"] = proxy_value * (1 - future_df["renewable_share"])
            future_df["renewable_penetration"] = future_df["renewable_share"]
        else:
            future_df["renewable_share"] = 0.0
            future_df["price_x_re_share"] = 0.0
            future_df["renewable_penetration"] = 0.0

        # supply_demand_ratio / supply_surplus
        if total_out_col in future_df.columns and load_col in future_df.columns:
            supply = future_df[total_out_col].fillna(0)
            demand = future_df[load_col].replace(0, np.nan).fillna(1)
            future_df["supply_demand_ratio"] = supply / demand
            future_df["supply_surplus"] = supply - demand.fillna(0)
        else:
            future_df["supply_demand_ratio"] = 0.0
            future_df["supply_surplus"] = 0.0

        # price momentum (用 proxy_value 的 diff)
        future_df["price_momentum_1h"] = 0.0
        future_df["price_momentum_6h"] = 0.0
        future_df["price_momentum_24h"] = 0.0
        for i in range(len(future_df)):
            if i >= 4:
                future_df.loc[future_df.index[i], "price_momentum_1h"] = float(proxy_value[i] - proxy_value[i-4])
            if i >= 24:
                future_df.loc[future_df.index[i], "price_momentum_6h"] = float(proxy_value[i] - proxy_value[i-24])
            if i >= 96:
                future_df.loc[future_df.index[i], "price_momentum_24h"] = float(proxy_value[i] - proxy_value[i-96])

        # price_vol (用 proxy_value 的滚动 std/mean)
        if n_hist >= 96:
            hist_vol_24h = float(np.std(hist_vals[-96:]) / (np.mean(np.abs(hist_vals[-96:])) + 0.01))
            future_df["price_vol_24h"] = hist_vol_24h
        else:
            future_df["price_vol_24h"] = 0.0
        if n_hist >= 672:
            hist_vol_7d = float(np.std(hist_vals[-672:]) / (np.mean(np.abs(hist_vals[-672:])) + 0.01))
            future_df["price_vol_7d"] = hist_vol_7d
        else:
            future_df["price_vol_7d"] = future_df["price_vol_24h"].iloc[0] if len(future_df) > 0 else 0.0

        # price_position (在日内波动中的相对位置, 用 proxy 近似)
        proxy_min = np.min(proxy_value)
        proxy_max = np.max(proxy_value)
        proxy_range = max(proxy_max - proxy_min, 1.0)
        future_df["price_position"] = (proxy_value - proxy_min) / proxy_range

        # peak_off_peak_spread (从 history 计算)
        peak_mask = future_df["peak_valley"] == 2
        off_mask = future_df["peak_valley"] == 0
        peak_vals = proxy_value[peak_mask.values]
        off_vals = proxy_value[off_mask.values]
        if len(peak_vals) > 0 and len(off_vals) > 0:
            future_df["peak_off_peak_spread"] = float(np.mean(peak_vals) - np.mean(off_vals))
        else:
            future_df["peak_off_peak_spread"] = 0.0

        # ── 补齐误差修正特征 (未来无预测, 全部初始化为0) ──
        error_cols = [
            "pred_error", "pred_error_lag_1d", "pred_error_lag_7d",
            "pred_error_bias_24h", "pred_error_std_24h", "pred_error_trend",
            "interval_coverage", "coverage_rate_24h",
            "pred_error_hour_bias", "pred_error_weekend", "pred_error_holiday",
            "pred_error_x_temp", "pred_error_x_wind",
            "pred_error_autocorr", "pred_error_regime",
        ]
        for ec in error_cols:
            if ec not in future_df.columns:
                future_df[ec] = 0.0

        # ── NaN 填充 ──
        for col in future_df.columns:
            if col not in ("dt", "province", "type") and future_df[col].dtype == np.float64:
                future_df[col] = future_df[col].fillna(
                    history[col].mean() if col in history.columns else 0)
        return future_df

    def _predict_with_model(self, model, features_df: pd.DataFrame,
                            feature_names: List[str], province: str,
                            target_type: str,
                            history: pd.DataFrame = None) -> pd.DataFrame:
        missing = [fn for fn in feature_names if fn not in features_df.columns]
        if missing:
            missing_df = pd.DataFrame(0.0, index=features_df.index, columns=missing)
            features_df = pd.concat([features_df, missing_df], axis=1)
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

                # 分位数模型已有独立的 P10/P90, 更信任模型; 单模型回退则更依赖历史残差
                is_quantile = ensemble.get("model_version", [""]).iloc[0] == "quantile_v1"
                mw = 0.7 if is_quantile else 0.3
                cw = 1.0 - mw
                final_lo = mw * model_lo + cw * cal_lo
                final_hi = mw * model_hi + cw * cal_hi

                # 时序衰减扩容: 越远的预测越不确定
                horizon_factor = 1.0 + 0.4 * np.arange(n) / max(n, 1)
                final_lo *= horizon_factor
                final_hi *= horizon_factor

                # 硬下限: 不低于 P50 的 3% (避免区间过窄失去参考价值)
                floor = np.abs(p50_vals) * 0.03
                final_lo = np.maximum(final_lo, floor)
                final_hi = np.maximum(final_hi, floor)

                # 上限: 不超过 P50 的 3x
                cap = np.abs(p50_vals) * 3.0
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
