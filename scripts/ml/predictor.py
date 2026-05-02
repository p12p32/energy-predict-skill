"""predictor.py — 预测执行器: 未来特征外推 + 分位数预测 + 趋势集成 + ECM

修复:
- 未来特征用日内模式外推 lag，气象获取失败用同小时均值
- ECM 修正传入真实残差
- 融合权重自适应
- 预测前集成跨类型特征
- 超时保护
"""
import logging
import time
import signal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

from scripts.ml.trainer import Trainer
from scripts.ml.trend import TrendModel
from scripts.ml.error_correction import ErrorCorrectionModel
from scripts.data.features import FeatureStore, FeatureEngineer
from scripts.data.fetcher import DataFetcher
from scripts.core.config import (load_config, validate_province_and_type,
                                  parse_type, get_prediction_config, get_cross_type_rules)


class _TimeoutException(Exception):
    pass


class Predictor:
    def __init__(self, trainer: Trainer = None,
                 store: FeatureStore = None):
        self.trainer = trainer or Trainer()
        self.store = store or FeatureStore()
        self.fetcher = DataFetcher()
        self.ecm = ErrorCorrectionModel()

    @staticmethod
    def _allow_negative(target_type: str) -> bool:
        ti = parse_type(target_type)
        return ti.base == "电价"

    def _find_latest_date(self, province: str, target_type: str):
        import glob, os
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            ".energy_data", "features",
        )
        pattern = os.path.join(base_dir, f"{province}_{target_type}_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
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
        pred_cfg = get_prediction_config()
        timeout = pred_cfg.get("timeout_seconds", 60)

        start_time = time.time()

        def _check_timeout():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise _TimeoutException(f"预测超时 ({timeout}s)")

        horizon_steps = horizon_hours * 4
        lookback_days = 14

        if reference_date:
            end_date = pd.to_datetime(reference_date)
        else:
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

        _check_timeout()

        # 用实际数据最新日期作为预测起点 (而非文件日期)
        if "dt" in history.columns and "value" in history.columns:
            valid = history.dropna(subset=["value"])
            if not valid.empty:
                actual_end = valid["dt"].max()
                start_date = actual_end - timedelta(days=lookback_days)
                history = history[history["dt"] >= start_date]
                end_date = actual_end

        # 跨类型特征集成
        ti = parse_type(target_type)
        if pred_cfg.get("auto_cross_type", True):
            try:
                history = self._add_cross_type_features(
                    history, province, target_type
                )
            except Exception as e:
                logger.warning("跨类型特征获取失败: %s", e)

        future_features = self._build_future_features(
            history, province, target_type, horizon_steps, end_date
        )

        _check_timeout()

        # 尝试分位数模型
        try:
            quantile_models = self.trainer.load_quantile_models(province, target_type)
            predictions = self._predict_quantile(
                quantile_models, future_features,
                province, target_type
            )
        except (FileNotFoundError, KeyError):
            model, feature_names, _ = self.trainer.load_model(province, target_type)
            predictions = self._predict_with_model(
                model, future_features, feature_names,
                province=province, target_type=target_type, history=history,
            )

        _check_timeout()

        trend_pred = self._predict_trend(history, horizon_steps)
        ensemble = self._ensemble(predictions, trend_pred, history, target_type)

        # ECM 残差修正 (核心修复: 传入真实残差)
        if len(history) > 200 and "value" in history.columns:
            try:
                self._apply_ecm_correction(
                    ensemble, history, province, target_type, horizon_steps
                )
            except Exception as e:
                logger.warning("ECM 残差修正失败 (%s/%s): %s",
                               province, target_type, e)

        # 自适应区间校准
        ensemble = self._calibrate_intervals(ensemble, history, province, target_type)

        # 光伏夜间置零: 20:00~05:45 基本无出力
        if "光伏" in target_type:
            night_mask = ensemble["dt"].dt.hour.isin(list(range(20, 24)) + list(range(0, 6)))
            if night_mask.any():
                ensemble.loc[night_mask, "p10"] = 0
                ensemble.loc[night_mask, "p50"] = 0
                ensemble.loc[night_mask, "p90"] = 0

        self.store.insert_predictions(ensemble)
        return ensemble

    def _add_cross_type_features(self, history: pd.DataFrame,
                                  province: str, target_type: str) -> pd.DataFrame:
        """加载同省份的其他类型特征，构建跨类型交互."""
        ti = parse_type(target_type)
        cfg = load_config()
        rules = cfg.get("cross_type_rules", {})

        # 找同省份的 sibling types
        sibling_key = None
        for key, siblings in rules.items():
            if key.endswith("_siblings"):
                for s in siblings:
                    if s == target_type or s == f"{ti.base}_{ti.sub}" if ti.sub else s == ti.base:
                        sibling_key = key
                        break

        if not sibling_key:
            return history

        siblings = rules[sibling_key]
        sibling_types = [s for s in siblings if s != target_type]

        if not sibling_types:
            return history

        end_dt = history["dt"].max()
        start_dt = end_dt - timedelta(days=14)

        cross_features = pd.DataFrame(index=history.index)

        for s_type in sibling_types:
            try:
                s_data = self.store.load_features(
                    province, s_type,
                    start_dt.strftime("%Y-%m-%d"),
                    (end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                if s_data.empty or "value" not in s_data.columns:
                    continue

                s_data = s_data[["dt", "value"]].rename(
                    columns={"value": f"cross_{s_type}_value"}
                )
                # 对齐到 15 分钟粒度
                s_data["dt"] = s_data["dt"].dt.floor("15min")
                cross_features = cross_features.merge(
                    s_data, on="dt", how="left"
                )
            except Exception:
                continue

        if cross_features.empty:
            return history

        # forward fill 有限数量 (最多 4 步 = 1 小时)
        for col in cross_features.columns:
            if col.startswith("cross_"):
                cross_features[col] = cross_features[col].ffill(limit=4).fillna(0)

        # 合并回 history
        result = history.copy()
        for col in cross_features.columns:
            if col not in result.columns:
                result[col] = cross_features[col].values

        return result

    def _build_future_features(self, history: pd.DataFrame,
                                province: str, target_type: str,
                                horizon_steps: int,
                                base_dt: datetime) -> pd.DataFrame:
        """构建未来特征 (用日内模式外推，而非 last_value)."""
        last_row = history.iloc[-1].copy()
        last_price = last_row.get("price", 0)
        ti = parse_type(target_type)

        future_times = pd.date_range(
            start=base_dt + timedelta(minutes=15),
            periods=horizon_steps,
            freq="15min",
        )

        # 计算日内模式 (最近 N 天)
        recent = history.tail(min(len(history), 7 * 96))
        hourly_pattern = np.zeros(96)
        hourly_std = np.zeros(96)
        if len(recent) >= 96:
            days = len(recent) // 96
            reshaped = recent["value"].iloc[:days * 96].values.reshape(days, 96)
            hourly_pattern = np.nanmean(reshaped, axis=0)
            hourly_std = np.nanstd(reshaped, axis=0)

        # 计算趋势（最近 7 天的均值变化率）
        trend_rate = 0.0
        if len(history) >= 2 * 96:
            recent_7d = history["value"].tail(7 * 96)
            older_7d = history["value"].iloc[-14 * 96:-7 * 96]
            if len(older_7d) > 0 and len(recent_7d) > 0:
                mean_new = recent_7d.mean()
                mean_old = older_7d.mean()
                if abs(mean_old) > 1e-8:
                    trend_rate = (mean_new - mean_old) / abs(mean_old) / 7  # 每天变化率

        # 7 天模式
        weekly_pattern = np.zeros(672)
        if len(recent) >= 672:
            weeks = len(recent) // 672
            reshaped = recent["value"].iloc[:weeks * 672].values.reshape(weeks, 672)
            weekly_pattern = np.nanmean(reshaped, axis=0)

        # 气象数据的同小时统计 (用于缺失填充)
        weather_hourly = {}
        for col in ["temperature", "humidity", "wind_speed", "wind_direction",
                     "solar_radiation", "precipitation", "pressure"]:
            if col in history.columns:
                grp = history.groupby("hour")[col].agg(["mean", "std"])
                weather_hourly[col] = grp

        # 滚动统计
        rolling_mean_24h = history["value"].tail(96).mean() if len(history) >= 96 else history["value"].mean()
        rolling_std_24h = history["value"].tail(96).std() if len(history) >= 96 else 0

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
                "quality_flag": 0,
            }

            # 日内索引
            hour_idx = ft.hour * 4 + ft.minute // 15
            day_progress = (ft - base_dt).total_seconds() / 86400

            # lag 特征: 日内模式 + 趋势
            if i < 96 and len(history) > 96:
                # 1天前: 从历史中精确取值
                hist_idx = len(history) - (96 - i)
                if 0 <= hist_idx < len(history):
                    row["value_lag_1d"] = history.iloc[hist_idx]["value"]
            elif len(hourly_pattern) > 0:
                # 超出窗口: 用日内模式 + 趋势
                base = hourly_pattern[hour_idx]
                adjusted = base * (1 + trend_rate * day_progress)
                row["value_lag_1d"] = max(adjusted, 0)
            else:
                row["value_lag_1d"] = rolling_mean_24h

            # 7天前 lag
            if i < 672 and len(history) > 672:
                hist_idx_7d = len(history) - (672 - i)
                if 0 <= hist_idx_7d < len(history):
                    row["value_lag_7d"] = history.iloc[hist_idx_7d]["value"]
            elif len(weekly_pattern) > 0:
                week_idx = (hour_idx + ft.dayofweek * 96) % 672
                row["value_lag_7d"] = weekly_pattern[week_idx]
            else:
                row["value_lag_7d"] = row["value_lag_1d"]

            # 滚动均值
            row["value_rolling_mean_24h"] = rolling_mean_24h * (1 + trend_rate * day_progress)

            # diff 特征: 趋势外推
            if len(history) > 96:
                last_1d = history["value"].iloc[-1]
                prev_1d = history["value"].iloc[-96] if len(history) >= 96 else last_1d
                daily_diff = last_1d - prev_1d
                row["value_diff_1d"] = daily_diff * (1 + trend_rate * day_progress * 0.5)
            else:
                row["value_diff_1d"] = 0.0

            if len(history) > 672:
                last_7d = history["value"].iloc[-1]
                prev_7d = history["value"].iloc[-672]
                weekly_diff = last_7d - prev_7d
                row["value_diff_7d"] = weekly_diff * (1 + trend_rate * day_progress * 0.3)
            else:
                row["value_diff_7d"] = 0.0

            rows.append(row)

        future_df = pd.DataFrame(rows)

        # 节假日 + 周期编码
        from scripts.data.holidays import add_holiday_features, add_cyclical_features
        future_df = add_holiday_features(future_df)
        future_df = add_cyclical_features(future_df)

        # 交互特征
        future_df["peak_valley"] = future_df["hour"].apply(
            lambda h: 2 if h in [8, 9, 10, 11, 17, 18, 19, 20] else
                      1 if h in [12, 13, 14, 21, 22] else 0
        )
        future_df["weekend_hour"] = future_df["is_weekend"].astype(int) * future_df["hour"]
        future_df["dow_hour"] = future_df["day_of_week"] * 24 + future_df["hour"]
        future_df["weekend_x_lag7d"] = future_df["is_weekend"].astype(int) * future_df["value_lag_7d"]
        future_df["hour_x_lag1d"] = future_df["hour"] * future_df["value_lag_1d"]

        # 气象特征: 用 Open-Meteo 获取预报
        weather_filled = False
        try:
            forecast_end_dt = base_dt + timedelta(days=8)
            weather = self.fetcher.fetch_weather(
                province,
                base_dt.strftime("%Y-%m-%d"),
                forecast_end_dt.strftime("%Y-%m-%d"),
            )
            if not weather.empty:
                weather = weather.copy()
                weather["dt"] = weather["dt"].dt.floor("15min")
                future_df["dt"] = future_df["dt"].dt.floor("15min")
                weather_cols = ["dt", "province", "temperature", "humidity",
                                "wind_speed", "wind_direction", "solar_radiation",
                                "precipitation", "pressure",
                                "cloud_cover", "wind_speed_10m", "wind_direction_10m",
                                "wind_gusts", "shortwave_radiation", "dni", "dhi"]
                w_avail = [c for c in weather_cols if c in weather.columns]
                w_subset = weather[w_avail].drop_duplicates(subset=["dt", "province"], keep="last")
                future_df = future_df.merge(
                    w_subset, on=["dt", "province"], how="left", suffixes=("", "_w")
                )
                # 用 _w 列覆盖 (预报数据优先)
                for c in w_avail:
                    if c + "_w" in future_df.columns:
                        future_df[c] = future_df[c + "_w"].fillna(future_df.get(c))
                        future_df.drop(columns=[c + "_w"], inplace=True, errors="ignore")
                future_df.drop(columns=["dt"], inplace=True, errors="ignore")
                weather_filled = True
                logger.info("气象预报获取成功: %s (%d 行)", province, len(weather))
        except Exception as e:
            logger.warning("天气预报获取失败 %s: %s, 降级到统计均值", province, e)

        # 降级: 用历史的同小时均值填充 (比全局均值更准)
        if not weather_filled or future_df[["temperature", "humidity", "wind_speed"]].isna().any().any():
            for col in ["temperature", "humidity", "wind_speed", "wind_direction",
                         "solar_radiation", "precipitation", "pressure"]:
                if col not in future_df.columns or future_df[col].isna().any():
                    if col in weather_hourly:
                        grp = weather_hourly[col]
                        for h in range(24):
                            mask = future_df["hour"] == h
                            if mask.any():
                                mean_val = grp.loc[h, "mean"] if h in grp.index else 0
                                future_df.loc[mask, col] = future_df.loc[mask, col].fillna(mean_val)
                    if col not in future_df.columns:
                        future_df[col] = history[col].mean() if col in history.columns else 0
                    else:
                        future_df[col] = future_df[col].fillna(
                            history[col].mean() if col in history.columns else 0
                        )

        # 深度天气特征
        from scripts.data.weather_features import WeatherFeatureEngineer
        wfe = WeatherFeatureEngineer()
        future_df = wfe.transform(future_df)

        # 波动率
        future_df["value_rolling_std_24h"] = rolling_std_24h
        future_df["value_range_24h"] = rolling_std_24h * 4
        future_df["value_rolling_max_24h"] = rolling_mean_24h + 2 * rolling_std_24h
        future_df["value_rolling_min_24h"] = max(rolling_mean_24h - 2 * rolling_std_24h, 0)

        # 天气交互
        future_df["temp_x_hour"] = future_df.get("temperature", 0) * future_df["hour"]
        future_df["wind_x_season"] = future_df.get("wind_speed", 0) * future_df["season"]

        # 多尺度 lag 外推 (新增的 2d~28d lag)
        extra_lags = {
            "value_lag_2d": 192, "value_lag_3d": 288, "value_lag_4d": 384,
            "value_lag_5d": 480, "value_lag_6d": 576,
            "value_lag_14d": 1344, "value_lag_21d": 2016, "value_lag_28d": 2688,
        }
        for lag_name, lag_steps in extra_lags.items():
            col = future_df[lag_name] = np.nan
            for i in range(min(horizon_steps, lag_steps)):
                hist_idx = len(history) - (lag_steps - i)
                if 0 <= hist_idx < len(history):
                    col.iloc[i] = history.iloc[hist_idx]["value"]
            # 超出窗口的用日内模式 + 趋势
            nan_mask = col.isna()
            if nan_mask.any():
                hour_indices = future_df.loc[nan_mask, "hour"] * 4
                for idx_pos in future_df.loc[nan_mask].index:
                    hi = int(hour_indices.get(idx_pos, 0)) % 96
                    dp = (future_df.loc[idx_pos, "dt"] - base_dt).total_seconds() / 86400
                    base = hourly_pattern[hi] if len(hourly_pattern) > 0 else rolling_mean_24h
                    adjusted = base * (1 + trend_rate * dp)
                    col.iloc[idx_pos] = max(adjusted, 0)

        # 多尺度滚动统计外推
        for feat_name, (window, func) in [
            ("value_rolling_mean_6h", (24, "mean")), ("value_rolling_mean_12h", (48, "mean")),
            ("value_rolling_mean_48h", (192, "mean")), ("value_rolling_mean_7d", (672, "mean")),
            ("value_rolling_std_7d", (672, "std")),
        ]:
            w, f = window, func
            if len(history) >= w:
                val = history["value"].tail(w).mean() if f == "mean" else history["value"].tail(w).std()
                dp_arr = (future_df["dt"] - base_dt).dt.total_seconds() / 86400
                future_df[feat_name] = val * (1 + trend_rate * dp_arr)

        # 风电专项特征
        if "wind_speed_10m" in future_df.columns:
            ws = future_df["wind_speed_10m"]
            if "wind_speed_100m" in future_df.columns:
                ws100 = future_df["wind_speed_100m"]
                future_df["wind_shear"] = np.where(
                    ws > 0.1, np.log(np.maximum(ws100, 0.01) / np.maximum(ws, 0.01)) / np.log(10), 0.0)
            if "wind_gusts" in future_df.columns:
                future_df["gust_factor"] = np.where(ws > 0.5, future_df["wind_gusts"] / ws, 0.0)
            future_df["wind_power"] = ws ** 3
            future_df["wind_effective"] = ((ws >= 3) & (ws <= 25)).astype(float) * ws
            if "wind_direction_10m" in future_df.columns:
                wd = future_df["wind_direction_10m"]
                future_df["wind_dir_var"] = 1.0 - np.sqrt(np.sin(np.radians(wd))**2 + np.cos(np.radians(wd))**2)

        # 光伏专项特征
        if "shortwave_radiation" in future_df.columns:
            ghi = future_df["shortwave_radiation"]
            if "temperature" in future_df.columns:
                future_df["pv_temp_coeff"] = ghi * (1 - 0.004 * np.maximum(future_df["temperature"] - 25, 0))
            if "dni" in future_df.columns and "dhi" in future_df.columns:
                total = future_df["dni"] + future_df["dhi"]
                future_df["clear_sky_index"] = np.where(total > 1, future_df["dni"] / total, 0)
            if "cloud_cover" in future_df.columns:
                future_df["cloud_attenuation"] = ghi * (1 - future_df["cloud_cover"] / 100.0)
            future_df["solar_bin"] = pd.cut(ghi, bins=[-1, 0, 100, 400, 800, 2000],
                                              labels=[0, 1, 2, 3, 4]).astype(float)
            future_df["solar_active"] = (future_df["hour"].isin(range(6, 20)) & (ghi > 10)).astype(float)
            future_df["solar_elevation"] = np.maximum(np.cos(np.abs(future_df["hour"] - 12) / 12 * np.pi), 0) ** 2
            if "dhi" in future_df.columns:
                future_df["dhi_dominance"] = np.where(ghi > 1, future_df["dhi"] / ghi, 0)
            if "cloud_cover" in future_df.columns:
                cc = future_df["cloud_cover"] / 100.0
                future_df["cloud_nonlinear"] = ghi * (1 - cc ** 1.5)
            if "hour_cos" in future_df.columns:
                future_df["ghi_x_cos"] = ghi * np.maximum(future_df["hour_cos"], 0)
            future_df["ghi_sqrt"] = np.sqrt(np.maximum(ghi, 0))

        # 周/月周期差异
        if "value_lag_7d" in future_df.columns and "value_lag_28d" in future_df.columns:
            future_df["weekly_vs_monthly"] = future_df["value_lag_7d"] - future_df["value_lag_28d"]

        # pred_error 列初始化
        for col in ["pred_error", "pred_error_lag_1d", "pred_error_lag_7d",
                     "pred_error_bias_24h", "pred_error_std_24h", "pred_error_trend",
                     "interval_coverage", "coverage_rate_24h",
                     "pred_error_hour_bias", "pred_error_weekend", "pred_error_holiday",
                     "pred_error_x_temp", "pred_error_x_wind",
                     "pred_error_autocorr", "pred_error_regime"]:
            if col not in future_df.columns:
                future_df[col] = 0.0

        return future_df

    def _predict_quantile(self, models: Dict, features_df: pd.DataFrame,
                           province: str, target_type: str) -> pd.DataFrame:
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

        allow_neg = self._allow_negative(target_type)
        raw_p10 = p10_model.predict(X) if p10_model else np.zeros(len(X))
        raw_p50 = p50_model.predict(X)
        raw_p90 = p90_model.predict(X) if p90_model else np.zeros(len(X))

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

        # 自适应权重: 根据历史一致性
        lgb_weight = self._compute_adaptive_weight(history, n)

        lgb_p50 = lgb_result["p50"].values
        ensemble_p50 = lgb_weight * lgb_p50 + (1 - lgb_weight) * trend_preds

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

    def _compute_adaptive_weight(self, history: pd.DataFrame, n: int) -> np.ndarray:
        """计算自适应融合权重.

        如果 LightGBM 历史表现好 → 权重高
        如果趋势模型表现好 → 权重低
        """
        if "value" not in history.columns or len(history) < 96 * 2:
            return np.array([max(0.3, 0.75 - 0.005 * i) for i in range(n)])

        # 最近 2 天的残差
        recent = history.tail(2 * 96)
        values = recent["value"].dropna().values
        if len(values) < 96:
            return np.array([max(0.3, 0.75 - 0.005 * i) for i in range(n)])

        # 趋势模型拟合
        trend = TrendModel()
        trend.fit(values)
        trend_fit = trend.predict_with_daily_pattern(len(values), values)

        # LightGBM 拟合 (简单均值模式)
        pattern = np.zeros(96)
        days = len(values) // 96
        reshaped = values[:days * 96].reshape(days, 96)
        lgb_pattern = np.mean(reshaped, axis=0)
        lgb_fit = np.tile(lgb_pattern, days)[:len(values)]

        # 计算各自 MAPE
        def _safe_mape(actual, pred):
            mask = np.abs(actual) > 1e-8
            if mask.sum() < 10:
                return 0.5
            return float(np.mean(np.abs(actual[mask] - pred[mask]) / np.abs(actual[mask])))

        mape_lgb = _safe_mape(values, lgb_fit)
        mape_trend = _safe_mape(values, trend_fit)

        # 权重: MAPE 低的更高
        total = mape_lgb + mape_trend + 1e-8
        base_weight = mape_trend / total  # trend MAPE 高 → lgb 权重高

        # 限制在 [0.3, 0.9]
        base_weight = np.clip(base_weight, 0.3, 0.9)

        # 远期衰减
        weights = np.array([
            max(0.3, base_weight - 0.003 * i)
            for i in range(n)
        ])
        return weights

    def _apply_ecm_correction(self, ensemble: pd.DataFrame,
                               history: pd.DataFrame,
                               province: str, target_type: str,
                               horizon_steps: int):
        """ECM 残差修正 — 核心修复: 传入真实残差."""
        recent = history["value"].tail(500).values

        # 计算历史残差: actual - predicted
        try:
            models = self.trainer.load_quantile_models(province, target_type)
            p50_model, feature_names = models.get("p50", (None, []))
        except Exception:
            try:
                model, feature_names, _ = self.trainer.load_model(province, target_type)
                p50_model = model
            except Exception:
                return

        if p50_model is None or not feature_names:
            return

        recent_features = history.tail(500).copy()
        for fn in feature_names:
            if fn not in recent_features.columns:
                recent_features[fn] = 0.0

        try:
            hist_pred = p50_model.predict(recent_features[feature_names].values)
        except Exception:
            return

        actuals = recent[:len(hist_pred)]
        residuals = actuals - hist_pred

        # 确保残差非零 (核心修复)
        if np.all(np.abs(residuals) < 1e-15):
            logger.debug("ECM: 残差全零，跳过修正")
            return

        # 获取小时信息
        recent_hours = None
        future_hours = None
        if "hour" in recent_features.columns:
            recent_hours = recent_features["hour"].values[-len(residuals):]
            future_hours = ensemble["dt"].dt.hour.values

        self.ecm.fit(residuals, recent_hours)
        correction = self.ecm.predict(
            residuals, horizon_steps, recent_hours, future_hours
        )

        allow_neg = self._allow_negative(target_type)
        ensemble["p50"] = ensemble["p50"].values + correction
        if not allow_neg:
            ensemble["p50"] = np.maximum(ensemble["p50"], 0)

        half_fix = correction / 2
        ensemble["p10"] = ensemble["p10"].values + half_fix
        ensemble["p90"] = ensemble["p90"].values + half_fix
        if not allow_neg:
            ensemble["p10"] = np.maximum(ensemble["p10"], 0)
            ensemble["p90"] = np.maximum(ensemble["p90"], 0)

    def _calibrate_intervals(self, ensemble: pd.DataFrame,
                             history: pd.DataFrame,
                             province: str, target_type: str) -> pd.DataFrame:
        """自适应区间校准."""
        try:
            if "value" not in history.columns or len(history) < 96:
                return ensemble

            try:
                models = self.trainer.load_quantile_models(province, target_type)
                feature_names = list(models.values())[0][1] if models else []
            except Exception:
                return ensemble

            if not feature_names:
                return ensemble

            recent = history.tail(672).copy()
            for fn in feature_names:
                if fn not in recent.columns:
                    recent[fn] = 0.0

            X_hist = recent[feature_names].values
            actual = recent["value"].values

            p50_m, _ = models.get("p50", (None, None))
            if p50_m is None:
                return ensemble

            hist_p50 = p50_m.predict(X_hist)

            denom = np.maximum(np.abs(hist_p50), 1.0)
            rel_residuals = np.abs(actual - hist_p50) / denom

            target_quantile = 0.80
            sigma_rel = float(np.quantile(rel_residuals, target_quantile))
            sigma_rel = max(sigma_rel, 0.02)
            sigma_rel *= 1.3

            p50 = np.maximum(ensemble["p50"].values, 1.0)
            model_lo = ensemble["p50"].values - ensemble["p10"].values
            model_hi = ensemble["p90"].values - ensemble["p50"].values
            cal_lo = sigma_rel * p50
            cal_hi = sigma_rel * p50

            final_lo = 0.4 * model_lo + 0.6 * cal_lo
            final_hi = 0.4 * model_hi + 0.6 * cal_hi
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

        # 限幅: P10 和 P90 不超过 P50 的合理倍数
        p50_vals = ensemble["p50"].values
        # 先对 P50 做合理性限幅 (基于历史数据的范围)
        if history is not None and "value" in history.columns:
            hist_max = history["value"].max()
            hist_min = history["value"].min()
            hist_range = hist_max - hist_min
            # P50 不应超出历史范围太多
            p50_upper = hist_max + hist_range * 0.5
            p50_lower = max(hist_min - hist_range * 0.5, 0)
            ensemble["p50"] = ensemble["p50"].clip(p50_lower, p50_upper)
            p50_vals = ensemble["p50"].values

        for col in ["p10", "p90"]:
            vals = ensemble[col].values
            max_dev = np.maximum(np.abs(p50_vals) * 0.5, 1.0)
            ensemble[col] = np.clip(vals, p50_vals - max_dev, p50_vals + max_dev)
            allow_neg_check = self._allow_negative(target_type)
            if not allow_neg_check:
                ensemble[col] = np.maximum(ensemble[col].values, 0)

        return ensemble

    def _predict_with_model(self, model, features_df: pd.DataFrame,
                            feature_names: list,
                            province: str = "",
                            target_type: str = "",
                            history: pd.DataFrame = None) -> pd.DataFrame:
        for fn in feature_names:
            if fn not in features_df.columns:
                features_df[fn] = 0.0
        X = features_df[feature_names].values
        predicted = model.predict(X)

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
            except Exception:
                pass

        p10 = predicted * (1 - 1.28 * residual_std)
        p90 = predicted * (1 + 1.28 * residual_std)

        allow_neg = self._allow_negative(target_type)
        return pd.DataFrame({
            "dt": features_df["dt"].values, "province": province, "type": target_type,
            "p50": predicted,
            "p10": p10 if allow_neg else np.maximum(p10, 0),
            "p90": p90 if allow_neg else np.maximum(p90, 0),
            "model_version": "v1",
        })
