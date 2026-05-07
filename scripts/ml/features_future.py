"""features_future.py — 未来特征构建: 从历史特征外推未来 N 步的特征矩阵."""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def build_future_features(history: pd.DataFrame,
                          province: str,
                          target_type: str,
                          horizon_steps: int,
                          base_dt: datetime,
                          store=None,
                          fetcher=None,
                          forecast_data: dict = None) -> pd.DataFrame:
    """从历史特征构建未来 horizon_steps 步的特征矩阵.

    参数:
        history: 历史特征 DataFrame, 含 dt/value/天气等列
        province: 省份名
        target_type: 类型 (e.g. "出力_风电_实际")
        horizon_steps: 预测步数
        base_dt: 预测起始日期 (最后已知时间点)
        store: FeatureStore 实例 (用于加载交叉类型数据)
        fetcher: DataFetcher 实例 (用于获取天气数据)

    返回:
        future_df: 未来 horizon_steps 行的特征 DataFrame
    """
    from scripts.core.config import parse_type

    if store is None:
        from scripts.data.features import FeatureStore
        store = FeatureStore()
    if fetcher is None:
        from scripts.data.fetcher import DataFetcher
        fetcher = DataFetcher()

    last_row = history.iloc[-1].copy()
    last_value = last_row.get("value", 0)
    last_price = last_row.get("price", 0)
    hist_vals = history["value"].values
    n_hist = len(hist_vals)

    # 构建 datetime → index 查找表
    hist_dt_index = pd.Series(np.arange(n_hist), index=history["dt"].values)

    def _find_pos(target_dt, offset_hours):
        lag_dt = target_dt - timedelta(hours=offset_hours)
        if lag_dt in hist_dt_index.index:
            return hist_dt_index[lag_dt]
        for delta_m in [0, 15, -15, 30, -30, 45, -45, 60, -60]:
            candidate = lag_dt + timedelta(minutes=delta_m)
            if candidate in hist_dt_index.index:
                return hist_dt_index[candidate]
        return None

    def _lookup_lag(target_dt, offset_hours):
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

        lag1 = _lookup_lag(ft, 24)
        lag7 = _lookup_lag(ft, 168)
        row["value_lag_1d"] = lag1
        row["value_lag_7d"] = lag7

        recent_96 = hist_vals[-96:] if n_hist >= 96 else hist_vals
        row["value_rolling_mean_24h"] = float(np.mean(recent_96))

        row["value_diff_1d"] = float(last_value - lag1) if last_value and lag1 else 0.0
        row["value_diff_7d"] = float(last_value - lag7) if last_value and lag7 else 0.0

        # 多尺度 lag
        row["value_lag_2d"] = _lookup_lag(ft, 48)
        row["value_lag_3d"] = _lookup_lag(ft, 72)
        row["value_lag_14d"] = _lookup_lag(ft, 336)

        # 多尺度 diff + 加速度
        lag2 = row["value_lag_2d"]
        lag3 = row["value_lag_3d"]
        row["value_diff_2d"] = float(last_value - lag2) if last_value and lag2 else 0.0
        row["value_diff_3d"] = float(last_value - lag3) if last_value and lag3 else 0.0
        row["value_accel_1d"] = float(row["value_diff_1d"] - (lag1 - lag2) if lag1 and lag2 else 0.0)

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

    # EMA 滑动均值
    if n_hist >= 672:
        alpha_1d = 0.30
        alpha_7d = 0.10
        alpha_30d = 0.03
        ema_1d = hist_vals[-1]
        ema_7d = hist_vals[-1]
        ema_30d = hist_vals[-1]
        for v in reversed(hist_vals):
            ema_1d = alpha_1d * v + (1 - alpha_1d) * ema_1d
            ema_7d = alpha_7d * v + (1 - alpha_7d) * ema_7d
            ema_30d = alpha_30d * v + (1 - alpha_30d) * ema_30d
        future_df["value_ema_1d"] = float(ema_1d)
        future_df["value_ema_7d"] = float(ema_7d)
        future_df["value_ema_30d"] = float(ema_30d)
        future_df["ema_cross_1d_7d"] = float(ema_1d - ema_7d) / (abs(ema_7d) + 1.0)
        future_df["ema_cross_7d_30d"] = float(ema_7d - ema_30d) / (abs(ema_30d) + 1.0)
    else:
        v_mean = float(np.mean(hist_vals)) if n_hist > 0 else 0.0
        for col in ["value_ema_1d", "value_ema_7d", "value_ema_30d"]:
            future_df[col] = v_mean
        future_df["ema_cross_1d_7d"] = 0.0
        future_df["ema_cross_7d_30d"] = 0.0

    # 白天/黑夜 + 时段细分
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

    # 气象列 fallback: 最近7天同时刻均值
    weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction",
                    "solar_radiation", "precipitation", "pressure"]
    for col in weather_cols:
        if col not in future_df.columns:
            if col in history.columns and len(history) >= 96:
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

    # 气象数据获取
    try:
        from datetime import timezone as _tz
        now = datetime.now(_tz.utc).replace(tzinfo=None)
        weather_start = base_dt.strftime("%Y-%m-%d")
        forecast_end = (base_dt + timedelta(days=8)).strftime("%Y-%m-%d")
        if base_dt < now - timedelta(days=1):
            hist_end_dt = min(base_dt + timedelta(days=8), now - timedelta(days=1))
            forecast_end = hist_end_dt.strftime("%Y-%m-%d")
            weather = fetcher.fetch_weather(
                province, weather_start, forecast_end, mode="historical")
        else:
            try:
                weather = fetcher.fetch_weather(
                    province, weather_start, forecast_end, mode="forecast")
            except Exception:
                weather = fetcher.fetch_weather(
                    province, weather_start, forecast_end, mode="historical")
        if not weather.empty:
            weather["dt_merge"] = weather["dt"].dt.floor("15min")
            future_df["dt_merge"] = future_df["dt"].dt.floor("15min")
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

    # 派生天气特征
    temp = future_df.get("temperature", pd.Series(20, index=future_df.index))
    hum = future_df.get("humidity", pd.Series(50, index=future_df.index))
    future_df["temp_extremity"] = np.abs(temp.values - 22) / 15
    if "humidity" in future_df.columns:
        hum_factor = 1.0 + np.clip((hum.values - 50) / 100, -0.2, 0.3)
        future_df["temp_extremity"] = future_df["temp_extremity"].values * hum_factor
    _te = pd.cut(future_df["temp_extremity"].fillna(0.5),
        bins=[0, 0.2, 0.5, 1.0, 100],
        labels=[0, 1, 2, 3], include_lowest=True).astype(int)
    future_df["tod_x_temp_extreme"] = future_df["time_of_day"] * 4 + _te.astype(int)

    # cloud_factor
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

    # 极端天气标志
    if "temperature" in future_df.columns:
        t_mean = history["temperature"].mean() if "temperature" in history.columns else 22
        t_std = history["temperature"].std() if "temperature" in history.columns else 5
        t_std = max(t_std, 0.1)
        future_df["temp_zscore"] = (future_df["temperature"] - t_mean) / t_std
        future_df["is_heat_wave"] = ((future_df["temperature"] > 35) & (future_df["temp_zscore"] > 2)).astype(int)
        future_df["is_cold_snap"] = ((future_df["temperature"] < -5) & (future_df["temp_zscore"] < -2)).astype(int)

    extreme_fut = pd.DataFrame(index=future_df.index)
    extreme_fut["hw"] = future_df.get("is_heat_wave", 0)
    extreme_fut["cs"] = future_df.get("is_cold_snap", 0)
    extreme_fut["sw"] = (future_df.get("wind_speed", 0) > 15).astype(int)
    extreme_fut["hp"] = (future_df.get("precipitation", 0) > 25).astype(int)
    future_df["extreme_weather_flag"] = (extreme_fut.sum(axis=1) > 0).astype(int)
    future_df["extreme_weather_count"] = extreme_fut.sum(axis=1).astype(int)

    # 极端值统计特征
    if "value" in history.columns:
        hist_v = history["value"].dropna()
        if len(hist_v) >= 96:
            future_df["value_zscore_24h"] = 0.0
            future_df["value_percentile_7d"] = 0.5
        else:
            future_df["value_zscore_24h"] = 0.0
            future_df["value_percentile_7d"] = 0.5
    else:
        future_df["value_zscore_24h"] = 0.0
        future_df["value_percentile_7d"] = 0.5

    future_df["extreme_x_tod"] = future_df["extreme_weather_flag"] * future_df["time_of_day"]
    future_df["heat_wave_x_daylight"] = future_df["is_heat_wave"] * future_df["is_daylight"]
    if "temp_zscore" in future_df.columns:
        future_df["tzscore_x_tod"] = future_df["temp_zscore"] * future_df["time_of_day"]
    future_df["val_extreme_x_weather"] = (
        np.abs(future_df["value_zscore_24h"]) * future_df["extreme_weather_flag"]
    )

    # 高级滚动统计
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

    # 周模式强度
    if n_hist >= 672 * 4:
        day_of_week_vals = history["day_of_week"].values if "day_of_week" in history.columns else np.zeros(n_hist)
        minute_of_day = history["dt"].dt.hour.values * 4 + history["dt"].dt.minute.values // 15
        dow_tod_groups = {}
        for t_idx in range(max(0, n_hist - 672 * 4), n_hist):
            key = (int(day_of_week_vals[t_idx]), int(minute_of_day[t_idx]))
            v = float(hist_vals[t_idx])
            if key not in dow_tod_groups:
                dow_tod_groups[key] = []
            dow_tod_groups[key].append(v)
        weekly_consistency = {}
        for key, vs in dow_tod_groups.items():
            if len(vs) >= 3:
                weekly_consistency[key] = 1.0 - min(1.0, float(np.std(vs)) / (abs(np.mean(vs)) + 1.0))
            else:
                weekly_consistency[key] = 0.0
        future_week = future_df["day_of_week"].values
        future_tod = future_df["hour"].values * 4
        future_df["weekly_pattern_corr"] = [
            weekly_consistency.get((int(future_week[i]), int(future_tod[i])), 0.0)
            for i in range(len(future_df))
        ]
    else:
        future_df["weekly_pattern_corr"] = 0.0

    # 多尺度波动率体制
    if n_hist >= 96:
        r96 = hist_vals[-96:]
        r672 = hist_vals[-min(672, n_hist):]
        vol_short = float(np.std(r96) / (np.mean(np.abs(r96)) + 0.01))
        vol_long = float(np.std(r672) / (np.mean(np.abs(r672)) + 0.01))
        future_df["volatility_regime"] = 0 if vol_short < 0.3 * vol_long else 1 if vol_short < 0.7 * vol_long else 2
        future_df["vol_ratio_short_long"] = vol_short / max(vol_long, 0.001)
        if n_hist >= 192:
            r96_prev = hist_vals[-192:-96]
            vol_prev = float(np.std(r96_prev) / (np.mean(np.abs(r96_prev)) + 0.01))
            future_df["vol_trend"] = float(vol_short - vol_prev)
        else:
            future_df["vol_trend"] = 0.0
    else:
        future_df["volatility_regime"] = 1
        future_df["vol_ratio_short_long"] = 1.0
        future_df["vol_trend"] = 0.0

    # Analog 天气匹配
    future_df["analog_value_mean"] = 0.0
    future_df["analog_value_std"] = 0.0
    future_df["analog_dist_min"] = 0.0
    is_volatile = any(kw in target_type for kw in ("风电", "光伏"))
    if is_volatile and n_hist >= 96 * 30:
        weather_features = []
        hist_arrays = {}
        for wcol in ["temperature", "wind_speed", "solar_radiation", "humidity", "pressure"]:
            if wcol in history.columns:
                hist_arrays[wcol] = history[wcol].values[-min(n_hist, 672*4):]
                weather_features.append(wcol)
        if len(weather_features) >= 2:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            hw = np.column_stack([hist_arrays[w][-min(n_hist, 672*4):] for w in weather_features])
            hw_scaled = scaler.fit_transform(hw)
            fut_weather = np.column_stack([
                future_df[w].fillna(0).values[:len(future_df)] for w in weather_features
            ])
            fut_weather_scaled = scaler.transform(fut_weather)
            analog_vals = []
            for i in range(len(future_df)):
                dists = np.sqrt(np.sum((hw_scaled - fut_weather_scaled[i]) ** 2, axis=1))
                k = min(3, len(dists) - 2)
                if k > 0:
                    top_k = np.argpartition(dists, k)[:k]
                    analogs = []
                    for idx in top_k:
                        if idx + 96 < len(hist_vals):
                            analogs.append(np.mean(hist_vals[idx:idx+96]))
                    if analogs:
                        analog_vals.append((np.mean(analogs), np.std(analogs), np.min(dists[top_k])))
                    else:
                        analog_vals.append((float(np.mean(hist_vals[-96:])), 0.0, 1.0))
                else:
                    analog_vals.append((float(np.mean(hist_vals[-96:])), 0.0, 1.0))
            future_df["analog_value_mean"] = [a[0] for a in analog_vals]
            future_df["analog_value_std"] = [a[1] for a in analog_vals]
            future_df["analog_dist_min"] = [a[2] for a in analog_vals]

    # 天气衍生特征
    wind = future_df.get("wind_speed", pd.Series(3, index=future_df.index))
    solar = future_df.get("solar_radiation", pd.Series(0, index=future_df.index))
    press = future_df.get("pressure", pd.Series(1013, index=future_df.index))
    temp_vals = temp.values
    hum_vals = hum.values
    wind_vals = wind.values
    solar_vals = solar.values

    future_df["CDD"] = np.maximum(temp_vals - 26, 0)
    future_df["HDD"] = np.maximum(18 - temp_vals, 0)
    future_df["THI"] = 0.8 * temp_vals + 0.2 * hum_vals * temp_vals / 100.0
    future_df["wind_power_potential"] = 0.5 * 1.225 * np.maximum(wind_vals, 0) ** 3
    ws_vals = np.maximum(wind_vals, 0)
    future_df["wind_power_curve"] = np.where(
        ws_vals < 3, 0,
        np.where(ws_vals > 25, 0,
        np.where(ws_vals >= 12, 1,
        ((ws_vals - 3) / 9) ** 3))
    )
    future_df["solar_potential"] = solar_vals / 1000.0
    future_df["solar_efficiency"] = np.clip(1.0 - 0.005 * (temp_vals - 25), 0.5, 1.0)

    # 天气 lag/diff
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

    # 连续高温天数
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

    # working_day_type
    if "is_work_weekend" in future_df.columns:
        future_df["working_day_type"] = future_df["is_work_weekend"].astype(int) * 2 + (~future_df["is_weekend"]).astype(int)
    else:
        future_df["working_day_type"] = (~future_df["is_weekend"]).astype(int)

    # D+1 预测值 → 交叉类型映射 (同时间戳实际值缺失时注入)
    _FORECAST_TO_CROSS: dict = {
        "solar": "出力_光伏_实际",
        "wind": "出力_风电_实际",
        "load": "负荷_系统_实际",
        "tie_load": "出力_联络线_实际",
    }
    _CROSS_TO_FORECAST: dict = {v: k for k, v in _FORECAST_TO_CROSS.items()}
    _injected_ct: set = set()  # 记录已注入的交叉类型

    # 交叉类型特征 — 按每种类型的 data_availability 延迟独立加载
    from scripts.core.config import get_available_date as _get_avail_date
    cross_types = [
        "出力_光伏_实际", "出力_总_实际", "出力_水电含抽蓄_实际",
        "出力_联络线_实际", "出力_非市场_实际", "出力_风电_实际", "负荷_系统_实际",
    ]
    for ct in cross_types:
        try:
            # 该交叉类型在 base_dt 时的实际可用截止日期
            ct_avail = _get_avail_date(province, ct, base_dt)
            ct_data = store.load_features(
                province, ct,
                (ct_avail - timedelta(days=14)).strftime("%Y-%m-%d"),
                (ct_avail + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if ct_data is not None and not ct_data.empty and "value" in ct_data.columns:
                ct_vals = ct_data["value"].values
                ct_dt_index = pd.Series(np.arange(len(ct_vals)), index=ct_data["dt"].values)
                ct_mean_96 = float(np.mean(ct_vals[-96:]) if len(ct_vals) >= 96 else np.mean(ct_vals))
                for i in range(len(future_df)):
                    ft = future_df.loc[future_df.index[i], "dt"]

                    def _ct_pos(offset_hours):
                        lag_dt = ft - timedelta(hours=offset_hours)
                        if lag_dt in ct_dt_index.index:
                            return ct_dt_index[lag_dt]
                        for dm in [0, 15, -15, 30, -30, 45, -45, 60, -60]:
                            c = lag_dt + timedelta(minutes=dm)
                            if c in ct_dt_index.index:
                                return ct_dt_index[c]
                        return None

                    v_pos = _ct_pos(0)
                    if v_pos is not None:
                        future_df.loc[future_df.index[i], f"{ct}_value"] = float(ct_vals[v_pos])
                    else:
                        # D+1 预测值注入: 同时间戳实际值未入库时用日前预测值替代
                        fc_key = _CROSS_TO_FORECAST.get(ct)
                        if fc_key and forecast_data and fc_key in forecast_data:
                            fc_vals = forecast_data[fc_key]
                            fc_idx = ft.hour * 4 + ft.minute // 15
                            if fc_idx < len(fc_vals):
                                future_df.loc[future_df.index[i], f"{ct}_value"] = float(fc_vals[fc_idx])
                                _injected_ct.add(ct)
                            else:
                                future_df.loc[future_df.index[i], f"{ct}_value"] = ct_mean_96
                        else:
                            future_df.loc[future_df.index[i], f"{ct}_value"] = ct_mean_96

                    l1_pos = _ct_pos(24)
                    future_df.loc[future_df.index[i], f"{ct}_lag_1d"] = (
                        float(ct_vals[l1_pos]) if l1_pos is not None else ct_mean_96)

                    l7_pos = _ct_pos(168)
                    future_df.loc[future_df.index[i], f"{ct}_lag_7d"] = (
                        float(ct_vals[l7_pos]) if l7_pos is not None else ct_mean_96)

                    l2_pos = _ct_pos(48)
                    future_df.loc[future_df.index[i], f"{ct}_lag_2d"] = (
                        float(ct_vals[l2_pos]) if l2_pos is not None else ct_mean_96)
                    l14_pos = _ct_pos(336)
                    future_df.loc[future_df.index[i], f"{ct}_lag_14d"] = (
                        float(ct_vals[l14_pos]) if l14_pos is not None else ct_mean_96)
        except Exception:
            for suffix in ["_value", "_lag_1d", "_lag_7d", "_lag_2d", "_lag_14d"]:
                col_name = f"{ct}{suffix}"
                if col_name not in future_df.columns:
                    future_df[col_name] = 0.0

    if _injected_ct:
        logger.info("D+1 预测值注入: %s ← forecast (替代缺失实际值, %d 步)",
                     ", ".join(sorted(_injected_ct)), horizon_steps)

    # 价格衍生特征
    proxy_value = future_df["value_lag_1d"].values
    load_col = "负荷_系统_实际_value"
    if load_col in future_df.columns:
        load_vals = future_df[load_col].replace(0, np.nan).values
        future_df["price_per_load"] = proxy_value / np.where(np.isnan(load_vals), 1.0, np.maximum(load_vals, 1.0))
        future_df["price_x_load"] = proxy_value * np.nan_to_num(load_vals, nan=0)
    else:
        if load_col in future_df.columns:
            load_vals = future_df[load_col].fillna(0).values
        else:
            load_vals = np.full(len(future_df), float(np.mean(hist_vals[-96:]) if n_hist >= 96 else last_value))
        future_df["price_per_load"] = proxy_value / np.maximum(np.abs(load_vals), 1.0)
        future_df["price_x_load"] = proxy_value * load_vals

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

    if total_out_col in future_df.columns and load_col in future_df.columns:
        supply = future_df[total_out_col].fillna(0)
        demand = future_df[load_col].replace(0, np.nan).fillna(1)
        future_df["supply_demand_ratio"] = supply / demand
        future_df["supply_surplus"] = supply - demand.fillna(0)
    else:
        future_df["supply_demand_ratio"] = 0.0
        future_df["supply_surplus"] = 0.0

    load_val = future_df[load_col].fillna(0).values if load_col in future_df.columns else np.zeros(len(future_df))
    wind_val = future_df[wind_col].fillna(0).values if wind_col in future_df.columns else np.zeros(len(future_df))
    solar_val = future_df[solar_col].fillna(0).values if solar_col in future_df.columns else np.zeros(len(future_df))
    future_df["residual_load"] = load_val - wind_val - solar_val
    future_df["residual_load_ratio"] = np.where(
        load_val > 0, future_df["residual_load"].values / load_val, 1.0)
    re_today = wind_val + solar_val
    if wind_col in future_df.columns and solar_col in future_df.columns:
        re_yesterday = (future_df.get(f"{wind_col.replace('_value', '_lag_1d')}", pd.Series(0, index=future_df.index)).fillna(0).values +
                       future_df.get(f"{solar_col.replace('_value', '_lag_1d')}", pd.Series(0, index=future_df.index)).fillna(0).values)
        load_yesterday = future_df.get(f"{load_col.replace('_value', '_lag_1d')}", pd.Series(load_val, index=future_df.index)).fillna(0).values if f"{load_col.replace('_value', '_lag_1d')}" in future_df.columns else load_val
        future_df["re_share_change_1d"] = np.where(
            load_yesterday > 0, re_today / np.maximum(load_val, 1) - re_yesterday / np.maximum(load_yesterday, 1), 0)
    else:
        future_df["re_share_change_1d"] = 0.0

    # 价格形成特征
    is_price = "电价" in target_type
    if is_price:
        proxy_vals = future_df["value_lag_1d"].values
        morning_peak = np.max([proxy_vals[i] for i in range(len(future_df)) if 7 <= future_df.loc[future_df.index[i], "hour"] <= 9], initial=np.mean(proxy_vals))
        evening_peak = np.max([proxy_vals[i] for i in range(len(future_df)) if 17 <= future_df.loc[future_df.index[i], "hour"] <= 20], initial=np.mean(proxy_vals))
        midday_valley = np.min([proxy_vals[i] for i in range(len(future_df)) if 11 <= future_df.loc[future_df.index[i], "hour"] <= 14], initial=np.mean(proxy_vals))
        night_valley = np.min([proxy_vals[i] for i in range(len(future_df)) if 2 <= future_df.loc[future_df.index[i], "hour"] <= 5], initial=np.mean(proxy_vals))
        future_df["peak_valley_spread"] = float(max(morning_peak, evening_peak) - min(midday_valley, night_valley))
        future_df["morning_evening_ratio"] = float(morning_peak / max(evening_peak, 0.01))
        load_peak = float(np.max(load_val)) if load_val.max() > 0 else 1.0
        future_df["load_factor"] = load_val / max(load_peak, 1.0)
        load_ramp = np.abs(np.diff(load_val, prepend=load_val[0]))
        future_df["load_ramp_rate"] = load_ramp / max(load_peak, 1.0)
        if "supply_demand_ratio" in future_df.columns:
            sdr = future_df["supply_demand_ratio"].values
            future_df["sdr_zscore"] = (sdr - np.mean(sdr)) / max(np.std(sdr), 0.001)
        else:
            future_df["sdr_zscore"] = 0.0
    else:
        for col in ["peak_valley_spread", "morning_evening_ratio", "load_factor",
                    "load_ramp_rate", "sdr_zscore"]:
            future_df[col] = 0.0

    # price momentum
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

    proxy_min = np.min(proxy_value)
    proxy_max = np.max(proxy_value)
    proxy_range = max(proxy_max - proxy_min, 1.0)
    future_df["price_position"] = (proxy_value - proxy_min) / proxy_range

    peak_mask = future_df["peak_valley"] == 2
    off_mask = future_df["peak_valley"] == 0
    peak_vals = proxy_value[peak_mask.values]
    off_vals = proxy_value[off_mask.values]
    if len(peak_vals) > 0 and len(off_vals) > 0:
        future_df["peak_off_peak_spread"] = float(np.mean(peak_vals) - np.mean(off_vals))
    else:
        future_df["peak_off_peak_spread"] = 0.0

    # 误差修正特征 (未来无预测, 初始化为0)
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

    # NaN 填充
    for col in future_df.columns:
        if col not in ("dt", "province", "type") and future_df[col].dtype == np.float64:
            future_df[col] = future_df[col].fillna(
                history[col].mean() if col in history.columns else 0)
    return future_df
