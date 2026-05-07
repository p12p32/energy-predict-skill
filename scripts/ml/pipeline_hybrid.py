"""pipeline_hybrid.py — 混合预测编排器: 物理链 + ML残差.

预测流程:
1. 气象预报 → solar / wind 物理预测
2. 气象 + 风光 → load 分解预测
3. net_load = load − solar − wind
4. net_load → price 预测
5. [可选] Transformer 残差修正
6. 物理约束
"""
import os, json, logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from scripts.ml.physics.solar_model import SolarParametricModel
from scripts.ml.physics.wind_model import WindParametricModel
from scripts.ml.physics.load_model import LoadDecompositionModel
from scripts.ml.physics.price_model import PriceStructuralModel
from scripts.ml.physics.net_load import NetLoadComputer
from scripts.ml.physics.base import PhysicalModelConfig
from scripts.ml.layers.constraints import PhysicalConstraints
from scripts.data.weather_features import WeatherFeatureEngineer
from scripts.data.features import FeatureStore
from scripts.core.config import load_config

logger = logging.getLogger(__name__)


class HybridPipeline:
    def __init__(self, model_dir: str = "models", store: FeatureStore = None):
        self.model_dir = model_dir
        self.store = store or FeatureStore()
        self.config = load_config()
        self._hybrid_cfg = self.config.get("hybrid", {})
        self._models: dict = {}  # lazy loaded
        self.net_load_computer = NetLoadComputer()
        self.constraints = PhysicalConstraints()
        self._weather_engineer = WeatherFeatureEngineer()

    def _model_key(self, province: str, model_type: str) -> str:
        return f"{province}_{model_type}"

    def _get_model(self, province: str, model_type: str):
        key = self._model_key(province, model_type)
        if key in self._models:
            return self._models[key]

        # Load from registry
        reg_path = os.path.join(self.model_dir, "model_registry.json")
        if not os.path.exists(reg_path):
            return None

        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        hybrid_reg = reg.get("hybrid", {}).get(key)
        if not hybrid_reg:
            return None

        prefix = hybrid_reg.get("model_prefix", "")
        base_dir = os.path.join(self.model_dir, province)

        model_cls = {
            "solar": SolarParametricModel,
            "wind": WindParametricModel,
            "load": LoadDecompositionModel,
            "price": PriceStructuralModel,
        }.get(model_type)

        if model_cls is None:
            return None

        coords = self.config.get("province_coords", {}).get(province, {})
        phys_cfg = self._hybrid_cfg.get("physics", {}).get(model_type, {})
        cfg = PhysicalModelConfig(
            model_type=phys_cfg.get("model_type", "parametric"),
            province=province,
            lat=coords.get("lat", 30.0),
            lon=coords.get("lon", 104.0),
        )

        model = model_cls(config=cfg)
        # prefix 已包含 model_type (e.g. "solar_20260504_155312"), 不要再拼接
        path = os.path.join(base_dir, f"{prefix}.lgbm")
        json_path = f"{os.path.splitext(path)[0]}_params.json"
        if os.path.exists(json_path):
            model.load(path)
            self._models[key] = model
            return model
        # LGBM fallback 模型没有 json params, 直接 load pkl
        if os.path.exists(path):
            try:
                model.load(path)
                self._models[key] = model
                return model
            except Exception:
                pass
        return None

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24, reference_date: str = None) -> pd.DataFrame:
        logger.info("HybridPipeline.predict: %s/%s, horizon=%dh", province, target_type, horizon_hours)

        # ── 0. 日期 + 频率检测 ──
        if reference_date:
            end_date = pd.to_datetime(reference_date)
        else:
            end_date = datetime.now()

        # 检测目标数据频率
        target_freq_minutes = 15
        target_points_per_hour = 4
        try:
            hist = self.store.load_features(province, target_type,
                                            end_date - timedelta(days=7), end_date)
            if hist is not None and len(hist) >= 2:
                delta = pd.to_datetime(hist["dt"]).diff().mode().iloc[0]
                target_freq_minutes = int(delta.total_seconds() / 60)
                target_points_per_hour = int(round(60 / target_freq_minutes))
        except Exception:
            pass

        target_steps = horizon_hours * target_points_per_hour
        physics_freq = 15  # 物理模型始终在15min分辨率运行
        physics_pts_per_hour = 4
        physics_steps = horizon_hours * physics_pts_per_hour

        # ── 1. 气象数据 (15min分辨率, 供物理模型) ──
        weather_physics = self._get_weather(province, end_date, physics_steps, physics_freq)
        if weather_physics is None or len(weather_physics) == 0:
            logger.warning("无气象数据, 回退统计基线")
            return self._fallback_predict(province, target_type, target_steps, end_date,
                                          target_freq_minutes)

        # ── 2. 风光物理预测 (15min) ──
        solar_15min = np.zeros(physics_steps)
        wind_15min = np.zeros(physics_steps)

        solar_model = self._get_model(province, "solar")
        if solar_model is not None:
            solar_15min = solar_model.predict(weather_physics)

        wind_model = self._get_model(province, "wind")
        if wind_model is not None:
            wind_15min = wind_model.predict(weather_physics)

        # ── 3. 负荷预测 (15min) ──
        load_model = self._get_model(province, "load")
        if load_model is not None:
            load_15min = load_model.predict(weather_physics, solar_15min, wind_15min)
        else:
            load_15min = np.zeros(physics_steps)

        # ── 4. 降采样到目标频率 ──
        if target_freq_minutes > physics_freq:
            ratio = int(target_freq_minutes / physics_freq)
            solar_pred = self._downsample(solar_15min, ratio)
            wind_pred = self._downsample(wind_15min, ratio)
            load_pred = self._downsample(load_15min, ratio)
            weather_target = self._downsample_weather(weather_physics, ratio)
            logger.info("降采样: 15min→%dmin, %d→%d点",
                        target_freq_minutes, physics_steps, target_steps)
        else:
            ratio = 1
            solar_pred = solar_15min[:target_steps]
            wind_pred = wind_15min[:target_steps]
            load_pred = load_15min[:target_steps]
            weather_target = weather_physics

        logger.info("Solar: mean=%.0f MW, Wind: mean=%.0f MW, Load: mean=%.0f MW",
                     np.mean(solar_pred), np.mean(wind_pred), np.mean(load_pred))

        # ── 5. 净负荷 ──
        net_load = self.net_load_computer.compute(load_pred, solar_pred, wind_pred)

        # ── 6. 根据 target_type 选择主预测 ──
        if "光伏" in target_type:
            p50 = solar_pred.copy()
        elif "风电" in target_type:
            p50 = wind_pred.copy()
        elif "负荷" in target_type or "联络线" in target_type or "直调" in target_type:
            p50 = load_pred.copy()
        elif "电价" in target_type:
            p50 = self._predict_price_target(province, target_type, weather_target,
                                              net_load, end_date, target_steps,
                                              target_points_per_hour)
        else:
            p50 = load_pred.copy() if np.any(load_pred) else solar_pred + wind_pred

        # ── 7. 物理约束 ──
        p10 = p50 * 0.90
        p90 = p50 * 1.10
        p50, p10, p90 = self.constraints.apply(
            p50, p10, p90, weather_target, target_type, province, self.config)

        # ── 8. 格式化输出 ──
        future_dts = pd.date_range(
            start=end_date, periods=target_steps, freq=f"{int(target_freq_minutes)}min")
        future_dts = future_dts[:target_steps]

        p50 = p50[:target_steps]
        p10 = p10[:target_steps]
        p90 = p90[:target_steps]

        result = pd.DataFrame({
            "dt": future_dts,
            "province": province,
            "type": target_type,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "model_version": "hybrid_v1",
        })

        if "电价" in target_type:
            net_load_decomp = self.net_load_computer.decompose(
                net_load, load_pred, solar_pred, wind_pred)
            result["net_load_mean"] = float(np.mean(net_load))
            result["renewable_share"] = float(
                np.mean(net_load_decomp.get("renewable_share", [0])))

            # 附加多周期趋势信号 (直接从 p50 计算)
            n_out = len(p50)
            for label, steps in [("trend_1h", 4), ("trend_4h", 16), ("trend_6h", 24)]:
                signal = np.zeros(n_out)
                for t in range(n_out):
                    if t + steps < n_out:
                        diff = p50[t + steps] - p50[t]
                        signal[t] = np.tanh(diff / (abs(p50[t]) + 1) * 10)
                    elif t > 0:
                        signal[t] = signal[t - 1]
                result[label] = signal[:target_steps]

        self.store.insert_predictions(result)
        return result

    @staticmethod
    def _downsample(arr: np.ndarray, ratio: int) -> np.ndarray:
        """降采样: 每ratio个点取均值."""
        n = len(arr) // ratio * ratio
        if n == 0:
            return arr
        return np.mean(arr[:n].reshape(-1, ratio), axis=1)

    @staticmethod
    def _downsample_weather(df: pd.DataFrame, ratio: int) -> pd.DataFrame:
        """降采样气象DataFrame: 每ratio行取均值."""
        if ratio <= 1:
            return df
        n = len(df) // ratio * ratio
        if n == 0:
            return df
        result_rows = []
        for i in range(0, n, ratio):
            chunk = df.iloc[i:i+ratio]
            row = {}
            for col in df.columns:
                if col == "dt":
                    row[col] = chunk[col].iloc[0]  # 保留起始时间
                elif pd.api.types.is_numeric_dtype(df[col]):
                    row[col] = chunk[col].mean()
                else:
                    row[col] = chunk[col].mode().iloc[0] if len(chunk[col].mode()) > 0 else chunk[col].iloc[0]
            result_rows.append(row)
        return pd.DataFrame(result_rows)

    def _predict_price_target(self, province, target_type, weather, net_load,
                               end_date, horizon_steps, points_per_hour):
        """电价预测: 物理 net_load + 日前电价 + 自回归价格信号 + Transformer 残差修正."""
        da_price = self._load_da_price(province, end_date.strftime("%Y%m%d"))
        lag_1d = self._load_price_lag(province, end_date, 1, horizon_steps)
        lag_7d = self._load_price_lag(province, end_date, 7, horizon_steps)

        # 加载过去实际价格 (供方向分类器使用)
        past_actual = self._load_price_history(province, end_date, 3, 96)

        price_model = self._get_model(province, "price")
        if price_model is not None:
            pred = price_model.predict_with_dir(net_load, weather, da_price,
                                                 lag_1d, lag_7d, past_actual)
        else:
            pred = self._price_statistical_baseline(province, end_date, horizon_steps, net_load)

        # Transformer 残差修正
        try:
            pred = self._apply_transformer_correction(
                province, end_date, pred, lag_1d, points_per_hour)
        except Exception as e:
            logger.debug("Transformer 修正跳过: %s", e)

        return pred

    def _apply_transformer_correction(self, province, end_date, lgbm_pred,
                                       lag_1d, points_per_hour):
        """用 Transformer 修正 LGBM 预测的时序不一致."""
        import os
        from scripts.ml.transformer.trainer import TransformerTrainer

        # 加载 Transformer 模型
        reg_path = os.path.join(self.model_dir, "model_registry.json")
        if not os.path.exists(reg_path):
            return lgbm_pred
        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        key = f"{province}_price"
        tf_prefix = reg.get("hybrid", {}).get(key, {}).get("transformer_path")
        if not tf_prefix or not os.path.exists(tf_prefix):
            return lgbm_pred

        # 加载过去 192 点实际价格作为 Transformer 输入
        pts_per_day = int(round(1440 / (60 / points_per_hour))) if points_per_hour else 96
        past_actual = self._load_price_history(province, end_date, 2, pts_per_day)
        if past_actual is None or len(past_actual) < pts_per_day:
            return lgbm_pred

        past_actual = past_actual[-pts_per_day * 2:]
        # past_phys: 用 lag_1d 作为简单基线 (Transformer 学习从基线→实际的映射)
        past_phys = past_actual.copy()
        if lag_1d is not None and len(lag_1d) >= pts_per_day:
            past_phys[-pts_per_day:] = lag_1d[:pts_per_day]

        future_phys = lgbm_pred[:pts_per_day]

        tf_trainer = TransformerTrainer(
            d_model=64, n_heads=4, n_layers=2,
            seq_len=pts_per_day, horizon=pts_per_day)
        tf_trainer.build_model()
        tf_trainer.load(tf_prefix)

        residuals = tf_trainer.predict_residuals(
            past_phys, past_actual, future_phys)
        corrected = future_phys + residuals[-len(future_phys):]

        logger.info("Transformer 修正: mean_residual=%.1f", float(np.mean(residuals)))
        return corrected

    def _load_price_history(self, province, end_date, days_back, pts_per_day):
        """加载过去 N 天实际价格序列."""
        try:
            import pymysql
            province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
            start_date = end_date - timedelta(days=days_back)
            conn = pymysql.connect(
                host="127.0.0.1", port=3306, user="root",
                password="root123456", database="electric_power_db")
            cursor = conn.cursor()
            cursor.execute('''
                SELECT date_key, time_point_id, price_value
                FROM f_price_15min
                WHERE province = %s AND date_key >= %s AND date_key < %s
                  AND price_market_id = 2
                ORDER BY date_key, time_point_id
            ''', (province_cn, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            rows = cursor.fetchall()
            conn.close()
            if rows and len(rows) >= pts_per_day:
                return np.array([float(r[2]) for r in rows], dtype=np.float64)
        except Exception:
            pass

        # Fallback: feature store
        try:
            hist = self.store.load_features(province, "电价_实时_实际",
                                            end_date - timedelta(days=days_back + 1),
                                            end_date)
            if hist is not None and len(hist) >= pts_per_day:
                return hist["value"].values[-pts_per_day * days_back:]
        except Exception:
            pass
        return None

    def _load_price_lag(self, province, end_date, days_back, n_steps):
        """加载历史价格作为滞后特征."""
        lag_date = end_date - timedelta(days=days_back)
        lag_date_key = lag_date.strftime("%Y%m%d")

        # 首先尝试从 DB 加载
        try:
            import pymysql
            province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
            conn = pymysql.connect(
                host="127.0.0.1", port=3306, user="root",
                password="root123456", database="electric_power_db")
            cursor = conn.cursor()
            cursor.execute('''
                SELECT time_point_id, price_value
                FROM f_price_15min
                WHERE province = %s AND date_key = %s AND price_market_id = 2
                ORDER BY time_point_id
                LIMIT 96
            ''', (province_cn, lag_date_key))
            rows = cursor.fetchall()
            conn.close()
            if rows and len(rows) >= 24:
                vals = np.array([float(r[1]) for r in rows], dtype=np.float64)
                # 根据点数扩展或裁剪到目标长度
                if len(vals) < n_steps:
                    vals = np.tile(vals, n_steps // len(vals) + 1)[:n_steps]
                return vals[:n_steps]
        except Exception:
            pass

        # 回退: 从 feature store 加载
        try:
            hist = self.store.load_features(province, "电价_实时_实际",
                                            lag_date - timedelta(hours=1),
                                            lag_date + timedelta(hours=25))
            if hist is not None and len(hist) >= 24:
                pts_per_day = 96
                vals = hist["value"].values[-pts_per_day:] if len(hist) >= pts_per_day else hist["value"].values
                if len(vals) < n_steps:
                    vals = np.tile(vals, n_steps // len(vals) + 1)[:n_steps]
                return vals[:n_steps]
        except Exception:
            pass

        return None

    def _price_statistical_baseline(self, province, end_date, n_steps, net_load):
        """统计基线: 昨日 + 7日均值 (保留之前的谷底修正逻辑)."""
        try:
            hist_end = end_date - timedelta(days=1)
            hist = self.store.load_features(province, "电价_实时_实际",
                                            hist_end - timedelta(days=14), hist_end)
            if hist is not None and len(hist) >= 96 * 2:
                pts_per_day = 96
                values = hist["value"].values
                yest = values[-pts_per_day:]
                avg7 = np.mean(values[-7 * pts_per_day:].reshape(7, pts_per_day), axis=0)

                yest_mean = float(np.mean(yest))
                yest_shape = yest - yest_mean
                avg7_mean = float(np.mean(avg7))
                anomaly = float(np.mean(np.abs(yest - avg7) / (np.abs(avg7) + 1e-6)))
                w_yest = np.clip(1.0 - anomaly, 0.3, 0.7)
                w_avg = 1.0 - w_yest

                blend_level = w_yest * yest_mean + w_avg * avg7_mean
                shape_w = 0.75
                avg7_shape = avg7 - avg7_mean
                p50_raw = blend_level + shape_w * yest_shape + (1 - shape_w) * avg7_shape

                repeated = np.tile(p50_raw, n_steps // pts_per_day + 1)[:n_steps]
                return repeated
        except Exception as e:
            logger.warning("统计基线失败: %s", e)

        return np.full(n_steps, 350.0)

    # ── 气象 ──

    def _get_weather(self, province, end_date, n_steps, freq_minutes=15):
        # 从本地 CSV 加载天气数据，取最近 N 天对应时段作为"持续性预报"
        try:
            import glob
            weather_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                ".energy_data", "weather")
            pattern = os.path.join(weather_dir, f"{province}_weather.csv")
            files = glob.glob(pattern)
            if files:
                weather = pd.read_csv(files[0])
                weather["dt"] = pd.to_datetime(weather["dt"])
                coords = self.config.get("province_coords", {}).get(province, {})
                lat = coords.get("lat", 30.0)
                weather = self._weather_engineer.transform(weather, lat=lat)

                latest = weather["dt"].max()
                lookback_start = latest - timedelta(days=14)
                recent = weather[(weather["dt"] >= lookback_start) & (weather["dt"] <= latest)]

                # 检测气象数据原始频率
                wx_delta = recent["dt"].diff().mode().iloc[0]
                wx_freq_hours = wx_delta.total_seconds() / 3600
                wx_pts_per_day = int(1440 / (wx_freq_hours * 60))

                target_pts_per_day = int(1440 / freq_minutes)

                # Step 1: 在气象数据原始频率上做持久性预报
                end_hour = end_date.floor("h") + pd.Timedelta(hours=1) if end_date.minute > 0 else end_date
                wx_dts = pd.date_range(start=end_date, periods=wx_pts_per_day + 1, freq=wx_delta)
                wx_dts = wx_dts[:wx_pts_per_day + 1]

                result_rows = []
                for target_dt in wx_dts:
                    tod = target_dt.time()
                    same_tod = recent[recent["dt"].dt.time == tod]
                    if len(same_tod) > 0:
                        result_rows.append(same_tod.iloc[-7:].mean(numeric_only=True).to_dict())
                    else:
                        hour_match = recent[recent["dt"].dt.hour == target_dt.hour]
                        if len(hour_match) > 0:
                            result_rows.append(hour_match.iloc[-7:].mean(numeric_only=True).to_dict())
                        else:
                            result_rows.append({})

                if not result_rows:
                    return self._zero_weather(forecast_dts)

                wx_forecast = pd.DataFrame(result_rows)
                wx_forecast["dt"] = wx_dts[:len(wx_forecast)]

                # Step 2: 如果目标频率高于气象频率，插值到目标分辨率
                if wx_pts_per_day < target_pts_per_day and len(wx_forecast) >= 2:
                    target_dts = pd.date_range(start=end_date, periods=n_steps,
                                               freq=f"{int(freq_minutes)}min")
                    wx_forecast = wx_forecast.set_index("dt")

                    numeric_cols = wx_forecast.select_dtypes(include=[np.number]).columns
                    result = pd.DataFrame(index=target_dts)

                    for col in numeric_cols:
                        s = wx_forecast[col].dropna()
                        if len(s) >= 2:
                            combined = s.reindex(s.index.union(target_dts)).sort_index()
                            result[col] = combined.interpolate(method="time").reindex(target_dts).values
                        else:
                            result[col] = s.iloc[0] if len(s) > 0 else 0.0

                    # 非数值列 (如 is_holiday) 用 forward-fill
                    for col in wx_forecast.columns:
                        if col not in numeric_cols:
                            s = wx_forecast[col].dropna()
                            if len(s) >= 1:
                                combined = s.reindex(s.index.union(target_dts)).sort_index()
                                result[col] = combined.ffill().reindex(target_dts).values
                            else:
                                result[col] = 0.0

                    result["dt"] = target_dts
                    return result.reset_index(drop=True)

                return wx_forecast.head(n_steps).reset_index(drop=True)

        except Exception as e:
            logger.warning("气象加载失败: %s", e)

        return self._zero_weather(end_date, n_steps, freq_minutes)

    def _zero_weather(self, end_date, n_steps, freq_minutes=15):
        dts = pd.date_range(start=end_date, periods=n_steps, freq=f"{int(freq_minutes)}min")
        wx = pd.DataFrame({"dt": dts})
        for col in ["temperature", "humidity", "wind_speed", "wind_direction",
                     "solar_radiation", "precipitation", "pressure", "cloud_factor",
                     "CDD", "HDD", "THI"]:
            wx[col] = 0.0
        return wx

    def _load_da_price(self, province, date_key):
        # 四川无独立日前市场, 返回 None 避免信息泄露
        if "四川" in province:
            return None
        try:
            import pymysql
            province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
            conn = pymysql.connect(
                host="127.0.0.1", port=3306, user="root",
                password="root123456", database="electric_power_db")
            cursor = conn.cursor()
            cursor.execute('''
                SELECT time_point_id, price_value
                FROM f_price_15min
                WHERE province = %s AND date_key = %s AND price_market_id = 1
                ORDER BY time_point_id
            ''', (province_cn, date_key))
            rows = cursor.fetchall()
            conn.close()
            if rows and len(rows) >= 24:
                return np.array([float(r[1]) for r in rows], dtype=np.float64)
        except Exception:
            pass
        return None

    def _fallback_predict(self, province, target_type, n_steps, end_date, freq_minutes=15):
        dts = pd.date_range(start=end_date, periods=n_steps, freq=f"{int(freq_minutes)}min")
        return pd.DataFrame({
            "dt": dts[:n_steps],
            "province": province,
            "type": target_type,
            "p10": np.zeros(n_steps),
            "p50": np.zeros(n_steps),
            "p90": np.zeros(n_steps),
            "model_version": "hybrid_v1_fallback",
        })
