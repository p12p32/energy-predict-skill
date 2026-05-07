"""trainer_hybrid.py — 混合管线训练编排器: 依赖顺序分层训练.

Tier 1 (并行): Solar + Wind — 只依赖气象
Tier 2:         Load — 依赖 Tier 1
Tier 3:         Price — 依赖 Tier 2
Tier 4 (并行):  Transformer — 依赖所有残差 (未来)
"""
import os, json, logging
from datetime import datetime
import numpy as np
import pandas as pd

from scripts.ml.physics.solar_model import SolarParametricModel
from scripts.ml.physics.wind_model import WindParametricModel
from scripts.ml.physics.load_model import LoadDecompositionModel
from scripts.ml.physics.price_model import PriceStructuralModel
from scripts.ml.physics.net_load import NetLoadComputer
from scripts.ml.physics.base import PhysicalModelConfig
from scripts.data.weather_features import WeatherFeatureEngineer
from scripts.core.config import load_config, get_provinces

logger = logging.getLogger(__name__)


class HybridTrainer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.config = load_config()
        self._hybrid_cfg = self.config.get("hybrid", {})
        self._phys_cfg = self._hybrid_cfg.get("physics", {})
        self._weather_engineer = WeatherFeatureEngineer()
        self.net_load = NetLoadComputer()
        self._weather_cache: dict = {}
        self._saved_prefixes: dict = {}

    def train_all(self) -> dict:
        results = {}
        provinces = get_provinces()

        for province in provinces:
            logger.info("=" * 50)
            logger.info("Hybrid 训练: %s", province)
            logger.info("=" * 50)

            # 加载气象数据
            weather = self._load_weather(province)
            if weather is None or len(weather) == 0:
                logger.warning("%s: 无气象数据, 跳过", province)
                continue

            self._saved_prefixes = {}

            p_result = {}

            # 发现可用类型
            available = self._scan_types(province)
            logger.info("可用类型: %s", list(available.keys()))

            # Tier 1: Solar + Wind (并行)
            solar_model = None
            wind_model = None
            solar_key = next((k for k in available if "光伏" in k), None)
            wind_key = next((k for k in available if "风电" in k), None)

            if solar_key:
                solar_model = self._train_solar(province, weather, available[solar_key])
                p_result["solar"] = solar_model.metrics if solar_model else None

            if wind_key:
                wind_model = self._train_wind(province, weather, available[wind_key])
                p_result["wind"] = wind_model.metrics if wind_model else None

            # Tier 2: Load
            solar_pred = None
            wind_pred = None
            if solar_model and solar_key:
                solar_pred = self._predict_on_history(solar_model, weather, available[solar_key])
            if wind_model and wind_key:
                wind_pred = self._predict_on_history(wind_model, weather, available[wind_key])

            load_model = None
            # 优先直调负荷 (系统总负荷), 联络线负荷不能代表全网供需
            load_keys = [k for k in available if "负荷" in k]
            if "直调" in str(load_keys):
                load_keys.sort(key=lambda k: 0 if "直调" in k else 1)
            for load_key in load_keys:
                load_model = self._train_load(province, weather, available[load_key],
                                               solar_pred, wind_pred)
                p_result["load"] = load_model.metrics if load_model else None
                logger.info("Load 训练完成: %s", load_key)
                break  # 只训练第一个 (优先直调)

            # Tier 3: Price
            price_model = None
            price_data = None
            for price_key, price_df in available.items():
                if "电价" in price_key:
                    price_data = price_df
                    price_model = self._train_price(province, weather, price_df,
                                                    load_model, solar_model, wind_model,
                                                    solar_pred, wind_pred, available)
                    if price_model and price_model.metrics:
                        p_result["price"] = price_model.metrics
                    break  # 只训练第一个电价类型

            # Tier 4: Transformer 残差修正
            if price_model is not None and price_data is not None:
                tf_metrics = self._train_transformer(province, weather, price_data,
                                                     price_model, load_model,
                                                     solar_model, wind_model,
                                                     solar_pred, wind_pred)
                if tf_metrics:
                    p_result["transformer"] = tf_metrics

            results[province] = p_result

            # 更新 registry
            self._update_registry(province, solar_model, wind_model, load_model,
                                  price_model)

        return {"status": "ok", "results": results}

    # ── 气象 ──

    def _load_weather(self, province):
        import glob
        key = province
        if key in self._weather_cache:
            return self._weather_cache[key]

        weather_dir = os.path.join(".energy_data", "weather")
        pattern = os.path.join(weather_dir, f"{province}_weather.csv")
        files = glob.glob(pattern)

        if not files:
            # 尝试 fetcher
            try:
                from scripts.data.fetcher import DataFetcher
                end = datetime.now()
                start = end - pd.Timedelta(days=180)
                fetcher = DataFetcher()
                weather = fetcher.fetch_weather(province, start, end)
                if weather is not None and len(weather) > 0:
                    coords = self.config.get("province_coords", {}).get(province, {})
                    lat = coords.get("lat", 30.0)
                    weather = self._weather_engineer.transform(weather, lat=lat)
                    self._weather_cache[key] = weather
                    return weather
            except Exception as e:
                logger.warning("fetcher 失败: %s", e)
            return None

        weather = pd.read_csv(files[0], parse_dates=["dt"] if "dt" in open(files[0]).readline() else False)
        if "dt" not in weather.columns and "time" in weather.columns:
            weather["dt"] = pd.to_datetime(weather["time"])
        elif "dt" not in weather.columns:
            weather["dt"] = pd.date_range(end=datetime.now(), periods=len(weather), freq="15min")

        coords = self.config.get("province_coords", {}).get(province, {})
        lat = coords.get("lat", 30.0)
        weather = self._weather_engineer.transform(weather, lat=lat)
        self._weather_cache[key] = weather
        return weather

    # ── 类型扫描 ──

    def _scan_types(self, province):
        import glob
        features_dir = os.path.join(".energy_data", "features")
        pattern = os.path.join(features_dir, f"{province}_*.parquet")
        files = glob.glob(pattern)

        available = {}
        for f in files:
            basename = os.path.basename(f)
            parts = basename.replace(".parquet", "").split("_")
            if len(parts) < 3:
                continue
            type_name = f"{parts[1]}_{parts[2]}_{parts[3]}" if len(parts) >= 4 else "_".join(parts[1:])

            try:
                df = pd.read_parquet(f)
                if "value" in df.columns and len(df) >= 96:
                    if "dt" not in df.columns:
                        continue
                    df["dt"] = pd.to_datetime(df["dt"])
                    available[type_name] = df
            except Exception:
                pass
        return available

    # ── Tier 1: Solar ──

    def _train_solar(self, province, weather, data):
        coords = self.config.get("province_coords", {}).get(province, {})
        solar_cfg = self._phys_cfg.get("solar", {})
        cfg = PhysicalModelConfig(
            model_type=solar_cfg.get("model_type", "parametric"),
            province=province,
            lat=coords.get("lat", 30.0),
            lon=coords.get("lon", 104.0),
        )

        model = SolarParametricModel(cfg)
        merged = self._merge_weather(data, weather)
        if merged is None or len(merged) == 0 or "value" not in merged.columns:
            return None

        model.fit(merged, merged["value"])
        self._saved_prefixes["solar"] = self._save_model(model, province, "solar")
        return model

    def _train_wind(self, province, weather, data):
        coords = self.config.get("province_coords", {}).get(province, {})
        wind_cfg = self._phys_cfg.get("wind", {})
        cfg = PhysicalModelConfig(
            model_type=wind_cfg.get("model_type", "parametric"),
            province=province,
            lat=coords.get("lat", 30.0),
            lon=coords.get("lon", 104.0),
        )

        model = WindParametricModel(cfg)
        merged = self._merge_weather(data, weather)
        if merged is None or len(merged) == 0 or "value" not in merged.columns:
            return None

        model.fit(merged, merged["value"])
        self._saved_prefixes["wind"] = self._save_model(model, province, "wind")
        return model

    # ── Tier 2: Load ──

    def _train_load(self, province, weather, data, solar_pred, wind_pred):
        coords = self.config.get("province_coords", {}).get(province, {})
        load_cfg = self._phys_cfg.get("load", {})
        cfg = PhysicalModelConfig(
            model_type="lightgbm",  # Load always uses ML for elasticity
            province=province,
            lat=coords.get("lat", 30.0),
            lon=coords.get("lon", 104.0),
        )

        model = LoadDecompositionModel(cfg)
        merged = self._merge_weather(data, weather)
        if merged is None or len(merged) == 0 or "value" not in merged.columns:
            return None

        # 构造包含风光预测的 weather
        if solar_pred is not None or wind_pred is not None:
            merged["solar_pred"] = solar_pred[:len(merged)] if solar_pred is not None else 0.0
            merged["wind_pred"] = wind_pred[:len(merged)] if wind_pred is not None else 0.0

        model.fit(merged, merged["value"])
        self._saved_prefixes["load"] = self._save_model(model, province, "load")
        return model

    # ── Tier 3: Price ──

    def _train_price(self, province, weather, price_data,
                     load_model, solar_model, wind_model,
                     solar_pred, wind_pred, available):
        coords = self.config.get("province_coords", {}).get(province, {})
        price_cfg = self._phys_cfg.get("price", {})
        cfg = PhysicalModelConfig(
            model_type="lightgbm",
            province=province,
            lat=coords.get("lat", 30.0),
            lon=coords.get("lon", 104.0),
        )

        merged = self._merge_weather(price_data, weather)
        if merged is None or len(merged) == 0 or "value" not in merged.columns:
            return None

        # 重构物理预测时序
        n = len(merged)

        if solar_model:
            solar_seq = self._predict_on_history(solar_model, weather, price_data, n)
        else:
            solar_seq = np.zeros(n)

        if wind_model:
            wind_seq = self._predict_on_history(wind_model, weather, price_data, n)
        else:
            wind_seq = np.zeros(n)

        if load_model:
            load_seq = load_model.predict(merged.iloc[-n:], solar_seq, wind_seq)
        else:
            load_seq = np.zeros(n)

        net_load_seq = self.net_load.compute(load_seq, solar_seq, wind_seq)

        # 加载 DA price (仅山东有独立日前市场 market_id=1)
        da_price = None
        if "山东" in province:
            try:
                import pymysql
                conn = pymysql.connect(host="127.0.0.1", port=3306, user="root",
                                       password="root123456", database="electric_power_db")
                cursor = conn.cursor()
                province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
                cursor.execute('''
                    SELECT time_point_id, price_value FROM f_price_15min
                    WHERE province = %s AND price_market_id = 1
                    ORDER BY date_key, time_point_id
                ''', (province_cn,))
                rows = cursor.fetchall()
                conn.close()
                if rows and len(rows) >= 24:
                    da_price = np.array([float(r[1]) for r in rows[-n:]], dtype=np.float64)
            except Exception:
                pass

        # 提取价格滞后特征 (来自 feature store)
        lag_1d = merged["value_lag_1d"].values[-n:] if "value_lag_1d" in merged.columns else None
        lag_7d = merged["value_lag_7d"].values[-n:] if "value_lag_7d" in merged.columns else None

        # 去掉 lag 特征为 NaN 的行 (前96行无历史)
        valid_mask = np.ones(n, dtype=bool)
        if lag_1d is not None:
            valid_mask &= ~np.isnan(lag_1d)
        if lag_7d is not None:
            valid_mask &= ~np.isnan(lag_7d)
        if not np.all(valid_mask):
            n_before = n
            merged_valid = merged.iloc[-n:].iloc[valid_mask].reset_index(drop=True)
            net_load_seq = net_load_seq[valid_mask]
            if lag_1d is not None:
                lag_1d = lag_1d[valid_mask]
            if lag_7d is not None:
                lag_7d = lag_7d[valid_mask]
            n = len(merged_valid)
            logger.info("  去除 NaN lag: %d → %d 样本", n_before, n)
        else:
            merged_valid = merged.iloc[-n:]

        model = PriceStructuralModel(cfg)
        model.fit(merged_valid, merged_valid["value"],
                  net_load=net_load_seq, da_price=da_price,
                  lag_1d=lag_1d, lag_7d=lag_7d)
        self._saved_prefixes["price"] = self._save_model(model, province, "price")
        return model

    # ── Tier 4: Transformer ──

    def _train_transformer(self, province, weather, price_data,
                           price_model, load_model, solar_model, wind_model,
                           solar_pred, wind_pred):
        """训练 Transformer 残差修正模型."""
        try:
            from scripts.ml.transformer.trainer import TransformerTrainer
        except ImportError:
            logger.info("PyTorch 未安装, 跳过 Transformer")
            return None

        merged = self._merge_weather(price_data, weather)
        if merged is None or len(merged) < 96 * 7:
            logger.info("数据不足, 跳过 Transformer")
            return None

        n = len(merged)

        # 重构物理预测
        if solar_model:
            solar_seq = self._predict_on_history(solar_model, weather, price_data, n)
        else:
            solar_seq = np.zeros(n)
        if wind_model:
            wind_seq = self._predict_on_history(wind_model, weather, price_data, n)
        else:
            wind_seq = np.zeros(n)
        if load_model:
            load_seq = load_model.predict(merged.iloc[-n:], solar_seq, wind_seq)
        else:
            load_seq = np.zeros(n)

        net_load_seq = self.net_load.compute(load_seq, solar_seq, wind_seq)

        # DA price
        da_price = None
        if "山东" in province:
            try:
                import pymysql
                conn = pymysql.connect(host="127.0.0.1", port=3306, user="root",
                                       password="root123456", database="electric_power_db")
                cursor = conn.cursor()
                province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
                cursor.execute('''
                    SELECT time_point_id, price_value FROM f_price_15min
                    WHERE province = %s AND price_market_id = 1
                    ORDER BY date_key, time_point_id
                ''', (province_cn,))
                rows = cursor.fetchall()
                conn.close()
                if rows and len(rows) >= 24:
                    da_price = np.array([float(r[1]) for r in rows[-n:]], dtype=np.float64)
            except Exception:
                pass

        lag_1d = merged["value_lag_1d"].values[-n:] if "value_lag_1d" in merged.columns else None
        lag_7d = merged["value_lag_7d"].values[-n:] if "value_lag_7d" in merged.columns else None

        # 过滤 NaN
        valid = np.ones(n, dtype=bool)
        if lag_1d is not None:
            valid &= ~np.isnan(lag_1d)
        if lag_7d is not None:
            valid &= ~np.isnan(lag_7d)
        n_valid = int(np.sum(valid))
        if n_valid < 96 * 7:
            logger.info("有效数据不足 (%d), 跳过 Transformer", n_valid)
            return None

        # 生成 LGBM 预测 (使用最后 20% 做验证, 避免全量过拟合)
        split = int(n_valid * 0.8)
        idx = np.where(valid)[0]
        train_idx = idx[:split]
        tf_idx = idx[split:]  # Transformer 训练数据 (out-of-sample)

        if len(tf_idx) < 96 * 3:
            logger.info("Transformer 训练数据不足 (%d), 跳过", len(tf_idx))
            return None

        # 用前80%数据重训 LGBM, 在后20%上生成 out-of-sample 预测
        import lightgbm as lgb
        y_raw = merged["value"].values.astype(float)
        X_all = price_model._build_price_features(
            net_load_seq, merged.iloc[-n:], da_price, lag_1d, lag_7d)

        tf_lgbm = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.02,
            min_child_samples=30, random_state=42, verbose=-1)
        tf_lgbm.fit(X_all[train_idx], y_raw[train_idx])
        phys_pred_full = tf_lgbm.predict(X_all[tf_idx])
        actuals_full = y_raw[tf_idx]
        dts_full = merged["dt"].values[-n:][tf_idx]

        # 训练 Transformer
        logger.info("Transformer 训练: %d 样本", len(tf_idx))
        tf_trainer = TransformerTrainer(
            d_model=64, n_heads=4, n_layers=2,
            seq_len=96, horizon=96, dropout=0.1,
            lr=0.001, batch_size=16, epochs=30, patience=8)
        tf_trainer.fit(phys_pred_full, actuals_full, dts_full, val_split=0.15)

        # 保存
        base_dir = os.path.join(self.model_dir, province)
        os.makedirs(base_dir, exist_ok=True)
        tf_path = os.path.join(base_dir, f"{self._saved_prefixes.get('price', 'price')}_transformer.pt")
        tf_trainer.save(tf_path)
        self._saved_prefixes["transformer"] = tf_path

        # 评估
        residuals = tf_trainer.predict_residuals(
            phys_pred_full[-192:-96], actuals_full[-192:-96],
            phys_pred_full[-96:])
        corrected = phys_pred_full[-96:] + residuals
        final_mae = float(np.mean(np.abs(corrected - actuals_full[-96:])))
        lgbm_mae = float(np.mean(np.abs(phys_pred_full[-96:] - actuals_full[-96:])))
        logger.info("Transformer 完成: LGBM_MAE=%.1f → Corrected_MAE=%.1f", lgbm_mae, final_mae)

        return {"lgbm_mae": lgbm_mae, "corrected_mae": final_mae}

    # ── 辅助 ──

    def _merge_weather(self, data, weather):
        if data is None or weather is None:
            return data
        if "dt" not in data.columns or "dt" not in weather.columns:
            return data

        data_dt = pd.to_datetime(data["dt"])
        weather_dt = pd.to_datetime(weather["dt"])

        merged = data.copy()
        weather_cols = ["temperature", "humidity", "wind_speed", "wind_direction",
                        "solar_radiation", "cloud_factor", "precipitation", "pressure",
                        "CDD", "HDD", "THI", "temp_extremity", "temp_zscore",
                        "is_heat_wave", "is_cold_snap", "extreme_weather_flag"]

        # 按最近时间对齐
        for col in weather_cols:
            if col in weather.columns:
                weather_vals = weather.set_index("dt")[col].reindex(
                    data_dt, method="nearest").values
                merged[col] = weather_vals

        return merged

    def _predict_on_history(self, model, weather, data_or_len, n=None):
        """在历史数据上生成物理预测序列."""
        if isinstance(data_or_len, pd.DataFrame):
            merged = self._merge_weather(data_or_len, weather)
            if n is not None:
                merged = merged.iloc[-n:]
            if merged is not None and len(merged) > 0:
                return model.predict(merged)
            return np.zeros(n or len(data_or_len))
        return np.zeros(n or data_or_len)

    def _save_model(self, model, province, model_type):
        base_dir = os.path.join(self.model_dir, province)
        os.makedirs(base_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{model_type}_{ts_str}"
        path = os.path.join(base_dir, f"{prefix}.lgbm")
        model.save(path)
        return prefix

    def _update_registry(self, province, solar_model, wind_model, load_model,
                         price_model=None):
        reg_path = os.path.join(self.model_dir, "model_registry.json")
        reg = {}
        if os.path.exists(reg_path):
            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)

        if "hybrid" not in reg:
            reg["hybrid"] = {}

        for model_type, model, saved_prefix in [
            ("solar", solar_model, self._saved_prefixes.get("solar")),
            ("wind", wind_model, self._saved_prefixes.get("wind")),
            ("load", load_model, self._saved_prefixes.get("load")),
            ("price", price_model, self._saved_prefixes.get("price")),
        ]:
            if model is None:
                continue
            key = f"{province}_{model_type}"
            entry = {
                "model_type": model.config.model_type,
                "model_prefix": saved_prefix or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "params": model.get_params(),
                "metrics": model.metrics,
                "updated_at": datetime.now().isoformat(),
            }
            if model_type == "price":
                tf_path = self._saved_prefixes.get("transformer")
                if tf_path:
                    entry["transformer_path"] = tf_path
            reg["hybrid"][key] = entry

        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2, ensure_ascii=False)

        logger.info("Registry 已更新: %s", province)
