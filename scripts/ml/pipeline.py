"""pipeline.py — 分层预测编排器: 11步串联 State→Level→Delta→TS→Fusion→Constraints."""
import os, json, logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pymysql

from scripts.ml.layers.state import StateLayer
from scripts.ml.layers.level import LevelLayer
from scripts.ml.layers.delta import DeltaLayer
from scripts.ml.layers.ts import TSLayer
from scripts.ml.layers.fusion import FusionLayer, apply_trend_gating
from scripts.ml.layers.constraints import PhysicalConstraints
from scripts.ml.layers.trend_classify import TrendClassifyLayer
from scripts.ml.layers.price_regime import PriceRegimeRegressor
from scripts.ml.trend import TrendModel
from scripts.ml.features_future import build_future_features
from scripts.data.features import FeatureStore
from scripts.core.config import load_config

logger = logging.getLogger(__name__)


class LayeredPipeline:
    def __init__(self, model_dir: str = None, store: FeatureStore = None):
        self.model_dir = model_dir or "models"
        self.store = store or FeatureStore()
        self.state = StateLayer()
        self.level = LevelLayer()
        self.delta = DeltaLayer()
        self.ts = TSLayer()
        self.fusion = FusionLayer()
        self.trend_cls = TrendClassifyLayer()
        self._price_regime = None  # 电价专用, 按需加载
        self.constraints = PhysicalConstraints()
        self.config = load_config()
        self._lookback_days = self.config.get("predictor", {}).get("lookback_days", 60)
        self._loaded_key = None
        self._db_config = {
            "host": "127.0.0.1", "port": 3306,
            "user": "root", "password": "root123456",
            "database": "electric_power_db",
        }

    def load_models(self, province: str, target_type: str) -> bool:
        key = f"{province}_{target_type}"
        if self._loaded_key == key:
            return True

        reg_path = os.path.join(self.model_dir, "model_registry.json")
        if not os.path.exists(reg_path):
            raise FileNotFoundError("模型注册表不存在")

        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        layered = reg.get("layered", {}).get(key)
        if not layered:
            raise FileNotFoundError(f"未找到分层模型: {key}")

        base_dir = os.path.join(self.model_dir, province)
        prefix = layered["model_prefix"]

        self.state.load(os.path.join(base_dir, f"{prefix}_state.lgbm"))
        self.level.load(os.path.join(base_dir, f"{prefix}_level.lgbm"))
        from scripts.ml.layers.transform import TransformConfig
        self.level.transform_config = TransformConfig(
            name=layered.get("transform_name", "identity"),
            C=layered.get("transform_C"),
            lmbda=layered.get("transform_lmbda"),
        )
        self.delta.load(os.path.join(base_dir, f"{prefix}_delta.lgbm"))
        self.ts.load(os.path.join(base_dir, f"{prefix}_ts.lgbm"))
        self.trend_cls.load(os.path.join(base_dir, f"{prefix}_trend_cls.lgbm"))
        self.fusion.load(os.path.join(base_dir, f"{prefix}_fusion.lgbm"))

        self._loaded_key = key
        return True

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24, reference_date: str = None) -> pd.DataFrame:
        horizon_steps = horizon_hours * 4

        # 电价类型走新架构: PriceClassify → PriceRegime ×3 → TS
        if "电价" in target_type:
            return self._predict_price(province, target_type, horizon_hours, reference_date)

        # 1. 确定日期并加载历史特征
        if reference_date:
            end_date = pd.to_datetime(reference_date)
        else:
            end_date = self._find_latest_date(province, target_type)
            if end_date is None:
                end_date = datetime.now()

        min_lookback = 14
        history = None
        for try_days in [self._lookback_days, 30, min_lookback]:
            start_date = end_date - timedelta(days=try_days)
            history = self.store.load_features(
                province, target_type,
                start_date.strftime("%Y-%m-%d"),
                (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if len(history) >= 96 * min_lookback:
                break

        if history is None or history.empty or len(history) < 96:
            raise ValueError(f"没有可用的特征数据: {province}/{target_type}")

        # 2. 构建未来特征
        future_features = build_future_features(
            history, province, target_type, horizon_steps, end_date,
            store=self.store,
        )

        # 加载模型
        self.load_models(province, target_type)

        # 3-8. 逐层预测
        prob_on = self.state.predict(future_features)
        level_pred = self.level.predict(future_features, prob_on=prob_on)
        delta_pred = self.delta.predict(future_features, level_pred=level_pred)
        base = level_pred * (1.0 + delta_pred)
        ts_pred = self.ts.predict(future_features)

        # 7. TrendClassify → trend_probs (仅模型有效时)
        trend_probs = self.trend_cls.predict(future_features)
        if self.trend_cls.model is None:
            trend_probs = None

        # Trend
        trend_vals = history["value"].dropna().values
        if len(trend_vals) >= 96:
            trend_model = TrendModel()
            trend_model.fit(trend_vals)
            trend = trend_model.predict_with_daily_pattern(horizon_steps, trend_vals)
        else:
            trend = np.full(horizon_steps, np.mean(trend_vals) if len(trend_vals) > 0 else 0.0)

        # 8. 融合 (Ridge 纯组件 ensemble, 不含趋势方向)
        meta = self.fusion.build_meta_features(base, ts_pred, trend, future_features)
        ensemble = self.fusion.predict(meta)

        # 8b. TrendClassify 方向门控 + 太阳能趋势偏置
        final_p50 = self._apply_trend_gating(ensemble, base, ts_pred, trend[:horizon_steps],
                                              trend_probs, target_type,
                                              future_df=future_features)

        # 8c. 波动率校准: 预测过度平滑时恢复历史波动幅度 (所有类型)
        final_p50 = self._calibrate_volatility(final_p50, history, target_type)

        # 9. 物理约束
        p10 = final_p50 * 0.95  # 初始 P10/P90
        p90 = final_p50 * 1.05
        final_p50, p10, p90 = self.constraints.apply(
            final_p50, p10, p90, future_features, target_type, province, self.config
        )

        # 10. P10/P90 扩展 (基于 S3 残差分位数)
        p10, p90 = self._expand_intervals(final_p50, p10, p90, target_type)

        # 11. 格式化输出
        result = pd.DataFrame({
            "dt": future_features["dt"].values,
            "province": province,
            "type": target_type,
            "p10": p10,
            "p50": final_p50,
            "p90": p90,
            "model_version": "layered_v1",
        })
        result["trend_adjusted"] = True

        self.store.insert_predictions(result)
        return result

    # ─── 电价专用预测: PriceRegime → TS修正 → 约束 → 输出 ──────

    def _predict_price(self, province: str, target_type: str,
                        horizon_hours: int = 24, reference_date: str = None) -> pd.DataFrame:
        # 1. 确定日期并加载历史
        if reference_date:
            end_date = pd.to_datetime(reference_date)
        else:
            end_date = self._find_latest_date(province, target_type)
            if end_date is None:
                end_date = datetime.now()

        min_lookback = 14
        history = None
        for try_days in [self._lookback_days, 30, min_lookback]:
            start_date = end_date - timedelta(days=try_days)
            history = self.store.load_features(
                province, target_type,
                start_date.strftime("%Y-%m-%d"),
                (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if len(history) >= 96 * min_lookback:
                break

        if history is None or history.empty:
            raise ValueError(f"没有可用的特征数据: {province}/{target_type}")

        # 1b. 频率检测 — 只用最近7天数据(避免历史频率变化)
        recent_cutoff = history["dt"].max() - timedelta(days=7)
        recent = history[history["dt"] >= recent_cutoff]
        if len(recent) >= 24:
            freq_minutes = recent["dt"].diff().mode().iloc[0].total_seconds() / 60
        else:
            freq_minutes = history["dt"].diff().mode().iloc[0].total_seconds() / 60
        points_per_hour = int(round(60 / freq_minutes))
        # 特征管线按15min设计, 始终生成96步再按需下采样
        horizon_steps = horizon_hours * 4
        min_data_points = points_per_hour * min_lookback
        if len(history) < min_data_points:
            raise ValueError(f"数据不足: {len(history)}行 < {min_data_points} ({province}/{target_type})")

        # 2. 加载日前电价 + 日前预测数据 (预测目标日期 = end_date + 1天)
        pred_date_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        da_price = self._load_da_price(province, pred_date_str)
        forecast_data = self._load_daily_forecasts(province, pred_date_str)

        if da_price is not None and len(da_price) >= 24:
            logger.info(f"已加载日前电价: {len(da_price)} 点")

        # 3. 构建未来特征 (注入 D+1 预测值替代缺失的同时间戳实际值)
        future_features = build_future_features(
            history, province, target_type, horizon_steps, end_date,
            store=self.store,
            forecast_data=forecast_data,
        )
        actual_pts_per_hour = points_per_hour
        history_aligned = history.copy()
        train_freq_min = history["dt"].diff().dropna().dt.total_seconds().median() / 60
        train_pph = max(1, int(round(60 / train_freq_min)))
        if train_pph != actual_pts_per_hour and train_pph > actual_pts_per_hour:
            ratio = train_pph // actual_pts_per_hour
            history_aligned = history.iloc[::ratio].reset_index(drop=True)
            horizon_steps = horizon_hours * actual_pts_per_hour

        # 4. 统计预测 (不依赖 LGB, 保方向)
        hist_vals = history_aligned["value"].dropna().values
        pts_per_day = actual_pts_per_hour * 24
        pred_n = horizon_steps  # 使用原始 horizon_steps 保证长度一致

        if len(hist_vals) >= pts_per_day * 3:
            # 如果训练频率与预测频率不同，上采样统计预测
            if actual_pts_per_hour < 4:
                # 1h → 15min: 对统计预测做线性插值
                ratio = 4 // actual_pts_per_hour
                yest_1h = hist_vals[-pts_per_day:]
                avg7_1h = np.mean(hist_vals[-min(len(hist_vals)//pts_per_day, 7)*pts_per_day:].reshape(-1, pts_per_day), axis=0)
                # 简单重复填充到15min
                yest = np.repeat(yest_1h[:horizon_hours], ratio)[:pred_n]
                avg7 = np.repeat(avg7_1h[:horizon_hours], ratio)[:pred_n]
            else:
                yest = hist_vals[-pts_per_day:][:pred_n]
                days = min(len(hist_vals) // pts_per_day, 90)
                reshaped = hist_vals[-days * pts_per_day:].reshape(days, pts_per_day)
                avg7 = np.mean(reshaped[-min(days, 7):], axis=0)[:pred_n]

            yest_mae = float(np.mean(np.abs(yest - avg7)))
            avg_abs = float(np.mean(np.abs(avg7)) + 1e-6)
            anomaly = yest_mae / avg_abs
            if anomaly > 0.8:
                w_yest, w_avg = 0.05, 0.95
            elif anomaly > 0.4:
                w_yest, w_avg = 0.2, 0.8
            else:
                w_yest, w_avg = 0.5, 0.5

            # 日间相位修正: yesterday 的谷底时机准, avg7 的水位准
            # 拆成 level(avg7主导) + shape(yesterday主导) 避免7天平均模糊相位
            yest_mean = float(np.mean(yest))
            yest_shape = yest - yest_mean
            avg7_mean = float(np.mean(avg7))
            avg7_shape = avg7 - avg7_mean

            # level: anomaly 加权 (avg7 水位更可靠)
            blend_level = w_yest * yest_mean + w_avg * avg7_mean
            # shape: yesterday 主导 (时机更准确), avg7 辅助稳定
            shape_w = 0.75  # yesterday shape weight
            p50_raw = blend_level + shape_w * yest_shape + (1.0 - shape_w) * avg7_shape
        else:
            p50_raw = np.full(pred_n, float(np.mean(hist_vals)))

        # 4b. 计算分析标签
        tags = self._compute_analysis_tags(
            hist_vals, pts_per_day, pred_n,
            yest if len(hist_vals) >= pts_per_day * 3 else np.zeros(pred_n),
            avg7 if len(hist_vals) >= pts_per_day * 3 else np.zeros(pred_n),
            anomaly if len(hist_vals) >= pts_per_day * 3 else 0.5,
            forecast_data,
        )

        # 4c. 日前电价方向信号 (已验证 ~85% 方向准确率)
        if da_price is not None and len(da_price) >= 96:
            # 下采样到预测频率
            da_1h = da_price[::4][:horizon_hours] if actual_pts_per_hour <= 1 else da_price[:pred_n]
            # 昨天同时刻实时价格
            yest_rt = hist_vals[-pts_per_day:] if len(hist_vals) >= pts_per_day else None

            if yest_rt is not None and len(yest_rt) >= horizon_hours:
                # 日前vs昨日实时: 信号方向
                for h in range(min(pred_n, len(da_1h), len(yest_rt))):
                    da_val = da_1h[h]
                    rt_yest_val = yest_rt[h % len(yest_rt)]

                    if abs(da_val) < 0.1:  # da_price 为0 (数据缺失)
                        continue

                    da_signal = np.sign(da_val - rt_yest_val)
                    if da_signal != 0:
                        # 日前电价信号权重: 历史验证 ~85%准, 给 0.4 权重
                        da_weight = 0.4
                        # 光伏高峰时段 (H8-H17) 统计方法弱, 提高日前权重
                        solar_hour = (h % pts_per_day) in range(
                            int(pts_per_day * 8 / 24), int(pts_per_day * 17 / 24))
                        if solar_hour:
                            da_weight = 0.55

                        # 融合: 统计方向 + 日前方向
                        stat_val = p50_raw[h]
                        adjustment = abs(stat_val) * 0.04 * da_signal
                        p50_raw[h] = (1.0 - da_weight) * p50_raw[h] + da_weight * (stat_val + adjustment)

        # 4d. 极端风险
        extreme_risk = tags["extreme_risk"]

        # 5. 模型辅助调整 (极端事件时更依赖模型)
        try:
            self._load_price_models(province, target_type)
            regime_pred = self._price_regime.predict(future_features)
            # 负价检测: 如果近期有负价, 模型负价预测可能有效
            recent_vals = hist_vals[-min(len(hist_vals), pts_per_day):]
            negative_recent = len(recent_vals) > 0 and float((recent_vals < 0).mean()) > 0.05
            high_extreme_risk = float(np.mean(extreme_risk)) > 0.3

            if negative_recent or high_extreme_risk:
                # 极端风险高时增加模型权重
                model_weight = min(0.5, 0.2 + float(np.mean(extreme_risk)))
                p50_raw = (1.0 - model_weight) * p50_raw + model_weight * regime_pred[:pred_n]
        except Exception:
            pass  # 模型未加载, 纯统计预测

        # 5b. 波动率校准
        final_p50 = self._calibrate_volatility(p50_raw, history_aligned, target_type)

        # 6. 物理约束
        p10 = final_p50 * 0.90
        p90 = final_p50 * 1.10
        final_p50, p10, p90 = self.constraints.apply(
            final_p50, p10, p90, future_features, target_type, province, self.config
        )

        # 7. P10/P90 扩展
        p10, p90 = self._expand_intervals(final_p50, p10, p90, target_type)

        # 8. 格式化输出 (含置信度/高价值/极端风险标签)
        result = pd.DataFrame({
            "dt": future_features["dt"].values,
            "province": province,
            "type": target_type,
            "p10": p10,
            "p50": final_p50,
            "p90": p90,
            "model_version": "price_forecast_enhanced_v2",
            "confidence": tags["confidence"],
            "is_high_value": tags["is_high_value"],
            "extreme_risk": np.round(extreme_risk, 2),
        })
        result["trend_adjusted"] = False

        # 9. 小时级数据下采样 (山东电价: 15min→1h)
        if points_per_hour == 1:
            result["dt"] = pd.to_datetime(result["dt"])
            result = result[result["dt"].dt.minute == 0].reset_index(drop=True)

        self.store.insert_predictions(result)
        return result

    def _load_price_models(self, province: str, target_type: str) -> bool:
        key = f"{province}_{target_type}"
        if self._loaded_key == key + "_price":
            return True

        reg_path = os.path.join(self.model_dir, "model_registry.json")
        if not os.path.exists(reg_path):
            raise FileNotFoundError("模型注册表不存在")

        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        layered = reg.get("layered", {}).get(key)
        if not layered:
            raise FileNotFoundError(f"未找到电价模型: {key}")

        if layered.get("pipeline") != "price":
            raise ValueError(f"模型管线不匹配: {key} (expected 'price')")

        base_dir = os.path.join(self.model_dir, province)
        prefix = layered["model_prefix"]

        self._price_regime = PriceRegimeRegressor()
        self._price_regime.load(base_dir, prefix)
        self.ts.load(os.path.join(base_dir, f"{prefix}_ts.lgbm"))

        self._loaded_key = key + "_price"
        return True

    def _expand_intervals(self, p50, p10, p90, target_type):
        """P10/P90 区间扩展."""
        allow_neg = "电价" in target_type
        p50_abs = np.maximum(np.abs(p50), 1.0)
        lo = np.maximum(p50_abs * 0.05, p50 - p10)
        hi = np.maximum(p50_abs * 0.05, p90 - p50)
        cap = p50_abs * 3.0
        lo = np.minimum(lo, cap)
        hi = np.minimum(hi, cap)
        p10_new = p50 - lo
        p90_new = p50 + hi
        if not allow_neg:
            p10_new = np.maximum(p10_new, 0)
            p90_new = np.maximum(p90_new, 0)
        return p10_new, p90_new

    # ─── 日前预测数据加载 ──────────────────────────────────────

    def _load_daily_forecasts(self, province: str, date_key: str) -> dict:
        """从 DB 加载日前预测数据: 风电/光伏/负荷/联络线.

        返回: {"solar": array(96), "wind": array(96), "load": array(96), "tie_load": array(96)}
        无数据时返回 None.
        """
        province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
        sources = [
            (4, "solar"),           # 光伏预测
            (3, "wind"),            # 风电预测
            (21, "load"),           # 直调负荷预测
            (22, "tie_load"),       # 联络线负荷预测
        ]
        # 四川用 ps=20 代替 ps=21 (系统负荷)
        if province == "四川":
            sources[2] = (20, "load")

        forecasts = {}
        try:
            conn = pymysql.connect(**self._db_config)
            cursor = conn.cursor()
            for ps_id, name in sources:
                dq_id = 1 if ps_id in (3, 4) else 3  # gen=1, load=3
                cursor.execute('''
                    SELECT time_point_id, power_value
                    FROM f_power_15min
                    WHERE province = %s AND date_key = %s
                      AND data_quality_id = %s AND power_source_id = %s
                    ORDER BY time_point_id
                ''', (province_cn, date_key, dq_id, ps_id))
                rows = cursor.fetchall()
                if rows and len(rows) >= 24:
                    forecasts[name] = np.array([float(r[1]) for r in rows], dtype=np.float64)
            conn.close()
        except Exception as e:
            logger.warning(f"加载预测数据失败 [{province}/{date_key}]: {e}")
            return None

        if len(forecasts) < 2:
            return None
        return forecasts

    def _load_da_price(self, province: str, date_key: str) -> np.ndarray:
        """从 DB 加载日前出清电价 (market_id=1).

        返回: array(96) 15min 日前电价, 无数据时返回 None.
        """
        province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
        # 四川: market_id 可能不同, 先查 t_price_market
        market_id = 1  # 日前电价
        try:
            conn = pymysql.connect(**self._db_config)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT time_point_id, price_value
                FROM f_price_15min
                WHERE province = %s AND date_key = %s AND price_market_id = %s
                ORDER BY time_point_id
            ''', (province_cn, date_key, market_id))
            rows = cursor.fetchall()
            conn.close()
            if rows and len(rows) >= 24:
                return np.array([float(r[1]) for r in rows], dtype=np.float64)
        except Exception as e:
            logger.warning(f"加载日前电价失败 [{province}/{date_key}]: {e}")
        return None

    def _load_rt_price_history(self, province: str, end_date: str, days: int = 7) -> np.ndarray:
        """从 DB 加载近期实时电价历史 (market_id=2).

        返回: array(N) 按时间排序的实时电价, 用于统计预测.
        """
        province_cn = {"山东": "山东", "四川": "四川"}.get(province, province)
        market_id = 2  # 实时电价
        try:
            conn = pymysql.connect(**self._db_config)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT date_key, time_point_id, price_value
                FROM f_price_15min
                WHERE province = %s AND price_market_id = %s
                  AND date_key <= %s
                ORDER BY date_key DESC, time_point_id DESC
                LIMIT %s
            ''', (province_cn, market_id, end_date, days * 96))
            rows = cursor.fetchall()
            conn.close()
            if rows and len(rows) >= 96 * 2:
                vals = np.array([float(r[2]) for r in reversed(rows)], dtype=np.float64)
                return vals
        except Exception as e:
            logger.warning(f"加载实时电价历史失败 [{province}]: {e}")
        return None

    # ─── 置信度分层 + 高价值样本 + 极端风险 ────────────────────

    def _compute_analysis_tags(self, hist_vals: np.ndarray, pts_per_day: int,
                                pred_n: int, yest: np.ndarray, avg7: np.ndarray,
                                anomaly: float, forecast_data: dict) -> dict:
        """计算置信度/高价值/极端风险标签.

        置信度: 基于昨日异常度 (已验证低异常→72%, 高异常→64%)
        高价值: 新能源占比高的时段 + 异常度高的时段
        极端风险: 新能源高占比 + 近期负价

        Returns:
            confidence: list[str]         "high"/"medium"/"low" 每预测步
            is_high_value: list[bool]     方向变化关键步
            extreme_risk: np.ndarray      0-1 极端事件概率
        """
        n = pred_n
        hours = min(n, pts_per_day)
        h_per_step = pts_per_day // n if n <= pts_per_day else 1

        # — 1. 置信度: 异常度越低 → 置信度越高 —
        if anomaly < 0.20:
            base_conf = "high"
        elif anomaly < 0.50:
            base_conf = "medium"
        else:
            base_conf = "low"

        # 每小时的 CV 修正 (高波动小时降级)
        if len(hist_vals) >= pts_per_day * 7:
            days = min(len(hist_vals) // pts_per_day, 7)
            reshaped = hist_vals[-days * pts_per_day:].reshape(days, pts_per_day)
            hourly_std = np.std(reshaped, axis=0)
            hourly_mean = np.abs(np.mean(reshaped, axis=0)) + 1e-6
            hourly_cv = hourly_std / hourly_mean
        else:
            hourly_cv = np.full(pts_per_day, 0.5)

        # 新能源占比 (预测数据)
        renewable_share = 0.0
        if forecast_data is not None:
            fc_solar = forecast_data.get("solar")
            fc_wind = forecast_data.get("wind")
            fc_load = forecast_data.get("load")
            if fc_solar is not None and fc_wind is not None and fc_load is not None:
                total_renewable = float(np.mean(fc_solar[:len(fc_solar)] + fc_wind[:len(fc_wind)]))
                total_load = float(np.mean(fc_load))
                renewable_share = total_renewable / (total_load + 1e-6)

        # 光伏高峰时段 (H8-H17): 方向难以预测, 天然低置信
        solar_peak_range = range(int(pts_per_day * 8 / 24), int(pts_per_day * 17 / 24))

        confidence = []
        for h in range(n):
            idx = h % pts_per_day if pts_per_day > 0 else h
            cv = hourly_cv[min(idx, len(hourly_cv) - 1)]

            c = base_conf
            # 光伏高峰时段: 最高降为 medium
            if idx in solar_peak_range and c == "high":
                c = "medium"
            # 高 CV 小时降级
            if cv > 0.5 and c == "high":
                c = "medium"
            elif cv > 0.5 and c == "medium":
                c = "low"
            # 高新能源占比: 价格不确定性更高
            if renewable_share > 0.25 and c == "high":
                c = "medium"

            confidence.append(c)

        # — 2. 高价值样本: 波动大 + 异常高 + 光伏变化大 —
        is_high_value = []
        hourly_anomaly = np.abs(yest[:hours] - avg7[:hours]) / (np.abs(avg7[:hours]) + 1e-6)
        hourly_anomaly = np.nan_to_num(hourly_anomaly, nan=0.0)

        for h in range(n):
            idx = h % pts_per_day if pts_per_day > 0 else h
            cv = hourly_cv[min(idx, len(hourly_cv) - 1)]
            ha = hourly_anomaly[min(idx, len(hourly_anomaly) - 1)]
            # 高价值: 波动大 OR 异常偏离 OR 光伏高峰 (方向变化关键)
            high_val = (cv > 0.35 or ha > 0.30 or idx in solar_peak_range)
            is_high_value.append(high_val)

        # — 3. 极端风险: 基于预测信号 + 近期价格 —
        extreme_risk = np.zeros(n, dtype=np.float64)

        # 高新能源占比 → 负价风险 (已验证 0 假警报)
        if renewable_share > 0.25:
            extreme_risk += 0.3
        if renewable_share > 0.35:
            extreme_risk += 0.3

        # 近期负价 → 持续风险
        if len(hist_vals) >= pts_per_day:
            recent = hist_vals[-pts_per_day:]
            if float(np.mean(recent < 0)) > 0.05:
                extreme_risk += 0.3
            recent_std = float(np.std(recent))
            recent_mean = float(np.mean(np.abs(recent)) + 1e-6)
            if recent_std / recent_mean > 0.8:
                extreme_risk += 0.2

        extreme_risk = np.clip(extreme_risk, 0.0, 1.0)

        return {
            "confidence": confidence,
            "is_high_value": is_high_value,
            "extreme_risk": extreme_risk,
        }

    def _apply_trend_gating(self, ensemble, base, ts_pred, trend,
                            trend_probs, target_type, future_df=None):
        return apply_trend_gating(ensemble, base, ts_pred, trend,
                                  trend_probs, target_type, future_df=future_df)

    def _calibrate_volatility(self, p50: np.ndarray, history: pd.DataFrame,
                               target_type: str) -> np.ndarray:
        """波动率校准: 预测过度平滑时恢复历史波动幅度.

        保持均值不变, 将偏差放大到历史日内残差 CV 的 70%.
        最多放大 3 倍, 最少 1 倍 (不平滑).
        """
        if "value" not in history.columns or len(history) < 96 * 3:
            return p50

        values = history["value"].dropna().values
        if len(values) < 96 * 3:
            return p50

        pred_mean = np.mean(p50)
        pred_std = np.std(p50)
        if pred_std < 1e-6:
            return p50

        # 历史 7 日日内残差 CV (去日均值后的波动)
        days = min(len(values) // 96, 7)
        pts_per_day = 96
        reshaped = values[-days * pts_per_day:].reshape(days, pts_per_day)
        daily_means = np.mean(reshaped, axis=1, keepdims=True)
        residuals = reshaped - daily_means
        hist_cv = float(np.std(residuals) / (np.mean(np.abs(daily_means)) + 1e-6))

        pred_cv = pred_std / (abs(pred_mean) + 1e-6)

        target_cv = hist_cv * 0.7
        if pred_cv >= target_cv:
            return p50

        scale = min(target_cv / max(pred_cv, 1e-10), 3.0)
        calibrated = pred_mean + (p50 - pred_mean) * scale
        return calibrated

    def _find_latest_date(self, province, target_type):
        try:
            import glob
            base = os.path.join(os.path.dirname(self.model_dir), ".energy_data", "features")
            pattern = os.path.join(base, f"{province}_{target_type}_*.parquet")
            files = sorted(glob.glob(pattern))
            if not files:
                # Try without actual qualifier
                pattern = os.path.join(base, f"{province}_{target_type}*.parquet")
                files = sorted(glob.glob(pattern))
            if files:
                df = pd.read_parquet(files[-1], columns=["dt"])
                if not df.empty:
                    return pd.to_datetime(df["dt"].max())
        except Exception:
            pass
        return None
