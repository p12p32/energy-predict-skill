"""strict_backtest.py — 严格时间序列回测

零数据泄露: 训练集截止到预测日前一天。
扩展窗口: 每7天重训一次，预测下7天。
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from scripts.ml.trainer import Trainer
from scripts.core.config import load_config


class StrictBacktester:
    def __init__(self, trainer: Trainer = None):
        self.trainer = trainer or Trainer()
        cfg = load_config()
        pred_cfg = cfg.get("predictor", {})
        self._min_train_days = pred_cfg.get("lookback_days", 30)
        self._retrain_every_days = 7

    def walk_forward(self, df: pd.DataFrame, province: str,
                     target_type: str, target_col: str = "value",
                     min_train_days: int = None,
                     retrain_every_days: int = None) -> Dict:
        """扩展窗口 walk-forward 回测.

        训练集始终在预测窗口之前，模拟真实生产环境。
        """
        min_train_days = min_train_days or self._min_train_days
        retrain_every_days = retrain_every_days or self._retrain_every_days
        steps_per_day = 96
        min_train_steps = min_train_days * steps_per_day

        df = df.sort_values("dt").reset_index(drop=True)

        if len(df) < min_train_steps + steps_per_day:
            return {"error": f"数据不足: 需要至少 {min_train_days}+1 天"}

        # 用实际行数找首次预测起点(不依赖日历天数, 兼容缺数据场景)
        unique_dates = sorted(df["dt"].dt.date.unique())
        train_cutoff_date = unique_dates[-1]  # fallback
        for d in unique_dates:
            pre_rows = (df["dt"].dt.date < d).sum()
            if pre_rows >= min_train_steps:
                train_cutoff_date = d
                break

        results = []
        train_end_date = train_cutoff_date

        while True:
            # 训练集: train_end_date 之前的所有数据
            train_mask = df["dt"].dt.date < train_end_date
            train_df = df[train_mask]

            # 测试集: train_end_date 之后的 retrain_every_days 天
            test_end_date = train_end_date + timedelta(days=retrain_every_days)
            test_mask = (df["dt"].dt.date >= train_end_date) & (df["dt"].dt.date < test_end_date)
            test_df = df[test_mask]

            if len(test_df) == 0:
                break
            if len(train_df) < min_train_steps:
                train_end_date = test_end_date  # 训练数据不够则前进, 不退出
                continue

            # 训练 LGB
            try:
                lgb_result = self.trainer.quick_train(
                    train_df, province, target_type, target_col
                )
                model = lgb_result["model"]
                feature_names = lgb_result["feature_names"]
            except Exception as e:
                print(f"  [WARN] 训练失败 {train_end_date}: {e}")
                train_end_date = test_end_date
                continue

            # 预测每一天
            for day_offset in range(retrain_every_days):
                day_date = train_end_date + timedelta(days=day_offset)
                day_mask = test_df["dt"].dt.date == day_date
                day_df = test_df[day_mask]

                if len(day_df) == 0:
                    continue

                # 对齐特征
                day_features = day_df[[c for c in feature_names if c in day_df.columns]]
                missing = [c for c in feature_names if c not in day_df.columns]
                for c in missing:
                    day_features[c] = 0.0

                preds = model.predict(day_features[feature_names].values)

                # 光伏残差还原: 模型预测的是 value - value_lag_1d, 加回基线
                if "光伏" in target_type and "value_lag_1d" in day_features.columns:
                    preds = day_features["value_lag_1d"].values + preds

                actuals = day_df[target_col].values

                for dt, pred, act in zip(day_df["dt"], preds, actuals):
                    results.append({
                        "dt": dt,
                        "predicted": float(pred),
                        "actual": float(act),
                        "train_days": len(train_df) // steps_per_day,
                        "test_date": day_date,
                    })

            # 前进
            train_end_date = test_end_date

            # 安全上限
            if train_end_date > unique_dates[-1]:
                break

        if not results:
            return {"error": "无法生成预测"}

        result_df = pd.DataFrame(results)
        metrics = self._compute_metrics(result_df)

        return {
            "province": province,
            "type": target_type,
            "n_predictions": len(result_df),
            "n_test_days": result_df["test_date"].nunique(),
            "test_date_range": [str(result_df["test_date"].min()), str(result_df["test_date"].max())],
            "metrics": metrics,
            "predictions": result_df,
        }

    @staticmethod
    def _compute_metrics(df: pd.DataFrame) -> Dict:
        actual = df["actual"].values
        predicted = df["predicted"].values

        # sMAPE
        denom = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denom > 1e-8
        smape = float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]))

        # MAE
        mae = float(np.mean(np.abs(actual - predicted)))

        # RMSE
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        # 按月的 MAPE
        df_monthly = df.copy()
        df_monthly["month"] = pd.to_datetime(df_monthly["dt"]).dt.to_period("M")
        by_month = {}
        for month, g in df_monthly.groupby("month"):
            a, p = g["actual"].values, g["predicted"].values
            d = (np.abs(a) + np.abs(p)) / 2
            m = d > 1e-8
            if m.sum() >= 10:
                by_month[str(month)] = round(float(np.mean(np.abs(a[m] - p[m]) / d[m])), 4)

        # 区间覆盖率 (P10/P90 用分位数近似)
        # 用预测值和实际值的偏差分布来估计区间
        errors = predicted - actual
        p10_err = np.percentile(errors, 10)
        p90_err = np.percentile(errors, 90)

        # 如果将来有 P10/P90，覆盖率 = mean(actual between p10 and p90)
        coverage = float(np.mean((actual >= predicted + p10_err) &
                                  (actual <= predicted + p90_err)))

        return {
            "smape": round(smape, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "by_month": by_month,
            "interval_coverage_80pct": round(coverage, 4),
        }


def run_full_strict_backtest(types_filter: List[str] = None,
                              min_train_days: int = 60,
                              retrain_every_days: int = 7) -> pd.DataFrame:
    """对所有 province/type 组合运行严格回测."""
    import json, glob, os

    with open("models/model_registry.json") as f:
        reg = json.load(f)

    bt = StrictBacktester()
    all_results = []

    for key, entry in sorted(reg.items()):
        if key == "广东_load":
            continue
        parts = key.split("_", 1)
        province = parts[0]
        target_type = entry.get("type_parts", entry.get("target_type", parts[1]))

        if types_filter and target_type not in types_filter:
            continue

        # 加载特征
        features_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                     ".energy_data", "features")
        pattern = os.path.join(features_dir, f"{province}_{target_type}_*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [SKIP] {province}/{target_type}: 无特征文件")
            continue

        df = pd.read_parquet(files[-1])
        print(f"\n{'='*60}")
        print(f"严格回测: {province}/{target_type} ({len(df)} 行)")

        result = bt.walk_forward(df, province, target_type,
                                 min_train_days=min_train_days,
                                 retrain_every_days=retrain_every_days)

        if "error" in result:
            print(f"  [FAIL] {result['error']}")
            continue

        m = result["metrics"]
        print(f"  测试天数: {result['n_test_days']}, 预测数: {result['n_predictions']}")
        print(f"  sMAPE: {m['smape']:.1%}, MAE: {m['mae']:.2f}, RMSE: {m['rmse']:.2f}")
        print(f"  80%区间覆盖率: {m['interval_coverage_80pct']:.1%}")

        all_results.append({
            "province": province,
            "type": target_type,
            "smape": m["smape"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "coverage_80pct": m["interval_coverage_80pct"],
            "n_predictions": result["n_predictions"],
            "n_test_days": result["n_test_days"],
        })

    summary = pd.DataFrame(all_results)
    if not summary.empty:
        summary = summary.sort_values("smape")
        print(f"\n{'='*80}")
        print(f"总排名 (sMAPE 升序)")
        print(f"{'='*80}")
        print(f'{"省份":5s} {"类型":25s} {"sMAPE":>8s} {"MAE":>10s} {"覆盖":>8s}')
        for _, row in summary.iterrows():
            print(f'{row["province"]:5s} {row["type"]:25s} '
                  f'{row["smape"]:>7.1%} {row["mae"]:>10.2f} {row["coverage_80pct"]:>7.1%}')

    return summary
