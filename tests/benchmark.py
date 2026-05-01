"""benchmark.py — 预测精度基准测试

测试流程:
  1. 生成带现实模式的合成数据
  2. 训练模型 + 回测评分
  3. 对比 3 个简单基准
  4. 多维度误差分解
  5. 运行自优化循环
  6. 输出报告

用法: python3 tests/benchmark.py
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.data_source import MemorySource
from src.data.features import FeatureEngineer
from src.ml.trainer import Trainer
from src.evolve.backtester import Backtester
from src.evolve.validator import Validator
from src.evolve.analyzer import Analyzer
from src.evolve.improver import Improver


# ============================================================
# Step 1: 生成合成数据
# ============================================================

def generate_synthetic_data(n_steps: int = 30000, seed: int = 42) -> pd.DataFrame:
    """生成15分钟粒度的合成电力负荷数据.

    包含特征:
      - 日内周期 (午后高峰, 夜间低谷)
      - 周周期 (周末低)
      - 季节周期 (夏季高, 冬季高)
      - 温度-负荷非线性关系 (高温暴增)
      - 节假日效应 (春节大降)
      - 随机噪声
    """
    np.random.seed(seed)
    dates = pd.date_range("2025-01-01", periods=n_steps, freq="15min")
    hour = dates.hour.values.astype(float)
    dow = dates.dayofweek.values.astype(float)
    month = dates.month.values.astype(float)
    day_of_year = dates.dayofyear.values.astype(float)

    # ── 日内模式 ──
    # 凌晨低谷 50, 早高峰+晚高峰 = 双峰
    hour_pattern = (
        30 * np.sin((hour - 6) * np.pi / 12)   # 早高峰
        + 20 * np.sin((hour - 14) * np.pi / 8)  # 晚高峰
    )

    # ── 周模式 ──
    weekend_effect = -15 * np.where(dow >= 5, 1, 0)
    monday_effect = -5 * np.where(dow == 0, 1, 0)

    # ── 季节模式 ──
    season_effect = 20 * np.sin((day_of_year - 15) * 2 * np.pi / 365)

    # ── 温度模拟 ──
    temp = (
        22
        + 15 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        + 5 * np.sin(hour * np.pi / 12)
        + np.random.normal(0, 3, n_steps)
    )

    # ── 温度-负荷关系 ──
    cooling = np.maximum(temp - 28, 0) * 4
    heating = np.maximum(10 - temp, 0) * 3

    # ── 节假日效应 ──
    is_spring = (dates >= "2025-01-28") & (dates <= "2025-02-04")
    holiday_effect = -30 * is_spring.astype(float)

    # ── 合成 ──
    base = 100
    load = (
        base
        + hour_pattern
        + weekend_effect
        + monday_effect
        + season_effect
        + cooling
        + heating
        + holiday_effect
        + np.random.normal(0, 3, n_steps)
    )
    load = np.maximum(load, 20)

    # ── 电价: 负荷驱动 + 随机 ──
    price = 0.30 + 0.003 * (load - base) + np.random.normal(0, 0.02, n_steps)
    price = np.maximum(price, 0.05)

    df = pd.DataFrame({
        "dt": dates,
        "province": "广东",
        "type": "load",
        "value": np.round(load, 2),
        "price": np.round(price, 3),
    })

    # 注入气象 (模拟已有气象数据的情况)
    df["temperature"] = np.round(temp, 1)
    df["humidity"] = np.round(60 + 20 * np.sin(day_of_year * 2 * np.pi / 365) + np.random.normal(0, 5, n_steps), 1)
    df["wind_speed"] = np.round(3 + 2 * np.sin(day_of_year * 2 * np.pi / 30) + np.random.normal(0, 1, n_steps), 1)
    df["wind_direction"] = np.round(180 + 90 * np.sin(day_of_year * np.pi / 30) + np.random.normal(0, 20, n_steps), 1)
    df["solar_radiation"] = np.maximum(0, np.round(300 * np.sin(hour * np.pi / 12) + np.random.normal(0, 30, n_steps), 1))
    df["precipitation"] = np.maximum(0, np.round(np.random.exponential(2, n_steps), 1))
    df["pressure"] = np.round(1013 + np.random.normal(0, 5, n_steps), 1)

    return df


# ============================================================
# Step 2: 基准模型
# ============================================================

def baseline_naive(actuals: np.ndarray, lag: int = 96) -> np.ndarray:
    """永续预测: 预测值 = 一天前同时刻的值."""
    n = len(actuals)
    pred = np.zeros(n)
    for i in range(n):
        pred[i] = actuals[max(0, i - lag)]
    return pred


def baseline_weekly(actuals: np.ndarray, lag: int = 672) -> np.ndarray:
    """上周同期: 预测值 = 七天前同时刻的值."""
    n = len(actuals)
    pred = np.zeros(n)
    for i in range(n):
        pred[i] = actuals[max(0, i - lag)]
    return pred


def baseline_mean(actuals: np.ndarray) -> np.ndarray:
    """均值预测: 所有历史值的均值."""
    mean_val = np.mean(actuals)
    return np.full(len(actuals), mean_val)


def calc_mape(actuals: np.ndarray, predictions: np.ndarray) -> float:
    mask = actuals != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])))


def calc_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actuals - predictions) ** 2)))


# ============================================================
# Step 3-7: 完整测试流程
# ============================================================

def run_benchmark():
    print("=" * 60)
    print("  电力预测系统 — 精度基准测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 生成数据 ──
    print("\n[1/6] 生成合成数据 (30,000步 ≈ 10天, 15分钟粒度)...")
    raw = generate_synthetic_data(n_steps=30000)
    print(f"  数据量: {len(raw)} 行, "
          f"日期范围: {raw['dt'].min()} ~ {raw['dt'].max()}")

    # ── 特征工程 ──
    print("\n[2/6] 特征工程...")
    engineer = FeatureEngineer()
    features = engineer.build_features_from_raw(raw)
    features = features.dropna(subset=["value"])
    print(f"  特征数: {len(features.columns)}, 有效行: {len(features)}")

    # ── 训练模型 ──
    print("\n[3/6] 训练 LightGBM 模型...")
    trainer = Trainer(model_dir="models")
    t0 = time.time()
    train_result = trainer.train(features, "广东", "load")
    elapsed = time.time() - t0
    print(f"  训练完成: {train_result['n_samples']}样本, "
          f"{train_result['n_features']}特征, 耗时{elapsed:.1f}s")

    # ── 回测验证 ──
    print("\n[4/6] 回测验证 (3个滚动窗口 × 14天训练 / 24h测试)...")
    backtester = Backtester(trainer)
    bt_result = backtester.evaluate_model(
        features, train_window_days=14, test_window_hours=24,
        province="广东", target_type="load",
    )
    model_mape = bt_result.get("overall_mape", 0)
    model_rmse = bt_result.get("overall_rmse", 0)
    print(f"  模型 MAPE: {model_mape:.2%}")
    print(f"  模型 RMSE: {model_rmse:.2f}")

    # ── 基准对比 ──
    print("\n[5/6] 基准模型对比...")
    # 用最后 2000 步做对比
    test_values = features["value"].values[-2000:]
    test_n = len(test_values)

    naive_pred = baseline_naive(test_values)[-test_n:]
    weekly_pred = baseline_weekly(test_values)[-test_n:]
    mean_pred = baseline_mean(test_values)

    baselines = {
        "永续(昨天同时刻)": calc_mape(test_values, naive_pred),
        "上周同期": calc_mape(test_values, weekly_pred),
        "历史均值": calc_mape(test_values, mean_pred),
    }

    best_baseline_name = min(baselines, key=baselines.get)
    best_baseline_mape = baselines[best_baseline_name]

    print(f"\n  基准模型 MAPE:")
    for name, mape in baselines.items():
        marker = " ← 最强基准" if name == best_baseline_name else ""
        print(f"    {name:20s}: {mape:7.2%}{marker}")

    # ── 对比 ──
    improvement = best_baseline_mape - model_mape
    rel_improvement = improvement / best_baseline_mape * 100 if best_baseline_mape > 0 else 0
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │ 你的模型 vs 最强基准:                 │")
    print(f"  │   基准: {best_baseline_name:20s} MAPE={best_baseline_mape:.2%} │")
    print(f"  │   你的: {'LightGBM':20s} MAPE={model_mape:.2%} │")
    if improvement > 0:
        print(f"  │   提升: {improvement:.2%} ({rel_improvement:.0f}%){'':>6}│")
    else:
        print(f"  │   ⚠️  比基准差 {abs(improvement):.2%}{'':>14}│")
    print(f"  └──────────────────────────────────────┘")

    # ── 多维度分解 ──
    print("\n  多维度误差分解:")
    by_season = bt_result.get("by_season", {})
    for season, info in sorted(by_season.items()):
        m = info.get("mape")
        if m is not None:
            bar = "█" * min(int(m * 200), 30)
            print(f"    {season:8s}: {m:6.2%} {bar}")

    by_time = bt_result.get("by_time_type", {})
    if by_time:
        wd = by_time.get('workday', {}).get('mape') or 0
        we = by_time.get('weekend', {}).get('mape') or 0
        print(f"    工作日: {wd:6.2%}")
        print(f"    周末:   {we:6.2%}")

    by_hour = bt_result.get("by_hour_bucket", {})
    if by_hour:
        for bucket, info in by_hour.items():
            m = info.get("mape") if info else None
            if m is not None:
                print(f"    {bucket:8s}: {m:6.2%}")

    # ── 自优化测试 ──
    print("\n[6/6] 自优化循环测试...")
    validator = Validator()
    analyzer = Analyzer()
    improver = Improver()

    # 模拟一次"发现偏差 → 自优化"循环
    diagnosis = analyzer.diagnose(bt_result)
    print(f"  诊断: {len(diagnosis)} 条")
    for d in diagnosis[:3]:
        print(f"    [{d.get('severity')}] {d.get('description')}")

    if diagnosis:
        baseline = {"overall_mape": model_mape}
        t0 = time.time()
        improvement_result = improver.improve(
            diagnosis, features.tail(4000), "广东", "load",
            baseline=baseline,
        )
        elapsed = time.time() - t0
        print(f"\n  优化结果 ({elapsed:.1f}s):")
        print(f"    策略: {improvement_result.get('selected_strategy', 'N/A')}")
        print(f"    测试假设: {improvement_result.get('hypotheses_tested', 0)} 个")
        print(f"    MAPE: {improvement_result.get('mape_before', 0):.2%} → "
              f"{improvement_result.get('mape_after', 0):.2%}")
        imp = improvement_result.get("improvement", 0)
        if imp > 0:
            print(f"    ✅ 改善: {imp:.2%}")
        else:
            print(f"    ⚠️  无改善")
    else:
        print("  无诊断结论，跳过优化")
        improvement_result = {}
        imp = 0

    # ── 最终评分 ──
    print("\n" + "=" * 60)
    print("  最终评分")
    print("=" * 60)

    score = 0
    details = []
    imp = improvement_result.get("improvement", 0) if (improvement_result and isinstance(improvement_result, dict)) else 0

    # 评分1: 模型是否超越基准 (0-40分)
    if improvement > 0:
        base_score = min(40, int(rel_improvement * 2))
        score += base_score
        details.append(f"  超越基准: +{base_score}/40 (提升 {rel_improvement:.0f}%)")
    else:
        details.append(f"  未超基准: +0/40")

    # 评分2: 绝对精度 (0-30分)
    if model_mape < 0.03:
        score += 30
        details.append(f"  绝对精度: +30/30 (MAPE<3%, 顶级)")
    elif model_mape < 0.06:
        score += 25
        details.append(f"  绝对精度: +25/30 (MAPE<6%, 良好)")
    elif model_mape < 0.10:
        score += 15
        details.append(f"  绝对精度: +15/30 (MAPE<10%, 可用)")
    else:
        details.append(f"  绝对精度: +0/30 (MAPE>{10}%)")

    # 评分3: 自优化有效 (0-20分)
    if imp > 0.01:
        score += 20
        details.append(f"  自优化: +20/20 (改善>1%)")
    elif imp > 0:
        score += 10
        details.append(f"  自优化: +10/20 (改善>0%)")
    else:
        details.append(f"  自优化: +0/20")

    # 评分4: 稳定性 (0-10分)
    if model_mape < 0.15:
        score += 10
        details.append(f"  稳定性: +10/10 (MAPE<15%)")

    for d in details:
        print(d)
    print(f"\n  总分: {score}/100")

    if score >= 80:
        print("  评级: 🌟 优秀 — 可以部署")
    elif score >= 60:
        print("  评级: ✅ 良好 — 实用级别")
    elif score >= 40:
        print("  评级: ⚠️ 可用 — 需要继续优化")
    else:
        print("  评级: ❌ 需改进 — 检查模型或数据")

    # ── 导出报告 ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_info": {"n_rows": len(raw), "n_features": len(features.columns)},
        "model_metrics": {"mape": model_mape, "rmse": model_rmse},
        "baselines": baselines,
        "best_baseline": {"name": best_baseline_name, "mape": best_baseline_mape},
        "improvement_vs_baseline": {"absolute": improvement, "relative_pct": rel_improvement},
        "multi_dimension": {
            "by_season": {k: v.get("mape") for k, v in (bt_result.get("by_season") or {}).items()},
            "by_time": {k: v.get("mape") for k, v in (bt_result.get("by_time_type") or {}).items()},
        },
        "optimization": improvement_result,
        "score": score,
    }

    report_path = "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  报告已导出: {report_path}")

    return report


if __name__ == "__main__":
    run_benchmark()
