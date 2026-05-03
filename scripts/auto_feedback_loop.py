"""auto_feedback_loop.py — 版本化自动反馈闭环

目录约定:
  /Users/pcy/ai/claude/v{N}/     ← 预测结果输出
  /Users/pcy/ai/openclaw/v{N}/   ← 用户反馈输入 (evaluation_report.md + improvement_suggestions.md)
  openclaw/v{N}/.processed       ← 标记已处理
  openclaw/v{N}/.skipped         ← 标记跳过

工作流: 检测到 openclaw/v{N} 有新反馈 → 分析建议 → 整改 → push → 预测到 claude/v{N+1}
"""

import os
import sys
import json
from datetime import datetime

OPENCLAW = "/Users/pcy/ai/openclaw"
CLAUDE_DIR = "/Users/pcy/ai/claude"
PROJ_DIR = "/Users/pcy/analysSkills"


def find_unprocessed_version():
    """扫描 openclaw 目录, 返回第一个未处理的版本号. 没有则返回 None."""
    if not os.path.isdir(OPENCLAW):
        return None

    versions = []
    for name in os.listdir(OPENCLAW):
        if name.startswith("v") and os.path.isdir(os.path.join(OPENCLAW, name)):
            try:
                vnum = int(name[1:])
                versions.append(vnum)
            except ValueError:
                continue

    versions.sort()

    for vnum in versions:
        vdir = os.path.join(OPENCLAW, f"v{vnum}")
        processed = os.path.exists(os.path.join(vdir, ".processed"))
        skipped = os.path.exists(os.path.join(vdir, ".skipped"))
        has_report = os.path.exists(os.path.join(vdir, "evaluation_report.md"))
        has_suggestions = os.path.exists(os.path.join(vdir, "improvement_suggestions.md"))

        if not processed and not skipped and (has_report or has_suggestions):
            return vnum

    return None


def mark_processed(vnum: int):
    """标记版本已处理."""
    vdir = os.path.join(OPENCLAW, f"v{vnum}")
    with open(os.path.join(vdir, ".processed"), "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def mark_skipped(vnum: int, reason: str = ""):
    """标记版本已跳过."""
    vdir = os.path.join(OPENCLAW, f"v{vnum}")
    with open(os.path.join(vdir, ".skipped"), "w") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{reason}")


def run_predictions(version: int):
    """运行所有预测并输出到 v{version}/."""
    sys.path.insert(0, PROJ_DIR)
    from scripts.ml.predictor import Predictor

    out_dir = os.path.join(CLAUDE_DIR, f"v{version}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(PROJ_DIR, "models/model_registry.json")) as f:
        reg = json.load(f)

    # 确定日期范围: 最新数据日期 + 接下来3天
    dates = ["2026-04-27", "2026-04-28", "2026-04-29"]
    p = Predictor()
    results = {"ok": 0, "fail": 0}

    for key, entry in sorted(reg.items()):
        if key == "广东_load":
            continue
        parts = key.split("_", 1)
        province = parts[0]
        target_type = entry.get("type_parts", entry.get("target_type", parts[1]))

        for date_str in dates:
            fname = f"{province}_{target_type}_{date_str}.txt"
            fpath = os.path.join(out_dir, fname)

            try:
                result = p.predict(
                    province=province, target_type=target_type,
                    reference_date=date_str, horizon_hours=24,
                )
                n_points = len(result)
                if n_points == 0:
                    raise RuntimeError("空结果")

                lines = [
                    f"版本: v{version}",
                    f"省份: {province}",
                    f"类型: {target_type}",
                    f"日期: {date_str}",
                    f"预测时点数: {n_points}",
                    f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 72,
                    f"{'时段':<6s} {'时间':<22s} {'P10':>10s} {'P50':>10s} {'P90':>10s}",
                ]

                for i, (_, row) in enumerate(result.iterrows()):
                    lines.append(
                        f"{i+1:<6d} {str(row['dt']):<22s} "
                        f"{row['p10']:10.2f} {row['p50']:10.2f} {row['p90']:10.2f}"
                    )

                lines.append("")
                lines.append(f"统计: P50最低={result['p50'].min():.2f}  "
                           f"P50最高={result['p50'].max():.2f}  "
                           f"P50均值={result['p50'].mean():.2f}")

                with open(fpath, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                results["ok"] += 1

            except Exception as e:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(f"v{version} 预测失败: {province}/{target_type} {date_str}\n错误: {e}\n")
                results["fail"] += 1

    return results


if __name__ == "__main__":
    vnum = find_unprocessed_version()
    if vnum is None:
        print(f"[{datetime.now()}] 无待处理版本")
        sys.exit(0)

    print(f"[{datetime.now()}] 发现待处理版本: v{vnum}")
    print(f"反馈文件: {os.path.join(OPENCLAW, f'v{vnum}')}")
    print("请 Claude Code 分析整改建议并实施。")
    print(f"处理后运行: mark_processed({vnum})")
    print(f"预测输出到: v{vnum + 1}")
