"""batch_predict.py — 批量预测: 所有省份×类型×日期, 输出到指定目录."""
import os, sys, logging
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

SKILL_HOME = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, SKILL_HOME)

from scripts.core.config import get_available_actual_types, get_provinces
from scripts.ml.pipeline import LayeredPipeline

ARGS = sys.argv[1:]
OUTPUT_DIR = ARGS.pop(0) if ARGS and not ARGS[0].startswith("--") else os.path.join(SKILL_HOME, "..", "ai", "claude", "v10")

DATES = ["2026-04-27", "2026-04-28", "2026-04-29"]
VERSION = "v10"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = LayeredPipeline()

for province in sorted(get_provinces()):
    types = get_available_actual_types(province)
    if not types:
        continue
    for target_type in sorted(types):
        for ref_date in DATES:
            fname = f"{province}_{target_type}_{ref_date}.txt"
            fpath = os.path.join(OUTPUT_DIR, fname)

            print(f"[{VERSION}] {province}/{target_type} @ {ref_date} ...", end=" ", flush=True)

            try:
                df = pipeline.predict(province, target_type,
                                      horizon_hours=24,
                                      reference_date=ref_date)
                n = len(df)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                lines = [
                    f"版本: {VERSION}",
                    f"省份: {province}",
                    f"类型: {target_type}",
                    f"日期: {ref_date}",
                    f"预测时点数: {n}",
                    f"生成时间: {ts}",
                    "=" * 72,
                    f"{'时段':<6} {'时间':<25} {'P10':>10} {'P50':>10} {'P90':>10}",
                ]

                for i, row in df.iterrows():
                    idx = i + 1
                    dt_str = str(row["dt"])
                    p10 = row.get("p10", 0)
                    p50 = row.get("p50", 0)
                    p90 = row.get("p90", 0)
                    lines.append(f"{idx:<6} {dt_str:<25} {p10:>10.2f} {p50:>10.2f} {p90:>10.2f}")

                with open(fpath, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")

                print(f"OK ({n} steps)")

            except Exception as e:
                print(f"FAIL: {e}")

print(f"\nDone. Output: {OUTPUT_DIR}")
