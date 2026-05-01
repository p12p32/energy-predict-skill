---
name: energy-predict
description: Use when users ask for electricity load forecasting, power output prediction, electricity price estimation, energy model training, backtesting, or self-improvement for Chinese provincial power grids.
---

# 电力预测 Skill

LightGBM 分位数回归 + 自进化闭环。覆盖 7 省 × 3 类型（出力/负荷/电价）。

## Setup（自动）

**Before any command, verify the environment. If not set up, do it automatically:**

```bash
# 自动推导 skill 根目录（不依赖环境变量）
SKILL_HOME="$(cd "$(dirname "$0")" && pwd)"

# Check: are we in a proper clone?
if [ ! -f "scripts/core/config.py" ]; then
  echo "❌ 缺少 scripts/ 目录。请先 git clone 整个仓库，不能只复制 SKILL.md："
  echo "   git clone https://github.com/p12p32/energy-predict-skill.git"
  echo "   将整个 energy-predict-skill/ 目录作为 skill 安装"
  exit 1
fi

# Auto-install if not done yet (check by trying to import scripts)
if ! python3 -c "from scripts.core.config import load_config" 2>/dev/null; then
  echo "首次使用，自动安装依赖..."
  bash assets/install.sh
  echo "安装完成"
fi
```

> Skill 目录已包含 `scripts/` + `assets/`。`energy-predict` 命令通过 alias 自动推导路径，不依赖 `ENERGY_HOME` 环境变量。

## Quick Reference

**每个命令执行前，先跑 Setup 自检。** 如果 skill 根目录 未设置或 `scripts/` 缺失，自动执行安装或提示 git clone。

| 命令 | 用途 |
|------|------|
| `/import <csv>` | 导入数据（需 dt, province, type, value 列） |
| `/predict <省> <类型> <小时>` | 预测未来 N 小时 |
| `/status` | 查看已训练模型 |
| `/explain <省> <类型>` | 特征重要性 Top 10 |
| `/backtest <省> <类型>` | 回测验证 + 诊断 |
| `/improve <省> <类型>` | 触发自主优化循环 |
| `/benchmark` | 精度基准测试 |
| `/export <省> <类型> <格式>` | 导出预测 (json/csv) |
| `/chart <省> <类型> [小时]` | ASCII 终端图表 |
| `/validate <省> <类型>` | 数据质量校验 |
| `/retrain [省] [类型]` | 全量重训练 |
| `/rollback <省> <类型>` | 模型版本回滚 |
| `/daemon <start\|once>` | 自动调度引擎 |

**类型映射:** 出力→output, 负荷→load, 电价→price
**时间映射:** N小时→N, N天→N×24, 下周→168

## When to Use

用户提到以下关键词时激活: 预测、出力、负荷、电价、电力预测、回测、模型评估、训练模型、重训练、特征重要性

## When NOT to Use

- 非电力领域的通用时间序列预测
- 数据不含 `dt`, `province`, `type`, `value` 列
- 省份不在 config 列表内（当前: 广东/云南/四川/江苏/山东/浙江/河南）
- 纯粹的数据分析/可视化（用 pandas/matplotlib 更直接）

## 预测维度

电力预测精度取决于特征覆盖度。以下维度重要性递减，**AI 应主动搜索和获取外部数据来丰富特征**：

### 1. 气象（出力 + 负荷核心驱动）

| 变量 | 列名 | 对负荷影响 | 对出力影响 |
|------|------|-----------|-----------|
| 温度 | `temperature` | 空调负荷 ↑↑ (夏>26°C, 冬<18°C) | 光伏效率 ↓ (高温) |
| 湿度 | `humidity` | 体感温度 → 空调负荷 | 火电效率 ↓ |
| 风速 | `wind_speed` | 体感温度 | 风电 P ∝ v³ |
| 太阳辐射 | `solar_radiation` | 照明负荷 ↓ | 光伏主力 |
| 降水量 | `precipitation` | 户外活动 ↓ → 负荷 ↓ | 水电来水 ↑ |
| 气压 | `pressure` | 天气系统指示 | 影响微弱 |

> 系统从以上 6 个基础变量自动派生：`CDD`(制冷度日)、`HDD`(采暖度日)、`THI`(温湿指数)、`wind_power_potential`(风功率 ∝ v³)、`solar_efficiency`(高温衰减)、`temp_change_1h/6h`(温度变化率)、`consecutive_hot_days`(连续高温)。

**数据获取：** 使用 WebSearch 搜索 "{省份} 历史天气数据" 或直接调用 Open-Meteo API (`https://archive-api.open-meteo.com/v1/archive`)，写入 CSV 的对应列。

### 2. 日历（负荷模式核心）

系统自动从 `dt` 列计算：`hour`、`day_of_week`、`is_weekend`、`season`，以及 sin/cos 周期编码。

节假日需外部获取——**春节是中国电力负荷最大扰动**（降幅 30-50%）：

| 影响等级 | 节假日 | 负荷特征 |
|---------|--------|---------|
| 极大 | 春节 | 工业负荷骤降，居民负荷上升 |
| 大 | 国庆 | 类似但幅度小 |
| 中 | 清明、五一、端午、中秋 | 短期波动 |

> 系统从 `dt` 自动计算：`is_holiday`、`is_work_weekend`(调休)、`days_to_holiday`(节前效应)、`days_from_holiday`(节后恢复)、`bridge_day`(桥接日)、`school_holiday`(寒暑假)、`working_day_type`(7 类精细编码)。节假日库已覆盖 2023-2032。

### 3. 经济（电价核心驱动）

| 变量 | 列名 | 影响 |
|------|------|------|
| 煤价 | `coal_price` | 火电边际成本 → 电价 (中国 60%+ 火电) |
| 碳价 | `carbon_price` | 碳排放成本 → 电价 |
| 工业产值 | `industrial_output` | 工业用电 → 负荷 |

> **数据获取：** 煤价可从秦皇岛煤炭网、中国煤炭市场网获取；碳价从全国碳排放权交易市场。

### 4. 电网调度（出力相关）

- 检修计划 → 出力骤降
- 省间联络线 → 出力/负荷变化
- 新能源弃电率 → 实际出力 < 理论出力

### AI 数据获取策略

**不要等用户提供数据。** 阅读上述维度后，主动：

1. **搜索**：用 WebSearch 找公开数据（气象局、能源局、交易中心）
2. **调用 API**：Open-Meteo (免费气象)、EIA (美国能源)、公开数据集
3. **写入 CSV**：将获取的数据按 `dt, province, type, value, [额外列...]` 格式写入
4. **导入**：`/import` 后系统自动处理所有列

> 系统不做任何预设的外部 API 调用。你拿到什么数据，就导入什么。模型自动利用所有列。

## Environment

所有命令在 skill 根目录下执行。Python 解释器自动探测（`python3`/`python`/`python3.11`）。

## Common Mistakes

| 错误 | 原因 | 修复 |
|------|------|------|
| `未找到模型` | 没训练就预测 | 先 `/import` → `/predict`（自动训练） |
| `无可用数据` | CSV 缺少 province/type 列 | 确保 CSV 有 `dt,province,type,value` |
| 预测值全相同 | 历史数据 < 96 步 | 至少导入 1 天以上数据 |
| LightGBM 缺编译 | 缺少 C 编译器 | `brew install libomp && pip install lightgbm` |
| 省份名不识别 | 不在 config 列表 | 用 7 省名之一，或编辑 `assets/config.yaml` |
| MAPE 突然升高 | 节假日/极端天气 | 触发 `/improve` 让模型自适应 |
| 特征只有时序 | 未搜集外部数据 | 读"预测维度"章节，搜索气象/日历/经济数据 |

---

## /predict <省份> <类型> <时间范围>

每次预测前自动跑轻量回测，检查模型健康度。如果 MAPE > 5%，自动告警建议 `/improve`。

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
result = o.predict('广东', 'load', 24)
# 模型健康检查
if result.get('health'):
    h = result['health']
    print(f'模型精度: MAPE={h[\"mape\"]:.2%} ({h[\"status\"]})')
    if h['status'] == 'degraded':
        print('⚠ 精度退化，建议 /improve 广东 load')
print()
for r in result.get('sample', []):
    print(f\"{r['dt']}  P50={r['p50']:,.0f}\")
"
```
"全国" → 遍历 config 省份列表。输出格式化为表格。

## /import <csv路径>

导入 CSV 后返回列清单。AI 自行对比"预测维度"章节判断覆盖：

```bash
python3 -c "from scripts.core.data_source import FileSource; import json; print(json.dumps(FileSource().import_csv('data/my_data.csv'), ensure_ascii=False))"
# → {"status":"ok", "rows":1000, "columns":["dt","province","type","value"], "extra_columns":[]}
```

**AI 收到 columns 后，对照上方"预测维度"表格：**
1. 气象列 (temperature/humidity/wind_speed/solar_radiation/precipitation/pressure) — 哪个缺？
2. 经济列 (coal_price/carbon_price/price) — 哪个缺？
3. 缺什么补什么，补全后重新 `/import`

**可用工具：** `scripts/data/fetcher.py` 提供 Open-Meteo 气象数据获取。AI 也可使用自己的数据源。CSV 是唯一接口。

## /backtest <省份> <类型>

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
import json; o = Orchestrator()
r = o.run_backtest_cycle('广东', 'load')
print(f'MAPE: {r.get(\"mape\")}'); print(json.dumps(r.get('diagnoses',[]), ensure_ascii=False))
"
```

## /status [省份]

```bash
python3 -c "
import os, json
p = 'models/model_registry.json'
if os.path.exists(p):
    with open(p) as f: r = json.load(f)
    for k, v in sorted(r.items()):
        print(f'{k}: {v[\"updated_at\"]} | {len(v[\"feature_names\"])} features | {len(v.get(\"versions\",[]))} versions')
else:
    print('No models yet')
"
```

## /retrain [省份] [类型]

```bash
python3 -c "from scripts.orchestrator import Orchestrator; Orchestrator().train_all()"
```

## /improve <省份> <类型>

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
import json; o = Orchestrator()
print(json.dumps(o.run_validation_cycle('广东', 'load'), ensure_ascii=False, indent=2, default=str))
"
```

## /explain <省份> <类型>

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
for f in o.explain('广东', 'load')['feature_importance'][:10]:
    print(f\"  #{f['rank']:2d} {f['feature']:<30s} {f['pct']:5.1f}%\")
"
```

## /validate <省份> <类型>

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
import json; o = Orchestrator()
print(json.dumps(o.validate_data('广东', 'load'), ensure_ascii=False, indent=2, default=str))
"
```

## /rollback <省份> <类型>

```bash
python3 -c "from scripts.orchestrator import Orchestrator; print(Orchestrator().rollback_model('广东', 'load'))"
```

## /export <省份> <类型> [格式]

```bash
python3 -c "from scripts.orchestrator import Orchestrator; print(Orchestrator().export('广东', 'load', fmt='csv'))"
```

## /chart <省份> <类型> [小时]

```bash
python3 -c "from scripts.orchestrator import Orchestrator; print(Orchestrator().chart('广东', 'load', 24))"
```

## /daemon <start|once>

```bash
python3 scripts/daemon.py --interval 15       # 持续运行
python3 scripts/daemon.py --once               # 单次
python3 scripts/daemon.py --source doris --interval 15
python3 scripts/daemon.py --watch /data/electricity --interval 15
```

## /benchmark

```bash
python3 tests/benchmark.py
```

---

## Data Flow

```
AI 搜索/采集外部数据 → 构造 CSV → /import
  → FileSource.import_csv() → FeatureEngineer → FeatureStore
  → Trainer.train() → models/*.lgb
  → Predictor.predict() → P10/P50/P90 + ECM 残差修正
  → Validator → 退化检测 → Improver → 重训 → 新模型部署
```

**AI 的职责是丰富特征维度；scripts/ 只负责处理，不负责获取外部数据。**

## Sub-Skills

- `references/energy-data.md` — 数据采集/质量/特征
- `references/energy-train.md` — 模型训练/注册/回滚
- `references/energy-predict-exec.md` — 预测执行/集成/导出
- `references/energy-backtest.md` — 回测验证/诊断
- `references/energy-improve.md` — 自主学习/策略知识库
