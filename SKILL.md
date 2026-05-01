---
name: energy-predict
description: Use when users ask for electricity load forecasting, power output prediction, electricity price estimation, energy model training, backtesting, or self-improvement for Chinese provincial power grids.
---

# 电力预测 Skill

LightGBM 分位数回归 + 自进化闭环。覆盖 7 省 × 3 类型（出力/负荷/电价）。

## 安装（必读）

> **仅复制 SKILL.md 无法使用。** 本 Skill 依赖 `scripts/` 和 `assets/` 目录中的 Python 代码和配置文件，两者必须与 SKILL.md 在同一目录下。

```bash
git clone https://github.com/p12p32/energy-predict-skill.git
cd energy-predict-skill
bash assets/install.sh
source ~/.zshrc  # 或 ~/.bashrc
```

安装后验证：
```bash
ls scripts/core/config.py && echo "✅ 安装正确" || echo "❌ 缺少 scripts/，请 git clone 整个仓库"
```

## Quick Reference

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

## Environment

所有命令在 `$ENERGY_HOME` 下执行。Python 解释器自动探测（`python3`/`python`/`python3.11`）。

## Common Mistakes

| 错误 | 原因 | 修复 |
|------|------|------|
| `未找到模型` | 没训练就预测 | 先 `/import` → `/predict`（自动训练） |
| `无可用数据` | CSV 缺少 province/type 列 | 确保 CSV 有 `dt,province,type,value` |
| 预测值全相同 | 历史数据 < 96 步 | 至少导入 1 天以上数据 |
| LightGBM 缺编译 | 缺少 C 编译器 | `brew install libomp && pip install lightgbm` |
| 省份名不识别 | 不在 config 列表 | 用 7 省名之一，或编辑 `assets/config.yaml` |
| MAPE 突然升高 | 节假日/极端天气 | 触发 `/improve` 让模型自适应 |

---

## /predict <省份> <类型> <时间范围>

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
result = o.predict('广东', 'load', 24)
for r in result.get('sample', []):
    print(f\"{r['dt']}  P50={r['p50']:,.0f}\")
"
```
"全国" → 遍历 config 省份列表。输出格式化为表格。

## /import <csv路径>

```bash
cd $ENERGY_HOME && python3 -c "from scripts.core.data_source import FileSource; FileSource().import_csv('data/my_data.csv')"
```

## /backtest <省份> <类型>

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
import json; o = Orchestrator()
r = o.run_backtest_cycle('广东', 'load')
print(f'MAPE: {r.get(\"mape\")}'); print(json.dumps(r.get('diagnoses',[]), ensure_ascii=False))
"
```

## /status [省份]

```bash
cd $ENERGY_HOME && python3 -c "
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
cd $ENERGY_HOME && python3 -c "from scripts.orchestrator import Orchestrator; Orchestrator().train_all()"
```

## /improve <省份> <类型>

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
import json; o = Orchestrator()
print(json.dumps(o.run_validation_cycle('广东', 'load'), ensure_ascii=False, indent=2, default=str))
"
```

## /explain <省份> <类型>

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
for f in o.explain('广东', 'load')['feature_importance'][:10]:
    print(f\"  #{f['rank']:2d} {f['feature']:<30s} {f['pct']:5.1f}%\")
"
```

## /validate <省份> <类型>

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
import json; o = Orchestrator()
print(json.dumps(o.validate_data('广东', 'load'), ensure_ascii=False, indent=2, default=str))
"
```

## /rollback <省份> <类型>

```bash
cd $ENERGY_HOME && python3 -c "from scripts.orchestrator import Orchestrator; print(Orchestrator().rollback_model('广东', 'load'))"
```

## /export <省份> <类型> [格式]

```bash
cd $ENERGY_HOME && python3 -c "from scripts.orchestrator import Orchestrator; print(Orchestrator().export('广东', 'load', fmt='csv'))"
```

## /chart <省份> <类型> [小时]

```bash
cd $ENERGY_HOME && python3 -c "from scripts.orchestrator import Orchestrator; print(Orchestrator().chart('广东', 'load', 24))"
```

## /daemon <start|once>

```bash
cd $ENERGY_HOME
python3 scripts/daemon.py --interval 15       # 持续运行
python3 scripts/daemon.py --once               # 单次
python3 scripts/daemon.py --source doris --interval 15
python3 scripts/daemon.py --watch /data/electricity --interval 15
```

## /benchmark

```bash
cd $ENERGY_HOME && python3 tests/benchmark.py
```

---

## Data Flow

```
CSV → FileSource.import_csv() → FeatureEngineer → FeatureStore
     → Trainer.train() → models/*.lgb
     → Predictor.predict() → P10/P50/P90 + ECM 残差修正
     → Validator → 退化检测 → Improver → 重训 → 新模型部署
```

## Sub-Skills

- `references/energy-data.md` — 数据采集/质量/特征
- `references/energy-train.md` — 模型训练/注册/回滚
- `references/energy-predict-exec.md` — 预测执行/集成/导出
- `references/energy-backtest.md` — 回测验证/诊断
- `references/energy-improve.md` — 自主学习/策略知识库
