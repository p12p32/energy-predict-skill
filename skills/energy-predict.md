---
name: energy-predict
description: 电力出力/负荷/电价预测 Skill。主入口，根据命令路由到对应子 Skill 层。
---

# 电力预测 Skill — 主入口

## 安装

```bash
git clone <repo> && cd energy-predict-skill && bash install.sh && source ~/.zshrc
```

安装后，所有命令行操作通过 `energy-predict` 入口执行。

## Skill 嵌套架构

```
用户命令
  ↓
energy-predict (主Skill, 本文件) — 命令路由
  ├── /predict     → energy-predict-exec (预测执行层)
  ├── /backtest    → energy-backtest (回测验证层)
  ├── /status      → energy-train (模型状态)
  ├── /retrain     → energy-train (训练层)
  ├── /import      → energy-data (数据层)
  ├── /improve     → energy-improve (优化层)
  ├── /explain     → energy-train (特征重要性)
  ├── /export      → energy-predict-exec (导出)
  ├── /chart       → energy-predict-exec (图表)
  ├── /validate    → energy-data (数据校验)
  └── /rollback    → energy-train (版本回滚)
```

源码按功能分层: `core/` | `data/` | `ml/` | `evolve/`

## 触发条件

用户提到以下关键词时激活: 预测、出力、负荷、电价、电力预测、回测、模型评估、训练模型、重训练

## 变量

所有命令在 `$ENERGY_HOME` 目录下执行。安装后自动设置。

---

## /predict <省份> <类型> <时间范围>

解析: 出力→output, 负荷→load, 电价→price。未来N小时→N, 未来N天→N×24, 下周→168, 下个月→720。

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
result = o.predict('广东', 'load', 24)
for r in result.get('sample', []):
    print(f\"{r['dt']}  P50={r['p50']:,.0f}\")
"
```

"全国" → 遍历 config.yaml provinces 列表。输出格式化为表格。

---

## /import <csv路径>

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.core.data_source import FileSource
FileSource().import_csv('data/my_data.csv')
"
```

CSV 必须包含: `dt`, `province`, `type`, `value`。可选: `price`。

---

## /backtest <省份> <类型>

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.run_backtest_cycle('广东', 'load')
print(f\"MAPE: {result.get('mape')}\")
print(f\"诊断: {json.dumps(result.get('diagnoses', []), ensure_ascii=False)}\")
"
```

---

## /status [省份]

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
import os, json
path = 'models/model_registry.json'
if os.path.exists(path):
    with open(path) as f:
        r = json.load(f)
    for k, v in sorted(r.items()):
        versions = len(v.get('versions', []))
        print(f'{k}: {v[\"updated_at\"]} | {len(v[\"feature_names\"])} features | {versions} versions')
else:
    print('尚未训练任何模型')
"
```

---

## /retrain [省份] [类型]

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
o.train_all()
"
```

---

## /improve <省份> <类型>

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.run_validation_cycle('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

---

## /explain <省份> <类型>

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
result = o.explain('广东', 'load')
for f in result['feature_importance'][:10]:
    print(f\"  #{f['rank']:2d} {f['feature']:<30s} {f['pct']:5.1f}%\")
"
```

---

## /rollback <省份> <类型>

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
result = o.rollback_model('广东', 'load')
print(result)
"
```

---

## /validate <省份> <类型>

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.validate_data('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

---

## /export <省份> <类型> [格式]

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
print(o.export('广东', 'load', fmt='csv'))
"
```

---

## /chart <省份> <类型> [小时]

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.orchestrator import Orchestrator
print(Orchestrator().chart('广东', 'load', 24))
"
```

---

## /data-source <file|doris>

配置 daemon 的持续数据来源:

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} -c "
from src.core.config import load_config
print(f\"当前: {load_config().get('data_source', 'file')}\")
"
# 改 config.yaml: data_source: doris → daemon 自动轮询 Doris
# 改 config.yaml: watch_dir: /path/to/data → 监控 CSV 目录
```

---

## /daemon <start|once>

```bash
cd $ENERGY_HOME
# 持续运行 (拉新数据 → 预测 → 验证 → 改进)
${PYTHON:-python3} src/daemon.py --interval 15
# 单次
${PYTHON:-python3} src/daemon.py --once
# Doris
${PYTHON:-python3} src/daemon.py --source doris --interval 15
# CSV 目录
${PYTHON:-python3} src/daemon.py --watch /data/electricity --interval 15
```

---

## /benchmark

```bash
cd $ENERGY_HOME && ${PYTHON:-python3} tests/benchmark.py
```

---

## 数据流

```
CSV → FileSource.import_csv() → FeatureEngineer → FeatureStore
     → Trainer.train() → models/
     → Predictor.predict() → 预测结果 + 写回文件
     → Validator → 退化检测 → Improver → 重训 → 新模型部署
```

## 子 Skill 层

- `energy-data` — 数据采集/质量/特征
- `energy-train` — 模型训练/注册/回滚
- `energy-predict-exec` — 预测执行/集成/导出
- `energy-backtest` — 回测验证/诊断
- `energy-improve` — 自主学习/策略知识库
