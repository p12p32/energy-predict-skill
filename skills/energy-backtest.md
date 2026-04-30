---
name: energy-backtest
description: 回测与验证层 — 实时验证、滚动窗口回测、多维度误差诊断
---

# 回测与验证层 Skill

被主 Skill `energy-predict` 调用。提供实时验证和回塑验证能力。

## 工作空间

`/Users/pcy/analysSkills`

## 能力列表

### 实时验证（循环A）

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
result = o.run_validation_cycle('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

### 回塑验证（循环B）

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
result = o.run_backtest_cycle('广东', 'load')
print(f'MAPE: {result.get(\"mape\")}')
print(f'按季节: {json.dumps(result.get(\"by_season\", {}), indent=2)}')
print(f'诊断: {result.get(\"diagnoses\", [])}')
"
```

### 诊断结果解读

输出中 `diagnoses` 字段包含诊断列表，每项结构：
```json
{
  "scenario": "summer",
  "root_cause": "nonlinear_heat_effect",
  "severity": "high",
  "description": "夏季高温导致负荷非线性增长，线性模型捕捉不足",
  "suggested_features": ["temperature²"]
}
```

severity 级别: critical > high > medium > low
