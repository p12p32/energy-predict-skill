---
name: energy-backtest
description: 回测验证层 Skill — 实时验证/回塑验证/误差诊断。被主 Skill 调用。
---

## 工作空间

`$ENERGY_HOME`

## 实时验证 (循环A)

```bash
cd $ENERGY_HOME && python3 -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.run_validation_cycle('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

## 回塑验证 (循环B)

```bash
cd $ENERGY_HOME && python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
result = o.run_backtest_cycle('广东', 'load')
print(f\"MAPE: {result.get('mape')}\")
print(f\"按季节: {result.get('by_season', {})}\")
print(f\"诊断: {result.get('diagnoses', [])}\")
"
```

## 诊断解读

severity: critical > high > medium > low
