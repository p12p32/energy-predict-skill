---
name: energy-predict-exec
description: 预测执行层 Skill — 集成预测+趋势+分位数。被主 Skill 调用。
---

## 工作空间

`$ENERGY_HOME`

## 预测

```bash
cd $ENERGY_HOME && python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
result = o.predict('广东', 'load', 24)
for r in result.get('sample', []):
    print(f\"{r['dt']}  P50={r['p50']:,.0f}\")
"
```

输出格式化为表格。如有 `trend_adjusted: True`，注明"集成预测 (LightGBM + 趋势模型)"。

## ASCII 图表

```bash
cd $ENERGY_HOME && python3 -c "
from src.orchestrator import Orchestrator
print(Orchestrator().chart('广东', 'load', 24))
"
```

## 导出结果

```bash
cd $ENERGY_HOME && python3 -c "
from src.orchestrator import Orchestrator
print(Orchestrator().export('广东', 'load', fmt='csv'))
"
```
