---
name: energy-predict-exec
description: Use when executing electricity predictions — runs LightGBM quantile models, trend ensemble, and error correction for provincial power data.
---

## 工作空间

所有命令在 skill 根目录下执行，路径自动推导。

## 预测

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
result = o.predict('广东', 'load', 24)
for r in result.get('sample', []):
    print(f\"{r['dt']}  P50={r['p50']:,.0f}\")
"
```

输出格式化为表格。如有 `trend_adjusted: True`，注明"集成预测 (LightGBM + 趋势模型)"。

## ASCII 图表

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
print(Orchestrator().chart('广东', 'load', 24))
"
```

## 导出结果

```bash
python3 -c "
from scripts.orchestrator import Orchestrator
print(Orchestrator().export('广东', 'load', fmt='csv'))
"
```
