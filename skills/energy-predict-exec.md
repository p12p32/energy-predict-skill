---
name: energy-predict-exec
description: 预测执行层 — LightGBM + 趋势模型集成预测
---

# 预测执行层 Skill

被主 Skill `energy-predict` 调用。执行预测并写入数据库。

## 工作空间

`/Users/pcy/analysSkills`

## 命令

### /predict-exec <省份> <类型> <时间范围>

```
/predict-exec 广东 load 24
```

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
result = o.predict('广东', 'load', 24)

print(f'预测步数: {result[\"n_predictions\"]}')
for r in result.get('sample', []):
    print(f\"  {r['dt']}  P50={r['p50']:,.0f}\")
"

# 多省
for p in ['广东', '云南', '四川']:
    try:
        result = o.predict(p, 'load', 24)
        print(f'{p}: {result[\"n_predictions\"]}步')
    except Exception as e:
        print(f'{p}: 预测失败 - {e}')
"
```

### 输出格式

将 predict 返回的 sample 列表格式化为表格：

```
┌──────────────────────┬──────────┬──────────┬──────────┐
│ 时间                  │ P10       │ P50       │ P90       │
├──────────────────────┼──────────┼──────────┼──────────┤
│ ...                   │ ...       │ ...       │ ...       │
└──────────────────────┴──────────┴──────────┴──────────┘

集成模式: LightGBM(75%) + 趋势模型(25%) 动态加权
```
