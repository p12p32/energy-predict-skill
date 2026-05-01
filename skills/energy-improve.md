---
name: energy-improve
description: 自我优化层 Skill — N假说竞技场+策略知识库。被主 Skill 调用。
---

## 工作空间

`$ENERGY_HOME`

## 触发优化

```bash
cd $ENERGY_HOME && python3 -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.run_validation_cycle('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

## 查看知识库

```bash
cd $ENERGY_HOME && python3 -c "
from src.evolve.improver import Improver
from src.core.data_source import FileSource
imp = Improver(data_source=FileSource())
knowledge = imp.query_knowledge('summer')
if knowledge:
    for k in knowledge[:10]:
        rate = k.get('success_count',0)/max(k.get('applied_count',0),1)*100
        print(f'{k.get(\"strategy_desc\",\"\")}: 成功率={rate:.0f}%')
else:
    print('知识库为空')
"
```

## 精度基准测试

```bash
cd $ENERGY_HOME && python3 tests/benchmark.py
```
