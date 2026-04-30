---
name: energy-improve
description: 自我优化层 — N假说竞技场 + 策略知识库 + 自动部署
---

# 自我优化层 Skill

被主 Skill `energy-predict` 调用。提供自主学习优化能力。

## 工作空间

`/Users/pcy/analysSkills`

## 能力列表

### 触发一轮自主优化

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
# 先执行验证以触发优化
result = o.run_validation_cycle('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

### 查看策略知识库

```bash
python3 -c "
from src.evolve.improver import Improver
from src.core.db import DorisDB

imp = Improver(DorisDB())
knowledge = imp.query_knowledge('summer')
if knowledge:
    for k in knowledge[:10]:
        rate = k['success_count']/max(k['applied_count'],1)*100
        print(f'{k[\"strategy_desc\"]}: 成功率={rate:.0f}%, avg_imp={k[\"avg_improvement\"]:.4f}')
else:
    print('知识库为空')
"
```

### 退役失效策略

```bash
python3 -c "
from src.core.db import DorisDB
db = DorisDB()
if db.table_exists('strategy_knowledge'):
    # 查看失败次数过多的策略
    df = db.query('''
        SELECT strategy_desc, applied_count, success_count, avg_improvement
        FROM strategy_knowledge
        WHERE applied_count - success_count >= 5 AND NOT retired
    ''')
    if not df.empty:
        print('以下策略连续失败过多，自动退役中:')
        for _, row in df.iterrows():
            print(f'  {row[\"strategy_desc\"]}: 失败{row[\"applied_count\"]-row[\"success_count\"]}次')
    else:
        print('没有需要退役的策略')
else:
    print('strategy_knowledge 表不存在')
"
```
