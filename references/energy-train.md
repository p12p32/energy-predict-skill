---
name: energy-train
description: Use when training or retraining LightGBM models, checking model status, viewing feature importance, or rolling back model versions.
---

## 工作空间

`$ENERGY_HOME`

## 全量训练

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
o.train_all()
"
```

## 查看模型

```bash
cd $ENERGY_HOME && python3 -c "
import os, json
path = 'models/model_registry.json'
if os.path.exists(path):
    with open(path) as f:
        r = json.load(f)
    for k, v in sorted(r.items()):
        print(f'{k}: {v[\"updated_at\"]} | {len(v[\"feature_names\"])} features')
"
```

## 特征重要性

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
result = o.explain('广东', 'load')
for f in result['feature_importance'][:10]:
    print(f\"  #{f['rank']:2d} {f['feature']:<30s} {f['pct']:5.1f}%\")
"
```

## 版本回滚

```bash
cd $ENERGY_HOME && python3 -c "
from scripts.orchestrator import Orchestrator
o = Orchestrator()
print(o.rollback_model('广东', 'load'))
"
```
