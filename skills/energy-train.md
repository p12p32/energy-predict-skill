---
name: energy-train
description: 模型训练层 — LightGBM 训练、模型注册、模型加载
---

# 训练层 Skill

被主 Skill `energy-predict` 调用。提供模型训练和模型管理能力。

## 工作空间

`/Users/pcy/analysSkills`

## 能力列表

### 全量训练所有省份/类型

```bash
python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
o.train_all()
"
```

### 训练单个省份/类型

```bash
python3 -c "
from src.ml.trainer import Trainer
from src.data.features import FeatureStore
from src.core.db import DorisDB
from datetime import datetime, timedelta

trainer = Trainer()
store = FeatureStore(DorisDB())

end = datetime.now()
start = end - timedelta(days=90)

df = store.load_features('广东', 'load',
    start.strftime('%Y-%m-%d'),
    (end + timedelta(days=1)).strftime('%Y-%m-%d'))

if not df.empty:
    result = trainer.train(df, '广东', 'load')
    print(f'训练完成: {result[\"n_samples\"]}样本, {result[\"n_features\"]}特征')
else:
    print('无可用数据')
"
```

### 查看已训练模型

```bash
python3 -c "
import os, json
path = 'models/model_registry.json'
if os.path.exists(path):
    with open(path) as f:
        registry = json.load(f)
    for k, v in sorted(registry.items()):
        print(f'{k}: {v[\"updated_at\"]} | {len(v[\"feature_names\"])} features')
else:
    print('尚未训练任何模型')
"
```
