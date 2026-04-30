---
name: energy-predict
description: 电力出力/负荷/电价预测系统 — 主入口。根据用户命令自动路由到对应的子 Skill 层。
---

# 电力预测系统 — 主入口

## 架构

```
用户命令
  ↓
energy-predict (主Skill, 本文件) — 命令路由
  ├── /predict     → energy-predict-exec (预测执行层)
  ├── /status      → energy-train (模型状态)
  ├── /backtest    → energy-backtest (回测验证层)
  ├── /retrain     → energy-train (训练层)
  ├── /data        → energy-data (数据层)
  ├── /improve     → energy-improve (优化层)
  └── /daemon      → 直接启动 daemon.py
```

子 Skill 层:
- `energy-data` — 数据采集 / 质量 / 特征
- `energy-train` — 模型训练 / 注册
- `energy-predict-exec` — 预测执行
- `energy-backtest` — 验证回测 / 诊断
- `energy-improve` — 自主优化 / 知识库

Python 源码:
- `src/core/` — 基础设施 (config, db)
- `src/data/` — 数据层 (fetcher, quality, holidays, features)
- `src/ml/` — ML引擎 (trainer, predictor, trend, executor)
- `src/evolve/` — 进化引擎 (validator, backtester, analyzer, improver)
- `src/orchestrator.py` — 总调度
- `src/daemon.py` — 自动调度引擎

## 触发条件

用户提到以下关键词时激活:
- 预测、出力、负荷、电价、电力预测
- 回测、模型评估、模型状态
- 训练模型、重训练、优化模型
- 电力数据、能源预测、电力报告

## 工作空间

`/Users/pcy/analysSkills`

---

## 命令路由

### /predict <省份> <类型> <时间范围> → energy-predict-exec

参数解析:
- 出力/output → "output" | 负荷/load → "load" | 电价/price → "price"
- "未来N小时" → horizon_hours=N | "未来N天" → N×24 | "下周" → 168 | "下个月" → 720
- "全国" → 遍历 config.yaml 中所有省份

执行(加载 energy-predict-exec 的指令):
```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
# 替换 province/type/hours 为解析后的值
result = o.predict('广东', 'load', 24)
print(f'预测: {result[\"n_predictions\"]}步')
for r in result.get('sample', []):
    print(f'  {r[\"dt\"]}  P50={r[\"p50\"]:,.0f}')
"
```

输出格式化为表格。如有 `trend_adjusted: True`，注明"集成预测 (LightGBM + 趋势模型)"。

---

### /backtest <省份> <类型> → energy-backtest

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.run_backtest_cycle('广东', 'load')
print(f'MAPE: {result.get(\"mape\")}')
print(f'诊断: {json.dumps(result.get(\"diagnoses\", []), ensure_ascii=False, indent=2)}')
"
```

---

### /status [省份] → energy-train

单省: 查 model_registry.json + 最近 MAPE
全国: 遍历所有省份/类型

```bash
python3 -c "
import os, json
path = 'models/model_registry.json'
if os.path.exists(path):
    with open(path) as f:
        registry = json.load(f)
    for k, v in sorted(registry.items()):
        print(f'{k}: updated={v[\"updated_at\"]}, features={len(v[\"feature_names\"])}')
else:
    print('尚未训练任何模型')
"
```

---

### /retrain [省份] [类型] → energy-train

```bash
python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
o.train_all()
"
```

或单省单类型时，参考 energy-train 中的单省训练命令。

---

### /data <操作> <参数> → energy-data

操作:
- `fetch <省份> <日期范围>` — 拉取气象数据
- `quality <省份> <日期>` — 数据质量检测
- `build <省份>` — 构建特征

参考 energy-data 中的对应命令。

---

### /improve <省份> <类型> → energy-improve

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.run_validation_cycle('广东', 'load')
if result.get('status') == 'improved':
    imp = result.get('improvement', {})
    print(f'优化完成: {imp.get(\"selected_strategy\")}')
    print(f'MAPE: {imp.get(\"mape_before\")} → {imp.get(\"mape_after\")}')
else:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

---

### /daemon <start|once>

```bash
# 持续运行
python3 src/daemon.py --interval 15 &

# 单次测试
python3 src/daemon.py --once
```

---

### /import <文件路径>

导入 CSV 数据（无需数据库）:

```bash
python3 -c "
from src.core.data_source import FileSource
fs = FileSource()
fs.import_csv('data/guangdong.csv')
print('导入完成')
"
```

CSV 必须包含列: `dt`(时间戳), `province`(省份), `type`(output/load/price), `value`, `price`(可选)

---

### /validate <省份> <类型>

数据校验:

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.validate_data('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

---

### /explain <省份> <类型>

查看特征重要性:

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json
o = Orchestrator()
result = o.explain('广东', 'load')
for f in result['feature_importance'][:10]:
    print(f\"  #{f['rank']:2d} {f['feature']:<30s} {f['pct']:5.1f}%\")
print(f\"版本数: {len(result.get('model_versions',[]))}\")
"
```

---

### /rollback <省份> <类型>

回滚到上版本:

```bash
python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
result = o.rollback_model('广东', 'load')
print(result)
"
```

---

### /export <省份> <类型> [格式]

导出预测(默认 json):

```bash
python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
result = o.export('广东', 'load', fmt='csv')
print(result)
"
```

---

### /chart <省份> <类型> [小时]

ASCII 折线图:

```bash
python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
print(o.chart('广东', 'load', 24))
"
```

---

## 错误处理

| 错误 | 处理 |
|------|------|
| `ModuleNotFoundError: No module named 'lightgbm'` | 提示安装: `pip install -r requirements.txt` |
| `pymysql.err.OperationalError` | 提示: "Doris 连接失败, 检查 config.yaml 中 doris 配置" |
| `FileNotFoundError: 未找到模型` | 提示: "该模型尚未训练, 请执行 /retrain 广东 load" |
| `ValueError: 未找到省份坐标` | 提示: "请在 config.yaml province_coords 中添加" |
| `no_data` | 提示: "数据库中无记录, 请检查 energy_raw 表" |
