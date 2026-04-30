---
name: energy-predict
description: 电力出力/负荷/电价预测系统。支持多时间尺度预测、模型回测、状态查看、重训练、自主优化和报告生成。自动定时 daemon 已内置。
---

# 电力预测 Skill

系统已内置 `src/daemon.py` 自动调度引擎（每15分钟预测→验证→优化闭环），也可以通过以下命令手动交互。

## 触发条件

用户提到以下关键词时激活此 Skill：
- 预测、出力、负荷、电价、电力预测
- 回测、模型评估、模型状态
- 训练模型、重训练、优化模型
- 电力数据、能源预测、电力报告

## 工作空间

所有命令在 `/Users/pcy/analysSkills` 目录执行。

---

## /predict — 执行预测

用户输入格式：
```
/predict <省份> <类型> <时间范围>
/predict 广东 负荷 未来24小时
/predict 云南 出力 下个月
/predict 全国 电价 未来7天
/predict 山东 负荷 下周
```

### 参数解析规则

**省份**：直接匹配 config.yaml 中 provinces 列表（广东、云南、四川、江苏、山东、浙江、河南）。"全国" → 遍历所有省份。

**类型**：
- 出力/output → "output"
- 负荷/load → "load"
- 电价/price → "price"

**时间范围**：
- "未来N小时" → horizon_hours=N
- "未来N天" → horizon_hours=N*24
- "下周" → horizon_hours=168
- "下个月" → horizon_hours=720
- 未指定 → 默认 24

### 执行步骤

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
# 单省
result = o.predict('广东', 'load', 24)
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

如果省份是"全国"，遍历所有省份并汇总。

### 输出格式

将结果格式化为表格输出给用户：

```
广东 负荷预测 (未来24小时)：
┌──────────────────────┬──────────┬──────────┬──────────┐
│ 时间                  │ P10       │ P50       │ P90       │
├──────────────────────┼──────────┼──────────┼──────────┤
│ 2026-05-01 14:00:00   │ 45,200   │ 48,000   │ 50,800   │
│ 2026-05-01 14:15:00   │ 45,100   │ 47,900   │ 50,700   │
│ ...                   │ ...      │ ...      │ ...      │
└──────────────────────┴──────────┴──────────┴──────────┘
```

如果有 trend_adjusted 字段为 True，说明这是 LightGBM + 趋势模型的集成预测。

---

## /backtest — 手动回测

```
/backtest 广东 负荷
```

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
result = o.run_backtest_cycle('广东', 'load')
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
"
```

输出：MAPE、RMSE、按季节/时段的多维度误差分解、诊断结论。

---

## /status — 查看模型状态

```
/status
/status 广东
```

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
import os

# 读取模型注册表
registry_path = os.path.join('models', 'model_registry.json')
if os.path.exists(registry_path):
    with open(registry_path) as f:
        registry = json.load(f)
    for key, info in sorted(registry.items()):
        print(f'{key}: updated={info[\"updated_at\"]}, '
              f'features={len(info[\"feature_names\"])}')
else:
    print('尚未训练任何模型')
"
```

---

## /retrain — 手动重训练

```
/retrain
/retrain 广东 负荷
```

```bash
python3 -c "
from src.orchestrator import Orchestrator
o = Orchestrator()
# 全量训练
o.train_all()
"
```

或单省单类型：

```bash
python3 -c "
from src.orchestrator import Orchestrator
from src.trainer import Trainer
from datetime import datetime, timedelta

o = Orchestrator()
end = datetime.now()
start = end - timedelta(days=90)

df = o.store.load_features('广东', 'load',
    start.strftime('%Y-%m-%d'),
    (end + timedelta(days=1)).strftime('%Y-%m-%d'))
if not df.empty:
    result = o.trainer.train(df, '广东', 'load')
    print(f'训练完成: samples={result[\"n_samples\"]}, features={result[\"n_features\"]}')
else:
    print('无可用数据')
"
```

---

## /improve — 手动触发自主优化

```
/improve 广东 负荷
```

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()
# 先做一次验证以获取触发状态
val_result = o.run_validation_cycle('广东', 'load')
print(json.dumps(val_result, ensure_ascii=False, indent=2, default=str))
"
```

输出：策略名、优化前后 MAPE 对比、知识库更新状态。

---

## /report — 生成完整预测报告

```
/report 广东 负荷 下周
```

综合执行 predict + backtest，输出结构：

```bash
python3 -c "
from src.orchestrator import Orchestrator
import json

o = Orchestrator()

# 预测
pred = o.predict('广东', 'load', 168)
print('=== 预测 ===')
print(f'预测步数: {pred[\"n_predictions\"]}')

# 回测
bt = o.run_backtest_cycle('广东', 'load')
print()
print('=== 回测 ===')
print(f'近期MAPE: {bt.get(\"mape\")}')
print(f'诊断: {bt.get(\"diagnoses\", [])}')
"
```

以自然语言汇总后输出给用户：
- 未来一周预测趋势
- 模型当前精度
- 潜在风险提示
- 是否处于自主优化中

---

## /daemon — 启动自动调度

```
/daemon start
/daemon once
```

```bash
# 持续运行
python3 src/daemon.py --interval 15 &

# 单次运行后退出（测试用）
python3 src/daemon.py --once
```

---

## 错误处理

| 错误 | 处理 |
|------|------|
| `ModuleNotFoundError` | 提示用户: `pip install -r requirements.txt` |
| `pymysql.err.OperationalError` | 提示: "Doris 连接失败，请检查 config.yaml 中的数据库配置" |
| `FileNotFoundError: 未找到模型` | 提示: "尚未训练该省份/类型的模型，请先执行 /retrain" |
| `未找到省份坐标` | 提示: "该省份不在 config.yaml 的 province_coords 中，请添加经纬度" |
| 返回 `no_data` | 提示: "数据库中没有该省份/类型的记录，请检查 energy_raw 表" |
