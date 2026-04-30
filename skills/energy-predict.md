---
name: energy-predict
description: 电力出力/负荷/电价预测系统。支持多时间尺度预测、模型回测、状态查看、重训练和报告生成。
---

# 电力预测 Skill

## 触发条件
当用户提到以下关键词时激活：
- 预测、出力、负荷、电价、电力预测
- 回测、模型评估
- 训练模型、重训练
- 电力数据、能源预测

## 命令列表

### /predict — 执行预测

```
/predict <省份|全国> <类型> <时间范围>

类型: 出力(output) | 负荷(load) | 电价(price)
时间范围: 未来N小时 | 未来N天 | 下周 | 下个月

示例:
  /predict 广东 负荷 未来24小时
  /predict 云南 出力 下个月
  /predict 全国 电价 未来7天
  /predict 山东 负荷 下周
```

实现：调用 `orchestrator.predict(province, target_type, horizon_hours)`

### /backtest — 手动回测

```
/backtest <省份> <类型> [时间窗口]

示例:
  /backtest 广东 负荷
  /backtest 云南 出力 过去30天
```

实现：调用 `orchestrator.run_backtest_cycle(province, target_type)`

### /status — 查看状态

```
/status [省份]

示例:
  /status           # 查看所有省份
  /status 广东      # 查看广东状态
```

实现：查询 predictions 表 + model_registry.json，返回最近 MAPE、模型更新时间、预测数

### /retrain — 手动重训练

```
/retrain [省份] [类型]

示例:
  /retrain              # 全量重训练
  /retrain 广东 负荷    # 单省单类型
```

实现：调用 `orchestrator.train_all()` 或针对性训练

### /improve — 手动触发优化

```
/improve <省份> <类型>

示例:
  /improve 广东 负荷
```

实现：调用 `orchestrator._run_improvement_cycle(province, target_type)`

### /report — 生成报告

```
/report <省份> <类型> <时间范围>

示例:
  /report 广东 负荷 下周
  /report 全国 电价 下个月
```

实现：调用 predict + backtest，汇总为结构化报告（含概率区间、风险提示）

## 工作流

```
/predict → orchestrator.predict() → predictor.py → 返回结果 + 写入 DB
/backtest → orchestrator.run_backtest_cycle() → backtester + analyzer
/status → 查询 DB + model_registry
/retrain → orchestrator.train_all()
/improve → orchestrator._run_improvement_cycle()
/report → predict + backtest 综合
```
