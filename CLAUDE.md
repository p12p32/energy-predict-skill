# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

中国省级电力预测系统，支持 7 省（广东/云南/四川/江苏/山东/浙江/河南）的出力/负荷/电价预测，模型自动进化。数据粒度 15 分钟（96 点/天），特征存储用 Parquet，原始数据 CSV。

## 常用命令

```bash
# 安装
bash assets/install.sh && source ~/.zshrc

# CLI 使用
energy-predict import data/your.csv           # 导入 CSV
energy-predict predict 广东 load 24           # 预测
energy-predict status                         # 模型状态
energy-predict explain 广东 出力_风电          # 特征重要性
energy-predict benchmark                      # 精度测试
energy-predict daemon --once                  # 单次守护运行

# 测试
pytest tests/ -v                              # 全部测试
pytest tests/test_orchestrator.py -v          # 单文件
pytest tests/test_orchestrator.py::TestOrchestrator::test_build_features -v  # 单用例

# Python 直接调用
python -c "from scripts.orchestrator import Orchestrator; o = Orchestrator(); o.setup(); print(o.predict('广东','load',24))"
```

## 架构

### 类型系统

所有类型用三段式 `{base}_{sub}_{value_type}` 表示（如 `出力_风电_实际`），**必须**通过 `config.py` 的 `parse_type()` → `TypeInfo` 解析，不要手动拆分类型字符串。

### 双管线预测

1. **LayeredPipeline**（11 步）: State(二分类) → Level(回归+变换) → Delta(误差修正) → TS(纯时序) → TrendClassify → Fusion(stacking) → VolatilityCalibration → Constraints → 输出
2. **HybridPipeline**（8 步）: 天气 → 光伏物理 → 风电物理 → 负荷分解(STL+LGB) → 净负荷 → 电价ML → Transformer残差 → Constraints → 输出

`config.yaml` 的 `hybrid.pipeline_mode` 控制运行哪个: `"layered"` / `"hybrid"` / `"both"`（双跑对比）。

### 关键调度

- **Orchestrator** (`scripts/orchestrator.py`): 总调度器，提供 `build_features`, `train_all_layered`, `train_all_hybrid`, `predict`, `validate_data`, `export` 方法
- **daemon.py**: cron 入口，每日 6 点运行。先检查数据就绪（按 `data_availability` 延迟配置），再构建特征/训练/预测
- **数据可用性**: `config.yaml` 的 `data_availability` 定义各 value_type 延迟天数（预测=-1, 实际=+1, 结算=+6），优先级: province_overrides > type_overrides > default_delays

### 数据源

FileSource（默认，零依赖）→ MemorySource（测试）→ DorisSource（生产 MySQL/Doris）。通过 `config.yaml` 的 `data_source` 切换。

### Layer 模式

所有 ML 层继承 `BaseLayer`（`scripts/ml/layers/base.py`），统一 `train()/predict()/save()/load()` 接口。模型存为 LightGBM `.lgbm` 格式。

### 特征工程

~200 特征分 9 大类（时间/天气/节假日/lag/rolling/跨类型/电价/预测误差）。特征版本由源文件 hash 控制（`.energy_data/features/.feature_version`），代码变更自动触发全量重建。支持增量追加（30 天重叠安全区）。

### 自进化循环

- Loop A: predict → 等实际值 → validate(sMAPE) → analyze(规则诊断) → retrain
- Loop B: 历史窗口回测 → 多维评分
- Loop C: `auto_feedback_loop.py` 监控外部反馈目录 → 分析建议 → 修复 → 推新预测

## 约定

- 用户界面、类型名、省份名、文档均为中文；代码标识符用英文
- commit 消息中文 + conventional-commit 前缀（feat/fix/refactor）
- 模型按省份存 `models/{省}/`，注册信息在 `models/model_registry.json`
- 运行时数据在 `.energy_data/`（gitignored）
- 测试用 MemorySource 隔离，无外部依赖
- 没有 linter/formatter 配置，保持现有代码风格即可
