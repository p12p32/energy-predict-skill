# 电力出力/负荷/电价预测系统 — 设计文档

> 版本: v1.1 | 日期: 2026-04-30 | 状态: 待审

---

## 1. 概述

### 1.1 目标

构建一个**自我进化的多时间尺度电力预测系统**，覆盖出力、负荷、电价三个预测目标。系统通过双循环验证机制持续优化模型，并以 OpenCode Skill 作为交互入口。

### 1.2 核心数据

| 维度 | 说明 |
|------|------|
| 数据库 | Apache Doris |
| 数据粒度 | 15 分钟级 |
| 核心字段 | province, timestamp, type(output/load), value, price |
| 覆盖范围 | 全国多省 |

### 1.3 预测范围

- **短期**: 15min ~ 24h 超短期/日前预测
- **中期**: 1天 ~ 14天
- **长期**: 15天以上趋势预测

---

## 2. 系统架构

### 2.1 脚本体系（6 核心 + 2 辅助 + 1 存储）

```
                    ┌──────────────────────────┐
                    │    orchestrator.py        │  总调度
                    │  (管理循环 A / B / C)      │
                    └──┬───┬────┬───┬───┬──────┘
           ┌──────────┘   │    │   │   └──────────┐
           ▼              ▼    ▼   ▼              ▼
    ┌──────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐
    │trainer   │  │predictor │ │validator │ │backtester│
    │  .py     │◄─│  .py     │ │  .py     │ │  .py     │
    └────┬─────┘  └──────────┘ └────┬─────┘ └──┬──┬────┘
         │              ▲           │           │  │
         │              │           ▼           │  │
         │         ┌──────────┐ ┌──────────┐    │  │
         │         │feature   │ │analyzer  │    │  │
         │         │_store.py │ │  .py     │    │  │
         │         └──────────┘ └────┬─────┘    │  │
         │              ▲            │           │  │
         │         ┌──────────┐      ▼           │  │
         │         │data      │ ┌──────────┐      │  │
         │         │_fetcher  │ │improver  │◄─────┘  │
         │         │  .py     │ │  .py     │─────────┘
         │         └──────────┘ └────┬─────┘   循环C:
         │                           │         improver→trainer
         │              ┌────────────┘         →backtester→improver
         │              ▼
         │       ┌──────────────┐
         │       │strategy_     │  策略知识库 (Doris)
         └──────►│knowledge_base│
                 └──────────────┘

辅助层:
  data_fetcher.py        — 外部数据拉取（气象、煤价、碳价等）
  feature_store.py       — 特征工程 + 特征存储至 Doris

存储层:
  strategy_knowledge_base — 策略效果历史 (Doris 表)
```

### 2.2 各脚本职责

| 脚本 | 职责 | 输入 → 输出 |
|------|------|------------|
| **orchestrator.py** | 总调度器，管理循环 A/B 的执行策略和触发条件 | 配置 → 调度指令 |
| **trainer.py** | 从 Doris 读取训练数据，训练/重训练模型 | 历史数据 + 特征表 → 模型文件 |
| **predictor.py** | 加载模型，产出未来时段预测 | 模型 + 外推特征 → 预测结果写入 Doris |
| **validator.py** | 实时验证：等真实数据到达后对比预测偏差 | 预测值 vs 真实值 → 偏差报告 + 预警 |
| **backtester.py** | 回塑验证：在历史上反复模拟预测，发现模型薄弱点 | 历史窗口 → 回测报告 |
| **analyzer.py** | 诊断误差根因（概念漂移、特征衰减、省份特异性等） | 偏差信号 → 诊断结论 |
| **improver.py** | 学习型优化引擎：生成 N 个改进假设 → 小规模训练 → backtester 打分 → 自动选最优方案 | 诊断结论 → 最优策略 + 部署指令 |
| **data_fetcher.py** | 采集外部数据（气象 API、煤价、碳价等），统一接口 | 外部 API → 原始数据 |
| **feature_store.py** | 特征工程 + 特征持久化至 Doris，trainer/predictor 共享 | 原始数据 → 特征表 |

### 2.3 脚本间调用关系

```
循环 A — 实时验证:
  predictor → (等待真实数据) → validator → analyzer → improver → trainer → (新模型部署)

循环 B — 回塑验证（单向）:
  backtester → analyzer → improver → trainer

循环 C — 自主学习闭环（核心进化引擎）:
  improver(生成N个假设) → trainer(快速训练) → backtester(打分裁决) → improver(选最优+记录知识)
  └──────────────────────── 反馈循环 ────────────────────────┘

外部数据流:
  data_fetcher → feature_store → trainer / predictor
```

---

## 3. 双循环自我增强

### 3.1 循环 A：实时验证

```
触发条件: 新一期真实数据入库（定时 / 数据到达事件）

流程:
1. validator 拉取近期预测记录，与最新真实值对比
2. 计算 MAPE / RMSE / 偏差方向，判断是否超出阈值
3. 若超阈值 → 触发 analyzer → improver → trainer 重训练链
4. 若未超阈值 → 记录指标，本轮结束

阈值策略:
- 短期预测 MAPE > 5%      → 触发
- 中期预测 MAPE > 10%     → 触发
- 长期预测趋势方向错误     → 触发
- 连续 3 轮同方向偏差      → 触发（即使未超阈值）
```

### 3.2 循环 B：回塑验证

```
触发条件:
- 每次重训练完成后自动触发
- 每周定时全量回测
- 季节更替前专项回测
- 节假日前一周围绕节假日窗口回测

回测策略:
┌─────────────────────┬──────────────────────────────────┐
│ 策略                │ 目的                             │
├─────────────────────┼──────────────────────────────────┤
│ 滚动窗口回测        │ 模拟实时部署场景，发现时间衰减    │
│ 省份交叉验证        │ 识别预测薄弱省份                 │
│ 极端天气回放        │ 验证极端气象下模型鲁棒性          │
│ 季节切换测试        │ 春→夏、秋→冬模型能否平滑过渡     │
│ 节假日专项          │ 春节/国庆等特殊时段预测一致性     │
│ 随机窗口采样        │ 避免窗口选择偏差                 │
└─────────────────────┴──────────────────────────────────┘
```

### 3.3 analyzer.py 诊断矩阵

```
信号                                        诊断结论
────────────────────────────────────────────────────────
某省份持续偏高 3 轮以上               → 省份特有因素未建模（装机变化？限电？）
风电出力预测周末系统偏差              → 周末用电结构不同，需加入 day_of_week 交互特征
夏季负荷预测整体偏低                  → 制冷需求非线性，加 temperature² 特征
电价波动期预测衰减                    → 切换到 CatBoost / 增加碳价特征
前后 3 天趋势方向都错了               → 拐点检测失败，减小平滑窗口
连续多轮无超阈值但偏差缓慢扩大        → 概念漂移，触发全量重训练
```

### 3.4 improver.py — 学习型优化引擎

**核心思想**：不做规则判断，而是像科学家一样——提出假设、实验验证、保留最优。

#### 工作流程

```
输入: analyzer 的诊断报告（误差在哪、什么场景下偏、偏多少）

        ┌─────────────────────────────────────────────┐
        │           improver.py 一轮迭代               │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  Step 1: 生成假设池                           │
        │  ┌───────────────────────────────────────┐   │
        │  │ 基于诊断 + 策略知识库，生成 N 个改进假设  │   │
        │  │                                        │   │
        │  │ H1: 加 wind_speed² 特征                 │   │
        │  │ H2: 最近7天样本权重 ×3                  │   │
        │  │ H3: 对高温样本过采样 (oversample=3x)    │   │
        │  │ H4: 切换模型为 CatBoost                 │   │
        │  │ H5: 省份 X 独立建模                     │   │
        │  │ H6: 减小训练窗口至30天                   │   │
        │  │ ...                                    │   │
        │  └───────────────────────────────────────┘   │
        │                                              │
        │  Step 2: 实验执行（并行）                      │
        │  ┌───────────────────────────────────────┐   │
        │  │ 每个假设 → trainer 快速训练 → 候选模型   │   │
        │  │ 可并行调用 trainer，共享数据缓存的读      │   │
        │  └───────────────────────────────────────┘   │
        │                                              │
        │  Step 3: 竞技场打分                           │
        │  ┌───────────────────────────────────────┐   │
        │  │ backtester 用多个窗口对每个候选模型回测   │   │
        │  │ 返回多维度分数矩阵:                       │   │
        │  │                                          │   │
        │  │      整体 夏季 冬季 工作日 周末 高温 春节   │   │
        │  │ H1:  5.2  7.1  3.8  4.5   6.1  9.2  18.3 │   │
        │  │ H2:  4.1  6.2  2.9  3.8   5.0  8.1  15.2 │   │
        │  │ H3:  4.8  5.3  4.1  4.2   5.8  6.9  20.1 │   │
        │  │ ...                                      │   │
        │  └───────────────────────────────────────┘   │
        │                                              │
        │  Step 4: 智能选择                             │
        │  ┌───────────────────────────────────────┐   │
        │  │ 综合评分：分析失败点权重 × 维度分数       │   │
        │  │ 选最优假设 → 全量训练 → 部署上线         │   │
        │  │ 记录"哪个策略在什么场景下有效" → 知识库   │   │
        │  └───────────────────────────────────────┘   │
        │                                              │
        └─────────────────────────────────────────────┘
```

#### 假设生成机制

```
种子策略库（人工初始化，约 20-30 条）:
┌────────────────────────────┬──────────────────────────┐
│ 策略模板                    │ 适用条件                  │
├────────────────────────────┼──────────────────────────┤
│ 加多项式特征 {col}²          │ 非线性残差大              │
│ 加交叉特征 {a}×{b}          │ 某条件下系统偏差           │
│ 近期样本上采样 ×{w}         │ 整体衰减 / 概念漂移        │
│ 减小训练窗口至 {n} 天       │ 长期趋势反转               │
│ 切换模型至 {model}          │ 特定场景当前模型效果差      │
│ {province} 独立建模         │ 省份偏差两极化              │
│ 极端条件样本过采样          │ 极端值预测不准              │
│ 加 day_of_week 交互特征     │ 周末/工作日模式差异大       │
│ 节假日样本过采样 ×{w}       │ 节假日预测不准              │
│ {col} 滑动窗口均值({n}h)    │ 时序平滑度不足              │
│ ...                        │ ...                       │
└────────────────────────────┴──────────────────────────┘

组合与变异:
  - 单一策略直接应用
  - 相关策略组合（如：加 wind² + 极端风速过采样）
  - 参数网格搜索（如：窗口大小尝试 7d/14d/30d/60d/90d）
  - 基于知识库历史成功率过滤低效方向
```

#### 选择策略

```
不是简单选 MAPE 最低的，而是有策略的选择:

1. 主分数: 诊断报告里"问题最严重"维度的 MAPE 降幅
   例：analyzer 说"夏季负荷崩了"→ 夏季 MAPE 权重 ×5

2. 稳健分: 其他维度至少不能变差
   要求: 其他维度 MAPE 不超过基线 ×1.05，否则淘汰

3. 复杂度罚分: 对过于复杂的假设轻微惩罚
   同等效果下优先选简单的（奥卡姆剃刀）

4. 历史一致性: 这个策略在历史上是否稳定有效
   查询 strategy_knowledge_base，排除"上次用了就崩"的策略
```

#### 输出

```
每轮迭代输出:
{
  "round": 7,
  "diagnosis": "广东夏季负荷 MAPE=14.2%，高温日尤其差",
  "hypotheses_tested": 12,
  "best_strategy": {
    "name": "高温日过采样3x + temperature²特征",
    "score_improvement": {
      "summer_mape": {"before": 14.2, "after": 6.8},
      "hot_day_mape": {"before": 22.1, "after": 9.3},
      "overall_mape": {"before": 8.1, "after": 5.4}
    }
  },
  "deployed": true,
  "knowledge_updated": true
}
```

### 3.5 backtester.py — 多维度精细打分

为支撑 improver 的假设验证，backtester 必须提供多粒度的误差分解：

```
每个候选模型回测输出:
{
  "model_id": "guangdong_load_short_h3",
  "overall_mape": 5.4,
  "overall_rmse": 3200,
  "bias_direction": "slightly_low",      # 系统偏差方向

  "by_season": {
    "spring": {"mape": 4.2, "samples": 8760},
    "summer": {"mape": 6.8, "samples": 11040},
    "autumn": {"mape": 4.5, "samples": 8760},
    "winter": {"mape": 5.1, "samples": 10520}
  },

  "by_time_type": {
    "workday":   {"mape": 4.8},
    "weekend":   {"mape": 6.2},
    "holiday":   {"mape": 8.1}
  },

  "by_weather": {
    "hot_day":   {"mape": 9.3,  "threshold": "temp>35"},
    "cold_day":  {"mape": 5.1,  "threshold": "temp<0"},
    "windy_day": {"mape": 7.2,  "threshold": "wind>8m/s"},
    "normal":    {"mape": 3.8}
  },

  "by_hour_bucket": {
    "peak":      {"mape": 5.2,  "hours": "8-12,17-21"},
    "valley":    {"mape": 4.1,  "hours": "0-7,22-23"},
    "flat":      {"mape": 6.5,  "hours": "12-16"}
  },

  "by_province_specific": {
    "before_holiday": {"mape": 12.3},    # 节前
    "after_holiday":  {"mape": 9.8},     # 节后
    "extreme_events": {"mape": 18.2}      # 限电/检修日
  }
}
```

### 3.6 strategy_knowledge_base — 策略记忆

持久化记录每条策略的历史表现，让 improver 越跑越聪明：

```sql
CREATE TABLE strategy_knowledge (
    strategy_hash   VARCHAR(64)  PRIMARY KEY,  -- 策略内容哈希
    strategy_desc   TEXT,                      -- 可读描述
    applied_count   INT,                       -- 应用次数
    success_count   INT,                       -- 成功次数(改善>5%)
    avg_improvement DOUBLE,                    -- 平均 MAPE 改善幅度
    best_scenario   VARCHAR(256),              -- 最有效场景
    worst_scenario  VARCHAR(256),              -- 最无效场景
    last_applied    DATETIME,                  -- 最近使用时间
    last_effect     DOUBLE,                    -- 最近一次效果
    retired         BOOLEAN DEFAULT FALSE      -- 是否退役
);

-- 查询: "哪些策略在夏季高温场景下有效？"
SELECT strategy_desc, avg_improvement
FROM strategy_knowledge
WHERE best_scenario LIKE '%summer%hot%'
  AND success_count / applied_count > 0.6
  AND NOT retired
ORDER BY avg_improvement DESC;
```

---

## 4. 外部数据集成

### 4.1 数据源分类

```
第一类：气象数据（对出力/负荷影响直接且重大）
  温度、风速/风向、太阳辐射、湿度、降水、气压
  来源: 和风天气 / Open-Meteo 等免费 API
  模式: 历史(训练) + 实况(修正) + 预报(预测输入)

第二类：成本数据（电价预测关键因子）
  动力煤价格(秦皇岛/BSPI)、碳交易价格、天然气价格
  来源: 公开数据平台 / 定期抓取
  模式: 定期更新写入 Doris

第三类：政策/基准数据
  各省上网电价基准、新能源补贴政策、电力市场规则
  来源: 发改委/能源局官网 / 手动维护
  模式: 低频更新（季度/年度）
```

### 4.2 data_fetcher.py 设计

```
统一接口:
  def fetch_weather(province, start_date, end_date, mode="historical")
  def fetch_coal_price(start_date, end_date)
  def fetch_carbon_price(start_date, end_date)

内部:
  - 封装各 API 差异，对外统一 DataFrame 格式
  - 自动处理限流、重试、缓存
  - 缺失值插补（线性插值 / 相邻站点补全）
```

### 4.3 特征存储 (feature_store.py)

所有特征一次性计算、落 Doris 特征表，trainer/predictor 共享：

```sql
CREATE TABLE IF NOT EXISTS energy_feature_store (
    dt              DATETIME      NOT NULL,
    province        VARCHAR(32)   NOT NULL,
    -- 气象特征
    temperature     DOUBLE        COMMENT '温度(°C)',
    feels_like      DOUBLE        COMMENT '体感温度(°C)',
    wind_speed      DOUBLE        COMMENT '风速(m/s)',
    wind_direction  DOUBLE        COMMENT '风向(°)',
    solar_radiation DOUBLE        COMMENT '太阳辐射(W/m²)',
    humidity        DOUBLE        COMMENT '相对湿度(%)',
    precipitation   DOUBLE        COMMENT '降水量(mm)',
    pressure        DOUBLE        COMMENT '气压(hPa)',
    -- 成本特征
    coal_price      DOUBLE        COMMENT '动力煤价格(元/吨)',
    carbon_price    DOUBLE        COMMENT '碳交易价格(元/吨)',
    -- 时间特征
    hour            TINYINT       COMMENT '小时(0-23)',
    day_of_week     TINYINT       COMMENT '星期(0-6)',
    day_of_month    TINYINT       COMMENT '日(1-31)',
    month           TINYINT       COMMENT '月(1-12)',
    is_holiday      BOOLEAN       COMMENT '是否节假日',
    is_weekend      BOOLEAN       COMMENT '是否周末',
    season          TINYINT       COMMENT '季节(1-4)',
    -- 滞后特征
    value_lag_1d    DOUBLE        COMMENT '一天前同期值',
    value_lag_7d    DOUBLE        COMMENT '七天前同期值',
    value_rolling_mean_24h DOUBLE  COMMENT '24小时滑动均值',
    -- 差分特征
    value_diff_1d   DOUBLE        COMMENT '日环比变化',
    value_diff_7d   DOUBLE        COMMENT '周环比变化',
    -- 时间戳分区键
    INDEX idx_province_dt (province, dt) USING BITMAP
)
DUPLICATE KEY (dt, province)
PARTITION BY RANGE (dt) ()
DISTRIBUTED BY HASH (province) BUCKETS 16;
```

---

## 5. 模型策略

### 5.1 模型选型

```
预测任务          主模型           备选模型           适用原因
─────────────────────────────────────────────────────────
短期出力/负荷    LightGBM         XGBoost           特征工程驱动，速度快
中期出力/负荷    Prophet          时序 Transformer   趋势+周期分解，节假日感知
长期趋势          Prophet          Holt-Winters      趋势外推
电价               CatBoost         LightGBM          高方差场景，序数特征处理强
```

### 5.2 概率预测

所有预测同时输出 P10 / P50 / P90 分位数：

```
输出格式:
{
  "dt": "2026-05-01 14:00",
  "province": "广东",
  "type": "load",
  "p10": 45200,
  "p50": 48000,
  "p90": 50800
}
```

```
实现方式:
- LightGBM/XGBoost: objective='quantile', alpha=[0.1, 0.5, 0.9]
- CatBoost: loss_function='Quantile:alpha=0.x'
- Prophet: uncertainty_samples + yhat_lower / yhat_upper
```

### 5.3 模型存储

```
models/
├── guangdong/
│   ├── load_shortterm_20260501.lgb     # 广东-负荷-短期
│   ├── output_shortterm_20260501.lgb   # 广东-出力-短期
│   ├── price_20260501.cbm              # 广东-电价
│   └── load_midterm_20260501.prophet   # 广东-负荷-中期
├── yunnan/
│   └── ...
└── model_registry.json                 # 模型索引 + 元信息
```

---

## 6. Skill 层设计

### 6.1 Skill 命令

```
/predict <省份> <类型> <时间范围>  → 预测
/predict 广东 负荷 未来24小时
/predict 云南 出力 下个月
/predict 全国 电价 未来7天

/backtest <范围>                    → 手动触发回测
/backtest 广东 过去30天

/status                            → 查看当前模型状态和最近误差
/status 广东

/retrain <范围>                    → 手动触发重训练
/retrain 全国

/report                            → 生成预测报告（含概率区间、风险提示）
/report 广东 负荷 下周
```

### 6.2 Skill → orchestrator 调用链

```
用户: /predict 广东 负荷 未来24小时
         │
    Skill 解析参数
         │
    orchestrator.predict(province="广东", type="load", horizon="24h")
         │
    predictor.py 加载模型 + 读特征表 + 输出预测
         │
    结果格式化为自然语言 + 图表返回
```

---

## 7. 增强模块分期

### Phase 1 — 基础闭环 + 学习型核心（必须，MVP）
- [ ] 6 核心脚本 + 2 辅助脚本（骨架）
- [ ] Doris 直连读写
- [ ] 短期预测（LightGBM）
- [ ] 循环 A 实时验证（validator → analyzer → improver 基础规则 → trainer）
- [ ] 循环 B 回塑验证（backtester 基础多维度打分）
- [ ] improver 假设生成 + 竞技场选优（种子策略库 20-30 条）
- [ ] strategy_knowledge_base 建表 + 读写
- [ ] Skill 对话入口

### Phase 2 — 外部数据注入
- [ ] 气象数据接入（data_fetcher + feature_store）
- [ ] 节假日/季节特征
- [ ] 成本数据接入（煤价/碳价）
- [ ] 特征表完整填充
- [ ] 带外部特征的训练流水线

### Phase 3 — 多尺度 + 概率预测
- [ ] 中期预测（Prophet）
- [ ] 长期趋势预测
- [ ] 概率预测（P10/P50/P90）
- [ ] 偏差自动补偿
- [ ] backtester 多尺度验证适配

### Phase 4 — 深度进化 + 工程加固
- [ ] improver 策略组合/变异（从种子到自动搜索）
- [ ] 省份间联动建模
- [ ] 多模型集成（ensemble voting）
- [ ] 异常事件自动检测
- [ ] 模型 A/B 线上竞技场（shadow deployment）

---

## 8. 技术约束

| 项目 | 选择 | 原因 |
|------|------|------|
| 语言 | Python 3.10+ | ML 生态成熟 |
| 数据读取 | pymysql / doris-client | Doris 原生连接 |
| ML 框架 | scikit-learn + LightGBM + Prophet | 轻量、可解释 |
| 任务调度 | 定时任务可由 orchestrator 管理，初期支持手动触发 |
| 配置管理 | YAML 配置文件 | 可读性高 |

---

## 9. 待确认事项

- [ ] 气象 API 具体选型（和风天气 vs Open-Meteo）— 调研结果待 librarian 返回
- [ ] 煤价/碳价数据源可行性确认 — 调研结果待 librarian 返回
- [ ] Doris 表 DDL（实际 Table Schema 确认）
- [ ] 省份列表完整确认
