# 能源预测系统架构

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           数据源层                                        │
│                                                                          │
│  MySQL: f_power_15min          MySQL: f_price_15min      Open-Meteo API  │
│  ┌──────────────────────┐     ┌─────────────────────┐    ┌────────────┐  │
│  │ data_quality_id=1    │     │ price_market_id=1   │    │ temperature│  │
│  │   出力_预测 (D+1)     │     │   日前电价 (D+1)     │    │ humidity   │  │
│  │ data_quality_id=2    │     │ price_market_id=2   │    │ wind_speed │  │
│  │   出力_实际 (D-1)     │     │   实时电价 (D-1)     │    │ radiation  │  │
│  │ data_quality_id=3    │     │ price_market_id=3   │    │ pressure   │  │
│  │   负荷_预测 (D+1)     │     │   中长期             │    └────────────┘  │
│  │ data_quality_id=4    │     │ price_market_id=4   │                    │
│  │   负荷_实际 (D-1)     │     │   日前出清           │                    │
│  └──────────────────────┘     │ price_market_id=5   │                    │
│                                │   实时出清           │                    │
│                                └─────────────────────┘                    │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         数据可用性感知层                                    │
│                                                                          │
│  config.yaml: data_availability                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  default_delays:    预测=-1  实际=1  结算=6                    │       │
│  │  type_overrides:    电价_实时=1  电价_结算=6                    │       │
│  │  province_overrides: {}                                        │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  运行日=4/26 时各类型可用截止:                                              │
│  ┌─────────────────────┬──────────┬────────────────┐                     │
│  │ 数据类型             │ delay    │ 可用截止        │                     │
│  ├─────────────────────┼──────────┼────────────────┤                     │
│  │ 出力_风电_预测       │   -1     │ 4/27 (D+1)     │                     │
│  │ 出力_风电_实际       │   +1     │ 4/25 (D-1)     │                     │
│  │ 负荷_总_预测         │   -1     │ 4/27 (D+1)     │                     │
│  │ 负荷_总_实际         │   +1     │ 4/25 (D-1)     │                     │
│  │ 电价_实时_实际       │   +1     │ 4/25 (D-1)     │                     │
│  │ 电价_日前_预测       │   -1     │ 4/27 (D+1)     │                     │
│  │ 电价_结算            │   +6     │ 4/20 (D-6)     │                     │
│  └─────────────────────┴──────────┴────────────────┘                     │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         daemon.py — 调度入口                              │
│                                                                          │
│  cron: 0 6 * * *                                                        │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ 1. 数据就绪检查 (_check_data_ready)                           │        │
│  │    每种 (province, type) 按 delay 判断是否已到库              │        │
│  │    未就绪 → 跳过并告警                                       │        │
│  │                                                              │        │
│  │ 2. 构建特征 (build_features)                                  │        │
│  │    每种类型独立按 get_available_date() 加载, 不互相拖累       │        │
│  │                                                              │        │
│  │ 3. 训练 (train_all_layered / train_all_hybrid)               │        │
│  │                                                              │        │
│  │ 4. 预测 (predict_layered / predict_hybrid)                    │        │
│  │    就绪的类型执行预测, 未就绪跳过                             │        │
│  └─────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. 特征工程管线

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    build_features_from_raw()                              │
│                                                                          │
│  raw data (CSV/DB)                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ 时间特征      │  │ 天气特征      │  │ 节假日特征    │  │ 周期编码    │ │
│  │ hour/dow/    │  │ temperature  │  │ is_holiday   │  │ hour_sin   │ │
│  │ month/season │  │ wind_speed   │  │ days_to/from │  │ hour_cos   │ │
│  │ is_weekend   │  │ solar_rad    │  │ work_weekend │  │ dow_sin/cos│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 滞后特征 (只在 value_type=实际 上计算)                              │   │
│  │   value_lag_1d  = shift(96)     value_diff_1d  = diff(96)         │   │
│  │   value_lag_7d  = shift(672)    value_diff_7d  = diff(672)        │   │
│  │   value_rolling_mean_24h = rolling(96)                            │   │
│  │   value_rolling_std_24h  = rolling(96).std()                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 交互特征                                                           │   │
│  │   peak_valley  weekend_hour  dow_hour  is_daylight  time_of_day  │   │
│  │   season_x_tod  temp_x_hour  wind_x_season  temp_x_humidity      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 交叉类型特征 v2 (add_cross_type_features_v2)                       │   │
│  │   层1: 同基类子类型交叉 (e.g. 风电出力 → 光伏出力)                 │   │
│  │   层2: 跨基类交叉 (e.g. 输出 → 负荷)                               │   │
│  │   层3: 物理/经济派生 (residual_demand, renewable_penetration)      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 电价专属特征 (add_price_features)                                  │   │
│  │   supply_demand_ratio  renewable_share  price_vol_24h/7d          │   │
│  │   price_momentum_1h/6h/24h  peak_off_peak_spread  load_factor    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 预测误差特征 (add_prediction_error_features, 延迟注入)              │   │
│  │   pred_error  pred_error_bias_24h  pred_error_regime             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3. 双管线预测架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      预测 → 按 pipeline_mode 分发                         │
│                                                                          │
│  predict(province, type, horizon)                                        │
│       │                                                                  │
│       ├── layered  ──────────────────────────────────────►               │
│       │   ┌──────────────────────────────────────────────┐              │
│       │   │         LayeredPipeline (11步)                │              │
│       │   │                                              │              │
│       │   │  load_features(history)                       │              │
│       │   │       │                                      │              │
│       │   │       ▼                                      │              │
│       │   │  build_future_features(history, horizon)     │              │
│       │   │       │                                      │              │
│       │   │       ▼                                      │              │
│       │   │  State(二分类) → Level(回归+变换)             │              │
│       │   │       │              │                       │              │
│       │   │       ▼              ▼                       │              │
│       │   │  Delta(残差修正)    TS(纯时序)               │              │
│       │   │       │              │                       │              │
│       │   │       ▼              ▼                       │              │
│       │   │  TrendClassify → Fusion(Stacking集成)        │              │
│       │   │       │                                      │              │
│       │   │       ▼                                      │              │
│       │   │  波动率校准 → PhysicalConstraints → 输出      │              │
│       │   └──────────────────────────────────────────────┘              │
│       │                                                                  │
│       ├── hybrid  ────────────────────────────────────────►              │
│       │   ┌──────────────────────────────────────────────┐              │
│       │   │         HybridPipeline (8步)                  │              │
│       │   │                                              │              │
│       │   │  气象 → Solar物理 / Wind物理 →                │              │
│       │   │       → Load分解(STL+LGB)                     │              │
│       │   │       → NetLoad = Load - Solar - Wind         │              │
│       │   │       → Price预测(NetLoad+LGB)                │              │
│       │   │       → [Transformer残差修正]                 │              │
│       │   │       → PhysicalConstraints → 输出            │              │
│       │   └──────────────────────────────────────────────┘              │
│       │                                                                  │
│       └── both  ──► 双跑对比, 默认输出 hybrid                            │
└──────────────────────────────────────────────────────────────────────────┘
```

## 4. 分层管线训练流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LayeredTrainer.train()                                 │
│                                                                          │
│  df (全量特征, 按dt排序)                                                  │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 0: TransformSelector → identity/log/asinh/yeo_johnson             │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 2: 时间切分  S1(70%) │ S2(20%) │ S3(10%)                         │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 3:  State 训练 (S1)           二分类: 出力>阈值?                  │
│  Phase 4:  Level 训练 (S1) + OOF    绝对值回归 + 自适应变换              │
│  Phase 5:  Delta 训练 (S1)          相对误差短期修正                     │
│  Phase 6:  TS 训练 (S1)             纯时序LGB                            │
│  Phase 6b: Trend 拟合 (S1)          趋势基线 (多日平均日内模式)          │
│  Phase 6c: TrendClassify 训练 (S1)  趋势方向分类                         │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 7: 前向预测 S2 →         _forward_predict(s1→s2)                  │
│           build_future_features(s1) → 各层预测 → 组件输出                │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 8:  Fusion 训练 (S2)        Stacking第二层 → 融合组件             │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 9:  S3 验证                 前向预测 → MAPE/RMSE/NRMSE            │
│       │                                                                  │
│       ▼                                                                  │
│  保存模型 → models/{province}/{type}_{timestamp}_*.lgbm                  │
│  注册表   → models/model_registry.json                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

## 5. 电价预测专用管线

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    _train_price / _predict_price                          │
│                                                                          │
│  输入: 电价_实时_实际 历史特征                                             │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 0:  频率检测 (最近7天, 15min/1h自适应)                             │
│  Phase 0b: 极端阈值 (山东=0, 其他=800 元/MWh)                             │
│  Phase 2:  时间切分 S1/S2/S3                                             │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 3-5: PriceRegime训练                                              │
│     ├── 三区间分类 (低/中/高/极端)                                        │
│     ├── 每区间独立 LGB 回归                                               │
│     └── 残差集成                                                         │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 6:  TS 层训练 (纯时序)                                             │
│       │                                                                  │
│       ▼                                                                  │
│  Phase 9:  S3 验证                                                       │
│                                                                          │
│  ═══════════════ 预测时额外输入 ═══════════════                           │
│                                                                          │
│  预测流程 (_predict_price):                                                │
│    1. 加载历史特征 (到 get_available_date 截止)                           │
│    2. build_future_features                                              │
│    3. 频率对齐                                                            │
│    4. 统计基线 (昨日+7日均值, anomaly加权)                                │
│    5. 加载 日前电价 (D+1, 四川=None)                                      │
│    6. 加载 日前出力/负荷预测 (D+1)                                        │
│    7. 日前电价方向信号 (85%准确率, 权重0.4)                                │
│    8. PriceRegime → TS修正 → 约束 → 输出                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

## 6. 数据时间线 (以4/26运行预测4/27为例)

```
                    4/20          4/25      4/26      4/27
                     │             │         │         │
  电价_结算          ██████████████▓         │         │    D-6, 最晚可用4/20
                     │             │         │         │
  出力/负荷_实际     │             ██████████▓         │    D-1, 最晚可用4/25
  电价_实时_实际     │             ██████████▓         │    D-1, 最晚可用4/25
                     │             │         │         │
  运行日             │             │         ████████  │    4/26 执行预测
                     │             │         │         │
  出力/负荷_预测     │             │         │  ████████    D+1, 4/27可用
  电价_日前_预测     │             │         │  ████████    D+1, 4/27可用
  气象预报           │             │         │  ████████    4/27可用
                     │             │         │         │
                     ▼             ▼         ▼         ▼

  训练数据范围:  [  ~4/20  ][  ~4/25  ]
                结算截断    实际截断

  预测输入:
    历史特征: 电价实际 to 4/25, 出力/负荷实际 to 4/25
    外部信号: 日前预测 4/27, 日前电价 4/27, 气象 4/27
```

## 7. 代码模块依赖

```
daemon.py ──► orchestrator.py ──► ml/pipeline.py ──► ml/layers/*
                          │        (LayeredPipeline)   ├── state.py
                          │                            ├── level.py
                          │                            ├── delta.py
                          │        ml/pipeline_hybrid.py├── ts.py
                          │        (HybridPipeline)    ├── fusion.py
                          │                            ├── constraints.py
                          │                            ├── trend_classify.py
                          │                            ├── price_regime.py
                          │                            └── training.py
                          │
                          ├──► ml/features_future.py
                          ├──► ml/trend.py
                          │
                          ├──► data/features.py ──► core/config.py
                          │    (FeatureEngineer/       ├── data_availability
                          │     FeatureStore)          ├── type 三段式解析
                          │                           └── province/type校验
                          ├──► data/weather_features.py
                          ├──► data/fetcher.py (Open-Meteo)
                          ├──► data/holidays.py
                          ├──► data/quality.py
                          │
                          ├──► core/data_source.py
                          │    (FileSource / DorisSource / MemorySource)
                          │
                          └──► core/db.py (MySQL/Doris)
                               core/monitoring.py
                               core/cleanup.py

ml/physics/                  ml/transformer/
  ├── solar_model.py           ├── corrector.py
  ├── wind_model.py            └── trainer.py
  ├── load_model.py
  ├── price_model.py
  ├── net_load.py
  └── base.py
```

## 8. 训练特征全集

训练时传入 LGB 的特征 = `df.columns - EXCLUDE_TRAIN_COLS - {"value"}`

```python
EXCLUDE_TRAIN_COLS = {"dt", "province", "type", "price",
                       "model_version", "p10", "p50", "p90", "trend_adjusted"}
```

### 8.1 时序特征 (来源: build_features_from_raw)

| 特征 | 说明 | 计算方式 |
|---|---|---|
| `value_lag_1d` | 1天前同期值 | `shift(96)` |
| `value_lag_2d` | 2天前同期值 | `shift(192)` ¹ |
| `value_lag_3d` | 3天前同期值 | `shift(288)` ¹ |
| `value_lag_7d` | 7天前同期值 | `shift(672)` |
| `value_lag_14d` | 14天前同期值 | `shift(1344)` ¹ |
| `value_diff_1d` | 日环比变化 | `diff(96)` |
| `value_diff_2d` | 2日环比变化 | `diff(192)` ¹ |
| `value_diff_3d` | 3日环比变化 | `diff(288)` ¹ |
| `value_diff_7d` | 周环比变化 | `diff(672)` |
| `value_accel_1d` | 日加速度 | `diff_1d(t) - diff_1d(t-96)` ¹ |
| `value_rolling_mean_24h` | 24h滑动均值 | `rolling(96).mean()` |
| `value_rolling_std_24h` | 24h滑动标准差 | `rolling(96).std()` |
| `value_rolling_max_24h` | 24h滑动最大值 | `rolling(96).max()` |
| `value_rolling_min_24h` | 24h滑动最小值 | `rolling(96).min()` |
| `value_range_24h` | 24h波动幅度 | `max - min` |
| `value_zscore_24h` | 24h标准化 | `(v - mean) / (std+ε)` |
| `value_percentile_7d` | 7日百分位 | `rolling(672).rank(pct)` |
| `value_ema_1d` | 1日EMA | `α=0.30` ¹ |
| `value_ema_7d` | 7日EMA | `α=0.10` ¹ |
| `value_ema_30d` | 30日EMA | `α=0.03` ¹ |
| `ema_cross_1d_7d` | 短-中EMA差 | `(ema_1d - ema_7d) / (|ema_7d|+1)` ¹ |
| `ema_cross_7d_30d` | 中-长EMA差 | `(ema_7d - ema_30d) / (|ema_30d|+1)` ¹ |
| `volatility_regime` | 波动率体制 | 0=低波/1=中波/2=高波 ¹ |
| `vol_ratio_short_long` | 短/长波动比 | `σ_96 / σ_672` ¹ |
| `vol_trend` | 波动率变化 | `σ_t - σ_{t-96}` ¹ |
| `weekly_pattern_corr` | 周模式一致性 | 最近4周同(dow,tod)的稳定性 ¹ |
| `value_sign` | 值符号 | `sign(value)`, 抽水蓄能等双向 ¹ |
| `value_sign_lag_1d` | 1天前符号 | `sign(lag_1d)` ¹ |
| `value_sign_change` | 符号变化 | `sign != sign_lag_1d` ¹ |

> ¹ = 仅在 `build_future_features` 中生成，不再写入特征库

### 8.2 时间/日历特征

| 特征 | 说明 |
|---|---|
| `hour` | 小时 (0-23) |
| `day_of_week` | 星期 (0=周一) |
| `day_of_month` | 日 (1-31) |
| `month` | 月 (1-12) |
| `is_weekend` | 是否周末 |
| `season` | 季节 (1=春/2=夏/3=秋/4=冬) |
| `is_holiday` | 是否节假日 |
| `is_work_weekend` | 是否调休工作日 |
| `days_to_holiday` | 距下一节假日天数 |
| `days_from_holiday` | 距上一节假日天数 |
| `hour_sin` | 小时 sin 编码 |
| `hour_cos` | 小时 cos 编码 |
| `dow_sin` | 星期 sin 编码 |
| `dow_cos` | 星期 cos 编码 |
| `month_sin` | 月 sin 编码 |
| `month_cos` | 月 cos 编码 |

### 8.3 时间交互特征

| 特征 | 说明 |
|---|---|
| `peak_valley` | 峰平谷: 2=峰/1=平/0=谷 |
| `weekend_hour` | `is_weekend * hour` |
| `dow_hour` | `dow * 24 + hour` |
| `is_daylight` | 白天 (6≤h≤18) |
| `time_of_day` | 时段: 0=夜/1=晨峰/2=上午/3=午间/4=下午/5=晚峰 |
| `season_x_tod` | `season * 6 + time_of_day` |
| `daylight_x_temp` | `is_daylight * 6 + temp_bucket` |
| `weekend_x_tod` | `is_weekend * 6 + time_of_day` |
| `tod_x_temp_extreme` | `time_of_day * 4 + temp_extremity_bucket` |
| `working_day_type` | 工作日类型 ¹ |

### 8.4 天气特征 (来源: weather_features.py)

**基础气象 (Open-Meteo API):**

| 特征 | 说明 |
|---|---|
| `temperature` | 温度 (°C) |
| `humidity` | 相对湿度 (%) |
| `wind_speed` | 风速 (m/s) |
| `wind_direction` | 风向 (°) |
| `solar_radiation` | 太阳辐射 (W/m²) |
| `precipitation` | 降水量 (mm) |
| `pressure` | 气压 (hPa) |

**派生天气:**

| 特征 | 说明 |
|---|---|
| `CDD` | 制冷度日: `max(T-26, 0)` |
| `HDD` | 采暖度日: `max(18-T, 0)` |
| `THI` | 温湿指数: `0.8*T + 0.2*RH*T/100` |
| `temp_extremity` | 温度极端程度: `|T-22|/15 * humidity_factor` |
| `temp_zscore` | 温度 Z-score |
| `is_heat_wave` | 热浪: `T>35 且 zscore>2` |
| `is_cold_snap` | 寒潮: `T<-5 且 zscore<-2` |
| `extreme_weather_flag` | 极端天气标志 (热浪/寒潮/大风/暴雨) |
| `extreme_weather_count` | 极端天气计数 |
| `consecutive_hot_days` | 连续高温天数 |
| `cloud_factor` | 云量因子: `actual_radiation / clear_sky_radiation` |
| `wind_power_potential` | 风功率理论值: `0.5 * ρ * v³` |
| `wind_power_curve` | 风速功率曲线归一化 |
| `solar_potential` | 光伏潜力: `radiation / 1000` |
| `solar_efficiency` | 光伏效率: `1 - 0.005*(T-25)` |

**天气交互:**

| 特征 | 说明 |
|---|---|
| `temp_x_hour` | `temperature * hour` |
| `wind_x_season` | `wind_speed * season` |
| `temp_x_humidity` | `temperature * humidity / 100` |
| `extreme_x_tod` | `extreme_flag * time_of_day` |
| `heat_wave_x_daylight` | `heat_wave * is_daylight` |
| `tzscore_x_tod` | `temp_zscore * time_of_day` |
| `val_extreme_x_weather` | `|value_zscore| * extreme_flag` |

**天气 lag:**

| 特征 | 说明 |
|---|---|
| `solar_radiation_lag_1d` | 昨天太阳辐射 |
| `solar_radiation_diff_1d` | 辐射日变化 |
| `wind_speed_lag_1d` | 昨天风速 |
| `wind_speed_diff_1d` | 风速日变化 |
| `temperature_lag_1d` | 昨天温度 |
| `temperature_diff_1d` | 温度日变化 |
| `temp_change_1h` | 1h温度变化 |
| `temp_change_6h` | 6h温度变化 |

### 8.5 交叉类型特征 (同时刻注入)

每个交叉类型注入 5 列:

| 特征 | 说明 |
|---|---|
| `{type}_value` | 同时刻交叉类型实际值 |
| `{type}_lag_1d` | 交叉类型 1 天前值 |
| `{type}_lag_7d` | 交叉类型 7 天前值 |
| `{type}_lag_2d` | 交叉类型 2 天前值 ¹ |
| `{type}_lag_14d` | 交叉类型 14 天前值 ¹ |

当前注入的交叉类型:
```
出力_光伏_实际, 出力_总_实际, 出力_水电含抽蓄_实际,
出力_联络线_实际, 出力_非市场_实际, 出力_风电_实际, 负荷_系统_实际
```

### 8.6 物理/经济派生特征

| 特征 | 说明 |
|---|---|
| `supply_demand_ratio` | 供需比: `总出力 / 总负荷` |
| `supply_surplus` | 供给过剩: `总出力 - 总负荷` |
| `renewable_share` / `renewable_penetration` | 可再生占比: `(风电+光伏) / 总出力` |
| `residual_load` | 剩余负荷: `负荷 - 风电 - 光伏` |
| `residual_load_ratio` | 剩余负荷比: `residual_load / load` |
| `re_share_change_1d` | 可再生占比日变化 |
| `thermal_marginal_cost` | 火电边际成本: `coal_price * 0.35` |
| `price_pressure_index` | 电价压力指数: `sdr * coal_price` |

### 8.7 电价专属特征

| 特征 | 说明 |
|---|---|
| `price_per_load` | 电价/负荷比 |
| `price_x_load` | 电价×负荷 |
| `price_x_re_share` | 电价×(1-可再生占比) |
| `price_vol_24h` | 24h电价波动率 |
| `price_vol_7d` | 7日电价波动率 |
| `price_momentum_1h` | 1h电价动量 |
| `price_momentum_6h` | 6h电价动量 |
| `price_momentum_24h` | 24h电价动量 |
| `price_position` | 日内电价位置 (0=谷/1=峰) |
| `peak_off_peak_spread` | 峰谷价差 |
| `morning_evening_ratio` | 晨峰/晚峰比 |
| `peak_valley_spread` | 峰谷价差(日内) |
| `load_factor` | 负荷率: `load / peak_load` |
| `load_ramp_rate` | 负荷爬坡率 |
| `sdr_zscore` | 供需比 Z-score |

### 8.8 预测误差反馈特征 (延迟注入)

| 特征 | 说明 |
|---|---|
| `pred_error` | 历史预测误差: `actual - p50` |
| `pred_error_lag_1d` | 1天前误差 |
| `pred_error_lag_7d` | 7天前误差 |
| `pred_error_bias_24h` | 24h系统性偏差 |
| `pred_error_std_24h` | 24h误差波动 |
| `pred_error_trend` | 误差趋势变化 |
| `interval_coverage` | 区间覆盖率 |
| `coverage_rate_24h` | 24h平均覆盖 |
| `pred_error_hour_bias` | 时点历史平均误差 |
| `pred_error_weekend` | 周末vs工作日误差差 |
| `pred_error_holiday` | 节假日误差放大系数 |
| `pred_error_x_temp` | 误差×温度 |
| `pred_error_x_wind` | 误差×风速 |
| `pred_error_autocorr` | 误差自相关(1天) |
| `pred_error_regime` | 持续偏差方向 (+1/-1/0) |

### 8.9 Analog 特征 (仅波动型: 风电/光伏)

| 特征 | 说明 |
|---|---|
| `analog_value_mean` | 相似天气日的平均出力 |
| `analog_value_std` | 相似天气日的出力标准差 |
| `analog_dist_min` | 最近似天气的欧氏距离 |

---

## 9. 配置要点

```yaml
# 数据源
data_source: file          # file | memory | doris

# 双管线模式
hybrid.pipeline_mode: both # layered | hybrid | both

# 数据可用性延迟
data_availability:
  default_delays:
    预测: -1               # D+1 日前预测
    实际: 1                # D-1 实际值
    结算: 6                # D-6 结算数据
  type_overrides:
    电价_实时: 1
    电价_结算: 6
  province_overrides: {}   # 预留省份差异

# 预测器 (LayeredPipeline)
predictor:
  lookback_days: 30
  lgb_weight_min: 0.45
  lgb_weight_decay: 0.002

# 训练器
trainer:
  quick_n_estimators: 500

# 覆盖范围
provinces: [广东, 云南, 四川, 江苏, 山东, 浙江, 河南]
types: [出力, 负荷, 电价]
```
