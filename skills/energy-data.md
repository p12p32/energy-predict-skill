---
name: energy-data
description: 电力数据层 — 外部数据采集、数据质量检测、节假日识别、特征工程
---

# 数据层 Skill

被主 Skill `energy-predict` 调用。提供数据采集、清洗和特征构建能力。

## 工作空间

`/Users/pcy/analysSkills`

## 能力列表

### 拉取气象历史数据

```bash
python3 -c "
from src.data.fetcher import DataFetcher
import json

f = DataFetcher()
df = f.fetch_weather('广东', '2026-04-01', '2026-04-30', mode='historical')
print(f'拉取 {len(df)} 条气象记录')
print(df.head(3).to_dict('records'))
"
```

### 拉取气象预报数据

```bash
python3 -c "
from src.data.fetcher import DataFetcher
f = DataFetcher()
df = f.fetch_weather('广东', '2026-05-01', mode='forecast', forecast_days=7)
print(f'拉取 {len(df)} 条预报记录')
"
```

### 数据质量检测

```bash
python3 -c "
from src.data.quality import DataQuality
from src.core.db import DorisDB
from src.core.config import load_config

cfg = load_config()
db = DorisDB()
df = db.query(f'SELECT * FROM {cfg[\"data\"][\"source_table\"]} WHERE province=\"广东\" LIMIT 10000')

dq = DataQuality()
result = dq.detect(df, 'value')
report = dq.report(result)
print(f'异常比例: {report[\"flagged_pct\"]}%')
print(f'IQR异常: {report[\"iqr_outliers\"]}, 跳变: {report[\"spike_outliers\"]}')
"
```

### 构建特征

```bash
python3 -c "
from src.data.features import FeatureStore, FeatureEngineer
from src.core.db import DorisDB

db = DorisDB()
store = FeatureStore(db)
store.ensure_tables()

engineer = FeatureEngineer()
raw = store.load_raw_data('广东', 'load', '2026-04-01', '2026-04-30')
if not raw.empty:
    features = engineer.build_features_from_raw(raw)
    count = store.insert_features(features)
    print(f'写入 {count} 条特征到 energy_feature_store')
"
```

### 错误处理

| 错误 | 处理 |
|------|------|
| `requests.exceptions.ConnectionError` | 提示: "Open-Meteo API 不可达，请检查网络连接" |
| `ValueError: 未找到省份坐标` | 提示: "请在 config.yaml 的 province_coords 中添加该省份经纬度" |
