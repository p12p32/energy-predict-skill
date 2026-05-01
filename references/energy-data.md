---
name: energy-data
description: Use when importing CSV data, building features, fetching weather data, or validating data quality for energy prediction pipelines.
---

## 工作空间

所有命令在 skill 根目录下执行，路径自动推导。

## 导入 CSV

```bash
python3 -c "
from scripts.core.data_source import FileSource
FileSource().import_csv('data/my_data.csv')
"
```

## 数据质量检测

```bash
python3 -c "
from scripts.data.quality import DataQuality
from scripts.data.features import FeatureStore
store = FeatureStore()
df = store.load_raw_data('广东', 'load', '2025-01-01', '2025-12-31')
if not df.empty:
    dq = DataQuality()
    report = dq.report(dq.detect(df))
    print(f'异常比例: {report[\"flagged_pct\"]}%')
"
```

## 构建特征

```bash
python3 -c "
from scripts.data.features import FeatureStore, FeatureEngineer
store = FeatureStore()
engineer = FeatureEngineer()
raw = store.load_raw_data('广东', 'load', '2025-01-01', '2025-12-31')
if not raw.empty:
    features = engineer.build_features_from_raw(raw)
    count = store.insert_features(features)
    print(f'写入 {count} 条特征')
"
```

## 拉取气象数据

```bash
python3 -c "
from scripts.data.fetcher import DataFetcher
f = DataFetcher()
df = f.fetch_weather('广东', '2025-01-01', '2025-01-31', mode='historical')
print(f'拉取 {len(df)} 条气象记录')
"
```
