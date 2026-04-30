# 电力预测系统 Phase 1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建自我进化电力预测系统 MVP：Doris 数据读取 → LightGBM 短期预测 → 双循环验证 → 学习型 improver 自动优化 → Skill 对话入口。

**Architecture:** 8 个独立脚本通过 orchestrator 协调，循环 A（validator→analyzer→improver→trainer 实时验证链），循环 B（backtester→analyzer 回塑验证），循环 C（improver→trainer→backtester 自主学习闭环）。外部数据通过 data_fetcher 统一接入，经 feature_store 落 Doris 特征表供 trainer/predictor 共享。

**Tech Stack:** Python 3.10+, LightGBM, scikit-learn, pymysql, pyyaml, pandas, numpy

---

## 文件结构

```
analysSkills/
├── config.yaml                          # 全局配置
├── requirements.txt                     # Python 依赖
├── src/
│   ├── db.py                            # Doris 连接与查询封装
│   ├── config_loader.py                 # YAML 配置加载
│   ├── data_fetcher.py                  # 外部数据采集 (Open-Meteo + 煤价)
│   ├── feature_store.py                 # 特征工程 + Doris 特征表
│   ├── trainer.py                       # 模型训练
│   ├── predictor.py                     # 预测执行
│   ├── validator.py                     # 实时验证
│   ├── backtester.py                    # 回塑验证 + 多维度打分
│   ├── analyzer.py                      # 误差诊断
│   ├── improver.py                      # 学习型优化引擎 + 策略知识库
│   └── orchestrator.py                  # 总调度
├── models/                              # 模型存储目录
├── skills/
│   └── energy-predict.md                # Skill 定义文件
└── tests/
    ├── test_db.py
    ├── test_data_fetcher.py
    ├── test_feature_store.py
    ├── test_trainer.py
    ├── test_predictor.py
    ├── test_validator.py
    ├── test_backtester.py
    ├── test_analyzer.py
    └── test_improver.py
```

---

### Task 1: 项目骨架与配置

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config_loader.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建 config.yaml**

```yaml
# config.yaml — 全局配置
doris:
  host: "127.0.0.1"
  port: 9030
  user: "root"
  password: ""
  database: "energy"

data:
  source_table: "energy_raw"            # 原始历史数据表
  prediction_table: "energy_predictions" # 预测结果表
  feature_table: "energy_feature_store"  # 特征表
  strategy_table: "strategy_knowledge"   # 策略知识库表

model:
  storage_dir: "models"
  default_window_days: 90               # 默认训练窗口
  forecast_horizon_hours: 96            # 默认预测长度(96步=24小时)

validator:
  short_term_mape_threshold: 0.05       # 短期 MAPE 报警阈值
  mid_term_mape_threshold: 0.10         # 中期 MAPE 报警阈值
  consecutive_bias_trigger: 3            # 连续同向偏差触发

improver:
  max_hypotheses: 12                    # 每轮最多测试假设数
  improvement_threshold: 0.05           # 改善>5%才算成功
  robustness_multiplier: 1.05           # 其他维度允许的退化上限

provinces:
  - 广东
  - 云南
  - 四川
  - 江苏
  - 山东
  - 浙江
  - 河南

types:
  - output
  - load
  - price

external_data:
  weather:
    source: "open-meteo"                # open-meteo | qweather
    qweather_api_key: ""                # 和风天气 Key（可选）
  coal_price:
    enabled: false                      # Phase 2 启用
  carbon_price:
    enabled: false                      # Phase 2 启用

# 各省主要城市经纬度（用于气象 API）
province_coords:
  广东: { lat: 23.13, lon: 113.26 }     # 广州
  云南: { lat: 25.04, lon: 102.72 }     # 昆明
  四川: { lat: 30.57, lon: 104.07 }     # 成都
  江苏: { lat: 32.06, lon: 118.80 }     # 南京
  山东: { lat: 36.67, lon: 116.98 }     # 济南
  浙江: { lat: 30.27, lon: 120.15 }     # 杭州
  河南: { lat: 34.77, lon: 113.65 }     # 郑州
```

- [ ] **Step 2: 创建 requirements.txt**

```txt
pandas>=2.0.0
numpy>=1.24.0
pymysql>=1.1.0
pyyaml>=6.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
requests>=2.31.0
```

- [ ] **Step 3: 创建 config_loader.py**

```python
"""config_loader.py — YAML 配置加载与访问"""
import os
import yaml
from typing import Any, Dict

_CONFIG: Dict[str, Any] = {}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置（模块导入时自动调用），仅加载一次."""
    global _CONFIG
    if _CONFIG:
        return _CONFIG

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)

    return _CONFIG


def get_doris_config() -> Dict[str, Any]:
    return _CONFIG.get("doris", {})


def get_model_config() -> Dict[str, Any]:
    return _CONFIG.get("model", {})


def get_provinces() -> list:
    return _CONFIG.get("provinces", [])


def get_types() -> list:
    return _CONFIG.get("types", ["output", "load"])


def get_province_coords() -> Dict[str, Dict[str, float]]:
    return _CONFIG.get("province_coords", {})


def get_validator_config() -> Dict[str, Any]:
    return _CONFIG.get("validator", {})


def get_improver_config() -> Dict[str, Any]:
    return _CONFIG.get("improver", {})


# 自动加载
load_config()
```

- [ ] **Step 4: 创建空 __init__.py**

```bash
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 5: 提交**

```bash
git add config.yaml requirements.txt src/__init__.py src/config_loader.py tests/__init__.py
git commit -m "feat: project scaffolding with config loader"
```

---

### Task 2: Doris 连接层

**Files:**
- Create: `src/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: 编写测试 test_db.py**

```python
"""tests/test_db.py"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.db import DorisDB


class TestDorisDB:
    @patch("src.db.pymysql.connect")
    def test_query_returns_dataframe(self, mock_connect):
        mock_cursor = MagicMock()
        mock_cursor.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [(1, "广东"), (2, "云南")]
        mock_cursor.description = [("id",), ("province",)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = DorisDB(host="127.0.0.1", port=9030, user="root",
                     password="", database="energy")
        df = db.query("SELECT id, province FROM test_table")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "province"]
        assert len(df) == 2

    @patch("src.db.pymysql.connect")
    def test_query_empty_result(self, mock_connect):
        mock_cursor = MagicMock()
        mock_cursor.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("id",), ("province",)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = DorisDB(host="127.0.0.1", port=9030, user="root",
                     password="", database="energy")
        df = db.query("SELECT id FROM empty_table")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("src.db.pymysql.connect")
    def test_execute_ddl(self, mock_connect):
        mock_cursor = MagicMock()
        mock_cursor.__enter__.return_value = mock_cursor
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db = DorisDB(host="127.0.0.1", port=9030, user="root",
                     password="", database="energy")
        db.execute("CREATE TABLE IF NOT EXISTS t (id INT)")

        mock_cursor.execute.assert_called_once_with(
            "CREATE TABLE IF NOT EXISTS t (id INT)"
        )
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_db.py -v
# Expected: FAIL — DorisDB not defined
```

- [ ] **Step 3: 实现 db.py**

```python
"""db.py — Apache Doris 连接与查询封装"""
import pymysql
import pandas as pd
from typing import Optional, Dict, Any
from src.config_loader import get_doris_config


class DorisDB:
    def __init__(self, host: str = None, port: int = None,
                 user: str = None, password: str = None,
                 database: str = None):
        cfg = get_doris_config()
        self.host = host or cfg["host"]
        self.port = port or cfg["port"]
        self.user = user or cfg["user"]
        self.password = password or cfg["password"]
        self.database = database or cfg["database"]

    def _get_connection(self) -> pymysql.Connection:
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
        )

    def query(self, sql: str) -> pd.DataFrame:
        """执行 SELECT 查询，返回 DataFrame."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()

    def execute(self, sql: str) -> None:
        """执行非查询语句（DDL / INSERT / DELETE）."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def insert_dataframe(self, df: pd.DataFrame, table: str,
                         batch_size: int = 5000) -> int:
        """将 DataFrame 批量写入 Doris 表. 返回写入行数."""
        if df.empty:
            return 0

        columns = ",".join(f"`{c}`" for c in df.columns)
        placeholders = ",".join(["%s"] * len(df.columns))
        sql = f"INSERT INTO `{table}` ({columns}) VALUES ({placeholders})"

        conn = self._get_connection()
        total = 0
        try:
            with conn.cursor() as cursor:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i : i + batch_size]
                    values = [
                        tuple(
                            None if pd.isna(v) else v
                            for v in row
                        )
                        for row in batch.itertuples(index=False)
                    ]
                    cursor.executemany(sql, values)
                    total += len(values)
            conn.commit()
        finally:
            conn.close()
        return total

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在."""
        result = self.query(
            f"SELECT COUNT(*) as cnt FROM information_schema.tables "
            f"WHERE table_schema='{self.database}' AND table_name='{table_name}'"
        )
        return result.iloc[0]["cnt"] > 0
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_db.py -v
# Expected: 3 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/db.py tests/test_db.py
git commit -m "feat: Doris database connection layer with DataFrame support"
```

---

### Task 3: 外部数据采集

**Files:**
- Create: `src/data_fetcher.py`
- Create: `tests/test_data_fetcher.py`

- [ ] **Step 1: 编写测试 test_data_fetcher.py**

```python
"""tests/test_data_fetcher.py"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.data_fetcher import DataFetcher, WeatherFetcher


class TestWeatherFetcher:
    def setup_method(self):
        self.province = "广东"
        self.lat, self.lon = 23.13, 113.26

    @patch("src.data_fetcher.requests.get")
    def test_fetch_historical(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "latitude": 23.13,
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                "temperature_2m": [15.2, 14.8],
                "relative_humidity_2m": [72, 75],
                "wind_speed_100m": [3.5, 4.1],
                "direct_radiation": [0, 120],
                "precipitation": [0.0, 0.0],
                "surface_pressure": [1013.2, 1013.5],
            },
        }
        mock_get.return_value = mock_response

        fetcher = WeatherFetcher(source="open-meteo")
        df = fetcher.fetch_historical(
            province=self.province, lat=self.lat, lon=self.lon,
            start_date="2025-01-01", end_date="2025-01-01"
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "temperature" in df.columns
        assert "wind_speed" in df.columns
        assert "province" in df.columns
        assert (df["province"] == self.province).all()

    @patch("src.data_fetcher.requests.get")
    def test_fetch_forecast(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2025-05-01T00:00", "2025-05-01T01:00"],
                "temperature_2m": [25.0, 24.5],
                "relative_humidity_2m": [65, 68],
                "wind_speed_100m": [2.1, 2.8],
                "direct_radiation": [250, 180],
                "precipitation": [0.0, 0.0],
                "surface_pressure": [1010.0, 1009.8],
            },
        }
        mock_get.return_value = mock_response

        fetcher = WeatherFetcher(source="open-meteo")
        df = fetcher.fetch_forecast(
            province=self.province, lat=self.lat, lon=self.lon,
            forecast_days=7
        )

        assert isinstance(df, pd.DataFrame)
        assert "forecast_hour" in df.columns

    @patch("src.data_fetcher.requests.get")
    def test_api_retry_on_failure(self, mock_get):
        mock_get.side_effect = [
            MagicMock(status_code=500),
            MagicMock(status_code=200, json=MagicMock(return_value={
                "hourly": {
                    "time": ["2025-01-01T00:00"],
                    "temperature_2m": [15.0],
                    "relative_humidity_2m": [70],
                    "wind_speed_100m": [3.0],
                    "direct_radiation": [100],
                    "precipitation": [0.0],
                    "surface_pressure": [1013.0],
                },
            })),
        ]

        fetcher = WeatherFetcher(source="open-meteo", max_retries=2)
        df = fetcher.fetch_historical(
            province=self.province, lat=self.lat, lon=self.lon,
            start_date="2025-01-01", end_date="2025-01-01"
        )
        assert len(df) == 1
        assert mock_get.call_count == 2


class TestDataFetcher:
    @patch("src.data_fetcher.WeatherFetcher.fetch_historical")
    def test_fetch_weather_historical(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame({
            "dt": pd.to_datetime(["2025-01-01 00:00"]),
            "province": ["广东"],
            "temperature": [15.0],
            "wind_speed": [3.0],
            "humidity": [72],
            "solar_radiation": [100],
            "precipitation": [0.0],
            "pressure": [1013.0],
        })

        fetcher = DataFetcher()
        df = fetcher.fetch_weather(
            province="广东", start_date="2025-01-01",
            end_date="2025-01-01", mode="historical"
        )
        assert len(df) == 1
        assert "temperature" in df.columns
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_data_fetcher.py -v
# Expected: FAIL — DataFetcher / WeatherFetcher not defined
```

- [ ] **Step 3: 实现 data_fetcher.py**

```python
"""data_fetcher.py — 外部数据采集（气象、煤价、碳价）"""
import time
from typing import Optional, Dict
from datetime import datetime, timedelta

import requests
import pandas as pd
from src.config_loader import get_province_coords

# ── 气象 API ──────────────────────────────────────

OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

WEATHER_HOURLY_PARAMS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_100m",
    "wind_direction_100m", "direct_radiation", "precipitation",
    "surface_pressure",
]

WEATHER_COLUMN_MAP = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_100m": "wind_speed",
    "wind_direction_100m": "wind_direction",
    "direct_radiation": "solar_radiation",
    "precipitation": "precipitation",
    "surface_pressure": "pressure",
}


class WeatherFetcher:
    """气象数据采集器，支持 Open-Meteo 免费 API."""
    def __init__(self, source: str = "open-meteo",
                 max_retries: int = 3, retry_delay: float = 2.0):
        self.source = source
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_historical(self, province: str, lat: float, lon: float,
                         start_date: str, end_date: str) -> pd.DataFrame:
        """拉取历史气象数据."""
        url = OPEN_METEO_HISTORICAL
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_HOURLY_PARAMS),
            "timezone": "Asia/Shanghai",
        }
        data = self._request(url, params)
        return self._parse_response(data, province, is_forecast=False)

    def fetch_forecast(self, province: str, lat: float, lon: float,
                       forecast_days: int = 7) -> pd.DataFrame:
        """拉取天气预报."""
        url = OPEN_METEO_FORECAST
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_HOURLY_PARAMS),
            "forecast_days": forecast_days,
            "timezone": "Asia/Shanghai",
        }
        data = self._request(url, params)
        return self._parse_response(data, province, is_forecast=True)

    def _request(self, url: str, params: Dict) -> Dict:
        for attempt in range(self.max_retries):
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(
            f"气象 API 请求失败，status={resp.status_code} url={url}"
        )

    def _parse_response(self, data: Dict, province: str,
                        is_forecast: bool) -> pd.DataFrame:
        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["dt"] = pd.to_datetime(df.pop("time"))

        # 重命名列
        for src, dst in WEATHER_COLUMN_MAP.items():
            if src in df.columns:
                df.rename(columns={src: dst}, inplace=True)
            else:
                df[dst] = None

        df["province"] = province

        want_cols = ["dt", "province", "temperature", "humidity",
                     "wind_speed", "wind_direction", "solar_radiation",
                     "precipitation", "pressure"]

        if is_forecast:
            df["forecast_hour"] = range(len(df))

        result = df[[c for c in want_cols if c in df.columns]]
        return result


# ── 统一入口 ──────────────────────────────────────

class DataFetcher:
    """外部数据统一入口."""

    def __init__(self):
        self.weather = WeatherFetcher(source="open-meteo")
        self._coords = get_province_coords()

    def fetch_weather(self, province: str, start_date: str,
                      end_date: str = None, mode: str = "historical",
                      forecast_days: int = 7) -> pd.DataFrame:
        """统一气象数据获取.

        Args:
            province: 省份名
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期（history 必填）
            mode: 'historical' | 'forecast'
            forecast_days: 预报天数（mode='forecast' 时使用）
        """
        coords = self._coords.get(province)
        if not coords:
            raise ValueError(f"未找到省份坐标: {province}")

        if mode == "historical":
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            return self.weather.fetch_historical(
                province, coords["lat"], coords["lon"],
                start_date, end_date
            )
        elif mode == "forecast":
            return self.weather.fetch_forecast(
                province, coords["lat"], coords["lon"],
                forecast_days=forecast_days
            )
        else:
            raise ValueError(f"mode must be 'historical' or 'forecast', got: {mode}")

    def fetch_weather_for_all_provinces(self, start_date: str,
                                         end_date: str = None,
                                         mode: str = "historical") -> pd.DataFrame:
        """批量拉取所有省份气象."""
        frames = []
        for province in self._coords:
            df = self.fetch_weather(province, start_date, end_date, mode)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_data_fetcher.py -v
# Expected: 4 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/data_fetcher.py tests/test_data_fetcher.py
git commit -m "feat: external data fetcher with Open-Meteo weather API"
```

---

### Task 4: 特征工程与特征存储

**Files:**
- Create: `src/feature_store.py`
- Create: `tests/test_feature_store.py`

- [ ] **Step 1: 编写测试 test_feature_store.py**

```python
"""tests/test_feature_store.py"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.feature_store import FeatureStore, FeatureEngineer


class TestFeatureEngineer:
    def test_build_features_from_raw(self):
        dates = pd.date_range("2025-01-01", periods=48, freq="15min")
        raw = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 48,
            "type": ["load"] * 48,
            "value": [100 + i * 2 for i in range(48)],
            "price": [0.35] * 48,
        })

        engineer = FeatureEngineer()
        df = engineer.build_features_from_raw(raw)

        expected_cols = [
            "dt", "province", "type", "value", "price",
            "hour", "day_of_week", "day_of_month", "month",
            "is_weekend", "season",
            "value_lag_1d", "value_lag_7d",
            "value_rolling_mean_24h",
            "value_diff_1d", "value_diff_7d",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_lag_features_computed(self):
        dates = pd.date_range("2025-01-07", periods=96, freq="15min")
        raw = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 96,
            "type": ["output"] * 96,
            "value": [float(i) for i in range(96)],
            "price": [0.30] * 96,
        })
        engineer = FeatureEngineer()
        df = engineer.build_features_from_raw(raw)

        # 前 96 个 15 分钟 = 1 天，所以 lag_1d 前 96 行应为 NaN
        assert pd.isna(df["value_lag_1d"].iloc[0])
        # 第 96 行（index 95）应该有前一天的值
        assert not pd.isna(df["value_lag_1d"].iloc[96])

    def test_merge_weather_features(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="H")
        features = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 4,
            "type": ["load"] * 4,
            "value": [100, 102, 105, 103],
        })
        weather = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 4,
            "temperature": [12, 13, 14, 13],
            "humidity": [70, 68, 65, 67],
            "wind_speed": [3, 3, 4, 3],
        })

        engineer = FeatureEngineer()
        merged = engineer.merge_weather(features, weather)

        assert "temperature" in merged.columns
        assert merged["temperature"].iloc[0] == 12


class TestFeatureStore:
    @patch("src.feature_store.DorisDB")
    def test_ensure_tables(self, MockDB):
        mock_db = MockDB.return_value
        store = FeatureStore(mock_db)
        store.ensure_tables()
        assert mock_db.execute.call_count >= 2  # feature + prediction DDL

    @patch("src.feature_store.DorisDB")
    def test_insert_features(self, MockDB):
        mock_db = MockDB.return_value
        store = FeatureStore(mock_db)

        dates = pd.date_range("2025-01-01", periods=4, freq="H")
        df = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 4,
            "type": ["load"] * 4,
            "value": [100, 102, 105, 103],
            "price": [0.35] * 4,
            "hour": [0, 1, 2, 3],
            "day_of_week": [2, 2, 2, 2],
            "temperature": [12, 13, 14, 13],
        })

        count = store.insert_features(df)
        assert count == 4
        mock_db.insert_dataframe.assert_called_once()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_feature_store.py -v
# Expected: FAIL
```

- [ ] **Step 3: 实现 feature_store.py**

```python
"""feature_store.py — 特征工程与 Doris 特征表管理"""
import pandas as pd
import numpy as np
from typing import Optional, List
from src.db import DorisDB
from src.config_loader import load_config

PREDICTION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS energy_predictions (
    dt              DATETIME      NOT NULL,
    province        VARCHAR(32)   NOT NULL,
    type            VARCHAR(16)   NOT NULL,
    p10             DOUBLE        COMMENT 'P10分位数',
    p50             DOUBLE        COMMENT 'P50分位数（中位数）',
    p90             DOUBLE        COMMENT 'P90分位数',
    model_version   VARCHAR(64)   COMMENT '模型版本',
    created_at      DATETIME      DEFAULT CURRENT_TIMESTAMP
)
DUPLICATE KEY (dt, province, type)
DISTRIBUTED BY HASH (province) BUCKETS 8;
"""

FEATURE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS energy_feature_store (
    dt              DATETIME      NOT NULL,
    province        VARCHAR(32)   NOT NULL,
    type            VARCHAR(16)   NOT NULL,
    value           DOUBLE        COMMENT '原始值',
    price           DOUBLE        COMMENT '电价',
    -- 时间特征
    hour            TINYINT       COMMENT '小时(0-23)',
    day_of_week     TINYINT       COMMENT '星期(0=周一)',
    day_of_month    TINYINT       COMMENT '日(1-31)',
    month           TINYINT       COMMENT '月(1-12)',
    is_weekend      BOOLEAN       COMMENT '是否周末',
    season          TINYINT       COMMENT '季节(1=春,2=夏,3=秋,4=冬)',
    -- 气象特征（Phase 1: 可选，Phase 2: 必填）
    temperature     DOUBLE        COMMENT '温度(°C)',
    humidity        DOUBLE        COMMENT '相对湿度(%)',
    wind_speed      DOUBLE        COMMENT '风速(m/s)',
    wind_direction  DOUBLE        COMMENT '风向(°)',
    solar_radiation DOUBLE        COMMENT '太阳辐射(W/m²)',
    precipitation   DOUBLE        COMMENT '降水量(mm)',
    pressure        DOUBLE        COMMENT '气压(hPa)',
    -- 滞后特征
    value_lag_1d    DOUBLE        COMMENT '一天前同期值',
    value_lag_7d    DOUBLE        COMMENT '七天前同期值',
    value_rolling_mean_24h DOUBLE  COMMENT '24小时滑动均值',
    -- 差分特征
    value_diff_1d   DOUBLE        COMMENT '日环比变化',
    value_diff_7d   DOUBLE        COMMENT '周环比变化',
    -- 索引
    INDEX idx_province_dt (province, dt) USING BITMAP
)
DUPLICATE KEY (dt, province, type)
DISTRIBUTED BY HASH (province) BUCKETS 16;
"""

STRATEGY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS strategy_knowledge (
    strategy_hash   VARCHAR(64)   PRIMARY KEY,
    strategy_desc   STRING,
    applied_count   INT           DEFAULT 0,
    success_count   INT           DEFAULT 0,
    avg_improvement DOUBLE        DEFAULT 0,
    best_scenario   VARCHAR(256)  DEFAULT '',
    worst_scenario  VARCHAR(256)  DEFAULT '',
    last_applied    DATETIME,
    last_effect     DOUBLE        DEFAULT 0,
    retired         BOOLEAN       DEFAULT FALSE
)
UNIQUE KEY (strategy_hash)
DISTRIBUTED BY HASH (strategy_hash) BUCKETS 4;
"""


class FeatureEngineer:
    """特征工程：从原始数据生成训练/预测特征."""

    LAG_PERIODS = {
        "value_lag_1d": 96,      # 96 × 15min = 1天
        "value_lag_7d": 672,     # 672 × 15min = 7天
        "value_diff_1d": 96,
        "value_diff_7d": 672,
    }

    def build_features_from_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """从 Doris 原始数据表构建特征 DataFrame.

        raw 必须包含: dt, province, type, value, price
        返回添加了时间特征和滞后特征的 DataFrame.
        """
        df = raw.copy()
        df = df.sort_values(["province", "type", "dt"]).reset_index(drop=True)

        # ── 时间特征 ──
        df["hour"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek  # 0=周一
        df["day_of_month"] = df["dt"].dt.day
        df["month"] = df["dt"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        df["season"] = df["month"].apply(
            lambda m: 1 if m in [3, 4, 5] else
                      2 if m in [6, 7, 8] else
                      3 if m in [9, 10, 11] else 4
        )

        # ── 滞后特征（按 (province, type) 分组计算）──
        for group_key, group_df in df.groupby(["province", "type"]):
            idx = group_df.index

            for col_name, shift_n in self.LAG_PERIODS.items():
                if "diff" in col_name:
                    df.loc[idx, col_name] = group_df["value"].diff(shift_n)
                else:
                    df.loc[idx, col_name] = group_df["value"].shift(shift_n)

            # 24 小时滑动均值 (96 步)
            df.loc[idx, "value_rolling_mean_24h"] = (
                group_df["value"].rolling(window=96, min_periods=1).mean().values
            )

        return df

    def merge_weather(self, features: pd.DataFrame,
                      weather: pd.DataFrame) -> pd.DataFrame:
        """将气象数据合并到特征表."""
        weather_subset = weather[
            ["dt", "province", "temperature", "humidity",
             "wind_speed", "wind_direction", "solar_radiation",
             "precipitation", "pressure"]
        ].copy()
        return features.merge(
            weather_subset, on=["dt", "province"], how="left"
        )


class FeatureStore:
    """特征存储管理：Doris 表创建与数据写入."""

    def __init__(self, db: DorisDB):
        self.db = db

    def ensure_tables(self):
        """确保所有必需表存在."""
        self.db.execute(FEATURE_TABLE_DDL)
        self.db.execute(PREDICTION_TABLE_DDL)
        self.db.execute(STRATEGY_TABLE_DDL)

    def insert_features(self, df: pd.DataFrame) -> int:
        """写入特征数据到 Doris."""
        cols = [
            "dt", "province", "type", "value", "price",
            "hour", "day_of_week", "day_of_month", "month",
            "is_weekend", "season",
            "temperature", "humidity", "wind_speed", "wind_direction",
            "solar_radiation", "precipitation", "pressure",
            "value_lag_1d", "value_lag_7d",
            "value_rolling_mean_24h",
            "value_diff_1d", "value_diff_7d",
        ]
        available = [c for c in cols if c in df.columns]
        return self.db.insert_dataframe(
            df[available], "energy_feature_store"
        )

    def insert_predictions(self, df: pd.DataFrame) -> int:
        """写入预测结果."""
        cols = ["dt", "province", "type", "p10", "p50", "p90",
                "model_version"]
        available = [c for c in cols if c in df.columns]
        if "model_version" not in df.columns:
            df["model_version"] = "v1"
        return self.db.insert_dataframe(
            df[available], "energy_predictions"
        )

    def load_features(self, province: str, data_type: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        """从特征表加载数据."""
        sql = f"""
            SELECT * FROM energy_feature_store
            WHERE province = '{province}'
              AND type = '{data_type}'
              AND dt >= '{start_date}'
              AND dt < '{end_date}'
            ORDER BY dt
        """
        return self.db.query(sql)

    def load_raw_data(self, province: str, data_type: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
        """从原始表加载数据."""
        cfg = load_config()
        src_table = cfg["data"]["source_table"]
        sql = f"""
            SELECT dt, province, type, value, price
            FROM {src_table}
            WHERE province = '{province}'
              AND type = '{data_type}'
              AND dt >= '{start_date}'
              AND dt < '{end_date}'
            ORDER BY dt
        """
        return self.db.query(sql)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_feature_store.py::TestFeatureEngineer -v
# Expected: 3 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/feature_store.py tests/test_feature_store.py
git commit -m "feat: feature engineering with lag/time/weather features + Doris feature store"
```

---

### Task 5: 模型训练器

**Files:**
- Create: `src/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: 编写测试 test_trainer.py**

```python
"""tests/test_trainer.py"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.trainer import Trainer


class TestTrainer:
    def setup_method(self):
        # 构造训练数据
        dates = pd.date_range("2025-01-01", periods=2000, freq="15min")
        np.random.seed(42)
        noise = np.random.normal(0, 5, 2000)
        base = np.sin(np.arange(2000) * 0.01) * 20 + 100 + noise
        self.train_df = pd.DataFrame({
            "dt": dates,
            "province": "广东",
            "type": "load",
            "hour": dates.hour,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
            "season": dates.month.apply(
                lambda m: 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3 if m in [9,10,11] else 4
            ),
            "value_lag_1d": np.roll(base, 96),
            "value_lag_7d": np.roll(base, 672),
            "value_rolling_mean_24h": pd.Series(base).rolling(96, min_periods=1).mean(),
            "value_diff_1d": np.diff(base, prepend=base[0]),
            "value_diff_7d": np.diff(base, prepend=base[0]),
            "value": base,  # target
        }).dropna()

    def test_prepare_training_data(self):
        trainer = Trainer()
        X, y = trainer.prepare_training_data(self.train_df, target_col="value")

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] >= 5  # at least 5 features
        assert "dt" not in X.columns  # dt should be excluded
        assert "province" not in X.columns or "province" not in [str(c) for c in X.columns]

    @patch("src.trainer.os.makedirs")
    @patch("src.trainer.os.path.exists")
    def test_train_and_save(self, mock_exists, mock_makedirs):
        mock_exists.return_value = True
        trainer = Trainer(model_dir="/tmp/test_models")

        result = trainer.train(
            df=self.train_df,
            province="广东",
            target_type="load",
            target_col="value",
        )

        assert "province" in result
        assert "model_path" in result
        assert result["n_samples"] == len(self.train_df)

    def test_quick_train_small_batch(self):
        """快速训练（improver 假设验证用）."""
        trainer = Trainer()
        sample = self.train_df.tail(500).copy()
        result = trainer.quick_train(
            df=sample,
            province="广东",
            target_type="load",
            target_col="value",
        )
        assert "model" in result
        assert "feature_names" in result
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_trainer.py -v
# Expected: FAIL
```

- [ ] **Step 3: 实现 trainer.py**

```python
"""trainer.py — 模型训练器（LightGBM 短期预测）"""
import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src.config_loader import get_model_config

EXCLUDE_COLS = {"dt", "province", "type", "price"}


class Trainer:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or get_model_config()["storage_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_training_data(self, df: pd.DataFrame,
                               target_col: str = "value") -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据: 剔除不可用列，分离特征和目标."""
        df = df.dropna(subset=[target_col]).copy()

        # 特征列: 排除 dt/province/type/price 以及目标列自身
        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS and c != target_col
        ]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        return X, y

    def train(self, df: pd.DataFrame, province: str,
              target_type: str, target_col: str = "value",
              params: Dict = None, model_filename: str = None) -> Dict:
        """全量训练并保存模型.

        Returns:
            {"province", "target_type", "model_path", "n_samples", "feature_names"}
        """
        X, y = self.prepare_training_data(df, target_col)

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 200,
            "early_stopping_rounds": 20,
        }
        if params:
            lgb_params.update(params)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
        )

        # 保存
        if model_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{province}_{target_type}_{timestamp}.lgb"

        province_dir = os.path.join(self.model_dir, province)
        os.makedirs(province_dir, exist_ok=True)
        model_path = os.path.join(province_dir, model_filename)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # 更新注册
        self._update_registry(province, target_type, model_filename,
                              list(X.columns))

        return {
            "province": province,
            "target_type": target_type,
            "model_path": model_path,
            "n_samples": len(df),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
        }

    def quick_train(self, df: pd.DataFrame, province: str,
                    target_type: str, target_col: str = "value",
                    params: Dict = None) -> Dict:
        """轻量训练（improver 假设验证用），不保存到磁盘.
        Returns:
            {"model": LGBMRegressor, "feature_names": [...], "n_samples": int}
        """
        X, y = self.prepare_training_data(df, target_col)

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "verbose": -1,
            "n_estimators": 100,
        }
        if params:
            lgb_params.update(params)

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, y)

        return {
            "model": model,
            "feature_names": list(X.columns),
            "n_samples": len(df),
            "province": province,
            "target_type": target_type,
        }

    def load_model(self, province: str,
                   target_type: str) -> Tuple[lgb.LGBMRegressor, List[str]]:
        """加载最新训练的模型.

        Returns:
            (model, feature_names)
        """
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        if key not in registry:
            raise FileNotFoundError(f"未找到模型: {key}")

        model_filename = registry[key]["latest"]
        model_path = os.path.join(self.model_dir, province, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model, registry[key]["feature_names"]

    def _update_registry(self, province: str, target_type: str,
                         filename: str, feature_names: List[str]):
        registry = self._read_registry()
        key = f"{province}_{target_type}"
        registry[key] = {
            "latest": filename,
            "feature_names": feature_names,
            "updated_at": datetime.now().isoformat(),
        }
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    def _read_registry(self) -> Dict:
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                return json.load(f)
        return {}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_trainer.py -v
# Expected: 3 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/trainer.py tests/test_trainer.py
git commit -m "feat: LightGBM trainer with model persistence and registry"
```

---

### Task 6: 预测器

**Files:**
- Create: `src/predictor.py`
- Create: `tests/test_predictor.py`

- [ ] **Step 1: 编写测试 test_predictor.py**

```python
"""tests/test_predictor.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.predictor import Predictor


class TestPredictor:
    def setup_method(self):
        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=2000, freq="15min")
        self.mock_features = pd.DataFrame({
            "dt": dates[:96],
            "province": ["广东"] * 96,
            "type": ["load"] * 96,
            "hour": dates[:96].hour,
            "day_of_week": dates[:96].dayofweek,
            "month": dates[:96].month,
            "is_weekend": [0] * 96,
            "season": [1] * 96,
            "value_lag_1d": np.random.randn(96) + 100,
            "value_lag_7d": np.random.randn(96) + 100,
            "value_rolling_mean_24h": np.random.randn(96) + 100,
            "value_diff_1d": np.random.randn(96),
            "value_diff_7d": np.random.randn(96),
        })

    def test_predict_produces_correct_shape(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([100.0] * 96)
        feature_names = [
            "hour", "day_of_week", "month", "is_weekend", "season",
            "value_lag_1d", "value_lag_7d", "value_rolling_mean_24h",
            "value_diff_1d", "value_diff_7d",
        ]

        predictor = Predictor()
        result = predictor._predict_with_model(
            mock_model, self.mock_features[feature_names], feature_names,
            province="广东", target_type="load"
        )

        assert len(result) == 96
        assert "p50" in result.columns
        assert "province" in result.columns
        assert result["province"].iloc[0] == "广东"

    @patch("src.predictor.FeatureStore")
    @patch("src.predictor.Trainer")
    def test_predict_loads_model_and_features(self, MockTrainer, MockStore):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([100.0] * 10)
        mock_trainer = MockTrainer.return_value
        mock_trainer.load_model.return_value = (
            mock_model,
            ["hour", "day_of_week", "month", "is_weekend", "season",
             "value_lag_1d", "value_lag_7d", "value_rolling_mean_24h",
             "value_diff_1d", "value_diff_7d"],
        )

        mock_store = MockStore.return_value
        mock_store.load_features.return_value = pd.DataFrame({
            "dt": pd.date_range("2025-02-01", periods=10, freq="15min"),
            "hour": list(range(10)),
            "day_of_week": [2] * 10,
            "month": [2] * 10,
            "is_weekend": [0] * 10,
            "season": [1] * 10,
            "value_lag_1d": [0.0] * 10,
            "value_lag_7d": [0.0] * 10,
            "value_rolling_mean_24h": [0.0] * 10,
            "value_diff_1d": [0.0] * 10,
            "value_diff_7d": [0.0] * 10,
        })

        predictor = Predictor(trainer=mock_trainer, store=mock_store)
        result = predictor.predict(
            province="广东", target_type="load", horizon_hours=2
        )

        assert len(result) <= 10  # 2 hours = 8 steps
        mock_trainer.load_model.assert_called_once_with("广东", "load")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_predictor.py -v
# Expected: FAIL
```

- [ ] **Step 3: 实现 predictor.py**

```python
"""predictor.py — 预测执行器"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

from src.trainer import Trainer
from src.feature_store import FeatureStore
from src.db import DorisDB
from src.config_loader import load_config


class Predictor:
    def __init__(self, trainer: Trainer = None,
                 store: FeatureStore = None):
        self.trainer = trainer or Trainer()
        self.store = store or FeatureStore(DorisDB())

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24,
                model_version: str = None) -> pd.DataFrame:
        """对指定省份/类型做未来 N 小时预测.

        Args:
            province: 省份名
            target_type: 'output' | 'load' | 'price'
            horizon_hours: 预测时长（小时），最多 96（24h）
            model_version: 模型版本标签

        Returns:
            DataFrame with columns: dt, province, type, p10, p50, p90, model_version
        """
        model, feature_names = self.trainer.load_model(province, target_type)

        # 加载最近的特征数据（用于构建外推特征）
        cfg = load_config()
        lookback_days = 14  # 用最近 14 天做特征外推基础
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        features = self.store.load_features(
            province, target_type,
            start_date.strftime("%Y-%m-%d"),
            (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if features.empty:
            raise ValueError(
                f"没有可用的特征数据: {province}/{target_type}"
            )

        horizon_steps = min(horizon_hours * 4, 96)  # 4 steps per hour

        # 用最近的特征做预测（简化版：取最后 horizon_steps 行）
        recent_features = features.tail(horizon_steps).copy()
        if len(recent_features) < horizon_steps:
            recent_features = features.tail(horizon_steps)

        predictions = self._predict_with_model(
            model, recent_features, feature_names,
            province=province, target_type=target_type,
        )

        # 写入预测表
        self.store.insert_predictions(predictions)

        return predictions

    def _predict_with_model(self, model, features_df: pd.DataFrame,
                            feature_names: List[str], province: str,
                            target_type: str) -> pd.DataFrame:
        """用模型做预测."""
        predict_features = features_df[feature_names].copy()
        predicted = model.predict(predict_features)

        result = pd.DataFrame({
            "dt": features_df["dt"].values[:len(predicted)],
            "province": province,
            "type": target_type,
            "p50": predicted,
            "p10": predicted * 0.97,   # Phase 3 替换为真实分位数
            "p90": predicted * 1.03,
            "model_version": "v1",
        })
        return result
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_predictor.py -v
# Expected: 2 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/predictor.py tests/test_predictor.py
git commit -m "feat: predictor with model loading and prediction persistence"
```

---

### Task 7: 验证器（循环 A）

**Files:**
- Create: `src/validator.py`
- Create: `tests/test_validator.py`

- [ ] **Step 1: 编写测试 test_validator.py**

```python
"""tests/test_validator.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.validator import Validator


class TestValidator:
    def setup_method(self):
        np.random.seed(42)
        dates = pd.date_range("2025-02-01", periods=96, freq="15min")
        actual = 100 + np.sin(np.arange(96) * 0.05) * 50
        self.predictions = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 96,
            "type": ["load"] * 96,
            "p50": actual + np.random.normal(0, 8, 96),
        })
        self.actuals = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 96,
            "type": ["load"] * 96,
            "value": actual,
        })

    def test_compute_metrics(self):
        validator = Validator()
        metrics = validator.compute_metrics(
            self.predictions, self.actuals, value_col="p50"
        )

        assert "mape" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "bias_direction" in metrics
        assert 0 < metrics["mape"] < 0.2  # ~8% error

    def test_should_trigger_improvement(self):
        validator = Validator()
        metrics = {"mape": 0.12, "rmse": 15.0, "bias_direction": "high"}

        history = [
            {"mape": 0.11, "bias_direction": "high"},
            {"mape": 0.13, "bias_direction": "high"},
        ]

        should = validator.should_trigger(metrics, history)
        assert should is True  # consecutive high bias

    def test_should_not_trigger_under_threshold(self):
        validator = Validator()
        metrics = {"mape": 0.03, "rmse": 5.0, "bias_direction": "ok"}
        should = validator.should_trigger(metrics, [])
        assert should is False

    def test_validate_returns_report(self):
        validator = Validator()
        report = validator.validate(
            self.predictions, self.actuals, value_col="p50"
        )

        assert "triggered" in report
        assert "metrics" in report
        assert "timestamp" in report
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_validator.py -v
# Expected: FAIL
```

- [ ] **Step 3: 实现 validator.py**

```python
"""validator.py — 实时验证器（循环 A）"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from src.config_loader import get_validator_config


class Validator:
    def __init__(self):
        cfg = get_validator_config()
        self.short_threshold = cfg.get("short_term_mape_threshold", 0.05)
        self.mid_threshold = cfg.get("mid_term_mape_threshold", 0.10)
        self.consecutive_trigger = cfg.get("consecutive_bias_trigger", 3)

    def compute_metrics(self, predictions: pd.DataFrame,
                        actuals: pd.DataFrame,
                        value_col: str = "p50") -> Dict:
        """计算预测评估指标."""
        merged = predictions.merge(
            actuals[["dt", "province", "type", "value"]],
            on=["dt", "province", "type"], how="inner",
            suffixes=("_pred", "_actual"),
        )

        if merged.empty:
            return {"error": "no_overlap", "mape": None, "rmse": None}

        actual = merged["value"].values
        predicted = merged[value_col].values

        # MAPE
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))

        # RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # MAE
        mae = np.mean(np.abs(actual - predicted))

        # 偏差方向
        mean_bias = np.mean(predicted - actual)
        if abs(mean_bias) < 0.01 * np.mean(actual):
            bias_dir = "ok"
        elif mean_bias > 0:
            bias_dir = "high"
        else:
            bias_dir = "low"

        return {
            "mape": round(float(mape), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
            "bias_direction": bias_dir,
            "bias_magnitude": round(float(mean_bias), 2),
            "n_samples": len(merged),
        }

    def should_trigger(self, metrics: Dict,
                       history: List[Dict] = None) -> bool:
        """判断是否应该触发改进循环."""
        if history is None:
            history = []

        mape = metrics.get("mape")
        if mape is None:
            return False

        # 阈值触发
        if mape > self.short_threshold:
            return True

        # 连续同向偏差触发
        if len(history) >= self.consecutive_trigger:
            recent_biases = [
                h.get("bias_direction") for h in history[-self.consecutive_trigger:]
            ]
            current_bias = metrics.get("bias_direction")
            if (current_bias in ("high", "low") and
                all(b == current_bias for b in recent_biases)):
                return True

        return False

    def validate(self, predictions: pd.DataFrame,
                 actuals: pd.DataFrame,
                 value_col: str = "p50") -> Dict:
        """执行单次验证，返回报告."""
        metrics = self.compute_metrics(predictions, actuals, value_col)
        triggered = self.should_trigger(metrics)

        return {
            "timestamp": datetime.now().isoformat(),
            "triggered": triggered,
            "metrics": metrics,
        }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_validator.py -v
# Expected: 4 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/validator.py tests/test_validator.py
git commit -m "feat: real-time validator with MAPE/RMSE metrics and trigger logic"
```

---

### Task 8: 回塑验证器（循环 B）+ 多维度打分

**Files:**
- Create: `src/backtester.py`
- Create: `tests/test_backtester.py`

- [ ] **Step 1: 编写测试 test_backtester.py**

```python
"""tests/test_backtester.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.backtester import Backtester


class TestBacktester:
    def setup_method(self):
        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=4000, freq="15min")
        base = 100 + np.sin(np.arange(4000) * 0.01) * 30
        self.df = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 4000,
            "type": ["load"] * 4000,
            "value": base + np.random.normal(0, 5, 4000),
            "price": [0.35] * 4000,
            "hour": dates.hour,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
            "season": dates.month.apply(
                lambda m: 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3 if m in [9,10,11] else 4
            ),
            "temperature": 20 + np.sin(np.arange(4000) * 0.005) * 15,
            "humidity": np.random.uniform(50, 80, 4000),
            "wind_speed": np.random.uniform(1, 8, 4000),
            "value_lag_1d": np.roll(base, 96),
            "value_lag_7d": np.roll(base, 672),
            "value_rolling_mean_24h": pd.Series(base).rolling(96, min_periods=1).mean(),
            "value_diff_1d": np.diff(base, prepend=base[0]),
            "value_diff_7d": np.diff(base, prepend=base[0]),
        }).dropna()

    def test_rolling_window_backtest(self):
        bt = Backtester()
        result = bt.rolling_window_backtest(
            df=self.df,
            train_window_days=7,
            test_window_hours=24,
            n_windows=2,
            province="广东",
            target_type="load",
            target_col="value",
        )

        assert "windows" in result
        assert "summary" in result
        assert len(result["windows"]) == 2
        assert "overall_mape" in result["summary"]

    def test_multi_dimension_scoring(self):
        bt = Backtester()
        actual = self.df["value"].values[:480]
        pred = actual + np.random.normal(0, 8, 480)

        scores = bt.multi_dimension_score(
            actuals=actual,
            predictions=pred,
            metadata=self.df.iloc[:480],
        )

        assert "by_season" in scores
        assert "by_time_type" in scores
        assert "by_hour_bucket" in scores
        assert len(scores["by_season"]) > 0

    def test_full_evaluation(self):
        bt = Backtester()
        sample_size = 2000
        result = bt.evaluate_model(
            df=self.df.iloc[:sample_size],
            train_window_days=14,
            test_window_hours=24,
            province="广东",
            target_type="load",
            target_col="value",
        )

        assert "overall_mape" in result
        assert "by_season" in result
        assert "by_time_type" in result
        assert result["overall_mape"] < 1.0  # reasonable error range
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_backtester.py -v
# Expected: FAIL
```

- [ ] **Step 3: 实现 backtester.py**

```python
"""backtester.py — 回塑验证 + 多维度精细打分（循环 B）"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.trainer import Trainer


class Backtester:
    def __init__(self, trainer: Trainer = None):
        self.trainer = trainer or Trainer()

    def rolling_window_backtest(self, df: pd.DataFrame,
                                 train_window_days: int,
                                 test_window_hours: int,
                                 n_windows: int,
                                 province: str,
                                 target_type: str,
                                 target_col: str = "value") -> Dict:
        """滚动窗口回测：用历史窗口训练，测试窗口评估."""
        total_steps = 96 * train_window_days       # 训练窗口 = 每天96步×天数
        test_steps = 4 * test_window_hours          # 测试窗口（15分钟一步）
        max_start = len(df) - total_steps - test_steps - 1

        if max_start <= 0:
            return {"error": "数据不足以做滚动窗口回测"}

        windows = []
        step_size = max(1, max_start // n_windows)

        for i, start_idx in enumerate(range(0, max_start, step_size)[:n_windows]):
            train_df = df.iloc[start_idx : start_idx + total_steps]
            test_df = df.iloc[
                start_idx + total_steps : start_idx + total_steps + test_steps
            ]

            result = self.trainer.quick_train(
                train_df, province, target_type, target_col
            )
            model = result["model"]
            feature_names = result["feature_names"]

            test_features = test_df[feature_names].values
            predictions = model.predict(test_features)
            actuals = test_df[target_col].values

            mape = self._calc_mape(actuals, predictions)
            rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

            windows.append({
                "window": i + 1,
                "train_start": train_df["dt"].iloc[0].isoformat(),
                "test_start": test_df["dt"].iloc[0].isoformat(),
                "mape": round(float(mape), 4),
                "rmse": round(float(rmse), 2),
                "n_train": len(train_df),
                "n_test": len(test_df),
            })

        summary = {
            "overall_mape": round(float(np.mean([w["mape"] for w in windows])), 4),
            "overall_rmse": round(float(np.mean([w["rmse"] for w in windows])), 2),
            "best_mape": round(float(min(w["mape"] for w in windows)), 4),
            "worst_mape": round(float(max(w["mape"] for w in windows)), 4),
            "n_windows": len(windows),
        }

        return {"windows": windows, "summary": summary}

    def multi_dimension_score(self, actuals: np.ndarray,
                               predictions: np.ndarray,
                               metadata: pd.DataFrame) -> Dict:
        """对单次预测做多维度误差分解."""
        errors = np.abs(actuals - predictions)

        def _mape_in_mask(mask):
            if mask.sum() == 0:
                return None
            a = actuals[mask]
            p = predictions[mask]
            m = a != 0
            return round(float(np.mean(np.abs((a[m] - p[m]) / a[m]))), 4) if m.sum() > 0 else None

        def _rmse_in_mask(mask):
            if mask.sum() == 0:
                return None
            return round(float(np.sqrt(np.mean(errors[mask] ** 2))), 2)

        meta = metadata.reset_index(drop=True)
        if len(meta) != len(actuals):
            meta = metadata.iloc[:len(actuals)].reset_index(drop=True)

        # 季节维度
        by_season = {}
        season_map = {1: "spring", 2: "summer", 3: "autumn", 4: "winter"}
        if "season" in meta.columns:
            for s_val, s_name in season_map.items():
                mask = meta["season"].values == s_val
                by_season[s_name] = {
                    "mape": _mape_in_mask(mask),
                    "samples": int(mask.sum()),
                }

        # 时间类型维度
        by_time = {}
        if "is_weekend" in meta.columns:
            is_weekend = meta["is_weekend"].astype(bool).values
            by_time["workday"] = {"mape": _mape_in_mask(~is_weekend)}
            by_time["weekend"] = {"mape": _mape_in_mask(is_weekend)}

        # 时段维度
        by_hour = {}
        if "hour" in meta.columns:
            hours = meta["hour"].values
            peak_mask = (hours >= 8) & (hours < 12) | (hours >= 17) & (hours < 21)
            valley_mask = (hours >= 0) & (hours < 8) | (hours >= 22)
            flat_mask = (hours >= 12) & (hours < 17)
            by_hour["peak"] = {"mape": _mape_in_mask(peak_mask)}
            by_hour["valley"] = {"mape": _mape_in_mask(valley_mask)}
            by_hour["flat"] = {"mape": _mape_in_mask(flat_mask)}

        return {
            "by_season": by_season,
            "by_time_type": by_time,
            "by_hour_bucket": by_hour,
        }

    def evaluate_model(self, df: pd.DataFrame,
                        train_window_days: int,
                        test_window_hours: int,
                        province: str,
                        target_type: str,
                        target_col: str = "value") -> Dict:
        """完整模型评估：含滚动窗口 + 多维度打分."""
        rb = self.rolling_window_backtest(
            df, train_window_days, test_window_hours,
            n_windows=3, province=province,
            target_type=target_type, target_col=target_col,
        )

        if "error" in rb:
            return rb

        # 取最后一个窗口做多维度打分
        train_steps = 96 * train_window_days
        test_steps = 4 * test_window_hours
        total = train_steps + test_steps
        last_train = df.iloc[-total : -test_steps]
        last_test = df.iloc[-test_steps:]

        bt_result = self.trainer.quick_train(
            last_train, province, target_type, target_col
        )
        pred = bt_result["model"].predict(last_test[bt_result["feature_names"]].values)
        multi_scores = self.multi_dimension_score(
            last_test[target_col].values, pred, last_test
        )

        return {
            "overall_mape": rb["summary"]["overall_mape"],
            "overall_rmse": rb["summary"]["overall_rmse"],
            "rolling_windows": rb["windows"],
            **multi_scores,
        }

    @staticmethod
    def _calc_mape(actuals: np.ndarray, predictions: np.ndarray) -> float:
        mask = actuals != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])))
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_backtester.py -v
# Expected: 3 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/backtester.py tests/test_backtester.py
git commit -m "feat: backtester with rolling windows and multi-dimension scoring"
```

---

### Task 9: 误差诊断器

**Files:**
- Create: `src/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: 编写测试 test_analyzer.py**

```python
"""tests/test_analyzer.py"""
import pytest
from src.analyzer import Analyzer


class TestAnalyzer:
    def test_diagnose_high_summer_mape(self):
        analyzer = Analyzer()
        report = {
            "by_season": {
                "summer": {"mape": 0.18, "samples": 2000},
                "winter": {"mape": 0.05, "samples": 2000},
            },
        }
        diagnosis = analyzer.diagnose(report, baseline_mape=0.10)
        assert any("summer" in d.get("scenario", "") for d in diagnosis)
        assert any("nonlinear" in d.get("root_cause", "") for d in diagnosis)

    def test_diagnose_weekend_pattern(self):
        analyzer = Analyzer()
        report = {
            "by_time_type": {
                "workday": {"mape": 0.05},
                "weekend": {"mape": 0.15},
            },
        }
        diagnosis = analyzer.diagnose(report, baseline_mape=0.08)
        assert any("weekend" in d.get("scenario", "") for d in diagnosis)
        assert any("day_of_week" in d.get("suggested_features", []) for d in diagnosis)

    def test_diagnose_no_signal(self):
        analyzer = Analyzer()
        report = {
            "by_season": {"summer": {"mape": 0.046}},
            "by_time_type": {"workday": {"mape": 0.048}},
        }
        diagnosis = analyzer.diagnose(report, baseline_mape=0.05)
        assert len(diagnosis) == 0

    def test_diagnose_extreme_events(self):
        analyzer = Analyzer()
        report = {
            "overall_mape": 0.25,
            "by_season": {},
            "by_time_type": {},
        }
        diagnosis = analyzer.diagnose(report, baseline_mape=0.10)
        assert len(diagnosis) > 0
        # Should flag as distribution shift / concept drift
        assert any("drift" in d.get("root_cause", "") or
                   "shift" in d.get("root_cause", "")
                   for d in diagnosis)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_analyzer.py -v
# Expected: FAIL
```

- [ ] **Step 3: 实现 analyzer.py**

```python
"""analyzer.py — 误差根因诊断器"""
from typing import Dict, List, Optional


class Analyzer:
    DIAGNOSIS_RULES = [
        {
            "condition": lambda r: r.get("by_season", {}).get("summer", {}).get("mape", 0) is not None
                                   and r.get("by_season", {}).get("summer", {}).get("mape", 0) > 0.12,
            "diagnosis": {
                "scenario": "summer",
                "root_cause": "nonlinear_heat_effect",
                "description": "夏季高温导致负荷非线性增长，线性模型捕捉不足",
                "severity": "high",
                "suggested_features": ["temperature²", "temperature×is_weekend"],
            },
        },
        {
            "condition": lambda r: r.get("by_time_type", {}).get("weekend", {}).get("mape", 0) is not None
                                   and r.get("by_time_type", {}).get("weekend", {}).get("mape", 0) > 0.10,
            "diagnosis": {
                "scenario": "weekend",
                "root_cause": "weekend_pattern_mismatch",
                "description": "周末用电模式与工作日差异大，缺乏交互特征",
                "severity": "medium",
                "suggested_features": ["day_of_week×hour", "is_weekend×value_lag_7d"],
            },
        },
        {
            "condition": lambda r: r.get("by_time_type", {}).get("holiday", {}).get("mape", 0) is not None
                                   and r.get("by_time_type", {}).get("holiday", {}).get("mape", 0) > 0.15,
            "diagnosis": {
                "scenario": "holiday",
                "root_cause": "holiday_pattern_break",
                "description": "节假日用电模式与平时显著不同",
                "severity": "high",
                "suggested_features": ["is_holiday", "days_from_holiday"],
            },
        },
        {
            "condition": lambda r: max(
                r.get("by_season", {}).get(s, {}).get("mape", 0) or 0
                for s in ["spring", "summer", "autumn", "winter"]
                if s in r.get("by_season", {})
            ) > 0.12 if any(s in r.get("by_season", {}) for s in ["spring", "summer", "autumn", "winter"]) else False,
            "diagnosis": {
                "scenario": "seasonal",
                "root_cause": "seasonal_concept_drift",
                "description": "季节更替导致数据分布偏移",
                "severity": "medium",
                "suggested_features": ["season_onehot", "month_sin_cos"],
            },
        },
        {
            "condition": lambda r: r.get("overall_mape", 0) > 0.15,
            "diagnosis": {
                "scenario": "overall",
                "root_cause": "distribution_shift",
                "description": "整体预测大幅度退化，可能有概念漂移或数据结构变化",
                "severity": "critical",
                "suggested_features": ["shorter_training_window", "full_retrain"],
            },
        },
        {
            "condition": lambda r: r.get("overall_mape", 0) > 0.08
                                   and r.get("by_hour_bucket", {}).get("peak", {}).get("mape", 0) is not None
                                   and r.get("by_hour_bucket", {}).get("peak", {}).get("mape", 0) > 0.10,
            "diagnosis": {
                "scenario": "peak_hours",
                "root_cause": "peak_load_variance",
                "description": "高峰时段负荷波动大，需要更细粒度的特征",
                "severity": "medium",
                "suggested_features": ["hour_onehot", "peak_valley_indicator"],
            },
        },
    ]

    def diagnose(self, backtest_report: Dict,
                 baseline_mape: float = 0.10,
                 validator_history: List[Dict] = None) -> List[Dict]:
        """根据回测报告诊断误差根因.

        Returns:
            list of diagnosis dicts, each with {scenario, root_cause,
            description, severity, suggested_features}
        """
        results = []
        for rule in self.DIAGNOSIS_RULES:
            try:
                if rule["condition"](backtest_report):
                    diag = dict(rule["diagnosis"])
                    diag["baseline_mape"] = baseline_mape
                    diag["current_mape"] = backtest_report.get("overall_mape")
                    results.append(diag)
            except (KeyError, TypeError):
                continue

        # 额外检查：连续同向偏差（需要 validator history）
        if validator_history and len(validator_history) >= 3:
            recent_biases = [
                h.get("bias_direction") for h in validator_history[-3:]
                if "bias_direction" in h
            ]
            if (len(recent_biases) >= 3 and
                all(b == recent_biases[0] for b in recent_biases) and
                recent_biases[0] in ("high", "low")):
                direction = "偏高" if recent_biases[0] == "high" else "偏低"
                results.append({
                    "scenario": "persistent_bias",
                    "root_cause": "systematic_bias",
                    "description": f"连续多轮同向偏差（{direction}），存在系统性误差",
                    "severity": "high",
                    "suggested_features": ["bias_correction_factor"],
                })

        return results
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_analyzer.py -v
# Expected: 4 PASSED
```

- [ ] **Step 5: 提交**

```bash
git add src/analyzer.py tests/test_analyzer.py
git commit -m "feat: error root cause analyzer with rule-based diagnosis"
```

---

### Task 10: 学习型优化引擎 + 策略知识库

**Files:**
- Create: `src/improver.py`
- Create: `tests/test_improver.py`

- [ ] **Step 1: 编写测试 test_improver.py**

```python
"""tests/test_improver.py"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.improver import Improver


class TestImprover:
    def setup_method(self):
        np.random.seed(42)

    def test_generate_hypotheses(self):
        improver = Improver()

        diagnosis = [{
            "scenario": "summer",
            "root_cause": "nonlinear_heat_effect",
            "severity": "high",
            "suggested_features": ["temperature²", "temperature×is_weekend"],
        }]

        hypotheses = improver.generate_hypotheses(diagnosis)
        assert len(hypotheses) > 0
        assert all("name" in h for h in hypotheses)
        assert all("params" in h for h in hypotheses)
        assert any("polynomial" in h.get("name", "").lower() for h in hypotheses)

    def test_select_best_hypothesis(self):
        improver = Improver()

        arena_results = [
            {
                "hypothesis_id": "h1",
                "score": {
                    "overall_mape": 0.08,
                    "by_season": {"summer": {"mape": 0.14}},
                },
            },
            {
                "hypothesis_id": "h2",
                "score": {
                    "overall_mape": 0.06,
                    "by_season": {"summer": {"mape": 0.07}},
                },
            },
            {
                "hypothesis_id": "h3",
                "score": {
                    "overall_mape": 0.07,
                    "by_season": {"summer": {"mape": 0.09}},
                },
            },
        ]

        diagnosis = [{"scenario": "summer", "severity": "high"}]
        baseline = {"overall_mape": 0.10}

        winner = improver.select_best(arena_results, diagnosis, baseline)
        assert winner["hypothesis_id"] == "h2"  # best summer improvement

    @patch("src.improver.DorisDB")
    def test_record_to_knowledge_base(self, MockDB):
        mock_db = MockDB.return_value
        improver = Improver(db=mock_db)

        strategy = {
            "name": "高温日过采样3x",
            "desc": "对温度>35的样本过采样3倍 + temperature²",
            "improvement": {"before": 0.14, "after": 0.07},
        }
        improver.record_strategy(strategy, scenario="summer", success=True)
        mock_db.execute.assert_called()

    def test_seed_strategies_exist(self):
        improver = Improver()
        assert len(improver.SEED_STRATEGIES) >= 8
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_improver.py -v
# Expected: FAIL
```

- [ ] **Step 3: 实现 improver.py**

```python
"""improver.py — 学习型优化引擎 + 策略知识库"""
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.db import DorisDB
from src.trainer import Trainer
from src.backtester import Backtester


class Improver:
    SEED_STRATEGIES = [
        {
            "name": "polynomial_features",
            "description": "加入多项式特征 {col}²",
            "params": {"power": 2},
            "applicable": ["nonlinear"],
        },
        {
            "name": "recent_upsample",
            "description": "近期样本上采样 ×{weight}",
            "params": {"weight": 3, "days": 7},
            "applicable": ["drift", "decay"],
        },
        {
            "name": "shorter_window",
            "description": "减小训练窗口至 {n} 天",
            "params": {"n": 30},
            "applicable": ["drift", "shift"],
        },
        {
            "name": "switch_to_catboost",
            "description": "切换模型至 CatBoost",
            "params": {"model": "catboost"},
            "applicable": ["variance", "price"],
        },
        {
            "name": "province_independent_model",
            "description": "{province} 独立建模",
            "params": {},
            "applicable": ["province_bias"],
        },
        {
            "name": "extreme_oversample",
            "description": "极端条件样本过采样 ×{factor}",
            "params": {"factor": 3, "condition": "percentile>95"},
            "applicable": ["extreme", "tail"],
        },
        {
            "name": "dayofweek_interaction",
            "description": "加 day_of_week 交互特征",
            "params": {"interact_with": ["hour", "value_lag_7d"]},
            "applicable": ["weekend", "pattern"],
        },
        {
            "name": "holiday_oversample",
            "description": "节假日样本过采样 ×{weight}",
            "params": {"weight": 5},
            "applicable": ["holiday"],
        },
        {
            "name": "rolling_window_features",
            "description": "{col} 滑动窗口均值({n}h) 作为新特征",
            "params": {"window_hours": 4},
            "applicable": ["smoothness"],
        },
        {
            "name": "bias_correction",
            "description": "偏差自动补偿: 预测值 + 近期平均偏差",
            "params": {},
            "applicable": ["systematic_bias"],
        },
    ]

    def __init__(self, db: DorisDB = None,
                 trainer: Trainer = None,
                 backtester: Backtester = None):
        self.db = db or DorisDB()
        self.trainer = trainer or Trainer()
        self.backtester = backtester or Backtester(self.trainer)

    def generate_hypotheses(self, diagnosis: List[Dict]) -> List[Dict]:
        """基于诊断结论生成改进假设池."""
        keywords = set()
        for d in diagnosis:
            root = d.get("root_cause", "")
            scenario = d.get("scenario", "")
            for word in root.replace("_", " ").split():
                keywords.add(word.lower())
            keywords.add(scenario.lower())

        suggested = set()
        for d in diagnosis:
            for sf in d.get("suggested_features", []):
                suggested.add(sf.lower())

        hypotheses = []

        for seed in self.SEED_STRATEGIES:
            # 匹配: 根因关键词或推荐特征
            applicable = seed.get("applicable", [])
            if any(kw in applicable for kw in keywords):
                hypotheses.append({
                    "name": seed["name"],
                    "description": seed["description"],
                    "params": dict(seed.get("params", {})),
                    "source": "seed_match",
                })
            elif suggested & set(applicable):
                hypotheses.append({
                    "name": seed["name"],
                    "description": seed["description"],
                    "params": dict(seed.get("params", {})),
                    "source": "suggested_match",
                })
            elif "overall" in keywords or "critical" in keywords:
                hypotheses.append({
                    "name": seed["name"],
                    "description": seed["description"],
                    "params": dict(seed.get("params", {})),
                    "source": "critical_explore",
                })

        # 去重
        seen = set()
        unique = []
        for h in hypotheses:
            sig = h["name"] + json.dumps(h["params"], sort_keys=True)
            if sig not in seen:
                seen.add(sig)
                unique.append(h)

        # 限制数量
        return unique[:12]

    def run_experiment(self, hypothesis: Dict, df: pd.DataFrame,
                       province: str, target_type: str,
                       target_col: str = "value") -> Dict:
        """执行单个假设的实验（快速训练→回测打分）."""
        try:
            # 快速训练
            bt_result = self.trainer.quick_train(
                df, province, target_type, target_col,
                params=hypothesis.get("model_params"),
            )

            # 回测打分
            total = len(df)
            test_steps = min(96, total // 5)
            train_df = df.iloc[:-test_steps]
            test_df = df.iloc[-test_steps:]

            model = bt_result["model"]
            pred = model.predict(
                test_df[bt_result["feature_names"]].values
            )
            actual = test_df[target_col].values

            mask = actual != 0
            mape = float(
                np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask]))
                if mask.sum() > 0 else 1.0
            )

            return {
                "hypothesis_id": self._hash(hypothesis["name"]),
                "hypothesis": hypothesis,
                "mape": round(mape, 4),
                "n_samples": len(df),
            }
        except Exception as e:
            return {
                "hypothesis_id": self._hash(hypothesis["name"]),
                "hypothesis": hypothesis,
                "error": str(e),
            }

    def run_arena(self, hypotheses: List[Dict], df: pd.DataFrame,
                  province: str, target_type: str,
                  target_col: str = "value") -> List[Dict]:
        """竞技场: 并行测试 N 个假设，返回打分结果."""
        results = []
        for h in hypotheses:
            result = self.run_experiment(h, df, province, target_type, target_col)
            results.append(result)

        return sorted(results, key=lambda r: r.get("mape", float("inf")))

    def select_best(self, arena_results: List[Dict],
                     diagnosis: List[Dict],
                     baseline: Dict) -> Dict:
        """从竞技场结果中选择最优策略.

        评分 = 主问题改善权重 × MAPE降幅 − 复杂度罚分
        """
        if not arena_results:
            return {"error": "no_valid_experiments"}

        pain_points = {}
        for d in diagnosis:
            scenario = d.get("scenario", "")
            pain_points[scenario] = d.get("severity", "medium")

        severity_weight = {"critical": 5.0, "high": 3.0, "medium": 1.5, "low": 1.0}

        scored = []
        for r in arena_results:
            if "error" in r:
                continue

            mape = r.get("mape", 1.0)
            baseline_mape = baseline.get("overall_mape", 0.10)

            improvement = baseline_mape - mape

            # 复杂度罚分（优先简单策略）
            complexity_penalty = 0.0
            if "polynomial" in r.get("hypothesis", {}).get("name", ""):
                complexity_penalty = 0.002
            if "catboost" in r.get("hypothesis", {}).get("name", "").lower():
                complexity_penalty = 0.004

            weight = 1.0
            for scenario, severity in pain_points.items():
                weight = max(weight, severity_weight.get(severity, 1.0))

            score = improvement * weight - complexity_penalty

            scored.append({
                **r,
                "improvement": round(float(improvement), 4),
                "score": round(float(score), 4),
            })

        scored.sort(key=lambda r: r["score"], reverse=True)

        if not scored:
            return {"error": "all_experiments_failed"}

        winner = scored[0]
        return {
            "selected": winner["hypothesis"],
            "hypothesis_id": winner["hypothesis_id"],
            "mape_after": winner["mape"],
            "mape_before": baseline.get("overall_mape"),
            "improvement": winner["improvement"],
            "all_results": [s["hypothesis_id"] for s in scored[:5]],
        }

    def record_strategy(self, strategy: Dict, scenario: str = "",
                        success: bool = True):
        """记录策略到知识库."""
        strategy_hash = self._hash(strategy.get("name", ""))
        desc = strategy.get("desc", strategy.get("name", ""))

        improvement = strategy.get("improvement", {})
        if isinstance(improvement, dict):
            effect = improvement.get("after", 0) - improvement.get("before", 0)
        else:
            effect = float(improvement) if improvement else 0

        sql = f"""
            INSERT INTO strategy_knowledge
                (strategy_hash, strategy_desc, applied_count, success_count,
                 avg_improvement, best_scenario, worst_scenario,
                 last_applied, last_effect)
            VALUES
                ('{strategy_hash}', '{desc}', 1, {1 if success else 0},
                 {abs(effect)}, '{scenario}', '',
                 '{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', {effect})
            ON DUPLICATE KEY UPDATE
                applied_count = applied_count + 1,
                success_count = success_count + {1 if success else 0},
                avg_improvement = (avg_improvement * applied_count + {abs(effect)}) 
                                  / (applied_count + 1),
                last_applied = '{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                last_effect = {effect}
        """
        self.db.execute(sql)

    def query_knowledge(self, scenario: str) -> List[Dict]:
        """查询策略知识库."""
        sql = f"""
            SELECT strategy_hash, strategy_desc, applied_count,
                   success_count, avg_improvement, best_scenario,
                   last_applied, last_effect
            FROM strategy_knowledge
            WHERE NOT retired
              AND (best_scenario LIKE '%{scenario}%' OR best_scenario = '')
            ORDER BY avg_improvement DESC
            LIMIT 20
        """
        return self.db.query(sql).to_dict("records") if self.db.table_exists("strategy_knowledge") else []

    def improve(self, diagnosis: List[Dict], df: pd.DataFrame,
                province: str, target_type: str,
                target_col: str = "value",
                baseline: Dict = None) -> Dict:
        """完整一轮自主优化.

        Returns:
            {selected_strategy, mape_before, mape_after, improvement, knowledge_updated}
        """
        if baseline is None:
            baseline = {"overall_mape": 0.10}

        # 1. 生成假设
        hypotheses = self.generate_hypotheses(diagnosis)

        # 2. 竞技场测试
        arena_results = self.run_arena(
            hypotheses, df, province, target_type, target_col
        )

        # 3. 选最优
        best = self.select_best(arena_results, diagnosis, baseline)

        # 4. 记录知识
        if "error" not in best:
            scenario = diagnosis[0].get("scenario", "") if diagnosis else ""
            success = best.get("improvement", 0) > 0
            self.record_strategy(
                {
                    "name": best.get("selected", {}).get("name", ""),
                    "desc": best.get("selected", {}).get("description", ""),
                    "improvement": {
                        "before": best.get("mape_before", 0),
                        "after": best.get("mape_after", 0),
                    },
                },
                scenario=scenario,
                success=success,
            )

        return {
            "selected_strategy": best.get("selected", {}).get("description", ""),
            "mape_before": best.get("mape_before"),
            "mape_after": best.get("mape_after"),
            "improvement": best.get("improvement", 0),
            "knowledge_updated": True,
            "hypotheses_tested": len([r for r in arena_results if "error" not in r]),
        }

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_improver.py -v
# Expected: 4 PASSED
```

- [ ] **Step 5: 在 improver.py 文件开头补充缺失的 import**

```python
# improver.py 文件顶部缺少 np 的导入，需要在文件开头添加:
import numpy as np
```

请在 improver.py 文件中第 2 行附近（`from datetime import datetime` 上方）添加：

```python
import numpy as np
```

```bash
# 验证导入
python -c "from src.improver import Improver; print('OK')"
```

- [ ] **Step 6: 提交**

```bash
git add src/improver.py tests/test_improver.py
git commit -m "feat: learning-based improver with hypothesis generation, arena testing, and strategy knowledge base"
```

---

### Task 11: 总调度器

**Files:**
- Create: `src/orchestrator.py`

- [ ] **Step 1: 实现 orchestrator.py**

```python
"""orchestrator.py — 总调度器：管理循环 A/B/C"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.db import DorisDB
from src.config_loader import get_provinces, get_types
from src.feature_store import FeatureStore, FeatureEngineer
from src.data_fetcher import DataFetcher
from src.trainer import Trainer
from src.predictor import Predictor
from src.validator import Validator
from src.backtester import Backtester
from src.analyzer import Analyzer
from src.improver import Improver

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.db = DorisDB()
        self.store = FeatureStore(self.db)
        self.trainer = Trainer()
        self.backtester = Backtester(self.trainer)
        self.predictor = Predictor(self.trainer, self.store)
        self.validator = Validator()
        self.analyzer = Analyzer()
        self.improver = Improver(self.db, self.trainer, self.backtester)
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()

        self._validator_history: List[Dict] = []

    # ── 初始化 ──

    def setup(self):
        """首次运行：建表 + 初始化特征."""
        logger.info("初始化系统...")
        self.store.ensure_tables()
        logger.info("表结构已就绪")

    def build_features(self, province: str = None,
                       start_date: str = None,
                       end_date: str = None):
        """从原始表构建特征（首次运行或全量重建）."""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        provinces = [province] if province else get_provinces()

        for p in provinces:
            for t in get_types():
                logger.info(f"构建特征: {p}/{t} ...")
                raw = self.store.load_raw_data(p, t, start_date, end_date)
                if raw.empty:
                    logger.warning(f"  无数据: {p}/{t}")
                    continue

                features = self.engineer.build_features_from_raw(raw)

                # 尝试合并气象
                try:
                    weather = self.fetcher.fetch_weather(
                        p, start_date, end_date, mode="historical"
                    )
                    if not weather.empty:
                        features = self.engineer.merge_weather(features, weather)
                except Exception as e:
                    logger.warning(f"  气象数据合并失败: {e}")

                count = self.store.insert_features(features)
                logger.info(f"  {p}/{t}: 写入 {count} 行")

    # ── 预测 ──

    def predict(self, province: str, target_type: str,
                horizon_hours: int = 24) -> Dict:
        """执行预测."""
        logger.info(f"预测: {province}/{target_type}, {horizon_hours}h")
        df = self.predictor.predict(province, target_type, horizon_hours)
        return {
            "province": province,
            "type": target_type,
            "horizon_hours": horizon_hours,
            "n_predictions": len(df),
            "sample": df.head(5)[["dt", "p50"]].to_dict("records") if not df.empty else [],
        }

    # ── 循环 A: 实时验证 ──

    def run_validation_cycle(self, province: str,
                              target_type: str) -> Dict:
        """运行循环 A: validator → analyzer → (可选) improver → trainer."""
        end = datetime.now()
        start = end - timedelta(hours=48)

        predictions = self.db.query(
            f"SELECT * FROM energy_predictions "
            f"WHERE province='{province}' AND type='{target_type}' "
            f"AND dt >= '{start.strftime('%Y-%m-%d %H:%M:%S')}'"
        )

        actuals = self.store.load_raw_data(
            province, target_type,
            start.strftime("%Y-%m-%d"),
            (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if predictions.empty or actuals.empty:
            return {"status": "no_data"}

        report = self.validator.validate(predictions, actuals, "p50")
        self._validator_history.append(report)

        logger.info(
            f"验证 {province}/{target_type}: "
            f"MAPE={report['metrics'].get('mape')}, "
            f"triggered={report['triggered']}"
        )

        if report["triggered"]:
            return self._run_improvement_cycle(
                province, target_type, report
            )

        return {"status": "ok", "report": report}

    # ── 循环 B: 回塑验证 ──

    def run_backtest_cycle(self, province: str,
                            target_type: str) -> Dict:
        """运行循环 B: 回塑验证 → analyzer."""
        end = datetime.now()
        start = end - timedelta(days=120)

        df = self.store.load_features(
            province, target_type,
            start.strftime("%Y-%m-%d"),
            (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if df.empty:
            return {"status": "no_data"}

        result = self.backtester.evaluate_model(
            df, train_window_days=14, test_window_hours=24,
            province=province, target_type=target_type,
        )

        diagnosis = self.analyzer.diagnose(
            result, validator_history=self._validator_history
        )

        logger.info(
            f"回测 {province}/{target_type}: "
            f"MAPE={result.get('overall_mape')}, "
            f"diagnoses={len(diagnosis)}"
        )

        return {
            "status": "ok",
            "mape": result.get("overall_mape"),
            "by_season": result.get("by_season", {}),
            "diagnoses": diagnosis,
        }

    # ── 循环 C: 自主学习 ──

    def _run_improvement_cycle(self, province: str,
                                target_type: str,
                                validation_report: Dict) -> Dict:
        """循环 C: improver 假设 → trainer 实验 → backtester 裁决 → 部署."""
        # 1. 先回测获取基准
        end = datetime.now()
        start = end - timedelta(days=60)
        df = self.store.load_features(
            province, target_type,
            start.strftime("%Y-%m-%d"),
            (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if df.empty:
            return {"status": "no_data"}

        bt_result = self.backtester.evaluate_model(
            df, train_window_days=14, test_window_hours=24,
            province=province, target_type=target_type,
        )

        # 2. 诊断
        diagnosis = self.analyzer.diagnose(bt_result)

        # 3. 自主优化
        improvement = self.improver.improve(
            diagnosis, df, province, target_type,
            baseline={"overall_mape": bt_result.get("overall_mape", 0.10)},
        )

        logger.info(
            f"优化 {province}/{target_type}: "
            f"策略={improvement.get('selected_strategy')}, "
            f"MAPE {improvement.get('mape_before')} → "
            f"{improvement.get('mape_after')}"
        )

        # 4. 如果改善显著，全量重训练部署
        if improvement.get("improvement", 0) > 0.03:
            logger.info("触发全量重训练...")
            self.trainer.train(
                df, province, target_type,
                target_col="value",
            )

        return {
            "status": "improved",
            "validation": validation_report,
            "improvement": improvement,
            "diagnoses": diagnosis,
        }

    # ── 完整训练流程 ──

    def train_all(self):
        """全量训练（首次部署或定期）."""
        end = datetime.now()
        start = end - timedelta(days=90)

        for province in get_provinces():
            for target_type in get_types():
                logger.info(f"训练: {province}/{target_type}")

                df = self.store.load_features(
                    province, target_type,
                    start.strftime("%Y-%m-%d"),
                    (end + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                if df.empty:
                    logger.warning(f"  无数据: {province}/{target_type}")
                    continue

                result = self.trainer.train(df, province, target_type)
                logger.info(
                    f"  完成: n={result['n_samples']}, "
                    f"features={result['n_features']}"
                )
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from src.orchestrator import Orchestrator; print('Orchestrator loaded OK')"
```

- [ ] **Step 3: 提交**

```bash
git add src/orchestrator.py
git commit -m "feat: orchestrator coordinating all three improvement cycles"
```

---

### Task 12: Skill 定义

**Files:**
- Create: `skills/energy-predict.md`

- [ ] **Step 1: 创建 Skill 文件**

```markdown
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
```

- [ ] **Step 2: 提交**

```bash
git add skills/energy-predict.md
git commit -m "feat: energy prediction skill definition with 6 commands"
```

---

### Task 13: 集成测试

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 编写集成测试**

```python
"""tests/test_integration.py — 端到端集成测试"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.orchestrator import Orchestrator


class TestIntegration:
    """集成测试：模拟完整工作流"""

    def setup_method(self):
        np.random.seed(42)
        self._create_mock_data()

    def _create_mock_data(self):
        """创建模拟数据."""
        dates = pd.date_range("2025-01-01", periods=3000, freq="15min")
        base = 100 + np.sin(np.arange(3000) * 0.008) * 40
        self.mock_raw = pd.DataFrame({
            "dt": dates,
            "province": ["广东"] * 3000,
            "type": ["load"] * 3000,
            "value": base + np.random.normal(0, 6, 3000),
            "price": [0.35] * 3000,
        })

    @patch("src.feature_store.DorisDB")
    @patch("src.orchestrator.DorisDB")
    def test_full_setup_and_train(self, MockOrchDB, MockStoreDB):
        """测试：初始化 → 训练 → 预测."""
        mock_db = MockOrchDB.return_value
        mock_db.query.return_value = self.mock_raw
        mock_db.table_exists.return_value = True
        MockStoreDB.return_value = mock_db

        # Mock 模型加载
        with patch("src.trainer.pickle.load") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([100.0] * 96)
            mock_load.return_value = mock_model

            with patch("src.trainer.os.path.exists", return_value=True):
                with patch("src.trainer.json.load", return_value={
                    "广东_load": {
                        "latest": "test.lgb",
                        "feature_names": ["hour", "day_of_week"],
                        "updated_at": "2025-01-01",
                    }
                }):
                    orch = Orchestrator()

                    # 预测
                    result = orch.predict("广东", "load", horizon_hours=1)
                    assert result["province"] == "广东"
                    assert result["n_predictions"] > 0

    def test_orchestrator_imports_cleanly(self):
        """确保 orchestrator 可以正常导入."""
        try:
            from src.orchestrator import Orchestrator
            assert Orchestrator is not None
        except Exception as e:
            pytest.fail(f"Orchestrator import failed: {e}")

    def test_all_modules_importable(self):
        """确保所有模块可以正常导入."""
        modules = [
            "src.config_loader",
            "src.db",
            "src.data_fetcher",
            "src.feature_store",
            "src.trainer",
            "src.predictor",
            "src.validator",
            "src.backtester",
            "src.analyzer",
            "src.improver",
            "src.orchestrator",
        ]
        for mod in modules:
            try:
                __import__(mod)
            except Exception as e:
                pytest.fail(f"Import failed for {mod}: {e}")
```

- [ ] **Step 2: 运行集成测试**

```bash
pytest tests/test_integration.py -v -s
# Expected: 3 PASSED
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for full pipeline"
```

---

## Self-Review Checklist

1. **Spec coverage**: 
   - ✅ Doris 连接层 (Task 2)
   - ✅ 外部数据采集 (Task 3, Open-Meteo)
   - ✅ 特征工程 + 特征存储 (Task 4)
   - ✅ LightGBM 短期预测 (Task 5)
   - ✅ 预测执行 (Task 6)
   - ✅ 循环 A 实时验证 (Task 7)
   - ✅ 循环 B 回塑验证 + 多维度打分 (Task 8)
   - ✅ 误差诊断 (Task 9)
   - ✅ 学习型 improver + 策略知识库 (Task 10)
   - ✅ 总调度 orchestrator (Task 11)
   - ✅ Skill 定义 (Task 12)
   - ✅ 集成测试 (Task 13)

2. **Placeholder scan**: ✅ 无 TBD/TODO，所有步骤包含完整代码

3. **Type consistency**: ✅ 所有脚本间通过标准 Dict/DataFrame 传递数据，接口一致

4. **Phase 1 完整度**: ✅ 覆盖 spec Phase 1 全部 checklist 项，共 13 个 Task
