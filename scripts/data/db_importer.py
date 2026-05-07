"""db_importer.py — 数据库导入器 (MySQL / Doris / MySQL 兼容协议)

从 MySQL/Doris 读取电力数据，JOIN 维度表，构造三段式 type 字符串，导出为 /import 可消费的 CSV。
Doris 兼容 MySQL 协议，使用相同的 pymysql 驱动，通过 data_source 参数切换配置段。

维度映射:
  t_data_quality:   1=出力预测, 2=出力实际, 3=负荷预测, 4=负荷实际, 5=出清电量
  t_power_source:   power_source_id → 风电/光伏/水电/火电/核电/...
  t_price_market:   price_market_id → 日前/实时/中长期/日前出清/实时出清
  t_date:           day_type (工作日/法定节假日/调休节假日)
  t_time_point:     hour, minute

电价特殊性: f_price_15min 没有 data_quality_id，所有电价数据都是实际交易价格 (value_type=实际)。
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import pymysql

logger = logging.getLogger(__name__)


def _query_to_df(conn: pymysql.Connection, sql: str) -> pd.DataFrame:
    """Execute SQL and return DataFrame using pymysql cursor (no SQLAlchemy needed)."""
    with conn.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        columns = [col[0] for col in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _query_chunks(conn: pymysql.Connection, sql: str, chunk_size: int):
    """Generator yielding DataFrames in chunks."""
    offset = 0
    while True:
        chunk_sql = f"{sql} LIMIT {chunk_size} OFFSET {offset}"
        df = _query_to_df(conn, chunk_sql)
        if df.empty:
            break
        yield df
        if len(df) < chunk_size:
            break
        offset += chunk_size


class EnergyDBImporter:
    """从 MySQL/Doris 读电力数据 → DataFrame / CSV."""

    # data_quality_id → (base_type, value_type)
    DATA_QUALITY_MAP = {
        1: ("出力", "预测"),
        2: ("出力", "实际"),
        3: ("负荷", "预测"),
        4: ("负荷", "实际"),
    }

    # power_source_id → sub_type (简化中文名)
    POWER_SOURCE_MAP = {
        1:  "总",
        3:  "风电",
        4:  "光伏",
        5:  "水电",
        6:  "水电含抽蓄",
        7:  "核电",
        8:  "火电",
        9:  "自备",
        10: "统调",
        11: "A类",
        12: "B类",
        13: "地方",
        14: "西电",
        20: "系统",
        21: "直调",
        22: "联络线",
        35: "试验",
        36: "储能",
        40: "非市场",
    }

    # price_market_id → sub_type
    PRICE_MARKET_MAP = {
        1: "日前",
        2: "实时",
        3: "中长期",
        4: "日前出清",
        5: "实时出清",
    }

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 user: Optional[str] = None, password: Optional[str] = None,
                 database: Optional[str] = None,
                 data_source: str = "auto",
                 connection: Optional[pymysql.Connection] = None):
        """初始化数据库导入器。

        Args:
            host/port/user/password/database: 直接指定连接参数
            data_source: 配置段名称 — "mysql" | "doris" | "auto"
                        "auto" 从 config 的 data_source 字段推断
            connection: 外部传入的连接对象 (优先级最高)
        """
        self._external_conn = connection

        if host is None:
            from scripts.core.config import load_config
            cfg = load_config()

            # 自动推断: config 的 data_source 字段
            if data_source == "auto":
                ds = cfg.get("data_source", "file")
                if ds in ("mysql", "doris"):
                    data_source = ds
                else:
                    data_source = "mysql"  # 默认

            db_cfg = cfg.get(data_source, {})
            self.host = db_cfg.get("host", "127.0.0.1")
            self.port = int(db_cfg.get("port", 9030 if data_source == "doris" else 3306))
            self.user = db_cfg.get("user", "root")
            self.password = db_cfg.get("password", "")
            self.database = db_cfg.get("database", "electric_power_db")
            self._source_type = data_source
        else:
            self.host = host
            self.port = port or 3306
            self.user = user or "root"
            self.password = password or ""
            self.database = database or "electric_power_db"
            self._source_type = "manual"

        self._dimension_cache: Dict[str, pd.DataFrame] = {}

    def _connect(self) -> pymysql.Connection:
        if self._external_conn:
            return self._external_conn
        return pymysql.connect(
            host=self.host, port=self.port,
            user=self.user, password=self.password,
            database=self.database, charset="utf8mb4",
        )

    def _load_dimension(self, table: str) -> pd.DataFrame:
        """加载维度表并缓存."""
        if table in self._dimension_cache:
            return self._dimension_cache[table]
        conn = self._connect()
        try:
            df = _query_to_df(conn, f"SELECT * FROM {table}")
            self._dimension_cache[table] = df
            return df
        finally:
            conn.close()

    # ================================================================
    # 事实表导入
    # ================================================================

    def import_power(self, province: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     quality_ids: Optional[List[int]] = None,
                     chunk_size: int = 100000) -> pd.DataFrame:
        """从 f_power_15min 导入出力/负荷数据.

        type 构造: {base}_{sub}_{value_type}
          例: 出力_风电_实际, 负荷_总_预测
        """
        if quality_ids is None:
            quality_ids = [1, 2, 3, 4]

        time_dim = self._load_dimension("t_time_point")
        time_map = dict(zip(time_dim["time_point_id"], time_dim["hour"]))
        minute_map = dict(zip(time_dim["time_point_id"], time_dim["minute"]))

        conn = self._connect()
        try:
            where_parts = [f"data_quality_id IN ({','.join(map(str, quality_ids))})"]
            if province:
                where_parts.append(f"province = '{province}'")
            if start_date:
                where_parts.append(f"date_key >= '{start_date}'")
            if end_date:
                where_parts.append(f"date_key <= '{end_date}'")
            where_clause = " AND ".join(where_parts)

            sql = f"""
                SELECT date_key, time_point_id, province,
                       power_source_id, data_quality_id, power_value
                FROM f_power_15min
                WHERE {where_clause}
                ORDER BY date_key, time_point_id, province, power_source_id
            """

            frames = []
            for chunk in _query_chunks(conn, sql, chunk_size):
                chunk = chunk.copy()
                chunk["hour"] = chunk["time_point_id"].map(time_map).fillna(0).astype(int)
                chunk["minute"] = chunk["time_point_id"].map(minute_map).fillna(0).astype(int)

                chunk["_base"] = chunk["data_quality_id"].map(
                    lambda x: self.DATA_QUALITY_MAP.get(x, ("未知", "实际"))[0]
                )
                chunk["_value_type"] = chunk["data_quality_id"].map(
                    lambda x: self.DATA_QUALITY_MAP.get(x, ("未知", "实际"))[1]
                )
                chunk["_sub"] = chunk["power_source_id"].map(
                    lambda x: self.POWER_SOURCE_MAP.get(x, f"ps{x}")
                )
                chunk["type"] = (
                    chunk["_base"] + "_" + chunk["_sub"] + "_" + chunk["_value_type"]
                )
                chunk["dt"] = pd.to_datetime(
                    chunk["date_key"].astype(str) + " " +
                    chunk["hour"].astype(str).str.zfill(2) + ":" +
                    chunk["minute"].astype(str).str.zfill(2) + ":00"
                )
                chunk = chunk.rename(columns={"power_value": "value"})
                chunk = chunk[["dt", "province", "type", "value",
                               "_base", "_sub", "_value_type",
                               "data_quality_id", "power_source_id"]]
                frames.append(chunk)

            if not frames:
                return pd.DataFrame()

            result = pd.concat(frames, ignore_index=True)
            result = result.dropna(subset=["value"])
            result = result.sort_values(["province", "type", "dt"]).reset_index(drop=True)

            logger.info("导入出力/负荷: %d 行, %d 种 type, %s 省",
                        len(result), result["type"].nunique(),
                        result["province"].nunique())
            return result

        finally:
            conn.close()

    def import_price(self, province: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     chunk_size: int = 50000) -> pd.DataFrame:
        """从 f_price_15min 导入电价数据.

        所有电价数据 value_type=实际 (没有预测/实际之分).
        type 构造: 电价_{sub}_实际
        """
        time_dim = self._load_dimension("t_time_point")
        time_map = dict(zip(time_dim["time_point_id"], time_dim["hour"]))
        minute_map = dict(zip(time_dim["time_point_id"], time_dim["minute"]))

        conn = self._connect()
        try:
            where_parts = ["1=1"]
            if province:
                where_parts.append(f"province = '{province}'")
            if start_date:
                where_parts.append(f"date_key >= '{start_date}'")
            if end_date:
                where_parts.append(f"date_key <= '{end_date}'")
            where_clause = " AND ".join(where_parts)

            sql = f"""
                SELECT date_key, time_point_id, province,
                       price_market_id, price_value
                FROM f_price_15min
                WHERE {where_clause}
                ORDER BY date_key, time_point_id, province
            """

            frames = []
            for chunk in _query_chunks(conn, sql, chunk_size):
                chunk = chunk.copy()
                chunk["hour"] = chunk["time_point_id"].map(time_map).fillna(0).astype(int)
                chunk["minute"] = chunk["time_point_id"].map(minute_map).fillna(0).astype(int)
                chunk["_sub"] = chunk["price_market_id"].map(
                    lambda x: self.PRICE_MARKET_MAP.get(x, f"pm{x}")
                )
                # 电价: 全部标记为 实际 (交易数据)
                chunk["type"] = "电价_" + chunk["_sub"] + "_实际"
                chunk["dt"] = pd.to_datetime(
                    chunk["date_key"].astype(str) + " " +
                    chunk["hour"].astype(str).str.zfill(2) + ":" +
                    chunk["minute"].astype(str).str.zfill(2) + ":00"
                )
                chunk = chunk.rename(columns={"price_value": "value"})
                chunk = chunk[["dt", "province", "type", "value",
                               "_sub", "price_market_id"]]
                frames.append(chunk)

            if not frames:
                return pd.DataFrame()

            result = pd.concat(frames, ignore_index=True)
            result = result.dropna(subset=["value"])
            result = result.sort_values(["province", "type", "dt"]).reset_index(drop=True)

            logger.info("导入电价: %d 行, %d 种 type, %s 省",
                        len(result), result["type"].nunique(),
                        result["province"].nunique())
            return result

        finally:
            conn.close()

    def import_all(self, province: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """导入全部数据 (出力 + 负荷 + 电价), 统一格式."""
        power_df = self.import_power(province, start_date, end_date)
        price_df = self.import_price(province, start_date, end_date)

        if power_df.empty and price_df.empty:
            return pd.DataFrame()

        cols = ["dt", "province", "type", "value"]
        power_subset = power_df[[c for c in cols if c in power_df.columns]]
        price_subset = price_df[[c for c in cols if c in price_df.columns]]

        result = pd.concat([power_subset, price_subset], ignore_index=True)
        result = result.sort_values(["province", "type", "dt"]).reset_index(drop=True)

        logger.info("导入总计: %d 行, %d 种 type, %d 省",
                     len(result), result["type"].nunique(),
                     result["province"].nunique())
        return result

    # ================================================================
    # 导出
    # ================================================================

    def export_to_csv(self, output_dir: Optional[str] = None,
                      province: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> List[str]:
        """按 (province, type) 分组导出 CSV 到 output_dir.

        每个文件格式: {province}_{type}.csv, /import 可直接消费.
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                ".energy_data", "raw",
            )
        os.makedirs(output_dir, exist_ok=True)

        data = self.import_all(province, start_date, end_date)
        if data.empty:
            return []

        files_written: List[str] = []
        for (p, t), group in data.groupby(["province", "type"]):
            fname = f"{p}_{t}.csv"
            fpath = os.path.join(output_dir, fname)
            group.to_csv(fpath, index=False)
            files_written.append(fpath)
            logger.info("导出: %s (%d 行)", fpath, len(group))

        return files_written

    # ================================================================
    # 数据概要
    # ================================================================

    def summary(self) -> Dict:
        """返回数据库中可导入数据的概览."""
        conn = self._connect()
        try:
            with conn.cursor() as c:
                c.execute("""
                    SELECT province, data_quality_id, power_source_id, COUNT(*) as cnt
                    FROM f_power_15min
                    GROUP BY province, data_quality_id, power_source_id
                    ORDER BY province, data_quality_id, power_source_id
                """)
                power_stats = [
                    {"province": r[0], "quality_id": r[1],
                     "quality_name": self.DATA_QUALITY_MAP.get(r[1], ("?", "?"))[0],
                     "power_source": self.POWER_SOURCE_MAP.get(r[2], f"ps{r[2]}"),
                     "count": r[3]}
                    for r in c.fetchall()
                ]

                c.execute("""
                    SELECT province, price_market_id, COUNT(*) as cnt
                    FROM f_price_15min
                    GROUP BY province, price_market_id
                    ORDER BY province, price_market_id
                """)
                price_stats = [
                    {"province": r[0],
                     "price_market": self.PRICE_MARKET_MAP.get(r[1], f"pm{r[1]}"),
                     "count": r[2]}
                    for r in c.fetchall()
                ]

            return {
                "source_type": self._source_type,
                "database": self.database,
                "host": self.host,
                "power_types": len(power_stats),
                "price_types": len(price_stats),
                "power_detail": power_stats[:30],
                "price_detail": price_stats,
            }
        finally:
            conn.close()


# ================================================================
# CLI 入口
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 MySQL/Doris 导入电力数据")
    parser.add_argument("--export", action="store_true", help="导出 CSV 到 .energy_data/raw/")
    parser.add_argument("--output-dir", type=str, default=None, help="CSV 输出目录")
    parser.add_argument("--province", type=str, default=None)
    parser.add_argument("--start", type=str, default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--summary", action="store_true", help="显示数据概要")
    args = parser.parse_args()

    importer = EnergyDBImporter()

    if args.summary:
        import json
        print(json.dumps(importer.summary(), ensure_ascii=False, indent=2, default=str))
    elif args.export:
        files = importer.export_to_csv(
            output_dir=args.output_dir,
            province=args.province,
            start_date=args.start,
            end_date=args.end,
        )
        print(f"导出完成: {len(files)} 个文件")
    else:
        data = importer.import_all(
            province=args.province,
            start_date=args.start,
            end_date=args.end,
        )
        print(f"导入 {len(data)} 行数据")
        print(f"type 分布:\n{data['type'].value_counts().to_string()}")
