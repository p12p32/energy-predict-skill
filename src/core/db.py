"""db.py — Apache Doris 连接与查询封装"""
import pymysql
import pandas as pd
from typing import Optional, Dict, Any
from src.core.config import get_doris_config


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
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def insert_dataframe(self, df: pd.DataFrame, table: str,
                         batch_size: int = 5000) -> int:
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
        result = self.query(
            f"SELECT COUNT(*) as cnt FROM information_schema.tables "
            f"WHERE table_schema='{self.database}' AND table_name='{table_name}'"
        )
        return result.iloc[0]["cnt"] > 0
