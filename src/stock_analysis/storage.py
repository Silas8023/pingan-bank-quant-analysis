# -*- coding: utf-8 -*-
"""SQLite 存储层：日 K 数据入库、增量更新与读取。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class StockDB:
    """本地 SQLite 行情库，支持 upsert 与自动更新。"""

    def __init__(self, db_path="output/pingan.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_kline (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume REAL, amount REAL,
                PRIMARY KEY (code, date)
            )""")
        self.conn.commit()

    def upsert_kline(self, code: str, df: pd.DataFrame) -> int:
        """把 OHLCV 数据写入库（INSERT OR REPLACE，按 code+date 去重）。"""
        rows = [
            (code, idx.strftime("%Y-%m-%d"),
             float(r["open"]), float(r["high"]), float(r["low"]),
             float(r["close"]), float(r["volume"]),
             float(r["amount"]) if "amount" in df.columns and pd.notna(r["amount"]) else None)
            for idx, r in df.iterrows()
        ]
        self.conn.executemany("""
            INSERT OR REPLACE INTO daily_kline
            (code, date, open, high, low, close, volume, amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
        self.conn.commit()
        return len(rows)

    def load_kline(self, code: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume, amount "
            "FROM daily_kline WHERE code=? ORDER BY date", self.conn, params=(code,))
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def auto_update(self, code: str, months=6, adjust="qfq", warmup_days=45) -> dict:
        """从行情接口拉取最新数据并入库，返回统计。"""
        from .data_fetcher import fetch_stock_data
        df = fetch_stock_data(code, months, adjust, warmup_days)
        inserted = self.upsert_kline(code, df)
        total = len(self.load_kline(code))
        return {"inserted_or_updated": inserted, "total_rows": total,
                "db_path": str(self.path)}

    def close(self) -> None:
        self.conn.close()
