from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


class ArticleDatabase:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("NEWS_DB_PATH", "data/news.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    publisher TEXT NOT NULL DEFAULT '',
                    published_at TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    naver_url TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL DEFAULT '',
                    collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_articles_published_at
                    ON articles(published_at);
                CREATE INDEX IF NOT EXISTS idx_articles_query
                    ON articles(query);
                """
            )

    def upsert_dataframe(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        required = {"title", "body", "publisher", "published_at", "url", "naver_url", "query"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"DB 저장 필수 컬럼이 없습니다: {sorted(missing)}")

        before = self.count()
        rows = frame[list(required)].to_dict("records")
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO articles
                    (title, body, publisher, published_at, url, naver_url, query)
                VALUES
                    (:title, :body, :publisher, :published_at, :url, :naver_url, :query)
                ON CONFLICT(url) DO UPDATE SET
                    title=excluded.title,
                    body=excluded.body,
                    publisher=excluded.publisher,
                    published_at=excluded.published_at,
                    naver_url=excluded.naver_url,
                    query=excluded.query
                """,
                rows,
            )
        return max(self.count() - before, 0)

    def query(self, start_date: date, end_date: date, queries: list[str] | None = None) -> pd.DataFrame:
        sql = """
            SELECT title, body, publisher, published_at, url, naver_url, query, collected_at
            FROM articles
            WHERE date(published_at) BETWEEN ? AND ?
        """
        params: list[str] = [start_date.isoformat(), end_date.isoformat()]
        if queries:
            placeholders = ",".join("?" for _ in queries)
            sql += f" AND query IN ({placeholders})"
            params.extend(queries)
        sql += " ORDER BY published_at DESC"
        with self.connect() as connection:
            return pd.read_sql_query(sql, connection, params=params)

    def query_month(self, year: int, month: int, queries: list[str] | None = None) -> pd.DataFrame:
        start = date(year, month, 1)
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        return self.query(start, next_month - timedelta(days=1), queries)

    def delete_older_than(self, cutoff_date: date) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM articles WHERE date(published_at) < ?",
                (cutoff_date.isoformat(),),
            )
            return cursor.rowcount

    def enforce_retention(self, retention_days: int | None = None, today: date | None = None) -> int:
        days = retention_days or int(os.getenv("NEWS_RETENTION_DAYS", "365"))
        cutoff = (today or date.today()) - timedelta(days=days)
        return self.delete_older_than(cutoff)

    def count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM articles").fetchone()
            return int(row["count"])

    def monthly_counts(self) -> pd.DataFrame:
        with self.connect() as connection:
            return pd.read_sql_query(
                """
                SELECT substr(published_at, 1, 7) AS month, COUNT(*) AS article_count
                FROM articles
                GROUP BY substr(published_at, 1, 7)
                ORDER BY month DESC
                """,
                connection,
            )
