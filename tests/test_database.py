from datetime import date

import pandas as pd

from src.database import ArticleDatabase


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "title": "서울 정책",
                "body": "서울시 정책 기사 요약",
                "publisher": "example.com",
                "published_at": "2026-07-15T09:00:00+09:00",
                "url": "https://example.com/1",
                "naver_url": "https://n.news.naver.com/1",
                "query": "서울",
            },
            {
                "title": "오래된 기사",
                "body": "과거 기사",
                "publisher": "example.com",
                "published_at": "2025-01-01T09:00:00+09:00",
                "url": "https://example.com/old",
                "naver_url": "",
                "query": "서울",
            },
        ]
    )


def test_upsert_is_idempotent_and_queryable(tmp_path):
    db = ArticleDatabase(tmp_path / "news.db")
    assert db.upsert_dataframe(sample_frame()) == 2
    assert db.upsert_dataframe(sample_frame()) == 0
    result = db.query(date(2026, 7, 1), date(2026, 7, 31), ["서울"])
    assert len(result) == 1
    assert result.iloc[0]["title"] == "서울 정책"


def test_retention_deletes_old_articles(tmp_path):
    db = ArticleDatabase(tmp_path / "news.db")
    db.upsert_dataframe(sample_frame())
    deleted = db.enforce_retention(retention_days=365, today=date(2026, 7, 15))
    assert deleted == 1
    assert db.count() == 1
