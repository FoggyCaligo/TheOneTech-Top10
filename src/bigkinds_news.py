from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from src.naver_news import clean_text

BIGKINDS_NEWS_URL = "https://tools.kinds.or.kr/search/news"


@dataclass(frozen=True)
class BigKindsConfig:
    access_key: str
    queries: tuple[str, ...]
    endpoint: str = BIGKINDS_NEWS_URL
    page_size: int = 100
    max_results_per_query: int = 1000
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> "BigKindsConfig":
        access_key = os.getenv("BIGKINDS_ACCESS_KEY", "").strip()
        if not access_key:
            raise ValueError("BIGKINDS_ACCESS_KEY를 .env에 설정해 주세요.")

        queries = tuple(
            value.strip()
            for value in os.getenv("NEWS_QUERIES", "서울").split(",")
            if value.strip()
        )
        if not queries:
            raise ValueError("NEWS_QUERIES에 검색어를 하나 이상 설정해 주세요.")

        return cls(
            access_key=access_key,
            queries=queries,
            endpoint=os.getenv("BIGKINDS_ENDPOINT", BIGKINDS_NEWS_URL).strip()
            or BIGKINDS_NEWS_URL,
            page_size=min(max(int(os.getenv("BIGKINDS_PAGE_SIZE", "100")), 1), 1000),
            max_results_per_query=max(
                int(os.getenv("BIGKINDS_MAX_RESULTS_PER_QUERY", "1000")), 1
            ),
            timeout_seconds=max(int(os.getenv("BIGKINDS_TIMEOUT_SECONDS", "30")), 1),
        )


def _extract_documents(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return_object = payload.get("return_object")
    nested = return_object if isinstance(return_object, dict) else {}
    candidates = [
        payload.get("documents"),
        payload.get("result"),
        payload.get("data"),
        nested.get("documents"),
        nested.get("result"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _extract_total(payload: dict[str, Any], fallback: int) -> int:
    return_object = payload.get("return_object")
    nested = return_object if isinstance(return_object, dict) else {}
    values = [
        payload.get("total"),
        payload.get("total_count"),
        nested.get("total"),
        nested.get("total_count"),
    ]
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return fallback


def _parse_published_at(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now().isoformat()
    for fmt, size in (
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%d", 10),
        ("%Y%m%d%H%M%S", 14),
        ("%Y%m%d", 8),
    ):
        try:
            return datetime.strptime(text[:size], fmt).isoformat()
        except ValueError:
            continue
    return text


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return clean_text(str(value))
    return ""


class BigKindsNewsClient:
    def __init__(self, config: BigKindsConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()

    def collect(self, start_date: date, end_date: date) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")

        rows: list[dict[str, str]] = []
        for query in self.config.queries:
            rows.extend(self._collect_query(query, start_date, end_date))

        columns = [
            "title",
            "body",
            "publisher",
            "published_at",
            "url",
            "naver_url",
            "query",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows, columns=columns)
        frame = frame.drop_duplicates(subset=["url"], keep="first")
        return frame.sort_values("published_at", ascending=False).reset_index(drop=True)

    def _collect_query(self, query: str, start_date: date, end_date: date) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        offset = 0
        total_seen = 0

        while total_seen < self.config.max_results_per_query:
            size = min(self.config.page_size, self.config.max_results_per_query - total_seen)
            payload = {
                "access_key": self.config.access_key,
                "argument": {
                    "query": query,
                    "published_at": {
                        "from": start_date.isoformat(),
                        "until": end_date.isoformat(),
                    },
                    "sort": {"date": "desc"},
                    "return_from": offset,
                    "return_size": size,
                    "fields": [
                        "news_id",
                        "title",
                        "content",
                        "hilight",
                        "published_at",
                        "provider",
                        "url",
                    ],
                },
            }
            response = self.session.post(
                self.config.endpoint,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
            documents = _extract_documents(response_payload)
            if not documents:
                break

            for item in documents:
                news_id = str(item.get("news_id") or item.get("id") or "").strip()
                url = str(item.get("url") or item.get("link") or "").strip()
                if not url:
                    url = f"bigkinds:{news_id}" if news_id else f"bigkinds:{query}:{offset}"
                rows.append(
                    {
                        "title": _first_text(item, "title", "news_title"),
                        "body": _first_text(item, "content", "body", "hilight", "summary"),
                        "publisher": _first_text(item, "provider", "provider_name", "publisher"),
                        "published_at": _parse_published_at(
                            item.get("published_at") or item.get("date")
                        ),
                        "url": url,
                        "naver_url": "",
                        "query": query,
                    }
                )

            total_seen += len(documents)
            offset += len(documents)
            total_available = _extract_total(response_payload, fallback=offset)
            if len(documents) < size or offset >= total_available:
                break

        return rows
