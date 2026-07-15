from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd
import requests

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
_TAG_RE = re.compile(r"<[^>]+>")
KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class NaverNewsConfig:
    client_id: str
    client_secret: str
    queries: tuple[str, ...]
    publisher_domains: tuple[str, ...] = ()
    page_size: int = 100
    max_results_per_query: int = 1000
    timeout_seconds: int = 20
    request_delay_seconds: float = 0.1

    @classmethod
    def from_env(cls) -> "NaverNewsConfig":
        client_id = os.getenv("NAVER_CLIENT_ID", "").strip()
        client_secret = os.getenv("NAVER_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise ValueError("NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 .env에 설정해 주세요.")

        queries = tuple(
            value.strip()
            for value in os.getenv("NEWS_QUERIES", "서울").split(",")
            if value.strip()
        )
        if not queries:
            raise ValueError("NEWS_QUERIES에 검색어를 하나 이상 설정해 주세요.")

        domains = tuple(
            value.lower().strip().removeprefix("www.")
            for value in os.getenv("NEWS_PUBLISHER_DOMAINS", "").split(",")
            if value.strip()
        )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            queries=queries,
            publisher_domains=domains,
            page_size=min(max(int(os.getenv("NAVER_PAGE_SIZE", "100")), 1), 100),
            max_results_per_query=min(
                max(int(os.getenv("NAVER_MAX_RESULTS_PER_QUERY", "1000")), 1), 1000
            ),
            timeout_seconds=max(int(os.getenv("NAVER_TIMEOUT_SECONDS", "20")), 1),
            request_delay_seconds=max(
                float(os.getenv("NAVER_REQUEST_DELAY_SECONDS", "0.1")), 0.0
            ),
        )


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    return " ".join(_TAG_RE.sub("", text).split())


def normalize_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def domain_allowed(url: str, allowed_domains: Iterable[str]) -> bool:
    allowed = tuple(allowed_domains)
    if not allowed:
        return True
    domain = normalize_domain(url)
    return any(domain == item or domain.endswith(f".{item}") for item in allowed)


def parse_pub_date(value: str) -> datetime:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(KST)


class NaverNewsClient:
    def __init__(self, config: NaverNewsConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-Naver-Client-Id": self.config.client_id,
            "X-Naver-Client-Secret": self.config.client_secret,
        }

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
        start = 1
        upper = self.config.max_results_per_query

        while start <= upper:
            response = self.session.get(
                NAVER_NEWS_URL,
                headers=self.headers,
                params={
                    "query": query,
                    "display": min(self.config.page_size, upper - start + 1),
                    "start": start,
                    "sort": "date",
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items", [])
            if not items:
                break

            reached_older_article = False
            for item in items:
                published = parse_pub_date(item.get("pubDate", ""))
                if published.date() < start_date:
                    reached_older_article = True
                    continue
                if published.date() > end_date:
                    continue

                original_url = item.get("originallink") or item.get("link") or ""
                if not original_url or not domain_allowed(
                    original_url, self.config.publisher_domains
                ):
                    continue

                rows.append(
                    {
                        "title": clean_text(item.get("title")),
                        # 네이버 검색 API는 기사 전문이 아니라 요약 passage를 반환한다.
                        "body": clean_text(item.get("description")),
                        "publisher": normalize_domain(original_url),
                        "published_at": published.isoformat(),
                        "url": original_url,
                        "naver_url": item.get("link", ""),
                        "query": query,
                    }
                )

            if reached_older_article or len(items) < self.config.page_size:
                break
            start += self.config.page_size
            if self.config.request_delay_seconds:
                time.sleep(self.config.request_delay_seconds)

        return rows
