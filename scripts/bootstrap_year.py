from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import ArticleDatabase
from src.naver_news import NaverNewsClient, NaverNewsConfig


DEFAULT_SUFFIXES = (
    "",
    "정책",
    "교통",
    "지하철",
    "버스",
    "주택",
    "부동산",
    "재개발",
    "복지",
    "청년",
    "교육",
    "경제",
    "일자리",
    "환경",
    "기후",
    "안전",
    "재난",
    "문화",
    "관광",
    "의회",
    "시장",
)


def build_bootstrap_queries(base_query: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    base = base_query.strip()
    if not base:
        raise ValueError("초기 적재 검색어는 비어 있을 수 없습니다.")

    queries: list[str] = []
    for suffix in suffixes:
        suffix = suffix.strip()
        query = base if not suffix else f"{base} {suffix}"
        if query not in queries:
            queries.append(query)
    return tuple(queries)


def load_suffixes() -> tuple[str, ...]:
    raw = os.getenv("BOOTSTRAP_QUERY_SUFFIXES", "").strip()
    if not raw:
        return DEFAULT_SUFFIXES
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return ("", *values) if "" not in values else values


def main() -> None:
    load_dotenv(ROOT / ".env")

    base_query = os.getenv("BOOTSTRAP_BASE_QUERY", "서울").strip() or "서울"
    queries = build_bootstrap_queries(base_query, load_suffixes())
    today = date.today()
    start_date = today - timedelta(days=365)

    base_config = NaverNewsConfig.from_env()
    config = NaverNewsConfig(
        client_id=base_config.client_id,
        client_secret=base_config.client_secret,
        queries=queries,
        publisher_domains=base_config.publisher_domains,
        page_size=base_config.page_size,
        max_results_per_query=base_config.max_results_per_query,
        timeout_seconds=base_config.timeout_seconds,
        request_delay_seconds=base_config.request_delay_seconds,
    )

    print(f"초기 적재 기간: {start_date} ~ {today}")
    print(f"기본 검색어: {base_query}")
    print(f"검색 조합: {len(queries)}개")
    for query in queries:
        print(f"  - {query}")

    client = NaverNewsClient(config)
    database = ArticleDatabase()

    frame = client.collect(start_date=start_date, end_date=today)
    added = database.upsert_dataframe(frame)
    deleted = database.enforce_retention(today=today)

    print(f"API에서 조건에 맞게 수집한 기사: {len(frame):,}개")
    print(f"DB 신규 추가 기사: {added:,}개")
    print(f"1년 초과 삭제 기사: {deleted:,}개")
    print(f"현재 DB 전체 기사: {database.count():,}개")
    print("초기 적재가 완료되었습니다. 이 스크립트는 최초 1회 실행용입니다.")


if __name__ == "__main__":
    main()
