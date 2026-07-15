from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import ArticleDatabase
from src.naver_news import NaverNewsClient, NaverNewsConfig


BOOTSTRAP_QUERY = "서울"


def main() -> None:
    load_dotenv(ROOT / ".env")

    today = date.today()
    start_date = today - timedelta(days=365)

    base_config = NaverNewsConfig.from_env()
    config = NaverNewsConfig(
        client_id=base_config.client_id,
        client_secret=base_config.client_secret,
        queries=(BOOTSTRAP_QUERY,),
        publisher_domains=base_config.publisher_domains,
        page_size=base_config.page_size,
        max_results_per_query=base_config.max_results_per_query,
        timeout_seconds=base_config.timeout_seconds,
        request_delay_seconds=base_config.request_delay_seconds,
    )

    print(f"초기 적재 기간: {start_date} ~ {today}")
    print(f"검색어: {BOOTSTRAP_QUERY}")
    print(
        "주의: 네이버 뉴스 검색 API는 단일 검색어당 최대 1,000건까지만 접근할 수 있어 "
        "최근 1년 전체 기사를 완전히 수집하지는 못합니다."
    )

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
