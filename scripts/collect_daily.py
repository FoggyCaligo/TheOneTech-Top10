from __future__ import annotations

import argparse
from datetime import date, timedelta

from dotenv import load_dotenv

from src.database import ArticleDatabase
from src.naver_news import NaverNewsClient, NaverNewsConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="네이버 뉴스 API 일일 증분 수집")
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="오늘을 포함해 다시 조회할 일수. 기본 2일은 지연 등록 기사를 보완합니다.",
    )
    args = parser.parse_args()

    load_dotenv()
    end_date = date.today()
    start_date = end_date - timedelta(days=max(args.days, 1) - 1)

    config = NaverNewsConfig.from_env()
    database = ArticleDatabase()
    articles = NaverNewsClient(config).collect(start_date, end_date)
    inserted = database.upsert_dataframe(articles)
    deleted = database.enforce_retention(today=end_date)

    print(
        f"수집 범위={start_date}~{end_date}, API 결과={len(articles)}, "
        f"신규 저장={inserted}, 만료 삭제={deleted}, DB 전체={database.count()}"
    )


if __name__ == "__main__":
    main()
