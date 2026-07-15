from datetime import date

import pandas as pd

from src.bigkinds_csv import normalize_bigkinds_csv
from src.bigkinds_csv import load_bigkinds_download


def test_normalize_bigkinds_csv_filters_and_maps_columns():
    frame = pd.DataFrame(
        [
            {
                "뉴스 식별자": "abc",
                "일자": "20260715",
                "언론사": "KBS",
                "제목": "<b>서울</b> 교통 정책",
                "본문": "서울시가 새 교통 정책을 발표했습니다.",
                "URL": "https://example.com/news/abc&amp;ref=DA",
                "위치": "서울",
                "키워드": "서울,교통,정책",
            },
            {
                "뉴스 식별자": "def",
                "일자": "20260715",
                "언론사": "KBS",
                "제목": "강원 날씨",
                "본문": "강원 폭염 소식입니다.",
                "URL": "https://example.com/news/def",
                "위치": "강원",
                "키워드": "강원,폭염",
            },
        ]
    )

    result = normalize_bigkinds_csv(
        frame,
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 31),
        regions=["서울"],
    )

    assert len(result) == 1
    assert result.iloc[0]["title"] == "서울 교통 정책"
    assert result.iloc[0]["body"] == "서울시가 새 교통 정책을 발표했습니다."
    assert result.iloc[0]["publisher"] == "KBS"
    assert result.iloc[0]["published_at"] == "2026-07-15T00:00:00"
    assert result.iloc[0]["url"] == "https://example.com/news/abc&ref=DA"
    assert result.iloc[0]["query"] == "서울"
    assert "교통" in result.iloc[0]["keyword_text"]


def test_load_bigkinds_download_supports_xlsx(tmp_path):
    path = tmp_path / "bigkinds.xlsx"
    frame = pd.DataFrame(
        [
            {
                "뉴스 식별자": "abc",
                "일자": "20260715",
                "언론사": "KBS",
                "제목": "서울 교통 정책",
                "본문": "서울시가 새 교통 정책을 발표했습니다.",
                "URL": "https://example.com/news/abc",
                "위치": "서울",
                "키워드": "서울,교통,정책",
            }
        ]
    )
    frame.to_excel(path, index=False)

    result = load_bigkinds_download(
        path,
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 31),
        regions=["서울"],
    )

    assert len(result) == 1
    assert result.iloc[0]["title"] == "서울 교통 정책"


def test_normalize_bigkinds_csv_can_use_entire_file_without_date_filter():
    frame = pd.DataFrame(
        [
            {
                "뉴스 식별자": "abc",
                "일자": "20250715",
                "언론사": "KBS",
                "제목": "서울 작년 기사",
                "본문": "서울 관련 작년 기사입니다.",
                "URL": "https://example.com/news/old",
                "위치": "서울",
                "키워드": "서울",
            },
            {
                "뉴스 식별자": "def",
                "일자": "20260715",
                "언론사": "KBS",
                "제목": "서울 올해 기사",
                "본문": "서울 관련 올해 기사입니다.",
                "URL": "https://example.com/news/new",
                "위치": "서울",
                "키워드": "서울",
            },
        ]
    )

    result = normalize_bigkinds_csv(
        frame,
        start_date=date.min,
        end_date=date.max,
        regions=["서울"],
    )

    assert len(result) == 2
