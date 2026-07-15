from __future__ import annotations

import html
import os
from datetime import date
from pathlib import Path

import pandas as pd

from src.naver_news import clean_text

DEFAULT_BIGKINDS_DOWNLOAD_PATH = "Top10_experiment/NewsResult_20250715-20260715_1year.csv"


def bigkinds_download_path() -> Path:
    return Path(
        os.getenv(
            "BIGKINDS_DOWNLOAD_PATH",
            os.getenv("BIGKINDS_CSV_PATH", DEFAULT_BIGKINDS_DOWNLOAD_PATH),
        )
    )


def bigkinds_csv_path() -> Path:
    return bigkinds_download_path()


def _parse_bigkinds_date(value: object) -> pd.Timestamp:
    text = str(value or "").strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def _row_region(row: pd.Series, regions: list[str]) -> str:
    haystack = " ".join(
        str(row.get(column, "") or "")
        for column in ("위치", "제목", "본문", "키워드", "통합 분류1", "통합 분류2", "통합 분류3")
    )
    for region in regions:
        if region and region in haystack:
            return region
    return regions[0] if len(regions) == 1 else "빅카인즈"


def normalize_bigkinds_csv(
    frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    required = {"일자", "언론사", "제목", "본문", "URL"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"빅카인즈 CSV 필수 컬럼이 없습니다: {sorted(missing)}")

    selected_regions = [region for region in (regions or []) if region]
    source = frame.copy()
    parsed_dates = source["일자"].map(_parse_bigkinds_date)
    source = source[parsed_dates.dt.date.between(start_date, end_date)].copy()
    parsed_dates = parsed_dates.loc[source.index]

    if selected_regions:
        region_text = source[
            [column for column in ("위치", "제목", "본문", "키워드", "통합 분류1", "통합 분류2", "통합 분류3") if column in source.columns]
        ].fillna("").astype(str).agg(" ".join, axis=1)
        mask = region_text.apply(lambda value: any(region in value for region in selected_regions))
        source = source[mask].copy()
        parsed_dates = parsed_dates.loc[source.index]

    if source.empty:
        return pd.DataFrame(
            columns=["title", "body", "publisher", "published_at", "url", "naver_url", "query"]
        )

    result = pd.DataFrame(
        {
            "title": source["제목"].map(clean_text),
            "body": source["본문"].map(clean_text),
            "publisher": source["언론사"].fillna("").astype(str).map(clean_text),
            "published_at": parsed_dates.dt.strftime("%Y-%m-%dT00:00:00"),
            "url": source["URL"].fillna("").astype(str).map(lambda value: html.unescape(value.strip())),
            "naver_url": "",
            "query": source.apply(lambda row: _row_region(row, selected_regions), axis=1),
        }
    )
    if "뉴스 식별자" in source.columns:
        empty_url = result["url"].eq("")
        result.loc[empty_url, "url"] = "bigkinds:" + source.loc[empty_url, "뉴스 식별자"].astype(str)
    result = result[result["url"].ne("")]
    return result.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)


def _read_bigkinds_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, dtype=str)
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype=str, encoding="cp949")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError("빅카인즈 다운로드 파일은 .csv, .xlsx, .xls만 지원합니다.")


def load_bigkinds_download(
    path: str | Path,
    start_date: date,
    end_date: date,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"빅카인즈 다운로드 파일을 찾을 수 없습니다: {file_path}")
    frame = _read_bigkinds_file(file_path)
    return normalize_bigkinds_csv(frame, start_date, end_date, regions)


def load_bigkinds_csv(
    path: str | Path,
    start_date: date,
    end_date: date,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    return load_bigkinds_download(path, start_date, end_date, regions)
