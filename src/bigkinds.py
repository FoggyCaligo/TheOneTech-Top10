from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import pandas as pd
import requests


class BigKindsError(RuntimeError):
    """빅카인즈 API 호출 또는 응답 변환 오류."""


@dataclass(frozen=True)
class BigKindsConfig:
    endpoint: str
    api_key: str
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    page_size: int = 100
    timeout_seconds: int = 30
    max_pages: int = 1000
    items_path: str = "documents"
    total_path: str = "total_hits"

    @classmethod
    def from_env(cls) -> "BigKindsConfig":
        endpoint = os.getenv("BIGKINDS_API_URL", "").strip()
        api_key = os.getenv("BIGKINDS_API_KEY", "").strip()
        if not endpoint or not api_key:
            raise BigKindsError(
                "BIGKINDS_API_URL과 BIGKINDS_API_KEY를 환경변수에 설정해야 합니다."
            )
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            auth_header=os.getenv("BIGKINDS_AUTH_HEADER", "Authorization"),
            auth_scheme=os.getenv("BIGKINDS_AUTH_SCHEME", "Bearer"),
            page_size=int(os.getenv("BIGKINDS_PAGE_SIZE", "100")),
            timeout_seconds=int(os.getenv("BIGKINDS_TIMEOUT_SECONDS", "30")),
            max_pages=int(os.getenv("BIGKINDS_MAX_PAGES", "1000")),
            items_path=os.getenv("BIGKINDS_ITEMS_PATH", "documents"),
            total_path=os.getenv("BIGKINDS_TOTAL_PATH", "total_hits"),
        )


DEFAULT_FIELD_MAP = {
    "id": ("news_id", "id", "article_id"),
    "title": ("title", "news_title"),
    "body": ("content", "body", "news_content"),
    "publisher": ("provider", "publisher", "provider_name"),
    "published_at": ("published_at", "date", "published_date"),
    "url": ("url", "news_url", "web_url"),
}


def _nested_get(payload: Any, dotted_path: str, default: Any = None) -> Any:
    current = payload
    for key in filter(None, dotted_path.split(".")):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _first_value(item: dict[str, Any], candidates: Iterable[str]) -> Any:
    for candidate in candidates:
        value = _nested_get(item, candidate)
        if value not in (None, ""):
            return value
    return ""


def normalize_documents(
    documents: list[dict[str, Any]],
    field_map: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    mapping = field_map or DEFAULT_FIELD_MAP
    rows = [
        {target: _first_value(item, candidates) for target, candidates in mapping.items()}
        for item in documents
    ]
    frame = pd.DataFrame(rows, columns=list(mapping))
    if frame.empty:
        return frame

    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    frame["body"] = frame["body"].fillna("").astype(str).str.strip()
    frame = frame[(frame["title"] != "") | (frame["body"] != "")]
    return frame.drop_duplicates(subset=["id", "url", "title"], keep="first").reset_index(drop=True)


class BigKindsClient:
    """승인 후 제공되는 빅카인즈 검색 API 명세를 설정으로 흡수하는 수집기.

    기본 요청 본문은 일반적인 검색 API 형태이며, 실제 승인 명세가 다르면
    build_payload 또는 환경변수 경로만 조정하면 된다.
    """

    def __init__(self, config: BigKindsConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        credential = self.config.api_key
        if self.config.auth_scheme:
            credential = f"{self.config.auth_scheme} {credential}"
        return {
            self.config.auth_header: credential,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def build_payload(
        self,
        *,
        start_date: date,
        end_date: date,
        publishers: list[str],
        query: str = "서울",
        page: int = 1,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "published_at": {
                "from": start_date.isoformat(),
                "until": end_date.isoformat(),
            },
            "provider": publishers,
            "page": page,
            "size": self.config.page_size,
            "sort": {"published_at": "desc"},
        }

    def search_page(
        self,
        *,
        start_date: date,
        end_date: date,
        publishers: list[str],
        query: str = "서울",
        page: int = 1,
    ) -> tuple[list[dict[str, Any]], int | None]:
        response = self.session.post(
            self.config.endpoint,
            headers=self._headers(),
            json=self.build_payload(
                start_date=start_date,
                end_date=end_date,
                publishers=publishers,
                query=query,
                page=page,
            ),
            timeout=self.config.timeout_seconds,
        )
        try:
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise BigKindsError(f"빅카인즈 API 요청 실패: {exc}") from exc
        except ValueError as exc:
            raise BigKindsError("빅카인즈 API가 JSON이 아닌 응답을 반환했습니다.") from exc

        items = _nested_get(payload, self.config.items_path)
        if not isinstance(items, list):
            raise BigKindsError(
                f"기사 배열을 찾지 못했습니다. BIGKINDS_ITEMS_PATH={self.config.items_path!r}를 확인하세요."
            )
        total = _nested_get(payload, self.config.total_path)
        return items, int(total) if isinstance(total, (int, float, str)) and str(total).isdigit() else None

    def collect(
        self,
        *,
        start_date: date,
        end_date: date,
        publishers: list[str],
        query: str = "서울",
        request_interval_seconds: float = 0.2,
    ) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
        if not publishers:
            raise ValueError("언론사를 하나 이상 선택해야 합니다.")

        all_documents: list[dict[str, Any]] = []
        for page in range(1, self.config.max_pages + 1):
            documents, total = self.search_page(
                start_date=start_date,
                end_date=end_date,
                publishers=publishers,
                query=query,
                page=page,
            )
            if not documents:
                break
            all_documents.extend(documents)
            if total is not None and len(all_documents) >= total:
                break
            if len(documents) < self.config.page_size:
                break
            time.sleep(request_interval_seconds)
        else:
            raise BigKindsError(
                f"최대 페이지 수({self.config.max_pages})에 도달했습니다. 기간을 나눠 수집하세요."
            )

        return normalize_documents(all_documents)
