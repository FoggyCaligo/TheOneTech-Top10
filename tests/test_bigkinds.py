from datetime import date

from src.bigkinds import BigKindsClient, BigKindsConfig, normalize_documents


def test_build_payload_uses_period_publishers_and_seoul_query():
    client = BigKindsClient(
        BigKindsConfig(endpoint="https://example.test/search", api_key="secret")
    )

    payload = client.build_payload(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 15),
        publishers=["한겨레", "KBS"],
        query="서울",
        page=3,
    )

    assert payload["query"] == "서울"
    assert payload["published_at"] == {
        "from": "2026-07-01",
        "until": "2026-07-15",
    }
    assert payload["provider"] == ["한겨레", "KBS"]
    assert payload["page"] == 3


def test_normalize_documents_maps_common_fields_and_removes_duplicates():
    documents = [
        {
            "news_id": "1",
            "title": "서울 교통 정책 발표",
            "content": "서울시가 새로운 정책을 발표했다.",
            "provider": "KBS",
            "published_at": "2026-07-15",
            "url": "https://example.test/1",
        },
        {
            "news_id": "1",
            "title": "서울 교통 정책 발표",
            "content": "서울시가 새로운 정책을 발표했다.",
            "provider": "KBS",
            "published_at": "2026-07-15",
            "url": "https://example.test/1",
        },
    ]

    frame = normalize_documents(documents)

    assert len(frame) == 1
    assert frame.loc[0, "title"] == "서울 교통 정책 발표"
    assert frame.loc[0, "body"] == "서울시가 새로운 정책을 발표했다."
    assert frame.loc[0, "publisher"] == "KBS"
