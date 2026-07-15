from datetime import date

from src.bigkinds_news import BigKindsConfig, BigKindsNewsClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return FakeResponse(self.payload)


def test_collect_normalizes_bigkinds_api_documents():
    session = FakeSession(
        {
            "return_object": {
                "total": 1,
                "documents": [
                    {
                        "news_id": "abc",
                        "title": "<b>서울</b> 교통 정책",
                        "content": "서울시가 새 교통 정책을 발표했습니다.",
                        "provider": "테스트신문",
                        "published_at": "2026-07-15",
                        "url": "https://example.com/news/abc",
                    }
                ],
            }
        }
    )
    config = BigKindsConfig(
        access_key="key",
        queries=("서울",),
        page_size=10,
        max_results_per_query=10,
    )

    result = BigKindsNewsClient(config, session=session).collect(
        date(2026, 7, 1), date(2026, 7, 31)
    )

    assert len(result) == 1
    assert result.iloc[0]["title"] == "서울 교통 정책"
    assert result.iloc[0]["body"] == "서울시가 새 교통 정책을 발표했습니다."
    assert result.iloc[0]["publisher"] == "테스트신문"
    assert result.iloc[0]["published_at"] == "2026-07-15T00:00:00"
    assert result.iloc[0]["query"] == "서울"
    request_payload = session.calls[0][1]["json"]
    assert request_payload["argument"]["published_at"] == {
        "from": "2026-07-01",
        "until": "2026-07-31",
    }
