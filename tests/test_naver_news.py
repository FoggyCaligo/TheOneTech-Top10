from datetime import date

from src.naver_news import (
    NaverNewsClient,
    NaverNewsConfig,
    clean_text,
    domain_allowed,
)


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

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return FakeResponse(self.payload)


def test_clean_text_removes_tags_and_decodes_html():
    assert clean_text("서울 <b>교통</b> &amp; 정책") == "서울 교통 & 정책"


def test_domain_allowed_supports_subdomains():
    assert domain_allowed("https://news.kbs.co.kr/a", ["kbs.co.kr"])
    assert not domain_allowed("https://example.com/a", ["kbs.co.kr"])


def test_collect_normalizes_naver_response():
    session = FakeSession(
        {
            "items": [
                {
                    "title": "<b>서울</b> 정책 발표",
                    "description": "서울시가 새 정책을 발표했다.",
                    "originallink": "https://news.kbs.co.kr/news/view.do?ncd=1",
                    "link": "https://n.news.naver.com/article/001/1",
                    "pubDate": "Wed, 15 Jul 2026 09:00:00 +0900",
                }
            ]
        }
    )
    config = NaverNewsConfig(
        client_id="id",
        client_secret="secret",
        queries=("서울",),
        publisher_domains=("kbs.co.kr",),
        request_delay_seconds=0,
    )
    result = NaverNewsClient(config, session=session).collect(
        date(2026, 7, 15), date(2026, 7, 15)
    )

    assert len(result) == 1
    assert result.iloc[0]["title"] == "서울 정책 발표"
    assert result.iloc[0]["publisher"] == "news.kbs.co.kr"
    assert result.iloc[0]["query"] == "서울"
