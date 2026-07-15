import pandas as pd
import pytest

from src.pipeline import normalize_text, prepare_articles


def test_normalize_text_removes_html_and_whitespace():
    assert normalize_text("<b>서울</b>   뉴스\n기사") == "서울 뉴스 기사"


def test_prepare_articles_builds_analysis_text():
    frame = pd.DataFrame(
        {
            "headline": ["서울시 교통 정책 발표" for _ in range(5)],
            "content": ["서울시는 새로운 대중교통 지원 정책을 발표했습니다." for _ in range(5)],
        }
    )
    result = prepare_articles(frame, title_col="headline", body_col="content")
    assert list(result.columns[-3:]) == ["title", "body", "text"]
    assert len(result) == 5
    assert result.iloc[0]["text"].startswith("서울시 교통 정책 발표")


def test_prepare_articles_rejects_missing_columns():
    with pytest.raises(ValueError, match="필수 컬럼"):
        prepare_articles(pd.DataFrame({"title": ["기사"]}))
