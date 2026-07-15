import pandas as pd
import pytest

import numpy as np

from src.pipeline import (
    _issue_score,
    _label_quality,
    normalize_text,
    prepare_articles,
    remove_exact_body_duplicates,
    remove_near_duplicates,
)


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


def test_label_quality_penalizes_generic_topic_terms():
    assert _label_quality("비트코인 · 가상화폐 · 거래소") == 1.0
    assert _label_quality("기자 · 사진 · 있다") == 0.0


def test_issue_score_discounts_loose_or_generic_clusters():
    strong = _issue_score(article_count=80, cohesion_score=0.9, label_quality=1.0)
    noisy = _issue_score(article_count=100, cohesion_score=0.3, label_quality=0.0)

    assert strong > noisy
    assert noisy == 0


def test_remove_near_duplicates_drops_identical_long_body_even_with_different_titles():
    repeated_body = "서울시가 같은 본문을 여러 제목으로 배포한 기사입니다. " * 4
    articles = pd.DataFrame(
        {
            "title": ["제목 A", "완전히 다른 제목 B", "다른 기사"],
            "body": [repeated_body, repeated_body, "전혀 다른 본문입니다. " * 4],
        }
    )
    embeddings = np.eye(3)

    deduped, deduped_embeddings = remove_near_duplicates(
        articles,
        embeddings,
        threshold=0.99,
    )

    assert len(deduped) == 2
    assert list(deduped["title"]) == ["제목 A", "다른 기사"]
    assert len(deduped_embeddings) == 2


def test_remove_near_duplicates_drops_identical_short_body_too():
    articles = pd.DataFrame(
        {
            "title": ["제목 A", "완전히 다른 제목 B"],
            "body": ["같은 본문", "같은 본문"],
        }
    )
    embeddings = np.eye(2)

    deduped, deduped_embeddings = remove_near_duplicates(
        articles,
        embeddings,
        threshold=0.99,
    )

    assert len(deduped) == 1
    assert len(deduped_embeddings) == 1


def test_remove_exact_body_duplicates_keeps_first_article():
    articles = pd.DataFrame(
        {
            "title": ["제목 A", "완전히 다른 제목 B", "다른 기사"],
            "body": ["같은 본문", "같은 본문", "다른 본문"],
        }
    )

    deduped = remove_exact_body_duplicates(articles)

    assert list(deduped["title"]) == ["제목 A", "다른 기사"]
