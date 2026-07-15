import pandas as pd
import pytest

import numpy as np

from src.pipeline import (
    _extract_document_keywords,
    _issue_score,
    _keyword_tfidf_vectors,
    _label_corpus_idf,
    _label_quality,
    _normalize_keyword_text,
    _period_weight,
    _strict_centroid_groups,
    _topic_keywords,
    refine_subclusters,
    normalize_text,
    prepare_articles,
    merge_subclusters,
    remove_near_duplicates_by_text,
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
    assert {"title", "body", "keyword_text", "dedupe_text", "cluster_text", "text"}.issubset(result.columns)
    assert len(result) == 5
    assert "서울시" in result.iloc[0]["text"]
    assert "정책" in result.iloc[0]["text"]
    assert "서울시는 새로운 대중교통" in result.iloc[0]["dedupe_text"]


def test_prepare_articles_uses_keyword_text_for_embedding_text():
    frame = pd.DataFrame(
        {
            "title": ["가상화폐 규제 발표" for _ in range(5)],
            "body": ["본문입니다. " * 10 for _ in range(5)],
            "keyword_text": ["비트코인,가상화폐,거래소" for _ in range(5)],
        }
    )

    result = prepare_articles(frame)

    assert "비트코인" in result.iloc[0]["text"]
    assert "본문입니다" not in result.iloc[0]["text"]
    assert result.iloc[0]["text"] == result.iloc[0]["cluster_text"]


def test_prepare_articles_combines_existing_and_extracted_keywords():
    frame = pd.DataFrame(
        {
            "title": ["가상화폐 규제 발표" for _ in range(5)],
            "body": ["정부가 거래소 투자 규제 방안을 발표했습니다." for _ in range(5)],
            "keyword_text": ["비트코인" for _ in range(5)],
        }
    )

    result = prepare_articles(frame)

    assert "비트코인" in result.iloc[0]["keyword_text"]
    assert "규제" in result.iloc[0]["keyword_text"]


def test_prepare_articles_extracts_keywords_when_keyword_text_is_missing():
    frame = pd.DataFrame(
        {
            "title": ["비트코인 거래소 규제 발표" for _ in range(5)],
            "body": ["정부가 가상화폐 거래소와 비트코인 투자 규제 방안을 발표했습니다." for _ in range(5)],
        }
    )

    result = prepare_articles(frame)

    assert result["keyword_text"].str.len().gt(0).all()
    assert "비트코인" in result.iloc[0]["text"]


def test_normalize_keyword_text_sorts_equivalent_keyword_sets():
    assert _normalize_keyword_text("슈퍼문, 블루문") == _normalize_keyword_text("블루문,슈퍼문")


def test_extract_document_keywords_prefers_kiwi_noun_terms(monkeypatch):
    class FakeToken:
        def __init__(self, form, tag):
            self.form = form
            self.tag = tag

    class FakeKiwi:
        def tokenize(self, text):
            return [
                FakeToken("Bitcoin", "SL"),
                FakeToken("market", "NNG"),
                FakeToken("running", "VV"),
            ]

    monkeypatch.setattr("src.pipeline._load_kiwi", lambda: FakeKiwi())

    keywords = _extract_document_keywords(pd.Series(["Bitcoin market is running."]))

    assert "bitcoin" in keywords.iloc[0]
    assert "market" in keywords.iloc[0]
    assert "running" not in keywords.iloc[0]


def test_keyword_tfidf_vectors_returns_one_vector_per_article():
    vectors = _keyword_tfidf_vectors(
        pd.Series(["비트코인 가상화폐 거래소", "블루문 슈퍼문 천문", "비트코인 거래소"])
    )

    assert vectors.shape[0] == 3


def test_topic_keywords_prefers_cluster_terms_that_are_rare_in_corpus():
    corpus = [
        "공통 발표 비트코인",
        "공통 발표 비트코인",
        "공통 발표 부동산",
        "공통 발표 증시",
        "공통 발표 정치",
    ]
    label_idf = _label_corpus_idf(corpus)

    topic = _topic_keywords(corpus[:2], top_n=2, corpus_idf=label_idf)

    assert topic.split(" · ")[0] == "비트코인"


def test_prepare_articles_rejects_missing_columns():
    with pytest.raises(ValueError, match="필수 컬럼"):
        prepare_articles(pd.DataFrame({"title": ["기사"]}))


def test_label_quality_only_checks_that_label_exists():
    assert _label_quality("비트코인 · 가상화폐 · 거래소") == 1.0
    assert _label_quality("기자 · 사진 · 있다") == 1.0
    assert _label_quality("") == 0.0


def test_issue_score_uses_period_weight():
    strong = _issue_score(article_count=80, cohesion_score=0.9, label_quality=1.0)
    weighted = _issue_score(
        article_count=80,
        cohesion_score=0.9,
        label_quality=1.0,
        period_weight=1.3,
    )

    assert weighted > strong


def test_period_weight_rewards_time_concentration():
    all_dates = pd.Series(pd.date_range("2026-01-01", "2026-01-31", freq="D"))
    concentrated_dates = pd.Series(pd.date_range("2026-01-10", "2026-01-12", freq="D"))
    spread_dates = pd.Series(pd.date_range("2026-01-01", "2026-01-31", freq="10D"))

    concentrated_weight, concentrated_days = _period_weight(concentrated_dates, all_dates)
    spread_weight, spread_days = _period_weight(spread_dates, all_dates)

    assert concentrated_days < spread_days
    assert concentrated_weight > spread_weight


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


def test_remove_near_duplicates_by_text_uses_dedupe_text_not_cluster_text():
    class FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            if texts == ["같은 원문", "같은 원문"]:
                return np.array([[1.0, 0.0], [1.0, 0.0]])
            raise AssertionError("dedupe_text should be used for duplicate detection")

    articles = pd.DataFrame(
        {
            "dedupe_text": ["같은 원문", "같은 원문"],
            "cluster_text": ["비트코인 규제", "가상화폐 거래소"],
            "body": ["본문 A", "본문 B"],
        }
    )

    deduped = remove_near_duplicates_by_text(
        articles,
        FakeModel(),
        threshold=0.99,
        text_col="dedupe_text",
    )

    assert len(deduped) == 1


def test_merge_subclusters_combines_similar_cluster_centroids():
    articles = pd.DataFrame(
        {
            "subcluster": [0, 0, 1, 1, 2, 2, -1],
        }
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.95, 0.05],
            [0.93, 0.07],
            [0.0, 1.0],
            [0.02, 0.98],
            [-1.0, 0.0],
        ]
    )

    merged = merge_subclusters(articles, embeddings, threshold=0.9)

    assert list(merged[:4].unique()) == [0]
    assert merged.iloc[4] == 1
    assert merged.iloc[6] == -1


def test_strict_centroid_groups_prevents_chain_merging():
    centroids = np.array(
        [
            [1.0, 0.0],
            [0.75, 0.66],
            [0.2, 0.98],
        ]
    )

    groups = _strict_centroid_groups(centroids, threshold=0.7)

    assert len(groups) == 2


def test_refine_subclusters_splits_far_articles_from_cluster():
    labels = np.array([0, 0, 0, 0, 0, 0])
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.01, 0.99],
            [0.02, 0.98],
        ]
    )

    refined = refine_subclusters(
        labels,
        embeddings,
        min_cluster_size=3,
        outlier_threshold=0.8,
    )

    assert len(set(refined)) > 1
