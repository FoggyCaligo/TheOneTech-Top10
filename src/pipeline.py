from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GENERIC_TOPIC_TERMS = {
    "기자",
    "사진",
    "전국",
    "오전",
    "오후",
    "있다",
    "했다",
    "한다",
    "대한",
    "관련",
    "지난",
    "오늘",
    "내일",
    "제공",
    "열린",
    "모습",
    "뉴스",
    "밝혔다",
}


@dataclass(frozen=True)
class AnalysisResult:
    articles: pd.DataFrame
    topics: pd.DataFrame


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_articles(
    frame: pd.DataFrame,
    title_col: str = "title",
    body_col: str = "body",
) -> pd.DataFrame:
    missing = [column for column in (title_col, body_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    articles = frame.copy()
    articles["title"] = articles[title_col].map(normalize_text)
    articles["body"] = articles[body_col].map(normalize_text)
    articles["text"] = (articles["title"] + ". " + articles["body"]).str.strip(". ")
    articles = articles[articles["text"].str.len() >= 20].reset_index(drop=True)
    if len(articles) < 5:
        raise ValueError("분석 가능한 기사가 5개 미만입니다.")
    return articles


def remove_exact_body_duplicates(
    articles: pd.DataFrame,
    body_col: str = "body",
) -> pd.DataFrame:
    if body_col not in articles.columns:
        return articles.copy()
    body_keys = articles[body_col].map(lambda value: normalize_text(value).lower())
    keep = ~body_keys.duplicated(keep="first")
    return articles.loc[keep].reset_index(drop=True)


def remove_near_duplicates(
    articles: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = 0.96,
) -> tuple[pd.DataFrame, np.ndarray]:
    articles = articles.reset_index(drop=True)
    body_keys = articles["body"].map(lambda value: normalize_text(value).lower())
    keep_body = ~body_keys.duplicated(keep="first")
    if not keep_body.all():
        kept_indices = np.flatnonzero(keep_body.to_numpy())
        articles = articles.iloc[kept_indices].reset_index(drop=True)
        embeddings = embeddings[kept_indices]

    similarity = cosine_similarity(embeddings)
    keep: list[int] = []
    removed: set[int] = set()

    for index in range(len(articles)):
        if index in removed:
            continue
        keep.append(index)
        duplicates = np.where(similarity[index] >= threshold)[0]
        removed.update(int(item) for item in duplicates if item > index)

    return articles.iloc[keep].reset_index(drop=True), embeddings[keep]


def _topic_keywords(texts: Iterable[str], top_n: int = 5) -> str:
    values = list(texts)
    if not values:
        return "분류되지 않음"
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=1,
            token_pattern=r"(?u)[가-힣A-Za-z0-9]{2,}",
        )
        matrix = vectorizer.fit_transform(values)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        order = scores.argsort()[::-1][:top_n]
        selected = [terms[index] for index in order if scores[index] > 0]
        return " · ".join(selected) if selected else "기타 이슈"
    except ValueError:
        return "기타 이슈"


def _cluster_cohesion(embeddings: np.ndarray) -> float:
    if len(embeddings) <= 1:
        return 1.0
    centroid = embeddings.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return 0.0
    centroid = centroid / norm
    similarities = cosine_similarity(embeddings, centroid).ravel()
    return float(np.clip(similarities.mean(), 0.0, 1.0))


def _label_quality(topic_name: str) -> float:
    terms = [term.strip() for term in topic_name.split("·") if term.strip()]
    if not terms:
        return 0.0
    generic_count = sum(1 for term in terms if term in GENERIC_TOPIC_TERMS)
    return round(1.0 - generic_count / len(terms), 4)


def _issue_score(article_count: int, cohesion_score: float, label_quality: float) -> float:
    # Noisy labels such as "기자 · 사진" should not outrank real issue clusters
    # merely because they are large.
    if label_quality <= 0:
        return 0.0
    cohesion_factor = max(cohesion_score, 0.0) ** 1.5
    label_factor = label_quality ** 3
    return round(article_count * cohesion_factor * label_factor, 4)


def analyze_articles(
    frame: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    min_cluster_size: int = 5,
    duplicate_threshold: float = 0.96,
    title_col: str = "title",
    body_col: str = "body",
) -> AnalysisResult:
    from sentence_transformers import SentenceTransformer

    articles = prepare_articles(frame, title_col=title_col, body_col=body_col)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        articles["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    articles, embeddings = remove_near_duplicates(
        articles,
        np.asarray(embeddings),
        threshold=duplicate_threshold,
    )

    if len(articles) < max(5, min_cluster_size):
        raise ValueError("중복 제거 후 군집화할 기사가 부족합니다.")

    reducer = umap.UMAP(
        n_neighbors=min(15, len(articles) - 1),
        n_components=2,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    coordinates = reducer.fit_transform(embeddings)

    effective_min_cluster_size = min(min_cluster_size, max(2, len(articles) // 2))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=max(1, effective_min_cluster_size // 2),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(coordinates)

    articles["cluster"] = labels.astype(int)
    articles["x"] = coordinates[:, 0]
    articles["y"] = coordinates[:, 1]

    topic_rows: list[dict[str, object]] = []
    valid = articles[articles["cluster"] >= 0]
    denominator = max(1, len(valid))
    for cluster_id, group in valid.groupby("cluster"):
        topic_name = _topic_keywords(group["text"], top_n=5)
        group_embeddings = embeddings[group.index.to_numpy()]
        cohesion_score = round(_cluster_cohesion(group_embeddings), 4)
        label_quality = _label_quality(topic_name)
        topic_rows.append(
            {
                "cluster": int(cluster_id),
                "topic": topic_name,
                "article_count": int(len(group)),
                "share_percent": round(len(group) / denominator * 100, 2),
                "cohesion_score": cohesion_score,
                "label_quality": label_quality,
                "issue_score": _issue_score(int(len(group)), cohesion_score, label_quality),
                "representative_title": group.iloc[0]["title"],
            }
        )

    topics = pd.DataFrame(topic_rows)
    if not topics.empty:
        topics = topics.sort_values(
            ["issue_score", "article_count", "cohesion_score", "cluster"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        topics.insert(0, "rank", np.arange(1, len(topics) + 1))
        topic_map = topics.set_index("cluster")["topic"].to_dict()
        articles["topic"] = articles["cluster"].map(topic_map).fillna("노이즈/기타")
    else:
        articles["topic"] = "노이즈/기타"

    return AnalysisResult(articles=articles, topics=topics)
