from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import hdbscan
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def remove_near_duplicates(
    articles: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = 0.96,
) -> tuple[pd.DataFrame, np.ndarray]:
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
            token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b",
        )
        matrix = vectorizer.fit_transform(values)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        order = scores.argsort()[::-1][:top_n]
        selected = [terms[index] for index in order if scores[index] > 0]
        return " · ".join(selected) if selected else "기타 이슈"
    except ValueError:
        return "기타 이슈"


def analyze_articles(
    frame: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    min_cluster_size: int = 5,
    duplicate_threshold: float = 0.96,
    title_col: str = "title",
    body_col: str = "body",
) -> AnalysisResult:
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

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(2, len(articles) // 2)),
        min_samples=max(2, min_cluster_size // 2),
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
        topic_rows.append(
            {
                "cluster": int(cluster_id),
                "topic": topic_name,
                "article_count": int(len(group)),
                "share_percent": round(len(group) / denominator * 100, 2),
                "representative_title": group.iloc[0]["title"],
            }
        )

    topics = pd.DataFrame(topic_rows)
    if not topics.empty:
        topics = topics.sort_values(
            ["article_count", "cluster"], ascending=[False, True]
        ).reset_index(drop=True)
        topics.insert(0, "rank", np.arange(1, len(topics) + 1))
        topic_map = topics.set_index("cluster")["topic"].to_dict()
        articles["topic"] = articles["cluster"].map(topic_map).fillna("노이즈/기타")
    else:
        articles["topic"] = "노이즈/기타"

    return AnalysisResult(articles=articles, topics=topics)
