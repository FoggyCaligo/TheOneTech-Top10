from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
    "본문",
    "본문입니다",
}
_KEYWORD_SPLIT_RE = re.compile(r"[,;/|·\n\r\t]+|\s{2,}")


@dataclass(frozen=True)
class AnalysisResult:
    articles: pd.DataFrame
    topics: pd.DataFrame


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _load_kiwi():
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        return None
    return Kiwi()


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_korean_noun_terms(text: str) -> str:
    kiwi = _load_kiwi()
    if kiwi is None:
        return ""

    tokens = kiwi.tokenize(text)
    noun_tokens: list[str] = []
    terms: list[str] = []
    for token in tokens:
        form = normalize_text(token.form)
        if len(form) < 2:
            noun_tokens.append("")
            continue
        if token.tag in {"NNG", "NNP", "SL", "SN"}:
            noun_tokens.append(form)
            terms.append(form)
        else:
            noun_tokens.append("")

    for left, right in zip(noun_tokens, noun_tokens[1:]):
        if left and right and left != right:
            terms.append(f"{left}{right}")

    return " ".join(terms)


def _extract_document_keywords(texts: pd.Series, top_n: int = 8) -> pd.Series:
    values = texts.fillna("").astype(str).map(normalize_text)
    if values.empty or values.str.len().sum() == 0:
        return pd.Series(["" for _ in range(len(values))], index=values.index)
    keyword_source = values.map(_extract_korean_noun_terms)
    if keyword_source.str.len().sum() == 0:
        keyword_source = values
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
            token_pattern=r"(?u)[가-힣A-Za-z0-9]{2,}",
        )
        matrix = vectorizer.fit_transform(keyword_source)
        terms = vectorizer.get_feature_names_out()
        rows: list[str] = []
        for row in matrix:
            scores = row.toarray().ravel()
            order = scores.argsort()[::-1][:top_n]
            selected = [
                terms[index]
                for index in order
                if scores[index] > 0 and terms[index] not in GENERIC_TOPIC_TERMS
            ]
            rows.append(",".join(selected))
        return pd.Series(rows, index=values.index)
    except ValueError:
        return pd.Series(["" for _ in range(len(values))], index=values.index)


def _normalize_keyword_text(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    normalized = _KEYWORD_SPLIT_RE.sub(",", text)
    terms: set[str] = set()
    for raw_term in normalized.split(","):
        term = normalize_text(raw_term).strip()
        tokens = term.split()
        if (
            len(term) < 2
            or term in GENERIC_TOPIC_TERMS
            or any(token in GENERIC_TOPIC_TERMS for token in tokens)
        ):
            continue
        terms.add(term)
    return " ".join(sorted(terms))


def prepare_articles(
    frame: pd.DataFrame,
    title_col: str = "title",
    body_col: str = "body",
    body_embedding_chars: int = 500,
) -> pd.DataFrame:
    missing = [column for column in (title_col, body_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    articles = frame.copy()
    articles["title"] = articles[title_col].map(normalize_text)
    articles["body"] = articles[body_col].map(normalize_text)
    source_text = articles["title"] + ". " + articles["body"].str.slice(0, body_embedding_chars)
    extracted_keywords = _extract_document_keywords(source_text)
    if "keyword_text" in articles.columns:
        articles["keyword_text"] = articles["keyword_text"].map(normalize_text)
        articles["keyword_text"] = (
            articles["keyword_text"] + " " + extracted_keywords
        ).map(_normalize_keyword_text)
    else:
        articles["keyword_text"] = extracted_keywords.map(_normalize_keyword_text)
    articles["dedupe_text"] = (articles["title"] + ". " + articles["body"]).str.strip(". ")
    articles["cluster_text"] = articles["keyword_text"]
    articles["text"] = articles["cluster_text"]
    articles = articles[
        articles["dedupe_text"].str.len().ge(20) & articles["cluster_text"].str.len().ge(2)
    ].reset_index(drop=True)
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


def remove_near_duplicates_by_text(
    articles: pd.DataFrame,
    model,
    threshold: float = 0.96,
    text_col: str = "dedupe_text",
) -> pd.DataFrame:
    if text_col not in articles.columns:
        return articles.copy()
    dedupe_embeddings = model.encode(
        articles[text_col].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    deduped, _ = remove_near_duplicates(
        articles,
        np.asarray(dedupe_embeddings),
        threshold=threshold,
    )
    return deduped


def _topic_keywords(
    texts: Iterable[str],
    top_n: int = 5,
    corpus_idf: dict[str, float] | None = None,
) -> str:
    values = list(texts)
    if not values:
        return "분류되지 않음"
    try:
        label_source = values
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            binary=True,
            use_idf=False,
            norm=None,
            max_features=3000,
            min_df=1,
            token_pattern=r"(?u)[가-힣A-Za-z0-9]{2,}",
        )
        matrix = vectorizer.fit_transform(label_source)
        cluster_df = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        idf = np.array([corpus_idf.get(term, 1.0) if corpus_idf else 1.0 for term in terms])
        scores = cluster_df * idf
        order = sorted(
            range(len(terms)),
            key=lambda index: (scores[index], cluster_df[index], len(terms[index]), terms[index]),
            reverse=True,
        )
        selected = [
            terms[index]
            for index in order
            if scores[index] > 0 and terms[index] not in GENERIC_TOPIC_TERMS
        ][:top_n]
        return " · ".join(selected) if selected else "기타 이슈"
    except ValueError:
        return "기타 이슈"


def _label_corpus_idf(texts: Iterable[str]) -> dict[str, float]:
    values = list(texts)
    if not values:
        return {}
    try:
        label_source = values
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            binary=True,
            use_idf=False,
            norm=None,
            max_features=5000,
            min_df=1,
            token_pattern=r"(?u)[가-힣A-Za-z0-9]{2,}",
        )
        matrix = vectorizer.fit_transform(label_source)
        document_frequency = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        article_count = max(1, len(values))
        idf = np.log((1 + article_count) / (1 + document_frequency)) + 1
        return {
            term: float(score)
            for term, score in zip(terms, idf)
            if term not in GENERIC_TOPIC_TERMS
        }
    except ValueError:
        return {}


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


def _cluster_template_score(group: pd.DataFrame) -> float:
    if group.empty or "cluster_text" not in group.columns:
        return 0.0
    top_signature_share = group["cluster_text"].value_counts(normalize=True).iloc[0]
    return round(float(np.clip(top_signature_share, 0.0, 1.0)), 4)


def _template_penalty(cohesion_score: float, template_score: float) -> float:
    if template_score < 0.25 and cohesion_score < 0.97:
        return 1.0
    repetition_penalty = 1.0 - max(0.0, template_score - 0.25) * 0.6
    cohesion_penalty = 0.82 if cohesion_score >= 0.985 else 0.9 if cohesion_score >= 0.97 else 1.0
    return round(float(np.clip(repetition_penalty * cohesion_penalty, 0.35, 1.0)), 4)


def _issue_score(
    article_count: int,
    cohesion_score: float,
    label_quality: float,
    template_penalty: float = 1.0,
) -> float:
    # Noisy labels such as "기자 · 사진" should not outrank real issue clusters
    # merely because they are large.
    if label_quality <= 0:
        return 0.0
    cohesion_factor = max(cohesion_score, 0.0) ** 1.5
    label_factor = label_quality ** 3
    return round(article_count * cohesion_factor * label_factor * template_penalty, 4)


def _keyword_tfidf_vectors(texts: pd.Series) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        token_pattern=r"(?u)[가-힣A-Za-z0-9]{2,}",
    )
    matrix = vectorizer.fit_transform(texts.fillna("").astype(str))
    max_components = min(50, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if max_components < 2:
        return matrix.toarray()
    reduced = TruncatedSVD(n_components=max_components, random_state=42).fit_transform(matrix)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return np.divide(reduced, norms, out=np.zeros_like(reduced), where=norms != 0)


def remove_repeated_keyword_templates(
    articles: pd.DataFrame,
    max_per_signature: int = 90,
) -> pd.DataFrame:
    if "cluster_text" not in articles.columns or articles.empty:
        return articles.copy()
    keep = articles.groupby("cluster_text", sort=False).cumcount() < max_per_signature
    return articles.loc[keep].reset_index(drop=True)


def _strict_centroid_groups(
    centroids: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    similarity = cosine_similarity(centroids)
    order = sorted(
        range(len(centroids)),
        key=lambda index: float(similarity[index].sum()),
        reverse=True,
    )
    groups: list[list[int]] = []

    for candidate in order:
        best_group_index: int | None = None
        best_score = -1.0
        for group_index, group in enumerate(groups):
            pairwise_ok = all(similarity[candidate, member] >= threshold for member in group)
            if not pairwise_ok:
                continue
            group_centroid = centroids[group].mean(axis=0, keepdims=True)
            norm = np.linalg.norm(group_centroid)
            if norm:
                group_centroid = group_centroid / norm
            score = float(cosine_similarity(centroids[candidate].reshape(1, -1), group_centroid)[0, 0])
            if score >= threshold and score > best_score:
                best_group_index = group_index
                best_score = score
        if best_group_index is None:
            groups.append([candidate])
        else:
            groups[best_group_index].append(candidate)

    return [sorted(group) for group in groups]


def merge_subclusters(
    articles: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = 0.72,
    article_threshold: float | None = None,
) -> pd.Series:
    labels = articles["subcluster"].to_numpy()
    valid_labels = sorted(int(label) for label in np.unique(labels) if label >= 0)
    merged = pd.Series(labels, index=articles.index, dtype="int64")
    if len(valid_labels) <= 1:
        return merged

    centroids: list[np.ndarray] = []
    for label in valid_labels:
        group_embeddings = embeddings[labels == label]
        centroid = group_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        centroids.append(centroid)

    centroid_matrix = np.vstack(centroids)
    components = _strict_centroid_groups(centroid_matrix, threshold=threshold)
    label_to_merged: dict[int, int] = {}
    for merged_id, component in enumerate(components):
        for component_index in component:
            label_to_merged[valid_labels[component_index]] = merged_id

    merged = pd.Series(
        [label_to_merged.get(int(label), -1) if label >= 0 else -1 for label in labels],
        index=articles.index,
        dtype="int64",
    )
    cutoff = article_threshold if article_threshold is not None else max(0.35, threshold - 0.2)
    next_cluster = int(merged.max()) + 1 if len(merged) else 0

    for cluster_id in sorted(int(value) for value in merged.unique() if value >= 0):
        cluster_mask = merged.eq(cluster_id).to_numpy()
        subclusters = {int(value) for value in labels[cluster_mask] if value >= 0}
        if len(subclusters) <= 1:
            continue

        cluster_embeddings = embeddings[cluster_mask]
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        similarities = cosine_similarity(embeddings, centroid).ravel()
        outlier_indices = np.flatnonzero(cluster_mask & (similarities < cutoff))
        if len(outlier_indices) == 0:
            continue

        outlier_subclusters = sorted({int(labels[index]) for index in outlier_indices if labels[index] >= 0})
        for subcluster in outlier_subclusters:
            subcluster_indices = np.flatnonzero(cluster_mask & (labels == subcluster) & (similarities < cutoff))
            if len(subcluster_indices) == 0:
                continue
            merged.iloc[subcluster_indices] = next_cluster
            next_cluster += 1

    return merged


def refine_subclusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_cluster_size: int,
    outlier_threshold: float = 0.45,
) -> np.ndarray:
    refined = labels.astype(int).copy()
    next_label = int(refined[refined >= 0].max()) + 1 if np.any(refined >= 0) else 0
    recluster_min_size = max(3, min_cluster_size // 2)

    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        indices = np.flatnonzero(refined == label)
        if len(indices) < max(3, min_cluster_size):
            continue

        group_embeddings = embeddings[indices]
        centroid = group_embeddings.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        similarities = cosine_similarity(group_embeddings, centroid).ravel()
        outlier_local = np.flatnonzero(similarities < outlier_threshold)
        if len(outlier_local) == 0:
            continue

        outlier_indices = indices[outlier_local]
        refined[outlier_indices] = -1
        if len(outlier_indices) < recluster_min_size:
            continue

        reclusterer = hdbscan.HDBSCAN(
            min_cluster_size=min(recluster_min_size, max(2, len(outlier_indices) // 2)),
            min_samples=max(1, recluster_min_size // 2),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        outlier_labels = reclusterer.fit_predict(embeddings[outlier_indices])
        for outlier_label in sorted(int(value) for value in np.unique(outlier_labels) if value >= 0):
            member_indices = outlier_indices[outlier_labels == outlier_label]
            refined[member_indices] = next_label
            next_label += 1

    return refined


def analyze_articles(
    frame: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    min_cluster_size: int = 5,
    duplicate_threshold: float = 0.96,
    subcluster_outlier_threshold: float = 0.45,
    fast_mode: bool = False,
    include_map: bool = False,
    title_col: str = "title",
    body_col: str = "body",
) -> AnalysisResult:
    articles = prepare_articles(frame, title_col=title_col, body_col=body_col)
    articles = remove_exact_body_duplicates(articles, body_col="body")

    if fast_mode:
        articles = remove_repeated_keyword_templates(
            articles,
            max_per_signature=max(60, min_cluster_size * 3),
        )
        embeddings = _keyword_tfidf_vectors(articles["cluster_text"])
    else:
        model = _load_sentence_transformer(model_name)
        articles = remove_near_duplicates_by_text(
            articles,
            model,
            threshold=duplicate_threshold,
            text_col="dedupe_text",
        )
        embeddings = model.encode(
            articles["cluster_text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings)

    if len(articles) < max(5, min_cluster_size):
        raise ValueError("중복 제거 후 군집화할 기사가 부족합니다.")

    effective_min_cluster_size = min(min_cluster_size, max(2, len(articles) // 2))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=max(1, effective_min_cluster_size // 2),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    if not fast_mode:
        labels = refine_subclusters(
            labels.astype(int),
            embeddings,
            min_cluster_size=effective_min_cluster_size,
            outlier_threshold=subcluster_outlier_threshold,
        )

    articles["subcluster"] = labels.astype(int)
    articles["cluster"] = articles["subcluster"]
    if include_map:
        reducer = umap.UMAP(
            n_neighbors=min(15, len(articles) - 1),
            n_components=2,
            min_dist=0.05,
            metric="cosine",
            random_state=42,
        )
        coordinates = reducer.fit_transform(embeddings)
        articles["x"] = coordinates[:, 0]
        articles["y"] = coordinates[:, 1]

    topic_rows: list[dict[str, object]] = []
    valid = articles[articles["cluster"] >= 0]
    denominator = max(1, len(valid))
    label_idf = _label_corpus_idf(valid["text"])
    for cluster_id, group in valid.groupby("cluster"):
        topic_name = _topic_keywords(group["text"], top_n=5, corpus_idf=label_idf)
        group_embeddings = embeddings[group.index.to_numpy()]
        cohesion_score = round(_cluster_cohesion(group_embeddings), 4)
        label_quality = _label_quality(topic_name)
        template_score = _cluster_template_score(group)
        template_penalty = _template_penalty(cohesion_score, template_score)
        topic_rows.append(
            {
                "cluster": int(cluster_id),
                "topic": topic_name,
                "article_count": int(len(group)),
                "share_percent": round(len(group) / denominator * 100, 2),
                "cohesion_score": cohesion_score,
                "label_quality": label_quality,
                "template_score": template_score,
                "template_penalty": template_penalty,
                "issue_score": _issue_score(
                    int(len(group)),
                    cohesion_score,
                    label_quality,
                    template_penalty=template_penalty,
                ),
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
