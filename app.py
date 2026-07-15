from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import analyze_articles


st.set_page_config(page_title="TheOneTech Top 10", layout="wide")
st.title("서울시 뉴스 이슈 Top 10")
st.caption("기사 의미 유사도를 기반으로 주제를 자동 군집화하고, 기사 수가 많은 이슈를 순위화합니다.")

uploaded = st.file_uploader("뉴스기사 CSV 업로드", type=["csv"])
with st.sidebar:
    st.header("분석 설정")
    title_col = st.text_input("제목 컬럼", value="title")
    body_col = st.text_input("본문 컬럼", value="body")
    min_cluster_size = st.slider("최소 군집 기사 수", min_value=3, max_value=30, value=5)
    duplicate_threshold = st.slider(
        "중복 판정 유사도",
        min_value=0.85,
        max_value=0.995,
        value=0.96,
        step=0.005,
    )

if uploaded is None:
    st.info("`title`, `body` 컬럼을 포함한 CSV를 올려 주세요. 선택 컬럼으로 `url`, `published_at`, `publisher`를 함께 둘 수 있습니다.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except UnicodeDecodeError:
    uploaded.seek(0)
    data = pd.read_csv(uploaded, encoding="cp949")

st.write(f"입력 기사: **{len(data):,}개**")

if st.button("Top 10 분석 실행", type="primary", use_container_width=True):
    with st.spinner("임베딩 모델 로드 및 군집화 중..."):
        try:
            result = analyze_articles(
                data,
                min_cluster_size=min_cluster_size,
                duplicate_threshold=duplicate_threshold,
                title_col=title_col,
                body_col=body_col,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    topics = result.topics.head(10)
    articles = result.articles

    col1, col2, col3 = st.columns(3)
    col1.metric("분석 기사", f"{len(articles):,}개")
    col2.metric("감지된 군집", f"{len(result.topics):,}개")
    col3.metric("기타/노이즈", f"{int((articles['cluster'] < 0).sum()):,}개")

    st.subheader("기사량 기준 Top 10 이슈")
    if topics.empty:
        st.warning("유효한 군집이 만들어지지 않았습니다. 최소 군집 기사 수를 낮춰 다시 실행해 보세요.")
    else:
        st.dataframe(
            topics[["rank", "topic", "article_count", "share_percent", "representative_title"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": "순위",
                "topic": "자동 추출 주제",
                "article_count": "기사 수",
                "share_percent": "군집 내 비중(%)",
                "representative_title": "대표 기사",
            },
        )

    st.subheader("기사 군집 지도")
    figure = px.scatter(
        articles,
        x="x",
        y="y",
        color="topic",
        hover_data=["title", "cluster"],
        title="UMAP 2차원 뉴스기사 군집",
    )
    figure.update_traces(marker={"size": 9, "opacity": 0.75})
    figure.update_layout(legend_title_text="주제", height=650)
    st.plotly_chart(figure, use_container_width=True)

    st.download_button(
        "분석 결과 CSV 다운로드",
        articles.to_csv(index=False).encode("utf-8-sig"),
        file_name="news_topic_clusters.csv",
        mime="text/csv",
        use_container_width=True,
    )
