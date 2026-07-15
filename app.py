from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.database import ArticleDatabase
from src.naver_news import NaverNewsClient, NaverNewsConfig
from src.pipeline import analyze_articles

load_dotenv()

st.set_page_config(page_title="TheOneTech Top 10", layout="wide")
st.title("지역 뉴스 이슈 Top 10")
st.caption(
    "네이버 뉴스 검색 API 결과를 SQLite에 누적하고, 선택 기간의 기사를 의미 유사도로 군집화합니다."
)

database = ArticleDatabase()
configured_queries = [
    value.strip() for value in os.getenv("NEWS_QUERIES", "서울").split(",") if value.strip()
]

with st.sidebar:
    st.header("분석 설정")
    min_cluster_size = st.slider("최소 군집 기사 수", 3, 30, 5)
    duplicate_threshold = st.slider(
        "중복 판정 유사도", 0.85, 0.995, 0.96, step=0.005
    )
    st.divider()
    st.metric("DB 전체 기사", f"{database.count():,}개")
    monthly = database.monthly_counts()
    if not monthly.empty:
        st.dataframe(monthly, use_container_width=True, hide_index=True)

source = st.radio(
    "기사 입력 방식",
    ["SQLite DB", "네이버 API 수집", "CSV 업로드"],
    horizontal=True,
)
data: pd.DataFrame | None = None

if source in {"SQLite DB", "네이버 API 수집"}:
    col1, col2 = st.columns(2)
    start_date = col1.date_input("시작일", value=date.today() - timedelta(days=30))
    end_date = col2.date_input("종료일", value=date.today())
    selected_queries = st.multiselect(
        "검색어",
        options=configured_queries,
        default=configured_queries,
        help="검색어 목록은 .env의 NEWS_QUERIES에서 설정합니다.",
    )

    if source == "네이버 API 수집":
        st.warning(
            "네이버 API에는 기간 필터가 없어 날짜순 결과를 순회한 뒤 프로그램에서 기간을 거릅니다. "
            "검색어당 최대 1,000건까지만 접근할 수 있으므로 장기간 일괄 수집보다 매일 증분 수집을 권장합니다."
        )
        if st.button("API에서 수집해 DB에 저장", type="primary", use_container_width=True):
            try:
                config = NaverNewsConfig.from_env()
                if selected_queries:
                    config = NaverNewsConfig(
                        client_id=config.client_id,
                        client_secret=config.client_secret,
                        queries=tuple(selected_queries),
                        publisher_domains=config.publisher_domains,
                        page_size=config.page_size,
                        max_results_per_query=config.max_results_per_query,
                        timeout_seconds=config.timeout_seconds,
                        request_delay_seconds=config.request_delay_seconds,
                    )
                with st.spinner("네이버 뉴스 API 수집 중..."):
                    collected = NaverNewsClient(config).collect(start_date, end_date)
                    inserted = database.upsert_dataframe(collected)
                    deleted = database.enforce_retention()
                st.success(
                    f"API 결과 {len(collected):,}개, 신규 저장 {inserted:,}개, "
                    f"1년 초과 삭제 {deleted:,}개"
                )
            except Exception as exc:
                st.error(str(exc))

    data = database.query(start_date, end_date, selected_queries or None)
    st.write(f"DB 조회 기사: **{len(data):,}개**")
    if not data.empty:
        st.download_button(
            "조회 결과 CSV 다운로드",
            data.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"news_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    uploaded = st.file_uploader("뉴스기사 CSV 업로드", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            data = pd.read_csv(uploaded, encoding="cp949")

if data is None or data.empty:
    st.info("기사를 수집하거나 DB 조회 기간을 조정하거나 CSV를 올려 주세요.")
    st.stop()

st.dataframe(data.head(30), use_container_width=True, hide_index=True)

if st.button("Top 10 분석 실행", type="primary", use_container_width=True):
    with st.spinner("임베딩 모델 로드 및 군집화 중..."):
        try:
            result = analyze_articles(
                data,
                min_cluster_size=min_cluster_size,
                duplicate_threshold=duplicate_threshold,
                title_col="title",
                body_col="body",
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
        st.warning("군집이 만들어지지 않았습니다. 최소 군집 기사 수를 낮춰 보세요.")
    else:
        st.dataframe(
            topics[["rank", "topic", "article_count", "share_percent", "representative_title"]],
            use_container_width=True,
            hide_index=True,
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
