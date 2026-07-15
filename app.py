from __future__ import annotations

import os
import inspect
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.bigkinds_csv import bigkinds_download_path, load_bigkinds_download, normalize_bigkinds_csv
from src.bigkinds_news import BigKindsConfig, BigKindsNewsClient
from src.database import ArticleDatabase
from src.naver_news import NaverNewsClient, NaverNewsConfig
from src.pipeline import analyze_articles, remove_exact_body_duplicates

load_dotenv()

st.set_page_config(page_title="TheOneTech Top 10", layout="wide")


@st.cache_data(show_spinner=False)
def run_analysis_cached(
    data: pd.DataFrame,
    min_cluster_size: int,
    duplicate_threshold: float,
    subcluster_outlier_threshold: float,
    fast_mode: bool,
    include_map: bool,
):
    analysis_kwargs = {
        "min_cluster_size": min_cluster_size,
        "duplicate_threshold": duplicate_threshold,
        "subcluster_outlier_threshold": subcluster_outlier_threshold,
        "fast_mode": fast_mode,
        "include_map": include_map,
        "title_col": "title",
        "body_col": "body",
    }
    supported_params = inspect.signature(analyze_articles).parameters
    analysis_kwargs = {
        key: value for key, value in analysis_kwargs.items() if key in supported_params
    }
    return analyze_articles(data, **analysis_kwargs)


st.title("지역 뉴스 이슈 Top 10")
st.caption(
    "빅카인즈 API, 네이버 API+DB, 빅카인즈 다운로드 파일 중 하나를 선택해 지역 뉴스 이슈를 군집화합니다."
)

database = ArticleDatabase()
configured_queries = [
    value.strip() for value in os.getenv("NEWS_QUERIES", "서울").split(",") if value.strip()
]

with st.sidebar:
    st.header("분석 설정")
    fast_mode = st.toggle(
        "빠른 모드",
        value=False,
        help="TF-IDF+SVD 기반으로 빠르게 군집화하고, 임베딩 기반 중복 제거와 군집 내부 재군집화를 건너뜁니다.",
    )
    include_map = st.toggle(
        "기사 군집 지도 생성",
        value=False,
        help="끄면 UMAP 좌표 계산을 생략해 Top N 표를 더 빨리 만듭니다.",
    )
    top_n = st.number_input("Top N 이슈 수", min_value=1, max_value=50, value=10, step=1)
    min_cluster_size = st.slider("최소 군집 기사 수", 3, 50, 30)
    duplicate_threshold = st.slider(
        "중복 판정 유사도",
        0.85,
        0.995,
        0.85,
        step=0.005,
        disabled=fast_mode,
        help="빠른 모드에서는 본문 완전중복만 제거하므로 이 값은 사용하지 않습니다.",
    )
    subcluster_outlier_threshold = st.slider(
        "군집 내부 이질 기사 제거 기준",
        0.20,
        0.80,
        0.45,
        step=0.01,
        help="높일수록 군집 중심에서 조금만 멀어도 이질 기사로 분리합니다.",
        disabled=fast_mode,
    )
    st.divider()
    st.metric("DB 전체 기사", f"{database.count():,}개")
    monthly = database.monthly_counts()
    if not monthly.empty:
        st.dataframe(monthly, use_container_width=True, hide_index=True)

source = st.radio(
    "기사 소스",
    ["빅카인즈 API", "네이버 API + DB", "빅카인즈 다운로드 파일"],
    horizontal=True,
)
data: pd.DataFrame | None = None

if source == "빅카인즈 다운로드 파일":
    use_file_date_filter = st.checkbox("파일 내부 기사에 기간 필터 적용", value=False)
else:
    use_file_date_filter = True

if use_file_date_filter:
    col1, col2, col3 = st.columns([1, 1, 1.4])
    start_date = col1.date_input("시작일", value=date.today() - timedelta(days=30))
    end_date = col2.date_input("종료일", value=date.today())
    selected_queries = col3.multiselect(
        "지역",
        options=configured_queries,
        default=configured_queries,
        help="지역 목록은 .env의 NEWS_QUERIES에서 설정합니다.",
    )
else:
    col1, col2 = st.columns([1.2, 2])
    selected_queries = col1.multiselect(
        "지역",
        options=configured_queries,
        default=configured_queries,
        help="파일 전체에서 선택 지역만 필터링합니다.",
    )
    col2.info("기간 필터를 적용하지 않고 빅카인즈 다운로드 파일 전체를 사용합니다.")
    start_date = date.min
    end_date = date.max

if source == "빅카인즈 API":
    st.info(
        "빅카인즈 API는 기간 조건을 요청에 직접 반영합니다. "
        ".env에 BIGKINDS_ACCESS_KEY를 설정해야 합니다."
    )
    if st.button("빅카인즈 API에서 수집", type="primary", use_container_width=True):
        try:
            config = BigKindsConfig.from_env()
            if selected_queries:
                config = BigKindsConfig(
                    access_key=config.access_key,
                    queries=tuple(selected_queries),
                    endpoint=config.endpoint,
                    page_size=config.page_size,
                    max_results_per_query=config.max_results_per_query,
                    timeout_seconds=config.timeout_seconds,
                )
            with st.spinner("빅카인즈 API 수집 중..."):
                data = BigKindsNewsClient(config).collect(start_date, end_date)
                st.session_state["bigkinds_api_data"] = data
            st.success(f"빅카인즈 API 결과: {len(data):,}개")
        except Exception as exc:
            st.error(str(exc))
    else:
        data = st.session_state.get("bigkinds_api_data")
        if data is not None:
            st.write(f"최근 빅카인즈 API 결과: **{len(data):,}개**")

elif source == "네이버 API + DB":
    st.warning(
        "네이버 API에는 기간 필터가 없어 최신 결과를 조회한 뒤 프로그램에서 기간을 거릅니다. "
        "검색어당 최대 1,000건까지만 접근할 수 있으므로 장기 수집은 매일 증분 수집을 권장합니다."
    )
    if st.button("네이버 API에서 수집해 DB에 저장", type="primary", use_container_width=True):
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
                f"네이버 API 결과 {len(collected):,}개, 신규 저장 {inserted:,}개, "
                f"보존 기간 초과 삭제 {deleted:,}개"
            )
        except Exception as exc:
            st.error(str(exc))

    data = database.query(start_date, end_date, selected_queries or None)
    st.write(f"DB 조회 기사: **{len(data):,}개**")
    if not data.empty:
        st.download_button(
            "DB 조회 결과 CSV 다운로드",
            data.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"news_db_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    input_mode = st.radio("파일 입력 방식", ["기본 경로 사용", "파일 업로드"], horizontal=True)
    if input_mode == "기본 경로 사용":
        path = st.text_input(
            "빅카인즈 다운로드 파일 경로",
            value=str(bigkinds_download_path()),
            help="BIGKINDS_DOWNLOAD_PATH 환경 변수로 기본 경로를 바꿀 수 있습니다.",
        )
        try:
            with st.spinner("빅카인즈 다운로드 파일 읽는 중..."):
                data = load_bigkinds_download(path, start_date, end_date, selected_queries)
            st.write(f"빅카인즈 다운로드 파일 조회 기사: **{len(data):,}개**")
        except Exception as exc:
            st.error(str(exc))

    uploaded = st.file_uploader("빅카인즈 다운로드 파일 업로드", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                try:
                    uploaded_frame = pd.read_csv(uploaded, dtype=str)
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    uploaded_frame = pd.read_csv(uploaded, dtype=str, encoding="cp949")
            else:
                uploaded_frame = pd.read_excel(uploaded, dtype=str)
            data = normalize_bigkinds_csv(uploaded_frame, start_date, end_date, selected_queries)
            st.write(f"업로드 파일 조회 기사: **{len(data):,}개**")
        except Exception as exc:
            st.error(str(exc))

if data is None or data.empty:
    st.info("기사를 수집하거나 DB 조회 기간을 조정하거나 CSV를 올려 주세요.")
    st.stop()

if data is None:
    raise SystemExit

preview_data = remove_exact_body_duplicates(data, body_col="body")
removed_exact_body_duplicates = len(data) - len(preview_data)
if removed_exact_body_duplicates:
    st.caption(
        f"본문이 완전히 같은 기사 {removed_exact_body_duplicates:,}개를 미리보기에서 제외했습니다. "
        "분석 실행 시에는 임베딩 유사도 기반 중복 제거도 추가로 적용됩니다."
    )
else:
    st.caption("본문 완전 중복은 발견되지 않았습니다. 분석 실행 시 임베딩 유사도 기반 중복 제거가 추가로 적용됩니다.")

st.dataframe(preview_data.head(30), use_container_width=True, hide_index=True)

if st.button("Top 10 분석 실행", type="primary", use_container_width=True):
    with st.spinner("임베딩 모델 로드 및 군집화 중..."):
        try:
            result = run_analysis_cached(
                data,
                min_cluster_size=min_cluster_size,
                duplicate_threshold=duplicate_threshold,
                subcluster_outlier_threshold=subcluster_outlier_threshold,
                fast_mode=fast_mode,
                include_map=include_map,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    topics = result.topics.head(int(top_n)).copy()
    articles = result.articles
    ranked_topic_map = {
        row["cluster"]: f"{int(row['rank'])}. {row['topic']}"
        for _, row in topics.iterrows()
    }
    topics["map_topic"] = topics["cluster"].map(ranked_topic_map)
    articles = articles.copy()
    articles["map_topic"] = articles["cluster"].map(ranked_topic_map)
    articles["map_topic"] = articles["map_topic"].fillna(
        articles["cluster"].map(lambda value: "노이즈/기타" if value < 0 else "Top N 외 군집")
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("분석 기사", f"{len(articles):,}개")
    col2.metric("감지된 군집", f"{len(result.topics):,}개")
    col3.metric("기타/노이즈", f"{int((articles['cluster'] < 0).sum()):,}개")

    st.subheader(f"기사 수 기준 Top {int(top_n)} 이슈")
    if topics.empty:
        st.warning("군집을 만들지 못했습니다. 최소 군집 기사 수를 낮춰 보세요.")
    else:
        topic_table = topics[
            [
                "rank",
                "map_topic",
                "issue_score",
                "article_count",
                "share_percent",
                "cohesion_score",
                "label_quality",
                "representative_title",
            ]
        ].rename(columns={"map_topic": "topic"})
        st.dataframe(
            topic_table,
            use_container_width=True,
            hide_index=True,
        )

    if include_map and {"x", "y"}.issubset(articles.columns):
        st.subheader("기사 군집 지도")
        figure = px.scatter(
            articles,
            x="x",
            y="y",
            color="map_topic",
            hover_data=["title", "cluster", "subcluster", "topic"],
            title="UMAP 2차원 뉴스기사 군집",
            category_orders={"map_topic": topics["map_topic"].tolist() + ["Top N 외 군집", "노이즈/기타"]},
        )
        figure.update_traces(marker={"size": 9, "opacity": 0.75})
        figure.update_layout(legend_title_text="Top N 주제", height=650)
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("기사 군집 지도 생성을 꺼서 UMAP 좌표 계산을 생략했습니다.")

    st.download_button(
        "분석 결과 CSV 다운로드",
        articles.to_csv(index=False).encode("utf-8-sig"),
        file_name="news_topic_clusters.csv",
        mime="text/csv",
        use_container_width=True,
    )
