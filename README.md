# TheOneTech Top 10

빅카인즈 API, 네이버 뉴스 검색 API+SQLite DB, 빅카인즈 다운로드 파일을 바탕으로 선택 기간의 지역 뉴스를 의미 유사도로 군집화해 Top N 이슈를 추출합니다.

## 기사 소스

앱에서 세 가지 소스 중 하나를 선택합니다.

- `빅카인즈 API`: `.env`의 `BIGKINDS_ACCESS_KEY`로 빅카인즈 API를 호출합니다.
- `네이버 API + DB`: 네이버 뉴스 검색 API로 수집한 뒤 SQLite DB에 저장하고, 선택 기간/지역으로 DB를 조회합니다.
- `빅카인즈 다운로드 파일`: 빅카인즈에서 내려받은 `csv`, `xlsx`, `xls` 파일을 읽어 분석합니다.

빅카인즈 다운로드 파일은 기본적으로 파일 내부 전체 기간을 사용합니다. 앱에서 `파일 내부 기사에 기간 필터 적용`을 켜면 시작일/종료일로 다시 거를 수 있습니다.

## 환경 설정

```bash
cp .env.example .env
```

```dotenv
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
BIGKINDS_ACCESS_KEY=...
NEWS_QUERIES=서울,광주,대전
BIGKINDS_DOWNLOAD_PATH=Top10_experiment/NewsResult_20250715-20260715_1year.csv
NEWS_DB_PATH=data/news.db
NEWS_RETENTION_DAYS=365
```

`NEWS_QUERIES`는 앱의 지역 선택 목록으로도 사용됩니다. 네이버 수집에서는 검색어 목록으로 쓰이고, 빅카인즈 다운로드 파일에서는 `위치`, `제목`, `본문`, `키워드`, `통합 분류` 컬럼에 해당 지역명이 포함된 기사를 남깁니다.

## 설치 및 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 네이버 수집 스크립트

최초 1년 초기 적재:

```bash
python scripts/bootstrap_year.py
```

매일 증분 수집:

```bash
python scripts/collect_daily.py
```

네이버 뉴스 검색 API는 언론사 필터 파라미터를 제공하지 않습니다. 따라서 전체 검색 결과 중 `originallink` 도메인이 `NEWS_PUBLISHER_DOMAINS` 허용 목록과 일치하는 기사만 저장합니다. 또한 API는 기사 전문이 아니라 제목, 요약 passage, 링크, 발행 시각을 제공합니다.

## 분석 흐름

1. 입력 데이터를 `title`, `body`, `publisher`, `published_at`, `url`, `query` 중심의 표준 컬럼으로 맞춥니다.
2. `title + body`를 합쳐 분석 텍스트를 만듭니다.
3. Sentence Transformer로 기사 임베딩을 만듭니다.
4. 같은 `body`가 반복되는 기사는 제목이 달라도 완전 중복으로 보고 먼저 제거합니다.
5. 남은 기사들은 임베딩 코사인 유사도로 한 번 더 중복 제거합니다. 앱의 `중복 판정 유사도` 슬라이더가 이 기준입니다.
6. UMAP으로 2차원 좌표를 만들고 HDBSCAN으로 군집화합니다.
7. 군집별 TF-IDF 키워드, 대표 기사, 점수, 지도 좌표를 생성합니다.

중복 판정은 `body`도 봅니다. 현재는 `body` 완전중복 제거와 `title + body` 임베딩 유사도 제거를 함께 사용합니다. 그래서 제목만 달라도 본문이 완전히 같으면 먼저 제거됩니다. 앱의 분석 버튼 위 미리보기 표도 본문 완전중복을 제외한 상태로 보여줍니다.

## Top N 순위 산정

Top N 표는 단순 기사 수만으로 정렬하지 않습니다. 현재 순위 기준은 `issue_score`입니다.

```text
issue_score = article_count × cohesion_factor × label_factor
cohesion_factor = cohesion_score ^ 1.5
label_factor = label_quality ^ 3
```

- `article_count`: 해당 군집의 기사 수입니다.
- `cohesion_score`: 군집 안 기사 임베딩이 중심점에 얼마나 가깝게 모였는지입니다. 1에 가까울수록 응집도가 높습니다.
- `label_quality`: 군집 라벨에 `기자`, `사진`, `있다` 같은 일반어가 적을수록 높습니다.
- `issue_score`: 크지만 잡음이 많은 군집은 강하게 낮추고, 조금 작아도 응집도와 라벨 품질이 좋은 군집은 올리기 위한 최종 점수입니다. `label_quality`가 0이면 점수도 0입니다.

표의 `rank`는 `issue_score`, `article_count`, `cohesion_score` 순으로 정렬해 부여합니다. 지도 범례는 Top N 표와 같은 rank 포함 라벨을 사용합니다. Top N 밖의 군집은 지도에서 `Top N 외 군집`으로 표시됩니다.

분석은 토픽 단위가 아니라 기사 단위로 시작합니다. 각 기사마다 `title + body` 임베딩을 만든 뒤, 기사끼리 의미적으로 가까운 것들을 HDBSCAN으로 군집화합니다. 토픽명은 군집이 만들어진 뒤 TF-IDF 키워드로 붙이는 후처리 라벨입니다.

## 해석상 주의

순위는 시민 여론이 아니라 수집된 기사 기준의 보도량과 군집 품질을 함께 반영한 값입니다. 빅카인즈 다운로드 파일은 사용자가 내려받은 범위에 의존하고, 네이버 검색 결과는 전체 언론 기사 아카이브를 보장하지 않습니다.

## 테스트

```bash
pytest
```
