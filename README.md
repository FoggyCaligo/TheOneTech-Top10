# TheOneTech Top 10

빅카인즈 API, 네이버 뉴스 검색 API+SQLite DB, 빅카인즈 다운로드 파일을 바탕으로 선택 기간의 지역 뉴스를 의미 유사도로 군집화해 기사 수 기준 Top 10 이슈를 추출합니다.

## 수집 구조

1. `.env`의 `NEWS_QUERIES`에 `서울,광주,대전`처럼 지역 검색어를 설정합니다.
2. 앱에서 기사 소스를 `빅카인즈 API`, `네이버 API + DB`, `빅카인즈 다운로드 파일` 중 하나로 선택합니다.
3. 기간과 지역을 선택합니다.
4. 빅카인즈 API는 기간 조건으로 직접 조회하고, 네이버 소스는 API 실행 후 SQLite DB를 조회합니다.
5. 빅카인즈 다운로드 파일은 `csv`, `xlsx`, `xls`를 읽어 표준 분석 컬럼으로 정규화합니다.
6. 네이버 수집 시에는 `originallink` 도메인이 메이저 언론사 허용 목록에 포함된 기사만 남깁니다.
7. 대시보드는 선택된 소스의 기사를 분석합니다.

네이버 뉴스 검색 API는 언론사 필터 파라미터를 제공하지 않습니다. 따라서 전체 검색 결과 중 원문 URL 도메인이 허용 목록과 일치하는 기사만 후처리로 저장합니다.

또한 API는 기사 전문이 아니라 제목, 기사 요약 passage, 원문 링크, 네이버 링크, 발행 시각을 제공합니다. 한 번에 최대 100건, 검색 시작 위치는 최대 1,000까지만 접근할 수 있어 전체 기사를 한 번에 완전 수집할 수는 없습니다.

## 환경 설정

```bash
cp .env.example .env
```

```dotenv
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
BIGKINDS_ACCESS_KEY=...
NEWS_QUERIES=서울,광주,대전
BIGKINDS_DOWNLOAD_PATH=Top10_experiment/NewsResult_20250715-20260715.csv
NEWS_PUBLISHER_DOMAINS=yna.co.kr,newsis.com,news1.kr,kbs.co.kr,imnews.imbc.com,sbs.co.kr,ytn.co.kr,jtbc.co.kr,chosun.com,joongang.co.kr,donga.com,hani.co.kr,khan.co.kr,hankookilbo.com,mk.co.kr,hankyung.com
NEWS_DB_PATH=data/news.db
NEWS_RETENTION_DAYS=365
```

`NEWS_PUBLISHER_DOMAINS`를 생략해도 코드의 기본 메이저 언론사 목록이 적용됩니다. 빈 설정 때문에 전체 언론사 수집으로 자동 전환되지는 않습니다.

빅카인즈 API를 사용할 때는 `BIGKINDS_ACCESS_KEY`를 설정하고 앱에서 `빅카인즈 API`를 선택합니다.

빅카인즈에서 내려받은 파일을 사용할 때는 앱에서 `빅카인즈 다운로드 파일`을 선택합니다. 기본 파일 경로는 `Top10_experiment/NewsResult_20250715-20260715.csv`이며, 필요하면 `.env`의 `BIGKINDS_DOWNLOAD_PATH`로 바꿀 수 있습니다. 다운로드 파일은 `csv`, `xlsx`, `xls`를 지원합니다.

## 설치 및 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 최초 1년 초기 적재

검색어를 `서울` 하나로 고정해, 실행 시점 기준 최근 1년 범위 안에서 네이버가 접근을 허용하는 최신 최대 1,000건을 조회합니다. 그중 메이저 언론사 기사만 DB에 저장합니다.

```bash
python scripts/bootstrap_year.py
```

같은 조건으로 반복 실행해도 URL 중복 데이터는 추가되지 않습니다.

## 매일 자동 수집

직접 한 번 실행:

```bash
python scripts/collect_daily.py
```

기본값은 오늘과 전날을 다시 조회해 지연 등록 기사를 보완합니다. 초기 적재와 동일한 메이저 언론사 도메인 필터가 적용됩니다.

Linux cron 예시:

```cron
10 6 * * * cd /path/to/TheOneTech-Top10 && /path/to/.venv/bin/python scripts/collect_daily.py >> logs/collector.log 2>&1
```

Windows에서는 작업 스케줄러에 `python scripts/collect_daily.py`를 매일 실행하도록 등록하면 됩니다.

## 분석 기능

- Sentence Transformer 기사 임베딩
- 코사인 유사도 기반 유사/중복 기사 제거
- UMAP 차원 축소와 HDBSCAN 군집화
- TF-IDF 기반 대표 키워드와 대표 기사 추출
- 기사 수 기준 Top 10 이슈 및 군집 산점도
- DB 조회 결과와 분석 결과 CSV 다운로드

## 테스트

```bash
pytest
```

## 해석상 주의

순위는 시민 여론이 아니라 수집된 기사 수, 즉 언론 보도량입니다. 네이버 검색 결과가 전체 언론 기사 아카이브를 보장하지 않으며, 메이저 언론사 필터는 네이버가 반환한 최대 1,000건 안에서만 적용됩니다.
