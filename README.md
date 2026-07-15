# TheOneTech Top 10

네이버 뉴스 검색 API 결과를 SQLite에 매일 누적하고, 선택 기간의 지역 뉴스를 의미 유사도로 군집화해 기사량 기준 Top 10 이슈를 추출합니다.

## 수집 구조

1. `.env`의 `NEWS_QUERIES`에 `서울,광주,대전`처럼 지역 검색어를 설정합니다.
2. 네이버 API를 날짜순으로 조회합니다.
3. `originallink` 도메인이 메이저 언론사 허용 목록에 포함된 기사만 남깁니다.
4. URL 기준으로 SQLite에 중복 없이 저장합니다.
5. 매 실행 시 `NEWS_RETENTION_DAYS`보다 오래된 기사를 삭제합니다.
6. 대시보드는 API를 다시 호출하지 않고 DB에서 기간별 기사를 조회해 분석합니다.

네이버 뉴스 검색 API는 언론사 필터 파라미터를 제공하지 않습니다. 따라서 전체 검색 결과 중 원문 URL 도메인이 허용 목록과 일치하는 기사만 후처리해 저장합니다.

또한 API는 기사 전문이 아니라 제목, 기사 요약 passage, 원문 링크, 발행시각을 제공합니다. 한 번에 최대 100건, 검색 시작 위치는 최대 1,000이므로 장기간 전체 기사를 한 번에 완전 수집할 수는 없습니다.

## 환경 설정

```bash
cp .env.example .env
```

```dotenv
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
NEWS_QUERIES=서울,광주,대전
NEWS_PUBLISHER_DOMAINS=yna.co.kr,newsis.com,news1.kr,kbs.co.kr,imnews.imbc.com,sbs.co.kr,ytn.co.kr,jtbc.co.kr,chosun.com,joongang.co.kr,donga.com,hani.co.kr,khan.co.kr,hankookilbo.com,mk.co.kr,hankyung.com
NEWS_DB_PATH=data/news.db
NEWS_RETENTION_DAYS=365
```

`NEWS_PUBLISHER_DOMAINS`를 생략해도 코드 내 기본 메이저 언론사 목록이 적용됩니다. 전체 언론사 저장으로 자동 전환되지 않으며, 목록은 회사 기준에 맞게 추가하거나 제거할 수 있습니다.

## 설치 및 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 최초 1회 초기 적재

검색어를 오직 `서울` 하나로 고정해, 실행 시점 기준 최근 1년 범위 안에서 네이버가 접근을 허용하는 최신 최대 1,000건을 조회합니다. 이 중 메이저 언론사 기사만 DB에 저장합니다.

```bash
python scripts/bootstrap_year.py
```

같은 조건으로 반복 실행해도 1,000건보다 더 과거로 내려가지는 않으며, URL 중복 데이터는 추가되지 않습니다.

## 매일 자동 수집

직접 한 번 실행:

```bash
python scripts/collect_daily.py
```

기본값은 오늘과 전날을 다시 조회해 늦게 등록된 기사를 보완합니다. 초기 적재와 동일한 메이저 언론사 도메인 필터가 적용됩니다.

Linux cron 예시 — 매일 오전 6시 10분:

```cron
10 6 * * * cd /path/to/TheOneTech-Top10 && /path/to/.venv/bin/python scripts/collect_daily.py >> logs/collector.log 2>&1
```

Windows에서는 작업 스케줄러에 `python scripts/collect_daily.py`를 매일 실행하도록 등록합니다.

## 분석 기능

- Sentence Transformer 기사 임베딩
- 코사인 유사도 기반 유사·중복 기사 제거
- UMAP 차원 축소와 HDBSCAN 군집화
- TF-IDF 기반 대표 키워드·대표 기사 추출
- 기사 수 기준 Top 10 및 군집 산점도
- DB 조회 결과와 분석 결과 CSV 다운로드

## 테스트

```bash
pytest
```

## 해석상 주의

순위는 시민 여론이 아니라 수집된 기사 내 보도량입니다. 네이버 검색 결과는 전체 언론 기사 아카이브를 보장하지 않으며, 메이저 언론사 필터는 네이버가 반환한 최대 1,000건 안에서 적용되므로 그 범위 밖의 메이저 언론사 기사는 수집할 수 없습니다.
