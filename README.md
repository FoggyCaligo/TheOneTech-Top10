# TheOneTech Top 10

네이버 뉴스 검색 API 결과를 SQLite에 매일 누적하고, 선택 기간의 지역 뉴스를 의미 유사도로 군집화해 기사량 기준 Top 10 이슈를 추출합니다.

## 수집 구조

1. `.env`의 `NEWS_QUERIES`에 `서울,광주,대전`처럼 검색어를 설정합니다.
2. 일일 수집 명령이 네이버 API를 날짜순으로 조회합니다.
3. 당일 또는 최근 2일 기사만 골라 URL 기준으로 SQLite에 중복 없이 저장합니다.
4. 매 실행 시 `NEWS_RETENTION_DAYS`보다 오래된 기사를 삭제합니다.
5. 대시보드는 API를 다시 호출하지 않고 DB에서 기간별 기사를 조회해 분석합니다.

네이버 뉴스 API는 기사 전문이 아니라 제목, 기사 요약 passage, 원문 링크, 발행시각을 제공합니다. 또한 한 번에 최대 100건, 검색 시작 위치는 최대 1,000이므로 장기간을 한 번에 완전 수집하기보다 매일 증분 수집하는 방식이 적합합니다.

## 환경 설정

```bash
cp .env.example .env
```

```dotenv
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
NEWS_QUERIES=서울,광주,대전
NEWS_PUBLISHER_DOMAINS=yna.co.kr,kbs.co.kr,sbs.co.kr,chosun.com,joongang.co.kr,donga.com,hani.co.kr,khan.co.kr,hankookilbo.com
NEWS_DB_PATH=data/news.db
NEWS_RETENTION_DAYS=365
```

`NEWS_PUBLISHER_DOMAINS`를 비워두면 검색 결과 전체를 저장합니다. 메이저 언론사만 대상으로 하려면 허용할 원문 도메인을 쉼표로 입력합니다.

## 설치 및 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 매일 자동 수집

직접 한 번 실행:

```bash
python scripts/collect_daily.py
```

기본값은 오늘과 전날을 다시 조회해 늦게 등록된 기사를 보완합니다. URL이 같으면 갱신만 하므로 중복 저장되지 않습니다.

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

순위는 시민 여론이 아니라 수집된 기사 내 보도량입니다. 네이버 검색 결과는 전체 언론 기사 아카이브를 보장하지 않으며, 검색어별 최대 접근 범위 때문에 기사량이 매우 많은 날에는 일부 결과가 누락될 수 있습니다.
