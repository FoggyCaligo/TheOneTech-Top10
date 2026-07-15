# TheOneTech Top 10

기간·언론사·검색어를 기준으로 빅카인즈 기사를 수집하거나 CSV를 업로드해, 의미 유사도 기반 뉴스 이슈 Top 10을 추출하는 MVP입니다.

## 주요 기능

- 빅카인즈 API에서 시작일·종료일·언론사·검색어(`서울` 기본값) 기반 기사 자동 수집
- 페이지를 반복 조회해 전체 결과를 CSV로 구성
- 기사 제목과 본문을 다국어 Sentence Transformer 임베딩으로 변환
- 코사인 유사도 기반 유사·중복 기사 제거
- UMAP 차원 축소와 HDBSCAN 군집화
- 군집별 TF-IDF 대표 키워드 및 대표 기사 추출
- 기사 수 기준 Top 10 이슈와 Plotly 군집 지도 출력
- 수집 원본 및 분석 결과 CSV 다운로드

## 빅카인즈 API 설정

승인 후 발급받은 API 명세에 맞춰 환경변수를 설정합니다.

```bash
cp .env.example .env
```

필수값:

```dotenv
BIGKINDS_API_URL=승인받은_뉴스검색_API_URL
BIGKINDS_API_KEY=발급받은_API_KEY
```

공개 페이지에서는 상세 요청·응답 명세가 확인되지 않으므로, 실제 승인 문서가 기본값과 다르면 다음 항목을 조정합니다.

```dotenv
BIGKINDS_AUTH_HEADER=Authorization
BIGKINDS_AUTH_SCHEME=Bearer
BIGKINDS_ITEMS_PATH=documents
BIGKINDS_TOTAL_PATH=total_hits
```

`BIGKINDS_ITEMS_PATH`와 `BIGKINDS_TOTAL_PATH`는 `result.documents`처럼 점 표기법을 지원합니다. 승인 명세에서 요청 본문의 키 이름 자체가 다를 경우 `src/bigkinds.py`의 `build_payload()`만 수정하면 됩니다.

기본 요청 형태는 다음과 같습니다.

```json
{
  "query": "서울",
  "published_at": {"from": "2026-07-01", "until": "2026-07-15"},
  "provider": ["한겨레", "KBS"],
  "page": 1,
  "size": 100,
  "sort": {"published_at": "desc"}
}
```

## CSV 형식

빅카인즈 응답은 다음 공통 형식으로 정규화됩니다. 같은 형식의 CSV를 직접 업로드할 수도 있습니다.

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `title` | 예 | 기사 제목 |
| `body` | 예 | 기사 본문 |
| `url` | 아니요 | 원문 URL |
| `published_at` | 아니요 | 발행 시각 |
| `publisher` | 아니요 | 언론사 |
| `id` | 아니요 | 기사 식별자 |

## 실행

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

최초 분석 시 Hugging Face에서 임베딩 모델을 내려받으므로 인터넷 연결이 필요합니다.

## 테스트

```bash
pytest
```

## 분석 해석 시 주의점

이 순위는 시민 여론의 찬반 비율이 아니라 **수집된 뉴스기사 안에서 보도량이 큰 주제 순위**입니다. 동일 보도자료를 여러 언론사가 전재하면 순위가 과대평가될 수 있어 중복 제거 임계값을 함께 조정해야 합니다.

## 남은 확인 사항

- 승인받은 빅카인즈 API의 실제 엔드포인트·인증 헤더·요청 필드명
- 언론사명이 API에서 요구하는 코드인지 표시명인지
- 한 요청당 최대 건수와 호출 한도
- 본문 전문 제공 및 저장 가능 범위
