# TheOneTech Top 10

서울시 뉴스기사를 의미 유사도로 군집화하고, 기사 수가 많은 주제 Top 10을 자동 추출하는 MVP입니다.

## 주요 기능

- 기사 제목과 본문을 다국어 Sentence Transformer 임베딩으로 변환
- 코사인 유사도 기반 유사·중복 기사 제거
- UMAP 차원 축소와 HDBSCAN 군집화
- 군집별 TF-IDF 대표 키워드 및 대표 기사 추출
- 기사 수 기준 Top 10 이슈 산출
- Plotly 기반 2차원 기사 군집 지도
- 분석 결과 CSV 다운로드

## 입력 데이터

CSV에 다음 컬럼이 필요합니다.

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `title` | 예 | 기사 제목 |
| `body` | 예 | 기사 본문 |
| `url` | 아니요 | 원문 URL |
| `published_at` | 아니요 | 발행 시각 |
| `publisher` | 아니요 | 언론사 |

제목과 본문 컬럼명은 화면 왼쪽 설정에서 바꿀 수 있습니다.

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

## 다음 단계

- 서울시 뉴스룸·RSS·검색 API 수집기 연결
- 날짜 범위 및 언론사 필터
- 군집명 품질 개선을 위한 한국어 형태소 분석 또는 LLM 요약
- 일자별 이슈 변화 추적 및 데이터베이스 저장
- Docker 및 정기 실행 배포
