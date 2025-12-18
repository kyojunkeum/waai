# waai

## 1. 프로젝트 개요 (Overview)

이 프로젝트는 작가들에게 자신만의 보조 작가를 채용하는 것 대신 AI 를 활용하여 시간이 지날수록 자신에게 최적화된 보조 작가를 활용할 수 있도록 고안된 프로젝트입니다. 
Writer Assistant AI. WAAI 가 당신만의 기호를 충족하는 보조 작가가 되어 기획, 웹검색 기반 자료조사, 당신의 인생그래프, 원고 합평까지 수행해 줍니다.

## 2. 왜 이 프로젝트가 필요한가? (Problem)

하나. 작가지망생이 가장 어렵고 감잡기가 힘든 부분이 바로 자료를 검색하고 정리하고 그것을 기반으로 글을 기획하는 일입니다. 
하나하나 검색해야 하고, 일기들을 하나하나 읽어봐야 하고, 여러 자료들을 끊임없이 들여다보며 하나의 완성된 뼈대, 기획을 하는 것이 가장 지난하고 중요합니다. 
WAAI 는 데이터를 검색하고, 저장하고, 그것들을 토대로 한 편의 글 또는 작품을 기획해줍니다. 어느 근거로부터 왔는지까지 정리해줍니다. 
즉, 근거 데이터 기반으로, 일기가 많다면, 나의 삶을 기반으로 글을 기획해주는 것입니다. 

둘. 작가지망생이 가장 필요로 하는 것은 작품을 쓰는 게 아닙니다. 작품을 쓰고 싶기 때문에 글을 쓰는 것이기에, 그보다는 내가 쓴 글을 평가받는 것이 가장 힘들고 곤란합니다. 
왜? 내 글의 평가를 맡길 만한 사람이 주변에 없습니다. 그러기 위해서는 기성 작가의 문하생으로 들어가거나, 민간 글쓰기 모임에 합류하여 합평을 해야 합니다. 
그러나, 기성 작가의 문하생이 되기란 정말 어려운 일이며, 민간 글쓰기 모임에 가는 것또한 두렵습니다. 섣불리 다가가기가 힘듭니다. 
WAAI 는 전국 공모전에서 수상한 진짜 작가가 만든 합평 기준을 활용하여 원고를 평가해줍니다. 총 14항목으로 이루어져 있으며, 합평 결과로 각 항목마다 10점 만점으로 점수가 책정되어 결과로 도출됩니다. 
즉, 근거 데이터 (규칙) 기반으로 원고를 평가하고 그 결과를 받아볼 수 있는 것입니다. 

셋. 작가지망생이 작가로서 쌓아가야 하는 가장 중요한 것이 '자신의 인생'입니다. 자신의 인생이 어땠는지, 어떤 사건이 있었는지, 그로 인한 심경의 변화는 무엇인지 등등이 작품 세계 에 녹아듭니다. 
WAAI 는 사용자가 기록한 일기를 기반으로 시간대별로 mood 와 mood_score 를 책정합니다. 그것을 그래프로 시각화합니다. 
실제로 개발하면서 보게 된 이 인생그래프가 가장 인상깊었습니다. 삶은 항상 좋지도 안좋지도 않다, 즉, 좋고 나쁨이 굴곡진 게 바로 인생이다 라는 것을 눈으로 목도할 수 있습니다.

## 3. 주요 기능 (Features)

하나. 데이터 정제 : 일기, 아이디어, 웹검색 등등 다양한 데이터를 기록하고, 자동으로 정제합니다. 
둘. 기획서 생성 : 여러 데이터들을 기반으로 사용자가 OpenWebUI 프롬프트에 친 "키워드1", "키워드2"를 기반으로 파일 목록을 수집하고, 종합 요약한 뒤, 최종적으로 기획서를 작성합니다. 
셋. 웹 검색 자료 저장 : 사용자가 "키워드1", "키워드2" 등을 입력하고 자료조사를 요청하면 조사한 기사의 텍스트를 추출하여 자동으로 저장합니다. 
넷. 원고 합평 : 사용자가 프롬프트에 원고를 입력하면 그것을 기반으로 보유한 합평기준규칙에 따라 평가하고, 항목당 점수를 책정하여 원고와 결과를 저장합니다. 

## 4. 시스템 구조 / 흐름 (Architecture) 

### 하나. 데이터 정제

### 동작 로직

[ User : /home/username/memory/ 에 정제되지 않은데이터 저장]

↓

[ Data-Format-Bot : 파일 생성을 감지하면 waai-[backend](http://backend.app.py)로 api/data/reformat-md API 호출 ]

↓

[ Waai-backend : 데이터 정제용 프롬프트를 활용해서 build_data_repair_prompt 함수 호출(프롬프트 포함) ]

↓

[ Ollama : ollama 컨테이너가 데이터 정제 후 응답 ]

↓

[ Waai-backend : 정제된 data 를 /home/username/waai/data/ 에 저장 ]

<img width="420" height="765" alt="image" src="https://github.com/user-attachments/assets/8740f6da-6df5-4dc6-bd53-74e5e3c77376" />

---------------------------------------------------

### 둘. 기획서 생성 기준

### 동작 로직

[ User : 사용자 프롬프트 입력 ]

↓

[ OpenWebUI : waai-backend 의  /api/plan/from-prompt API 호출 ]

↓

[ Waai-backend : 날짜/키워드 등 기획서 생성 근거 조건 추출을 위해 Ollama llm 호출 ]

↓

[ Ollama : 날짜/키워드 등 기획서 생성 근거 조건 추출 후 MCP-bridge 에 근거 조건 결과 전송 ]

↓

[ MCP-bridge : MCP-filesystem 에게 파일 목록 수집 요청 ]

↓

[ MCP-filesystem : diary, ideas, web_research, works, bible 에서 조건에 맞는 파일 목록 수집하여 MCP-bridge 로 전송 ]

↓

[ MCP-bridge : 받은 파일목록과 요약 전용 프롬프트로 Ollama llm 호출 ]

↓

[ Ollama : 파일 목록을 읽어 종합 요약 생성 후 MCP-bridge 로 전송 ]

↓

[ MCP-bridge : 파일 목록과 요약 정보를 waai-backend 로 전송 ]

↓

[ Waai-backend : 파일 목록과 종합 요약, 그리고 기획서 생성 전용 프롬프트로 Ollama llm 호출 ]

↓

[ Ollama : 기획서 생성 후 waai-backend 로 전송 ] 

↓

[ Waai-backend : 생성된 기획서를 /waai/data/outputs/ 에 저장 및 OpenWebUI 에 응답 전송 ] 

↓

[ OpenWebUI  : 생성된 기획서 일부를 UI 에 출력 ]

<img width="893" height="797" alt="image" src="https://github.com/user-attachments/assets/bff85c5f-f63c-44e3-ad93-8b1cdf00169d" />

---------------------------------------------------

### 셋. 웹 검색

### 동작 로직

[ User : UI 에서 “키워드1” 자료 조사해줘 프롬프트 입력 ] 

↓

[ OpenWebUI : UI와 연동된 waai-backend의 /api/web_search/fetch API 를 호출 ]

↓

[ Waai-backend : 프롬프트의 ‘, “ 로 감싸진 항목들을 제거(정규화) ]

↓

[ Waai-backend : api/web_search/fetch 함수를 호출하여 SEARXNG 혹은 google_news_rss 검색 수행 ]

↓

[ Waai-backend : 검색한 결과 수집된 링크를 리스트 형태로 수집 ]

↓

[ Waai-backend : 링크, 제목, 본문 만 남도록 불필요한 텍스트 항목 제거 후 MCP-Playwright 호출] 

↓

[ MCP-Playwright : url 브라우저 진입 후 텍스트 추출한 데이터를 waai-backend로 전송 ]

↓

[ Waai-backend : 데이터를 종합하여 /waai/data/web_research/ 에 데이터(md파일) 저장 ]

↓

[ OpenWebUI : 최종 응답을 기반으로 조사된 링크 목록을 사용자에게 출력 ]

<img width="529" height="821" alt="image" src="https://github.com/user-attachments/assets/8854aefa-c749-478b-b372-f54c90ced8c7" />

---------------------------------------------------

### 넷. 원고 합평

### 동작 로직

[ User : 입력된 원고를 프롬프트에 입력

↓

[ OpenWebUI : waai-backend 에 /api/critique API 호출 ]

↓

[ Waai-backend : 받은 원고를 txt 형태로 /waai/data/critique/objects/ 에 저장 ]

↓

[ Waai-backend : /waai/data/critique/criteria/ 에 있는 합평 기준 불러오기 ]

↓

[ Waai-backend : 원고 + 합평 기준 + 합평 전용 프롬프트로 Ollama llm 호출 ]

↓

[ Ollama : 합평 기준별로 원고를 읽고 점수 책정 및 결과 도출 후 waai-backend 로 결과 전송 ]

↓

[ Waai-backend : 결과 물을 md 파일로 /waai/data/critique/results/ 에 저장 및 OpenWebUI 에 응답 전송 ]

↓

[ OpenWebUI : 완성된 합평 결과 일부를 UI 에 출력 ]

<img width="696" height="784" alt="image" src="https://github.com/user-attachments/assets/f7e4b636-e1a3-4bd4-90c9-2fc6556afa16" />

---------------------------------------------------

## 6. 사용 기술 (Tech Stack)

하나. Core Architecture

## RAG 인덱싱/실행 가이드 (V4.0 추가)

### 인덱싱 순서(권장)
- 1) web_research, bible : 외부 자료/메모 먼저
- 2) critique/criteria   : 합평 기준 14개 이상 확보
- 3) ideas, works        : 창작 아이디어/기존 작품
- 4) diary               : 일기 데이터

### 초기 인덱싱 커맨드 예시
```bash
curl -X POST http://localhost:8000/api/rag/index_all \
  -H "Content-Type: application/json" \
  -d '{
    "doc_types": [
      "web_research", "bible",
      "critique_criteria", "critique", "critique/results",
      "ideas", "works",
      "diary"
    ],
    "force": true
  }'
```

### 환경변수 주요 목록
- `OLLAMA_BASE_URL` (기본 `http://ollama:11434`)
- `OLLAMA_EMBED_MODEL` (예: `nomic-embed-text`)
- `CHROMA_PERSIST_DIR` (기본 `/waai/chroma`, 권한 안 되면 `~/.waai/chroma` 폴백)
- `CHROMA_COLLECTION` (기본 `documents`)
- 기존 설정: `OUTPUT_ROOT`, `CRITIQUE_*`, `DIARY_ROOT`, `IDEAS_ROOT`, `WEB_RESEARCH_ROOT`, `WORKS_ROOT`, `BIBLE_ROOT`, `PLAYWRIGHT_MCP_URL` 등

### 폴백 전략 (use_rag=false)
- 기획/합평 요청에서 `use_rag=false`이거나 RAG 쿼리 실패 시 자동으로 기존 V3.4 MCP 요약/합평 흐름으로 폴백합니다.
- RAG 미사용 상태에서도 기존 기능은 동일하게 동작하며, RAG 사용 시에만 citations md가 생성되고 [S번호] 인용이 강제됩니다.
  - Docker / Docker Compose (컨테이너로 분리)
  - Microservice-oriented Design (Backend / MCP / Bot / UI)

둘. Backend & API
  - Python 3.11
  - FastAPI
  - Pydantic (요청/응답 데이터 검증 _ OpenWebUI HTTP Tool 연동 안정성)
  - CORS Middleware (OpenWebUI <-> Backend 통신 지원)

셋. LLM & AI Inference
  - Ollama
  - Qwen
  - Prompt Engineering (목적별 프롬프트 분리)

넷. MCP (Model Context Protocol)
  - MCP Filesystem Server (파일 리소스 제공, 조건 기반 파일 필터링)
  - MCP Playwright Server (웹 크롤링, JS 렌더링 기반 본문 추출)
  - MCP Bridge (Backend 연결다리; LLM - Filesystem - Bridge - Backend - UI; 프롬프트 -> 컨텍스트 -> 생성 파이프라인의 핵심)

다섯. Web Research & Crawling
  - Google Search (검색 및 URL 수집)
  - Playwright (크롤링, 본문 추출)
  - Custom HTML Parser (제목, 링크, 본문 등 추출, 불필요한 스크립트/광고 제거)
  - 자동 저장 파이프라인 (검색 -> 본문 추출 -> 텍스트 저장 -> 후처리)

여섯. Data Processing & Formatting
  - Markdown + YAML Front Matter (모든 데이터를 구조화)
  - Data-Format-Bot (Raw text -> 데이터 정제, 타입별 스키마 유지 등)
  - LLM-Assisted Metadata Enrichment (일부 필드 자동 생성)

일곱. OpenWebUI Integration
  - OpenWebUI (사용자 프롬프트 입력 UI, HTTP Tool 기반 API 호출)
  - HTTP Tool Design (기능별 API 분리)
  - Container Network Integration

여덟. Storage Strategy
  - File-based Knowledge Store

아홉. Operations
  - waai-monitor (각 컨테이너별 헬스체크 모니터링)

열. Design Philosophy
  - Human-readable AI : 모든 AI 결과는 사람이 읽고 검토 가능한 문서로 남긴다
  - LLM is a Reasoner, not a Database : 의미, 해석, 창작은 LLM이 / 구조, 보장은 코드가 수행
  - 점진적 고도화 수행 중 : V2(일기 기반) -> V3(데이터 확장 및 웹검색, 합평 기능) -> V4(VectorDB/RAG 시스템) -> V5(학습모델)


## 7. 실행 방법 (How To Run)

docker-compose.yml 이 있는 루트경로 /waai/ 로 진입합니다. 
docker compose up -d --build 를 통해 이미지를 생성하면 필요한 프로그램을 설치합니다. 
단, 컨테이너 활성화 후 ollama 컨테이너에 들어가서 ollma, qwen:7b 모델을 별도로 설치해야 합니다. 
시작 전에 /home/witness(username)/memory/{diary, works, ideas, bible, webresearch, critique} 폴더를 미리 생성하시기 바랍니다. 
데이터 경로 이름은 기호에 따라 변경해도 됩니다. 단, 백엔드의 app.py 나 mcp-filesystem 처럼 파일경로를 참조하는 부분을 모두 수정해야 합니다.

## 8. 한계와 개선 방향 (Limitations & Future Work)

구조적RAG 구성은 완료되었지만 향후 VectorDB 를 활용하여 정확도와 정밀도를 높여야 합니다. 
또한, 학습모델을 도입해서 기획서 생성과 합평기능을 할 때 좀더 최적화된 모델이 되도록 성능을 향상해야 합니다. 

## 9. 이 프로젝트를 통해 얻은 것 (Lesson & Learn)

하나. LLM 서비스가 어떠한 방식으로 동작하는지 이해하게 됐습니다. 모두 구현하는 줄 알았지만 실질적으로는 LLM 은 이미 생성된 모델을 사용하고, API 를 통해 거대언어모델을 사용하는 구조를 이해했습니다. 

둘. MCP와 LLM 이 어떻게 상호작용할 수 있는지 이해했습니다. LLM 은 '뇌'역할을, MCP 는 '손'과 '발'이 된다는 것이 어떠한 이유에서 표현된 것인지 알게 됐습니다. 
고 ChatGPT와 같은 범용 거대언어모델에서 '도구'라는 개념이 어떤식으로 구현되는지 (API 통해 백엔드 기능 호출), 사용자 프롬프트를 어떤 API로 전송해서 llm 에게 질의하여 어떤 시너지를 낼 수 있는지를 확인했습니다. 

셋. LLM 을 잘 활용하면 기존에 사람 한 명이 하던 일을 둘, 셋 이 하는 것처럼 효율을 낼 수 있습니다. 그리고 '학습'이라는 특성으로 인해 시간이 지날수록 최적화된 모델을 사용할 수 있다는 것이 매력포인트입니다. 

마지막. 최종적으로 WAAI 라는 나만의 보조 작가 AI를 얻게 되었습니다. 앞으로 고도화할 길이 멀지만, 하나씩 차근차근 기능을 추가해가면서 나만의 강력한 보조 작가 AI 로서 성장시킬 것입니다.
