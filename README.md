# waai

## 1. 프로젝트 개요 (Overview)

이 프로젝트는 작가들에게 자신만의 보조 작가를 채용하는 것 대신 AI 를 활용하여 시간이 지날수록 자신에게 최적화된 보조 작가를 활용할 수 있도록 고안된 프로젝트입니다. 
Writer Assistant AI. WAAI 가 당신만의 기호를 충족하는 보조 작가가 되어 기획, 웹검색 기반 자료조사, 당신의 인생그래프, 원고 합평, 브레인스토밍 까지 수행해 줍니다.
특히, WAAI는 데이터 정제, 합평과 같은 규격화된 일은 refine 모델 / 기획서 생성과 같이 창의성과 구조화가 필요한 일은 creative 모델 / 브레인스토밍과 같이 극대화된 창의성이 필요한 일은 brainstorm 모델이 수행합니다. 
목적별로 모델 파라미터를 최적화 설계/구현된 LLM 모델이 당신과 아이디어를 나누고, 자료를 정리해주며, 기획서를 만들어줍니다. 그리고 확실한 평가 기준을 통해 원고를 합평해줍니다.

## 2. 왜 이 프로젝트가 필요한가? (Problem)

하나. 작가지망생이 가장 어렵고 감잡기가 힘든 부분이 바로 자료를 검색하고 정리하고 그것을 기반으로 글을 기획하는 일입니다. 
하나하나 검색해야 하고, 일기들을 하나하나 읽어봐야 하고, 여러 자료들을 끊임없이 들여다보며 하나의 완성된 뼈대, 기획을 하는 것이 가장 지난하고 중요합니다. 
WAAI 는 데이터를 검색하고, 저장하고, 그것들을 토대로 한 편의 글 또는 작품을 기획해줍니다. 심지어 WAAI와 함께 대화하며 최적화된 창의 모델과 아이디어의 브레인스토밍을 할 수 있고, 그 결과까지 정리하여 저장해줍니다.

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

다섯. 브레인 스토밍 : 사용자가 AI와 프롬프트를 통해 다양한 아이디어를 나눕니다. 이때 브레인스토밍에 최적화된 모델과 대화를 하게 되며, 아이디어를 정제하여 저장하고 싶다면 /api/brainstorm/finalize 도구를 선택한 뒤 프롬프트를 보내면 
대화를 기반으로 3가지의 아이디어를 도출하여 정리한 뒤 정제하여 /waai/data/ideas 에 저장합니다.

## 4. 시스템 구조 / 흐름 (Architecture) 

### 하나. 데이터 정제

### 동작 로직

[ User : /home/username/memory/ 에 정제되지 않은데이터 저장]

↓

[ Data-Format-Bot : 파일 생성을 감지하면 waai-[backend](http://backend.app.py)로 api/data/reformat-md API 호출 ]

↓

[ Waai-backend : 데이터 정제용 프롬프트를 활용해서 build_data_repair_prompt 함수 호출(프롬프트 포함) ]

↓

[ Ollama : qweb2.5-refine-ko 모델이 데이터 정제 후 응답 ]

↓

[ Waai-backend : validate_date_front_matter 함수에서 메타 데이터 유효성 검증 후 누락 발견 시 llm 재호출 ] 

↓

[ Ollama : qweb2.5-refine-ko 모델이 데이터 재정제 후 응답 ]

↓

[ Waai-backend : 정제된 data 를 /home/username/waai/data/ 에 저장 ]

<img width="631" height="890" alt="image" src="https://github.com/user-attachments/assets/d023394a-98f1-47c9-bcce-7f694fd5a954" />


---------------------------------------------------

### 둘. 기획서 생성 기준

### 동작 로직

[ User : 사용자 프롬프트 입력 ]

↓

[ OpenWebUI : LLM 에게 사용자 프롬프트와 waai-backend 의  /api/plan/from-prompt API 호출 요청 ]

↓

[ Ollama : qwen2.5-refine-ko 모델이 사용자 프롬프트와 도구 호출 여부를 검토하여 도구 호출 ]

↓

[ Waai-backend : 날짜/키워드 등 기획서 생성 근거 조건 추출을 위해 Ollama llm 호출 ]

↓

[ Ollama : qwen2.5-refine-ko 모델이 날짜/키워드 등 기획서 생성 근거 조건 추출 후 MCP-bridge 에 근거 조건 결과 전송 ]

↓

[ MCP-bridge : MCP-filesystem 에게 파일 목록 수집 요청 ]

↓

[ MCP-filesystem : diary, ideas, web_research, works, bible 에서 조건에 맞는 파일 목록 수집하여 MCP-bridge 로 전송 ]

↓

[ MCP-bridge : 받은 파일목록과 요약 전용 프롬프트로 Ollama llm 호출 ]

↓

[ Ollama : qwen2.5-refine-ko 모델이 파일 목록을 읽어 종합 요약 생성 후 MCP-bridge 로 전송 ]

↓

[ MCP-bridge : 파일 목록과 요약 정보를 waai-backend 로 전송 ]

↓

[ Waai-backend : 파일 목록과 종합 요약, 그리고 기획서 생성 전용 프롬프트로 Ollama llm 호출 ]

↓

[ Ollama : qwen2-creative-ko 모델이 기획서 생성 후 waai-backend 로 전송 ] 

↓

[ Waai-backend : validate_date_front_matter 함수에서 메타 데이터 유효성 검증 후 누락 발견 시 llm 재호출 ] 

↓

[ Ollama : qwen2-creative-ko 모델이 기획서 재생성 후 응답 ]

↓

[ Waai-backend : 생성된 기획서를 /waai/data/outputs/ 에 저장 및 OpenWebUI 에 응답 전송 ] 

↓

[ OpenWebUI  : 생성된 기획서 일부를 UI 에 출력 ]

<img width="1000" height="851" alt="image" src="https://github.com/user-attachments/assets/3f08556a-b63f-4f73-9618-0d623d41c315" />
<img width="998" height="678" alt="image" src="https://github.com/user-attachments/assets/b8f3375f-7c29-4e76-ac84-5c3480957fde" />


---------------------------------------------------

### 셋. 웹 검색

### 동작 로직

[ User : UI 에서 “키워드1” 자료 조사해줘 프롬프트 입력 ] 

↓

[ OpenWebUI : 사용자 프롬프트와 /api/web_search/fetch API 호출을 LLM 에게 요청 ]

↓

[ Ollama : 사용자 프롬프트와 도구 호출 여부를 검토하여 waai-backend의 /api/web_search/fetch API 도구 호출 ]

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

[ Waai-backend : 데이터를 종합하여 /home/witness/memory/webresearch/ 에 raw_data 저장 ]

↓

[ OpenWebUI : 최종 응답을 기반으로 조사된 링크 목록을 사용자에게 출력 ]

↓

[ data-format-bot : data/reformat-md API 호출을 통해 데이터 정제 수행 ]

<img width="421" height="894" alt="image" src="https://github.com/user-attachments/assets/e89572d3-f9ce-4455-a137-6501df43a3ec" />


---------------------------------------------------

### 넷. 원고 합평

### 동작 로직

[ User : 입력된 원고를 프롬프트에 입력 ]

↓

[ OpenWebUI : LLM 에게 waai-backend 에 원고와 /api/critique API 호출 요청 ]

↓

[ Ollama : qweb2.5-refine-ko 모델이 도구 호출 여부를 검토하여 waai-backend 에 /api/critique API 호출 ]

↓

[ Waai-backend : 받은 원고를 txt 형태로 /waai/data/critique/objects/ 에 저장 ]

↓

[ Waai-backend : /waai/data/critique/criteria/ 에 있는 합평 기준 불러오기 ]

↓

[ Waai-backend : 원고 + 합평 기준 + 합평 전용 프롬프트로 Ollama llm 호출 ]

↓

[ Ollama : qweb2.5-refine-ko 모델이 합평 기준별로 원고를 읽고 점수 책정 및 결과 도출 후 waai-backend 로 결과 전송 ]

↓

[ Waai-backend : validate_date_front_matter 함수에서 메타 데이터 유효성 검증 후 누락 발견 시 llm 재호출 ] 

↓

[ Ollama : qweb2.5-refine-ko 모델이 합평 결과 재생성 후 응답 ]

↓

[ Waai-backend : 결과 물을 md 파일로 /waai/data/critique/results/ 에 저장 및 OpenWebUI 에 응답 전송 ]

↓

[ OpenWebUI : 완성된 합평 결과 일부를 UI 에 출력 ]

<img width="536" height="895" alt="image" src="https://github.com/user-attachments/assets/f833ac26-2201-45a6-9803-77f1dd2b1ccf" />


---------------------------------------------------

### 다섯. 브레인스토밍

### 동작 로직

[ 사용자 프롬프트 : qwen2-brainstorm-ko 모델을 활용해서 다양하게 아이디어 주고받기 ]

↓

[ 사용자 프롬프트 : /api/brainstorm/finalize 를 호출하여 지금까지의 채팅을 바탕으로 아이디어 생성 요청 ]

↓

[ Ollama : qwen2-brainstorm-ko 모델이 /api/brianstorm/finalize API 호출 ]

↓

[ Waai-backend : /api/brainstorm/finalize 함수 실행 전 가장 최근 채팅방의 모든메시지 조회하여 200개 소팅 ]

↓

[ Waai-backend : LLM 에게 아이디어 구체화 요청 ]

↓

[ Ollama : qwen2-brainstorm-ko 모델이 주어진 데이터를 기반으로 주요 아이디어 3개를 1000자 이내로 각각 구체화하고, 이를 /home/witness/memory/ideas/ 에 txt 파일로 각각 저장 ]

↓

[ Data-format-bot : 생성된 ideas.txt 파일들을 data/reformat-md 로 정제 ]

<img width="697" height="894" alt="image" src="https://github.com/user-attachments/assets/29bc10dd-2d12-413b-bf19-72dc3efd0a00" />

---------------------------------------------------

## 6. 사용 기술 (Tech Stack)

하나. Core Architecture
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
  - Modelfiles (목적별 모델 파라미터 최적화)
    - qwen2.5-refine-ko : 데이터 정제, 합평, 기획서 생성 전 프롬프트 조건 추출, 파일 종합 요약 등
    - qwen2-creative-ko : 기획서 생성
    - qwen2-brainstorm-ko : 브레인스토밍 전용 아이디어 확장

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
  - 점진적 고도화 수행 중 :
    [완료] V1(프로토타입; 기본 대화형 AI) -> V2(일기 기반 기획서 생성 AI) -> V3(데이터(ideas,bible,works,web_research) 확장, 기능 확장(웹검색, 합평, 브레인스토밍 기능), 목적별 모델 최적화, 모니터링 고도화)
    [계획] -> V4(VectorDB/RAG 시스템) -> V5(학습모델)


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

하나. LLM 서비스가 어떠한 방식으로 동작하는지 이해하게 됐습니다. 모두 구현하는 줄 알았지만 실질적으로는 LLM 은 이미 생성된 모델을 사용하고, API 를 통해 거대언어모델을 다양한 방식으로 호출하여 여러 결과물을 얻는 구조를 이해했습니다. 

둘. MCP와 LLM 이 어떻게 상호작용할 수 있는지 이해했습니다. LLM 은 '뇌'역할을, MCP 는 '손'과 '발'이 된다는 것이 어떠한 이유에서 표현된 것인지 알게 됐습니다. 
ChatGPT와 같은 범용 거대언어모델에서 '도구'라는 개념이 어떤식으로 구현되는지 (API 통해 백엔드 기능 호출), 사용자 프롬프트를 어떤 API로 전송해서 llm 에게 질의하여 어떤 시너지를 낼 수 있는지를 확인했습니다. 

셋. LLM 의 모델 파라미터의 조정과 목적별 모델을 분기하여 다른 형태의 결과물을 다양하게 얻을 수 있다는 것을 깨달았습니다. 규격에 맞춘 모델, 기획서 생성 모델, 창의력 모델 등등 같은 모델이어도 파라미터를 통해 모델의 출력을 조정할 수 있습니다.

마지막. 최종적으로 WAAI 라는 나만의 보조 작가 AI를 얻게 되었습니다. 앞으로 고도화할 길이 멀지만, 하나씩 차근차근 기능을 추가해가면서 강력한 보조 작가 AI 로서 성장시킬 것입니다.
