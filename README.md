# AGI Framework

**Multi-LLM Agent 기반 사용자 상호작용 및 능동학습 AGI 프레임워크**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

## 개요

본 프레임워크는 다중 LLM 에이전트 기반의 AGI(Artificial General Intelligence) 시스템으로, 각 에이전트가 독립적으로 작동하면서 상호 연계하여 복잡한 작업을 수행합니다.

### 주요 특징

- **다중 에이전트 아키텍처**: 5개의 전문 에이전트가 협력하여 작업 수행
- **GNN 기반 지식 그래프**: 그래프 신경망을 활용한 지식 정렬 및 검색
- **LoRA 기반 태스크 어댑터**: 경량 파인튜닝을 통한 태스크별 최적화
- **자기 평가 강화학습**: 응답 품질 평가 및 지속적 개선
- **비동기 메시지 통신**: 확장 가능한 에이전트 간 통신 구조

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Multi LLM Agent 기반 AGI 프레임워크                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐       ┌────────────────┐       ┌────────────────┐   │
│  │ 지식 큐레이션   │       │   모더레이터     │       │   컨텍스트      │   │
│  │    에이전트     │◄─────►│    에이전트     │◄─────►│    에이전트     │   │
│  ├────────────────┤       ├────────────────┤       ├────────────────┤   │
│  │• GNN 임베딩     │       │• 의도 분류      │      │• 히스토리       │   │
│  │• 그래프 정렬    │       │• 계획 수립      │       │• 상황그래프     │   │
│  │• 지식 검색      │       │• 작업 조율      │       │                │   │
│  └────────────────┘       └────────────────┘       └────────────────┘   │
│           │                        │                        │           │
│           │                        ▼                        │           │
│           │             ┌────────────────────┐              │           │
│           └────────────►│   오케스트레이터     │◄─────────────┘          │
│                         │    (메시지 버스)    │                          │
│                         └──────────┬─────────┘                          │
│                                    │                                    │
│           ┌────────────────────────┴────────────────────────┐           │
│           ▼                                                 ▼           │
│  ┌────────────────┐                                 ┌────────────────┐  │
│  │  추론 에이전트  │◄───────────────────────────────►│  학습 에이전트   │  │
│  ├────────────────┤                                 ├────────────────┤  │
│  │• LoRA 어댑터    │                                 │• 자기 평가     │  │
│  │• 태스크 모듈    │                                 │• 리워드 모델    │  │
│  │• 불확실도 추정  │                                 │• PPO 학습      │   │
│  └────────────────┘                                 └────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 에이전트 구성

### 1. 모더레이터 에이전트 (Moderator Agent)
- 사용자 의도 분류 (knowledge_query, task_execution, learning_request 등)
- 과업 분해 및 실행 계획 수립
- 에이전트 간 작업 조율 및 실행 흐름 관리

### 2. 지식 큐레이션 에이전트 (Knowledge Curation Agent)
- Multi-modal Knowledge Graph 관리
- GNN(GCN, GAT, GraphSAGE) 기반 노드 임베딩
- 쿼리-그래프 정렬 및 컨텍스트 기반 지식 검색

### 3. 컨텍스트 에이전트 (Context Agent)
- SQLite 기반 대화 히스토리 관리
- 상황 정보 그래프 구성 (시간적 감쇠 적용)
- 장기 문맥 검색 및 회수

### 4. 추론 에이전트 (Reasoning Agent)
- LoRA Adapter 기반 태스크별 추론
- 교육자료 생성, 소견 생성 등 다양한 태스크 모듈
- 불확실도 추정

### 5. 학습 에이전트 (Learning Agent)
- 자기 평가 기반 응답 품질 평가
- 선호도 데이터 수집 및 리워드 모델 학습
- PPO 기반 강화학습 루프

## 설치

```bash
# 저장소 클론
git clone https://github.com/cuknlp/agi-framework.git
cd agi-framework

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# 개발 모드 설치
pip install -e .
```

## 빠른 시작

### 기본 사용법

```python
import asyncio
from src.core.orchestrator import AGIFramework

async def main():
    # 시스템 생성 및 시작
    orchestrator = AGIFramework.create_default_system()
    await orchestrator.start()
    
    # 쿼리 처리
    result = await orchestrator.process_request(
        query="딥러닝이란 무엇인가요?",
        session_id="my_session"
    )
    
    print(result)
    
    # 시스템 종료
    await orchestrator.stop()

asyncio.run(main())
```

### 개별 에이전트 사용

```python
from src.core.message_bus import MessageBus
from src.agents.knowledge_curation_agent import KnowledgeCurationAgent

# 메시지 버스 생성
bus = MessageBus()

# 지식 에이전트 생성
knowledge_agent = KnowledgeCurationAgent(message_bus=bus)
await knowledge_agent.on_start()

# 지식 추가
node_id = knowledge_agent.add_knowledge(
    label="Large Language Model",
    node_type="concept",
    properties={"acronym": "LLM"}
)

# 지식 검색
results = knowledge_agent.search_knowledge("언어 모델", top_k=5)
```

## 예제 실행

```bash
# 기본 사용 예제
python examples/basic_usage.py

# 시나리오 기반 데모
python examples/scenario_demo.py

# 테스트 실행
pytest tests/ -v
```

## 프로젝트 구조

```
agi-framework/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml           # 설정 파일
├── src/
│   ├── __init__.py
│   ├── agents/               # 에이전트 모듈
│   │   ├── base_agent.py
│   │   ├── moderator_agent.py
│   │   ├── knowledge_curation_agent.py
│   │   ├── context_agent.py
│   │   ├── reasoning_agent.py
│   │   └── learning_agent.py
│   ├── core/                 # 핵심 모듈
│   │   ├── orchestrator.py
│   │   └── message_bus.py
│   ├── models/               # 모델 모듈
│   │   ├── lora_adapter.py
│   │   └── reward_model.py
│   ├── knowledge/            # 지식 그래프 모듈
│   │   ├── knowledge_graph.py
│   │   └── graph_embeddings.py
│   └── utils/                # 유틸리티
│       └── logging_utils.py
├── tests/                    # 테스트
│   ├── test_agents.py
│   └── test_integration.py
└── examples/                 # 예제
    ├── basic_usage.py
    └── scenario_demo.py
```

## 설정

`config/config.yaml`에서 각 에이전트와 시스템 설정을 변경할 수 있습니다:

```yaml
# 모더레이터 설정
moderator:
  max_planning_steps: 10
  parallel_execution: true

# 지식 큐레이션 설정
knowledge_curation:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  gnn_model: "GCN"

# 추론 에이전트 설정
reasoning:
  base_model: "meta-llama/Llama-2-7b-hf"
  lora_adapters:
    enabled: true
    rank: 16
```
---

*본 프로젝트는 AGI 시스템의 다중 에이전트 통합 운영 가능성을 실제 시나리오 기반으로 검증하기 위한 토이 프로젝트입니다.*

