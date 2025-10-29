# src/core/planner_prompt.py

import json
from typing import Dict, Any
from src.services.data_service import data_service

from typing import Dict, Any
from src.services.data_service import data_service

# --- 1. 프롬프트 템플릿 상수 정의 ---

# Planner의 역할과 정체성을 정의하는 시스템 메시지입니다.
SYSTEM_MESSAGE = """
당신은 소상공인 전문 AI 컨설턴트의 '최고 전략 책임자(Planner)'입니다.
당신의 유일한 임무는 주어진 [상황 정보]를 바탕으로, 문제를 해결하기 위한 단계별 실행 계획을 JSON 형식으로 수립하는 것입니다.
당신은 직접 답변을 생성하지 않으며, 오직 실행 계획만을 출력합니다.
"""

# Planner가 계획을 수립할 때 반드시 따라야 할 핵심 규칙입니다.
PLANNING_RULES = """
**[계획 수립 4원칙 (매우 중요)]**
1.  **목표 지향:** 사용자의 최종 목표를 달성하기 위한 가장 효율적인 경로를 설계해야 합니다.
2.  **도구 전문성 활용:** 각 도구의 전문 분야를 정확히 이해하고, 문제에 가장 적합한 전문가(도구)에게 임무를 할당해야 합니다.
3.  **신중한 전문가 호출:** `action_card_generator`는 매우 유능하지만 비용이 높은 전문가입니다. 사용자가 명시적으로 "실행 카드", "n주 플랜", "종합 솔루션", "전략 제안" 등을 요구할 때만 호출하세요. 일반적인 분석이나 검색 요청에 남용해서는 안 됩니다.
4.  **출력 형식 준수:** 당신의 최종 출력물은 오직 JSON 배열이어야 합니다. 서론, 결론, 부연 설명 등은 절대 포함하지 마세요.
"""

# Planner가 생성해야 할 최종 JSON 출력의 스키마 예시입니다.
PLAN_JSON_SCHEMA = """
[
  {
    "tool_name": "사용할 도구의 이름",
    "tool_input": {
      "도구의 Pydantic 스키마에 맞는 인자들": "값"
    },
    "thought": "이 도구를 왜, 어떤 목적으로 사용하는지에 대한 나의 생각"
  }
]
"""

# --- 2. 프롬프트 빌더 함수 ---

def build_planner_prompt(state: Dict[str, Any], effective_tool_descriptions: str) -> str:
    """
    AgentState와 사용 가능한 도구 설명을 바탕으로 Planner LLM을 위한 최종 프롬프트를 생성합니다.

    Args:
        state: 현재 대화의 AgentState 딕셔너리.
        effective_tool_descriptions: 이번 턴에 사용 가능한 도구들의 설명 문자열.

    Returns:
        LLM에 전달될 전체 프롬프트 문자열.
    """
    # 1. 상태(State)에서 컨텍스트 정보 추출
    try:
        profile = state.get("current_profile", {})
        store_id = profile.get("profile_id")
        recent_messages = state.get('messages', [])[-5:]
        user_query = recent_messages[-1].content if recent_messages else "질문을 찾을 수 없음"
        conversation_history = "\n".join([f"- {msg.type}: {msg.content}" for msg in recent_messages])
    except (IndexError, KeyError) as e:
        print(f"경고: State에서 정보 추출 중 오류 발생 - {e}")
        store_id = None
        user_query = "질문을 찾을 수 없음"
        conversation_history = "대화 기록 없음"

    # 2. 현재 프로필에 대한 요약 정보 구성
    if store_id:
        available_info = data_service.get_summary_for_planner(store_id)
        available_info_str = "\n".join([f"- {k}: {v}" for k, v in available_info.items()])
    else:
        available_info_str = "현재 조회된 가맹점 프로필 정보가 없습니다."

    # 3. 프롬프트의 각 섹션을 조합
    context_section = f"""
**[상황 정보]**
1.  **최근 대화 기록:**
    {conversation_history}

2.  **사용자의 가장 최근 요청:**
    "{user_query}"

3.  **현재 프로필에 이미 확보된 주요 정보:**
    (아래 정보로 답변이 충분하다면, `data_analyzer`를 호출하지 마세요.)
    {available_info_str}
"""

    tools_section = f"""
**[이번 작업에서 사용 가능한 도구 목록]**
{effective_tool_descriptions}
"""

    output_instruction_section = f"""
---
위 모든 정보를 바탕으로, 사용자의 요청을 해결하기 위한 실행 계획을 아래 JSON 스키마를 준수하는 JSON 배열 형식으로만 생성하세요.
다른 어떤 설명도 추가하지 말고, 오직 JSON 객체만 출력해야 합니다.

**[JSON 스키마]**
```json
{PLAN_JSON_SCHEMA}
```
"""

    # 4. 모든 섹션을 결합하여 최종 프롬프트 완성
    final_prompt = (
        f"{SYSTEM_MESSAGE}\n"
        f"{context_section}\n"
        f"{tools_section}\n"
        f"{PLANNING_RULES}\n"
        f"{output_instruction_section}"
    )

    return final_prompt