# src/core/planner_prompt.py

import json
from typing import Dict, Any
from src.services.data_service import data_service

SYSTEM_MESSAGE = """
당신은 소상공인 전문 AI 컨설턴트의 '최고 전략 책임자(Planner)'입니다.
당신의 유일한 임무는 주어진 [상황 정보]를 바탕으로, 문제를 해결하기 위한 단계별 실행 계획을 수립하는 것입니다.
당신은 직접 답변을 생성하지 않으며, 오직 실행 계획만을 출력합니다.
"""

PLANNING_RULES = """
**[계획 수립 4원칙 (매우 중요)]**
1.  **목표 지향:** 사용자의 최종 목표를 달성하기 위한 가장 효율적인 경로를 설계해야 합니다.
2.  **도구 전문성 활용:** 각 도구의 전문 분야를 정확히 이해하고, 문제에 가장 적합한 전문가(도구)에게 임무를 할당해야 합니다.
3.  **신중한 전문가 호출:** `action_card_generator`는 매우 유능하지만 비용이 높은 전문가입니다. 사용자가 명시적으로 "실행 카드", "n주 플랜", "종합 솔루션", "전략 제안" 등을 요구할 때만 호출하세요. 일반적인 분석이나 검색 요청에 남용해서는 안 됩니다.
4.  **출력 형식 준수:** 당신의 최종 출력물은 오직 번호가 매겨진 실행 계획 목록이어야 합니다. 서론, 결론, 부연 설명 등은 절대 포함하지 마세요.
"""

# 도구 설명은 한 곳에서 관리하여, 새로운 도구가 추가되거나 변경될 때 여기만 수정하면 됩니다.
TOOL_DESCRIPTIONS = {
    "data_analyzer": "프로필에 없는 상세 수치 데이터(예: 시간대별, 메뉴별, 고객 세그먼트별)를 원본 CSV 파일에서 직접 심층 분석하는 '데이터 과학자'입니다.",
    "rag_searcher": "내부 지식 베이스에서 소상공인 관련 마케팅 전략, 실행 가이드, 성공 사례 등 검증된 정보를 검색하는 '사내 자료 분석가'입니다.",
    "action_card_generator": "수집된 모든 정보를 종합하여 구체적인 실행 방안이 담긴 '실행 카드'나 'n주 플랜'을 생성하는 '수석 컨설턴트'입니다. 가장 마지막에 호출되는 경우가 많습니다.",
    "marketing_idea_generator": "분석된 데이터나 트렌드를 바탕으로 창의적인 마케팅 아이디어를 브레인스토밍하는 '마케팅 전문가'입니다.",
    "update_profile": "사용자와의 대화에서 얻은 새로운 정보를 가맹점 프로필 데이터베이스에 업데이트하는 '비서'입니다. (예: 사장님의 사업 목표, 연령대 등)",
    "video_recommender": "사용자의 질문과 프로필에 맞춰 관련된 학습 영상을 검색하고, 각 영상의 내용을 요약하여 맞춤 추천하는 '미디어 큐레이터'입니다. 텍스트 설명 외에 시청각 자료가 필요할 때 사용하세요.",
    "policy_recommender": "사용자의 프로필(업종, 지역 등)을 바탕으로 가장 적합한 정부/지자체 지원사업을 검색하고 맞춤 추천하는 '정책 전문가'입니다. '지원금', '보조금', '정책' 관련 질문에 사용하세요.",
}

# ==============================================================================
# 프롬프트 빌더 함수 (Prompt Builder Function)
# ==============================================================================

def build_planner_prompt(state: Dict[str, Any]) -> str:
    """
    LangGraph의 AgentState를 입력받아 Planner LLM을 위한 최종 프롬프트를 생성합니다.

    Args:
        state (Dict[str, Any]): 현재 대화의 AgentState 딕셔너리.

    Returns:
        str: LLM에 전달될 전체 프롬프트 문자열.
    """
    
    # 1. 상태에서 필요한 정보 추출
    try:
        profile = state.get("current_profile", {})
        # messages 리스트의 마지막 메시지가 사용자 질문이라고 가정
        store_id = profile.get("profile_id")
        user_query = state['messages'][-1].content
    except (IndexError, KeyError) as e:
        # 상태 구조가 예상과 다를 경우를 대비한 방어 코드
        print(f"경고: State에서 정보를 추출하는 중 오류 발생 - {e}")
        profile = {}
        store_id = None
        user_query = "정보를 찾을 수 없음"

    # 2. 컨텍스트 정보 구성
    if store_id:
        available_info = data_service.get_summary_for_planner(store_id)
        available_info_str = "\n".join([f"- {k}: {v}" for k, v in available_info.items()])
    else:
        available_info_str = "현재 조회된 가맹점 프로필 정보가 없습니다."

    context_section = f"""
**[상황 정보]**
1.  **사용자의 최근 요청:**
    "{user_query}"
2.  **현재 프로필에 이미 확보된 주요 정보:**
    (아래 정보로 답변이 충분하다면, `data_analyzer`를 호출하지 마세요.)
{available_info_str}
"""

    # 3. 사용 가능한 도구 설명 구성
    tool_desc_str = "\n".join(
        [f"- `{name}`: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()]
    )
    tools_section = f"""
**[사용 가능한 전문가(도구) 목록]**
{tool_desc_str}
"""

    # 4. 최종 프롬프트 조합
    final_prompt = f"""{SYSTEM_MESSAGE}
{context_section}
{tools_section}
{PLANNING_RULES}

---
위 모든 정보를 바탕으로, 사용자의 요청을 해결하기 위한 실행 계획을 `[Tool: 도구이름] 구체적인 지시사항` 형식으로만 생성하세요.
"""

    return final_prompt