# src/features/action_card_generation/tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json

from .adapter import profile_to_agent1_like_json
from src.utils.errors import create_tool_error

from .agent import build_agent2_prompt, call_gemini_for_action_card
from src.features.data_analysis.tool import data_analysis_tool
from src.core.common_tools.rag_search_tool import rag_search_tool
from src.features.profile_management.tool import get_profile
from src.services.data_service import data_service 

class ActionCardGeneratorInput(BaseModel):
    user_query: str = Field(..., description="사용자의 원본 질문")
    profile: dict = Field(..., description="현재 상담 중인 가맹점의 전체 프로필")

def _format_action_card_result(agent2_json: dict) -> str:
    """Agent-2의 JSON 출력을 Streamlit에서 보여주기 좋은 Markdown 문자열로 변환합니다."""
    if not agent2_json or "recommendations" not in agent2_json or not agent2_json["recommendations"]:
        return "실행 카드를 생성하는 데 실패했거나, 추가 정보가 필요하여 생성을 보류했습니다. (오류 또는 tool_calls 확인 필요)"

    output = "### 💡 소상공인 맞춤 실행 카드 제안\n\n"
    for card in agent2_json.get("recommendations", []):
        output += f"#### **{card.get('title', '제목 없음')}**\n"
        output += f"- **🎯 타겟:** {card.get('what', '—')}\n"
        output += f"- **📢 채널:** {', '.join(card.get('where', [])) if isinstance(card.get('where'), list) else card.get('where', '—')}\n"
        output += f"- **📝 방법:** {', '.join(card.get('how', [])) if isinstance(card.get('how'), list) else card.get('how', '—')}\n"
        output += f"- **✍️ 카피 예시:** {' / '.join(card.get('copy', [])) if isinstance(card.get('copy'), list) else card.get('copy', '—')}\n"
        
        kpi = card.get('kpi', {})
        kpi_text = f"측정 지표: {kpi.get('target', '—')}"
        if kpi.get('range'):
            kpi_text += f", 목표 구간: {kpi['range'][0]} ~ {kpi['range'][1]}"
        output += f"- **📈 KPI:** {kpi_text}\n"
        
        output += f"- **🔍 근거:** {', '.join(card.get('evidence', [])) if isinstance(card.get('evidence'), list) else card.get('evidence', '—')}\n\n"
    
    return output

@tool(args_schema=ActionCardGeneratorInput)
def generate_action_card(user_query: str, profile: dict) -> str:
    """
    실행 카드 생성을 위한 전문가 에이전트(Agent2)를 루프(Loop) 방식으로 호출합니다.
    Agent2가 만족스러운 결과를 낼 때까지 필요한 정보를 대신 수집하여 제공합니다.
    """
    print("--- 🛠️ Tool: generate_action_card (Phase 2 수정 버전) 호출됨 ---")
    
    if not profile:
        return "오류: 프로필 정보가 없어 실행 카드를 생성할 수 없습니다."

    max_turns = 3  # 루프 최대 횟수를 3회로 조정하여 비용 및 시간 관리
    collected_data = []
    
    # Adapter: 프로필을 Agent2가 이해하는 형식으로 변환
    agent1_like_json = profile_to_agent1_like_json(profile, user_query)
    store_id = profile.get("profile_id")

    # 초기 RAG 검색을 수행
    rag_query = f"{profile.get('core_data', {}).get('basic_info', {}).get('industry_main', '')} 업종의 {user_query}"
    initial_rag_context = data_service.search_for_context(query=rag_query)

    for i in range(max_turns):
        print(f"--- [Agent2 Loop] Turn {i+1}/{max_turns} ---")

        prompt = build_agent2_prompt(agent1_like_json, initial_rag_context, collected_data)
        agent2_result = call_gemini_for_action_card(prompt)
        
        tool_calls = agent2_result.get("tool_calls")
        if not tool_calls:
            print("--- [Agent2 Loop] 완료. 최종 카드 생성. ---")
            return _format_action_card_result(agent2_result)
            
        print(f"--- [Agent2 Loop] Tool 호출 요청 감지: {tool_calls} ---")
        for call in tool_calls:
            tool_name = call.get("tool_name")
            query = call.get("query")
            result = ""
            
            try:
                # Phase 1에서 수정한 Tool들을 직접 호출
                if tool_name == "data_analyzer":
                    print(f"--- 🤵 비서: Agent2의 요청으로 데이터 분석 수행 -> '{query}' ---")
                    result = data_analysis_tool.invoke({"query": query, "store_id": store_id})
                elif tool_name == "rag_searcher":
                    print(f"--- 🤵 비서: Agent2의 요청으로 RAG 검색 수행 -> '{query}' ---")
                    result = data_service.search_for_context(query=query)
                else:
                    result = f"알 수 없는 도구 요청: {tool_name}"
            except Exception as e:
                result = create_tool_error(tool_name, e)
            
            collected_data.append((f"[Tool: {tool_name}] {query}", result))

    print("--- [Agent2 Loop] 최대 턴 도달. 마지막 생성 시도. ---")
    final_prompt = build_agent2_prompt(agent1_like_json, initial_rag_context, collected_data)
    final_result = call_gemini_for_action_card(final_prompt)
    return _format_action_card_result(final_result)