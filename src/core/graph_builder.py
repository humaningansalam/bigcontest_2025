# src/core/graph_builder.py

import json
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from src.core.state import AgentState
import streamlit as st
import os
from langgraph.checkpoint.memory import MemorySaver

from src.core.common_tools.marketing_idea_tool import marketing_idea_generator_tool
from src.core.common_tools.rag_search_tool import rag_search_tool
from src.features.profile_management.tool import get_profile, update_profile
from src.features.data_analysis.tool import data_analysis_tool
from src.features.action_card_generation.tool import generate_action_card
from src.features.video_recommendation.tool import video_recommender_tool
from src.features.policy_recommendation.tool import policy_recommender_tool
from src.services.data_service import data_service
from src.core.intent_classifier import classify_intent
from .planner_prompt import build_planner_prompt
from src.utils.errors import create_tool_error
from src.config import PRIMARY_MODEL_NAME


    
load_dotenv()

google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0)

tools = {
    "data_analyzer": data_analysis_tool,
    "action_card_generator": generate_action_card,
    "marketing_idea_generator": marketing_idea_generator_tool,
    "get_profile": get_profile,
    "update_profile": update_profile,
    "rag_searcher": rag_search_tool,
    "video_recommender": video_recommender_tool,
    "policy_recommender": policy_recommender_tool,
}

#  단순 응답을 위한 노드 
def simple_responder_node(state: AgentState):
    """
    'greeting', 'unknown' 등 간단한 의도에 대해 LLM 호출 없이 즉시 답변하는 경량 노드.
    """
    print("--- 👋 Simple Responder 활동 시작 ---")
    user_query = state['messages'][-1].content
    intent = classify_intent(user_query) # 의도를 다시 한번 확인
    
    if intent == 'greeting':
        response_content = "안녕하세요, 사장님! 무엇을 도와드릴까요?"
    else: # 'unknown' 또는 기타 처리 불가능한 의도
        response_content = "죄송합니다. 질문을 명확하게 이해하지 못했습니다. '우리 가게 재방문율 분석해줘'와 같이 구체적으로 질문해주시겠어요?"
        
    # 최종 상태에 메시지를 추가하여 반환
    return {"messages": [AIMessage(content=response_content)]}

# --- Router Node ---
def router_node(state: AgentState) -> dict:
    user_query = state['messages'][-1].content
    intent = classify_intent(user_query)
    print(f"--- 🚦 분석된 의도: {intent} ---")
    
    # "bigcon_request" 의도일 경우, Planner를 건너뛰고 Executor에게 직접 지시
    if intent == "bigcon_request":
        print("--- [Router] Agent2 직접 호출 결정 ---")
        plan = [f"[Tool: action_card_generator] {user_query}"]
        return {
            "next_node": "executor", 
            "plan": plan,
            "past_steps": []
        }
    
    # 동영상 추천
    elif intent == "video_recommendation":
        print("--- [Router] 동영상 추천 도구 직접 호출 결정 ---")
        plan = [f"[Tool: video_recommender] {user_query}"]
        return {"next_node": "executor", "plan": plan, "past_steps": []}

    # 정책 추천
    elif intent == "policy_recommendation":
        print("--- [Router] 지원사업 추천 도구 직접 호출 결정 ---")
        plan = [f"[Tool: policy_recommender] {user_query}"]
        return {"next_node": "executor", "plan": plan, "past_steps": []}

    # 복합적인 분석이 필요할 때만 Planner 호출
    elif intent in ["data_analysis", "marketing_idea", "rag_search"]:
        print("--- [Router] Planner 호출 결정 ---")
        return {"next_node": "planner"}
        
    # 프로필 조회
    elif intent == "profile_query":
        print("--- [Router] Synthesizer 직접 호출 결정 (프로필 기반 답변) ---")
        return {"next_node": "synthesizer", "plan": []}
        
    # 인사나 알 수 없는 질문은 새로운 'simple_responder'가 처리하도록 합니다.
    else: # greeting, unknown
        print("--- [Router] 단순 응답 노드(Simple Responder) 직접 호출 결정 ---")
        return {"next_node": "simple_responder"}

# --- Planner Node ---
def planner_node(state: AgentState):
    print("--- 🤔 Planner 활동 시작 (강화 버전) ---")
    
    prompt = build_planner_prompt(state)
    
    response = llm.invoke(prompt)
    plan = [step.strip() for step in response.content.split('\n') if step.strip() and '[Tool:' in step]
    
    print(f"--- 📝 수립된 계획 ---\n" + "\n".join(plan))
    return {"plan": plan, "past_steps": []}

# --- Executor Node ---
def executor_node(state: AgentState):
    print("--- ⚙️ Executor 활동 시작 (Phase 2 수정 버전) ---")
    
    if not state.get("plan"):
        return {}

    step = state["plan"][0]
    
    try:
        tool_name = step.split("[Tool: ")[1].split("]")[0]
        query = step.split("]")[1].strip()
    except IndexError:
        error_result = "오류: 계획의 형식이 잘못되었습니다. [Tool: 도구이름] 형식으로 작성되어야 합니다."
        return {"plan": state["plan"][1:], "past_steps": state.get("past_steps", []) + [(step, error_result)]}

    print(f"--- [실행] 도구: {tool_name} // 지시사항: {query} ---")

    past_step_result = ""
    if tool_name in tools:
        tool = tools[tool_name]
        try:
            # [최종 수정] 각 도구의 Pydantic 모델에 맞는 정확한 인자 딕셔너리 생성
            invoke_args = {}
            user_input_query = state["messages"][-1].content # 사용자의 원본 질문

            if tool_name == "data_analyzer":
                invoke_args = {"query": query, "store_id": state.get("current_profile", {}).get("profile_id")}
            
            elif tool_name == "action_card_generator":
                # 이 도구는 Planner의 지시사항(query)이 아닌 사용자의 원본 질문(user_input_query)을 받도록 설계했었음
                invoke_args = {"user_query": user_input_query, "profile": state.get("current_profile")}

            elif tool_name == "video_recommender":
                # 이 도구도 사용자의 원본 질문을 기반으로 개인화 추천
                invoke_args = {"user_query": user_input_query, "profile": state.get("current_profile")}

            elif tool_name == "marketing_idea_generator":
                # 이 도구는 Planner가 정제한 지시사항(query)을 주제(topic)로 받음
                invoke_args = {"topic": query}

            elif tool_name == "policy_recommender":
                invoke_args = {"user_query": state["messages"][-1].content, "profile": state.get("current_profile")}

            else: # web_searcher, rag_search_tool 등 'query' 인자만 받는 기본 도구들
                invoke_args = {"query": query}
            
            result = tool.invoke(invoke_args)
            past_step_result = str(result)

            # Pandas Agent 또는 Action Card Generator가 최종 답변을 생성하면 바로 종료
            if (tool_name == "data_analyzer" and "Final Answer:" in past_step_result) or \
               (tool_name == "action_card_generator") or \
               (tool_name == "video_recommender") or \
               (tool_name == "policy_recommender"):
                
                final_content = past_step_result
                if "Final Answer:" in past_step_result:
                    final_content = past_step_result.split("Final Answer:", 1)[1].strip()

                return {
                    "plan": [],
                    "past_steps": state.get("past_steps", []) + [(step, past_step_result)],
                    "messages": state.get("messages", []) + [AIMessage(content=final_content)],
                    "is_final_answer": True
                }

        except Exception as e:
            print(f"--- 🚨 EXECUTOR가 도구 호출 중 예외 발생: {e} ---")
            past_step_result = create_tool_error(tool_name, e)
    else:
        past_step_result = f"오류: '{tool_name}'은(는) 알 수 없는 도구입니다."

    return {"plan": state["plan"][1:], "past_steps": state.get("past_steps", []) + [(step, past_step_result)]}

# --- Synthesizer Node (RAG-Aware) ---
def synthesizer_node(state: AgentState):
    print("--- ✍️ Synthesizer 최종 답변 작성 (논리 강화 버전) ---")
    
    user_query = state['messages'][-1].content
    profile_json_str = json.dumps(state.get('current_profile'), ensure_ascii=False, indent=2)
    
    # Tool을 사용한 경우
    if state.get("past_steps"):
        evidence = "\n\n".join(
            [f"**실행 내용:** {step}\n**결과:**\n{result}" for step, result in state.get("past_steps")]
        )
        base_context = f"**[수집된 근거 자료]**\n{evidence}\n\n**[참고: 가맹점 프로필]**\n{profile_json_str}"
    
    # Tool을 사용하지 않은 경우 
    else:
        prompt_check = f"""사용자의 질문이 주어진 프로필 정보만으로 답변 가능한지 'yes' 또는 'no'로만 답해주세요.
        
        [프로필 정보]
        {data_service.get_summary_for_planner(state['current_profile']['profile_id'])}
        
        [사용자 질문]
        "{user_query}"
        
        답변 (yes/no):"""
        
        is_profile_sufficient = llm.invoke(prompt_check).content.strip().lower()

        if 'yes' in is_profile_sufficient:
            print("--- [Synthesizer] 프로필 정보만으로 답변 가능. RAG 검색 생략. ---")
            base_context = f"**[가맹점 프로필 정보]**\n{profile_json_str}"
        else:
            print("--- [Synthesizer] 외부 정보 필요. RAG 검색 수행. ---")
            rag_context = data_service.search_for_context(query=user_query)
            base_context = f"**[참고 자료]**\n{rag_context}\n\n**[가맹점 프로필 정보]**\n{profile_json_str}"

    prompt = f"""당신은 전문 컨설턴트입니다. 아래 [사용자 질문]에 대해, 주어진 [핵심 근거]만을 바탕으로 친절하고 명확하게 최종 답변을 작성해주세요.
만약 [참고 자료]가 있다면, 해당 내용을 인용하여 답변의 신뢰도를 높여주세요.

**[사용자 질문]**
{user_query}

**[핵심 근거]**
{base_context}

**[최종 답변]**
"""
    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}




# --- 최종 그래프 구성 ---
workflow = StateGraph(AgentState)

# 1. 모든 노드를 그래프에 추가합니다.
workflow.add_node("router", router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("simple_responder", simple_responder_node)

# 2. 그래프의 시작점을 'router'로 설정합니다.
workflow.set_entry_point("router")

# 3. 각 노드를 순서대로 연결합니다.

# 'router'의 결정에 따라 'planner' 또는 'synthesizer'로 분기합니다.
workflow.add_conditional_edges(
    "router",
    lambda state: state.get("next_node"),
    {
        "planner": "planner",
        "executor": "executor", 
        "synthesizer": "synthesizer",
        "simple_responder": "simple_responder"
    }
)

# 'planner'는 항상 'executor'로 연결됩니다.
workflow.add_edge("planner", "executor")

# 이 함수가 executor 노드 다음에 어디로 갈지 결정합니다.
def after_executor_logic(state: AgentState):
    # 1순위: 특별 전문가(data_analyzer, action_card_generator)가 최종 답변을 만들었는가?
    if state.get("is_final_answer"):
        # 그렇다면 즉시 종료(END)하라는 신호를 보냅니다.
        return "end" 
    
    # 2순위: 남은 계획이 있는가?
    elif state.get("plan"):
        # 그렇다면 executor를 다시 실행하라는 신호를 보냅니다.
        return "continue"
        
    # 3순위: 모든 계획이 끝났고, 최종 답변이 생성되지 않았는가?
    else:
        # 그렇다면 synthesizer를 실행하라는 신호를 보냅니다.
        return "synthesize"

# 위 함수가 반환하는 신호('end', 'continue', 'synthesize')에 따라
# 다음에 실행할 노드를 매핑합니다.
workflow.add_conditional_edges(
    "executor",
    after_executor_logic,
    {
        "continue": "executor",
        "synthesize": "synthesizer",
        "end": END 
    }
)

# 'synthesizer'는 항상 마지막이며, 그래프를 종료(END)합니다.
workflow.add_edge("synthesizer", END)

# 4. 그래프를 컴파일합니다.
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

