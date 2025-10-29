# src/core/graph_builder.py

import json
import os
from typing import Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.config import PRIMARY_MODEL_NAME
from src.core.intent_classifier import classify_intent
from src.core.state import AgentState
from src.core.tool_registry import tool_registry
from src.utils.errors import create_tool_error
from .planner_prompt import build_planner_prompt

# --- 1. 전역 설정 및 LLM 초기화 ---
load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# 프로젝트의 핵심 LLM을 정의
llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0)

# 등록된 모든 도구의 설명 정보를 가져오고 Planner가 계획 수립 시 참고
TOOL_DESCRIPTIONS = tool_registry.get_all_descriptions()

# --- 2. 그래프 노드(Graph Nodes) 정의 ---

def simple_responder_node(state: AgentState) -> dict:
    """'greeting', 'unknown' 등 간단한 의도에 대해 즉시 답변하는 경량 노드입니다."""
    print("--- 👋 Simple Responder 활동 시작 ---")
    user_query = state['messages'][-1].content
    intent = classify_intent(user_query)

    if intent == 'greeting':
        response_content = "안녕하세요, 사장님! 무엇을 도와드릴까요?"
    else:
        response_content = "죄송합니다. 질문을 명확하게 이해하지 못했습니다. '우리 가게 재방문율 분석해줘'와 같이 구체적으로 질문해주시겠어요?"

    return {
        "messages": [AIMessage(content=response_content)],
        "final_output": response_content
    }


def router_node(state: AgentState) -> dict:
    """사용자 질문의 의도를 분석하여 워크플로우를 적절한 다음 노드로 분기합니다."""
    print("--- 🚦 Router 활동 시작 ---")
    user_query = state['messages'][-1].content
    intent = classify_intent(user_query)
    print(f"--- 분석된 의도: {intent} ---")

    # 시나리오 1: 복잡한 분석이 필요하여 Planner에게 계획 수립을 요청
    if intent in ["data_analysis", "marketing_idea", "general_rag_search"]:
        print("--- [Router] Planner 호출 결정 ---")
        allowed_tools = ["rag_searcher"] if intent == "general_rag_search" else None
        return {"next_node": "planner", "allowed_tools": allowed_tools}

    # 시나리오 2: 단일 도구로 해결 가능하여 Planner를 건너뛰고 바로 Executor 호출
    elif intent in ["bigcon_request", "video_recommendation", "policy_recommendation"]:
        print(f"--- [Router] 단일 도구({intent}) 직접 호출 결정 ---")
        tool_map = {
            "bigcon_request": "action_card_generator",
            "video_recommendation": "video_recommender",
            "policy_recommendation": "policy_recommender"
        }
        tool_name = tool_map[intent]
        plan = [{
            "tool_name": tool_name,
            "tool_input": {"user_query": user_query},
            "thought": f"사용자 의도 '{intent}'에 따라 {tool_name} 도구를 직접 호출합니다."
        }]
        return {"next_node": "executor", "plan": plan, "past_steps": []}

    # 시나리오 3: 도구 사용 없이 프로필 정보만으로 답변 가능
    elif intent == "profile_query":
        print("--- [Router] Synthesizer 직접 호출 결정 (프로필 기반 답변) ---")
        return {"next_node": "synthesizer", "plan": []}

    # 시나리오 4: 인사 등 간단한 응답
    else:
        print("--- [Router] Simple Responder 호출 결정 ---")
        return {"next_node": "simple_responder"}


def planner_node(state: AgentState) -> dict:
    """사용자 요청과 현재 상태를 바탕으로 LLM을 사용하여 단계별 실행 계획을 수립합니다."""
    print("--- 🤔 Planner 활동 시작 ---")
    allowed_list = state.get("allowed_tools")
    effective_tools = {key: TOOL_DESCRIPTIONS[key] for key in allowed_list if key in TOOL_DESCRIPTIONS} if allowed_list else TOOL_DESCRIPTIONS
    effective_tool_descriptions_str = "\n".join([f"- `{name}`: {desc}" for name, desc in effective_tools.items()])

    prompt = build_planner_prompt(state, effective_tool_descriptions_str)
    planner_chain = llm | JsonOutputParser()
    plan_json = planner_chain.invoke(prompt)

    print(f"--- 📝 수립된 계획 ---\n" + json.dumps(plan_json, indent=2, ensure_ascii=False))
    return {"plan": plan_json, "past_steps": []}


def executor_node(state: AgentState) -> dict:
    """계획(plan)의 첫 번째 단계를 실행하고, 그 결과를 상태(state)에 추가합니다."""
    print("--- ⚙️ Executor 활동 시작 ---")
    plan = state.get("plan", [])
    if not plan:
        return {}

    step = plan[0]
    tool_name = step.get("tool_name")
    invoke_args = step.get("tool_input", {}).copy()

    print(f"--- [실행] 도구: {tool_name} // 인자: {invoke_args} ---")
    print(f"--- [사고 과정] {step.get('thought', 'N/A')} ---")

    try:
        tool = tool_registry.get_tool(tool_name)

        # 도구가 필요로 하는 컨텍스트(profile, user_query 등)를 동적으로 주입
        tool_meta = tool_registry.get_tool_metadata(tool_name)
        current_profile = state.get("current_profile")
        if tool_meta.get("needs_profile") and current_profile:
            invoke_args["profile"] = current_profile
        if tool_meta.get("needs_user_query"):
            invoke_args["user_query"] = state["messages"][-1].content
        if tool_meta.get("needs_store_id") and current_profile:
            invoke_args["store_id"] = current_profile.get("profile_id")

        tool_output_dict = tool.invoke(invoke_args)
        tool_output = ToolOutput.model_validate(tool_output_dict)

        past_steps = state.get("past_steps", []) + [(json.dumps(step, ensure_ascii=False), tool_output.content)]
        sources = state.get("sources", []) + tool_output.sources
        updated_state = {"past_steps": past_steps, "sources": sources}

        # 도구가 '최종 답변'을 생성했는지 여부에 따라 상태를 업데이트
        if tool_output.is_final_answer:
            updated_state.update({
                "is_final_answer": True,
                "final_output": tool_output.content,
                "messages": state["messages"] + [AIMessage(content=tool_output.content)],
                "plan": []  # 계획 종료
            })
        else:
            updated_state["plan"] = plan[1:]  # 다음 계획으로 이동

        return updated_state

    except Exception as e:
        print(f"--- 🚨 EXECUTOR 예외 발생: {e} ---")
        error_message = create_tool_error(tool_name, e)
        past_steps = state.get("past_steps", []) + [(json.dumps(step, ensure_ascii=False), error_message)]
        return {"plan": plan[1:], "past_steps": past_steps}


def synthesizer_node(state: AgentState) -> dict:
    """지금까지 수집된 모든 정보(past_steps)를 종합하여 최종 답변을 생성합니다."""
    print("--- ✍️ Synthesizer 최종 답변 작성 ---")
    user_query = state['messages'][-1].content
    conversation_history = "\n".join([f"- {msg.type}: {msg.content}" for msg in state.get('messages', [])[:-1]])
    profile_json_str = json.dumps(state.get('current_profile'), ensure_ascii=False, indent=2)

    if state.get("past_steps"):
        evidence = "\n\n".join([f"**실행 내용:** {step}\n**결과:**\n{str(result)[:1000]}..." for step, result in state.get("past_steps")])
        base_context = f"**[수집된 근거 자료]**\n{evidence}\n\n**[참고: 가맹점 프로필]**\n{profile_json_str}"
    else:
        base_context = f"**[가맹점 프로필 정보]**\n{profile_json_str}"

    prompt = f"""당신은 전문 컨설턴트입니다. 아래 [사용자 질문]에 대해, 주어진 [핵심 근거]와 [이전 대화 내용]을 종합적으로 고려하여 친절하고 명확하게 최종 답변을 작성해주세요.
만약 [핵심 근거]에 [출처]나 [참고 자료]가 포함되어 있다면, 해당 내용을 인용하여 답변의 신뢰도를 높여주세요.

**[이전 대화 내용]**
{conversation_history}

**[사용자 질문]**
{user_query}

**[핵심 근거]**
{base_context}

**[최종 답변]**
"""
    response = llm.invoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "final_output": response.content
    }


def after_executor_logic(state: AgentState) -> str:
    """Executor 노드 실행 후, 다음으로 이동할 경로를 결정하는 조건부 로직"""
    if state.get("is_final_answer"):
        # 도구가 최종 답변을 생성했으므로 워크플로우를 즉시 종료
        return "end"
    elif state.get("plan"):
        # 아직 실행할 계획이 남아있으므로 Executor를 다시 실행
        return "continue"
    else:
        # 모든 계획이 끝났으므로, 수집된 정보를 종합하여 최종 답변을 생성
        return "synthesize"


# --- 3. 그래프(Graph) 구성 및 컴파일 ---

workflow = StateGraph(AgentState)

# 그래프에 각 노드를 추가
workflow.add_node("router", router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("simple_responder", simple_responder_node)

# 워크플로우의 시작점을 'router'로 설정
workflow.set_entry_point("router")

# 노드 간의 연결(엣지)을 정의
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
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    after_executor_logic,
    {
        "continue": "executor",      # 계속 실행
        "synthesize": "synthesizer",  # 종합
        "end": END                   # 종료
    }
)
workflow.add_edge("synthesizer", END)
workflow.add_edge("simple_responder", END) # simple_responder는 항상 마지막

# 대화 기록을 관리하기 위한 메모리 세이버를 설정
memory = MemorySaver()

# 최종적으로 그래프를 컴파일하여 실행 가능한 객체를 생성
graph = workflow.compile(checkpointer=memory)