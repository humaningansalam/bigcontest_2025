# src/core/state.py

from typing import List, TypedDict, Annotated, Dict, Any, NotRequired 
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Plan-and-Execute 모델을 위한 새로운 상태 정의.
    NotRequired를 사용하여 선택적 필드를 명시적으로 관리합니다.
    """
    # 기존 대화 기록 
    messages: Annotated[list, add_messages]
    
    # 현재 사용자 프로필
    current_profile: NotRequired[Dict[str, Any]]

    # Planner가 생성한 앞으로 해야 할 일들의 목록
    plan: NotRequired[List[Dict[str, Any]]]
    
    # Executor가 실행한 단계와 그 결과 (수집된 근거 자료)
    past_steps: NotRequired[List[tuple]]
    
    # 다음으로 실행할 노드의 이름 (조건부 엣지에서 사용)
    next_node: NotRequired[str]

    # 최종 답변 여부
    is_final_answer: NotRequired[bool]

    # rag 검색 결과
    sources: NotRequired[List[Dict[str, Any]]]

    # Planner가 이 턴에서 사용할 수 있도록 허용된 도구 목록
    allowed_tools: NotRequired[List[str]]

    # 최종 답변
    final_output: NotRequired[str]