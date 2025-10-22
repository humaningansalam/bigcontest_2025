# src/core/state.py

from typing import List, TypedDict, Annotated, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Plan-and-Execute 모델을 위한 새로운 상태 정의
    """
    # 기존 대화 기록
    messages: Annotated[list, add_messages]
    
    # 현재 사용자 프로필
    current_profile: Dict[str, Any] | None

    # Planner가 생성한 앞으로 해야 할 일들의 목록
    plan: List[str]
    
    # Executor가 실행한 단계와 그 결과 (수집된 근거 자료)
    past_steps: List[tuple]
    
    # 다음으로 실행할 노드의 이름 (조건부 엣지에서 사용)
    next_node: str

    # 최종 답변 여부
    is_final_answer: bool = False
    