# src/core/common_tools/api_call_tool.py

# 현재는 더미 구현된 외부 API 호출 기능을 시뮬레이션하는 예제입니다.

import json
from langchain_core.messages import HumanMessage
from src.utils.errors import create_tool_error

class PolicySearchInput(BaseModel):
    topic: str = Field(..., description="검색할 정책 자금 주제 (예: 청년 창업)")

@tool(args_schema=PolicySearchInput)
def policy_search_tool(topic: str) -> str: 
    """
    소상공인 관련 정책 자금 정보를 반환하는 가상의 API 함수
    """
    
    if "청년" in topic:
        return json.dumps({
            "product_name": "청년 소상공인 특별자금",
            "interest_rate": "2.5%",
            "limit": "1억원 이내",
            "conditions": "만 39세 이하 청년 창업가"
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "product_name": "일반 소상공인 성장자금",
            "interest_rate": "3.0%~",
            "limit": "5억원 이내",
            "conditions": "업력 1년 이상 소상공인"
        }, ensure_ascii=False)

def _call_api(state: dict) -> str:
    """
    외부 API를 호출하는 '도구'입니다.
    state 대신 간단한 query를 받아 문자열 결과를 반환합니다.
    """
    return _get_policy_fund_info(state)

