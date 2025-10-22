# src/utils/errors.py 
import json
from typing import TypedDict

class ToolError(TypedDict):
    status: str
    tool_name: str
    query: str
    message: str
    details: str | None

def create_tool_error(tool_name: str, e: Exception, query: str = "N/A") -> str:
    """표준화된 에러 dict를 JSON 문자열로 반환합니다."""
    error_payload: ToolError = {
        "status": "error",
        "tool_name": tool_name,
        "query": query, 
        "message": f"{tool_name} 실행 중 오류 발생",
        "details": f"{type(e).__name__}: {str(e)}"
    }
    return json.dumps(error_payload, ensure_ascii=False)