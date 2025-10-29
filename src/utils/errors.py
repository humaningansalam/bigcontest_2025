# src/utils/errors.py 


def create_tool_error(tool_name: str, e: Exception, query: str = "N/A") -> str:
    """표준화된 에러 메시지 문자열을 반환합니다."""
    error_details = f"{type(e).__name__}: {str(e)}".replace('"', "'") 
    return f"ERROR: 도구 '{tool_name}' 실행 중 오류가 발생했습니다. (입력: '{query}', 상세: {error_details})"