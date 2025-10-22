# src/core/common_tools/web_search_tool.py

from langchain_community.tools.tavily_search import TavilySearchResults
from utils.errors import create_tool_error
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WebSearchInput(BaseModel):
    query: str = Field(..., description="웹에서 검색할 내용")

@tool(args_schema=WebSearchInput)
def web_search_tool(query: str) -> str:
    """
    Tavily를 사용하여 웹 검색을 수행하고 결과를 문자열로 반환합니다.
    """
    print("--- 🔍 웹 검색 도구 실행 ---")
    try:
        search_tool = TavilySearchResults(max_results=3)
        search_result = search_tool.invoke(query)
        return str(search_result)
    except Exception as e:
        return create_tool_error("web_searcher", e)
