# src/core/common_tools/rag_search_tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
from src.services.data_service import data_service
from src.utils.errors import create_tool_error

class RagSearchInput(BaseModel):
    query: str = Field(..., description="지식 베이스에서 검색할 질문이나 키워드")
    collection_types: List[str] | None = Field(
        default=["strategy", "guide", "trend"], 
        description="검색할 컬렉션 타입. (예: ['strategy', 'guide', 'local', 'case'])"
    )

@tool(args_schema=RagSearchInput)
def rag_search_tool(query: str, collection_types: List[str] | None = None) -> str:
    """
    내부 지식 베이스(RAG)에서 소상공인 관련 전략, 가이드, 트렌드 정보를 검색하여
    요약된 문자열 컨텍스트를 반환합니다.
    """
    print(f"--- 🛠️ Tool: rag_search_tool 호출됨 -> DataService에 위임 ---")
    try:
        if collection_types is None:
            collection_types = ("strategy", "guide", "trend", "case", "local") 
            
        context_str = data_service.search_for_context(query, collection_types=collection_types)
        
        if not context_str or "찾을 수 없습니다" in context_str:
            return f"'{query}'에 대한 정보를 내부 지식 베이스에서 찾지 못했습니다."
            
        return context_str
    except Exception as e:
        return create_tool_error("rag_searcher", e, query=query)