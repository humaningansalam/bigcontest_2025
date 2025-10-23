# src/core/common_tools/rag_search_tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from src.services.data_service import data_service
from src.utils.errors import create_tool_error

class RagSearchInput(BaseModel):
    query: str = Field(..., description="지식 베이스에서 검색할 질문이나 키워드")
    collection_types: List[str] | None = Field(
        default=["strategy", "guide", "trend"], 
        description="검색할 컬렉션 타입. (예: ['strategy', 'guide', 'local', 'case'])"
    )

@tool(args_schema=RagSearchInput)
def rag_search_tool(query: str, collection_types: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    내부 지식 베이스(RAG)에서 정보를 검색하여,
    가공되지 않은 원본 소스(Source) 객체의 리스트를 반환합니다.
    """
    print(f"--- 🛠️ Tool: rag_search_tool 호출됨 -> DataService에 위임 ---")
    try:
        if collection_types is None:
            collection_types = ("strategy", "guide", "trend", "case", "local") 
            
        sources_list = data_service.search_for_sources(query, collection_types=collection_types)
        
        return sources_list
        
    except Exception as e:
        print(create_tool_error("rag_searcher", e, query=query))
        return []