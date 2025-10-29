# src/core/common_tools/rag_search_tool.py

from typing import List, Dict, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.core.common_models import ToolOutput
from src.core.tool_registry import tool_registry
from src.services.data_service import data_service
from src.utils.errors import create_tool_error

# --- 도구 설명 및 입력 스키마 정의 ---

TOOL_DESCRIPTION = """내부 지식 베이스에서 검증된 정보를 검색하는 '사내 자료 분석가'입니다.
- **주요 사용처:** 시장 트렌드, 마케팅 전략, 실행 가이드 등 특정 가맹점과 무관한 일반적인 정보를 찾을 때 사용합니다.
- **핵심 파라미터:** `collection_types`를 사용하여 검색 범위를 `trend`, `strategy`, `guide` 등으로 한정할 수 있습니다. 이를 통해 더 정확한 결과를 얻을 수 있습니다.
- **지시:** Planner는 사용자의 질문 의도에 가장 적합한 `collection_types` 1~2개를 반드시 지정하여 계획을 수립해야 합니다."""


class RagSearchInput(BaseModel):
    """RAG 검색 도구의 입력 파라미터를 정의하고 유효성을 검사합니다."""
    query: str = Field(
        ...,
        description="지식 베이스에서 검색할 질문이나 키워드"
    )
    collection_types: List[str] | None = Field(
        default=None,
        description="검색할 컬렉션 타입 리스트. 예: ['strategy', 'guide']. 지정하지 않으면 모든 타입을 검색합니다."
    )


# --- 도구 구현 ---

@tool_registry.register(
    name="rag_searcher",
    description=TOOL_DESCRIPTION
)
@tool(args_schema=RagSearchInput)
def rag_search_tool(query: str, collection_types: List[str] | None = None) -> dict:
    """
    내부 지식 베이스(RAG)에서 정보를 검색하고, 검색 결과를 요약한 텍스트와
    원본 소스(Source) 객체 리스트를 함께 반환합니다.
    """
    print(f"--- 🛠️ Tool: rag_searcher 호출됨 (Query: '{query}') ---")
    try:
        # collection_types가 None이면 모든 타입을 검색하도록 기본값 설정
        search_collections = tuple(collection_types) if collection_types else ("strategy", "guide", "trend", "case")

        # 실제 검색 작업은 DataService에 위임
        sources_list = data_service.search_for_sources(query, collection_types=search_collections)

        if not sources_list:
            content = f"'{query}'와 관련된 자료를 찾지 못했습니다."
        else:
            # 검색 결과를 Synthesizer가 이해하기 좋은 요약 텍스트로 가공
            context_parts = []
            for i, src in enumerate(sources_list):
                title = src.get('title', '제목 없음')
                content_part = str(src.get('content', ''))[:500]  
                context_parts.append(f"[{i+1}] 제목: {title}\n   내용 요약: {content_part}...")
            content = "\n\n".join(context_parts)

        print(f"--- [RAG Tool] '{query}' 검색 결과 {len(sources_list)}건 발견 ---")

        return ToolOutput(content=content, sources=sources_list).model_dump()

    except Exception as e:
        error_content = create_tool_error("rag_searcher", e, query=query)
        return ToolOutput(content=error_content).model_dump()