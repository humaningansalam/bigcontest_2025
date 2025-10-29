# src/core/tool_registry.py 

from typing import Dict, Any
from langchain_core.tools import BaseTool

class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}
    _tool_descriptions: Dict[str, str] = {}
    _tool_metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls, 
        name: str, 
        description: str,
        needs_profile: bool = False,
        needs_user_query: bool = False,
        needs_store_id: bool = False
    ):
        """
        도구를 레지스트리에 등록하는 데코레이터.
        실행에 필요한 컨텍스트 메타데이터도 함께 등록합니다.
        """
        def decorator(tool_obj: BaseTool):
            if not isinstance(tool_obj, BaseTool):
                raise TypeError("Registered object must be a LangChain BaseTool instance.")
            
            cls._tools[name] = tool_obj
            cls._tool_descriptions[name] = description
            cls._tool_metadata[name] = {
                "needs_profile": needs_profile,
                "needs_user_query": needs_user_query,
                "needs_store_id": needs_store_id,
            }
            print(f"✅ Tool '{name}' registered with metadata.")
            return tool_obj
        return decorator

    @classmethod
    def get_tool(cls, name: str) -> BaseTool:
        if name not in cls._tools:
            raise ValueError(f"Tool '{name}' not found.")
        return cls._tools[name]
    
    @classmethod
    def get_tool_metadata(cls, name: str) -> Dict[str, Any]:
        """도구에 필요한 컨텍스트 메타데이터를 가져옵니다."""
        return cls._tool_metadata.get(name, {})

    @classmethod
    def get_all_tools(cls) -> Dict[str, BaseTool]:
        return cls._tools.copy()

    @classmethod
    def get_all_descriptions(cls) -> Dict[str, str]:
        return cls._tool_descriptions.copy()


tool_registry = ToolRegistry()