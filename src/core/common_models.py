# src/core/common_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ToolOutput(BaseModel):
    """
    모든 Tool이 반환해야 하는 표준 출력 형식입니다.
    Pydantic 모델을 사용하여 런타임 타입 검증과 기본값을 보장합니다.
    """
    content: str
    is_final_answer: bool = False
    sources: List[Dict[str, Any]] = Field(default_factory=list)