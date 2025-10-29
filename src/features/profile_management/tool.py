# src/features/profile_management/tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any

from src.services.profile_service import profile_manager
from src.utils.errors import create_tool_error
from src.core.common_models import ToolOutput

GET_PROFILE_TOOL_DESCRIPTION = "가맹점의 ID를 받아 전체 프로필 정보를 JSON 객체(딕셔너리)로 조회합니다." 
UPDATE_PROFILE_TOOL_DESCRIPTION = "사용자와의 대화에서 얻은 새로운 정보를 가맹점 프로필 데이터베이스에 업데이트하는 '비서'입니다. (예: 사장님의 사업 목표, 연령대 등)"

class GetProfileInput(BaseModel):
    """가맹점 ID로 프로필을 조회하기 위한 입력 스키마"""
    store_id: str = Field(..., description="조회할 가맹점의 고유 ID(ENCODED_MCT)")

class UpdateProfileInput(BaseModel):
    """프로필의 특정 부분을 업데이트하기 위한 입력 스키마"""
    store_id: str = Field(..., description="업데이트할 가맹점의 고유 ID")
    section: str = Field(..., description="업데이트할 최상위 섹션 (예: 'extended_features')")
    key: str = Field(..., description="섹션 내에서 업데이트할 키 (예: 'owner_info')")
    data_to_update: Dict[str, Any] = Field(..., description="업데이트할 내용을 담은 딕셔너리")

# --- Tool 함수 정의 ---

@tool_registry.register(
    name="get_profile",
    description=GET_PROFILE_TOOL_DESCRIPTION
)
@tool(args_schema=GetProfileInput)
def get_profile(store_id: str) -> dict:
    """가맹점의 ID를 받아 전체 프로필 정보를 JSON 객체(딕셔너리)로 조회합니다."""
    print(f"--- 🛠️ Tool: get_profile 호출됨 (ID: {store_id}) ---")
    profile = profile_manager.get_profile(store_id) 
    if profile:
        return ToolOutput(content=profile, is_final_answer=True).model_dump()
    else:
        error_content = create_tool_error(
            "get_profile", 
            Exception(f"ID '{store_id}'에 해당하는 프로필을 찾을 수 없습니다."),
            query=f"store_id={store_id}"
        )
        return ToolOutput(content=error_content).model_dump()

@tool_registry.register(
    name="update_profile",
    description=UPDATE_PROFILE_TOOL_DESCRIPTION
)
@tool(args_schema=UpdateProfileInput)
def update_profile(store_id: str, section: str, key: str, data_to_update: dict) -> dict:
    """가맹점 프로필의 특정 부분을 새로운 정보로 업데이트합니다."""
    print(f"--- 🛠️ Tool: update_profile 호출됨 (ID: {store_id}, Section: {section}) ---")
    success = profile_manager.update_profile(store_id, section, key, data_to_update) 
    if success:
        return ToolOutput(content=f"'{store_id}' 프로필의 '{section}.{key}'가 성공적으로 업데이트되었습니다.", is_final_answer=True).model_dump()
    else:
        error_content = create_tool_error(
            "update_profile", 
            Exception("프로필 업데이트 중 서버 측 오류 발생"),
            query=f"store_id={store_id}"
        )
        return ToolOutput(content=error_content).model_dump()