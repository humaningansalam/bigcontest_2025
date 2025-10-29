# src/features/video_recommendation/tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from dotenv import load_dotenv

from src.services.data_service import data_service
from src.utils.errors import create_tool_error
from src.config import PRIMARY_MODEL_NAME
from .prompts import create_video_recommendation_prompt
from src.core.common_models import ToolOutput

load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

TOOL_DESCRIPTION = "사용자의 질문과 프로필에 맞춰 관련된 학습 영상을 검색하고, 각 영상의 내용을 요약하여 맞춤 추천하는 '미디어 큐레이터'입니다. 텍스트 설명 외에 시청각 자료가 필요할 때 사용하세요."

class VideoRecommenderInput(BaseModel):
    user_query: str = Field(..., description="사용자의 원본 질문 또는 영상 추천을 위한 주제")
    profile: Dict[str, Any] = Field(..., description="추천의 개인화를 위한 현재 가맹점 프로필")

@tool_registry.register(
    name="video_recommender",
    description=TOOL_DESCRIPTION,
    needs_profile=True,
    needs_user_query=True
)
@tool(args_schema=VideoRecommenderInput)
def video_recommender_tool(user_query: str, profile: Dict[str, Any]) -> dict:
    """
    사용자의 질문과 프로필을 바탕으로 가장 적합한 학습 영상을 검색하고,
    각 영상의 핵심 내용을 요약하여 맞춤 추천합니다.
    """
    print(f"--- ✨ Feature: video_recommender_tool (Agent-mode) 호출됨 ---")
    try:
        industry = profile.get("core_data", {}).get("basic_info", {}).get("industry_main", "소상공인")
        search_query = f"{industry} {user_query}"
        
        sources = data_service.search_for_sources(
            query=search_query, 
            collection_types=("video",)
        )

        if not sources:
            return ToolOutput(content=f"'{user_query}'에 대한 맞춤 추천 영상을 찾지 못했습니다.").model_dump()
        
        llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0.3)
        
        recommendation_prompt = create_video_recommendation_prompt(profile, sources, user_query)
        
        final_recommendation = llm.invoke(recommendation_prompt).content
        
        return ToolOutput(content=final_recommendation, is_final_answer=True, sources=sources).model_dump()

    except Exception as e:
        error_content = create_tool_error("video_recommender", e, query=user_query)
        return ToolOutput(content=error_content).model_dump()