# src/features/policy_recommendation/tool.py

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
from .prompts import create_policy_recommendation_prompt

load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

class PolicyRecommenderInput(BaseModel):
    user_query: str = Field(..., description="사용자의 원본 질문 또는 지원사업 추천을 위한 주제")
    profile: Dict[str, Any] = Field(..., description="추천의 개인화를 위한 현재 가맹점 프로필")

@tool(args_schema=PolicyRecommenderInput)
def policy_recommender_tool(user_query: str, profile: Dict[str, Any]) -> str:
    """
    사용자의 질문과 프로필을 바탕으로 가장 적합한 정부/지자체 지원사업을 맞춤 추천합니다.
    """
    print(f"--- ✨ Feature: policy_recommender_tool 호출됨 ---")
    try:
        # 1. 사용자 프로필을 바탕으로 검색어 생성
        industry = profile.get("core_data", {}).get("basic_info", {}).get("industry_main", "")
        address = profile.get("core_data", {}).get("basic_info", {}).get("address_district", "")
        # 예: "서울 카페 인력 지원"
        search_query = f"{address} {industry} {user_query}"
        
        # 2. DataService를 통해 'case_studies_and_policies' 컬렉션 검색
        sources = data_service.search_for_sources(
            query=search_query, 
            collection_types=("case",) # 'case' 타입으로 검색 요청
        )

        if not sources:
            return f"'{user_query}'와 관련된 맞춤 지원사업을 찾지 못했습니다."
        
        # 3. LLM을 사용하여 개인화된 추천사 생성
        llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0.1)
        
        recommendation_prompt = create_policy_recommendation_prompt(profile, sources, user_query)
        final_recommendation = llm.invoke(recommendation_prompt).content
        
        return final_recommendation

    except Exception as e:
        return create_tool_error("policy_recommender", e, query=user_query)