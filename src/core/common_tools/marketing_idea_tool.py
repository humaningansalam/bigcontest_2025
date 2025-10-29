# src/core/common_tools/marketing_idea_tool.py

from langchain_google_genai import ChatGoogleGenerativeAI
from config import PRIMARY_MODEL_NAME
from dotenv import load_dotenv
import streamlit as st
import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.core.tool_registry import tool_registry

from src.utils.errors import create_tool_error
from src.core.common_models import ToolOutput

load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0.7) # 창의성을 위해 온도를 약간 높임


TOOL_DESCRIPTION = "분석된 데이터나 트렌드를 바탕으로 창의적인 마케팅 아이디어를 브레인스토밍하는 '마케팅 전문가'입니다."

class MarketingIdeaInput(BaseModel):
    topic: str = Field(..., description="마케팅 아이디어 생성을 위한 기반이 될 주제나 데이터 분석 결과")

@tool_registry.register(
    name="marketing_idea_generator",
    description=TOOL_DESCRIPTION
)
@tool(args_schema=MarketingIdeaInput)
def marketing_idea_generator_tool(topic: str) -> dict:
    """
    주어진 주제(데이터 분석 결과, 트렌드)를 바탕으로 마케팅 아이디어를 생성합니다.
    """
    print("--- 💡 마케팅 아이디어 생성 도구 실행 ---")
    
    try:
        prompt = f"""당신은 데이터 기반 마케팅 아이디어 전문가입니다.
    아래에 제공된 '분석 결과 및 트렌드'를 바탕으로, 소상공인 매장을 위한 **구체적이고 실행 가능한 마케팅 아이디어 3가지**를 각각의 근거와 함께 제안해주세요.

    **[분석 결과 및 트렌드]**
    {topic}

    **[마케팅 아이디어 제안 (구체적인 실행 방안과 근거 포함)]**
    1. **아이디어**: ...
    - **근거**: ...
    2. ...
    """
        response = llm.invoke(prompt)
        return ToolOutput(content=response.content).model_dump()
    except Exception as e:
        error_content = create_tool_error("marketing_idea_generator", e, query=topic)
        return ToolOutput(content=error_content).model_dump()
