# src/features/data_analysis/tool.py

import os, pandas as pd, traceback
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from src.config import PRIMARY_MODEL_NAME
from dotenv import load_dotenv
import streamlit as st
from src.utils.errors import create_tool_error
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .prompts import create_pandas_agent_prompt

load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

class DataAnalysisInput(BaseModel):
    query: str = Field(..., description="데이터 분석을 위해 Pandas Agent에게 전달할 질문")
    store_id: str | None = Field(None, description="분석의 중심이 되는 특정 가맹점의 ID")

@tool(args_schema=DataAnalysisInput)
def data_analysis_tool(query: str, store_id: str | None = None) -> str:
    """
    Pandas DataFrame을 사용하여 원본 데이터에 대한 복잡한 질문에 답변합니다.
    """
    print(f"--- 🛠️ Tool: data_analysis_tool 호출됨 (ID: {store_id}) ---")
    try:
        # 1. data_service로부터 데이터프레임을 받아옵니다.
        df_map, dataframes = data_service.get_dataframes()
        if not dataframes:
            return "오류: 분석할 데이터프레임이 없습니다."

        # 2. 이 파일 내에서 프롬프트와 Agent를 생성합니다.
        agent_prefix = create_pandas_agent_prompt(df_map, store_id)
        llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0)
        
        pandas_agent = create_pandas_dataframe_agent(
            llm, dataframes, prefix=agent_prefix,
            agent_executor_kwargs={"handle_parsing_errors": True},
            verbose=True, allow_dangerous_code=True
        )
        
        # 3. Agent를 실행합니다.
        response = pandas_agent.invoke({"input": query})
        return response.get("output", "결과를 찾을 수 없습니다.")
        
    except Exception as e:
        return create_tool_error("data_analyzer", e, query=query)


