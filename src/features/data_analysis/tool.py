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
from src.core.tool_registry import tool_registry

load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))


TOOL_DESCRIPTION = "프로필에 없는 상세 수치 데이터(예: 시간대별, 메뉴별, 고객 세그먼트별)를 원본 CSV 파일에서 직접 심층 분석하는 '데이터 과학자'입니다."

class DataAnalysisInput(BaseModel):
    query: str = Field(..., description="데이터 분석을 위해 Pandas Agent에게 전달할 질문")
    store_id: str | None = Field(None, description="분석의 중심이 되는 특정 가맹점의 ID")

@tool_registry.register(
    name="data_analyzer",
    description=TOOL_DESCRIPTION,
    needs_store_id=True
)
@tool(args_schema=DataAnalysisInput)
def data_analysis_tool(query: str, store_id: str | None = None) -> dict:
    """
    Pandas DataFrame을 사용하여 원본 데이터에 대한 복잡한 질문에 답변합니다.
    """
    print(f"--- 🛠️ Tool: data_analysis_tool 호출됨 (ID: {store_id}) ---")
    try:
        df_map, dataframes = data_service.get_dataframes()
        if not dataframes:
            return ToolOutput(content="오류: 분석할 데이터프레임이 없습니다.").model_dump()

        agent_prefix = create_pandas_agent_prompt(df_map, store_id)
        llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0)
        
        pandas_agent = create_pandas_dataframe_agent(
            llm, dataframes, prefix=agent_prefix,
            agent_executor_kwargs={"handle_parsing_errors": True},
            verbose=True, allow_dangerous_code=True
        )
        
        try:
            response = pandas_agent.invoke({"input": query})
            output_text = response.get("output", "결과를 찾을 수 없습니다.")

            is_final = "Final Answer:" in output_text
            content = output_text.split("Final Answer:", 1)[1].strip() if is_final else output_text

            return ToolOutput(content=content, is_final_answer=is_final, sources=None).model_dump()
        except Exception as e:
            error_content = create_tool_error("data_analysis", e, query=query)
            return ToolOutput(content=error_content).model_dump()
        
    except Exception as e:
        error_content = create_tool_error("data_analysis", e, query=query)
        return ToolOutput(content=error_content).model_dump()


