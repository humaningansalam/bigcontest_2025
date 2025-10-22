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

load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

class DataAnalysisInput(BaseModel):
    query: str = Field(..., description="데이터 분석을 위해 Pandas Agent에게 전달할 질문")
    store_id: str | None = Field(None, description="분석의 중심이 되는 특정 가맹점의 ID")

# 정] @tool 데코레이터와 Pydantic 모델을 사용하여 명확한 인터페이스 정의
@tool(args_schema=DataAnalysisInput)
def data_analysis_tool(query: str, store_id: str | None = None) -> str:
    """
    Pandas DataFrame을 사용하여 원본 데이터에 대한 복잡한 질문에 답변합니다.
    특정 가맹점 컨텍스트가 필요할 경우 store_id를 함께 제공해야 합니다.
    """
    print(f"--- 🛠️ Tool: data_analysis_tool 호출됨 (ID: {store_id}) ---")
    data_dir = "./data/"
    try:
        # [삭] state에서 정보를 추출하는 로직 제거
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files: return "분석할 CSV 파일이 'data' 폴더에 없습니다."
        
        
        df_map = {} 
        for f in csv_files:
            file_path = os.path.join(data_dir, f)
            try: df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError: df = pd.read_csv(file_path, encoding='cp949')
            df_map[f] = df
        
        dataframes = list(df_map.values())

    except Exception as e:
        return create_tool_error("data_analyzer", e)

    llm = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME, google_api_key=google_api_key, temperature=0)
    
    # 동적으로 DataFrame 변수와 파일 이름을 매핑하는 정보를 생성합니다.
    df_info_str = "\n".join([f"- df{i+1}: '{filename}'" for i, filename in enumerate(df_map.keys())])

    context_injection_prompt = ""
    if store_id:
        context_injection_prompt =  f"""
**[현재 분석 컨텍스트]**
- 당신은 지금 가맹점 ID가 '{store_id}'인 특정 가맹점에 대해 컨설팅하고 있습니다.
- 사용자가 '우리 가게', '해당 매장' 등 자신을 지칭하면, 이는 ID '{store_id}'를 의미합니다.
- 'ENCODED_MCT' 컬럼을 사용하여 이 가맹점의 데이터를 찾을 수 있습니다.
- 다른 가맹점과의 비교 분석 요청이 있을 수 있으니, 전체 데이터는 필터링하지 말고 사용하세요.
"""

    # 사용자님의 지침과 저의 보완 사항을 결합한 최종 prefix
    agent_prefix = f"""
당신은 Python Pandas 전문가이며, 주어진 DataFrame(df1, df2, ...)을 사용하여 데이터 분석 질문에 답변하는 임무를 맡았습니다.

**[중요 지침]**
1.  **컨텍스트 인지:** 현재 가맹점 ID '{store_id}'에 대한 분석 요청임을 인지하고, '우리 가게' 등의 표현은 이 ID를 의미하는 것으로 해석하세요. 'ENCODED_MCT' 컬럼을 활용하세요.
2.  **자유로운 분석:** 질문에 답하기 위해 필요한 모든 Python 코드를 자유롭게 생성하고 실행하세요. 여러 단계의 `Action`을 사용해도 좋습니다.
3.  **데이터 한계 명시:** 분석에 필요한 데이터가 없다면, 그 사실을 최종 답변에 명확히 포함시키세요.
4.  **최종 보고:** 모든 분석이 끝나면, 반드시 `Final Answer:` 키워드로 시작하는 최종 요약 보고서를 작성하여 작업을 마무리하세요.


**[사용 가능한 DataFrame 정보]**
당신에게는 다음과 같은 파일들이 DataFrame으로 주어졌습니다. 코드를 작성할 때 이 정보를 반드시 참고하여 올바른 변수(df1, df2 등)를 사용해야 합니다.
{df_info_str}

**[분석 파일 설명서]**
- 'big_data_set1_f.csv'의 컬럼 설명은 '2025_빅콘테스트_데이터_레이아웃_20250902_데이터셋1.csv' 파일을 참고하세요.
- 'big_data_set2_f.csv'의 컬럼 설명은 '2025_빅콘테스트_데이터_레이아웃_20250902_데이터셋2.csv' 파일을 참고하세요.
- 'big_data_set3_f.csv'의 컬럼 설명은 '2025_빅콘테스트_데이터_레이아웃_20250902_데이터셋3.csv' 파일을 참고하세요.
"""

    try:
        pandas_agent = create_pandas_dataframe_agent(
            llm, 
            dataframes,
            prefix=agent_prefix,
            agent_executor_kwargs={"handle_parsing_errors": True}, 
            verbose=True,
            allow_dangerous_code=True
        )
        
        response = pandas_agent.invoke({"input": query})
        return response.get("output", "결과를 찾을 수 없습니다.")
    except Exception as e:
        return create_tool_error("data_analyzer", e)
    
