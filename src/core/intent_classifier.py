# src/core/intent_classifier.py

import os
import re
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import PRIMARY_MODEL_NAME

# --- 1. 설정 및 초기화 ---
load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# 의도 분류를 위한 전용 LLM 인스턴스
try:
    INTENT_LLM = ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL_NAME,
        temperature=0.0,  
        google_api_key=google_api_key
    )
except Exception as e:
    print(f"❌ Intent LLM 초기화 오류: {e}")
    INTENT_LLM = None


# --- 2. 프롬프트 구성 요소 (Prompt Components) ---

# LLM에게 학습시킬 Few-shot 예시 데이터
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"input": "우리 가게 정보 알려줘", "output": "profile_query"},
    {"input": "{고향***} 프로필 보여줘", "output": "profile_query"},

    {"input": "재방문율 4주 플랜 작성해줘", "output": "bigcon_request"},
    {"input": "{돔카*} 매출 증대 방안 제시", "output": "bigcon_request"},
    {"input": "마케팅 채널 추천 및 홍보안 작성해줘", "output": "bigcon_request"},

    {"input": "요즘 외식 트렌드가 어때?", "output": "general_rag_search"},
    {"input": "고객 관리 기법에 대해 알려줘", "output": "general_rag_search"},

    {"input": "재방문율 30% 이하 매장 특성 분석", "output": "data_analysis"},

    {"input": "신규 고객 유치 아이디어", "output": "marketing_idea"},

    {"input": "도움될만한 영상 추천해줘", "output": "video_recommendation"},
    {"input": "마케팅 관련 유튜브 영상 찾아줘", "output": "video_recommendation"},

    {"input": "지원사업 추천해줘", "output": "policy_recommendation"},
    {"input": "정부 지원 받을 거 없어?", "output": "policy_recommendation"},

    {"input": "안녕", "output": "greeting"},
]

# 시스템 메시지는 LLM의 역할과 목표, 그리고 분류할 카테고리 목록을 정의
SYSTEM_MESSAGE_TEMPLATE = """당신은 사용자 질문의 의도를 분류하는 전문 AI입니다.
사용자의 질문을 분석하여 다음 카테고리 중 가장 적합한 하나를 선택하여 그 키워드만 답변하세요.

**[카테고리]**
- `profile_query`: 특정 가맹점의 기본 현황이나 프로필을 조회.
- `general_rag_search`: "요즘 트렌드", "마케팅 기법" 등 **특정 가게와 무관한 일반적인 지식이나 정보**를 RAG에서 검색 요청.
- `bigcon_request`: 특정 가맹점에 대한 심층 진단, 실행 카드, 솔루션, n주 플랜 등 구체적인 해결책을 요청.
- `data_analysis`: 일반적인 조건으로 데이터를 분석 요청.
- `marketing_idea`: 창의적인 마케팅 아이디어를 생성 요청.
- `video_recommendation`: 주제와 관련된 학습용 동영상 추천을 요청.
- `policy_recommendation`: 정부 지원사업, 보조금, 혜택 추천을 요청.
- `greeting`: 간단한 인사나 대화 시작.
- `unknown`: 위 카테고리에 해당하지 않는 질문.
"""

def _build_intent_classifier_chain():
    """의도 분류를 위한 LangChain 체인을 동적으로 생성합니다."""
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=FEW_SHOT_EXAMPLES,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_TEMPLATE),
        few_shot_prompt,
        HumanMessagePromptTemplate.from_template("사용자 질문: {user_query}\n---\n분류된 의도 키워드:"),
    ])

    return final_prompt | INTENT_LLM | StrOutputParser()

# --- 3. 핵심 기능 함수 ---

def _fallback_logic(query: str) -> str:
    """LLM 호출 실패 시, 키워드와 정규식을 사용하여 의도를 추정하는 폴백 함수입니다."""
    print("⚠️ 경고: Intent LLM 호출에 실패하여 키워드 기반으로 의도를 추정합니다.")
    q = query.lower()
    store_name_pattern = re.search(r'\{.+\*+\}', query)

    # 우선순위가 높은 규칙부터 순서대로 검사합니다.
    if store_name_pattern:
        if any(k in q for k in ["플랜", "실행 카드", "솔루션", "방안", "전략", "진단"]):
            return "bigcon_request"
        if any(k in q for k in ["프로필", "정보", "현황", "조회"]):
            return "profile_query"

    if any(k in q for k in ["분석", "특성", "비교", "통계"]): return "data_analysis"
    if any(k in q for k in ["트렌드", "최신", "방법"]): return "general_rag_search"
    if any(k in q for k in ["아이디어", "제안", "이벤트"]): return "marketing_idea"
    if any(k in q for k in ["영상", "유튜브", "동영상"]): return "video_recommendation"
    if any(k in q for k in ["지원사업", "정부 지원", "보조금"]): return "policy_recommendation"
    if any(k in q for k in ["안녕", "hi", "hello"]): return "greeting"

    # 다른 규칙에 해당하지 않지만 상호명 패턴이 있다면 프로필 조회로 간주
    if store_name_pattern:
        return "profile_query"

    return "unknown"


def classify_intent(user_query: str) -> str:
    """
    사용자 질문의 의도를 분류합니다.

    LLM을 사용한 분류를 우선 시도하고, 실패 시 폴백 로직을 사용합니다.

    Args:
        user_query: 사용자가 입력한 원본 질문 문자열.

    Returns:
        분류된 의도 키워드 문자열 (예: 'profile_query').
    """
    if not INTENT_LLM:
        return _fallback_logic(user_query)

    try:
        chain = _build_intent_classifier_chain()
        intent = chain.invoke({"user_query": user_query})
        return intent.strip().lower()
    except Exception as e:
        print(f"❌ 의도 분류 중 오류 발생: {e}")
        return _fallback_logic(user_query)