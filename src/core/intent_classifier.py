# src/core/intent_classifier.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import re

from src.config import PRIMARY_MODEL_NAME
import streamlit as st
from dotenv import load_dotenv  

load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# --- LLM 및 프롬프트 설정 ---
try:
    INTENT_LLM = ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL_NAME,
        temperature=0.0,
        google_api_key=google_api_key
    )
except Exception as e:
    print(f"❌ Intent LLM 초기화 오류: {e}")
    INTENT_LLM = None

# --- LLM 및 프롬프트 설정 ---
#  예시(Examples)를 파이썬 리스트로 분리
examples = [
    {"input": "우리 가게 정보 알려줘", "output": "profile_query"},
    {"input": "{고향***} 프로필 보여줘", "output": "profile_query"},
    
    {"input": "재방문율 4주 플랜 작성해줘", "output": "bigcon_request"},
    {"input": "{돔카*} 매출 증대 방안 제시", "output": "bigcon_request"},
    {"input": "마케팅 채널 추천 및 홍보안 작성해줘", "output": "bigcon_request"},
    
    {"input": "재방문율 30% 이하 매장 특성 분석", "output": "data_analysis"},
    
    {"input": "2025년 카페 트렌드", "output": "web_search"},
    
    {"input": "신규 고객 유치 아이디어", "output": "marketing_idea"},
    
    {"input": "도움될만한 영상 추천해줘", "output": "video_recommendation"},
    {"input": "마케팅 관련 유튜브 영상 찾아줘", "output": "video_recommendation"},
    {"input": "신규 고객 유치 전략에 대한 동영상 없어?", "output": "video_recommendation"},

    {"input": "지원사업 추천해줘", "output": "policy_recommendation"},
    {"input": "정부 지원 받을 거 없어?", "output": "policy_recommendation"},
    {"input": "서울시에서 하는 소상공인 정책 알려줘", "output": "policy_recommendation"},
    {"input": "안녕", "output": "greeting"},

]

# 각 예시를 포맷팅할 템플릿 정의
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# FewShot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 템플릿 구성 (예시 부분 제거)
final_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """당신은 사용자 질문의 의도를 분류하는 전문 AI입니다.
사용자의 질문을 분석하여 다음 카테고리 중 가장 적합한 하나를 선택하여 그 키워드만 답변하세요.

**[카테고리]**
- `profile_query`: 특정 가맹점의 기본 현황이나 프로필을 조회.
- `bigcon_request`: 특정 가맹점에 대한 심층 진단, 실행 카드, 솔루션, n주 플랜 등 구체적인 해결책을 요청.
- `data_analysis`: 일반적인 조건으로 데이터를 분석 요청.
- `web_search`: 외부 정보, 최신 트렌드, 경쟁사 등을 검색 요청.
- `marketing_idea`: 창의적인 마케팅 아이디어를 생성 요청.
- `video_recommendation`: 주제와 관련된 학습용 동영상 추천을 요청.
- `policy_recommendation`: 정부 지원사업, 보조금, 혜택 추천을 요청.
- `greeting`: 간단한 인사나 대화 시작.
- `unknown`: 위 카테고리에 해당하지 않는 질문.
"""
        ),
        # few_shot_prompt가 여기에 예시들을 동적으로 삽입해줍니다.
        few_shot_prompt,
        HumanMessagePromptTemplate.from_template("사용자 질문: {user_query}\n---\n분류된 의도 키워드:"),
    ]
)


def classify_intent(user_query: str) -> str:
    """LLM을 사용하여 사용자 질문의 의도를 파악합니다."""
    
    # 폴백 로직 강화
    def fallback_logic(query: str) -> str:
        print("⚠️ 경고: Intent LLM 호출에 실패하여 키워드 기반으로 의도를 추정합니다.")
        q = query.lower()
        
        # 정규 표현식을 사용하여 {상호명***} 패턴 확인
        store_name_pattern = re.search(r'\{.+\*+\}', query)

        # 1순위: Bigcon 특화 키워드 + 상호명 패턴
        if any(k in q for k in ["플랜", "실행 카드", "솔루션", "방안", "전략", "진단"]) and store_name_pattern:
            return "bigcon_request"
        
        # 2순위: 프로필 조회 + 상호명 패턴
        if any(k in q for k in ["프로필", "정보", "현황", "조회"]) and store_name_pattern:
            return "profile_query"

        # 3순위: 범용 분석 (상호명 패턴이 없을 때 더 가능성 높음)
        if any(k in q for k in ["분석", "특성", "비교", "통계"]):
            return "data_analysis"
        if any(k in q for k in ["검색", "트렌드", "최신", "방법"]):
            return "web_search"
        if any(k in q for k in ["아이디어", "제안", "이벤트", "캠페인"]):
            return "marketing_idea"
        if any(k in q for k in ["안녕", "hi", "hello"]):
            return "greeting"
        if any(k in q for k in ["영상", "영상 추천", "영상 추천해줘"]):
            return "video_recommendation"
        if any(k in q for k in ["지원사업", "정부 지원", "보조금", "혜택"]):
            return "policy_recommendation"
        # 4순위: 상호명 패턴만 단독으로 들어온 경우 (예: "{돔카*}")
        if store_name_pattern:
            return "profile_query"

        return "unknown"

    if not INTENT_LLM:
        return fallback_logic(user_query)

    try:
        chain = final_prompt | INTENT_LLM | StrOutputParser()
        intent = chain.invoke({"user_query": user_query})
        return intent.strip().lower()
    except Exception as e:
        print(f"❌ 의도 분류 중 오류 발생: {e}")
        return fallback_logic(user_query)