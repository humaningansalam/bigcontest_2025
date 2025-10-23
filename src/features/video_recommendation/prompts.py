# src/features/video_recommendation/prompts.py

from typing import Dict, Any, List

def create_video_recommendation_prompt(profile: Dict[str, Any], sources: List[Dict[str, Any]], user_query: str) -> str:
    """
    LLM이 검색된 영상들을 요약하고 개인화된 추천사를 생성하도록 지시하는 프롬프트를 만듭니다.
    """
    
    # 프로필에서 컨텍스트 정보 추출
    industry = profile.get("core_data", {}).get("basic_info", {}).get("industry_main", "소상공인")
    business_age = profile.get("core_data", {}).get("basic_info", {}).get("business_age_months", 0)

    # 검색된 영상 정보들을 텍스트로 변환
    source_texts = []
    for i, src in enumerate(sources):
        title = src.get('title', '제목 없음')
        creator = src.get('creator', '알 수 없는 제작자')
        url = src.get('url', '#')
        script_summary = src.get('content', '스크립트 정보 없음') 
        
        source_texts.append(f"""
### 영상 {i+1}: "{title}"
- 제작자: {creator}
- URL: {url}
- 스크립트 내용: {script_summary[:500]}... 
""")

    sources_str = "\n".join(source_texts)

    prompt = f"""당신은 소상공인 전문 비즈니스 컨설턴트입니다.
당신의 임무는 아래 [상황 정보]와 검색된 [영상 정보]를 바탕으로, 소상공인 사장님께 맞춤형 학습 영상을 추천하는 것입니다.

**[상황 정보]**
- **사장님 업종:** {industry}
- **사장님 업력:** 약 {business_age}개월
- **사장님 질문:** "{user_query}"

**[검색된 영상 정보]**
{sources_str}

**[당신의 임무]**
1.  각 영상의 **'스크립트 내용'을 1~2 문장으로 객관적으로 요약**하여 **'핵심 내용'**을 작성해주세요.
2.  그 요약된 내용을 바탕으로, 이 영상이 사장님의 현재 상황과 질문에 **왜 도움이 되는지** 분석하여 **'추천 이유'**를 작성해주세요.
3.  아래 **[출력 형식]**을 반드시 준수하여 최종 추천 메시지를 생성해주세요.
4.  가장 관련성이 높은 1~2개의 영상만 선별하여 추천하세요.

**[출력 형식]**
사장님의 '{user_query}' 고민 해결에 도움이 될 만한 영상을 추천해 드립니다.

- **[영상 제목]([검색된 영상 정보]에서 찾은 실제 URL)**
  - **핵심 내용:** (여기에 영상 스크립트의 객관적인 요약본을 작성)
  - **추천 이유:** (여기에 이 영상이 왜 사장님께 도움이 되는지 개인화된 추천 이유를 작성)
"""
    return prompt