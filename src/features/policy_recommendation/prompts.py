# src/features/policy_recommendation/prompts.py

from typing import Dict, Any, List

def create_policy_recommendation_prompt(profile: Dict[str, Any], sources: List[Dict[str, Any]], user_query: str) -> str:
    """
    LLM이 검색된 지원사업들을 분석하고 개인화된 추천사를 생성하도록 지시하는 프롬프트를 만듭니다.
    """
    
    # 프로필에서 컨텍스트 정보 추출
    industry = profile.get("core_data", {}).get("basic_info", {}).get("industry_main", "소상공인")
    address = profile.get("core_data", {}).get("basic_info", {}).get("address_district", "정보 없음")
    
    source_texts = []
    for i, src in enumerate(sources):
        title = src.get('공고명', '공고명 없음')
        agency = src.get('주관기관', '정보 없음')
        target = src.get('지원대상', '정보 없음')
        url = src.get('접수처_url', '#') 
        content = src.get('content', '내용 없음')
        
        source_texts.append(f"""
### 지원사업 {i+1}: "{title}"
- 주관기관: {agency}
- 지원대상: {target}
- 접수처 URL: {url}
- 사업 내용: {content[:500]}... 
""")

    sources_str = "\n".join(source_texts)

    prompt = f"""당신은 정부 및 지자체 지원사업 전문 컨설턴트입니다.
당신의 임무는 아래 [상황 정보]와 검색된 [지원사업 정보]를 바탕으로, 소상공인 사장님께 가장 적합한 지원사업을 추천하는 것입니다.

**[상황 정보]**
- **사장님 업종:** {industry}
- **사장님 주소:** {address}
- **사장님 질문:** "{user_query}"

**[검색된 지원사업 정보]**
{sources_str}

**[당신의 임무]**
1.  각 지원사업의 **'지원대상'**과 **'사업 내용'**을 꼼꼼히 분석하세요.
2.  사장님의 **업종과 지역**을 고려했을 때, **신청 자격이 되거나 가장 관련성이 높은 사업**을 1~2개 선별하세요.
3.  선별된 각 사업에 대해, **왜 사장님께 이 사업이 도움이 되는지** 명확한 **'추천 이유'**를 작성해주세요.
4.  아래 **[출력 형식]**을 반드시 준수하여 최종 추천 메시지를 생성해주세요. **접수처 URL을 반드시 포함해야 합니다.**

**[출력 형식]**
사장님께서 관심을 가지실 만한 지원사업 정보를 찾아봤어요.

- **[지원사업 공고명 1]**
  - **주관기관:** (주관기관 정보)
  - **지원대상:** (지원대상 정보)
  - **추천 이유:** (여기에 이 사업이 왜 사장님께 도움이 되는지 개인화된 추천 이유를 작성)
  - **신청 바로가기:** [검색된 지원사업 정보]에서 찾은 실제 접수처 URL

- **[지원사업 공고명 2]**
  - **주관기관:** (주관기관 정보)
  - **지원대상:** (지원대상 정보)
  - **추천 이유:** (여기에 이 사업이 왜 사장님께 도움이 되는지 개인화된 추천 이유를 작성)
  - **신청 바로가기:** [검색된 지원사업 정보]에서 찾은 실제 접수처 URL
"""
    return prompt