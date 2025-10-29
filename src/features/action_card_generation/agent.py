# src/features/action_card_generation/agent.py
import os
import json
import re
import traceback
from pathlib import Path
from jsonschema import Draft7Validator, ValidationError
from dotenv import load_dotenv
import streamlit as st
import os
    
load_dotenv()

# --- 경로 설정 ---
# 이 파일의 위치를 기준으로 상위 폴더(src)를 참조하여 경로를 설정
APP_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = APP_ROOT / 'action_card_generation' / 'actioncard.schema.json'
OUTPUT_DIR = APP_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 스키마 캐싱을 위한 전역 변수 ---
_SCHEMA_CACHE = None
_SCHEMA_VALIDATOR = None

def load_actioncard_schema():
    """
    actioncard.schema.json 파일을 로드하고 유효성 검사기(Validator)를 생성합니다.
    한 번 로드된 스키마는 메모리에 캐시하여 반복적인 파일 I/O를 방지합니다.
    """
    global _SCHEMA_CACHE, _SCHEMA_VALIDATOR
    if _SCHEMA_CACHE is not None and _SCHEMA_VALIDATOR is not None:
        return _SCHEMA_CACHE, _SCHEMA_VALIDATOR
    
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"스키마 파일을 찾을 수 없습니다: {SCHEMA_PATH}")
    
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        _SCHEMA_CACHE = json.load(f)
    
    _SCHEMA_VALIDATOR = Draft7Validator(_SCHEMA_CACHE)
    return _SCHEMA_CACHE, _SCHEMA_VALIDATOR

def build_agent2_prompt(agent1_like_json: dict, rag_context: str, collected_data: list = None) -> str:
    """
    Agent2가 실행 카드를 생성하기 위한 최종 프롬프트를 구성합니다.
    """
    try:
        schema, _ = load_actioncard_schema()
        tool_schema_desc_example = """
[
  {
    "tool_name": "data_analyzer",
    "query": "분석할 내용"
  },
  {
    "tool_name": "rag_searcher",
    "query": "검색할 내용"
  }
]
        """
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"--- 🚨 Agent2: 스키마 로딩 오류: {e} ---")
        schema_text = '{"error": "스키마 로딩 실패"}'
        tool_schema_desc_example = '{"error": "스키마 로딩 실패"}'

    additional_info = ""
    if collected_data:
        additional_info += "\n[추가 수집 정보]\n"
        for i, (step, result) in enumerate(collected_data):
            additional_info += f"--- 정보 {i+1} ---\n요청 내용: {step}\n수집 결과: {result}\n"

    tool_rule = (
        "**[정보 부족 시 Tool 사용 규칙]**\n"
        "- 만약 실행 카드를 만들기에 정보가 부족하다고 판단되면, `recommendations`는 반드시 빈 배열(`[]`)로 설정하세요.\n"
        "- 그리고 나서, 부족한 정보를 얻기 위해 `tool_calls` 필드에 필요한 도구와 질문을 명시하세요.\n"
        f"- `tool_calls` 필드의 형식은 다음과 같습니다:\n```json\n{tool_schema_desc_example}\n```\n"
        '- **예시:** `{"recommendations": [], "tool_calls": [{"tool_name": "data_analyzer", "query": "20대 여성 시간대별 방문 데이터 분석"}]}`\n'
        "- 정보가 충분하여 최종 실행 카드를 생성할 수 있다면, `tool_calls` 필드는 생략하거나 빈 배열로 두세요."
    )

    guide = f"""당신은 한국 소상공인 컨설턴트이며, 주어진 정보를 바탕으로 JSON 형식의 실행 카드만 생성합니다.

[가맹점 데이터]
{json.dumps(agent1_like_json, ensure_ascii=False, indent=2)}

[전문가 마케팅 지식]
{rag_context}
{additional_info}

{tool_rule}

반드시 아래 스키마를 준수하는 JSON 객체 하나만 출력하세요. 다른 설명은 절대 추가하지 마세요.
[액션카드 스키마(JSON)]
{schema_text}
"""
    return guide

def _extract_text_from_gemini_response(resp):
    """Gemini API 응답 객체에서 텍스트 콘텐츠를 안전하게 추출합니다."""
    text = ""
    try:
        if resp and resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                text += part.text
    except (AttributeError, IndexError):
        try:
            text = resp.text
        except Exception:
            pass
    return (text or "").strip()

def call_gemini_for_action_card(prompt_text: str, model_name='gemini-2.5-flash') -> dict:
    """
    주어진 프롬프트로 Gemini API를 호출하고, 스키마에 맞는 JSON 결과를 반환합니다.
    #1의 call_gemini_agent2 함수와 동일한 로직입니다.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Gemini API를 사용하려면 'pip install google-generativeai'를 설치해야 합니다.")

    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        raise ValueError('GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.')
    genai.configure(api_key=api_key)

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    generation_config = {"temperature": 0.2, "top_p": 0.9}

    try:
        _, schema_validator = load_actioncard_schema()
    except Exception as e:
        schema_validator = None
        print(f"--- ⚠️ Agent2: 스키마 유효성 검사기 로딩 실패: {e} ---")

    last_error = "알 수 없는 오류"
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = model.generate_content(prompt_text)
        text = _extract_text_from_gemini_response(response)

        if not text:
            finish_reason = "N/A"
            try:
                finish_reason = response.candidates[0].finish_reason.name
            except Exception:
                pass
            last_error = f"LLM 응답이 비어 있습니다. (종료 사유: {finish_reason})"
        else:
            # 응답 텍스트에서 JSON 블록만 정확히 추출
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                last_error = f"응답에서 JSON 블록을 찾지 못했습니다. 원본 응답: {text[:200]}..."
            else:
                json_text = match.group(0)
                try:
                    core_json = json.loads(json_text)
                    if schema_validator:
                        schema_validator.validate(core_json)
                    
                    # 모든 검증 통과: 성공적으로 결과 반환
                    print('--- ✅ Agent2: 실행 카드 생성 및 유효성 검사 완료 ---')
                    return core_json
                except (json.JSONDecodeError, ValidationError) as e:
                    last_error = f"JSON 처리/유효성 검사 실패: {e}\n--- 원본 JSON 텍스트 ---\n{json_text[:500]}..."

    except Exception as e:
        last_error = f"Gemini API 호출 실패: {type(e).__name__} - {e}"
        print(traceback.format_exc())

    # 위 과정에서 하나라도 실패하면 최종적으로 폴백(Fallback) 응답 반환
    print(f'--- 🚨 Agent2: 실행 카드 생성 실패. 폴백 응답을 반환합니다. (사유: {last_error}) ---')
    fallback_response = {
        "recommendations": [{
            "title": "⚠️ AI 모델 응답 오류",
            "what": "실행 카드를 생성하는 데 문제가 발생했습니다.",
            "how": ["잠시 후 다시 시도해주세요.", f"오류 사유: {last_error}"],
            "evidence": ["AI 모델 호출 또는 응답 처리 단계에서 예외가 발생했습니다."]
        }]
    }
    return fallback_response