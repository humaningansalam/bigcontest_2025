# src/main_app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from src.core.graph_builder import graph
from src.features.profile_management.resolver import resolve_store_id_from_name
from src.services.profile_service import profile_manager

st.set_page_config(page_title="소상공인 AI 비밀상담사 🤖", layout="wide")
st.title("🏪 소상공인 AI 비밀상담사")

# --- 세션 상태 초기화 ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "current_profile" not in st.session_state:
    st.session_state.current_profile = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. 세션 시작: 프로필 로드 ---
if not st.session_state.current_profile:
    st.info("상담을 위해 먼저 가맹점 정보를 불러옵니다. 상호명은 `{상호명***}` 형식으로 입력해주세요.")
    store_name_query = st.text_input("분석할 가맹점명을 입력하세요.", placeholder="예: 성동구 {고향***}")

    if st.button("상담 시작"):
        if store_name_query:
            with st.spinner(f"'{store_name_query}' 정보를 찾는 중..."):
                store_id = resolve_store_id_from_name(store_name_query)
                if store_id:
                    profile = profile_manager.get_profile(store_id)
                    if "error" not in profile:
                        st.session_state.current_profile = profile
                        st.session_state.messages = [
                            AIMessage(content=f"안녕하세요! '{profile['core_data']['basic_info']['store_name_masked']}' 사장님. 반갑습니다. 무엇을 도와드릴까요?")
                        ]
                        st.rerun()
                    else:
                        st.error(f"프로필 로딩 실패: {profile['error']}")
                else:
                    st.error("해당하는 가맹점을 찾을 수 없습니다. 상호명을 확인해주세요.")
        else:
            st.warning("가맹점명을 입력해주세요.")
else:
    # --- 2. 대화 진행 ---
    profile_name = st.session_state.current_profile['core_data']['basic_info']['store_name_masked']
    st.success(f"현재 '{profile_name}' 가맹점에 대해 상담 중입니다.")

    # 채팅 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human"):
            st.markdown(prompt)

        with st.chat_message("ai"):
            with st.status("AI가 요청을 분석하고 있습니다...", expanded=True) as status:
                inputs = {
                    "messages": [HumanMessage(content=prompt)],
                    "current_profile": st.session_state.current_profile
                }
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # 최종 상태를 저장할 변수
                final_state = None 
                
                for chunk in graph.stream(inputs, config=config):
                    # 각 노드의 출력을 확인하고 UI 업데이트
                    if "router" in chunk:
                        status.update(label="의도를 파악하고 최적의 전문가를 찾는 중...")
                    
                    if "planner" in chunk and chunk["planner"].get("plan"):
                        plan_steps = "\n".join(f"⏳ {step}" for step in chunk["planner"]["plan"])
                        status.update(label=f"작업 계획 수립 완료!\n{plan_steps}")

                    if "executor" in chunk and chunk["executor"].get("past_steps"):
                        past_steps_str = "\n".join(f"✅ {step[0]}" for step in chunk["executor"]["past_steps"])
                        remaining_plan_str = "\n".join(f"⏳ {step}" for step in chunk["executor"]["plan"])
                        status.update(label=f"작업 수행 중...\n{past_steps_str}\n{remaining_plan_str}")

                    if "synthesizer" in chunk:
                        status.update(label="수집된 정보를 종합하여 최종 답변을 생성 중...")
                        final_response = chunk["synthesizer"]["messages"][-1].content

                    final_state = chunk
                
                status.update(label="답변이 완료되었습니다!", state="complete")
            
            # --- 최종 답변 추출 로직 ---
            final_response = ""
            if final_state:
                last_node_key = list(final_state.keys())[0]
                messages = final_state[last_node_key].get("messages")
                if messages:
                    final_response = messages[-1].content

            st.markdown(final_response)

        if final_response:
            st.session_state.messages.append(AIMessage(content=final_response))