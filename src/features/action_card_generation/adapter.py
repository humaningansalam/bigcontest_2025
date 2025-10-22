# src/features/action_card_generation/adapter.py

import json

def profile_to_agent1_like_json(profile: dict, user_query: str) -> dict:
    """
    현재 대화의 프로필 컨텍스트를 Agent2가 이해할 수 있는
    Agent1 출력과 유사한 JSON 형식으로 변환합니다.
    """
    print("--- 🔄 컨텍스트 어댑터: Profile -> Agent1 JSON 형식으로 변환 ---")
    
    core = profile.get("core_data", {})
    basic = core.get("basic_info", {})
    perf = core.get("performance_metrics", {})
    cust = core.get("customer_profile", {})
    ts = core.get("time_series_summary", {})

    # Agent1 JSON의 kpis 필드를 프로필에서 매핑
    kpis = {
        "revisit_rate_avg": cust.get("revisit_rate_latest_percent"),
        "new_rate_avg": cust.get("new_customer_rate_latest_percent"),
        "youth_share_avg": None, # 프로필에 해당 정보가 없다면 None
        "age_top_segments": [
            # 프로필의 main_customer_segment를 파싱하여 유사한 구조로 만들 수 있음
            # 예: {"label": "20-30대 여성", "value": 75.0}
        ],
        "customer_mix_detail": cust.get("customer_type_ratio"),
        "avg_ticket_band_label": perf.get("avg_spending_per_customer_band"),
    }
    
    # Agent1 JSON의 context 필드를 프로필에서 매핑
    context = {
        "intent": user_query,
        "parsed": {"original_question": user_query},
        "merchant": {
            "encoded_mct": profile.get("profile_id"),
            "masked_name": basic.get("store_name_masked"),
            "address": basic.get("address_district"),
            "category": basic.get("industry_main"),
        },
    }
    
    # Agent1 JSON의 최종 구조에 맞게 조합
    agent1_like_json = {
        "context": context,
        "kpis": kpis,
        "limits": ["프로필 기반 요약 정보입니다. 상세 데이터는 추가 분석이 필요할 수 있습니다."],
        "quality": "normal",
        "period": {"months": basic.get("business_age_months")},
        "sample": {"merchants_covered": 1},
    }
    
    return agent1_like_json