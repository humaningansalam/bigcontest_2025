# src/services/data_service.py

import pandas as pd
from functools import lru_cache
from typing import Dict, Any, Tuple, List
import os

# 서비스는 다른 서비스나 도구, 유틸리티를 '사용'하는 역할을 합니다.
from .profile_service import profile_manager
from src.services.rag_service import search_unified_rag_for_context, search_unified_rag_for_sources

class DataService:
    """
    프로젝트의 모든 데이터 관련 작업을 중앙에서 처리하는 서비스 계층입니다.
    - 프로필 조회
    - 데이터 심층 분석 (캐싱 적용)
    - RAG 검색 (캐싱 적용)
    - Planner를 위한 요약 정보 제공
    """
    def __init__(self):
        self.profile_manager = profile_manager
        self.df_map, self.dataframes = self._load_dataframes()
        print("✅ DataService: 초기화 완료.")

    def get_profile(self, store_id: str) -> Dict[str, Any] | None:
        """ID로 단일 프로필을 안전하게 조회합니다."""
        print(f"--- [DataService] 프로필 조회 요청: {store_id} ---")
        return self.profile_manager.get_profile(store_id)

    def _load_dataframes(self) -> Tuple[Dict[str, pd.DataFrame], List[pd.DataFrame]]:
        """CSV 파일들을 로드하여 딕셔너리와 리스트 형태로 반환합니다."""
        df_map = {}
        data_dir = "./data/"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            print("⚠️ 경고: 'data' 폴더에 분석할 CSV 파일이 없습니다.")
            return {}, []

        for f in csv_files:
            file_path = os.path.join(data_dir, f)
            try: df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError: df = pd.read_csv(file_path, encoding='cp949')
            df_map[f] = df
        
        return df_map, list(df_map.values())

    def get_dataframes(self) -> Tuple[Dict[str, pd.DataFrame], List[pd.DataFrame]]:
        """캐싱된 데이터프레임들을 제공합니다."""
        return self.df_map, self.dataframes

    def get_summary_for_planner(self, store_id: str) -> Dict[str, Any]:
        """
        Planner가 계획 수립에 필요한 핵심 요약 정보만 추출하여 제공합니다.
        LLM이 거대한 JSON 대신 소화하기 쉬운 정보만 받게 되어 판단 정확도가 올라갑니다.
        """
        print(f"--- [DataService] Planner용 프로필 요약 생성: {store_id} ---")
        profile = self.get_profile(store_id)
        if not profile:
            return {"오류": "프로필을 찾을 수 없습니다."}

        core = profile.get("core_data", {})
        basic = core.get("basic_info", {})
        perf = core.get("performance_metrics", {})
        cust = core.get("customer_profile", {})
        ts = core.get("time_series_summary", {})

        summary = {
            "상호명": basic.get("store_name_masked"),
            "업종": basic.get("industry_main"),
            "위치": basic.get("address_district"),
            "업력(개월)": basic.get("business_age_months"),
            "최신_재방문율(%)": cust.get("revisit_rate_latest_percent"),
            "최신_신규고객비율(%)": cust.get("new_customer_rate_latest_percent"),
            "매출_추세(6개월)": ts.get("sales_trend_6m"),
            "재방문율_추세(6개월)": ts.get("revisit_rate_trend_6m"),
            "상권 내 매출 순위(상위 %)": perf.get("sales_rank_in_district_percentile"),
            "주요_고객층": [seg['segment'] for seg in cust.get("top_customer_segments", [])]
        }
        
        return {k: v for k, v in summary.items() if v is not None and v != []}

    @lru_cache(maxsize=256)
    def search_for_context(self, query: str, collection_types: tuple[str, ...] | None = None) -> str:
        """
        Synthesizer와 같이 단순 문자열 컨텍스트가 필요한 경우 사용합니다.
        """
        print(f"--- [DataService] RAG 컨텍스트 검색 실행: {query} ---")
        return search_unified_rag_for_context(query, list(collection_types) if collection_types else None)

    @lru_cache(maxsize=128)
    def search_for_sources(self, query: str, collection_types: tuple[str, ...] | None = None) -> List[Dict[str, Any]]:
        """
        video_recommender와 같이 구조화된 전체 정보가 필요한 경우 사용합니다.
        """
        print(f"--- [DataService] RAG 소스 검색 실행: {query} ---")
        return search_unified_rag_for_sources(query, list(collection_types) if collection_types else None)


# 프로젝트 전역에서 사용할 싱글톤(Singleton) 인스턴스 생성
data_service = DataService()