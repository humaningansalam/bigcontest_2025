# scripts/populate_chromadb.py
import chromadb
import json
import os
from tqdm import tqdm
from pathlib import Path

# ===============================================
# 1. 설정 변수
# ===============================================

# 프로젝트 루트 경로를 기준으로 동적으로 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
PROFILE_JSON_PATH = DATA_PATH / 'store_profiles.json'

# ChromaDB 데이터를 저장할 로컬 디렉토리 경로
CHROMA_DB_PATH = str(DATA_PATH / 'chroma_db')

# 컬렉션 이름 설정
COLLECTION_NAME = "store_profiles"
BATCH_SIZE = 100

# ===============================================
# 2. 핵심 함수: 검색용 문서 생성기 (고도화 버전)
# ===============================================

def create_document_from_profile(profile: dict) -> str:
    """
    고도화된 프로필 JSON 객체를 받아 ChromaDB 검색에 사용할
    핵심 요약 텍스트 문서를 생성합니다.
    """
    try:
        core = profile.get("core_data", {})
        basic = core.get("basic_info", {})
        perf = core.get("performance_metrics", {})
        customer = core.get("customer_profile", {})
        ts_summary = core.get("time_series_summary", {})
        extended = profile.get("extended_features", {})

        # --- 검색 품질을 높이기 위한 정보 조합 ---

        # 1. 기본 정보
        info_parts = [
            f"가맹점명 {basic.get('store_name_masked', '')}",
            f"위치 {basic.get('address_district', '')} {basic.get('commercial_district', '')}",
            f"업종 {basic.get('industry_main', '')}",
            f"업력 {basic.get('business_age_months', 0)}개월",
            "프랜차이즈" if extended.get('is_franchise') else "개인점포",
            "배달 서비스 운영" if extended.get('has_delivery_service') else ""
        ]

        # 2. 성과 요약
        perf_parts = [
            f"최신 매출은 {perf.get('sales_amount_band', '알 수 없음')} 구간",
            f"상권 내 매출 순위는 상위 {perf.get('sales_rank_in_district_percentile', 100)}%",
            f"최근 6개월 매출 추세는 {ts_summary.get('sales_trend_6m', '안정')}",
        ]

        # 3. 고객 분석 요약 (고도화된 정보 활용)
        customer_parts = [
            f"최신 재방문율 {customer.get('revisit_rate_latest_percent', 0)}%",
            f"최근 6개월 재방문율 추세는 {ts_summary.get('revisit_rate_trend_6m', '안정')}",
        ]
        top_segments = customer.get('top_customer_segments', [])
        if top_segments:
            # 상위 고객층 정보를 문장으로 생성
            top_segments_str = ", ".join([f"{seg['segment']}({seg['ratio']}%)" for seg in top_segments])
            customer_parts.append(f"주요 고객층은 {top_segments_str}")

        # 모든 정보를 조합하여 하나의 긴 문장으로 만듭니다.
        # 마침표(.)로 구분하여 문맥적 의미를 부여합니다.
        document_text = ". ".join(filter(None, info_parts + perf_parts + customer_parts))
        
        return document_text

    except Exception as e:
        print(f"Warning: 프로필 ID {profile.get('profile_id')} 처리 중 오류 발생 - {e}")
        return profile.get('profile_id', '') # 오류 발생 시 ID만이라도 반환

# ===============================================
# 3. 메인 실행 로직
# ===============================================

def main():
    print("ChromaDB 데이터 적재를 시작합니다 (로컬 파일 기반)...")

    # --- 1. ChromaDB 클라이언트 및 컬렉션 설정 ---
    print(f"ChromaDB 데이터베이스를 '{CHROMA_DB_PATH}' 경로에 설정합니다.")
    # 로컬 파일 시스템에 데이터를 저장하는 PersistentClient 사용
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 기존 컬렉션이 있다면 삭제하고 새로 생성 (데이터 일관성을 위해)
    try:
        if COLLECTION_NAME in [c.name for c in client.list_collections()]:
            print(f"기존 '{COLLECTION_NAME}' 컬렉션을 삭제합니다.")
            client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"컬렉션 확인/삭제 중 오류 발생: {e}")
        print("ChromaDB 서버가 다른 곳에서 실행 중일 수 있습니다. 확인 후 다시 시도해주세요.")
        return
        
    print(f"'{COLLECTION_NAME}' 컬렉션을 새로 생성합니다.")
    collection = client.create_collection(name=COLLECTION_NAME)

    # --- 2. 프로필 데이터 로드 ---
    if not os.path.exists(PROFILE_JSON_PATH):
        print(f"오류: '{PROFILE_JSON_PATH}' 파일을 찾을 수 없습니다.")
        print("먼저 `scripts/create_profiles.py`를 실행하여 프로필 파일을 생성해주세요.")
        return

    print(f"'{PROFILE_JSON_PATH}' 파일에서 프로필 데이터를 로드합니다.")
    with open(PROFILE_JSON_PATH, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    
    print(f"총 {len(profiles)}개의 프로필을 처리합니다.")

    # --- 3. 데이터 준비 ---
    documents = []
    metadatas = []
    ids = []

    for profile in tqdm(profiles, desc="프로필 데이터 준비 중"):
        if not profile.get('profile_id'):
            continue

        # 임베딩할 요약 텍스트(문서) 생성
        doc = create_document_from_profile(profile)
        documents.append(doc)
        
        # 검색 결과로 원본 전체 JSON을 참조할 수 있도록 메타데이터에 저장
        meta = {"profile_json": json.dumps(profile, ensure_ascii=False)}
        metadatas.append(meta)
        
        # 각 문서의 고유 ID 설정
        ids.append(str(profile['profile_id'])) # ID는 문자열이어야 함

    # --- 4. ChromaDB에 데이터 일괄 적재 ---
    total_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"데이터를 {BATCH_SIZE}개씩 나누어 총 {total_batches}개의 배치로 적재합니다.")
    
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="ChromaDB에 적재 중"):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_documents = documents[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]
        
        # add 메서드를 사용하여 데이터를 컬렉션에 추가
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        
    print("\nChromaDB 데이터 적재 완료!")
    print(f"'{COLLECTION_NAME}' 컬렉션에 총 {collection.count()}개의 프로필이 저장되었습니다.")
    print(f"데이터베이스는 '{CHROMA_DB_PATH}' 디렉토리에 저장되었습니다.")
    
    # --- 5. 테스트 쿼리 ---
    print("\n--- 테스트 쿼리 실행 ---")
    test_query_text = "성동구에서 재방문율이 낮은 카페"
    print(f"질의: '{test_query_text}'")
    
    try:
        results = collection.query(
            query_texts=[test_query_text],
            n_results=3 # 상위 3개 결과 반환
        )
        
        print("가장 유사한 가맹점 TOP 3:")
        if not results['ids'] or not results['ids'][0]:
            print("  결과 없음")
        else:
            for i, doc_id in enumerate(results['ids'][0]):
                print(f"  {i+1}. ID: {doc_id}")
                print(f"     유사도 점수(Distance): {results['distances'][0][i]:.4f}")
                print(f"     요약 문서: {results['documents'][0][i]}")
            
    except Exception as e:
        print(f"테스트 쿼리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()