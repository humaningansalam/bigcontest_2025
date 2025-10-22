# check_local_db.py
import chromadb
from pathlib import Path
import json

# --- 설정 ---
# 프로젝트 루트를 기준으로 DB 경로를 설정합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
LOCAL_CHROMA_DB_PATH = str(DATA_PATH / 'chroma_db')

# 확인할 컬렉션 목록 (migrate 스크립트와 동일하게)
COLLECTIONS_TO_CHECK = [
    "store_profiles",
    "strategies_and_theories",
    "practical_guides",
    "market_trends_and_data"
]

def main():
    print("="*50)
    print("ChromaDB 로컬 데이터베이스 상태 확인을 시작합니다.")
    print(f"DB 경로: {LOCAL_CHROMA_DB_PATH}")
    print("="*50)

    # --- 1. DB 클라이언트 연결 ---
    try:
        client = chromadb.PersistentClient(path=LOCAL_CHROMA_DB_PATH)
        print("✅ 로컬 DB 클라이언트 연결 성공!")
    except Exception as e:
        print(f"❌ 로컬 DB 클라이언트 연결 실패: {e}")
        print("  - 마이그레이션 스크립트가 정상적으로 실행되었는지 확인해주세요.")
        print(f"  - '{LOCAL_CHROMA_DB_PATH}' 경로가 존재하는지 확인해주세요.")
        return

    # --- 2. 전체 컬렉션 목록 확인 ---
    try:
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        print(f"\n[INFO] 현재 DB에 존재하는 컬렉션: {collection_names}")
    except Exception as e:
        print(f"❌ 컬렉션 목록을 가져오는 데 실패했습니다: {e}")
        return

    # --- 3. 각 컬렉션별 상세 정보 확인 ---
    all_collections_ok = True
    for collection_name in COLLECTIONS_TO_CHECK:
        print("\n" + "-"*40)
        print(f"🔍 '{collection_name}' 컬렉션 확인 중...")

        if collection_name not in collection_names:
            print(f"  ❌ 오류: '{collection_name}' 컬렉션이 DB에 존재하지 않습니다!")
            all_collections_ok = False
            continue

        try:
            collection = client.get_collection(name=collection_name)
            
            # 3-1. 문서 수 확인
            count = collection.count()
            if count > 0:
                print(f"  ✅ 문서 수: {count} 개")
            else:
                print(f"  ⚠️ 경고: 문서가 0개입니다. 데이터가 제대로 적재되지 않았을 수 있습니다.")
                all_collections_ok = False
                continue

            # 3-2. 샘플 데이터 확인 (처음 2개)
            print("  - 샘플 데이터 (상위 2개):")
            sample_data = collection.peek(limit=2)
            
            for i in range(len(sample_data['ids'])):
                doc_id = sample_data['ids'][i]
                document = sample_data['documents'][i]
                metadata = sample_data['metadatas'][i]
                
                print(f"\n    [샘플 {i+1}]")
                print(f"    - ID: {doc_id}")
                
                # 'store_profiles' 컬렉션은 메타데이터의 JSON을 예쁘게 출력
                if collection_name == 'store_profiles' and 'profile_json' in metadata:
                    profile = json.loads(metadata['profile_json'])
                    store_name = profile.get('core_data', {}).get('basic_info', {}).get('store_name_masked', 'N/A')
                    print(f"    - 가맹점명: {store_name}")
                    print("    - 메타데이터: (프로필 JSON, 정상)")
                else:
                    print(f"    - 메타데이터: {metadata}")
                
                print(f"    - 문서 내용 (앞 50자): {document[:50]}...")

        except Exception as e:
            print(f"  ❌ '{collection_name}' 컬렉션 확인 중 오류 발생: {e}")
            all_collections_ok = False
    
    # --- 4. 최종 결과 ---
    print("\n" + "="*50)
    if all_collections_ok:
        print("🎉 최종 확인 완료: 모든 주요 컬렉션이 정상적으로 로드된 것으로 보입니다.")
    else:
        print("🔥 최종 확인 결과: 일부 컬렉션에서 문제가 발견되었습니다. 위의 로그를 확인해주세요.")
    print("="*50)


if __name__ == "__main__":
    main()