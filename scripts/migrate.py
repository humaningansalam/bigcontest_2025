# scripts/migrate_db_to_local.py
import chromadb
import os
from tqdm import tqdm
from pathlib import Path

# ===============================================
# 1. 설정 변수
# ===============================================

# --- 원본: 외부 서버 정보 ---
REMOTE_CHROMA_HOST = os.getenv("CHROMA_HOST", "us.duckdns.org")
REMOTE_CHROMA_PORT = int(os.getenv("CHROMA_PORT", 19905))

# --- 대상: 로컬 DB 경로 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
LOCAL_CHROMA_DB_PATH = str(DATA_PATH / 'chroma_db')

# --- 마이그레이션할 컬렉션 목록 ---
# 외부 서버에 존재하는 컬렉션 이름들을 모두 적어줍니다.
COLLECTIONS_TO_MIGRATE = [
    "strategies_and_theories",
    "practical_guides",
    "market_trends_and_data"
]

# 한 번에 가져올 데이터 개수 (메모리 사용량에 따라 조절)
BATCH_SIZE = 500

# ===============================================
# 2. 메인 실행 로직
# ===============================================

def main():
    print("ChromaDB 마이그레이션을 시작합니다: 외부 서버 -> 로컬 파일")

    # --- 1. 원본(외부 서버) 클라이언트 연결 ---
    print(f"\n--- 1. 원본 서버({REMOTE_CHROMA_HOST}:{REMOTE_CHROMA_PORT})에 연결 ---")
    try:
        remote_client = chromadb.HttpClient(host=REMOTE_CHROMA_HOST, port=REMOTE_CHROMA_PORT)
        remote_client.heartbeat()
        print("✅ 원본 서버 연결 성공.")
    except Exception as e:
        print(f"❌ 원본 서버 연결 실패: {e}")
        return

    # --- 2. 대상(로컬 DB) 클라이언트 생성 ---
    print(f"\n--- 2. 대상 로컬 DB('{LOCAL_CHROMA_DB_PATH}') 준비 ---")
    try:
        # 로컬 DB 폴더가 없다면 생성
        os.makedirs(LOCAL_CHROMA_DB_PATH, exist_ok=True)
        local_client = chromadb.PersistentClient(path=LOCAL_CHROMA_DB_PATH)
        print("✅ 대상 로컬 DB 준비 완료.")
    except Exception as e:
        print(f"❌ 대상 로컬 DB 생성 실패: {e}")
        return

    # --- 3. 각 컬렉션별 데이터 마이그레이션 ---
    for collection_name in COLLECTIONS_TO_MIGRATE:
        print("\n" + "="*50)
        print(f"'{collection_name}' 컬렉션 마이그레이션 시작...")

        # --- 3-1. 원본 컬렉션에서 데이터 가져오기 ---
        try:
            print("  - 원본 데이터 가져오는 중...")
            remote_collection = remote_client.get_collection(name=collection_name)
            
            # get() 메서드로 모든 데이터를 가져옴 (include 모든 항목)
            # count()로 전체 개수를 가져와서 offset으로 순회할 수 있지만,
            # 데이터가 아주 많지 않다면 get()으로 한 번에 가져오는 것이 간단함.
            count = remote_collection.count()
            if count == 0:
                print(f"  - '{collection_name}' 컬렉션에 데이터가 없습니다. 건너뜁니다.")
                continue

            # 데이터가 매우 많을 경우를 대비한 배치 처리
            all_data = {"ids": [], "documents": [], "metadatas": []}
            for offset in tqdm(range(0, count, BATCH_SIZE), desc="  - 데이터 다운로드 중"):
                batch = remote_collection.get(
                    limit=BATCH_SIZE,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                all_data["ids"].extend(batch["ids"])
                all_data["documents"].extend(batch["documents"])
                all_data["metadatas"].extend(batch["metadatas"])

            print(f"  - 총 {len(all_data['ids'])}개의 문서를 가져왔습니다.")

        except Exception as e:
            print(f"❌ 원본 컬렉션 '{collection_name}'에서 데이터 가져오기 실패: {e}")
            continue

        # --- 3-2. 대상 컬렉션에 데이터 쓰기 ---
        try:
            print("  - 대상 로컬 DB에 데이터 쓰는 중...")
            # 기존에 로컬에 동일 이름의 컬렉션이 있다면 삭제하고 새로 생성
            if collection_name in [c.name for c in local_client.list_collections()]:
                print(f"  - 기존 로컬 '{collection_name}' 컬렉션을 삭제합니다.")
                local_client.delete_collection(name=collection_name)
            
            local_collection = local_client.create_collection(name=collection_name)

            # 가져온 데이터를 배치 단위로 로컬 DB에 upsert
            for i in tqdm(range(0, len(all_data['ids']), BATCH_SIZE), desc="  - 로컬 DB에 업로드 중"):
                local_collection.upsert(
                    ids=all_data['ids'][i:i+BATCH_SIZE],
                    documents=all_data['documents'][i:i+BATCH_SIZE],
                    metadatas=all_data['metadatas'][i:i+BATCH_SIZE]
                )
            
            print(f"✅ '{collection_name}' 컬렉션 마이그레이션 성공! (총 {local_collection.count()}개 문서)")

        except Exception as e:
            print(f"❌ 대상 로컬 DB에 '{collection_name}' 컬렉션 쓰기 실패: {e}")
            
    print("\n" + "="*50)
    print("모든 컬렉션의 마이그레이션 작업이 완료되었습니다.")
    print(f"이제 로컬 DB 경로 '{LOCAL_CHROMA_DB_PATH}'를 사용하여 개발을 진행할 수 있습니다.")


if __name__ == "__main__":
    main()