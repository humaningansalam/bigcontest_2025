# scripts/populate_rag_single_source.py
import chromadb
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from pathlib import Path

# ===============================================
# 1. 설정 변수 (이 부분만 수정하여 재사용 가능)
# ===============================================

# --- 데이터베이스 설정 (로컬 파일 시스템) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
CHROMA_DB_PATH = str(DATA_PATH / 'chroma_db')

# --- 이 스크립트로 처리할 데이터 소스 정보 ---

# ✅ 1. 데이터를 저장할 컬렉션 이름
# (예: "strategies_and_theories", "practical_guides", "market_trends_and_data")
COLLECTION_NAME = "market_trends_and_data"

# ✅ 2. 텍스트 파일이 저장된 폴더 경로
# (예: DATA_PATH / "rag_sources/academic_materials")
DATA_FOLDER_PATH = DATA_PATH / "rag_sources/industry_trends/Nasmedia"

# ✅ 3. 이 데이터 소스의 기본 메타데이터
BASE_METADATA = {
    "source_group": "industry_trends",
    "source_name": "Nasmedia",
    "base_url": "https://www.nasmedia.co.kr/" # 문서별 URL이 없다면 기본 URL 사용
}

# ===============================================
# 2. ChromaDB 클라이언트 및 텍스트 분할기 준비
# ===============================================

# --- ChromaDB 클라이언트 연결 (로컬) ---
print(f"ChromaDB 데이터베이스를 '{CHROMA_DB_PATH}' 경로에 설정합니다...")
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
except Exception as e:
    print(f"❌ ChromaDB 클라이언트 생성 실패: {e}")
    exit()

# --- 텍스트 분할기(Chunker) 준비 ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

# ===============================================
# 3. 메인 실행 로직
# ===============================================

def main():
    print("\n" + "="*50)
    print(f"RAG 데이터 처리 시작: '{DATA_FOLDER_PATH.name}' -> '{COLLECTION_NAME}'")

    # --- 1. 컬렉션 생성 또는 가져오기 ---
    print(f"'{COLLECTION_NAME}' 컬렉션을 준비합니다...")
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"✅ 컬렉션 준비 완료. 현재 문서 수: {collection.count()}")
    except Exception as e:
        print(f"❌ 컬렉션 '{COLLECTION_NAME}' 준비 실패: {e}")
        return

    # --- 2. 데이터 폴더에서 텍스트 파일 목록 가져오기 ---
    print(f"'{DATA_FOLDER_PATH}' 폴더에서 .txt 파일을 검색합니다...")
    try:
        txt_files = [f for f in os.listdir(DATA_FOLDER_PATH) if f.endswith('.txt')]
        if not txt_files:
            print(f"⚠️ 경고: '{DATA_FOLDER_PATH}' 폴더에 .txt 파일이 없습니다. 작업을 종료합니다.")
            return
        print(f"총 {len(txt_files)}개의 파일을 찾았습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{DATA_FOLDER_PATH}' 폴더를 찾을 수 없습니다.")
        return

    # --- 3. 파일 처리 및 데이터 준비 ---
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for filename in tqdm(txt_files, desc="파일 처리 중"):
        file_path = os.path.join(DATA_FOLDER_PATH, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                document_title = filename.replace('.txt', '')
                
                # 기본 메타데이터 복사 후 개별 청크 정보 추가
                metadata = BASE_METADATA.copy()
                metadata["document_title"] = document_title
                # metadata["url"] = BASE_METADATA["base_url"] + document_title # 필요시 URL 규칙 정의
                
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                # 각 청크에 고유 ID 부여 (컬렉션명 + 파일명 + 청크 번호)
                all_ids.append(f"{COLLECTION_NAME}_{filename}_{i}")

        except Exception as e:
            print(f"\n⚠️ 파일 '{filename}' 처리 중 오류 발생: {e}")

    # --- 4. ChromaDB에 일괄 업로드 (upsert) ---
    if all_chunks:
        print(f"\n총 {len(all_chunks)}개의 청크를 ChromaDB에 업로드합니다...")
        
        try:
            collection.upsert(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print("✅ 업로드 완료!")
            print(f"'{COLLECTION_NAME}' 컬렉션의 최종 문서 수: {collection.count()}")
        except Exception as e:
            print(f"❌ 업로드 실패: {e}")
    else:
        print("업로드할 데이터가 없습니다.")

    print("="*50 + "\n")

if __name__ == "__main__":
    main()