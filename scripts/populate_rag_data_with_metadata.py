# scripts/populate_rag_with_metadata.py

import chromadb
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from pathlib import Path

# ===============================================
# 1. 설정 변수
# ===============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
CHROMA_DB_PATH = str(DATA_PATH / 'chroma_db')

COLLECTION_NAME = "case_studies_and_policies"
DATA_FOLDER_PATH = DATA_PATH / "rag_sources/서울_지원사업_txt"

# ===============================================
# 2. 유틸리티 함수 (핵심 수정 부분)
# ===============================================

def parse_metadata_from_file(content: str) -> tuple[dict, str]:
    """
    파일 내용에서 메타데이터와 본문을 분리하여 파싱합니다.
    """
    metadata = {}
    
    # '### 사업개요'를 기준으로 메타데이터와 본문 분리
    parts = re.split(r'\n###\s*사업개요', content, 1)
    metadata_section = parts[0]
    # 본문에는 '사업개요' 제목을 다시 포함시켜 의미 명확화
    body_section = "### 사업개요\n" + parts[1] if len(parts) > 1 else ""

    # 메타데이터 섹션에서 key-value 파싱
    # 예: "공고명: [서울] ..." -> key='공고명', value='[서울] ...'
    for line in metadata_section.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            # 메타데이터 key는 검색 필터링을 위해 영문 소문자로 변환
            key_clean = key.strip().lower().replace(' ', '_')
            metadata[key_clean] = value.strip()

    return metadata, body_section

# ===============================================
# 3. ChromaDB 클라이언트 및 텍스트 분할기 준비
# ===============================================

print(f"ChromaDB 데이터베이스를 '{CHROMA_DB_PATH}' 경로에 설정합니다...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", "### ", ". ", " ", ""],
    length_function=len,
)

# ===============================================
# 4. 메인 실행 로직
# ===============================================

def main():
    print("\n" + "="*50)
    print(f"RAG 데이터 처리(메타데이터 포함) 시작: '{DATA_FOLDER_PATH.name}' -> '{COLLECTION_NAME}'")

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"✅ 컬렉션 준비 완료. 현재 문서 수: {collection.count()}")

    try:
        txt_files = [f for f in os.listdir(DATA_FOLDER_PATH) if f.endswith('.txt')]
        if not txt_files:
            print(f"⚠️ 경고: '{DATA_FOLDER_PATH}' 폴더에 .txt 파일이 없습니다.")
            return
    except FileNotFoundError:
        print(f"❌ 오류: '{DATA_FOLDER_PATH}' 폴더를 찾을 수 없습니다.")
        return

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for filename in tqdm(txt_files, desc="파일 처리 중"):
        file_path = DATA_FOLDER_PATH / filename
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # [수정] 파일 내용에서 메타데이터와 본문을 파싱
            file_metadata, body_content = parse_metadata_from_file(content)
            
            # 본문만 Chunking 대상으로 함
            chunks = text_splitter.split_text(body_content)

            for i, chunk in enumerate(chunks):
                # 파일에서 파싱한 메타데이터를 기반으로 각 청크의 메타데이터 생성
                chunk_metadata = file_metadata.copy()
                chunk_metadata["source_file"] = filename
                
                all_chunks.append(chunk)
                all_metadatas.append(chunk_metadata)
                all_ids.append(f"{COLLECTION_NAME}_{filename}_{i}")

        except Exception as e:
            print(f"\n⚠️ 파일 '{filename}' 처리 중 오류 발생: {e}")

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

if __name__ == "__main__":
    main()