# src/services/rag_service.py

import chromadb
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# 1. 설정 변수


# --- 옵션 1: 로컬 파일 기반 ChromaDB (기본값) ---
# 프로젝트 루트를 기준으로 DB 경로를 동적으로 설정합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
CHROMA_DB_PATH = str(DATA_PATH / 'chroma_db')

# --- 옵션 2: 외부 서버 기반 ChromaDB (주석 처리) ---
# 나중에 외부 서버를 사용하려면 아래 주석을 풀고, 위 CHROMA_DB_PATH를 주석 처리하세요.
# CHROMA_HOST = os.getenv("CHROMA_HOST", "us-hun.duckdns.org")
# CHROMA_PORT = int(os.getenv("CHROMA_PORT", 19905))


# --- 공통 설정 ---
# 검색 대상 컬렉션 이름들을 중앙에서 관리합니다.
COLLECTIONS = {
    "profile": "store_profiles",
    "strategy": "strategies_and_theories",
    "guide": "practical_guides",
    "trend": "market_trends_and_data",
    "video": "learning_videos",
    "case": "case_studies_and_policies"
}

# 클라이언트 객체를 캐싱하기 위한 전역 변수
_client = None


#  ChromaDB 클라이언트 생성 함수


def get_chroma_client():
    """
    ChromaDB 클라이언트를 초기화하고 연결 상태를 확인하는 싱글톤 함수.
    설정에 따라 로컬 클라이언트 또는 원격 클라이언트를 생성합니다.
    """
    global _client
    if _client is not None:
        return _client

    try:
        # --- 로컬 클라이언트 생성 (기본) ---
        print(f"로컬 ChromaDB에 연결 시도 중... (경로: {CHROMA_DB_PATH})")
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        # 로컬 클라이언트는 컬렉션 목록 조회로 연결 상태를 확인합니다.
        _client.list_collections() 
        print("✅ 통합 RAG: 로컬 ChromaDB에 성공적으로 연결되었습니다.")

        # --- 외부 서버 클라이언트 생성 (주석 처리) ---
        # print(f"외부 ChromaDB 서버에 연결 시도 중... (주소: {CHROMA_HOST}:{CHROMA_PORT})")
        # _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # _client.heartbeat()
        # print("✅ 통합 RAG: 외부 ChromaDB 서버에 성공적으로 연결되었습니다.")

    except Exception as e:
        print(f"❌ 통합 RAG: ChromaDB 연결에 실패했습니다: {e}")
        _client = None
    
    return _client

def _perform_search(client, query: str, collection_types: list[str], n_results: int) -> list[dict]:
    """ 실제 ChromaDB 검색을 수행하고 원본 결과 리스트를 반환합니다."""
    all_results = []
    seen_docs = set()

    if collection_types is None:
        collection_types = ["strategy", "guide", "trend", "case", "local"] 

    for ctype in collection_types:
        collection_name = COLLECTIONS.get(ctype)
        if not collection_name: continue
        try:
            collection = client.get_collection(name=collection_name)
            results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
            if results and results['documents']:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    if doc not in seen_docs:
                        all_results.append({'doc': doc, 'meta': meta, 'collection': ctype})
                        seen_docs.add(doc)
        except Exception as e:
            print(f"⚠️ RAG 검색 중 '{collection_name}' 컬렉션에서 오류: {e}")
    return all_results

def search_unified_rag_for_context(query: str, collection_types: list[str] = None, n_results: int = 3) -> str:
    """
    LLM 프롬프트에 넣기 좋은 '문자열 컨텍스트'만 생성하여 반환합니다.
    """
    client = get_chroma_client()
    if not client: return "RAG 시스템에 연결할 수 없습니다."
    
    all_results = _perform_search(client, query, collection_types, n_results)
    if not all_results: return "관련 정보를 찾을 수 없습니다."

    context_str = ""
    for i, res in enumerate(all_results[:5]):
        meta = res.get('meta', {})
        doc = res.get('doc', '')
        ctype = res.get('collection', 'unknown')
        title = meta.get('title', meta.get('document_title', 'N/A'))
        source_info = f"[출처:{i+1}|{ctype.upper()}] 제목: {title}"
        context_str += f"{source_info}\n내용 요약: {doc[:200]}...\n\n"
    return context_str

def search_unified_rag_for_sources(query: str, collection_types: list[str] = None, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    메타데이터와 본문(content)을 모두 포함한 '구조화된 소스 리스트'를 반환합니다.
    """
    client = get_chroma_client()
    if not client: return []
    
    all_results = _perform_search(client, query, collection_types, n_results)
    if not all_results: return []

    sources_list = []
    for res in all_results[:5]:
        source_item = res.get('meta', {}).copy()
        source_item['content'] = res.get('doc', '')
        sources_list.append(source_item)
    return sources_list