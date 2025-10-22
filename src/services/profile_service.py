# services/profile_service.py

import threading
import json
from .rag_service import get_chroma_client 
from src.utils.errors import create_tool_error

# 이 클래스의 인스턴스가 프로필 DB에 대한 유일한 접근 지점이 됩니다.
class ProfileManager:
    """
    프로필 데이터에 대한 모든 CRUD(Create, Read, Update, Delete) 작업을 중앙에서 관리하고,
    스레드 안전성(Thread-Safety)을 보장하는 클래스.
    """
    def __init__(self):
        """
        초기화 시 스레드 잠금(Lock)과 ChromaDB 컬렉션을 설정합니다.
        """
        self._lock = threading.Lock()
        self.client = get_chroma_client()
        
        if self.client:
            try:
                # store_profiles 컬렉션이 존재한다고 가정합니다.
                self.collection = self.client.get_collection(name="store_profiles")
                print("✅ ProfileManager: 'store_profiles' 컬렉션에 성공적으로 연결되었습니다.")
            except Exception as e:
                print(f"❌ ProfileManager: 'store_profiles' 컬렉션을 가져오는 데 실패했습니다: {e}")
                self.collection = None
        else:
            self.collection = None

    def get_profile(self, store_id: str) -> dict | None:
        """ID로 단일 프로필을 안전하게 조회합니다."""
        if not self.collection:
            print("⚠️ ProfileManager: 컬렉션이 없어 프로필 조회를 건너뜁니다.")
            return None
            
        print(f"--- 락(Lock) 획득 시도 (작업: get_profile, ID: {store_id}) ---")
        with self._lock: 
            print(f"--- 락(Lock) 획득 성공. 프로필 조회 시작 ---")
            try:
                result = self.collection.get(ids=[str(store_id)], include=["metadatas"])
                if result and result['ids']:
                    profile_str = result['metadatas'][0]['profile_json']
                    return json.loads(profile_str)
                else:
                    return None
            except Exception as e:
                print(f"⚠️ 프로필 조회 중 DB 오류 발생 (ID: {store_id}): {e}")
                return None
            finally:
                print(f"--- 락(Lock) 해제 (작업: get_profile, ID: {store_id}) ---")

    def update_profile(self, store_id: str, section: str, key: str, data_to_update: dict) -> bool:
        """특정 프로필의 일부를 안전하게 업데이트합니다."""
        if not self.collection:
            print("⚠️ ProfileManager: 컬렉션이 없어 프로필 업데이트를 건너뜁니다.")
            return False

        print(f"--- 락(Lock) 획득 시도 (작업: update_profile, ID: {store_id}) ---")
        with self._lock: 
            print(f"--- 락(Lock) 획득 성공. 프로필 업데이트 시작 ---")
            try:
                # 1. 먼저 현재 데이터를 읽어옵니다.
                original_data = self.collection.get(ids=[str(store_id)], include=["metadatas", "documents"])
                if not original_data['ids']:
                    print(f"⚠️ 업데이트할 프로필을 찾을 수 없음 (ID: {store_id})")
                    return False
                
                profile = json.loads(original_data['metadatas'][0]['profile_json'])
                existing_doc = original_data['documents'][0]
                
                # 2. 메모리에서 데이터를 수정한 뒤
                profile.setdefault(section, {}).setdefault(key, {}).update(data_to_update)
                
                # 3. DB에 다시 씁니다 (upsert).
                self.collection.upsert(
                    ids=[str(store_id)],
                    documents=[existing_doc], # document는 변경하지 않음
                    metadatas=[{"profile_json": json.dumps(profile, ensure_ascii=False)}]
                )
                return True
            except Exception as e:
                print(f"⚠️ 프로필 업데이트 중 DB 오류 발생 (ID: {store_id}): {e}")
                return False
            finally:
                print(f"--- 락(Lock) 해제 (작업: update_profile, ID: {store_id}) ---")

# 프로젝트 전역에서 사용할 싱글톤(Singleton) 인스턴스
profile_manager = ProfileManager()