"""
Gemini API 키를 순환(round-robin)으로 제공하는 관리자.
멀티스레드 환경을 고려하여 Lock으로 보호.
"""
import threading
from config import GEMINI_API_KEYS

class GeminiKeyManager:
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("최소 1개 이상의 Gemini API 키가 필요합니다.")
        self._keys = keys
        self._index = 0
        self._lock = threading.Lock()

    def get_next_key(self) -> str:
        """현재 키를 반환하고 인덱스를 다음으로 이동"""
        with self._lock:
            key = self._keys[self._index]
            self._index = (self._index + 1) % len(self._keys)
            return key

# 전역 싱글톤
key_manager = GeminiKeyManager(GEMINI_API_KEYS)
