import os

def load_api_keys():
    """GEMINI_API_KEY_1 ~ GEMINI_API_KEY_8 환경변수를 읽어 리스트로 반환"""
    keys = []
    for i in range(1, 9):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    # 만약 1~8이 없으면 단일 키 폴백
    if not keys:
        single = os.getenv("GEMINI_API_KEY")
        if single:
            keys.append(single)
    return keys

GEMINI_API_KEYS = load_api_keys()
CHROMA_DB_PATH = "./chroma_deal_db"
