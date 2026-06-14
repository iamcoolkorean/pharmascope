from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DB_PATH
from utils.key_manager import key_manager

class RotatingEmbeddings(GoogleGenerativeAIEmbeddings):
    """임베딩 요청마다 다른 키를 사용"""
    def embed_documents(self, texts, **kwargs):
        temp_emb = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key_manager.get_next_key()
)
        )
        return temp_emb.embed_documents(texts, **kwargs)

    def embed_query(self, text, **kwargs):
        temp_emb = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key_manager.get_next_key()
)
        )
        return temp_emb.embed_query(text, **kwargs)

def get_embeddings():
    """순환 임베딩 인스턴스 반환"""
    return RotatingEmbeddings(model="models/embedding-001")

def load_existing_store():
    """저장된 Chroma DB를 로드 (순환 임베딩 사용)"""
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings()
    )
