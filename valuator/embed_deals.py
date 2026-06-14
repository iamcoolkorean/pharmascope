"""
과거 딜 데이터 CSV를 읽어 벡터 DB로 변환합니다.
프로젝트 루트에서 'python valuator/embed_deals.py'로 실행합니다.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from valuator.deal_data_generator import generate_sample_deals
from valuator.rag_valuator import prepare_deal_documents, build_vector_store

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "sample_deals.csv")
    if not os.path.exists(csv_path):
        df = generate_sample_deals()
        df.to_csv(csv_path, index=False)
        print(f"{csv_path} 생성됨")
    else:
        print(f"{csv_path} 이미 존재함")

    docs = prepare_deal_documents(csv_path)
    build_vector_store(docs)
    print("Chroma 벡터 DB 저장 완료 → chroma_deal_db/")
