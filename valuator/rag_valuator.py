"""
과거 딜 데이터를 Chroma 벡터 DB에 임베딩하고,
RAG로 유사한 딜을 검색하여 예상 계약 조건을 생성합니다.
"""
import pandas as pd
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.vector_store import load_existing_store as load_vector_store
from utils.key_manager import key_manager
from config import CHROMA_DB_PATH

def prepare_deal_documents(csv_path: str) -> list[Document]:
    """CSV 파일에서 Document 리스트 생성"""
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = (
            f"질환: {row['disease_area']}, 기술: {row['technology']}, "
            f"단계: {row['phase_at_deal']}, 계약금: ${row['upfront_million']}M, "
            f"마일스톤 총액: ${row['milestone_total_million']}M, 로열티: {row['royalty_rate_percent']}%"
        )
        metadata = {
            "deal_id": row["deal_id"],
            "upfront": row["upfront_million"],
            "milestones": row["milestone_total_million"],
            "royalty": row["royalty_rate_percent"]
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def build_vector_store(documents: list[Document]):
    """문서 임베딩 후 Chroma 벡터 DB 생성 및 저장"""
    emb = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=key_manager.get_next_key()
    )
    Chroma.from_documents(
        documents=documents,
        embedding=emb,
        persist_directory=CHROMA_DB_PATH
    )

def retrieve_similar_deals(
    query_disease: str,
    query_modality: str,
    query_phase: str,
    vectorstore: Chroma,
    top_k: int = 5
) -> list[Document]:
    """질환, 모달리티, 임상 단계로 유사한 과거 딜 검색"""
    query_text = f"질환: {query_disease}, 기술: {query_modality}, 단계: {query_phase}"
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query_text)

def retrieve_similar_deals_with_scores(
    query_disease: str,
    query_modality: str,
    query_phase: str,
    vectorstore: Chroma,
    top_k: int = 5
) -> tuple[list[Document], list[float]]:
    """질환, 모달리티, 임상 단계로 유사한 과거 딜을 검색하고 유사도 점수도 함께 반환"""
    query_text = f"질환: {query_disease}, 기술: {query_modality}, 단계: {query_phase}"
    results = vectorstore.similarity_search_with_score(query_text, k=top_k)
    docs = [r[0] for r in results]
    scores = [r[1] for r in results]  # distance (낮을수록 유사)
    return docs, scores

def predict_deal_structure(
    target_name: str,
    disease: str,
    modality: str,
    phase: str,
    similar_deals: list[Document],
    llm: ChatGoogleGenerativeAI
) -> dict:
    """검색된 유사 딜을 바탕으로 LLM이 계약 조건 예측"""
    context = "\n".join([doc.page_content for doc in similar_deals])

    prompt = PromptTemplate.from_template(
        """당신은 제약 BD 전문가입니다. 아래 과거 라이선스 인 딜 사례를 참고하여,
주어진 타겟 약물의 예상 계약 조건을 예측하세요.

[타겟 정보]
- 후보 이름: {target_name}
- 질환: {disease}
- 기술 모달리티: {modality}
- 현재 임상 단계: {phase}

[참고할 과거 딜]
{context}

분석을 바탕으로 다음 계약 조건을 추정하고, 근거를 간략히 설명하세요.
출력은 JSON 형식으로 제공하세요.
{{
  "upfront_million": <예상 계약금 ($M)>,
  "milestone_total_million": <예상 마일스톤 총액 ($M)>,
  "royalty_rate_percent": <예상 로열티 비율 (%)>,
  "rationale": "<추정 근거>"
}}
JSON만 출력하세요.
"""
    )
    chain = prompt | llm
    result = chain.invoke({
        "target_name": target_name,
        "disease": disease,
        "modality": modality,
        "phase": phase,
        "context": context
    })
    try:
        content = result.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        return json.loads(content)
    except Exception:
        return {
            "upfront_million": None,
            "milestone_total_million": None,
            "royalty_rate_percent": None,
            "rationale": "파싱 실패"
        }
