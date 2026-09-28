"""
PharmaScope FastAPI 백엔드
- 기존 Streamlit 로직을 API로 노출
- 엔드포인트:
  GET  /api/health              → 상태 확인
  POST /api/scan                → 후보 리스트
  POST /api/evaluate            → 개별 후보 딜 예측 + 유사도
  POST /api/report              → BD 리포트 생성
"""
import os
import time
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 기존 모듈 재사용
from scanner.clinical_trials import search_clinical_trials, extract_relevant_fields
from valuator.deal_data_generator import generate_sample_deals
from valuator.rag_valuator import (
    load_vector_store,
    prepare_deal_documents,
    build_vector_store,
    retrieve_similar_deals_with_scores,
    predict_deal_structure,
)
from reporter.report_generator import generate_bd_report
from reporter.financials import get_financial_health
from utils.llm import get_gemini
from config import CHROMA_DB_PATH

# ─────────── FastAPI 앱 초기화 ───────────
app = FastAPI(title="PharmaScope API", version="1.0")

# CORS: Netlify 도메인 허용 (배포 후 실제 도메인으로 좁히는 것을 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 배포 후 ["https://your-app.netlify.app"]로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────── 서버 시작 시 1회 초기화 ───────────
GEMINI = None
VECTOR_STORE = None

@app.on_event("startup")
async def startup_event():
    """서버 부팅 시 LLM과 벡터 DB를 한 번만 로드"""
    global GEMINI, VECTOR_STORE
    GEMINI = get_gemini()

    # 벡터 DB 없으면 자동 생성
    if not os.path.exists(CHROMA_DB_PATH):
        print("🔧 벡터 DB 최초 구축 중...")
        if not os.path.exists("valuator/sample_deals.csv"):
            df = generate_sample_deals()
            df.to_csv("valuator/sample_deals.csv", index=False)
        docs = prepare_deal_documents("valuator/sample_deals.csv")
        build_vector_store(docs)
        print("✅ 벡터 DB 구축 완료")

    VECTOR_STORE = load_vector_store()
    print("🚀 PharmaScope API 준비 완료")

# ─────────── 요청/응답 스키마 ───────────
class ScanRequest(BaseModel):
    disease: str
    modality: Optional[str] = ""
    phase: Optional[str] = "모든 단계"   # "Phase 1" | "Phase 2" | "Phase 3" | "모든 단계"

class EvaluateRequest(BaseModel):
    nct_id: str
    sponsor: str
    disease: str
    modality: Optional[str] = ""
    phase: str
    intervention: Optional[str] = ""

class ReportRequest(BaseModel):
    targets: List[Dict]

# ─────────── 엔드포인트 ───────────
@app.get("/api/health")
def health():
    return {"status": "ok", "vector_store_loaded": VECTOR_STORE is not None}

@app.post("/api/scan")
def scan(req: ScanRequest):
    """
    질환/모달리티/단계로 임상시험 검색. 결과 없으면 자동 조건 완화.
    """
    disease = req.disease.strip()
    modality = (req.modality or "").strip()
    phase = req.phase

    if not disease:
        raise HTTPException(status_code=400, detail="질환명을 입력하세요.")

    cur_modality = modality or None
    cur_phase_api = None if phase == "모든 단계" else phase.replace(" ", "").upper()
    found = False
    df = None

    # 시도 1
    raw = search_clinical_trials(disease, cur_modality, cur_phase_api)
    df = extract_relevant_fields(raw)
    if not df.empty and cur_phase_api:
        df = df[df["임상단계"].str.contains(phase, na=False)]
    if not df.empty:
        found = True

    # 시도 2: 모달리티 제거
    if not found and modality:
        cur_modality = None
        raw = search_clinical_trials(disease, cur_modality, cur_phase_api)
        df = extract_relevant_fields(raw)
        if not df.empty and cur_phase_api:
            df = df[df["임상단계"].str.contains(phase, na=False)]
        if not df.empty:
            found = True

    # 시도 3: 단계 확장
    if not found and phase != "모든 단계":
        cur_phase_api = None
        raw = search_clinical_trials(disease, cur_modality, cur_phase_api)
        df = extract_relevant_fields(raw)
        if not df.empty:
            found = True

    if not found or df is None or df.empty:
        return {"candidates": [], "message": "조건을 완화해도 결과가 없습니다."}

    df = df.head(10)
    # JSON 직렬화를 위해 NaN → None 변환
    df = df.where(df.notna(), None)
    return {
        "candidates": df.to_dict(orient="records"),
        "message": f"{len(df)}건의 후보를 찾았습니다."
    }

@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    """
    개별 후보에 대해 유사 딜 검색 + LLM 계약 조건 예측.
    """
    if VECTOR_STORE is None or GEMINI is None:
        raise HTTPException(status_code=503, detail="서버 초기화가 완료되지 않았습니다.")

    # 모달리티가 비어 있으면 intervention의 첫 단어 사용
    mod_str = req.modality.strip() if req.modality else ""
    if not mod_str:
        inter = req.intervention or ""
        mod_str = inter.split(",")[0].strip()[:30] if inter else "기타"

    target_name = f"{req.nct_id} / {req.sponsor}"

    try:
        similar_docs, scores = retrieve_similar_deals_with_scores(
            req.disease[:50], mod_str, req.phase, VECTOR_STORE
        )
    except Exception as e:
        similar_docs, scores = [], []

    # 유사도 점수 (거리 → 0~100%)
    similarity = 0.0
    if scores:
        similarity = round(max([1/(1+s) for s in scores]) * 100, 1)

    # LLM 예측
    try:
        pred = predict_deal_structure(
            target_name, req.disease[:50], mod_str, req.phase,
            similar_docs, GEMINI
        )
    except Exception as e:
        pred = {"upfront_million": None, "milestone_total_million": None,
                "royalty_rate_percent": None, "rationale": f"오류: {e}"}

    # 재무 안전성
    try:
        fin = get_financial_health(req.sponsor)
    except Exception:
        fin = "재무 정보 조회 실패"

    return {
        "nct_id": req.nct_id,
        "sponsor": req.sponsor,
        "similarity": similarity,
        "prediction": pred,
        "financial_health": fin,
    }

@app.post("/api/report")
def report(req: ReportRequest):
    """
    상위 타겟 리스트를 받아 LLM이 BD 리포트를 생성.
    """
    if GEMINI is None:
        raise HTTPException(status_code=503, detail="LLM 초기화가 완료되지 않았습니다.")
    if not req.targets:
        raise HTTPException(status_code=400, detail="타겟이 없습니다.")
    try:
        md = generate_bd_report(req.targets, GEMINI)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 실패: {e}")
    return {"report_markdown": md}
