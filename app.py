"""
PharmaScope Streamlit 프로토타입
- 사용자 입력 → Target Scanner → Deal Valuator → Report Generator
"""
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from scanner.clinical_trials import search_clinical_trials, extract_relevant_fields
from valuator.rag_valuator import (
    load_vector_store,
    retrieve_similar_deals,
    predict_deal_structure
)
from reporter.report_generator import generate_bd_report
from utils.llm import get_gemini

# .env 파일이 있을 경우 로드 (Codespaces 시크릿은 자동 환경변수)
load_dotenv()

st.set_page_config(page_title="PharmaScope", layout="wide")
st.title("💊 PharmaScope – AI 라이선스 인 타겟 발굴 에이전트")

# 세션 초기화
if "gemini" not in st.session_state:
    st.session_state.gemini = get_gemini()

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_vector_store()
    except Exception as e:
        st.error(f"벡터 DB를 불러올 수 없습니다. 먼저 'python valuator/embed_deals.py'를 실행하세요.\n{str(e)}")
        st.stop()

# 사이드바 입력 폼
with st.sidebar:
    st.header("🔍 탐색 조건")
    disease = st.text_input("질환명 (예: 비알콜성 지방간염)", value="NASH")
    modality = st.text_input("기술 모달리티 (예: siRNA, ADC)", value="siRNA")
    phase = st.selectbox("임상 단계", ["Phase 1", "Phase 2", "Phase 3", "모든 단계"])
    st.markdown("---")
    search_btn = st.button("타겟 스캔 시작", type="primary")

if search_btn:
    if not disease:
        st.warning("질환명을 입력해주세요.")
    else:
        # 1. 타겟 스캐너
        with st.spinner("임상시험 DB 검색 중..."):
            phase_api = None if phase == "모든 단계" else phase.replace(" ", "").upper()
            raw_studies = search_clinical_trials(disease, modality, phase_api)
            df_all = extract_relevant_fields(raw_studies)
            if df_all.empty:
                st.error("조건에 맞는 임상시험이 없습니다.")
                st.stop()
            if phase != "모든 단계":
                df_all = df_all[df_all["임상단계"].str.contains(phase, na=False)]
            df_all = df_all.head(10)
            st.success(f"총 {len(df_all)} 건의 후보를 찾았습니다.")
            st.dataframe(df_all, use_container_width=True)

        # 2. Deal Valuator
        st.subheader("💰 유사 과거 딜 분석 & 예상 계약 조건")
        predictions = []
        for _, row in df_all.iterrows():
            disease_str = row["질환"][:50]
            modality_str = modality if modality else row["중재법"]
            phase_str = row["임상단계"]
            target_name = row["NCT_ID"] + " / " + row["스폰서"]

            similar = retrieve_similar_deals(
                disease_str, modality_str, phase_str,
                st.session_state.vector_store
            )
            deal_pred = predict_deal_structure(
                target_name, disease_str, modality_str, phase_str,
                similar, st.session_state.gemini
            )
            predictions.append(deal_pred)

        display_df = df_all.copy()
        display_df["예상 계약금($M)"] = [p.get("upfront_million","?") for p in predictions]
        display_df["예상 마일스톤($M)"] = [p.get("milestone_total_million","?") for p in predictions]
        display_df["예상 로열티(%)"] = [p.get("royalty_rate_percent","?") for p in predictions]
        st.dataframe(display_df, use_container_width=True)

        # 3. 인사이트 리포트 (상위 3개)
        st.subheader("📑 BD 인사이트 리포트 (상위 3개 타겟)")
        top3_indices = display_df.index[:3]
        top3_targets = []
        for idx in top3_indices:
            row = df_all.loc[idx]
            top3_targets.append({
                "name": row["NCT_ID"] + " / " + row["스폰서"],
                "nct_id": row["NCT_ID"],
                "phase": row["임상단계"],
                "sponsor": row["스폰서"],
                "moa": row["중재법"],
                "deal_prediction": predictions[idx],
                "financial_health": "데이터 없음 (MVP)"
            })

        report = generate_bd_report(top3_targets, st.session_state.gemini)
        st.markdown(report)

        st.download_button(
            label="📥 보고서 다운로드 (Markdown)",
            data=report,
            file_name="pharmascope_report.md",
            mime="text/markdown"
        )
else:
    st.info("왼쪽 사이드바에서 조건을 입력한 후 '타겟 스캔 시작' 버튼을 눌러주세요.")
