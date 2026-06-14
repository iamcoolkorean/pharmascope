"""
PharmaScope – AI 라이선스 인 타겟 발굴 에이전트
- 검색 자동 완화, 유사도 점수, Plotly 차트
- 자가 진단 try-except 적용
"""
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from dotenv import load_dotenv

from scanner.clinical_trials import search_clinical_trials, extract_relevant_fields
from valuator.deal_data_generator import generate_sample_deals
from valuator.rag_valuator import (
    load_vector_store,
    prepare_deal_documents,
    build_vector_store,
    retrieve_similar_deals_with_scores,
    predict_deal_structure
)
from reporter.report_generator import generate_bd_report
from utils.llm import get_gemini
from config import CHROMA_DB_PATH

load_dotenv()

st.set_page_config(page_title="PharmaScope", layout="wide")
st.title("💊 PharmaScope – AI 라이선스 인 타겟 발굴 에이전트")

# ── 세션 초기화 ──
if "gemini" not in st.session_state:
    st.session_state.gemini = get_gemini()

if "vector_store" not in st.session_state:
    if not os.path.exists(CHROMA_DB_PATH):
        with st.spinner("🔧 벡터 DB 최초 구축 중... (약 30초 소요)"):
            if not os.path.exists("sample_deals.csv"):
                df = generate_sample_deals()
                df.to_csv("sample_deals.csv", index=False)
            docs = prepare_deal_documents("sample_deals.csv")
            build_vector_store(docs)
        st.success("✅ 벡터 DB 구축 완료!")
    try:
        st.session_state.vector_store = load_vector_store()
    except Exception as e:
        st.error(f"벡터 DB를 불러올 수 없습니다.\n{str(e)}")
        st.stop()

# ── 사이드바 입력 ──
with st.sidebar:
    st.header("🔍 탐색 조건")
    disease = st.text_input("질환명 (예: NASH)", value="NASH")
    modality = st.text_input("기술 모달리티 (예: siRNA)", value="")
    phase = st.selectbox("임상 단계", ["Phase 1", "Phase 2", "Phase 3", "모든 단계"])
    st.markdown("---")
    search_btn = st.button("타겟 스캔 시작", type="primary")

if search_btn:
    if not disease:
        st.warning("질환명을 입력해주세요.")
        st.stop()

    # ── 검색 및 자동 완화 ──
    with st.spinner("임상시험 DB 검색 중..."):
        cur_modality = modality
        cur_phase_api = None if phase == "모든 단계" else phase.replace(" ", "").upper()
        found = False

        # 시도 1: 사용자 입력 그대로
        raw = search_clinical_trials(disease, cur_modality, cur_phase_api)
        df_all = extract_relevant_fields(raw)

        # Phase 필터: df_all이 비어 있지 않을 때만 적용
        if not df_all.empty and cur_phase_api:
            df_all = df_all[df_all["임상단계"].str.contains(phase, na=False)]

        if not df_all.empty:
            found = True
            st.success(f"✅ 원하신 조건으로 {len(df_all)} 건의 후보를 찾았습니다.")
        else:
            # 시도 2: 모달리티 제거
            if modality:
                st.warning("📭 원하신 조건에 맞는 시험이 없습니다. 모달리티를 제거하고 재검색합니다...")
                cur_modality = None
                raw = search_clinical_trials(disease, cur_modality, cur_phase_api)
                df_all = extract_relevant_fields(raw)
                if not df_all.empty and cur_phase_api:
                    df_all = df_all[df_all["임상단계"].str.contains(phase, na=False)]
                if not df_all.empty:
                    found = True
                    st.success(f"✅ 모달리티 없이 {len(df_all)} 건의 후보를 찾았습니다.")
            # 시도 3: 단계도 완화
            if not found and phase != "모든 단계":
                st.warning("📭 여전히 결과가 없습니다. 임상 단계를 모든 단계로 확장합니다...")
                cur_phase_api = None
                raw = search_clinical_trials(disease, cur_modality, cur_phase_api)
                df_all = extract_relevant_fields(raw)
                if not df_all.empty:
                    found = True
                    st.success(f"✅ 모든 임상 단계에서 {len(df_all)} 건의 후보를 찾았습니다.")
            if not found:
                st.error("🛑 모든 조건을 완화했지만 결과가 없습니다. 질환명을 확인하거나 다른 검색어를 시도해보세요.")
                st.stop()

        df_all = df_all.head(10)

    # ── Deal Valuator + 유사도 점수 (자가 진단) ──
    st.subheader("💰 유사 과거 딜 분석 & 예상 계약 조건")

    predictions = []
    similarity_scores = []

    try:
        for idx, row in df_all.iterrows():
            disease_str = row["질환"][:50]
            if modality:
                mod_str = modality
            else:
                inter = row["중재법"]
                mod_str = inter.split(",")[0].strip()[:30] if inter else "기타"
            phase_str = row["임상단계"]
            target_name = row["NCT_ID"] + " / " + row["스폰서"]

            st.write(f"⏳ 처리 중: {target_name}") 

            # 점수 포함 검색 시도
            try:
                similar_docs, scores = retrieve_similar_deals_with_scores(
                    disease_str, mod_str, phase_str,
                    st.session_state.vector_store
                )
                if scores:
                    best_score = max([1/(1+s) for s in scores]) * 100
                    similarity_scores.append(round(best_score, 1))
                else:
                    similarity_scores.append(0.0)
            except Exception as e:
                st.error(f"유사도 검색 오류 ({target_name}): {str(e)}")
                similarity_scores.append(0.0)
                similar_docs = []

            # 예측 시도
            try:
                deal_pred = predict_deal_structure(
                    target_name, disease_str, mod_str, phase_str,
                    similar_docs, st.session_state.gemini
                )
            except Exception as e:
                st.error(f"딜 예측 오류 ({target_name}): {str(e)}")
                deal_pred = {"upfront_million": None, "milestone_total_million": None, "royalty_rate_percent": None, "rationale": str(e)}

            predictions.append(deal_pred)

    except Exception as e:
        st.error(f"Deal Valuator 전체 오류: {str(e)}")
        st.stop()

    # 결과 표시
    display_df = df_all.copy()
    display_df["유사도 점수"] = similarity_scores
    display_df["예상 계약금($M)"] = [p.get("upfront_million","?") for p in predictions]
    display_df["예상 마일스톤($M)"] = [p.get("milestone_total_million","?") for p in predictions]
    display_df["예상 로열티(%)"] = [p.get("royalty_rate_percent","?") for p in predictions]
    st.dataframe(display_df, use_container_width=True)

    # ── Plotly 차트: 예상 계약금 비교 ──
    try:
        st.subheader("📊 예상 계약금 비교")
        chart_df = display_df[display_df["예상 계약금($M)"] != "?"].copy()
        if not chart_df.empty:
            chart_df["예상 계약금($M)"] = pd.to_numeric(chart_df["예상 계약금($M)"])
            fig = px.bar(
                chart_df,
                x="NCT_ID",
                y="예상 계약금($M)",
                color="예상 계약금($M)",
                color_continuous_scale="Blues",
                title="타겟별 예상 계약금 (Upfront)",
                labels={"NCT_ID": "임상시험 ID", "예상 계약금($M)": "계약금 (백만 달러)"}
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("예상 계약금 정보가 없어 차트를 그릴 수 없습니다.")
    except Exception as e:
        st.error(f"차트 생성 오류: {str(e)}")

    # ── 인사이트 리포트 ──
    st.subheader("📑 BD 인사이트 리포트 (상위 3개 타겟)")
    top3_indices = display_df.index[:min(3, len(display_df))]
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

    with st.spinner("인사이트 리포트 생성 중..."):
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
