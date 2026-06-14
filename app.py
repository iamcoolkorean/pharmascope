    # ── Deal Valuator + 유사도 점수 (자가 진단 추가) ──
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
                similar_docs = []  # 빈 리스트로 처리

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

    # 차트도 try-except로 감싸기
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
