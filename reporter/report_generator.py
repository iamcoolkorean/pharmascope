"""
BD 인사이트 리포트 생성 체인 (LangChain)
"""
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.llm import get_gemini

def generate_bd_report(
    targets: list[dict],
    gemini: ChatGoogleGenerativeAI
) -> str:
    """상위 타겟 리스트를 입력받아 마크다운 형식의 BD 보고서를 생성합니다."""
    targets_text = ""
    for i, t in enumerate(targets, 1):
        deal = t.get("deal_prediction", {})
        targets_text += f"""
**후보 {i}**: {t['name']}  
- NCT ID: {t.get('nct_id','N/A')}  
- 현재 단계: {t['phase']}  
- 개발사: {t['sponsor']}  
- 작용 기전: {t.get('moa','정보 없음')}  
- 예상 계약금: ${deal.get('upfront_million','?')}M  
- 예상 마일스톤 총액: ${deal.get('milestone_total_million','?')}M  
- 예상 로열티: {deal.get('royalty_rate_percent','?')}%  
- 재무 안전성: {t.get('financial_health','분석 불가')}
"""

    prompt = PromptTemplate.from_template(
        """당신은 제약 라이선스 인 전문가입니다. 아래 후보 약물들에 대해 BD팀에 제공할 1페이지 분량의 인사이트 보고서를 마크다운으로 작성하세요.

보고서는 다음 섹션을 포함해야 합니다.
1. **Executive Summary** (3문장)
2. **후보별 분석**
   - 기술 경쟁력
   - 재무 안전성 평가
   - 예상 협상 전략
3. **종합 권고**

[타겟 후보 정보]
{targets}

보고서는 간결하고 비즈니스 의사결정에 도움이 되도록 작성하세요.
마크다운 문법을 사용하며, 헤딩(##)과 글머리 기호를 활용하세요.
"""
    )
    chain = prompt | gemini
    result = chain.invoke({"targets": targets_text})
    return result.content
