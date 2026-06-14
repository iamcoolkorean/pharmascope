"""
ClinicalTrials.gov API v2를 이용한 임상시험 검색 모듈.
"""
import requests
import time
import pandas as pd
from typing import List, Dict, Optional

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def search_clinical_trials(
    condition: str,
    intervention: Optional[str] = None,
    phase: Optional[str] = None,
    status: str = "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
    page_size: int = 100,
    max_pages: int = 10
) -> List[Dict]:
    """임상시험 데이터를 API에서 가져옵니다."""
    query_parts = [condition]
    if intervention:
        query_parts.append(intervention)
    query_term = " AND ".join(query_parts)

    params = {
        "query.term": query_term,
        "filter.overallStatus": status,
        "pageSize": page_size,
        "format": "json"
    }
    if phase:
        params["filter.phase"] = phase

    all_studies = []
    next_page_token = None

    for page in range(max_pages):
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print(f"API 오류: {response.status_code} - {response.text}")
            break

        data = response.json()
        studies = data.get("studies", [])
        all_studies.extend(studies)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.5)

    return all_studies

def extract_relevant_fields(studies: List[Dict]) -> pd.DataFrame:
    """API 응답에서 필요한 필드만 뽑아내 DataFrame 생성"""
    rows = []
    for s in studies:
        protocol = s.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {})
        description = protocol.get("descriptionModule", {}).get("briefSummary", "")
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])

        rows.append({
            "NCT_ID": identification.get("nctId"),
            "제목": identification.get("briefTitle", ""),
            "상태": status_module.get("overallStatus", ""),
            "임상단계": ",".join(design.get("phases", [])),
            "질환": ", ".join(conditions),
            "중재법": ", ".join([i.get("name", "") for i in interventions]),
            "스폰서": sponsor.get("leadSponsor", {}).get("name", ""),
            "등록일": status_module.get("startDateStruct", {}).get("date", ""),
            "요약": description[:200]
        })
    return pd.DataFrame(rows)
