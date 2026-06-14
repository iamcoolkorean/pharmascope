"""
과거 5년간의 공개 라이선스-인 딜 데이터를 모사한 예제 CSV 생성기.
"""
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_sample_deals(num_deals: int = 30, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    diseases = ["NASH", "비만", "ADC 항암", "알츠하이머", "류마티스 관절염", "당뇨병"]
    modalities = ["siRNA", "ADC", "경구용 GLP-1", "CAR-T", "항체", "펩타이드"]
    phases = ["Phase 1", "Phase 2", "Phase 3"]

    deals = []
    for i in range(num_deals):
        disease = random.choice(diseases)
        modality = random.choice(modalities)
        phase = random.choice(phases)

        upfront = round(random.uniform(10, 500), 1)
        total_milestones = round(random.uniform(200, 5000), 1)
        royalty = round(random.uniform(5, 20), 1)

        deal_date = datetime(2019, 1, 1) + timedelta(days=random.randint(0, 1500))

        deals.append({
            "deal_id": f"D{2020 + i//10}-{i:03d}",
            "deal_date": deal_date.strftime("%Y-%m-%d"),
            "licensor": f"바이오텍{chr(65+i%26)}",
            "licensee": random.choice(["화이자", "노바티스", "로슈", "릴리", "아스트라제네카"]),
            "technology": modality,
            "disease_area": disease,
            "phase_at_deal": phase,
            "upfront_million": upfront,
            "milestone_total_million": total_milestones,
            "royalty_rate_percent": royalty,
            "description": f"{modality} 기반 {disease} 치료제. {phase} 단계."
        })
    return pd.DataFrame(deals)

if __name__ == "__main__":
    df = generate_sample_deals()
    df.to_csv("sample_deals.csv", index=False)
    print("샘플 딜 데이터 생성 완료: sample_deals.csv")
