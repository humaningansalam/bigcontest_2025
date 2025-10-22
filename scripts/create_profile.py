# scripts/create_profiles.py
import pandas as pd
import numpy as np
from datetime import datetime
import json
from scipy import stats
import warnings
from pathlib import Path

# 스크립트 실행 시 발생할 수 있는 경고 메시지를 무시합니다.
warnings.filterwarnings('ignore')

# ===============================================
# 1. 모듈화된 분석 함수
# ===============================================

def calculate_business_age(open_date_str, current_date=datetime.now()):
    """
    개설일 문자열(YYYYMMDD 형식)을 받아 현재 날짜 기준 업력(개월)을 계산합니다.
    """
    if pd.isna(open_date_str):
        return None
    try:
        # 정수형으로 변환 후 문자열로 다시 변환하여 파싱 오류 방지
        open_date = datetime.strptime(str(int(open_date_str)), '%Y%m%d')
        # 총 일수를 30.44 (평균 월 일수)로 나누어 개월 수 계산
        age_in_months = (current_date - open_date).days / 30.44
        return int(age_in_months)
    except (ValueError, TypeError):
        return None

def analyze_trend(series: pd.Series, period=6) -> str:
    """
    시계열 데이터(Pandas Series)를 받아 최근 N개월의 추세를
    'upward', 'downward', 'stable' 중 하나로 반환합니다.
    """
    # 유효한 데이터가 2개 미만이면 추세 분석 불가
    valid_series = series.dropna()
    if len(valid_series) < 2:
        return 'stable'
    
    # 최근 N개월 데이터만 선택
    recent_series = valid_series.tail(period)
    if len(recent_series) < 2:
        return 'stable'

    x = np.arange(len(recent_series))
    y = recent_series.values
    
    try:
        # 선형 회귀 실행하여 기울기 계산
        slope, _, _, _, _ = stats.linregress(x, y)
    except ValueError:
        return 'stable' # 데이터가 선형 회귀에 부적합할 경우
    
    # 기울기 임계값 (데이터의 스케일에 따라 조정 가능)
    if slope > 0.1: return 'upward'
    elif slope < -0.1: return 'downward'
    else: return 'stable'

def analyze_customer_segments(row: pd.Series, top_n=3) -> list:
    """
    데이터 행을 받아, 인구통계학적 세그먼트를 분석하여
    상위 N개의 주요 고객층과 그 비율을 리스트로 반환합니다.
    """
    demographics = {
        '20대 이하 여성': row.get('M12_FME_1020_RAT', 0),
        '30대 여성': row.get('M12_FME_30_RAT', 0),
        '40대 여성': row.get('M12_FME_40_RAT', 0),
        '50대 여성': row.get('M12_FME_50_RAT', 0),
        '60대 이상 여성': row.get('M12_FME_60_RAT', 0),
        '20대 이하 남성': row.get('M12_MAL_1020_RAT', 0),
        '30대 남성': row.get('M12_MAL_30_RAT', 0),
        '40대 남성': row.get('M12_MAL_40_RAT', 0),
        '50대 남성': row.get('M12_MAL_50_RAT', 0),
        '60대 이상 남성': row.get('M12_MAL_60_RAT', 0),
    }
    
    # NaN 값을 0으로 처리하고, 비율이 높은 순으로 정렬
    sorted_segments = sorted(
        [(seg, ratio) for seg, ratio in demographics.items() if pd.notna(ratio) and ratio > 0],
        key=lambda item: item[1],
        reverse=True
    )
    
    # 상위 N개 결과를 지정된 형식으로 포맷팅
    top_segments = [
        {"segment": seg, "ratio": round(float(ratio), 2)} for seg, ratio in sorted_segments[:top_n]
    ]
    
    return top_segments

def analyze_customer_types(row: pd.Series) -> dict:
    """
    고객 유형(거주자, 직장인, 유동인구) 비율을 분석하여
    딕셔셔리 형태로 반환합니다.
    """
    types = {
        "resident_pct": row.get('RC_M1_SHC_RSD_UE_CLN_RAT'),
        "worker_pct": row.get('RC_M1_SHC_WP_UE_CLN_RAT'),
        "floating_pct": row.get('RC_M1_SHC_FLP_UE_CLN_RAT')
    }
    # NaN이 아닌 값만 필터링하고 소수점 2자리까지 반올림하여 반환
    return {k: round(float(v), 2) for k, v in types.items() if pd.notna(v)}

# ===============================================
# 2. 메인 실행 로직
# ===============================================

def main():
    print("프로필 생성을 시작합니다...")
    
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data'

    # --- 데이터 로드 및 전처리 ---
    print("1. 데이터 로드 및 전처리 중...")
    try:
        df1 = pd.read_csv(data_path / 'big_data_set1_f.csv', encoding='cp949')
        df2 = pd.read_csv(data_path / 'big_data_set2_f.csv', encoding='cp949')
        df3 = pd.read_csv(data_path / 'big_data_set3_f.csv', encoding='cp949')
    except FileNotFoundError as e:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. '{e.filename}'")
        print("폴더에 CSV 파일들이 있는지 확인해주세요.")
        return

    for df in [df2, df3]:
        for col in df.columns:
            # -999999.9 와 같은 이상치를 NaN으로 변환
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].replace(-999999.9, np.nan)

    # --- 최신 월 데이터(스냅샷)와 전체 시계열 데이터 분리 ---
    latest_df2 = df2.sort_values('TA_YM').drop_duplicates(subset='ENCODED_MCT', keep='last')
    latest_df3 = df3.sort_values('TA_YM').drop_duplicates(subset='ENCODED_MCT', keep='last')

    # --- 데이터 병합 (최신 월 기준) ---
    df1_selected = df1[[
        'ENCODED_MCT', 'MCT_NM', 'MCT_SIGUNGU_NM', 'HPSN_MCT_BZN_CD_NM', 
        'HPSN_MCT_ZCD_NM', 'ARE_D', 'MCT_ME_D', 'MCT_BRD_NUM'
    ]].copy()

    merged_df = pd.merge(df1_selected, latest_df2, on='ENCODED_MCT', how='left')
    merged_df = pd.merge(merged_df, latest_df3, on='ENCODED_MCT', how='left', suffixes=('_df2', '_df3'))
    
    # 추세 분석을 위해 원본 시계열 데이터를 가맹점 ID로 그룹화
    grouped_df2 = df2.groupby('ENCODED_MCT')
    grouped_df3 = df3.groupby('ENCODED_MCT')

    print("2. 가맹점별 프로필 카드 생성 중...")
    all_profiles = []
    
    for _, row in merged_df.iterrows():
        store_id = row['ENCODED_MCT']
        
        store_ts_df2 = grouped_df2.get_group(store_id) if store_id in grouped_df2.groups else pd.DataFrame()
        store_ts_df3 = grouped_df3.get_group(store_id) if store_id in grouped_df3.groups else pd.DataFrame()

        sales_band_map = {'1_10%이하': 1, '2_10-25%': 2, '3_25-50%': 3, '4_50-75%': 4, '5_75-90%': 5, '6_90%초과(하위 10% 이하)': 6}
        sales_series_numeric = store_ts_df2['RC_M1_SAA'].map(sales_band_map) if not store_ts_df2.empty else pd.Series(dtype='float64')
        revisit_series = store_ts_df3['MCT_UE_CLN_REU_RAT'] if not store_ts_df3.empty else pd.Series(dtype='float64')

        profile = {
            "profile_id": store_id,
            "last_updated": datetime.now().isoformat(),
            "core_data": {
                "basic_info": {
                    "store_name_masked": row.get('MCT_NM'),
                    "address_district": row.get('MCT_SIGUNGU_NM'),
                    "commercial_district": row.get('HPSN_MCT_BZN_CD_NM'),
                    "industry_main": row.get('HPSN_MCT_ZCD_NM'),
                    "open_date": row.get('ARE_D'),
                    "close_date": row.get('MCT_ME_D'), 
                    "business_age_months": calculate_business_age(row.get('ARE_D'))
                },
                "performance_metrics": {
                    "latest_period": row.get('TA_YM_df2'),
                    "sales_amount_band": row.get('RC_M1_SAA'),
                    "sales_count_band": row.get('RC_M1_TO_UE_CT'),
                    "avg_spending_per_customer_band": row.get('RC_M1_AV_NP_AT'),
                    "sales_rank_in_district_percentile": row.get('M12_SME_BZN_SAA_PCE_RT'),
                    "sales_rank_in_industry_percentile": row.get('M12_SME_RY_SAA_PCE_RT')
                },
                "time_series_summary": {
                    "sales_trend_6m": analyze_trend(sales_series_numeric, period=6),
                    "revisit_rate_trend_6m": analyze_trend(revisit_series, period=6)
                },
                "customer_profile": {
                    "revisit_rate_latest_percent": row.get('MCT_UE_CLN_REU_RAT'),
                    "new_customer_rate_latest_percent": row.get('MCT_UE_CLN_NEW_RAT'),
                    "top_customer_segments": analyze_customer_segments(row, top_n=3),
                    "customer_type_ratio": analyze_customer_types(row)
                }
            },
            "extended_features": {
                "is_franchise": bool(pd.notna(row.get('MCT_BRD_NUM'))),
                "has_delivery_service": row.get('DLV_SAA_RAT') is not np.nan and row.get('DLV_SAA_RAT') > 0,
                "owner_info": {"age_band": None, "gender": None, "business_goals": []},
                "interaction_history": {"recommended_programs": [], "executed_campaigns": []}
            }
        }
        all_profiles.append(profile)

    output_filename = data_path / 'store_profiles.json'
    print(f"3. 생성된 프로필 {len(all_profiles)}개를 '{output_filename}' 파일로 저장합니다.")
    
    def default_converter(o):
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o) if pd.notna(o) else None
        if isinstance(o, (pd.Timestamp, datetime)): return o.isoformat()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, ensure_ascii=False, indent=2, default=default_converter)
        
    print("프로필 생성 완료!")

if __name__ == "__main__":
    main()