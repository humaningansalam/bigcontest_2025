# src/features/profile_management/resolver.py

import re
import unicodedata
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher

def read_csv_smart(path):
    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='replace')

def _normalize_compare(value: str | None) -> str:
    if value is None: return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", "", text)
    return text.upper()

def load_set1(shinhan_dir: Path) -> pd.DataFrame:
    p = shinhan_dir / 'big_data_set1_f.csv'
    if not p.exists():
        raise FileNotFoundError(f"가맹점 정보 파일을 찾을 수 없습니다: {p}")
    df = read_csv_smart(p)
    ren = {}
    for c in df.columns:
        cu = str(c).upper()
        if cu == 'ENCODED_MCT': ren[c] = 'ENCODED_MCT'
        elif 'SIGUNGU' in cu:   ren[c] = 'SIGUNGU'
        elif cu == 'MCT_NM':    ren[c] = 'MCT_NM'
    df = df.rename(columns=ren)
    df = df.loc[:, ~df.columns.duplicated()]
    keep = ['ENCODED_MCT','MCT_NM','SIGUNGU']
    for k in keep:
        if k not in df.columns: df[k] = pd.NA
    df = df[keep].drop_duplicates('ENCODED_MCT')
    if 'ENCODED_MCT' in df.columns:
        df['ENCODED_MCT'] = df['ENCODED_MCT'].apply(lambda v: str(v).strip() if pd.notna(v) else '')
    return df

def resolve_merchant(masked_name, mask_prefix, sigungu, merchants_df):
    if merchants_df is None or merchants_df.empty or not masked_name:
        return None
    df = merchants_df.copy()
    df['_norm_name'] = df['MCT_NM'].apply(_normalize_compare)
    df['_norm_sigungu'] = df['SIGUNGU'].apply(_normalize_compare)
    norm_sigungu = _normalize_compare(sigungu)
    prefix_norm = _normalize_compare(mask_prefix)
    base = df
    if norm_sigungu:
        exact = base[base['_norm_sigungu'] == norm_sigungu]
        base = exact if not exact.empty else base[base['_norm_sigungu'].str.contains(norm_sigungu, na=False)]
    if base.empty: return None
    rule1 = base[base['_norm_name'].str.startswith(prefix_norm, na=False)] if prefix_norm else base.iloc[0:0]
    if len(rule1) == 1:
        return {'encoded_mct': str(rule1.iloc[0]['ENCODED_MCT'])}
    def _score_rows(frame):
        scores = [SequenceMatcher(None, prefix_norm, r or '').ratio() for r in frame['_norm_name']]
        frame['__score'] = scores
        return frame
    top = _score_rows(rule1 if not rule1.empty else base).sort_values('__score', ascending=False)
    if not top.empty and top.iloc[0]['__score'] > 0.7:
        return {'encoded_mct': str(top.iloc[0]['ENCODED_MCT'])}
    return None



_merchants_df = None

def _get_merchants_df():
    global _merchants_df
    if _merchants_df is None:
        print("--- 💾 가맹점 정보(Set1) 최초 로딩... ---")
        shinhan_dir = Path("data")
        _merchants_df = load_set1(shinhan_dir)
    return _merchants_df

def resolve_store_id_from_name(store_name_query: str) -> str | None:
    """사용자 쿼리에서 가맹점 ID를 찾아 반환하는 최종 인터페이스 함수."""
    print(f"--- 🔍 상호명으로 ID 확인 시작: {store_name_query} ---")
    try:
        merchants_df = _get_merchants_df()
    except FileNotFoundError as e:
        print(f"--- 🚨 오류: {e} ---")
        return None
    sigungu_match = re.search(r'([가-힣]{2,}구)', store_name_query)
    sigungu = sigungu_match.group(1) if sigungu_match else "성동구"
    mask_match = re.search(r'\{([^{}]+)\}', store_name_query)
    masked_name = mask_match.group(1).strip() if mask_match else None
    if not masked_name:
        print("--- ⚠️ 쿼리에서 {상호명***} 패턴을 찾을 수 없음 ---")
        return None
    mask_prefix = masked_name.split('*', 1)[0].strip()
    resolved_merchant = resolve_merchant(masked_name, mask_prefix, sigungu, merchants_df)
    if resolved_merchant and resolved_merchant.get('encoded_mct'):
        store_id = resolved_merchant['encoded_mct']
        print(f"--- ✅ ID 확인 완료: {store_id} ---")
        return store_id
    else:
        print(f"--- ❌ ID를 찾지 못함 ---")
        return None