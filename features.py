"""
features.py - Feature registry with graceful degradation
Core idea: features grouped by tiers. More input = more tiers = better score.
Missing fields default to -999 so tree models still work.
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── FEATURE TIERS ─────────────────────────────────────────────────────────────
TIER_1_FEATURES = [
    'TransactionAmt', 'amt_log', 'amt_decimal', 'amt_is_round',
    'hour', 'day', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_night'
]
TIER_2_FEATURES = [
    'card1', 'card2', 'card3', 'card5',
    'card1_freq', 'card2_freq', 'amt_zscore', 'card1_amt_mean',
]
TIER_3_FEATURES = [
    'addr1', 'addr2', 'addr1_freq', 'addr2_freq',
    'P_emaildomain', 'R_emaildomain',
    'P_emaildomain_freq', 'R_emaildomain_freq',
    'same_email', 'P_email_risk', 'addr_mismatch',
]
TIER_4_FEATURES = (
    ['DeviceType', 'DeviceInfo', 'id_num_missing'] +
    [f'id_{i:02d}' for i in range(1, 39)]
)
TIER_5_FEATURES = (
    ['uid_count', 'uid2_count', 'uid_amt_sum', 'uid_amt_max'] +
    [f'{c}_uid_mean'  for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
    [f'{c}_uid_std'   for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
    [f'{c}_uid2_mean' for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
    [f'M{i}_uid_mean'  for i in range(1, 10)] +
    [f'M{i}_uid2_mean' for i in range(1, 10)]
)

KNOWN_CAT_COLS = (
    ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
     'DeviceType', 'DeviceInfo'] +
    [f'id_{i:02d}' for i in range(12, 39)]
)


def get_available_tier(transaction: dict) -> int:
    """Detect which tier is available from input fields."""
    if transaction.get('uid_count') is not None:        return 5
    if transaction.get('DeviceType') is not None:       return 4
    if transaction.get('P_emaildomain') is not None:    return 3
    if any(transaction.get(f'card{i}') for i in range(1, 7)): return 2
    return 1


def engineer_single(transaction: dict, uid_stats: Optional[dict] = None) -> dict:
    """
    Engineer all features from a single transaction dict.
    Graceful degradation: missing fields → -999, model still scores.
    uid_stats: precomputed dict from compute_uid_stats() for tier 5.
    """
    feat = {}

    # ── Tier 1: Amount + Time ─────────────────────────────────────────────────
    amt = float(transaction.get('TransactionAmt') or 0)
    dt  = float(transaction.get('TransactionDT')  or 0)

    feat['TransactionAmt'] = amt
    feat['amt_log']        = float(np.log1p(amt))
    feat['amt_decimal']    = float(amt - int(amt))
    feat['amt_is_round']   = int(feat['amt_decimal'] == 0)
    feat['hour']           = int((dt // 3600) % 24)
    feat['day']            = int((dt // 86400) % 7)
    feat['hour_sin']       = float(np.sin(2 * np.pi * feat['hour'] / 24))
    feat['hour_cos']       = float(np.cos(2 * np.pi * feat['hour'] / 24))
    feat['day_sin']        = float(np.sin(2 * np.pi * feat['day'] / 7))
    feat['day_cos']        = float(np.cos(2 * np.pi * feat['day'] / 7))
    feat['is_weekend']     = int(feat['day'] >= 5)
    feat['is_night']       = int(feat['hour'] >= 22 or feat['hour'] <= 5)

    # ── Tier 2: Card ──────────────────────────────────────────────────────────
    feat['card1'] = int(transaction.get('card1')  or -999)
    feat['card2'] = float(transaction.get('card2') or -999)
    feat['card3'] = float(transaction.get('card3') or -999)
    feat['card5'] = float(transaction.get('card5') or -999)
    feat['card1_freq']     = -999  # needs full dataset
    feat['card2_freq']     = -999
    feat['card1_amt_mean'] = -999
    feat['amt_zscore']     = -999

    # ── Tier 3: Email + Address ───────────────────────────────────────────────
    p_email  = str(transaction.get('P_emaildomain') or 'unknown')
    r_email  = str(transaction.get('R_emaildomain') or 'unknown')
    high_risk = {'protonmail.com', 'anonymous.com', 'guerrillamail.com', 'mail.com'}
    feat['same_email']    = int(p_email == r_email)
    feat['P_email_risk']  = int(p_email in high_risk)
    feat['addr_mismatch'] = int(
        str(transaction.get('addr1') or '') != str(transaction.get('addr2') or ''))
    feat['addr1']               = float(transaction.get('addr1') or -999)
    feat['addr2']               = float(transaction.get('addr2') or -999)
    feat['addr1_freq']          = -999
    feat['addr2_freq']          = -999
    feat['P_emaildomain_freq']  = -999
    feat['R_emaildomain_freq']  = -999

    # ── Tier 4: Identity ──────────────────────────────────────────────────────
    id_fields = [f'id_{i:02d}' for i in range(1, 12)]
    feat['id_num_missing'] = sum(1 for f in id_fields if transaction.get(f) is None)
    for i in range(1, 39):
        col = f'id_{i:02d}'
        feat[col] = float(transaction.get(col) or -999)

    # ── Tier 5: UID aggregations ──────────────────────────────────────────────
    if uid_stats:
        card1   = str(transaction.get('card1', ''))
        addr1   = str(transaction.get('addr1', ''))
        d1_raw  = transaction.get('D1')
        d1_norm = round(float(d1_raw) - dt / 86400) if d1_raw is not None else -1
        uid     = f"{card1}_{addr1}_{d1_norm}"
        uid2    = f"{card1}_{addr1}"

        uid_d  = uid_stats.get('uid',  {}).get(uid,  {})
        uid2_d = uid_stats.get('uid2', {}).get(uid2, {})

        feat['uid_count']  = uid_d.get('count',  -999)
        feat['uid2_count'] = uid2_d.get('count', -999)
        feat['uid_amt_sum']= uid_d.get('amt_sum', -999)
        feat['uid_amt_max']= uid_d.get('amt_max', -999)

        for col in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']:
            feat[f'{col}_uid_mean']  = uid_d.get(f'{col}_mean',  -999)
            feat[f'{col}_uid_std']   = uid_d.get(f'{col}_std',   -999)
            feat[f'{col}_uid2_mean'] = uid2_d.get(f'{col}_mean', -999)
        for m in range(1, 10):
            feat[f'M{m}_uid_mean']  = uid_d.get(f'M{m}_mean',  -999)
            feat[f'M{m}_uid2_mean'] = uid2_d.get(f'M{m}_mean', -999)
    else:
        feat['uid_count']  = -999
        feat['uid2_count'] = -999
        feat['uid_amt_sum']= -999
        feat['uid_amt_max']= -999
        for col in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']:
            feat[f'{col}_uid_mean']  = -999
            feat[f'{col}_uid_std']   = -999
            feat[f'{col}_uid2_mean'] = -999
        for m in range(1, 10):
            feat[f'M{m}_uid_mean']  = -999
            feat[f'M{m}_uid2_mean'] = -999

    return feat


def compute_uid_stats(train_df: pd.DataFrame) -> dict:
    """
    Precompute per-UID aggregation statistics from training data.
    Run once after training, save with joblib.
    """
    print("Computing UID statistics from training data...")
    df = train_df.copy()

    m_cols = [f'M{i}' for i in range(1, 10) if f'M{i}' in df.columns]
    for col in m_cols:
        if df[col].dtype == object:
            df[col] = df[col].map({'T': 1, 'F': 0}).fillna(-1)

    df['D1_norm'] = (df['D1'] - df['TransactionDT'] / 86400).round(0)
    df['uid']  = (df['card1'].astype(str) + '_' +
                  df['addr1'].astype(str) + '_' +
                  df['D1_norm'].fillna(-1).astype(str))
    df['uid2'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)

    agg_cols = [c for c in
                ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15'] + m_cols
                if c in df.columns]

    uid_stats, uid2_stats = {}, {}

    grp   = df.groupby('uid')
    counts = grp['uid'].count()
    stats  = grp[agg_cols].agg(['mean','std']).fillna(0)
    amts   = grp['TransactionAmt'].agg(['sum','max'])

    for uid_val in counts.index:
        d = {'count': int(counts[uid_val]),
             'amt_sum': float(amts.loc[uid_val, 'sum']),
             'amt_max': float(amts.loc[uid_val, 'max'])}
        for col in agg_cols:
            d[f'{col}_mean'] = float(stats.loc[uid_val, (col, 'mean')])
            d[f'{col}_std']  = float(stats.loc[uid_val, (col, 'std')])
        uid_stats[uid_val] = d

    grp2    = df.groupby('uid2')
    counts2 = grp2['uid2'].count()
    stats2  = grp2[agg_cols].agg(['mean']).fillna(0)

    for uid2_val in counts2.index:
        d = {'count': int(counts2[uid2_val])}
        for col in agg_cols:
            d[f'{col}_mean'] = float(stats2.loc[uid2_val, (col, 'mean')])
        uid2_stats[uid2_val] = d

    print(f"  UIDs: {len(uid_stats):,}  UID2s: {len(uid2_stats):,}")
    return {'uid': uid_stats, 'uid2': uid2_stats}
