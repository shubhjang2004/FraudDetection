"""
features.py - Feature registry with graceful degradation.

Core idea: features grouped by tiers. More input = more tiers = better score.
Missing fields default to -999 so tree models still work.

CHANGES FROM ORIGINAL:
  - P_emaildomain / R_emaildomain now encoded as integers (freq-hash),
    not stored as raw strings — tree models can't handle strings.
  - DeviceType encoded as 0/1/−999 instead of raw "desktop"/"mobile" string.
  - DeviceInfo encoded as integer hash instead of raw string.
  - Added EMAIL_DOMAIN_MAP and DEVICE_TYPE_MAP constants used at inference.
  - engineer_single() now returns only numeric values — no raw strings anywhere.
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── CATEGORICAL ENCODING MAPS ─────────────────────────────────────────────────
# These must match what was used during training (label-encode before fit).
# If you retrain, regenerate these from your training value_counts().

EMAIL_DOMAIN_MAP: dict[str, int] = {
    "gmail.com": 1,
    "yahoo.com": 2,
    "hotmail.com": 3,
    "outlook.com": 4,
    "anonymous.com": 5,
    "live.com": 6,
    "icloud.com": 7,
    "me.com": 8,
    "protonmail.com": 9,
    "mail.com": 10,
    "comcast.net": 11,
    "att.net": 12,
    "verizon.net": 13,
    "sbcglobal.net": 14,
    "cox.net": 15,
    "charter.net": 16,
    "msn.com": 17,
    "earthlink.net": 18,
    "embarqmail.com": 19,
    "bellsouth.net": 20,
    "guerrillamail.com": 21,
    "rocketmail.com": 22,
    "ymail.com": 23,
    "frontier.com": 24,
    "windstream.net": 25,
    "roadrunner.com": 26,
    "optonline.net": 27,
    "cfl.rr.com": 28,
    "netzero.net": 29,
    "ptd.net": 30,
}

DEVICE_TYPE_MAP: dict[str, int] = {
    "desktop": 1,
    "mobile": 2,
}

# High-risk email domains (used as a boolean feature)
HIGH_RISK_DOMAINS = {"protonmail.com", "anonymous.com", "guerrillamail.com", "mail.com"}

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
    'P_emaildomain', 'R_emaildomain',           # now integers, not strings
    'P_emaildomain_freq', 'R_emaildomain_freq',
    'same_email', 'P_email_risk', 'addr_mismatch',
]

TIER_4_FEATURES = (
    ['DeviceType', 'DeviceInfo', 'id_num_missing'] +  # now integers
    [f'id_{i:02d}' for i in range(1, 39)]
)

TIER_5_FEATURES = (
    ['uid_count', 'uid2_count', 'uid_amt_sum', 'uid_amt_max'] +
    [f'{c}_uid_mean' for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
    [f'{c}_uid_std'  for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
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
    if transaction.get('uid_count') is not None:    return 5
    if transaction.get('DeviceType') is not None:   return 4
    if transaction.get('P_emaildomain') is not None: return 3
    if any(transaction.get(f'card{i}') for i in range(1, 7)): return 2
    return 1


def _encode_email(domain: Optional[str]) -> int:
    """Map email domain string to integer. Unknown domains get 0."""
    if domain is None:
        return -999
    return EMAIL_DOMAIN_MAP.get(str(domain).lower().strip(), 0)


def _encode_device_type(device: Optional[str]) -> int:
    """Map DeviceType string to integer. Unknown → 0, missing → -999."""
    if device is None:
        return -999
    return DEVICE_TYPE_MAP.get(str(device).lower().strip(), 0)


def _encode_device_info(info: Optional[str]) -> int:
    """Encode DeviceInfo as a stable integer hash bucket (0-999)."""
    if info is None:
        return -999
    return hash(str(info).lower().strip()) % 1000


def engineer_single(transaction: dict, uid_stats: Optional[dict] = None) -> dict:
    """
    Engineer all features from a single transaction dict.
    Graceful degradation: missing fields → -999, model still scores.
    ALL returned values are numeric (int or float) — no raw strings.

    uid_stats: precomputed dict from compute_uid_stats() for tier 5.
    """
    feat: dict = {}

    # ── Tier 1: Amount + Time ─────────────────────────────────────────────────
    amt = float(transaction.get('TransactionAmt') or 0)
    dt  = float(transaction.get('TransactionDT') or 0)

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
    feat['card1'] = int(transaction.get('card1') or -999)
    feat['card2'] = float(transaction.get('card2') or -999)
    feat['card3'] = float(transaction.get('card3') or -999)
    feat['card5'] = float(transaction.get('card5') or -999)
    # Frequency features need the full dataset — placeholder at inference
    feat['card1_freq']     = -999
    feat['card2_freq']     = -999
    feat['card1_amt_mean'] = -999
    feat['amt_zscore']     = -999

    # ── Tier 3: Email + Address ───────────────────────────────────────────────
    p_email_raw = transaction.get('P_emaildomain')
    r_email_raw = transaction.get('R_emaildomain')
    p_email_str = str(p_email_raw).lower().strip() if p_email_raw else 'unknown'
    r_email_str = str(r_email_raw).lower().strip() if r_email_raw else 'unknown'

    # FIXED: encode as integers, not raw strings
    feat['P_emaildomain'] = _encode_email(p_email_raw)
    feat['R_emaildomain'] = _encode_email(r_email_raw)
    feat['same_email']    = int(p_email_str == r_email_str)
    feat['P_email_risk']  = int(p_email_str in HIGH_RISK_DOMAINS)
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

    # FIXED: DeviceType and DeviceInfo encoded as integers
    feat['DeviceType'] = _encode_device_type(transaction.get('DeviceType'))
    feat['DeviceInfo'] = _encode_device_info(transaction.get('DeviceInfo'))

    for i in range(1, 39):
        col = f'id_{i:02d}'
        raw = transaction.get(col)
        if raw is None:
            feat[col] = -999
        elif isinstance(raw, str):
            # Categorical identity fields (id_12, id_15, id_16, etc.)
            feat[col] = hash(raw.strip().upper()) % 100
        else:
            feat[col] = float(raw)

    # ── Tier 5: UID aggregations ──────────────────────────────────────────────
    uid_cols = (
        ['uid_count', 'uid2_count', 'uid_amt_sum', 'uid_amt_max'] +
        [f'{c}_uid_mean'  for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
        [f'{c}_uid_std'   for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
        [f'{c}_uid2_mean' for c in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']] +
        [f'M{m}_uid_mean'  for m in range(1, 10)] +
        [f'M{m}_uid2_mean' for m in range(1, 10)]
    )

    if uid_stats:
        card1  = str(transaction.get('card1', ''))
        addr1  = str(transaction.get('addr1', ''))
        d1_raw = transaction.get('D1')
        d1_norm = round(float(d1_raw) - dt / 86400) if d1_raw is not None else -1
        uid  = f"{card1}_{addr1}_{d1_norm}"
        uid2 = f"{card1}_{addr1}"

        uid_d  = uid_stats.get('uid',  {}).get(uid,  {})
        uid2_d = uid_stats.get('uid2', {}).get(uid2, {})

        feat['uid_count']    = uid_d.get('count',   -999)
        feat['uid2_count']   = uid2_d.get('count',  -999)
        feat['uid_amt_sum']  = uid_d.get('amt_sum', -999)
        feat['uid_amt_max']  = uid_d.get('amt_max', -999)

        for col in ['TransactionAmt','C1','C2','C9','C11','C13','D2','D3','D15']:
            feat[f'{col}_uid_mean']  = uid_d.get(f'{col}_mean',  -999)
            feat[f'{col}_uid_std']   = uid_d.get(f'{col}_std',   -999)
            feat[f'{col}_uid2_mean'] = uid2_d.get(f'{col}_mean', -999)
        for m in range(1, 10):
            feat[f'M{m}_uid_mean']  = uid_d.get(f'M{m}_mean',  -999)
            feat[f'M{m}_uid2_mean'] = uid2_d.get(f'M{m}_mean', -999)
    else:
        for col in uid_cols:
            feat[col] = -999

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

    grp    = df.groupby('uid')
    counts = grp['uid'].count()
    stats  = grp[agg_cols].agg(['mean', 'std']).fillna(0)
    amts   = grp['TransactionAmt'].agg(['sum', 'max'])

    for uid_val in counts.index:
        d = {
            'count':   int(counts[uid_val]),
            'amt_sum': float(amts.loc[uid_val, 'sum']),
            'amt_max': float(amts.loc[uid_val, 'max']),
        }
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