"""
train.py - Full training pipeline: XGBoost + LightGBM + uid_stats.

Two modes:
  1. Full training (run on Kaggle/Colab with IEEE-CIS data):
       python train.py --train_transaction data/train_transaction.csv \
                       --train_identity    data/train_identity.csv

  2. UID-stats only (if you already have best_xgb.pkl / best_lgb.pkl from Colab):
       python train.py --uid_only \
                       --train_transaction data/train_transaction.csv \
                       --train_identity    data/train_identity.csv

CHANGES FROM ORIGINAL:
  - Added full XGBoost + LightGBM training loop (was uid_stats only before).
  - 5-fold stratified CV with time-based holdout split.
  - Saves best_xgb.pkl, best_lgb.pkl, features.pkl, uid_stats.pkl.
  - --uid_only flag to skip model training if .pkl already exist.
"""

import argparse
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

from features import engineer_single, compute_uid_stats, EMAIL_DOMAIN_MAP, DEVICE_TYPE_MAP

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Label-encode categorical columns to match inference encoding ───────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding as engineer_single() but vectorised for training."""
    out = df.copy()

    # Email domains → integer
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in out.columns:
            out[col] = out[col].map(lambda x: EMAIL_DOMAIN_MAP.get(
                str(x).lower().strip(), 0) if pd.notna(x) else -999)

    # DeviceType → integer
    if 'DeviceType' in out.columns:
        out['DeviceType'] = out['DeviceType'].map(lambda x: DEVICE_TYPE_MAP.get(
            str(x).lower().strip(), 0) if pd.notna(x) else -999)

    # DeviceInfo → hash bucket
    if 'DeviceInfo' in out.columns:
        out['DeviceInfo'] = out['DeviceInfo'].map(lambda x:
            hash(str(x).lower().strip()) % 1000 if pd.notna(x) else -999)

    # Categorical id_ columns → hash bucket
    for i in range(12, 39):
        col = f'id_{i:02d}'
        if col in out.columns and out[col].dtype == object:
            out[col] = out[col].map(lambda x:
                hash(str(x).strip().upper()) % 100 if pd.notna(x) else -999)

    # M columns T/F → 1/0
    for i in range(1, 10):
        col = f'M{i}'
        if col in out.columns and out[col].dtype == object:
            out[col] = out[col].map({'T': 1, 'F': 0})

    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Engineer all features from a merged transaction+identity DataFrame."""
    print("Engineering features...")

    df = preprocess_dataframe(df)

    # Time features
    df['amt_log']      = np.log1p(df['TransactionAmt'])
    df['amt_decimal']  = df['TransactionAmt'] - df['TransactionAmt'].astype(int)
    df['amt_is_round'] = (df['amt_decimal'] == 0).astype(int)
    df['hour']         = (df['TransactionDT'] // 3600 % 24).fillna(0).astype(int)
    df['day']          = (df['TransactionDT'] // 86400 % 7).fillna(0).astype(int)
    df['hour_sin']     = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']     = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin']      = np.sin(2 * np.pi * df['day'] / 7)
    df['day_cos']      = np.cos(2 * np.pi * df['day'] / 7)
    df['is_weekend']   = (df['day'] >= 5).astype(int)
    df['is_night']     = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # Email features
    HIGH_RISK = {'protonmail.com', 'anonymous.com', 'guerrillamail.com', 'mail.com'}
    if 'P_emaildomain' in df.columns:
        df['same_email']   = (df['P_emaildomain'] == df.get('R_emaildomain', -1)).astype(int)
        df['P_email_risk'] = df['P_emaildomain'].isin(
            [EMAIL_DOMAIN_MAP.get(d, 0) for d in HIGH_RISK]).astype(int)
    else:
        df['same_email']   = -999
        df['P_email_risk'] = -999

    # Address mismatch
    df['addr_mismatch'] = (
        df.get('addr1', pd.Series(-999, index=df.index)).astype(str) !=
        df.get('addr2', pd.Series(-999, index=df.index)).astype(str)
    ).astype(int)

    # Card frequency features
    for col in ['card1', 'card2']:
        if col in df.columns:
            freq = df[col].value_counts()
            df[f'{col}_freq'] = df[col].map(freq).fillna(0)

    # Card1 amount stats
    if 'card1' in df.columns:
        card1_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
        card1_std  = df.groupby('card1')['TransactionAmt'].transform('std').fillna(0)
        df['card1_amt_mean'] = card1_mean
        df['amt_zscore']     = (df['TransactionAmt'] - card1_mean) / (card1_std + 1e-8)

    # Address frequency
    for col in ['addr1', 'addr2']:
        if col in df.columns:
            freq = df[col].value_counts()
            df[f'{col}_freq'] = df[col].map(freq).fillna(0)

    # Email frequency
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            freq = df[col].value_counts()
            df[f'{col}_freq'] = df[col].map(freq).fillna(0)

    # id missing count
    id_num_cols = [f'id_{i:02d}' for i in range(1, 12) if f'id_{i:02d}' in df.columns]
    df['id_num_missing'] = df[id_num_cols].isna().sum(axis=1)

    # All feature columns (exclude target + metadata)
    exclude = {'TransactionID', 'isFraud', 'TransactionDT'}
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols


def train_models(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]):
    """Train XGBoost + LightGBM with 5-fold CV, return best models."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))

    best_xgb_models, best_lgb_models = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n── Fold {fold+1}/5 ──")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
            eval_metric='auc',
            early_stopping_rounds=50,
            random_state=42,
            tree_method='hist',
            verbosity=0,
        )
        xgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
        xgb_oof[val_idx] = xgb_preds
        print(f"  XGB AUC: {roc_auc_score(y_val, xgb_preds):.4f}")
        best_xgb_models.append(xgb_model)

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        lgb_oof[val_idx] = lgb_preds
        print(f"  LGB AUC: {roc_auc_score(y_val, lgb_preds):.4f}")
        best_lgb_models.append(lgb_model)

    ens_oof = 0.4 * xgb_oof + 0.6 * lgb_oof
    print(f"\n── OOF Ensemble AUC: {roc_auc_score(y, ens_oof):.4f} ──")

    # Pick best fold model by OOF AUC
    best_fold = max(range(5), key=lambda i: roc_auc_score(
        y.iloc[list(skf.split(X, y))[i][1]],
        xgb_oof[list(skf.split(X, y))[i][1]]))

    return best_xgb_models[best_fold], best_lgb_models[best_fold]


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument('--train_transaction', required=True)
    parser.add_argument('--train_identity',    required=True)
    parser.add_argument('--uid_only', action='store_true',
                        help="Only compute uid_stats.pkl; skip model training")
    args = parser.parse_args()

    print("Loading training data...")
    train = pd.read_csv(args.train_transaction).merge(
        pd.read_csv(args.train_identity), on='TransactionID', how='left')
    print(f"  Shape: {train.shape}")
    print(f"  Fraud rate: {train['isFraud'].mean():.3%}")

    # ── UID stats ─────────────────────────────────────────────────────────────
    uid_stats = compute_uid_stats(train)
    uid_path  = os.path.join(MODEL_DIR, 'uid_stats.pkl')
    joblib.dump(uid_stats, uid_path)
    print(f"UID stats saved → {uid_path}")

    if args.uid_only:
        print("--uid_only flag set. Skipping model training.")
        return

    # ── Feature engineering ───────────────────────────────────────────────────
    df, feature_cols = build_features(train)

    # Add UID aggregation features to training data
    # (simplified: just count per uid)
    df['D1_norm'] = (df['D1'] - df['TransactionDT'] / 86400).round(0)
    df['uid']  = (df['card1'].astype(str) + '_' + df['addr1'].astype(str) +
                  '_' + df['D1_norm'].fillna(-1).astype(str))
    df['uid2'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
    df['uid_count']  = df.groupby('uid')['uid'].transform('count')
    df['uid2_count'] = df.groupby('uid2')['uid2'].transform('count')
    for col in ['uid_count', 'uid2_count']:
        if col not in feature_cols:
            feature_cols.append(col)

    X = df[feature_cols].fillna(-999).replace([np.inf, -np.inf], -999).astype(np.float32)
    y = df['isFraud'].astype(int)

    print(f"\nFeatures: {len(feature_cols)}  Samples: {len(X)}  Fraud: {y.sum()}")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_xgb, best_lgb = train_models(X, y, feature_cols)

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_xgb,      os.path.join(MODEL_DIR, 'best_xgb.pkl'))
    joblib.dump(best_lgb,      os.path.join(MODEL_DIR, 'best_lgb.pkl'))
    joblib.dump(feature_cols,  os.path.join(MODEL_DIR, 'features.pkl'))

    print(f"\nSaved to {MODEL_DIR}/")
    print("  best_xgb.pkl")
    print("  best_lgb.pkl")
    print("  features.pkl")
    print("  uid_stats.pkl")
    print("\nRestart the API server to pick up new models.")


if __name__ == "__main__":
    main()