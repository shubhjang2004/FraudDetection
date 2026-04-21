"""
train.py - Generate uid_stats.pkl from your training data.
Run this ONCE after Kaggle training to enable Tier 5 features at inference.

Usage:
  python train.py --train_transaction data/train_transaction.csv \
                  --train_identity    data/train_identity.csv
"""

import argparse
import joblib
import os
import pandas as pd

from features import compute_uid_stats

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_transaction', required=True)
    parser.add_argument('--train_identity',    required=True)
    args = parser.parse_args()

    print("Loading training data...")
    train = pd.read_csv(args.train_transaction).merge(
        pd.read_csv(args.train_identity), on='TransactionID', how='left')
    print(f"  Shape: {train.shape}")

    uid_stats = compute_uid_stats(train)

    out_path = os.path.join(MODEL_DIR, 'uid_stats.pkl')
    joblib.dump(uid_stats, out_path)
    print(f"\nUID stats saved to {out_path}")
    print(f"  UIDs:  {len(uid_stats['uid']):,}")
    print(f"  UID2s: {len(uid_stats['uid2']):,}")
    print("\nRestart the API server to load tier 5 features.")


if __name__ == "__main__":
    main()
