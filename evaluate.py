"""
evaluate.py - Generate holdout evaluation metrics for resume.
Run after training: python evaluate.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
    roc_curve, precision_recall_curve
)

MODEL_DIR = "models"
os.makedirs("plots", exist_ok=True)


def main():
    print("Loading models and features...")
    xgb_model = joblib.load(f"{MODEL_DIR}/best_xgb.pkl")
    lgb_model  = joblib.load(f"{MODEL_DIR}/best_lgb.pkl")
    features   = joblib.load(f"{MODEL_DIR}/features.pkl")

    print("Loading test data (last 20% of train by time)...")
    # Load full training data and use last 20% as holdout
    try:
        train = pd.read_csv("data/train_transaction.csv")
        train = train.sort_values('TransactionDT')
        split = int(len(train) * 0.8)
        test_raw = train.iloc[split:].copy()
        y_test   = test_raw['isFraud'].values
    except FileNotFoundError:
        print("data/train_transaction.csv not found.")
        print("Run from project root with data/ folder present.")
        return

    # Build features using the same pipeline
    from features import engineer_single
    print(f"Engineering features for {len(test_raw)} holdout samples...")

    rows = []
    for _, row in test_raw.iterrows():
        feat = engineer_single(row.to_dict())
        rows.append(feat)

    X_test = pd.DataFrame(rows)
    for col in features:
        if col not in X_test.columns:
            X_test[col] = -999
    X_test = X_test[features].fillna(-999).astype(np.float32)

    # Predictions
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    ens_probs = 0.4 * xgb_probs + 0.6 * lgb_probs

    # Optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, ens_probs)
    f1s        = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_thresh = thresholds[np.argmax(f1s)]
    y_pred      = (ens_probs >= best_thresh).astype(int)

    auc  = roc_auc_score(y_test, ens_probs)
    ap   = average_precision_score(y_test, ens_probs)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print("\n" + "="*55)
    print("HOLDOUT EVALUATION — USE THESE ON YOUR RESUME")
    print("="*55)
    print(f"  AUC-ROC:           {auc:.4f}  →  {auc*100:.1f}%")
    print(f"  Avg Precision:     {ap:.4f}  →  {ap*100:.1f}%")
    print(f"  F1 Score:          {f1:.4f}  →  {f1*100:.1f}%")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  Optimal Threshold: {best_thresh:.3f}")
    print(f"\n  RESUME LINE:")
    print(f"  Trained XGBoost+LightGBM ensemble on 590K+ IEEE-CIS")
    print(f"  transactions achieving {auc*100:.1f}% AUC and {f1*100:.1f}% F1")
    print("="*55)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, ens_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Fraud Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=150)
    plt.close()
    print("\nPlots saved to plots/")


if __name__ == "__main__":
    main()
