"""
scorer.py - Load trained models and score transactions at inference time.
Graceful degradation: works with 1 field or 400 fields.

CHANGES FROM ORIGINAL:
  - SHAP: replaced deprecated list-check with modern Explanation object API.
    shap.TreeExplainer(model)(df) returns an Explanation; use .values not [1].
  - Added _explainer_cache so the SHAP explainer is only built once.
  - get_model_status() now catches all exception types cleanly.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Optional
from features import engineer_single, get_available_tier

MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Lazy-loaded globals
_xgb_model    = None
_lgb_model    = None
_feature_cols = None
_explainer    = None
_uid_stats    = None


def _load_models():
    global _xgb_model, _lgb_model, _feature_cols, _explainer, _uid_stats

    if _xgb_model is not None:
        return  # already loaded

    print("Loading models...")

    xgb_path  = os.path.join(MODEL_DIR, 'best_xgb.pkl')
    lgb_path  = os.path.join(MODEL_DIR, 'best_lgb.pkl')
    feat_path = os.path.join(MODEL_DIR, 'features.pkl')
    uid_path  = os.path.join(MODEL_DIR, 'uid_stats.pkl')

    if not os.path.exists(xgb_path):
        raise FileNotFoundError(
            f"Model not found at {xgb_path}. "
            "Run the Colab training notebook first and copy models/ here."
        )

    _xgb_model    = joblib.load(xgb_path)
    _lgb_model    = joblib.load(lgb_path)
    _feature_cols = joblib.load(feat_path)

    # UID stats optional — enables tier 5
    if os.path.exists(uid_path):
        _uid_stats = joblib.load(uid_path)
        print(f"  UID stats loaded: {len(_uid_stats.get('uid', {})):,} UIDs")
    else:
        print("  UID stats not found — tier 5 features disabled (score still works)")

    # FIXED: use modern SHAP Explanation API
    _explainer = shap.TreeExplainer(_xgb_model)
    print("Models loaded.")


def _build_feature_row(transaction: dict) -> pd.DataFrame:
    """Build a single-row DataFrame aligned to training feature columns."""
    _load_models()

    feat_dict = engineer_single(transaction, uid_stats=_uid_stats)

    row = {}
    for col in _feature_cols:
        row[col] = feat_dict.get(col, -999)

    df = pd.DataFrame([row])[_feature_cols]
    df = df.fillna(-999).replace([np.inf, -np.inf], -999)
    return df.astype(np.float32)


def score_transaction(transaction: dict) -> tuple[float, list[dict], int]:
    """
    Score a single transaction.
    Returns (risk_score, top_shap_features, tier_used)
    """
    _load_models()

    tier = get_available_tier(transaction)
    df   = _build_feature_row(transaction)

    # Ensemble: 40% XGBoost + 60% LightGBM
    xgb_score = float(_xgb_model.predict_proba(df)[0][1])
    lgb_score = float(_lgb_model.predict_proba(df)[0][1])
    risk_score = 0.4 * xgb_score + 0.6 * lgb_score

    # FIXED: modern SHAP returns an Explanation object, not a list
    shap_explanation = _explainer(df)          # returns shap.Explanation
    shap_row = shap_explanation.values[0]      # shape: (n_features,) or (n_features, n_classes)

    # For binary classifiers shap_row may be 2D (n_features, 2) — take class-1 column
    if shap_row.ndim == 2:
        shap_row = shap_row[:, 1]

    feat_df = pd.DataFrame({
        'feature':     _feature_cols,
        'value':       df.iloc[0].values,
        'shap_impact': shap_row,
    })

    # Only show features that are actually present (not -999 placeholders)
    feat_df = feat_df[feat_df['value'] != -999]

    top_features = (
        feat_df
        .reindex(feat_df['shap_impact'].abs().sort_values(ascending=False).index)
        .head(5)
        .to_dict('records')
    )

    return float(risk_score), top_features, tier


def get_model_status() -> dict:
    """Return model health info for /health endpoint."""
    try:
        _load_models()
        tier5_ready = _uid_stats is not None
        return {
            "status":        "healthy",
            "models_loaded": ["xgboost", "lightgbm"],
            "feature_count": len(_feature_cols),
            "tier5_ready":   tier5_ready,
            "uid_count":     len(_uid_stats.get('uid', {})) if tier5_ready else 0,
        }
    except FileNotFoundError as e:
        return {"status": "models_not_found", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}