"""
main.py - FastAPI Fraud Detection API
Run: uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from models import (
    TransactionRequest, FraudScoreResponse, FeatureImpact,
    BatchTransactionRequest, BatchFraudScoreResponse, HealthResponse
)
from scorer  import score_transaction, get_model_status
from advisor import explain_transaction

app = FastAPI(
    title="Fraud Detection API",
    description="""
## AI-Powered Transaction Fraud Detection

**Graceful degradation**: works with just `TransactionAmt` up to full 300+ feature transactions.

| Tier | Required Fields | Confidence |
|------|----------------|------------|
| 1 | Amount only | Low |
| 2 | + Card info | Moderate |
| 3 | + Email + Address | Good |
| 4 | + Device + Identity | High |
| 5 | + UID history (precomputed) | Maximum |

### Risk Levels
| Score | Level | Action |
|-------|-------|--------|
| 0.0–0.3 | 🟢 LOW | APPROVE |
| 0.3–0.7 | 🟡 MEDIUM | REVIEW |
| 0.7–1.0 | 🔴 HIGH | BLOCK |
    """,
    version="2.0.0"
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    status = get_model_status()
    return HealthResponse(**status)


@app.post("/score", response_model=FraudScoreResponse, tags=["Scoring"])
def score_single(request: TransactionRequest):
    """
    Score a single transaction. Only TransactionAmt required.
    More fields provided = higher confidence score.
    """
    try:
        txn_dict = request.model_dump(exclude_none=False)

        risk_score, top_features, tier = score_transaction(txn_dict)
        result = explain_transaction(txn_dict, risk_score, top_features, tier)

        return FraudScoreResponse(
            risk_score=result['risk_score'],
            risk_level=result['risk_level'],
            recommended_action=result['recommended_action'],
            explanation=result['explanation'],
            top_features=[FeatureImpact(**f) for f in result['top_features']],
            feature_tier=result['feature_tier'],
            tier_note=result['tier_note'],
            model=result['model']
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score-batch", response_model=BatchFraudScoreResponse, tags=["Scoring"])
def score_batch(request: BatchTransactionRequest):
    """Score up to 50 transactions. Each can have different fields."""
    try:
        results = []
        for txn in request.transactions:
            txn_dict = txn.model_dump(exclude_none=False)
            risk_score, top_features, tier = score_transaction(txn_dict)
            result = explain_transaction(txn_dict, risk_score, top_features, tier)
            results.append(FraudScoreResponse(
                risk_score=result['risk_score'],
                risk_level=result['risk_level'],
                recommended_action=result['recommended_action'],
                explanation=result['explanation'],
                top_features=[FeatureImpact(**f) for f in result['top_features']],
                feature_tier=result['feature_tier'],
                tier_note=result['tier_note'],
                model=result['model']
            ))
        return BatchFraudScoreResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Fraud Detection API v2",
        "docs":    "Visit /docs for Swagger UI",
        "health":  "Visit /health for model status"
    }
