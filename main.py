"""
main.py - FastAPI Fraud Detection API

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs

CHANGES FROM ORIGINAL:
  - Batch endpoint (/score-batch) now only calls Claude for HIGH-risk transactions.
    Medium and low-risk use fast_explain() (no API call) — saves cost + latency.
  - Added CORS middleware so a frontend can call the API.
  - /health now returns 503 when models are missing, not 200.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    TransactionRequest, FraudScoreResponse, FeatureImpact,
    BatchTransactionRequest, BatchFraudScoreResponse, HealthResponse
)
from scorer import score_transaction, get_model_status
from advisor import explain_transaction, fast_explain

app = FastAPI(
    title="Fraud Detection API",
    description="""
## AI-Powered Transaction Fraud Detection

**Graceful degradation**: works with just `TransactionAmt` up to full 300+ feature transactions.

| Tier | Required Fields       | Confidence |
|------|-----------------------|------------|
| 1    | Amount only           | Low        |
| 2    | + Card info           | Moderate   |
| 3    | + Email + Address     | Good       |
| 4    | + Device + Identity   | High       |
| 5    | + UID history (precomputed) | Maximum |

### Risk Levels
| Score   | Level     | Action  |
|---------|-----------|---------|
| 0.0–0.3 | 🟢 LOW    | APPROVE |
| 0.3–0.7 | 🟡 MEDIUM | REVIEW  |
| 0.7–1.0 | 🔴 HIGH   | BLOCK   |
""",
    version="2.1.0"
)

# Allow local frontend / demos to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    status = get_model_status()
    # Return 503 when models aren't loaded so load-balancers know to wait
    if status.get("status") != "healthy":
        raise HTTPException(status_code=503, detail=status)
    return HealthResponse(**status)


@app.post("/score", response_model=FraudScoreResponse, tags=["Scoring"])
def score_single(request: TransactionRequest):
    """
    Score a single transaction. Only TransactionAmt is required.
    Always calls Claude for a natural-language explanation.
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
            model=result['model'],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score-batch", response_model=BatchFraudScoreResponse, tags=["Scoring"])
def score_batch(request: BatchTransactionRequest):
    """
    Score up to 50 transactions.
    HIGH-risk transactions get a full Claude explanation.
    MEDIUM / LOW-risk get a fast rule-based explanation (no Claude API call).
    This keeps batch latency low and Claude costs minimal.
    """
    try:
        results = []
        for txn in request.transactions:
            txn_dict = txn.model_dump(exclude_none=False)
            risk_score, top_features, tier = score_transaction(txn_dict)

            # Only call Claude for HIGH-risk transactions
            if risk_score > 0.7:
                result = explain_transaction(txn_dict, risk_score, top_features, tier)
            else:
                result = fast_explain(txn_dict, risk_score, top_features, tier)

            results.append(FraudScoreResponse(
                risk_score=result['risk_score'],
                risk_level=result['risk_level'],
                recommended_action=result['recommended_action'],
                explanation=result['explanation'],
                top_features=[FeatureImpact(**f) for f in result['top_features']],
                feature_tier=result['feature_tier'],
                tier_note=result['tier_note'],
                model=result['model'],
            ))

        return BatchFraudScoreResponse(results=results, total=len(results))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Fraud Detection API v2.1",
        "docs":    "Visit /docs for Swagger UI",
        "health":  "Visit /health for model status",
    }