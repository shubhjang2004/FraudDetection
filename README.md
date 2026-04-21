# Fraud Detection System v2
**XGBoost + LightGBM + Claude — Graceful degradation across 5 feature tiers**

## Setup

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key" > .env

# Copy models from Colab
mkdir models/
cp best_xgb.pkl best_lgb.pkl features.pkl models/

# Optional: generate UID stats for Tier 5 (best accuracy)
python train.py --train_transaction data/train_transaction.csv \
                --train_identity    data/train_identity.csv

uvicorn main:app --reload
python test_score.py
python evaluate.py
```

| File | Purpose |
|------|---------|
| `features.py` | Tiered feature engineering + UID stats |
| `scorer.py` | Model inference + SHAP |
| `advisor.py` | Claude explanation layer |
| `models.py` | Pydantic schemas |
| `main.py` | FastAPI routes |
| `train.py` | Generate uid_stats.pkl |
| `evaluate.py` | Resume metrics |
| `test_score.py` | 5 test scenarios |
