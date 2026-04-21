"""
advisor.py - Claude-powered fraud explanation layer.
Adapts explanation depth based on available feature tier.
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a fraud analyst AI at a payments company.
You receive transaction data, a fraud risk score (0-1), the feature tier 
(1=minimal data, 5=full data), and the top SHAP features driving the score.

Your job:
1. Explain in plain English WHY this transaction looks suspicious (or safe)
2. List the top 3 specific risk factors with actual values
3. Give a recommended action: BLOCK / REVIEW / APPROVE
4. If data is limited (tier 1-2), note that confidence is lower

Keep response to 4-6 sentences. Be direct. Use actual values from the transaction.
Write as a system report — no "I" or first person."""

TIER_CONTEXT = {
    1: "⚠️ LIMITED DATA — Only amount and time available. Low confidence score.",
    2: "📊 PARTIAL DATA — Card info available. Moderate confidence.",
    3: "📧 GOOD DATA — Card + email + address available. Good confidence.",
    4: "🔍 FULL DATA — Device and identity info available. High confidence.",
    5: "✅ COMPLETE DATA — Full UID history available. Maximum confidence.",
}


def explain_transaction(
    transaction: dict,
    risk_score: float,
    top_features: list[dict],
    tier: int = 3
) -> dict:
    """Call Claude to explain a fraud decision."""

    tier_note = TIER_CONTEXT.get(tier, "")

    features_text = "\n".join([
        f"  - {f['feature']}: value={f['value']:.3f}, SHAP impact={f['shap_impact']:+.3f}"
        for f in top_features[:5]
    ]) or "  No significant features available"

    prompt = f"""TRANSACTION FRAUD ANALYSIS

Data Confidence: {tier_note}

Transaction Details:
  Amount:           ${transaction.get('TransactionAmt', 'N/A')}
  Hour of Day:      {transaction.get('hour', 'N/A')}
  Is Weekend:       {transaction.get('is_weekend', 'N/A')}
  Is Night:         {transaction.get('is_night', 'N/A')}
  Card Type:        {transaction.get('card4', 'N/A')} / {transaction.get('card6', 'N/A')}
  Purchaser Email:  {transaction.get('P_emaildomain', 'N/A')}
  Recipient Email:  {transaction.get('R_emaildomain', 'N/A')}
  Address Mismatch: {transaction.get('addr_mismatch', 'N/A')}
  Device Type:      {transaction.get('DeviceType', 'N/A')}
  UID Tx Count:     {transaction.get('uid_count', 'N/A')}

Fraud Risk Score: {risk_score:.3f} ({risk_score*100:.1f}%)
Risk Level: {"🔴 HIGH" if risk_score > 0.7 else "🟡 MEDIUM" if risk_score > 0.3 else "🟢 LOW"}
Feature Tier: {tier}/5

Top SHAP Features Driving This Score:
{features_text}

Analyze this transaction and provide:
1. Plain-English explanation of the score
2. Top 3 risk factors with specific values
3. Recommended action: BLOCK / REVIEW / APPROVE
4. Confidence note if data is limited (tier < 3)
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    explanation = message.content[0].text

    action = "REVIEW"
    if "BLOCK"   in explanation.upper(): action = "BLOCK"
    elif "APPROVE" in explanation.upper(): action = "APPROVE"

    return {
        "risk_score":         round(risk_score, 4),
        "risk_level":         "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW",
        "recommended_action": action,
        "explanation":        explanation,
        "top_features":       top_features,
        "feature_tier":       tier,
        "tier_note":          tier_note,
        "model":              "claude-sonnet-4-20250514"
    }
