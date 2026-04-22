"""
test_score.py - 5 test scenarios covering all feature tiers.

Run while the API is live:
    uvicorn main:app --reload   # in one terminal
    python test_score.py        # in another

Each scenario tests a different data tier and edge case.
"""

import json
import sys
import requests

BASE = "http://localhost:8000"


def check_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    if r.status_code != 200:
        print(f"❌ Health check failed: {r.status_code} — {r.text}")
        sys.exit(1)
    data = r.json()
    print(f"✅ API healthy | features={data.get('feature_count')} "
          f"| tier5={data.get('tier5_ready')}\n")


def score(payload: dict, label: str):
    r = requests.post(f"{BASE}/score", json=payload, timeout=30)
    if r.status_code != 200:
        print(f"❌ {label}: HTTP {r.status_code} — {r.text}\n")
        return
    d = r.json()
    print(f"{'='*60}")
    print(f"Scenario : {label}")
    print(f"Tier     : {d['feature_tier']}/5  ({d['tier_note']})")
    print(f"Score    : {d['risk_score']:.4f}  →  {d['risk_level']}")
    print(f"Action   : {d['recommended_action']}")
    print(f"Model    : {d['model']}")
    if d['top_features']:
        top = d['top_features'][0]
        print(f"Top feat : {top['feature']} = {top['value']:.3f} "
              f"(SHAP {top['shap_impact']:+.3f})")
    print(f"Explain  : {d['explanation'][:200]}...")
    print()


def score_batch(payloads: list[dict]):
    r = requests.post(f"{BASE}/score-batch",
                      json={"transactions": payloads}, timeout=60)
    if r.status_code != 200:
        print(f"❌ Batch: HTTP {r.status_code} — {r.text}\n")
        return
    d = r.json()
    print(f"{'='*60}")
    print(f"BATCH ({d['total']} transactions)")
    for i, res in enumerate(d['results']):
        print(f"  [{i+1}] {res['risk_level']:6s}  score={res['risk_score']:.4f}  "
              f"action={res['recommended_action']}  model={res['model']}")
    print()


if __name__ == "__main__":
    check_health()

    # ── Scenario 1: Tier 1 — amount only ──────────────────────────────────────
    score(
        {"TransactionAmt": 49.99},
        "TIER 1 — Amount only (low confidence)"
    )

    # ── Scenario 2: Tier 2 — card info ────────────────────────────────────────
    score(
        {
            "TransactionAmt": 299.00,
            "TransactionDT": 86400,
            "card1": 12345,
            "card2": 321.0,
            "card4": "visa",
            "card6": "debit",
        },
        "TIER 2 — Card info (moderate confidence)"
    )

    # ── Scenario 3: Tier 3 — suspicious (high-risk email, address mismatch) ───
    score(
        {
            "TransactionAmt": 2499.00,
            "TransactionDT": 3600 * 3,          # 3am
            "card1": 99999,
            "card4": "visa",
            "card6": "credit",
            "P_emaildomain": "protonmail.com",  # high-risk domain
            "R_emaildomain": "gmail.com",
            "addr1": 100.0,
            "addr2": 999.0,                     # mismatch
        },
        "TIER 3 — Suspicious: protonmail + address mismatch + night transaction"
    )

    # ── Scenario 4: Tier 4 — normal desktop transaction ───────────────────────
    score(
        {
            "TransactionAmt": 19.99,
            "TransactionDT": 86400 * 2 + 3600 * 14,  # Monday 2pm
            "card1": 55555,
            "card4": "mastercard",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "addr1": 200.0,
            "addr2": 200.0,
            "DeviceType": "desktop",
            "DeviceInfo": "Windows",
            "id_01": -1.0,
            "id_02": 500.0,
        },
        "TIER 4 — Normal desktop transaction (should be LOW)"
    )

    # ── Scenario 5: Tier 3 — large round-number transaction ───────────────────
    score(
        {
            "TransactionAmt": 5000.00,          # large round number
            "TransactionDT": 3600 * 23,         # 11pm
            "card1": 77777,
            "card4": "visa",
            "card6": "credit",
            "P_emaildomain": "yahoo.com",
            "R_emaildomain": "yahoo.com",
            "addr1": 300.0,
            "addr2": 300.0,
        },
        "TIER 3 — Large round-number late-night transaction"
    )

    # ── Scenario 6: Batch — 3 transactions (mix of risk levels) ──────────────
    score_batch([
        {"TransactionAmt": 12.50, "card1": 11111, "card4": "visa",
         "P_emaildomain": "gmail.com", "addr1": 100.0, "addr2": 100.0},
        {"TransactionAmt": 1500.00, "card1": 22222, "card4": "mastercard",
         "P_emaildomain": "protonmail.com", "addr1": 50.0, "addr2": 999.0},
        {"TransactionAmt": 89.99, "card1": 33333, "card4": "discover",
         "P_emaildomain": "hotmail.com", "addr1": 200.0, "addr2": 200.0},
    ])
