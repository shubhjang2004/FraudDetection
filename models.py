"""
models.py - Pydantic request/response schemas.
All fields optional except TransactionAmt — graceful degradation.

CHANGES FROM ORIGINAL:
  - Replaced deprecated class Config inner-class with model_config = {} (Pydantic v2 style).
  - Added json_schema_extra directly on model_config.
  - No field-level changes.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class TransactionRequest(BaseModel):
    """
    Fraud scoring request. Only TransactionAmt is required.
    More fields = better score accuracy.
    Tier 1 (amt only) → Tier 5 (full data with UID history).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "TransactionAmt": 117.50,
                "TransactionDT": 86400,
                "card1": 12345,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "DeviceType": "desktop",
            }
        }
    )

    # Tier 1 — always required
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount USD")
    TransactionDT:  Optional[int]   = Field(None, description="Seconds offset from reference time")
    ProductCD:      Optional[str]   = Field(None, description="Product code W/H/C/S/R")

    # Tier 2 — card info
    card1: Optional[int]   = Field(None, description="Card identifier 1")
    card2: Optional[float] = Field(None, description="Card identifier 2")
    card3: Optional[float] = Field(None, description="Card identifier 3")
    card4: Optional[str]   = Field(None, description="Card network: visa/mastercard/etc")
    card5: Optional[float] = Field(None, description="Card identifier 5")
    card6: Optional[str]   = Field(None, description="Card type: debit/credit")

    # Tier 3 — address + email
    addr1:         Optional[float] = Field(None, description="Billing zip")
    addr2:         Optional[float] = Field(None, description="Billing country")
    P_emaildomain: Optional[str]   = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str]   = Field(None, description="Recipient email domain")
    dist1:         Optional[float] = None
    dist2:         Optional[float] = None

    # Tier 4 — device + identity
    DeviceType: Optional[str] = Field(None, description="desktop/mobile")
    DeviceInfo: Optional[str] = None

    # Identity numeric (id_01 to id_11)
    id_01: Optional[float] = None
    id_02: Optional[float] = None
    id_03: Optional[float] = None
    id_04: Optional[float] = None
    id_05: Optional[float] = None
    id_06: Optional[float] = None
    id_07: Optional[float] = None
    id_08: Optional[float] = None
    id_09: Optional[float] = None
    id_10: Optional[float] = None
    id_11: Optional[float] = None

    # Identity categorical (id_12 to id_38)
    id_12: Optional[str]   = None
    id_13: Optional[float] = None
    id_14: Optional[float] = None
    id_15: Optional[str]   = None
    id_16: Optional[str]   = None
    id_17: Optional[float] = None
    id_18: Optional[float] = None
    id_19: Optional[float] = None
    id_20: Optional[float] = None
    id_21: Optional[float] = None
    id_22: Optional[float] = None
    id_23: Optional[str]   = None
    id_24: Optional[float] = None
    id_25: Optional[float] = None
    id_26: Optional[float] = None
    id_27: Optional[str]   = None
    id_28: Optional[str]   = None
    id_29: Optional[str]   = None
    id_30: Optional[str]   = None
    id_31: Optional[str]   = None
    id_32: Optional[float] = None
    id_33: Optional[str]   = None
    id_34: Optional[str]   = None
    id_35: Optional[str]   = None
    id_36: Optional[str]   = None
    id_37: Optional[str]   = None
    id_38: Optional[str]   = None

    # Tier 5 — D columns for UID construction
    D1:  Optional[float] = None
    D2:  Optional[float] = None
    D3:  Optional[float] = None
    D15: Optional[float] = None

    # C columns
    C1:  Optional[float] = None
    C2:  Optional[float] = None
    C3:  Optional[float] = None
    C4:  Optional[float] = None
    C5:  Optional[float] = None
    C6:  Optional[float] = None
    C9:  Optional[float] = None
    C11: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None

    # M columns
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None


class FeatureImpact(BaseModel):
    feature:     str
    value:       float
    shap_impact: float


class FraudScoreResponse(BaseModel):
    risk_score:          float
    risk_level:          str
    recommended_action:  str
    explanation:         str
    top_features:        list[FeatureImpact]
    feature_tier:        int   = Field(..., description="1=minimal  5=full data")
    tier_note:           str   = Field(..., description="Data confidence description")
    model:               str


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., max_length=50)


class BatchFraudScoreResponse(BaseModel):
    results: list[FraudScoreResponse]
    total:   int


class HealthResponse(BaseModel):
    status:         str
    models_loaded:  Optional[list[str]] = None
    feature_count:  Optional[int]       = None
    tier5_ready:    Optional[bool]      = None
    uid_count:      Optional[int]       = None
    error:          Optional[str]       = None