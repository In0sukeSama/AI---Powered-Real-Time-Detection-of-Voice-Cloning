"""
NLP / Conversational Intelligence — Interface Contract
Owner: Member 1

This module defines the STABLE interface between the NLP component and
the rest of the system (Risk Engine, Real-Time Pipeline). Per Section
28.2 of the project handoff, other members should only depend on this
contract, not on internal implementation details.

INPUT
-----
A single new utterance (str) of transcribed speech, plus the running
conversation state (see conversation_state.py). The NLP component does
NOT own audio or STT — it only consumes text.

OUTPUT
------
A UtteranceResult (per-utterance) is folded into a ConversationRiskResult
(cumulative), matching this JSON shape:

{
  "risk_score": 0.87,          # float 0.0-1.0
  "risk_level": "HIGH",        # LOW | CAUTION | HIGH
  "intent": "financial_fraud", # dominant intent label, or "benign"
  "signals": ["sensitive_information_request", "urgency"],
  "explanation": ["Requests sensitive banking information", "Creates urgency"],
  "confidence": 0.91
}

ERROR BEHAVIOR
---------------
- Empty/whitespace-only utterance -> risk_score 0.0, intent "benign",
  no exception raised.
- Unclassifiable/garbled text -> returns lowest-confidence LOW result
  rather than raising.

DEPENDENCIES
------------
- Pure Python standard library only for the baseline engine (no network
  calls, no model downloads) — satisfies Section 6 privacy requirements
  and keeps the component runnable offline in CI.
- Optional: an injectable `llm_classify_fn` callback for a future
  agentic/LLM-backed classifier. If not provided, the rule/pattern
  engine below is used. Swapping this in must NOT change the JSON
  contract above.

VERSION: 0.1.0 (baseline rule engine)
"""

from dataclasses import dataclass, field
from typing import List, Optional


RISK_LEVELS = ("LOW", "CAUTION", "HIGH")

RISK_CATEGORIES = (
    "financial_solicitation",
    "sensitive_information_request",
    "otp_pin_password_request",
    "payment_or_transfer_request",
    "lottery_prize_fraud",
    "bank_impersonation",
    "government_police_impersonation",
    "emergency_relative_scam",
    "account_compromise_claim",
    "threat_intimidation",
    "urgency_time_pressure",
    "secrecy_isolation_request",
    "remote_access_request",
    "software_installation_request",
    "suspicious_link_action_request",
    "personal_data_request",
)


@dataclass
class UtteranceResult:
    """Result of classifying a single utterance in context."""
    utterance: str
    risk_score: float
    risk_level: str
    intent: str
    signals: List[str] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_json(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 3),
            "risk_level": self.risk_level,
            "intent": self.intent,
            "signals": self.signals,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ConversationRiskResult:
    """Cumulative conversation-level result (Section 5)."""
    turn_count: int
    risk_score: float
    risk_level: str
    dominant_intent: str
    accumulated_signals: List[str] = field(default_factory=list)
    history: List[UtteranceResult] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "risk_score": round(self.risk_score, 3),
            "risk_level": self.risk_level,
            "intent": self.dominant_intent,
            "signals": self.accumulated_signals,
        }
