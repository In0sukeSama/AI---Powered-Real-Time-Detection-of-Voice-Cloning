"""
integration/nlp_to_risk_adapter.py — thin glue between the NLP/Groq
component and Member 3's fuse_risk(). No architecture changes to either
side; this file absorbs the interface differences.

Signal name mapping is deliberately small and explicit. Unknown signals
(anything not in SIGNAL_MAP) pass through unchanged -- fuse_risk() already
ignores signals it doesn't recognize (see fusion.py's _extract_reasons),
so this never crashes, it just means that signal won't get a human-
readable reason line yet.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_upgrade import classify
from src.risk.fusion import fuse_risk

SIGNAL_MAP = {
    "otp_pin_password_request": "otp_request",
    "urgency_time_pressure": "urgency",
    "bank_impersonation": "impersonation",
    "government_police_impersonation": "impersonation",
    "secrecy_isolation_request": "secrecy",
    "account_compromise_claim": "account_threat",
    "sensitive_information_request": "sensitive_information_request",  # already matches
    "remote_access_request": "remote_access_request",  # already matches
    "payment_or_transfer_request": "financial_solicitation",
    "lottery_prize_fraud": "financial_solicitation",
}


def _map_signals(signals):
    return [SIGNAL_MAP.get(s, s) for s in signals]


def get_final_risk(text, prior_signals=None, deepfake_score=0.0, classification="REAL", mode=None):
    """
    Runs the full NLP -> fusion pipeline for one utterance.

    deepfake_score/classification default to a "REAL, no risk" mock so this
    can be called standalone (mock-audio testing) before Member 2's real
    output is wired in -- see integration/mock_audio.py for named presets,
    or integration/audio_adapter.py for Member 2's actual component.
    """
    nlp_result = classify(text, prior_signals=prior_signals, mode=mode)  # UtteranceResult

    payload = {
        "deepfake_score": deepfake_score,
        "classification": classification,
        "nlp": {
            "risk_score": nlp_result.risk_score,
            "signals": _map_signals(nlp_result.signals),
        },
    }

    fused = fuse_risk(payload)
    return {
        "nlp_result": nlp_result.to_json(),
        "fused_result": fused.to_dict(),
    }


def get_final_risk_from_audio(text, audio_result, prior_signals=None, mode=None):
    """
    Same as get_final_risk(), but takes Member 2's actual output dict
    directly (e.g. from integration.audio_adapter.get_audio_result())
    instead of hand-specified deepfake_score/classification. Passes
    Member 2's dict straight through with no key translation -- see the
    schema note in audio_adapter.py for why that's safe.
    """
    return get_final_risk(
        text, prior_signals=prior_signals, mode=mode,
        deepfake_score=audio_result["deepfake_score"],
        classification=audio_result["classification"],
    )


if __name__ == "__main__":
    result = get_final_risk("Tell me the OTP you just received.",
                             deepfake_score=0.1, classification="REAL")
    import json
    print(json.dumps(result, indent=2))
