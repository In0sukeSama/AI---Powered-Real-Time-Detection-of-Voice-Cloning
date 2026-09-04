"""
Mock producers for Member 1 (NLP) and Member 2 (Deepfake).
Used to test the Risk Fusion Engine before real modules are integrated.
"""


def mock_deepfake(score, classification=None):
    """
    Create a mock deepfake detector output.
    """

    if classification is None:
        classification = "SYNTHETIC" if score >= 0.5 else "REAL"

    return {
        "deepfake_score": score,
        "classification": classification
    }


def mock_nlp(score, intent="none", signals=None):
    """
    Create a mock NLP risk detector output.
    """

    if signals is None:
        signals = []

    if score >= 0.8:
        risk_level = "HIGH"
    elif score >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "intent": intent,
        "signals": signals
    }


def build_payload(deepfake, nlp):
    """
    Combine Member 1 and Member 2 outputs
    into the format expected by the fusion engine.
    """

    return {
        **deepfake,
        "nlp": nlp
    }


SCENARIOS = {
    "1_real_normal": build_payload(
        mock_deepfake(0.10),
        mock_nlp(0.05)
    ),

    "2_synthetic_normal": build_payload(
        mock_deepfake(0.91),
        mock_nlp(0.08)
    ),

    "3_real_suspicious": build_payload(
        mock_deepfake(0.12),
        mock_nlp(
            0.88,
            intent="financial_fraud",
            signals=[
                "sensitive_information_request",
                "urgency"
            ]
        )
    ),

    "4_synthetic_suspicious": build_payload(
        mock_deepfake(0.93),
        mock_nlp(
            0.87,
            intent="financial_fraud",
            signals=[
                "sensitive_information_request",
                "urgency",
                "otp_request"
            ]
        )
    ),

    "5_hard_negative": build_payload(
        mock_deepfake(0.15),
        mock_nlp(0.10)
    )
}