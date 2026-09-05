from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FusionConfig:
    weight_deepfake: float = 0.35
    weight_nlp: float = 0.65

    threshold_high: float = 0.70
    threshold_caution: float = 0.40

    min_nlp_for_high: float = 0.35
    nlp_alone_forces_high: float = 0.80

    def __post_init__(self):
        total = self.weight_deepfake + self.weight_nlp

        if abs(total - 1.0) > 1e-9:
            raise ValueError("Fusion weights must sum to 1.0")


DEFAULT_CONFIG = FusionConfig()


@dataclass
class RiskResult:
    final_risk_score: float
    risk_level: str
    action: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    inputs_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "final_risk_score": round(self.final_risk_score, 3),
            "risk_level": self.risk_level,
            "action": self.action,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "inputs_used": self.inputs_used,
        }


class InvalidRiskInputError(ValueError):
    pass


def _clamp_score(value, name, warnings):
    if value is None:
        warnings.append(f"{name} missing; using neutral score 0.5")
        return 0.5

    try:
        value = float(value)
    except (TypeError, ValueError):
        warnings.append(f"{name} invalid; using neutral score 0.5")
        return 0.5

    if value != value:
        warnings.append(f"{name} is NaN; using neutral score 0.5")
        return 0.5

    if value < 0 or value > 1:
        warnings.append(
            f"{name} out of range; clamped to [0, 1]"
        )
        value = max(0.0, min(1.0, value))

    return value


def _extract_reasons(nlp):
    reasons = []

    signals = nlp.get("signals", [])

    signal_reasons = {
        "sensitive_information_request":
            "Sensitive personal/banking information requested",

        "otp_request":
            "One-time password (OTP) or PIN requested",

        "urgency":
            "Urgency or pressure tactics detected",

        "impersonation":
            "Possible impersonation detected",

        "secrecy":
            "Secrecy or isolation tactics detected",

        "remote_access_request":
            "Remote-access request detected",

        "financial_solicitation":
            "Financial solicitation detected",

        "account_threat":
            "Threat involving account/security detected",
    }

    for signal in signals:
        if signal in signal_reasons:
            reasons.append(signal_reasons[signal])

    return reasons


def fuse_risk(payload, config=DEFAULT_CONFIG):
    if not isinstance(payload, dict):
        raise InvalidRiskInputError(
            "Risk payload must be a dictionary"
        )

    warnings = []
    reasons = []

    deepfake_score = _clamp_score(
        payload.get("deepfake_score"),
        "deepfake_score",
        warnings
    )

    nlp = payload.get("nlp")

    if not isinstance(nlp, dict):
        warnings.append(
            "NLP output missing or invalid; using neutral NLP score"
        )
        nlp = {}

    nlp_score = _clamp_score(
        nlp.get("risk_score"),
        "nlp_score",
        warnings
    )

    classification = str(
        payload.get("classification", "")
    ).upper()

    reasons.extend(_extract_reasons(nlp))

    if classification == "SYNTHETIC" or deepfake_score >= 0.60:
        reasons.append(
            "Voice shows strong synthetic/AI-generated characteristics"
        )

    final_score = (
        config.weight_deepfake * deepfake_score
        + config.weight_nlp * nlp_score
    )

    if final_score >= config.threshold_high:
        risk_level = "HIGH"
    elif final_score >= config.threshold_caution:
        risk_level = "CAUTION"
    else:
        risk_level = "LOW"

    # Severe conversational risk can force HIGH
    if nlp_score >= config.nlp_alone_forces_high:
        risk_level = "HIGH"
        warnings.append(
            f"Escalated to HIGH: nlp_score {nlp_score:.2f} "
            f">= nlp_alone_forces_high "
            f"{config.nlp_alone_forces_high:.2f} "
            "(severe conversational risk is treated as "
            "sufficient on its own, since a real-voice human "
            "scammer is still a real scam)"
        )

    # Safety rule:
    # synthetic voice alone must not automatically mean scam
    if risk_level == "HIGH" and nlp_score < config.min_nlp_for_high:
        risk_level = "CAUTION"
        warnings.append(
            f"Downgraded HIGH to CAUTION because nlp_score "
            f"{nlp_score:.2f} < min_nlp_for_high "
            f"{config.min_nlp_for_high:.2f}"
        )

    action_map = {
        "LOW": "MONITOR",
        "CAUTION": "WARN",
        "HIGH": "ALERT",
    }

    if not reasons:
        reasons.append(
            "No significant risk signals detected"
        )

    return RiskResult(
        final_risk_score=final_score,
        risk_level=risk_level,
        action=action_map[risk_level],
        reasons=reasons,
        warnings=warnings,
        inputs_used={
            "deepfake_score": deepfake_score,
            "nlp_score": nlp_score,
            "classification": classification,
        },
    )