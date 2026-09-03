import pytest

from src.risk.fusion import (
    FusionConfig,
    InvalidRiskInputError,
    fuse_risk,
)


def make_payload(deepfake_score, nlp_score, signals=None):
    return {
        "deepfake_score": deepfake_score,
        "classification": (
            "SYNTHETIC" if deepfake_score >= 0.5 else "REAL"
        ),
        "nlp": {
            "risk_score": nlp_score,
            "risk_level": "HIGH" if nlp_score >= 0.8
            else "CAUTION" if nlp_score >= 0.4
            else "LOW",
            "intent": "financial_fraud",
            "signals": signals or [],
        },
    }


def test_low_risk():
    result = fuse_risk(make_payload(0.1, 0.05))

    assert result.risk_level == "LOW"
    assert result.action == "MONITOR"


def test_synthetic_voice_alone_not_high():
    result = fuse_risk(make_payload(0.93, 0.08))

    assert result.risk_level != "HIGH"


def test_high_nlp_alone_is_high():
    result = fuse_risk(make_payload(0.12, 0.88))

    assert result.risk_level == "HIGH"
    assert result.action == "ALERT"


def test_high_deepfake_and_high_nlp():
    result = fuse_risk(make_payload(0.93, 0.87))

    assert result.risk_level == "HIGH"
    assert result.action == "ALERT"


def test_medium_risk():
    result = fuse_risk(make_payload(0.5, 0.5))

    assert result.risk_level in {"CAUTION", "HIGH"}


def test_out_of_range_scores_are_clamped():
    result = fuse_risk(make_payload(2.0, -1.0))

    assert 0.0 <= result.final_risk_score <= 1.0
    assert len(result.warnings) > 0


def test_missing_nlp_does_not_crash():
    payload = {
        "deepfake_score": 0.8,
        "classification": "SYNTHETIC",
    }

    result = fuse_risk(payload)

    assert 0.0 <= result.final_risk_score <= 1.0
    assert len(result.warnings) > 0


def test_missing_deepfake_uses_neutral_value():
    payload = {
        "nlp": {
            "risk_score": 0.2,
            "signals": [],
        }
    }

    result = fuse_risk(payload)

    assert 0.0 <= result.final_risk_score <= 1.0
    assert len(result.warnings) > 0


def test_nonnumeric_score_does_not_crash():
    result = fuse_risk(make_payload("invalid", 0.2))

    assert 0.0 <= result.final_risk_score <= 1.0
    assert len(result.warnings) > 0


def test_high_risk_has_explanation():
    result = fuse_risk(
        make_payload(
            0.93,
            0.87,
            ["sensitive_information_request", "urgency"],
        )
    )

    assert len(result.reasons) > 0


def test_low_risk_has_explanation():
    result = fuse_risk(make_payload(0.1, 0.05))

    assert len(result.reasons) > 0


def test_invalid_payload_type():
    with pytest.raises(InvalidRiskInputError):
        fuse_risk("invalid")


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        FusionConfig(
            weight_deepfake=0.5,
            weight_nlp=0.6,
        )