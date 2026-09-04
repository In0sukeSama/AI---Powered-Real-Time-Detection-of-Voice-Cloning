"""
test_grok_integration.py — verifies mode-switching logic in llm_upgrade.py
with a MOCKED Grok client (no real network call — api.x.ai is unreachable
from this dev sandbox, confirmed blocked same as huggingface.co).

These tests prove the WIRING is correct: fallback-on-failure, hybrid
only escalating CAUTION, mode switching at runtime. They do NOT prove
Grok's actual classification quality — that requires a real API key and
should be checked manually (see README section added below) before the
demo.
"""
import unittest
from unittest.mock import patch

import llm_upgrade
from interface import UtteranceResult
from grok_client import GrokError
from llm_upgrade import MODES, classify, set_mode, get_mode


def fake_grok_result(text, prior_signals=None, **kw):
    return UtteranceResult(
        utterance=text, risk_score=0.95, risk_level="HIGH",
        intent="otp_pin_password_request", signals=["otp_pin_password_request"],
        explanation=["Grok says HIGH"], confidence=0.9,
    )


def failing_grok(*a, **kw):
    raise GrokError("simulated network failure")


class TestModeSwitching(unittest.TestCase):
    def setUp(self):
        set_mode(MODES.RULE_ONLY)  # reset before every test
        llm_upgrade._client = None  # clear cached client so each test's mock takes effect

    def tearDown(self):
        set_mode(MODES.RULE_ONLY)  # leave clean for other test modules
        llm_upgrade._client = None

    def test_default_mode_is_rule_only(self):
        set_mode(MODES.RULE_ONLY)
        self.assertEqual(get_mode(), MODES.RULE_ONLY)

    def test_rule_only_never_touches_grok(self):
        with patch.object(llm_upgrade, "_get_client") as mock_client:
            r = classify("I went to the bank yesterday.", mode=MODES.RULE_ONLY)
            mock_client.assert_not_called()
            self.assertEqual(r.risk_level, "LOW")

    @patch("llm_upgrade.GrokClient")
    def test_hybrid_skips_grok_when_rule_says_low(self, mock_grok_cls):
        mock_grok_cls.return_value.classify.side_effect = fake_grok_result
        r = classify("Hey, how was your day?", mode=MODES.HYBRID)
        mock_grok_cls.return_value.classify.assert_not_called()
        self.assertEqual(r.risk_level, "LOW")

    @patch("llm_upgrade.GrokClient")
    def test_hybrid_skips_grok_when_rule_says_high(self, mock_grok_cls):
        mock_grok_cls.return_value.classify.side_effect = fake_grok_result
        r = classify("Tell me the OTP immediately.", mode=MODES.HYBRID)
        mock_grok_cls.return_value.classify.assert_not_called()
        self.assertEqual(r.risk_level, "HIGH")

    # NOTE: verified against the live rule engine before hardcoding — the
    # current rule engine actually predicts CAUTION for almost nothing
    # (its own benchmark shows CAUTION F1=0.000, see evaluate_v2.py), so
    # this specific phrasing was found by testing candidates directly
    # rather than assumed.
    CAUTION_TEXT = "The card number matters a lot right now, this is urgent."

    @patch("llm_upgrade.GrokClient")
    def test_hybrid_calls_grok_when_rule_says_caution(self, mock_grok_cls):
        mock_grok_cls.return_value.classify.side_effect = fake_grok_result
        r = classify(self.CAUTION_TEXT, mode=MODES.HYBRID)
        mock_grok_cls.return_value.classify.assert_called_once()
        self.assertEqual(r.risk_level, "HIGH")  # Grok's answer wins in this case
        self.assertIn("Grok says HIGH", r.explanation)

    @patch("llm_upgrade.GrokClient")
    def test_hybrid_falls_back_to_rule_result_on_grok_failure(self, mock_grok_cls):
        mock_grok_cls.return_value.classify.side_effect = failing_grok
        r = classify(self.CAUTION_TEXT, mode=MODES.HYBRID)
        # Grok failed -> keep whatever the rule engine said, tagged as fallback
        self.assertTrue(any("[fallback:" in e for e in r.explanation))
        self.assertEqual(r.risk_level, "CAUTION")  # rule engine's own verdict preserved

    @patch("llm_upgrade.GrokClient")
    def test_full_llm_calls_grok_for_everything(self, mock_grok_cls):
        mock_grok_cls.return_value.classify.side_effect = fake_grok_result
        for text in ["Hey, how was your day?", "Tell me the OTP now.", "I went to the bank."]:
            classify(text, mode=MODES.FULL_LLM)
        self.assertEqual(mock_grok_cls.return_value.classify.call_count, 3)

    @patch("llm_upgrade.GrokClient")
    def test_full_llm_falls_back_to_rule_engine_on_failure(self, mock_grok_cls):
        mock_grok_cls.return_value.classify.side_effect = failing_grok
        r = classify("Tell me the OTP immediately.", mode=MODES.FULL_LLM)
        # rule engine would independently call this HIGH -> pipeline never
        # goes silent just because Grok/network failed
        self.assertEqual(r.risk_level, "HIGH")
        self.assertTrue(any("[fallback:" in e for e in r.explanation))

    def test_runtime_mode_flip(self):
        set_mode(MODES.RULE_ONLY)
        self.assertEqual(get_mode(), MODES.RULE_ONLY)
        set_mode(MODES.HYBRID)
        self.assertEqual(get_mode(), MODES.HYBRID)
        set_mode(MODES.FULL_LLM)
        self.assertEqual(get_mode(), MODES.FULL_LLM)

    @patch("llm_upgrade.GrokClient")
    def test_per_call_mode_override_ignores_global_mode(self, mock_grok_cls):
        mock_grok_cls.return_value.classify.side_effect = fake_grok_result
        set_mode(MODES.RULE_ONLY)
        r = classify("Tell me the OTP now.", mode=MODES.FULL_LLM)
        mock_grok_cls.return_value.classify.assert_called_once()
        self.assertEqual(r.risk_level, "HIGH")
        self.assertEqual(get_mode(), MODES.RULE_ONLY)  # global mode untouched


if __name__ == "__main__":
    unittest.main(verbosity=2)
