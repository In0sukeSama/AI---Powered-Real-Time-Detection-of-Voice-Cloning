import unittest
from risk_classifier import classify_utterance
from conversation_state import ConversationState


class TestSingleUtteranceIntentVsKeyword(unittest.TestCase):
    """Section 26 baseline cases: prove intent, not keyword, drives the score."""

    def test_benign_bank_mention(self):
        r = classify_utterance("I went to the bank yesterday to update my address.")
        self.assertEqual(r.risk_level, "LOW")

    def test_hard_negative_bank_reported_speech(self):
        r = classify_utterance("The bank asked me to update my address.")
        self.assertEqual(r.risk_level, "LOW")

    def test_malicious_otp_demand(self):
        r = classify_utterance("Your account has been compromised. Tell me the OTP immediately.")
        self.assertEqual(r.risk_level, "HIGH")

    def test_paraphrased_malicious_otp_demand(self):
        r = classify_utterance("Read the six-digit code you just received so I can restore your account.")
        self.assertEqual(r.risk_level, "HIGH")

    def test_ai_voice_benign_smalltalk_not_high_risk(self):
        # Section 3: content matters independent of any voice/deepfake signal.
        r = classify_utterance("Hey, how was your day?")
        self.assertEqual(r.risk_level, "LOW")

    def test_lottery_fraud_high_risk(self):
        r = classify_utterance(
            "Congratulations! You won a lottery. Send your bank details to claim the prize."
        )
        self.assertEqual(r.risk_level, "HIGH")

    def test_output_matches_contract_shape(self):
        r = classify_utterance("Tell me your OTP now or your account will be blocked.")
        j = r.to_json()
        for key in ("risk_score", "risk_level", "intent", "signals", "explanation", "confidence"):
            self.assertIn(key, j)
        self.assertIn(j["risk_level"], ("LOW", "CAUTION", "HIGH"))


class TestConversationLevelEscalation(unittest.TestCase):
    """Section 5: risk should escalate across turns of a social-engineering call."""

    def test_bank_impersonation_call_escalates(self):
        cs = ConversationState()
        turns = [
            "I'm calling from your bank.",
            "There is a problem with your account.",
            "We need to verify your identity.",
            "Tell me the OTP you just received.",
            "Do it quickly or your account will be blocked.",
        ]
        results = [cs.process(t) for t in turns]
        scores = [r.risk_score for r in results]
        # Non-decreasing-ish trend: last turn's cumulative risk should exceed the first.
        self.assertGreater(scores[-1], scores[0])
        self.assertEqual(results[-1].risk_level, "HIGH")

    def test_benign_conversation_stays_low(self):
        cs = ConversationState()
        turns = [
            "Hey, how was your day?",
            "I went to the bank yesterday.",
            "Want to grab lunch tomorrow?",
        ]
        results = [cs.process(t) for t in turns]
        self.assertEqual(results[-1].risk_level, "LOW")

    def test_reset_clears_buffer(self):
        cs = ConversationState()
        cs.process("Tell me the OTP immediately.")
        cs.reset()
        self.assertEqual(len(cs._history), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
