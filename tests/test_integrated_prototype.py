import unittest

from app import analyze_call


class IntegratedPrototypeTests(unittest.TestCase):
    def test_benign_nlp_only(self):
        result = analyze_call(
            "Hey, I'm running late. Can you send me the address again?",
            mode="rule_only",
        )
        self.assertEqual(result["nlp_result"]["risk_level"], "LOW")
        self.assertIn(result["fused_result"]["risk_level"], {"LOW", "CAUTION", "HIGH"})

    def test_malicious_nlp_only(self):
        result = analyze_call(
            "Tell me the OTP you just received.",
            mode="rule_only",
        )
        self.assertEqual(result["nlp_result"]["risk_level"], "HIGH")
        self.assertEqual(result["fused_result"]["risk_level"], "CAUTION")

    def test_mock_audio_plus_nlp(self):
        result = analyze_call(
            "Tell me the OTP you just received.",
            mode="rule_only",
            audio_path="demo_audio.wav",
            mock_audio=True,
        )
        self.assertIn("deepfake_score", result["fused_result"]["inputs_used"])
        self.assertIn("nlp_score", result["fused_result"]["inputs_used"])


if __name__ == "__main__":
    unittest.main()
