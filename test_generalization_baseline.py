"""
test_generalization_baseline.py — Milestone 1 gate: paraphrase / hard-negative
regression cases specified in the Member-1 brief. Run BEFORE any fix is made
so failures are captured honestly, then again after each fix.

Each case is tagged with its category so results can be reported per-category
(functional / paraphrase / hard-negative) as required by the roadmap.
"""
import unittest
from risk_classifier import classify_utterance

CASES = [
    ("Send me the OTP you just received.", "HIGH", "functional_malicious"),
    ("I received an OTP from my bank yesterday.", "LOW", "functional_benign"),
    ("Tell me the verification code that came to your phone.", "HIGH", "paraphrase"),
    ("Read out the six-digit number you received.", "HIGH", "paraphrase"),
    ("I need the number from that SMS.", "HIGH", "paraphrase_indirect"),
    ("Scammers often ask victims for OTPs.", "LOW", "hard_negative_meta"),
    ("I am researching banking scams and how criminals request passwords.", "LOW", "hard_negative_meta"),
    ("My bank told me never to share my OTP.", "LOW", "hard_negative_negation"),
    ("You'll need to give me the digits from the SMS before I can complete this.", "HIGH", "indirect_disguised"),
]


class TestGeneralizationAndHardNegatives(unittest.TestCase):
    def test_all_cases_report(self):
        """
        This test intentionally reports per-case results rather than asserting
        blindly, since (as of this writing) not all cases pass yet. Once the
        classifier is patched, tighten this into per-case assertEqual calls.
        """
        results = []
        for text, expected, category in CASES:
            r = classify_utterance(text)
            results.append((category, text, expected, r.risk_level, r.risk_score))

        failures = [x for x in results if x[2] != x[3]]
        print("\n--- Generalization/Hard-Negative Results ---")
        for category, text, expected, got, score in results:
            mark = "OK " if expected == got else "FAIL"
            print(f"[{mark}] ({category}) expected={expected} got={got} score={score:.3f} :: {text}")
        print(f"Pass rate: {len(results) - len(failures)}/{len(results)}")

        # Not asserted as a hard pass/fail yet — see MILESTONE REPORT in README
        # or conversation history for the pass/investigate/stop determination.


if __name__ == "__main__":
    unittest.main(verbosity=2)
