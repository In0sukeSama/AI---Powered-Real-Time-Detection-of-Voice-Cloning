"""
test_detector.py
-----------------
Unit tests for the Audio / Deepfake Detection component.

Run with:
    cd deepfake_detection/tests && python -m pytest test_detector.py -v
or:
    python -m unittest test_detector.py
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deepfake_detector import DeepfakeDetector, DetectionResult
from audio_features import extract_feature_vector, FEATURE_NAMES
from synthetic_dataset import generate_dataset, SAMPLE_RATE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "deepfake_model.pkl")


class TestInterfaceContract(unittest.TestCase):
    """Verify output always matches the agreed team interface contract."""

    def setUp(self):
        self.detector = DeepfakeDetector(mock=True)

    def test_mock_output_shape(self):
        result = self.detector.predict(signal=np.random.randn(16000), sample_rate=16000)
        self.assertIn("deepfake_score", result)
        self.assertIn("classification", result)
        self.assertEqual(len(result), 2, "Output must only contain contract fields")

    def test_score_is_float_in_range(self):
        result = self.detector.predict(signal=np.random.randn(16000), sample_rate=16000)
        self.assertIsInstance(result["deepfake_score"], float)
        self.assertGreaterEqual(result["deepfake_score"], 0.0)
        self.assertLessEqual(result["deepfake_score"], 1.0)

    def test_classification_is_valid_label(self):
        result = self.detector.predict(signal=np.random.randn(16000), sample_rate=16000)
        self.assertIn(result["classification"], ("SYNTHETIC", "HUMAN"))

    def test_classification_consistent_with_score(self):
        for _ in range(20):
            sig = np.random.randn(16000).astype(np.float32) * np.random.rand()
            result = self.detector.predict(signal=sig, sample_rate=16000)
            if result["deepfake_score"] >= self.detector.threshold:
                self.assertEqual(result["classification"], "SYNTHETIC")
            else:
                self.assertEqual(result["classification"], "HUMAN")


class TestMockMode(unittest.TestCase):
    """Mock mode must be usable by other members before the real model exists."""

    def test_mock_deterministic_for_same_input_key(self):
        detector = DeepfakeDetector(mock=True)
        r1 = detector.predict(audio_path="call_abc.wav")
        r2 = detector.predict(audio_path="call_abc.wav")
        self.assertEqual(r1, r2, "Mock output should be reproducible for the same input")

    def test_mock_varies_across_different_inputs(self):
        detector = DeepfakeDetector(mock=True)
        r1 = detector.predict(audio_path="call_1.wav")
        r2 = detector.predict(audio_path="call_2.wav")
        self.assertNotEqual(
            r1["deepfake_score"], r2["deepfake_score"],
            "Mock scores should differ across distinct inputs (not hardcoded)"
        )

    def test_mock_does_not_require_model_file(self):
        # Should not raise even if no trained model exists on disk
        detector = DeepfakeDetector(mock=True)
        result = detector.predict(signal=np.zeros(1000), sample_rate=16000)
        self.assertIn("deepfake_score", result)


class TestFeatureExtraction(unittest.TestCase):

    def test_feature_vector_shape(self):
        sig = np.random.randn(16000).astype(np.float32)
        features = extract_feature_vector(sig, 16000)
        self.assertEqual(features.shape[0], len(FEATURE_NAMES))

    def test_feature_vector_no_nans(self):
        sig = np.random.randn(16000).astype(np.float32)
        features = extract_feature_vector(sig, 16000)
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())

    def test_handles_silence(self):
        sig = np.zeros(16000, dtype=np.float32)
        features = extract_feature_vector(sig, 16000)
        self.assertEqual(features.shape[0], len(FEATURE_NAMES))
        self.assertFalse(np.isnan(features).any())

    def test_handles_very_short_clip(self):
        sig = np.random.randn(200).astype(np.float32)  # shorter than one frame
        features = extract_feature_vector(sig, 16000)
        self.assertEqual(features.shape[0], len(FEATURE_NAMES))
        self.assertFalse(np.isnan(features).any())

    def test_handles_empty_signal(self):
        sig = np.array([], dtype=np.float32)
        features = extract_feature_vector(sig, 16000)
        self.assertEqual(features.shape[0], len(FEATURE_NAMES))


@unittest.skipUnless(os.path.exists(MODEL_PATH), "Trained model not found; run train_model.py first")
class TestRealModelInference(unittest.TestCase):
    """
    Sanity checks against the trained placeholder model. These validate the
    pipeline wiring (features -> scaler -> classifier -> contract), not
    real-world deepfake-detection accuracy (see docs/README.md limitations).
    """

    def setUp(self):
        self.detector = DeepfakeDetector(model_path=MODEL_PATH)

    def test_real_inference_returns_contract_shape(self):
        signals, labels, sr = generate_dataset(n_per_class=2, seed=1)
        for sig in signals:
            result = self.detector.predict(signal=sig, sample_rate=sr)
            self.assertIn("deepfake_score", result)
            self.assertIn(result["classification"], ("SYNTHETIC", "HUMAN"))

    def test_separates_placeholder_classes_reasonably(self):
        # Not a claim about real-world performance -- just confirms the
        # trained pipeline recovers the (easy, synthetic) training signal.
        signals, labels, sr = generate_dataset(n_per_class=5, seed=99)
        correct = 0
        for sig, label in zip(signals, labels):
            result = self.detector.predict(signal=sig, sample_rate=sr)
            predicted = "synthetic" if result["classification"] == "SYNTHETIC" else "human"
            correct += int(predicted == label)
        accuracy = correct / len(signals)
        self.assertGreater(accuracy, 0.7)

    def test_silence_does_not_crash(self):
        result = self.detector.predict(signal=np.zeros(16000), sample_rate=16000)
        self.assertIn("deepfake_score", result)


class TestErrorHandling(unittest.TestCase):

    def test_missing_model_raises_clear_error_when_not_mock(self):
        with self.assertRaises(FileNotFoundError):
            DeepfakeDetector(model_path="/nonexistent/path/model.pkl", mock=False)

    def test_predict_requires_input(self):
        detector = DeepfakeDetector(mock=True)
        with self.assertRaises(ValueError):
            detector.predict()


if __name__ == "__main__":
    unittest.main(verbosity=2)
