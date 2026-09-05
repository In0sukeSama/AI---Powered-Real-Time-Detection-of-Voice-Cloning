"""
deepfake_detector.py
---------------------
Member 2 component: Audio / Deepfake Detection.

Implements the stable interface contract agreed with the rest of the team
(see Section 28.2 of the project handoff):

    {
        "deepfake_score": 0.93,     # float in [0, 1], higher = more likely synthetic
        "classification": "SYNTHETIC"  # "SYNTHETIC" | "HUMAN"
    }

This module is designed to be dropped into the Risk Fusion Engine (Member 3)
and the Real-Time Pipeline (Member 4) without either needing to know how
detection is implemented internally.

Usage
-----
    from deepfake_detector import DeepfakeDetector

    detector = DeepfakeDetector()  # loads trained model if available
    result = detector.predict(audio_path="call_clip.wav")
    # -> {"deepfake_score": 0.87, "classification": "SYNTHETIC"}

    result = detector.predict(signal=np.array([...]), sample_rate=16000)

Mock mode
---------
If dependent audio/model artifacts aren't available yet (e.g. teammates
integrating before your model is trained), use `DeepfakeDetector(mock=True)`
which returns structurally valid, deterministic-but-varied outputs so the
rest of the pipeline can be developed and tested independently
(Section 28.3 of the handoff: "Mock interfaces are mandatory").
"""

from __future__ import annotations

import os
import pickle
import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from audio_features import extract_feature_vector, load_wav

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "deepfake_model.pkl")
CLASSIFICATION_THRESHOLD = 0.5


@dataclass
class DetectionResult:
    deepfake_score: float
    classification: str

    def to_dict(self) -> dict:
        return {
            "deepfake_score": round(float(self.deepfake_score), 4),
            "classification": self.classification,
        }


class DeepfakeDetector:
    """
    Voice deepfake detector. Produces a synthetic-likelihood score and
    classification label for a given audio clip, per the team interface
    contract.

    Parameters
    ----------
    model_path : str
        Path to a trained model bundle (.pkl) produced by train_model.py.
    mock : bool
        If True, skip real inference and return mock outputs. Useful when
        this component isn't trained/ready yet but other members need to
        develop against the interface (see handoff Section 28.3).
    threshold : float
        Score threshold above which classification = "SYNTHETIC".
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        mock: bool = False,
        threshold: float = CLASSIFICATION_THRESHOLD,
    ):
        self.mock = mock
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self.sample_rate = 16000
        self.version = "mock-0.0"

        if not mock:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model bundle not found at {model_path}. "
                f"Run train_model.py first, or pass mock=True to develop "
                f"against the interface without a trained model."
            )
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.sample_rate = bundle.get("sample_rate", 16000)
        self.version = bundle.get("version", "unknown")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def predict(
        self,
        audio_path: Optional[str] = None,
        signal: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ) -> dict:
        """
        Run deepfake detection on an audio clip.

        Provide either `audio_path` (a WAV file) OR `signal` + `sample_rate`
        (an in-memory waveform, e.g. from a live-call audio buffer).

        Returns
        -------
        dict matching the interface contract:
            {"deepfake_score": float, "classification": "SYNTHETIC"|"HUMAN"}
        """
        if audio_path is None and signal is None:
            raise ValueError("Provide either audio_path or signal.")

        if self.mock:
            return self._mock_predict(audio_path or "in-memory-signal").to_dict()

        if audio_path is not None:
            signal, sr = load_wav(audio_path)
        else:
            sr = sample_rate or self.sample_rate

        if signal is None or len(signal) == 0:
            # Empty/silent input: known limitation, see docs. Return a
            # low-confidence neutral result rather than raising, so a
            # brief silence in a live call doesn't crash the pipeline.
            return DetectionResult(deepfake_score=0.5, classification="HUMAN").to_dict()

        features = extract_feature_vector(np.asarray(signal, dtype=np.float32), sr)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        score = float(self.model.predict_proba(features_scaled)[0, 1])

        classification = "SYNTHETIC" if score >= self.threshold else "HUMAN"
        return DetectionResult(deepfake_score=score, classification=classification).to_dict()

    # -----------------------------------------------------------------
    # Mock mode
    # -----------------------------------------------------------------

    def _mock_predict(self, seed_key: str) -> DetectionResult:
        """
        Deterministic pseudo-random mock output, seeded from input identity
        so the same file/signal always yields the same mock result during
        a testing session (helps teammates write reproducible integration
        tests against this mock).
        """
        digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        classification = "SYNTHETIC" if score >= self.threshold else "HUMAN"
        return DetectionResult(deepfake_score=score, classification=classification)


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            detector = DeepfakeDetector()
        except FileNotFoundError as e:
            print(f"[warn] {e}\nFalling back to mock mode.\n")
            detector = DeepfakeDetector(mock=True)
        print(detector.predict(audio_path=path))
    else:
        print("Usage: python deepfake_detector.py <path_to_wav>")
        print("Running mock-mode self-check instead...\n")
        detector = DeepfakeDetector(mock=True)
        print(detector.predict(audio_path="example_call.wav"))
