"""
integration/audio_adapter.py — thin wrapper around Member 2's
DeepfakeDetector. No changes to Member 2's files.

NOTE ON A BUG IN MEMBER 2'S OWN CODE (not fixed here, just worked around):
deepfake_detector.py uses a relative import (`from .audio_features import
...`), which requires it to be imported as part of a package. Member 2's
own `examples/example_usage.py` imports it as a top-level module instead
(via sys.path insertion), which actually breaks with
"ImportError: attempted relative import with no known parent package" --
verified by running their example script exactly as documented. This
adapter avoids the problem by importing it correctly as
`deepfake_detection.src.deepfake_detector` (namespace package import),
which works without needing any __init__.py files added. Recommend
Member 2 fix their own example script separately; not touched here.

SCHEMA NOTE: Member 2's classification values are "SYNTHETIC"/"HUMAN".
Member 3's fuse_risk() only branches on the literal string "SYNTHETIC"
(falls back to the numeric deepfake_score >= 0.60 threshold otherwise) --
confirmed by reading fusion.py directly. So "HUMAN" vs "REAL" never
affects any fusion decision, only cosmetic pass-through text in the
output. No translation needed for this component -- pass Member 2's dict
straight through.

MODEL STATUS: no trained model bundle (deepfake_model.pkl) is present in
Member 2's submission -- only train_model.py exists, no artifact. Real
(non-mock) mode will raise FileNotFoundError until Member 2 trains and
ships that file. Defaults to mock=True here for that reason; flip to
mock=False the moment the .pkl exists.
"""
import os
import sys

_DEEPFAKE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "deepfake_detection")
_DEEPFAKE_ROOT = os.path.dirname(_DEEPFAKE_DIR)
_DEEPFAKE_SRC = os.path.join(_DEEPFAKE_DIR, "src")
# Member 2's detector uses a flat import (`from audio_features import ...`).
# Add its src directory so the detector can be imported without modifying Member 2's code.
sys.path.insert(0, _DEEPFAKE_SRC)
sys.path.insert(0, _DEEPFAKE_ROOT)

from deepfake_detection.src.deepfake_detector import DeepfakeDetector  # noqa: E402

_detector = None
_detector_mock_flag = None


def _get_detector(mock: bool = True):
    global _detector, _detector_mock_flag
    if _detector is None or _detector_mock_flag != mock:
        try:
            _detector = DeepfakeDetector(mock=mock)
        except FileNotFoundError as e:
            # No trained model yet -- fall back to mock rather than crash
            # the pipeline, same graceful behavior Member 2's own CLI uses.
            print(f"[audio_adapter] {e}\nFalling back to mock mode.")
            _detector = DeepfakeDetector(mock=True)
            mock = True
        _detector_mock_flag = mock
    return _detector


def get_audio_result(audio_path=None, signal=None, sample_rate=None, mock=True):
    """
    Returns Member 2's exact contract dict, unmodified:
        {"deepfake_score": float, "classification": "SYNTHETIC"|"HUMAN"}
    """
    detector = _get_detector(mock=mock)
    return detector.predict(audio_path=audio_path, signal=signal, sample_rate=sample_rate)


if __name__ == "__main__":
    print(get_audio_result(audio_path="example_call.wav", mock=True))
