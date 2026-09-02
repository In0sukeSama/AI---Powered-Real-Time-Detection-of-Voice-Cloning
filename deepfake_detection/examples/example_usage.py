"""
example_usage.py
-----------------
Demonstrates the Audio / Deepfake Detection component end-to-end:
generating example clips, writing a WAV file, and running inference
both in real mode and mock mode.

Run:
    cd deepfake_detection/examples && python3 example_usage.py
"""

import os
import sys
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deepfake_detector import DeepfakeDetector
from synthetic_dataset import generate_dataset, SAMPLE_RATE


def save_wav(path: str, signal: np.ndarray, sr: int):
    pcm = np.clip(signal, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main():
    print("=== 1. Real-mode inference (trained model) ===")
    signals, labels, sr = generate_dataset(n_per_class=1, seed=7)

    example_dir = os.path.dirname(__file__)
    human_path = os.path.join(example_dir, "example_human.wav")
    synthetic_path = os.path.join(example_dir, "example_synthetic.wav")
    save_wav(human_path, signals[0], sr)
    save_wav(synthetic_path, signals[1], sr)

    detector = DeepfakeDetector()  # loads deepfake_model.pkl
    print(f"Ground truth: {labels[0]:10s} -> {detector.predict(audio_path=human_path)}")
    print(f"Ground truth: {labels[1]:10s} -> {detector.predict(audio_path=synthetic_path)}")

    print("\n=== 2. In-memory signal inference (no file I/O; live-call style) ===")
    live_buffer = signals[0]  # pretend this came from a rolling audio buffer
    result = detector.predict(signal=live_buffer, sample_rate=sr)
    print(f"Live buffer result: {result}")

    print("\n=== 3. Mock mode (for teammates before this model is ready) ===")
    mock_detector = DeepfakeDetector(mock=True)
    print(mock_detector.predict(audio_path="incoming_call_042.wav"))
    print(mock_detector.predict(audio_path="incoming_call_043.wav"))

    print("\n=== 4. Interface contract shape (what Member 3's Risk Engine receives) ===")
    result = detector.predict(audio_path=synthetic_path)
    assert set(result.keys()) == {"deepfake_score", "classification"}
    print("Contract OK:", result)


if __name__ == "__main__":
    main()
