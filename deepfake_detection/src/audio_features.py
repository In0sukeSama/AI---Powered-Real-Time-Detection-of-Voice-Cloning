"""
audio_features.py
-----------------
Feature extraction for the voice deepfake detector.

This version uses the same feature pipeline as our tested baseline:
    - 13 MFCC means
    - 13 MFCC standard deviations
    - spectral centroid
    - spectral bandwidth
    - spectral rolloff
    - zero-crossing rate

Total: 30 features.

Audio is normalized to:
    - 16 kHz sample rate
    - mono
    - exactly 4 seconds
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 4
TARGET_LENGTH = TARGET_SAMPLE_RATE * TARGET_DURATION


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and convert it to mono 16 kHz audio.

    Returns
    -------
    samples : np.ndarray
        Mono float32 audio.
    sample_rate : int
        Always 16000 Hz.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    samples, sample_rate = sf.read(path, dtype="float32")

    # Convert stereo/multi-channel audio to mono.
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    samples = samples.astype(np.float32)

    # Resample to 16 kHz if necessary.
    if sample_rate != TARGET_SAMPLE_RATE:
        samples = librosa.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE
        )

    # Force exactly 4 seconds.
    if len(samples) > TARGET_LENGTH:
        samples = samples[:TARGET_LENGTH]

    elif len(samples) < TARGET_LENGTH:
        samples = np.pad(
            samples,
            (0, TARGET_LENGTH - len(samples))
        )

    return samples.astype(np.float32), TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_feature_vector(
    signal: np.ndarray,
    sr: int
) -> np.ndarray:
    """
    Extract the 30-dimensional feature vector used by the detector.

    Features:
        13 MFCC means
        13 MFCC standard deviations
        spectral centroid
        spectral bandwidth
        spectral rolloff
        zero-crossing rate

    Returns
    -------
    np.ndarray
        Shape: (30,)
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.size == 0:
        signal = np.zeros(
            TARGET_LENGTH,
            dtype=np.float32
        )

    # Convert to mono if necessary.
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Resample if input is not 16 kHz.
    if sr != TARGET_SAMPLE_RATE:
        signal = librosa.resample(
            signal,
            orig_sr=sr,
            target_sr=TARGET_SAMPLE_RATE
        )

    # Force exactly 4 seconds.
    if len(signal) > TARGET_LENGTH:
        signal = signal[:TARGET_LENGTH]

    elif len(signal) < TARGET_LENGTH:
        signal = np.pad(
            signal,
            (0, TARGET_LENGTH - len(signal))
        )

    sr = TARGET_SAMPLE_RATE

    # -----------------------------------------------------------------------
    # MFCC
    # -----------------------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # -----------------------------------------------------------------------
    # Spectral features
    # -----------------------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    ).mean()

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    ).mean()

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr
    ).mean()

    # -----------------------------------------------------------------------
    # Zero crossing rate
    # -----------------------------------------------------------------------

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        signal
    ).mean()

    # -----------------------------------------------------------------------
    # Combine all features
    # -----------------------------------------------------------------------

    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zero_crossing_rate
        ]
    ])

    return feature_vector.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + [f"mfcc_std_{i}" for i in range(13)]
    + [
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "zero_crossing_rate"
    ]
)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("Feature extractor loaded successfully.")
    print("Number of features:", len(FEATURE_NAMES))

    if len(FEATURE_NAMES) == 30:
        print("Feature count check: PASS")
    else:
        print("Feature count check: FAIL")