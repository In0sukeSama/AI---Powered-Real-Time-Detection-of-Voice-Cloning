"""
synthetic_dataset.py
---------------------
Generates a small synthetic waveform dataset that caricatures two classes:

  * "human"    - voiced signals with natural pitch jitter, formant-like
                 resonances, and broadband noise (breathiness), which
                 approximate some statistical properties of real speech.
  * "synthetic"- signals that are unnaturally periodic (near-zero jitter),
                 spectrally flatter/smoother, and have a cleaner harmonic
                 structure -- properties commonly reported for
                 vocoder/TTS output vs. natural recordings.

IMPORTANT / KNOWN LIMITATION
-----------------------------
This is a *placeholder* dataset used only to exercise the full pipeline
(feature extraction -> training -> inference -> interface contract) in an
offline environment with no internet access and no bundled audio corpus.

It is NOT a substitute for training on a real deepfake-speech corpus such
as ASVspoof 2019/2021, WaveFake, or In-the-Wild. Before this component is
considered "real", retrain `train_model.py` on real data using the same
`extract_feature_vector` pipeline. The interface contract
(deepfake_score / classification) will not need to change.
"""

from __future__ import annotations

import numpy as np


SAMPLE_RATE = 16000


def _make_human_like(duration_s: float, rng: np.random.Generator) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE

    f0_base = rng.uniform(90, 220)  # fundamental frequency range (voice pitch)
    # natural pitch jitter: small random walk in F0
    jitter_walk = np.cumsum(rng.normal(0, 0.8, size=n)) / SAMPLE_RATE
    f0 = f0_base + jitter_walk * 50

    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    signal = np.zeros(n)
    # a handful of harmonics with natural falloff + slight random detune (formant-ish)
    for k in range(1, 6):
        detune = rng.normal(0, 0.002)
        amp = 1.0 / k ** 1.2
        signal += amp * np.sin(k * phase * (1 + detune))

    # breathiness / broadband noise component (natural voices are noisier)
    noise = rng.normal(0, 0.05, size=n)
    signal = signal + noise

    # amplitude envelope (natural loudness variation)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(1.5, 3.5) * t + rng.uniform(0, np.pi))
    signal = signal * env

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    return signal.astype(np.float32)


def _make_synthetic_like(duration_s: float, rng: np.random.Generator) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE

    f0 = rng.uniform(90, 220)  # near-constant F0: unnaturally stable pitch
    phase = 2 * np.pi * f0 * t

    signal = np.zeros(n)
    for k in range(1, 6):
        amp = 1.0 / k  # cleaner harmonic falloff (more "ideal")
        signal += amp * np.sin(k * phase)

    # very low, smooth noise floor (vocoders often produce cleaner output)
    noise = rng.normal(0, 0.005, size=n)
    signal = signal + noise

    # smoother, more regular amplitude envelope
    env = 0.7 + 0.3 * np.sin(2 * np.pi * 2.0 * t)
    signal = signal * env

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    return signal.astype(np.float32)


def generate_dataset(n_per_class: int = 60, seed: int = 42):
    """
    Returns
    -------
    signals : list[np.ndarray]
    labels  : list[str]  ("human" | "synthetic")
    sample_rate : int
    """
    rng = np.random.default_rng(seed)
    signals, labels = [], []

    for _ in range(n_per_class):
        duration = rng.uniform(1.5, 3.0)
        signals.append(_make_human_like(duration, rng))
        labels.append("human")

    for _ in range(n_per_class):
        duration = rng.uniform(1.5, 3.0)
        signals.append(_make_synthetic_like(duration, rng))
        labels.append("synthetic")

    return signals, labels, SAMPLE_RATE
