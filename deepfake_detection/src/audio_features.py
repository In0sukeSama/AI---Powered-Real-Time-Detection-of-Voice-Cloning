"""
audio_features.py
------------------
Lightweight audio feature extraction for voice deepfake detection.

Design goal: zero heavyweight dependencies (no librosa/torchaudio required).
Uses only numpy + scipy + Python's built-in `wave` module so this component
can run in constrained / offline hackathon environments and is easy for
teammates to install.

If librosa is available in your environment, you can swap in richer
features later (see docs/README.md, "Upgrading features").
"""

from __future__ import annotations

import wave
import struct
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Load a mono/stereo PCM WAV file without external dependencies.

    Returns
    -------
    samples : np.ndarray, float32, normalized to [-1, 1], mono
    sample_rate : int
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sampwidth == 1:
        dtype = np.uint8
        max_val = 128.0
    elif sampwidth == 4:
        dtype = np.int32
        max_val = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if sampwidth == 1:
        samples = samples - 128.0  # uint8 is offset-encoded

    samples = samples / max_val

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, framerate


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _frame_signal(signal: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0):
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    frames = np.stack([
        signal[i * hop_len: i * hop_len + frame_len] for i in range(n_frames)
    ])
    window = np.hanning(frame_len)
    return frames * window


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    """Minimal mel filterbank implementation (replaces librosa.filters.mel)."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        f_center = max(f_center, f_left + 1)
        f_right = max(f_right, f_center + 1)
        for k in range(f_left, f_center):
            if k < fbank.shape[1]:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if k < fbank.shape[1]:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank


def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 13, n_mels: int = 26) -> np.ndarray:
    """Compute MFCC-like coefficients using only numpy/scipy (DCT-II via numpy)."""
    frames = _frame_signal(signal, sr)
    n_fft = frames.shape[1]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    power = (mag ** 2) / n_fft

    fbank = _mel_filterbank(sr, n_fft, n_mels)
    mel_energy = power @ fbank.T
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel = np.log(mel_energy)

    # DCT-II (orthonormal) to get cepstral coefficients
    N = log_mel.shape[1]
    dct_basis = np.cos(
        np.pi / N * (np.arange(N) + 0.5)[None, :] * np.arange(n_mfcc)[:, None]
    )
    mfcc = log_mel @ dct_basis.T  # (frames, n_mfcc)
    return mfcc


def spectral_flatness(signal: np.ndarray, sr: int) -> float:
    """
    Spectral flatness (Wiener entropy): ratio of geometric to arithmetic mean
    of the power spectrum. Synthetic/vocoder audio tends to show unnaturally
    smooth or unnaturally flat spectral structure compared to natural speech.
    """
    freqs = rfftfreq(len(signal), 1 / sr)
    mag = np.abs(rfft(signal)) + 1e-10
    geo_mean = np.exp(np.mean(np.log(mag)))
    arith_mean = np.mean(mag)
    return float(geo_mean / arith_mean)


def pitch_jitter_proxy(signal: np.ndarray, sr: int) -> float:
    """
    Approximate jitter (cycle-to-cycle pitch period variation) using
    autocorrelation-based pitch tracking per short frame. Natural voices
    have small but nonzero jitter; some TTS/vocoder outputs are
    unnaturally stable (very low jitter) or have irregular artifacts
    (very high jitter).
    """
    frame_len = int(sr * 0.04)
    hop = int(sr * 0.02)
    if len(signal) < frame_len * 2:
        return 0.0

    periods = []
    min_lag = int(sr / 400)  # 400 Hz upper bound
    max_lag = int(sr / 60)   # 60 Hz lower bound

    for start in range(0, len(signal) - frame_len, hop):
        frame = signal[start:start + frame_len]
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-4:
            continue
        autocorr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if max_lag >= len(autocorr):
            continue
        search = autocorr[min_lag:max_lag]
        if len(search) == 0:
            continue
        peak_lag = np.argmax(search) + min_lag
        if autocorr[0] > 0 and autocorr[peak_lag] / autocorr[0] > 0.3:
            periods.append(peak_lag / sr)

    if len(periods) < 3:
        return 0.0

    periods = np.array(periods)
    diffs = np.abs(np.diff(periods))
    jitter = float(np.mean(diffs) / (np.mean(periods) + 1e-10))
    return jitter


def spectral_flux(signal: np.ndarray, sr: int) -> float:
    """Frame-to-frame spectral change. Vocoded speech often has unnaturally
    smooth spectral trajectories (low flux) due to how neural vocoders
    generate frames."""
    _, _, Zxx = stft(signal, fs=sr, nperseg=min(512, len(signal)))
    mag = np.abs(Zxx)
    if mag.shape[1] < 2:
        return 0.0
    diffs = np.diff(mag, axis=1)
    flux = np.sqrt(np.sum(diffs ** 2, axis=0))
    return float(np.mean(flux))


def high_freq_energy_ratio(signal: np.ndarray, sr: int, cutoff_hz: float = 4000.0) -> float:
    """
    Ratio of energy above cutoff_hz to total energy. Many TTS/vocoder
    pipelines roll off or artificially shape high-frequency content
    differently from natural recordings/telephony channels.
    """
    freqs = rfftfreq(len(signal), 1 / sr)
    mag = np.abs(rfft(signal)) ** 2
    total = np.sum(mag) + 1e-10
    high = np.sum(mag[freqs >= cutoff_hz])
    return float(high / total)


def extract_feature_vector(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Combine MFCC statistics + hand-crafted acoustic artifacts into a single
    fixed-length feature vector suitable for a classical ML classifier
    (RandomForest/GradientBoosting/SVM).

    Returns a 1D np.ndarray of shape (n_features,).
    """
    if len(signal) == 0:
        signal = np.zeros(1024, dtype=np.float32)

    mfcc = extract_mfcc(signal, sr)
    mfcc_mean = mfcc.mean(axis=0)
    mfcc_std = mfcc.std(axis=0)

    flatness = spectral_flatness(signal, sr)
    jitter = pitch_jitter_proxy(signal, sr)
    flux = spectral_flux(signal, sr)
    hf_ratio = high_freq_energy_ratio(signal, sr)

    feature_vec = np.concatenate([
        mfcc_mean, mfcc_std,
        [flatness, jitter, flux, hf_ratio],
    ])
    return feature_vec.astype(np.float32)


FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + [f"mfcc_std_{i}" for i in range(13)]
    + ["spectral_flatness", "pitch_jitter", "spectral_flux", "high_freq_ratio"]
)
