# Audio / Deepfake Detection Component (Member 2)

**Project:** SIH26104 — AI-Powered Real-Time Detection and Prevention of Voice-Cloning Impersonation Attacks
**Owner:** Member 2 — Audio / Deepfake Detection
**Status:** Working prototype, placeholder training data (see Limitations)

---

## 1. What this component does

Given an audio clip (a full call recording or a short buffer from a live
call), it estimates whether the **voice sounds AI-generated / synthetic**.

Per the project's north-star design (handoff Section 3), this component
does **not** decide whether a call is a scam. It only answers: *"is the
voice potentially synthetic?"* That signal is combined with the NLP
Conversational Risk component (Member 1) downstream in the Risk Fusion
Engine (Member 3).

## 2. Interface contract

This is the stable contract the rest of the team depends on. Do not change
field names/types without team agreement (handoff Section 28.2).

**Input:** a mono audio waveform, either as a WAV file path or an in-memory
`numpy` array + sample rate.

**Output:**
```json
{
  "deepfake_score": 0.93,
  "classification": "SYNTHETIC"
}
```
- `deepfake_score`: float in `[0, 1]`. Higher = more likely AI-generated/synthetic.
- `classification`: `"SYNTHETIC"` if `deepfake_score >= threshold` (default `0.5`), else `"HUMAN"`.

## 3. Usage

```python
from deepfake_detector import DeepfakeDetector

detector = DeepfakeDetector()  # loads trained model from ../deepfake_model.pkl
result = detector.predict(audio_path="call_clip.wav")
# {"deepfake_score": 0.87, "classification": "SYNTHETIC"}

# Or with an in-memory buffer (e.g. a rolling window from a live call):
result = detector.predict(signal=audio_array, sample_rate=16000)
```

### Mock mode (for teammates integrating before this model is ready)

```python
detector = DeepfakeDetector(mock=True)
result = detector.predict(audio_path="incoming_call.wav")
# Deterministic per input, structurally valid, no model file needed.
```

Per handoff Section 28.3, Members 3 and 4 should use `mock=True` to build
and test the Risk Fusion Engine and Real-Time Pipeline against this
contract without waiting for the real model.

## 4. Architecture

```
Audio (file or buffer)
        |
        v
Feature extraction (audio_features.py)
  - MFCC (13 coeffs, mean + std across frames)
  - Spectral flatness
  - Pitch jitter proxy (autocorrelation-based)
  - Spectral flux
  - High-frequency energy ratio
        |
        v
StandardScaler -> RandomForestClassifier (scikit-learn)
        |
        v
{"deepfake_score": float, "classification": str}
```

### Why these features?

Rather than a single black-box embedding model (which would need a large
labeled corpus and GPU training time we don't have in a hackathon), the
prototype uses hand-crafted acoustic features known in the anti-spoofing
literature to differ between natural speech and vocoder/TTS output:

- **MFCCs** — standard spectral-envelope representation used across nearly
  all speech/anti-spoofing systems.
- **Spectral flatness** — vocoder output can be spectrally "smoother" or
  more tonal than natural voiced speech.
- **Pitch jitter** — natural voices have small, irregular cycle-to-cycle
  pitch variation; some synthetic speech is unnaturally stable or has
  characteristic artifacts.
- **Spectral flux** — frame-to-frame spectral change; neural vocoders can
  produce unnaturally smooth trajectories.
- **High-frequency energy ratio** — TTS/vocoder pipelines often shape
  high-frequency content differently than microphone/telephony recordings.

These are the "top features" the model actually relied on when trained
(see training console output) — `mfcc_mean`, `high_freq_ratio`, and
`spectral_flatness` dominated.

## 5. Dependencies

- `numpy`, `scipy`, `scikit-learn` only. No `librosa`/`torchaudio`/GPU
  required — this was a deliberate choice for portability in constrained
  environments (e.g. this prototype was built with no internet access).
- WAV loading uses Python's built-in `wave` module (PCM 8/16/32-bit, mono
  or stereo).

**Upgrading features:** if `librosa` becomes available, you can swap
`extract_feature_vector` internals for richer features (e.g. real CQCC/LFCC
features used in ASVspoof baselines) without changing the function
signature or the interface contract.

## 6. Known limitations — READ BEFORE DEMO/INTEGRATION

1. **Training data is synthetic/placeholder, not real deepfake audio.**
   `synthetic_dataset.py` generates two caricatured waveform classes
   (natural-ish vs. unnaturally-clean) purely to exercise the full
   pipeline offline. The 100% test accuracy reflects that these synthetic
   classes are easy to separate — it is **not** a claim about real-world
   deepfake-detection accuracy.
   **Action before the real demo:** retrain on a real corpus (ASVspoof
   2019/2021 LA, WaveFake, or In-the-Wild are standard choices) using the
   same `extract_feature_vector` pipeline and `train_model.py` structure.
2. **No voiced/unvoiced segmentation.** Features are computed over the
   whole clip; very short or heavily noisy clips may produce unreliable
   jitter estimates (mitigated by returning `0.0` jitter when too few
   pitch periods are detected, but this is a simplification).
3. **No channel/codec robustness testing yet.** Real phone calls go
   through codecs (e.g. AMR, Opus) that reshape the spectrum; this hasn't
   been validated against codec-compressed audio.
4. **Single-speaker-segment assumption.** The component assumes the clip
   is one speaker's voice; it doesn't yet handle diarization for
   multi-speaker recordings.
5. **Latency not yet benchmarked** on realistic streaming buffer sizes;
   current implementation processes a whole clip at once rather than
   incrementally.

## 7. Files

| File | Purpose |
|---|---|
| `src/audio_features.py` | WAV loading + feature extraction (no external audio deps) |
| `src/synthetic_dataset.py` | Placeholder training data generator (swap for real corpus) |
| `src/train_model.py` | Trains and saves the classifier bundle |
| `src/deepfake_detector.py` | Main `DeepfakeDetector` class — the component's public interface |
| `tests/test_detector.py` | Unit tests: contract shape, mock mode, edge cases, real-model sanity checks |
| `examples/example_usage.py` | End-to-end runnable demo |
| `deepfake_model.pkl` | Trained model bundle (model + scaler + metadata) |

## 8. Retraining on real data

1. Replace `generate_dataset()` in `train_model.py` with a loader for your
   real corpus (waveforms + `"human"`/`"synthetic"` labels).
2. Everything else (`build_training_matrix`, `extract_feature_vector`,
   the classifier, the saved bundle format) stays the same.
3. Re-run `python3 train_model.py` — it overwrites `deepfake_model.pkl` in
   place. `DeepfakeDetector` picks up the new model automatically; no
   interface change needed for Members 3/4.

## 9. Version / assumptions

- Model version: see `bundle["version"]` in `deepfake_model.pkl` (currently `0.1.0`, placeholder data).
- Assumes mono audio, any sample rate (feature extraction is sample-rate aware).
- Classification threshold: `0.5` (tunable via `DeepfakeDetector(threshold=...)`), should be re-tuned once trained on real data (e.g. optimize for target false-accept rate, since in this application false negatives on genuine scams are costlier than false positives on friends using voice filters).
