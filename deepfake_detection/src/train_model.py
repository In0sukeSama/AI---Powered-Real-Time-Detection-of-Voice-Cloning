"""
train_model.py
--------------
Train the voice deepfake detector using the prepared baseline dataset.

Training data:
    800 REAL
    800 FAKE

Features:
    30 features extracted by audio_features.py

Model:
    StandardScaler + RandomForestClassifier

The resulting model bundle is compatible with
deepfake_detector.py.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .audio_features import (
    FEATURE_NAMES,
    extract_feature_vector,
    load_wav,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "baseline_v2",
)

REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepfake_model.pkl",
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset():

    features = []
    labels = []

    # ---------------------------------------------------------
    # REAL = 0
    # FAKE = 1
    # ---------------------------------------------------------

    for folder, label, name in [
        (REAL_DIR, 0, "REAL"),
        (FAKE_DIR, 1, "FAKE"),
    ]:

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ]

        files.sort()

        print(f"{name} files found: {len(files)}")

        for i, audio_path in enumerate(files, start=1):

            try:

                signal, sr = load_wav(audio_path)

                feature_vector = extract_feature_vector(
                    signal,
                    sr
                )

                features.append(feature_vector)
                labels.append(label)

            except Exception as exc:

                print(
                    f"Skipping {audio_path}: {exc}"
                )

            if i % 100 == 0:
                print(
                    f"  Processed {i}/{len(files)} {name} files"
                )

    X = np.stack(features)
    y = np.array(labels)

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():

    print("\n========================================")
    print("TRAINING VOICE DEEPFAKE DETECTOR")
    print("========================================")

    print("\nLoading dataset...")

    X, y = load_dataset()

    print("\nDataset loaded!")

    print("Feature matrix:", X.shape)
    print("Labels:", y.shape)

    print(
        "REAL samples:",
        int(np.sum(y == 0))
    )

    print(
        "FAKE samples:",
        int(np.sum(y == 1))
    )

    print(
        "Number of features:",
        len(FEATURE_NAMES)
    )

    # ---------------------------------------------------------
    # StandardScaler
    # ---------------------------------------------------------

    print("\nFitting StandardScaler...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Scaling complete.")

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_scaled,
        y
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Save model bundle
    # ---------------------------------------------------------

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "sample_rate": 16000,
        "trained_on": "ASVspoof2021_DF_baseline_v2",
        "training_samples": len(y),
        "version": "1.0.0",
    }

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            bundle,
            f
        )

    print("\n========================================")
    print("MODEL SAVED SUCCESSFULLY")
    print("========================================")

    print("Path:", MODEL_PATH)
    print("Features:", len(FEATURE_NAMES))
    print("Training samples:", len(y))
    print("Model version:", bundle["version"])

    return bundle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()