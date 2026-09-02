"""
train_model.py
---------------
Trains the deepfake-vs-human classifier and saves it to disk.

Currently trains on the synthetic placeholder dataset (see
synthetic_dataset.py). Swap `generate_dataset()` for a real data loader
(e.g. ASVspoof protocol files) when real audio is available -- everything
downstream (feature extraction, model, interface) stays the same.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from audio_features import extract_feature_vector, FEATURE_NAMES
from synthetic_dataset import generate_dataset


def build_training_matrix(signals, labels, sample_rate):
    X = np.stack([extract_feature_vector(sig, sample_rate) for sig in signals])
    y = np.array([1 if lbl == "synthetic" else 0 for lbl in labels])  # 1 = synthetic
    return X, y


DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "deepfake_model.pkl")


def train(save_path: str = DEFAULT_SAVE_PATH, n_per_class: int = 80, seed: int = 42):
    signals, labels, sr = generate_dataset(n_per_class=n_per_class, seed=seed)
    X, y = build_training_matrix(signals, labels, sr)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=seed, class_weight="balanced"
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    print("=== Evaluation on held-out synthetic-dataset split ===")
    print(classification_report(y_test, y_pred, target_names=["human", "synthetic"]))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    except ValueError:
        pass

    feature_importance = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop features:")
    for name, imp in feature_importance[:8]:
        print(f"  {name:20s} {imp:.3f}")

    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "sample_rate": sr,
        "trained_on": "synthetic_placeholder_dataset",
        "version": "0.1.0",
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle to {save_path}")
    return bundle


if __name__ == "__main__":
    train()
