from __future__ import annotations

import os

from deepfake_detector import DeepfakeDetector


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

REAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "evaluation",
    "real"
)

FAKE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "evaluation",
    "fake"
)


def evaluate():

    detector = DeepfakeDetector()

    results = []

    # ---------------------------------------------------------
    # REAL AUDIO
    # ---------------------------------------------------------

    print("\n========================================")
    print("EVALUATING REAL VOICES")
    print("========================================")

    for filename in sorted(os.listdir(REAL_DIR)):

        if not filename.lower().endswith(".wav"):
            continue

        path = os.path.join(
            REAL_DIR,
            filename
        )

        result = detector.predict(
            audio_path=path
        )

        results.append(
            (0, result)
        )

    # ---------------------------------------------------------
    # FAKE AUDIO
    # ---------------------------------------------------------

    print("\n========================================")
    print("EVALUATING FAKE VOICES")
    print("========================================")

    for filename in sorted(os.listdir(FAKE_DIR)):

        if not filename.lower().endswith(".wav"):
            continue

        path = os.path.join(
            FAKE_DIR,
            filename
        )

        result = detector.predict(
            audio_path=path
        )

        results.append(
            (1, result)
        )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    total = len(results)

    correct = 0

    true_real = 0
    false_fake = 0
    false_real = 0
    true_fake = 0

    for true_label, result in results:

        predicted_label = (
            1
            if result["classification"] == "SYNTHETIC"
            else 0
        )

        if predicted_label == true_label:
            correct += 1

        if true_label == 0 and predicted_label == 0:
            true_real += 1

        elif true_label == 0 and predicted_label == 1:
            false_fake += 1

        elif true_label == 1 and predicted_label == 0:
            false_real += 1

        elif true_label == 1 and predicted_label == 1:
            true_fake += 1

    accuracy = correct / total

    real_precision = (
        true_real / (true_real + false_real)
        if (true_real + false_real) > 0
        else 0
    )

    real_recall = (
        true_real / (true_real + false_fake)
        if (true_real + false_fake) > 0
        else 0
    )

    fake_precision = (
        true_fake / (true_fake + false_fake)
        if (true_fake + false_fake) > 0
        else 0
    )

    fake_recall = (
        true_fake / (true_fake + false_real)
        if (true_fake + false_real) > 0
        else 0
    )

    real_f1 = (
        2 * real_precision * real_recall
        / (real_precision + real_recall)
        if (real_precision + real_recall) > 0
        else 0
    )

    fake_f1 = (
        2 * fake_precision * fake_recall
        / (fake_precision + fake_recall)
        if (fake_precision + fake_recall) > 0
        else 0
    )

    macro_f1 = (real_f1 + fake_f1) / 2

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print("\n========================================")
    print("HELD-OUT EVALUATION RESULTS")
    print("========================================")

    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2%}")

    print("\nREAL")
    print(f"Precision     : {real_precision:.3f}")
    print(f"Recall        : {real_recall:.3f}")
    print(f"F1 Score      : {real_f1:.3f}")

    print("\nSYNTHETIC")
    print(f"Precision     : {fake_precision:.3f}")
    print(f"Recall        : {fake_recall:.3f}")
    print(f"F1 Score      : {fake_f1:.3f}")

    print("\nMacro F1      :", f"{macro_f1:.3f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(f"REAL predicted REAL      : {true_real}")
    print(f"REAL predicted SYNTHETIC : {false_fake}")
    print(f"FAKE predicted REAL      : {false_real}")
    print(f"FAKE predicted SYNTHETIC : {true_fake}")


if __name__ == "__main__":
    evaluate()