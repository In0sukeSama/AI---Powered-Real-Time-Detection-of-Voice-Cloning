"""
evaluate_v2.py — Full metrics benchmark of the UNMODIFIED rule engine against
dataset_v2.json. Computes per-class precision/recall/F1, macro-F1, confusion
matrix, false-positive/false-negative rate (HIGH vs not-HIGH), latency, and
per-split/context_type breakdowns. No changes made to risk_classifier.py or
conversation_state.py.
"""
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, "..")
from risk_classifier import classify_utterance
from conversation_state import ConversationState

LABELS = ["LOW", "CAUTION", "HIGH"]


def predict(record, timings):
    t0 = time.perf_counter()
    if len(record["turns"]) == 1:
        r = classify_utterance(record["turns"][0]["text"])
        level = r.risk_level
    else:
        cs = ConversationState()
        result = None
        for t in record["turns"]:
            result = cs.process(t["text"])
        level = result.risk_level
    timings.append(time.perf_counter() - t0)
    return level


def per_class_prf(confusion):
    """confusion: dict[(gold,pred)] -> count. Returns per-class P/R/F1."""
    results = {}
    for label in LABELS:
        tp = confusion.get((label, label), 0)
        fp = sum(c for (g, p), c in confusion.items() if p == label and g != label)
        fn = sum(c for (g, p), c in confusion.items() if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[label] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": sum(c for (g, p), c in confusion.items() if g == label)}
    return results


def main():
    data = json.load(open("dataset_v2.json"))
    confusion = defaultdict(int)
    per_split = defaultdict(lambda: defaultdict(int))
    per_context = defaultdict(lambda: defaultdict(int))
    timings = []
    failures = defaultdict(list)

    for r in data:
        pred = predict(r, timings)
        gold = r["label"]
        confusion[(gold, pred)] += 1
        per_split[r["split"]][(gold, pred)] += 1
        per_context[r["context_type"]][(gold, pred)] += 1
        if pred != gold:
            failures[r["split"]].append((r["conversation_id"], gold, pred,
                                          " | ".join(t["text"] for t in r["turns"])))

    total = len(data)
    correct = sum(c for (g, p), c in confusion.items() if g == p)

    print(f"=== OVERALL: {correct}/{total} = {100*correct/total:.1f}% accuracy ===\n")

    print("--- Confusion matrix (rows=gold, cols=pred) ---")
    header = "        " + "".join(f"{l:>10}" for l in LABELS)
    print(header)
    for gold in LABELS:
        row = f"{gold:>8}" + "".join(f"{confusion.get((gold,p),0):>10}" for p in LABELS)
        print(row)
    print()

    print("--- Per-class Precision / Recall / F1 ---")
    prf = per_class_prf(confusion)
    macro_f1 = sum(v["f1"] for v in prf.values()) / len(prf)
    for label, v in prf.items():
        print(f"  {label:8} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (support={v['support']})")
    print(f"  MACRO-F1: {macro_f1:.3f}\n")

    # FPR/FNR framed as: is HIGH vs not-HIGH (the actionable alert decision)
    tp = confusion.get(("HIGH", "HIGH"), 0)
    fn = sum(c for (g, p), c in confusion.items() if g == "HIGH" and p != "HIGH")
    fp = sum(c for (g, p), c in confusion.items() if g != "HIGH" and p == "HIGH")
    tn = sum(c for (g, p), c in confusion.items() if g != "HIGH" and p != "HIGH")
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    print(f"--- HIGH-alert decision framing ---")
    print(f"  False Positive Rate (non-HIGH gold flagged HIGH): {fpr:.3f} ({fp}/{fp+tn})")
    print(f"  False Negative Rate (HIGH gold missed): {fnr:.3f} ({fn}/{fn+tp})\n")

    print("--- Accuracy per split ---")
    for split, conf in per_split.items():
        s_total = sum(conf.values())
        s_correct = sum(c for (g, p), c in conf.items() if g == p)
        print(f"  {split:20} {s_correct}/{s_total} ({100*s_correct/s_total:.1f}%)")
    print()

    print("--- Accuracy per context_type ---")
    for ctx, conf in per_context.items():
        s_total = sum(conf.values())
        s_correct = sum(c for (g, p), c in conf.items() if g == p)
        print(f"  {ctx:20} {s_correct}/{s_total} ({100*s_correct/s_total:.1f}%)")
    print()

    print(f"--- Latency ---")
    avg_ms = 1000 * sum(timings) / len(timings)
    max_ms = 1000 * max(timings)
    print(f"  avg={avg_ms:.4f}ms  max={max_ms:.4f}ms  n={len(timings)}\n")

    print("--- Failures by split ---")
    for split, fails in failures.items():
        print(f"  [{split}] {len(fails)} failures:")
        for cid, gold, pred, text in fails:
            print(f"    [{cid}] gold={gold} pred={pred} :: {text}")
    print()


if __name__ == "__main__":
    main()
