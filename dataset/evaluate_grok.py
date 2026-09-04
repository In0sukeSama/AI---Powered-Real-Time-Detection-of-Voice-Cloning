"""
evaluate_grok.py — same benchmark as evaluate_v2.py, but using
llm_upgrade.classify() so you can see whether Grok actually improves on
the rule-engine baseline (60.8% accuracy, macro-F1 0.439, CAUTION F1=0.000
— see evaluate_v2.py) rather than assuming it does.

REQUIRES a real XAI_API_KEY and real network access to api.x.ai (this
was never run against the live API from the dev sandbox — see README).

This makes real Grok API calls — 74 calls for the SIH set alone (one per
single-turn record, one per turn for multi-turn records). Cheap, but not
free. Check your credit balance isn't zero first.

Usage:
    export XAI_API_KEY="..."
    python3 evaluate_grok.py hybrid       # or: full_llm
"""
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, "..")
from conversation_state import ConversationState
from llm_upgrade import MODES, classify, set_mode

LABELS = ["LOW", "CAUTION", "HIGH"]


def predict(record, timings, mode):
    t0 = time.perf_counter()
    if len(record["turns"]) == 1:
        r = classify(record["turns"][0]["text"], mode=mode)
        level = r.risk_level
    else:
        cs = ConversationState()
        result = None
        for t in record["turns"]:
            result = cs.process(t["text"], llm_classify_fn=lambda txt, sig: classify(txt, sig, mode=mode))
        level = result.risk_level
    timings.append(time.perf_counter() - t0)
    return level


def per_class_prf(confusion):
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    if mode not in (MODES.HYBRID.value, MODES.FULL_LLM.value):
        print(f"Usage: python3 evaluate_grok.py [hybrid|full_llm]  (got {mode!r})")
        sys.exit(1)
    set_mode(mode)
    print(f"=== Evaluating mode={mode} against dataset_v2.json (74 records) ===")
    print("This makes real Grok API calls. Ctrl+C to abort.\n")

    data = json.load(open("dataset_v2.json"))
    confusion = defaultdict(int)
    per_split = defaultdict(lambda: defaultdict(int))
    timings = []
    failures = defaultdict(list)
    fallback_count = 0

    for i, r in enumerate(data):
        pred = predict(r, timings, mode)
        gold = r["label"]
        confusion[(gold, pred)] += 1
        per_split[r["split"]][(gold, pred)] += 1
        if pred != gold:
            failures[r["split"]].append((r["conversation_id"], gold, pred,
                                          " | ".join(t["text"] for t in r["turns"])))
        print(f"  [{i+1}/{len(data)}] {r['conversation_id']} gold={gold} pred={pred}"
              f"{'  <-- MISS' if pred != gold else ''}")

    total = len(data)
    correct = sum(c for (g, p), c in confusion.items() if g == p)

    print(f"\n=== OVERALL: {correct}/{total} = {100*correct/total:.1f}% accuracy (mode={mode}) ===")
    print(f"    Rule-engine baseline for comparison: 45/74 = 60.8%, macro-F1 0.439\n")

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

    print("--- Accuracy per split ---")
    for split, conf in per_split.items():
        s_total = sum(conf.values())
        s_correct = sum(c for (g, p), c in conf.items() if g == p)
        print(f"  {split:20} {s_correct}/{s_total} ({100*s_correct/s_total:.1f}%)")
    print()

    avg_ms = 1000 * sum(timings) / len(timings)
    max_ms = 1000 * max(timings)
    print(f"--- Latency (includes network round-trip) ---")
    print(f"  avg={avg_ms:.1f}ms  max={max_ms:.1f}ms  n={len(timings)}\n")

    print("--- Failures by split ---")
    for split, fails in failures.items():
        print(f"  [{split}] {len(fails)} failures:")
        for cid, gold, pred, text in fails:
            print(f"    [{cid}] gold={gold} pred={pred} :: {text}")
    print()


if __name__ == "__main__":
    main()
