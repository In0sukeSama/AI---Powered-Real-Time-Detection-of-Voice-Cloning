"""
evaluate_baseline.py — Benchmark the EXISTING rule engine (unmodified) against
the pilot dataset. For multi-turn conversations, feeds turns sequentially
through ConversationState and takes its final cumulative risk_level as the
conversation-level prediction, mirroring how it will actually be used.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "..")
from risk_classifier import classify_utterance
from conversation_state import ConversationState


def predict(record):
    if len(record["turns"]) == 1:
        r = classify_utterance(record["turns"][0]["text"])
        return r.risk_level
    cs = ConversationState()
    result = None
    for t in record["turns"]:
        result = cs.process(t["text"])
    return result.risk_level


def collapse(label):
    """Collapse CAUTION into HIGH-side-of-decision-boundary for a binary
    comparison against this pilot's LOW/HIGH-only gold labels, and report
    CAUTION separately so it isn't silently hidden either way."""
    return label


def evaluate(data):
    per_split = defaultdict(lambda: {"correct": 0, "total": 0, "confusion": defaultdict(int),
                                      "failures": []})
    overall = {"correct": 0, "total": 0}

    for r in data:
        pred = predict(r)
        gold = r["label"]
        split = r["split"]
        per_split[split]["total"] += 1
        overall["total"] += 1
        per_split[split]["confusion"][(gold, pred)] += 1
        # Treat CAUTION prediction as a "not clearly correct, not clearly wrong"
        # miss against this pilot's binary gold labels -- counted as incorrect,
        # but shown separately in the confusion matrix so it's not conflated
        # with a confident wrong answer.
        is_correct = (pred == gold)
        if is_correct:
            per_split[split]["correct"] += 1
            overall["correct"] += 1
        else:
            per_split[split]["failures"].append(
                (r["conversation_id"], gold, pred, " | ".join(t["text"] for t in r["turns"]))
            )

    return per_split, overall


if __name__ == "__main__":
    data = json.load(open("dataset_pilot.json"))
    per_split, overall = evaluate(data)

    print(f"OVERALL ACCURACY: {overall['correct']}/{overall['total']} "
          f"({100*overall['correct']/overall['total']:.1f}%)\n")

    for split in ("train", "validation", "test", "paraphrase_test",
                  "hard_negative_test", "context_test", "unseen_test"):
        s = per_split.get(split)
        if not s or s["total"] == 0:
            continue
        acc = 100 * s["correct"] / s["total"]
        print(f"--- {split} : {s['correct']}/{s['total']} ({acc:.1f}%) ---")
        print(f"  Confusion (gold->pred): {dict(s['confusion'])}")
        if s["failures"]:
            print("  Failures:")
            for cid, gold, pred, text in s["failures"]:
                print(f"    [{cid}] gold={gold} pred={pred} :: {text}")
        print()
