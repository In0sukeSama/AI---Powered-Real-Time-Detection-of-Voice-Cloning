"""
quality_checks.py — Milestone 2 quality gates. Run BEFORE using the dataset
for any benchmarking. Prints a report; does not silently fix anything.
"""
import json
import re
from collections import Counter, defaultdict
from itertools import combinations

VALID_LABELS = {"LOW", "CAUTION", "HIGH"}
VALID_SPLITS = {"train", "validation", "test", "paraphrase_test",
                "hard_negative_test", "unseen_test", "context_test", "caution_test"}
REQUIRED_FIELDS = {"conversation_id", "scenario", "scenario_family", "context_type",
                    "difficulty", "turns", "label", "intent", "risk_signals", "split"}


def load(path):
    with open(path) as f:
        return json.load(f)


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return set(text.split())


def full_text(record):
    return " ".join(t["text"] for t in record["turns"])


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_checks(data):
    report = []

    # 1. Missing/malformed fields
    malformed = []
    for r in data:
        missing = REQUIRED_FIELDS - set(r.keys())
        if missing:
            malformed.append((r.get("conversation_id", "?"), missing))
    report.append(("Missing/malformed fields", malformed if malformed else "none"))

    # 2. Label / split validity
    bad_labels = [r["conversation_id"] for r in data if r["label"] not in VALID_LABELS]
    bad_splits = [r["conversation_id"] for r in data if r["split"] not in VALID_SPLITS]
    report.append(("Invalid labels", bad_labels if bad_labels else "none"))
    report.append(("Invalid splits", bad_splits if bad_splits else "none"))

    # 3. Exact duplicate detection (on full concatenated text)
    seen = {}
    exact_dupes = []
    for r in data:
        t = full_text(r).strip().lower()
        if t in seen:
            exact_dupes.append((r["conversation_id"], seen[t]))
        else:
            seen[t] = r["conversation_id"]
    report.append(("Exact duplicates", exact_dupes if exact_dupes else "none"))

    # 4. Near-duplicate detection (token Jaccard >= 0.85) across DIFFERENT splits
    near_dupes = []
    norm_cache = {r["conversation_id"]: normalize(full_text(r)) for r in data}
    for a, b in combinations(data, 2):
        if a["split"] == b["split"]:
            continue
        sim = jaccard(norm_cache[a["conversation_id"]], norm_cache[b["conversation_id"]])
        if sim >= 0.85:
            near_dupes.append((a["conversation_id"], b["conversation_id"], round(sim, 2),
                                a["split"], b["split"]))
    report.append(("Near-duplicates across splits (Jaccard>=0.85)", near_dupes if near_dupes else "none"))

    # 5. Scenario-family leakage: unseen_test families must not appear in train/validation
    train_val_families = {r["scenario_family"] for r in data if r["split"] in ("train", "validation")}
    unseen_families = {r["scenario_family"] for r in data if r["split"] == "unseen_test"}
    leaked_families = train_val_families & unseen_families
    report.append(("Scenario-family leakage (train/val vs unseen_test)",
                    sorted(leaked_families) if leaked_families else "none"))

    # 6. Class distribution
    label_counts = Counter(r["label"] for r in data)
    report.append(("Label distribution (overall)", dict(label_counts)))

    per_split_labels = defaultdict(Counter)
    for r in data:
        per_split_labels[r["split"]][r["label"]] += 1
    report.append(("Label distribution (per split)", {k: dict(v) for k, v in per_split_labels.items()}))

    # 7. Scenario/context_type distribution
    report.append(("context_type distribution", dict(Counter(r["context_type"] for r in data))))
    report.append(("scenario_family distribution", dict(Counter(r["scenario_family"] for r in data))))

    # 8. Split sizes
    report.append(("Split sizes", dict(Counter(r["split"] for r in data))))

    return report


def audit_external_dataset(path):
    """
    Phase 9 addition: quality checks for the curated_external_v1.json schema,
    which differs from the SIH schema (id/text/label/scenario_family/source_*
    instead of conversation_id/turns/split). Additive -- does not touch the
    SIH-schema functions above.
    """
    data = load(path)
    report = []

    required = {"id", "text", "label", "scenario_family", "source_type",
                "source_file", "source_record", "source_label",
                "curation_reason", "indirect_request_candidate", "raw_text"}
    missing = [(r.get("id", "?"), required - set(r.keys())) for r in data if required - set(r.keys())]
    report.append(("Missing fields", missing if missing else "none"))

    bad_labels = [r["id"] for r in data if r["label"] not in VALID_LABELS]
    report.append(("Invalid labels", bad_labels if bad_labels else "none"))

    empty_text = [r["id"] for r in data if not r["text"].strip()]
    report.append(("Empty text", empty_text if empty_text else "none"))

    ids = [r["id"] for r in data]
    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    report.append(("Duplicate IDs", dup_ids if dup_ids else "none"))

    texts = [r["text"].strip().lower() for r in data]
    dup_texts = {t: c for t, c in Counter(texts).items() if c > 1}
    report.append(("Duplicate texts", dup_texts if dup_texts else "none"))

    provenance = Counter(r["source_type"] for r in data)
    report.append(("source_type distribution", dict(provenance)))
    unexpected_source_types = [r["id"] for r in data
                                if r["source_type"] != "external_generated_unknown_source"]
    report.append(("Records with unexpected source_type", unexpected_source_types or "none"))

    report.append(("Label distribution", dict(Counter(r["label"] for r in data))))

    by_label = defaultdict(Counter)
    for r in data:
        by_label[r["label"]][r["scenario_family"]] += 1
    report.append(("Scenario distribution by class", {k: dict(v) for k, v in by_label.items()}))

    return report


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset_pilot.json"
    if "curated_external" in path:
        print(f"Detected external-schema dataset: {path}\n")
        for title, result in audit_external_dataset(path):
            print(f"--- {title} ---")
            print(result)
            print()
    else:
        data = load(path)
        print(f"Total records: {len(data)}\n")
        for title, result in run_checks(data):
            print(f"--- {title} ---")
            print(result)
            print()
