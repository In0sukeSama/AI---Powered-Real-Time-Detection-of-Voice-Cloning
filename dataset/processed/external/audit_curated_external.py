"""
audit_curated_external.py — Phase 2 audit of curated_external_v1.json.
Read-only: does not modify the curated file or any SIH data.
"""
import json
import re
import statistics
from collections import Counter, defaultdict

VALID_LABELS = {"LOW", "CAUTION", "HIGH"}
REQUIRED_FIELDS = {"id", "text", "label", "scenario_family", "source_type",
                    "source_file", "source_record", "source_label",
                    "curation_reason", "indirect_request_candidate", "raw_text"}


def load(path):
    with open(path) as f:
        return json.load(f)


def audit_schema(data):
    print("=== A. SCHEMA ===")
    missing = [(r.get("id", "?"), REQUIRED_FIELDS - set(r.keys())) for r in data
               if REQUIRED_FIELDS - set(r.keys())]
    print("Missing fields:", missing if missing else "none")

    bad_labels = [r["id"] for r in data if r["label"] not in VALID_LABELS]
    print("Invalid labels:", bad_labels if bad_labels else "none")

    empty_text = [r["id"] for r in data if not r["text"].strip()]
    print("Empty text:", empty_text if empty_text else "none")

    ids = [r["id"] for r in data]
    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    print("Duplicate IDs:", dup_ids if dup_ids else "none")

    texts = [r["text"].strip().lower() for r in data]
    dup_texts_count = Counter(texts)
    dup_texts = {t: c for t, c in dup_texts_count.items() if c > 1}
    print(f"Duplicate texts: {len(dup_texts)} distinct texts duplicated"
          f" ({sum(dup_texts.values())} total records)" if dup_texts else "Duplicate texts: none")
    print()


def audit_class_distribution(data):
    print("=== B. CLASS DISTRIBUTION ===")
    print(dict(Counter(r["label"] for r in data)))
    print()


def audit_scenario_distribution(data):
    print("=== C. SCENARIO DISTRIBUTION (by class) ===")
    by_label = defaultdict(Counter)
    for r in data:
        by_label[r["label"]][r["scenario_family"]] += 1
    for label in ("LOW", "CAUTION", "HIGH"):
        print(f"  {label}:")
        for fam, n in sorted(by_label[label].items(), key=lambda x: -x[1]):
            print(f"    {fam:25} {n}")
    all_families = set()
    for c in by_label.values():
        all_families |= set(c.keys())
    print(f"\n  Total distinct scenario_family values: {len(all_families)}")
    print(f"  Families: {sorted(all_families)}")
    print()


def audit_length(data):
    print("=== D. LENGTH ANALYSIS ===")
    by_label = defaultdict(list)
    for r in data:
        by_label[r["label"]].append(len(r["text"]))
    for label in ("LOW", "CAUTION", "HIGH"):
        lens = by_label[label]
        print(f"  {label:8} n={len(lens):4} mean={statistics.mean(lens):.1f} "
              f"median={statistics.median(lens):.1f} min={min(lens)} max={max(lens)} "
              f"stdev={statistics.stdev(lens):.1f}")
    # confound check: overall correlation-ish signal via mean gap
    low_mean = statistics.mean(by_label["LOW"])
    high_mean = statistics.mean(by_label["HIGH"])
    caution_mean = statistics.mean(by_label["CAUTION"])
    print(f"\n  HIGH mean is {high_mean/low_mean:.2f}x LOW mean, "
          f"{high_mean/caution_mean:.2f}x CAUTION mean.")
    print("  This confirms the length confound flagged in the curation report is NOT resolved:")
    print("  a classifier could reach non-trivial accuracy using length alone as a shortcut.")
    print()


def audit_templates(data):
    print("=== E. TEMPLATE/ARTIFACT ANALYSIS ===")
    placeholder_pat = re.compile(r"\[[A-Za-z]+\]")
    remaining_placeholders = [r["id"] for r in data if placeholder_pat.search(r["text"])]
    print(f"Records with bracket placeholders remaining in cleaned 'text': "
          f"{len(remaining_placeholders)} {remaining_placeholders if remaining_placeholders else ''}")

    def skeleton(text, n=6):
        t = re.sub(r"[^a-z\s]", "", text.lower())
        return " ".join(t.split()[:n])

    skels = Counter(skeleton(r["text"]) for r in data)
    repeated = {k: v for k, v in skels.items() if v >= 3}
    print(f"Opening 6-word skeletons repeated >=3 times: {len(repeated)}")
    for k, v in sorted(repeated.items(), key=lambda x: -x[1])[:8]:
        print(f"   ({v}x) {k!r}")

    # naturalization artifact check: leftover "the number"/"the amount"/"the company"
    # generic-noun substitutions the curator used to replace placeholders
    generic_subs = re.compile(r"\bthe (number|amount|company|name|date|time|money|link)\b", re.IGNORECASE)
    with_generic_subs = sum(1 for r in data if generic_subs.search(r["text"]))
    print(f"\nRecords containing naturalized generic-noun substitutions "
          f"(e.g. 'the number', 'the amount' standing in for a placeholder): {with_generic_subs}/{len(data)}")
    print("  -> These are a residual, more subtle artifact: a classifier could still learn")
    print("     'the number'/'the amount' as source-fingerprint tokens rather than content.")
    print()


def audit_indirect(data):
    print("=== F. INDIRECT-REQUEST ANALYSIS (HIGH only) ===")
    high = [r for r in data if r["label"] == "HIGH"]
    indirect = [r for r in high if r.get("indirect_request_candidate") is True]
    print(f"HIGH records: {len(high)}, indirect_request_candidate=true: {len(indirect)} "
          f"({100*len(indirect)/len(high):.1f}%)")
    by_family = Counter(r["scenario_family"] for r in indirect)
    print("Breakdown by scenario_family:")
    for fam, n in sorted(by_family.items(), key=lambda x: -x[1]):
        print(f"  {fam:25} {n}")
    print()


if __name__ == "__main__":
    data = load("curated_external_v1.json")
    audit_schema(data)
    audit_class_distribution(data)
    audit_scenario_distribution(data)
    audit_length(data)
    audit_templates(data)
    audit_indirect(data)
