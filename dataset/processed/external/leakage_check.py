"""
leakage_check.py — Phase 3: compare curated_external_v1.json against the
SIH dataset (all splits, frozen and non-frozen). Documents threshold used.
Does not modify anything.
"""
import json
import re

FROZEN_SPLITS = {"test", "paraphrase_test", "hard_negative_test",
                  "context_test", "unseen_test", "caution_test"}
NEAR_DUP_THRESHOLD = 0.60  # documented: chosen because SIH-internal near-dup
                           # checks use 0.85 on much shorter texts; external
                           # records are longer, so raw token-Jaccard tends
                           # lower for genuinely similar content. 0.60 is
                           # exploratory/conservative -- flags candidates for
                           # human review, not automatic exclusion.


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return set(text.split())


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    external = json.load(open("curated_external_v1.json"))
    sih = json.load(open("../../dataset_v2.json"))

    sih_norm = [(r["conversation_id"], r["split"],
                 normalize(" ".join(t["text"] for t in r["turns"])))
                for r in sih]
    ext_norm = [(r["id"], r["label"], normalize(r["text"])) for r in external]

    exact_matches = []
    near_matches = []
    frozen_overlaps = []

    sih_exact = {}
    for cid, split, norm in sih_norm:
        key = " ".join(sorted(norm))
        sih_exact.setdefault(key, []).append((cid, split))

    for eid, elabel, enorm in ext_norm:
        ekey = " ".join(sorted(enorm))
        if ekey in sih_exact:
            for cid, split in sih_exact[ekey]:
                exact_matches.append((eid, cid, split))

        best_sim = 0.0
        best_match = None
        for cid, split, snorm in sih_norm:
            sim = jaccard(enorm, snorm)
            if sim > best_sim:
                best_sim = sim
                best_match = (cid, split)
        if best_sim >= NEAR_DUP_THRESHOLD:
            near_matches.append((eid, elabel, best_match[0], best_match[1], round(best_sim, 3)))
            if best_match[1] in FROZEN_SPLITS:
                frozen_overlaps.append((eid, elabel, best_match[0], best_match[1], round(best_sim, 3)))

    print(f"External records checked: {len(external)}")
    print(f"SIH records compared against: {len(sih)}")
    print(f"Exact text matches: {len(exact_matches)} -> {exact_matches if exact_matches else 'none'}")
    print(f"Near matches (Jaccard>={NEAR_DUP_THRESHOLD}): {len(near_matches)}")
    for m in sorted(near_matches, key=lambda x: -x[4]):
        print(f"    ext={m[0]} ({m[1]}) ~ sih={m[2]} ({m[3]}) sim={m[4]}")
    print(f"\nFrozen-split overlaps: {len(frozen_overlaps)}")
    for m in frozen_overlaps:
        print(f"    ext={m[0]} ({m[1]}) ~ sih={m[2]} (FROZEN:{m[3]}) sim={m[4]}")

    excluded_ids = sorted(set(m[0] for m in near_matches))
    print(f"\nRecords recommended for exclusion from training (any near-match to ANY SIH split, "
          f"frozen or not, out of caution): {len(excluded_ids)}")
    print(excluded_ids if excluded_ids else "none")

    with open("leakage_report.json", "w") as f:
        json.dump({
            "threshold_used": NEAR_DUP_THRESHOLD,
            "threshold_rationale": "Exploratory/conservative threshold for longer external "
                                    "texts vs SIH's shorter records; flags candidates for human "
                                    "review rather than asserting definitive duplication.",
            "external_records_checked": len(external),
            "sih_records_compared": len(sih),
            "exact_matches": exact_matches,
            "near_matches": near_matches,
            "frozen_overlaps": frozen_overlaps,
            "recommended_exclusions": excluded_ids,
        }, f, indent=2)
    print("\nWrote leakage_report.json")


if __name__ == "__main__":
    main()
