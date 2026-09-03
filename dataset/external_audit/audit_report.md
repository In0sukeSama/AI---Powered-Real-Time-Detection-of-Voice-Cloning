# External Dataset Audit — "Scam and Non-Scam Call Conversation Dataset"

Audited: raw zip at `dataset/raw/external/scam_non_scam_calls.zip` (untouched).
Extracted (read-only inspection copy) at `dataset/raw/external/extracted/`.

## 1. Dataset files
- `English_NonScam.txt` (79,832 bytes)
- `English_Scam.txt` (161,538 bytes)
- No README, no metadata file, no CSV/JSON schema file, no license file. Two flat `.txt` files only.

## 2. Total records
- Scam: 400 records
- Non-scam: 400 records
- Records are blank-line-separated blocks of plain text. Scam records are numbered ("1.\t...", "2.\t..."); non-scam records are not numbered.

## 3. Class distribution
400 / 400 — perfectly balanced by construction (this is a curated 50/50 split, not naturally-occurring class balance — worth remembering if it's combined with SIH data, since SIH's own distribution is very different).

## 4. Fields/schema
None. Each record is a single unstructured paragraph of caller-side monologue. No turn structure, no speaker labels, no conversation_id, no scenario tag, no label field beyond which file it lives in.

## 5. Conversation characteristics
- **Single-turn only** — 0/400 in either file span multiple lines/turns. This dataset cannot contribute multi-turn or context-dependent examples at all.
- Length: scam avg 394.7 chars (range 155–1429); non-scam avg 195.1 chars (range 116–372). Scam records are consistently ~2x longer — itself a potential shortcut signal a model could latch onto instead of learning content (a real risk to flag before training).
- Both files use bracket placeholders extensively: `[Greetings]`, `[Company]`, `[Name]`, `[Date]`, `[Number]`, etc. — 373/400 scam and 400/400 non-scam records contain at least one placeholder. This is clearly template-based/synthetic generation, not verbatim transcripts of real calls.

## 6. Scenario families found (keyword-based, approximate)
| Scenario | Scam hits | Non-scam hits |
|---|---|---|
| bank_otp | 122 | 0 |
| government_police | 99 | 12 |
| investment | 99 | 10 |
| delivery_payment | 49 | 25 |
| lottery | 13 | 0 |
| romance_emergency | 12 | 5 |
| employment | 16 | 9 |
| refund | 8 | — |
| subscription | 5 | 8 |
| charity | 4 | 4 |
| tech_support | 3 | — |
| utility_medical (benign-only category) | 3 | 29 |

Bank/OTP, government/police, and investment dominate the scam side. Non-scam is dominated by ordinary utility/pharmacy/clinic/delivery calls.

## 7. Provenance
The zip contains **no provenance metadata whatsoever** — no field or file distinguishes real-world sourced material from LLM-generated/augmented material at the record level. Given the dense, uniform placeholder templating (`[Company]`, `[Greetings]`, `[Title][Name]`) across nearly all records, this reads as entirely template-generated content, not raw transcripts. I cannot verify or refute the Kaggle description's claim about "publicly available scam experiences" from the file contents alone — there's simply no way to tell which records (if any) trace to a real source. **This must be treated as synthetic/unknown-provenance data, not real-world conversation data, in any documentation going forward.**

## 8. Synthetic/generated content
Strong indicators of template generation: uniform bracket-placeholder scheme, repeated openings (12 records literally start "this is [Company] and we are from we are..." — an artifact-looking phrase, possibly a generation glitch), and clustered opening skeletons (e.g. 4 records start "This is from the police department we have...", 4 start "Congratulations on your first successful investment as you..."). This is template/LLM-augmented data, consistent with what the Kaggle description claims, but with zero record-level tagging to confirm which specific records are which.

## 9. Duplicates
- Scam: 3 distinct texts duplicated (6 total duplicate records) — small, but real (should be deduplicated before any training use).
- Non-scam: 0 exact duplicates.

## 10. Near duplicates
No near-duplicates found against the SIH dataset at the Jaccard≥0.85 threshold used internally for SIH's own near-dup checks (see `external_audit/overlap_report.json`). Within the external dataset itself, near-duplicate detection wasn't exhaustively run pairwise (400×400×2 comparisons) but the repeated-opening-skeleton check above surfaces the clearest templated clusters.

## 11. SIH dataset overlap
No exact or near-duplicate text overlap found against any SIH record (max Jaccard similarity found anywhere: 0.40, and that reflects shared generic scam vocabulary like "congratulations... won... lottery... prize... claim", not a paraphrase of an actual SIH sentence). See `overlap_report.json` for full method and numbers.

## 12. Frozen-test overlap
**No leakage found.** No exact or near-duplicate matches against `test`, `paraphrase_test`, `hard_negative_test`, `context_test`, `unseen_test`, or `caution_test`. The exploratory topical-overlap pass (lower threshold, for characterization only — see overlap_report.json) shows expected thematic clustering around bank/OTP and lottery scenarios, which both datasets happen to cover, but nothing that would let a model "memorize" a frozen test example via the external data.

## 13. Potential LOW examples
Most of the 400 non-scam records look like clean LOW candidates (routine utility/pharmacy/clinic/delivery/bank-service calls with no manipulative framing). Estimated ~370-390 of 400 usable as LOW after removing the ~28 that mention scam-adjacent vocabulary (see #14).

## 14. Potential CAUTION examples
This is the most important finding for our current gap. **28 non-scam records mention scam-adjacent vocabulary in a legitimate-service context** — e.g., "this is [Company] Bank... verify recent transactions... routine security check, no action required" or "verify a recent online transaction... did you authorize this payment?" These are exactly the "genuinely ambiguous, plausible-either-way" texture our current CAUTION class needs and currently lacks (F1=0.000, 0 training examples). They are NOT automatically CAUTION-worthy by virtue of vocabulary alone — each would need individual review, since some read as clearly legitimate once you account for full context (e.g. bank calling about its own recorded transaction, addressed to name, no request for the customer to provide anything back). A genuine CAUTION set would need hand-selection from this pool, not bulk import.

## 15. Potential HIGH examples
The bulk of the 400 scam records (~370+ after removing 6 duplicates and the "we are from we are" glitch cluster) look like reasonable HIGH candidates — explicit fraud framing (fake grants, fake lottery wins, fake police/government threats, fake investment guarantees) matching several of our existing and planned scenario families.

## 16. Unusable/ambiguous examples
- The 6 duplicate scam records (should be deduped).
- The ~12 records with the glitchy "this is [Company] and we are from we are" opening (looks like a template-generation artifact, not natural language — worth excluding or manually rewriting rather than importing verbatim).
- Any record whose only content is the bracket-placeholder template itself without enough surrounding context to make a clear scenario judgment.

## 17. Linguistic diversity
Genuinely useful additions over our hand-authored set:
- **175/400 scam records (44%) contain no obvious imperative demand verb** (send/give/share/confirm/etc.) — this is a large pool of naturally-occurring indirect-request-style malicious text, exactly the category our rule engine (and our own hand-authored dataset) is thinnest on.
- Real variety in authority-impersonation framing (police, government, ISP, "credit card fraud department", "local police department") beyond our current bank/lottery-heavy set.
- The 28 non-scam scam-vocabulary records are a genuine, independently-sourced pool of hard-negative/CAUTION material we did not author ourselves — valuable specifically because it's not from the same author (me) as our existing hard negatives, reducing single-author bias in the dataset.

Does NOT provide: any multi-turn/context-dependent examples (0 records span turns), so it cannot help with our weakest area (context_test, 40% baseline accuracy) or multi-turn escalation at all.

## 18. New scenario coverage
Adds meaningful volume in `government_police` (99 vs our 2), `investment` (99 vs our 3), and general `delivery_payment` (49 vs our 3) beyond what we hand-authored — real value if curated rather than bulk-imported, given the length/template artifacts noted above.

## 19. Major risks
1. **Length confound**: scam records are ~2x longer on average than non-scam — a model could learn "long text → scam" instead of learning content, especially if this dataset dominates a training mix.
2. **No provenance tagging**: cannot honestly label any individual record as "real-world" vs "generated" — must document the whole set as unknown/synthetic provenance if used.
3. **No CAUTION field**: every record is binary scam/non-scam; any CAUTION assignment requires our own manual judgment call, which reintroduces single-author labeling bias for exactly the class we were trying to diversify.
4. **No multi-turn data**: doesn't touch our weakest measured area at all.
5. **Template artifacts**: bracket placeholders and the "we are from we are" glitch cluster would need cleanup/rewriting before use, not verbatim import.

## 20. Recommended filtering
Before any future use:
- Drop the 6 exact-duplicate scam records.
- Drop or manually rewrite the ~12 glitch-opening records.
- Hand-review (not bulk-label) the 28 scam-vocabulary non-scam records as CAUTION candidates — likely yields somewhere well under 28 genuinely ambiguous examples once reviewed individually, since several read as clearly-legitimate on inspection.
- Replace bracket placeholders with concrete filled-in values (a placeholder-heavy training set risks a model that only recognizes the literal string "[Company]" rather than generalizing).
- Do not use record length as a feature or allow it to correlate trivially with the label in whatever training pipeline eventually gets built.

## 21. Recommended training strategy
Given the evidence: **Option E (auxiliary linguistic/scenario diversity), with a possible narrow slice of Option C (curated merge) for the indirect-request and CAUTION-candidate pools specifically** — not Option A (direct use) and not full Option C (wholesale merge).
Reasoning: the dataset is single-turn only (can't help our biggest weakness), has real template/length artifacts, has zero CAUTION labeling, and has zero provenance tagging. Its real value is narrow and specific: (a) the 175 indirect-phrased scam records, and (b) the 28 hard-negative/CAUTION candidates — both directly address gaps in our own dataset. Everything else is redundant with what we already have or introduces risk (length confound, template artifacts) without adding much.
Do not use for auxiliary pretraining (Option D) — 800 short template records isn't enough scale to meaningfully pretrain anything a compact transformer's own pretraining wouldn't already cover; that option would add engineering complexity without clear justification here.

## 22. Estimated useful records after filtering
Rough, conservative estimate pending actual manual review (not a final count):
- ~370-380 usable HIGH candidates (scam, minus duplicates/glitches) — but likely only a curated subset (perhaps 40-80) would actually be selected, prioritizing the indirect-request subset and scenario diversity, not all 370+.
- ~15-25 usable CAUTION candidates after manual review of the 28-record pool.
- LOW candidates: our own dataset isn't short on LOW examples, so bulk-importing more LOW records is lower priority than the above two.

---

# DECISION

**External dataset status: 🟡 USEFUL BUT REQUIRES SIGNIFICANT CURATION**

Explanation: There is real, specific value here — indirect-phrased malicious examples and independently-sourced hard-negative/CAUTION candidates, both of which target exact gaps in our current dataset. But it cannot be used as-is: no provenance tagging, no multi-turn data, template/length artifacts, zero CAUTION labels, and a length confound that risks teaching a model the wrong signal. Any use requires manual curation record-by-record, not bulk import or automated label mapping.

---

# MILESTONE STATUS

**Dataset audit:** PASS (thorough inspection completed, no fabricated statistics)
**Leakage protection:** PASS (no exact/near-duplicate contamination of any frozen split found)
**Provenance:** FAIL (no provenance metadata exists in the source at all — this is a property of the external dataset, not a gap in our audit process)
**Label compatibility:** INVESTIGATE (binary scam/non-scam does not map cleanly to LOW/CAUTION/HIGH; usable CAUTION candidates exist but require individual manual review, not automatic derivation)

**Overall: 🟡 INVESTIGATE**

**Next recommended step:** Do not merge yet. If approved, the next concrete task would be a manual curation pass: (1) dedupe and remove glitch records, (2) hand-review the 28 CAUTION candidates and assign SIH-schema labels individually, (3) hand-select a subset of the 175 indirect-phrased scam records for HIGH, (4) convert all selected records into the existing SIH schema (turns/scenario_family/split/etc.) with `provenance: "external_generated_unknown_source"` explicitly tagged, and (5) re-run the full quality-check + leakage pipeline before adding anything to train/validation. This is a curation task, not a bulk-import task, and should go through the same schema-design review step as the original pilot dataset before execution.

**STOP HERE. Waiting for review before any curation or merging work begins.**
