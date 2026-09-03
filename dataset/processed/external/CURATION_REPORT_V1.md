# External Dataset Curation V1

## Result
- Raw records: **800** (400 scam + 400 non-scam)
- Exact duplicates removed: **6 scam**, **0 non-scam**
- Curated records: **260**
  - HIGH: **120**
  - CAUTION: **40**
  - LOW: **100**

## Curation decisions
- HIGH: stratified across scenario families; intentionally includes many indirect/non-imperative examples.
- CAUTION: 40 curator-reassigned non-scam examples where legitimate context coexists with a potentially sensitive request.
- LOW: 100 benign/hard-negative examples across multiple domains.
- Bracket placeholders were naturalized in the derived `text` field to reduce obvious template-token artifacts.
- `raw_text` and source record numbers are retained for traceability.

## Quality checks
- Exact duplicates remaining: **0**
- Same-label near-duplicate pairs (similarity >= 0.92): **0**
- Empty texts: **0**
- Bracket placeholders remaining in cleaned text: **0**

## Class lengths
| Class | N | Mean chars | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| LOW | 100 | 199.7 | 191.5 | 143 | 348 |
| CAUTION | 40 | 209.7 | 197.0 | 157 | 377 |
| HIGH | 120 | 391.6 | 320.0 | 177 | 1305 |

## Important limitations
- The archive has no reliable per-record provenance metadata, so provenance is recorded as `external_generated_unknown_source`.
- It is single-turn and therefore does not solve the SIH multi-turn/context-data problem.
- The source is strongly templated; naturalization reduces one artifact but does not eliminate all stylistic bias.
- CAUTION is a curator-created label, not a source-provided label.
- This should be treated as an **auxiliary training/diversity source**, not the sole benchmark.
- The original SIH frozen evaluation material was not modified.

## Recommended next use
Use this as an auxiliary training source, keep provenance during experiments, and evaluate final models only on untouched SIH validation/test material.
