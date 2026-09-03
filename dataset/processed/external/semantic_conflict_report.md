# Phase 4 — Semantic Conflict Audit

Method: keyword/pattern-assisted sampling + manual reading, since semantic
alignment can't be fully automated with regex. Every specific claim below is
backed by an actual record ID that can be re-checked.

## Finding 1 — Scenario-family mistagging (data quality, not our error)
`EXT-H-111` (an "automated hardship-program notification bot" pitching student
loan relief) is tagged `scenario_family: tech_support`. It has nothing to do
with tech support. Sampling the rest of the `tech_support`-tagged HIGH bucket
(`EXT-H-107`–`EXT-H-110`) shows more of the same: `EXT-H-107`–`109` are retail
flash-sale/product marketing copy ("VIP Access", "Flash Sale Alert", "Reserve
Yours Now"), not tech-support scams at all. `scenario_family` in this curated
file is noisy and should not be trusted as-is for scenario-balance decisions.

## Finding 2 — A meaningful slice of "HIGH" is telemarketing/spam, not social engineering
This is the most important finding for our specific project. Our task is
voice-cloning-enabled **social engineering** — impersonation, manufactured
urgency/authority, and requests for sensitive information or money. A
keyword-assisted scan of all 120 HIGH records found:
- **16/120 (13.3%)** match a telemarketing/spam-style pattern (e.g. "press 1",
  "flash sale", "pre-approved", "act fast", "limited time") with **no**
  co-occurring social-engineering pattern (no OTP/PIN/password/impersonation/
  account-compromise/threat language).
- **19/120 (15.8%)** match core social-engineering patterns directly.
- The remaining ~86 use varied phrasing that this narrow heuristic doesn't
  classify either way — this is expected (heuristics are narrow, not a full
  count) and needs human judgment, not just this scan.

Concretely: `EXT-H-107`–`109` (product flash-sale ads) and `EXT-H-112`,
`EXT-H-113`, `EXT-H-115` (loan/debt-consolidation robocalls ending in "press 1
to speak with a specialist") read as generic telemarketing fraud/spam, not
attacks matching our specific taxonomy (no impersonation claim, no request
for sensitive personal data, no urgency-driven credential/payment demand —
just a marketing hook + call-forward). Under our SIH risk taxonomy these
would plausibly land LOW-to-CAUTION, not HIGH. This is a genuine taxonomy
mismatch between the external source's binary "scam/non-scam" label and our
project's specific "social-engineering intent" definition — not a labeling
error on the curator's part, since these genuinely are "scam" by a broader
definition, just not the kind our project is built to detect.

## Finding 3 — "other" is an oversized, uninformative scenario bucket
45/100 LOW records (45%) and 16/120 HIGH records fall under `scenario_family:
other`. This isn't a conflict with SIH per se, but it limits how precisely we
can reason about scenario coverage/balance for Experiment C filtering below —
"other" tells us nothing about what's actually in it.

## Finding 4 — LOW class does not show meaningful semantic conflict
A scan of all 100 LOW records for sensitive-action vocabulary (OTP/PIN/card
number/wire transfer/remote access/etc.) found only 1 match (`EXT-L-026`),
which on inspection is a benign product-recommendation message with no actual
conflict ("the company is offering a discount on... headphones... should we
check..."). LOW looks clean.

## Finding 5 — CAUTION class matches our intended definition reasonably well
Sampling `CAUTION`-labeled banking records (`EXT-C-001`, `EXT-C-002`,
`EXT-C-016`, `EXT-C-017`) shows genuinely ambiguous "verify a transaction /
confirm recent activity" framing — legitimate-sounding bank-service language
that also resembles a common scam pretext. This matches the conceptual
definition we set for CAUTION in Milestone 2 (plausible either way, evidence
insufficient for HIGH) reasonably well, better than I expected going in given
CAUTION was curator-assigned rather than source-provided.

## Overall interpretation
The external dataset's "scam" label is broader than our "HIGH" (malicious
social-engineering intent) — it includes plain telemarketing/spam alongside
genuine social-engineering attempts. Bulk-importing all 120 HIGH records
would teach a "is this a sales pitch" signal alongside the "is this a
social-engineering attack" signal we actually want, on top of the
already-documented length confound. This directly informs the Experiment C
filtering criteria below.
