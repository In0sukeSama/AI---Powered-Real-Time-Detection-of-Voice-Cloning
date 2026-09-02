# Milestone 2 — Dataset Schema & Taxonomy

## Schema
See `dataset_pilot.json`. Each record:

| field | type | notes |
|---|---|---|
| conversation_id | str | unique |
| scenario | str | specific narrative, e.g. `otp_bank_impersonation` |
| scenario_family | str | broader grouping used for leakage control |
| context_type | str | standard / paraphrase / indirect / hard_negative / context_dependent / escalation / unseen_scenario |
| difficulty | str | easy / medium / hard (subjective, assigned at authoring time) |
| turns | list[{turn_id, speaker, text}] | preserves order for multi-turn |
| label | str | gold LOW/CAUTION/HIGH for the conversation as a whole |
| intent | str | dominant intent, or "benign" |
| risk_signals | list[str] | gold signal tags (subset of RISK_CATEGORIES in interface.py) |
| split | str | train / validation / test / paraphrase_test / hard_negative_test / unseen_test / context_test |

## Scenario families in this pilot
- `bank_otp` — bank/OTP impersonation (used in train + paraphrase_test)
- `lottery` — prize/lottery fraud (used in train)
- `delivery_payment` — fake courier/delivery payment fraud (unseen_test only)
- `employment_verification` — fake job-offer document/fee fraud (unseen_test only)
- `account_recovery` — fake "help you recover your account" fraud (unseen_test only)
- `tech_support` — fake tech-support remote-access fraud (unseen_test only)
- `benign_life` — ordinary conversation, no scam scenario_family (train/test)
- `benign_meta_discussion` — talking about scams without being one (hard_negative_test)

## Leakage control
1. `unseen_test` scenario_families ∩ (train ∪ validation) scenario_families = ∅ (checked programmatically).
2. No exact-duplicate text anywhere in the dataset.
3. No near-duplicate text (normalized token-overlap ≥ 0.85) across different splits.

## What's deliberately hard in this pilot
- Indirect malicious requests with no imperative verb ("I need the number from that SMS.") — the exact category the current baseline fails on. Kept in `context_test`/`test` as documented failures, not fixed.
- Hard negatives with dense scam vocabulary but meta/reporting register.
- Context-dependent pairs: identical final utterance, different preceding context, different gold label.
- Multi-turn escalation vs. multi-turn flat-benign-with-scam-vocabulary (the "researcher" conversation).
