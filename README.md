# NLP / Conversational Intelligence Component
**Owner:** Member 1 (per Section 28 of SIH26104 handoff)
**Version:** 0.1.0 — baseline rule/pattern engine

## What this is
Classifies conversational risk from transcribed text (Sections 4, 5, 7, 26
of the handoff). Distinguishes **intent** from **keyword presence**: it
looks for a sensitive topic *combined with* a live directive aimed at the
listener (or urgency/secrecy/threat), and downweights past-tense narration
or reported speech ("I went to the bank" / "the bank asked me to..."). It
also tracks a short-lived rolling conversation state so risk can escalate
turn-by-turn instead of judging each sentence alone.

## Files
- `interface.py` — the stable contract (dataclasses + JSON shape). Other
  members should only import from here + the two functions below.
- `risk_classifier.py` — `classify_utterance(text, prior_signals=None, llm_classify_fn=None)`
  → single-utterance result.
- `conversation_state.py` — `ConversationState.process(text)` → cumulative
  conversation-level result; `.reset()` discards the buffer.
- `test_nlp_component.py` — unit tests, including the four baseline cases
  from Section 26 (benign, malicious, paraphrased malicious, hard negative)
  plus the Section 3 "AI voice ≠ scam" case and a 5-turn escalation case.

## Interface contract (what Risk Engine / Pipeline can rely on)
```json
{
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "intent": "otp_pin_password_request",
  "signals": ["otp_pin_password_request", "urgency_time_pressure"],
  "explanation": ["Directs the listener to take a sensitive action", "Creates urgency / time pressure"],
  "confidence": 0.91
}
```
`risk_level` is always one of `LOW | CAUTION | HIGH`. `risk_score` is a
float 0.0–1.0. This shape will not change when the engine underneath is
swapped (see below) — only the contents may get more accurate.

## Example usage
```python
from conversation_state import ConversationState

cs = ConversationState()
for utterance in ["I'm calling from your bank.", "Tell me the OTP you just received."]:
    result = cs.process(utterance)
    print(result.to_json())
```

## Error behavior
- Empty/whitespace input → `LOW`, intent `benign`, no exception.
- Garbled/unclassifiable text → falls through to lowest-confidence `LOW`,
  never raises.

## Why a rule engine and not an LLM call, for the baseline
Section 6 of the handoff requires local processing and no upload of raw
transcripts to the cloud on the core path. A pattern-based engine runs
fully offline, has zero external dependencies, and its behavior is
auditable — useful for proving the "intent not keywords" requirement in
Section 26 before wiring up real STT.

## Upgrade path — agentic/LLM classifier
Both `classify_utterance()` and `ConversationState.process()` accept an
optional `llm_classify_fn(text, prior_signals) -> UtteranceResult`
callback. Swap in a local/on-device LLM or an agentic pipeline later
without changing the contract that Members 3/4/5 depend on. Recommended
next step if you want to explore this: keep the rule engine as a fast
first-pass filter, and only invoke the (heavier) LLM to double-check
CAUTION-level or ambiguous utterances — that keeps latency and any
cloud-dependency to a minimum while improving accuracy on edge cases.

## Known limitations (v0.1.0)
- English-only patterns; no multilingual/code-switching support yet.
- Regex-based entity/demand detection will miss heavily reworded attacks
  that use no recognizable entity terms at all (e.g., fully novel social
  engineering phrasing) — this is the main argument for the LLM upgrade
  path above.
- Sarcasm, quoting a scammer ("he told me to send my OTP, obviously I
  didn't"), and other negation patterns aren't yet handled.
- Not tuned/validated on real call transcripts — thresholds (0.3/0.6) are
  reasonable defaults tuned against the Section 26 test cases, not a
  labeled dataset.

## Dependencies
Python standard library only (`re`, `dataclasses`, `collections`, `typing`).
No network calls, no model downloads.

## Tests
```
python3 -m unittest test_nlp_component.py -v
```
All 10 tests currently pass.
