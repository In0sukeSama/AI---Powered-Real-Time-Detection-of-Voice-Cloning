"""
grok_client.py — thin wrapper around the Groq API (OpenAI-compatible).

NOTE ON NAMING: this file is called "grok_client.py" for historical reasons
(the original plan was xAI's Grok). It actually talks to Groq (groq.com,
console.groq.com) — a DIFFERENT company running open-source models
(Llama, Mixtral, etc.) on custom fast inference hardware. Grok (xAI) and
Groq are unrelated companies with confusingly similar names. Class/file
names were kept as-is to avoid touching every other file that imports
this one (llm_upgrade.py, test_grok_integration.py, demo_cli.py) — only
the internals (auth env var, base URL, default model) changed.

This is the ONLY file in the NLP component that makes a network call. Kept
isolated so:
  - it's easy to audit/remove if the privacy stance changes,
  - it's easy to mock in tests (see test_grok_integration.py),
  - a failure here can never crash the pipeline (see llm_upgrade.py, which
    always catches GrokError and falls back to the rule engine).

Config (environment variables, no secrets committed to the repo):
  GROQ_API_KEY  - required to actually call Groq. If unset, GrokClient
                  raises GrokError immediately on first call (fail fast,
                  caller falls back to the rule engine). Get one free,
                  no credit card, at https://console.groq.com/keys.
  GROK_MODEL    - optional, defaults to "openai/gpt-oss-120b" (good quality,
                  reliable JSON-following, currently active on Groq's free
                  tier). For lower latency/cost, try "openai/gpt-oss-20b".
                  NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant
                  were deprecated by Groq on 2026-06-17 -- if you see a
                  "model_not_found" 404, check
                  https://console.groq.com/docs/models for the current
                  catalog, since Groq's model lineup changes frequently.
  GROK_BASE_URL - optional, defaults to "https://api.groq.com/openai/v1".
  GROK_TIMEOUT_S - optional, defaults to 6 seconds. Keep this LOW: a slow
                  cloud call must not stall a "real-time" pipeline. If it
                  times out, the caller falls back to the rule engine.

NOTE: api.groq.com was not reachable from the Claude sandbox this was
written in either (same network-allowlist restriction as huggingface.co
and api.x.ai). This module has only been exercised with a mocked HTTP
layer here — test it against the real API with your own key before the
demo (which is exactly what's happening in this session, on the user's
own machine).
"""

import json
import os
import re
from typing import List, Optional

import requests

import load_env  # loads .env automatically if it exists
from interface import RISK_CATEGORIES, RISK_LEVELS, UtteranceResult


class GrokError(Exception):
    """Raised for any Groq call failure: missing key, network, timeout,
    bad status code, or a response that doesn't parse into our contract.
    Callers should always catch this and fall back to the rule engine —
    never let this propagate into the pipeline."""


_SYSTEM_PROMPT = f"""You are a conversational risk classifier for a voice-call \
scam-detection system. You will be given ONE new utterance from an ongoing \
phone call, plus signal names already seen earlier in the SAME call (may be \
empty). Classify the risk that this call is a social-engineering / scam \
attempt, based on INTENT, not just keyword presence.

Rules:
- Past-tense narration or reported speech about a sensitive topic \
("I went to the bank", "the bank asked me to update my address") is LOW risk.
- A live directive aimed at the listener to reveal/send/confirm a sensitive \
item (OTP, PIN, password, card/account details, a payment) is HIGH risk, \
even if phrased indirectly with no imperative verb \
("I need the number from that SMS.").
- Meta-discussion ABOUT scams ("scammers often ask for OTPs", "I'm researching \
banking fraud") is LOW risk — it is not a live demand.
- Legitimate-sounding verification requests with no urgency/threat/secrecy \
and a plausible business reason are usually CAUTION, not HIGH.
- Ordinary benign conversation is LOW.
- If earlier signals in this call already indicate risk, weight that context: \
repeated pressure across turns should push risk upward even if this one \
utterance alone looks milder.

Valid intent values (pick the single best match, or "benign"):
{list(RISK_CATEGORIES)}

Respond with ONLY a JSON object, no prose, no markdown fences, matching \
EXACTLY this shape:
{{
  "risk_score": <float 0.0-1.0>,
  "risk_level": <one of {list(RISK_LEVELS)}>,
  "intent": <one of the valid intent values above, or "benign">,
  "signals": [<zero or more valid intent values>],
  "explanation": [<one or two short human-readable reasons>],
  "confidence": <float 0.0-1.0>
}}"""


def _extract_json(raw: str) -> dict:
    """Groq is instructed to return raw JSON, but some models (notably
    reasoning models like openai/gpt-oss-*) sometimes add preamble/trailing
    text even when told not to. Strip markdown fences defensively, then
    fall back to locating the first {...} JSON object in the text if a
    direct parse fails."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first balanced {...} block and try that instead.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("No JSON object found in response", cleaned, 0)


class GrokClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, timeout_s: Optional[float] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("GROK_MODEL", "openai/gpt-oss-120b")
        self.base_url = base_url or os.environ.get("GROK_BASE_URL", "https://api.groq.com/openai/v1")
        self.timeout_s = timeout_s or float(os.environ.get("GROK_TIMEOUT_S", "6"))

    def classify(self, text: str, prior_signals: Optional[List[str]] = None) -> UtteranceResult:
        if not self.api_key:
            raise GrokError("GROQ_API_KEY not set — cannot call Groq. Get a free key "
                             "(no credit card) at https://console.groq.com/keys")

        user_content = json.dumps({
            "utterance": text,
            "prior_signals_this_call": prior_signals or [],
        })

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.0,
                },
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise GrokError(f"Network error calling Grok: {e}") from e

        if resp.status_code != 200:
            raise GrokError(f"Grok returned HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise GrokError(f"Could not parse Grok response: {e}") from e

        return _to_utterance_result(text, parsed)


def _to_utterance_result(text: str, parsed: dict) -> UtteranceResult:
    """Validate + coerce Grok's JSON into our stable contract. Raises
    GrokError on anything malformed so the caller falls back cleanly."""
    try:
        risk_level = parsed["risk_level"]
        if risk_level not in RISK_LEVELS:
            raise ValueError(f"invalid risk_level {risk_level!r}")

        risk_score = float(parsed["risk_score"])
        risk_score = max(0.0, min(1.0, risk_score))

        intent = parsed.get("intent", "benign")
        signals = [s for s in parsed.get("signals", []) if s in RISK_CATEGORIES]
        explanation = list(parsed.get("explanation", []))[:5]
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (KeyError, ValueError, TypeError) as e:
        raise GrokError(f"Grok response failed validation: {e}") from e

    return UtteranceResult(
        utterance=text,
        risk_score=risk_score,
        risk_level=risk_level,
        intent=intent,
        signals=signals,
        explanation=explanation or ["Grok classification (no explanation returned)"],
        confidence=confidence,
    )
