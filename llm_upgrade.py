"""
llm_upgrade.py — plugs Grok into the existing NLP component through the
llm_classify_fn extension point that risk_classifier.py / conversation_state.py
already expose (see README "Upgrade path"). Does not modify those files.

Three modes, switchable at RUNTIME (no restart needed — set_mode() just
flips a module-level variable, so a demo control panel / keypress can call
it live):

  RULE_ONLY  - existing rule engine only. Free, offline, ~0.15ms/utterance.
  HYBRID     - rule engine runs first (always). If its risk_level is
               CAUTION (the class it's worst at — see evaluate_v2.py:
               0.000 F1), Grok is called to refine that single utterance.
               LOW/HIGH from the rule engine are trusted as-is. This is
               the pattern the README already recommends.
  FULL_LLM   - Grok classifies every utterance. Rule engine is skipped
               entirely UNLESS Grok fails (network/timeout/bad response),
               in which case the rule-engine result is used and the
               explanation notes the fallback — the pipeline must never
               go silent just because a demo WiFi hiccups.

Usage:
    from llm_upgrade import set_mode, classify, MODES

    set_mode(MODES.HYBRID)
    result = classify("Tell me the OTP you just received.", prior_signals=[])

    # or, per-call override without touching the global mode:
    result = classify(text, prior_signals, mode=MODES.FULL_LLM)

Wiring into conversation_state.ConversationState (unchanged file):
    from conversation_state import ConversationState
    from llm_upgrade import classify

    cs = ConversationState()
    result = cs.process(text, llm_classify_fn=lambda t, sig: classify(t, sig))
    # ^ if current mode is RULE_ONLY this callback still runs the rule
    #   engine internally, so behavior is identical to not passing it.
"""

import logging
from enum import Enum
from typing import List, Optional

from interface import UtteranceResult
from risk_classifier import classify_utterance
from grok_client import GrokClient, GrokError

logger = logging.getLogger("llm_upgrade")


class MODES(str, Enum):
    RULE_ONLY = "rule_only"
    HYBRID = "hybrid"
    FULL_LLM = "full_llm"


_current_mode: str = MODES.RULE_ONLY
_client: Optional[GrokClient] = None


def set_mode(mode: str) -> None:
    """Flip the global mode at runtime. Valid at any point, including
    mid-call — e.g. a demo host can switch RULE_ONLY -> HYBRID live if
    the panel wants to see the "smarter" path without a restart."""
    global _current_mode
    mode = MODES(mode)
    _current_mode = mode
    logger.info("llm_upgrade mode set to %s", mode)


def get_mode() -> str:
    return _current_mode


def _get_client() -> GrokClient:
    global _client
    if _client is None:
        _client = GrokClient()
    return _client


def _mark_fallback(result: UtteranceResult, reason: str) -> UtteranceResult:
    """Tag a rule-engine result as a fallback so it's visible in the JSON
    output (useful for the demo UI to show "Grok unavailable, used
    offline engine" rather than pretending nothing happened)."""
    result.explanation = list(result.explanation) + [f"[fallback: {reason}]"]
    return result


def classify(
    text: str,
    prior_signals: Optional[List[str]] = None,
    mode: Optional[str] = None,
) -> UtteranceResult:
    """
    Single entry point used by both single-utterance and conversation-level
    callers. Never raises — any Grok failure falls back to the rule engine,
    per the existing "never raises" error-behavior contract in interface.py.
    """
    effective_mode = MODES(mode) if mode is not None else _current_mode
    prior_signals = prior_signals or []

    if effective_mode == MODES.RULE_ONLY:
        return classify_utterance(text, prior_signals=prior_signals)

    if effective_mode == MODES.HYBRID:
        rule_result = classify_utterance(text, prior_signals=prior_signals)
        if rule_result.risk_level != "CAUTION":
            return rule_result
        try:
            return _get_client().classify(text, prior_signals=prior_signals)
        except GrokError as e:
            logger.warning("Grok call failed in HYBRID mode, keeping rule result: %s", e)
            return _mark_fallback(rule_result, str(e))

    if effective_mode == MODES.FULL_LLM:
        try:
            return _get_client().classify(text, prior_signals=prior_signals)
        except GrokError as e:
            logger.warning("Grok call failed in FULL_LLM mode, falling back to rule engine: %s", e)
            rule_result = classify_utterance(text, prior_signals=prior_signals)
            return _mark_fallback(rule_result, str(e))

    raise AssertionError(f"unhandled mode {effective_mode!r}")  # pragma: no cover
