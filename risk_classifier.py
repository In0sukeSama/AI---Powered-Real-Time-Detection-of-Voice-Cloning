"""
risk_classifier.py — Baseline intent/risk classifier for a single utterance.

Design goal (Section 26 of handoff): distinguish intent from keywords.
"I went to the bank yesterday" and "Tell me the OTP now" both mention
sensitive-adjacent nouns, but only one is a *demand for action* aimed at
extracting something risky. So classification here is based on:

  1. Which risk category's ENTITIES are present (bank, OTP, prize, etc.)
  2. Whether the sentence has a DEMAND STRUCTURE targeting the listener
     (imperative mood / "tell me", "send", "give", "click", "install",
     "verify", second-person directive + sensitive object)
  3. Whether it's PAST-TENSE / FIRST-PERSON NARRATION (the classic
     benign-mention pattern: "I went to...", "I called...", "he asked me
     to update my address")
  4. Modifier signals layered on top: urgency, secrecy, threat.

A pure keyword hit with no demand structure and/or in narrated/past
context is downweighted heavily -> this is what makes "I went to the
bank yesterday" score LOW while "tell me the OTP" scores HIGH even
though neither example needs exact keyword overlap to work correctly
on new phrasing (see paraphrase test in tests/).

This is a transparent, auditable baseline meant to prove the intent-vs-
keyword distinction cheaply and offline. It is explicitly designed to
be swappable for a learned/agentic classifier later (see interface.py).
"""

import re
from typing import Callable, List, Optional, Tuple

from interface import UtteranceResult, RISK_CATEGORIES


# --- Entity lexicons: what the sentence is ABOUT -----------------------

ENTITY_PATTERNS = {
    "otp_pin_password_request": [r"\botp\b", r"\bpin\b", r"\bpassword\b",
                                  r"\bverification code\b", r"\bsix.?digit\b",
                                  r"\bcode you (just )?received\b",
                                  r"\b(the )?digits? (from|in) (that|this|the) (sms|message|text)\b",
                                  r"\bnumbers? from (that|this|the) (sms|message|text)\b"],
    "payment_or_transfer_request": [r"\btransfer\b", r"\bpayment\b", r"\bpay\b",
                                     r"\bwire\b", r"\bupi\b", r"\bgift card\b"],
    "sensitive_information_request": [r"\bbank details\b", r"\baccount number\b",
                                       r"\bcard number\b", r"\bcvv\b", r"\baadhaar\b",
                                       r"\bssn\b", r"\bdate of birth\b"],
    "lottery_prize_fraud": [r"\blottery\b", r"\bprize\b", r"\bwon\b.*\bclaim\b", r"\bcongratulations\b.*\bwon\b"],
    "bank_impersonation": [r"\bcalling from your bank\b", r"\byour bank\b", r"\bbank account\b"],
    "government_police_impersonation": [r"\bpolice\b", r"\bcourt\b", r"\bincome tax\b", r"\bcustoms\b", r"\barrest\b"],
    "emergency_relative_scam": [r"\bi('?m| am) in (an )?accident\b", r"\bneed money now\b", r"\bhospital\b.*\bmoney\b"],
    "account_compromise_claim": [r"\baccount (has been |is )?compromised\b", r"\bsuspicious activity\b",
                                  r"\bproblem with your account\b", r"\baccount will be blocked\b"],
    "threat_intimidation": [r"\bwill be arrested\b", r"\blegal action\b", r"\byou will be blocked\b", r"\bpenalty\b"],
    "urgency_time_pressure": [r"\bimmediately\b", r"\bright now\b", r"\bquickly\b", r"\burgent\b",
                               r"\bdo it now\b", r"\bwithin (an? )?(hour|minute)\b", r"\bor your\b"],
    "secrecy_isolation_request": [r"\bdon'?t tell\b", r"\bkeep this (a )?secret\b", r"\bdon'?t (hang up|disconnect)\b"],
    "remote_access_request": [r"\bteamviewer\b", r"\banydesk\b", r"\bremote access\b", r"\bscreen ?share\b"],
    "software_installation_request": [r"\binstall (this |the )?app\b", r"\bdownload (this |the )?app\b"],
    "suspicious_link_action_request": [r"\bclick (this |the )?link\b", r"\bopen (this )?link\b"],
    "personal_data_request": [r"\bfull name and address\b", r"\bmother'?s maiden name\b"],
}

# --- Demand structure: is the SPEAKER directing the LISTENER to act now? -

DEMAND_VERBS = [
    r"\btell me\b", r"\bsend\b", r"\bgive me\b", r"\bshare\b", r"\bread (me |out )?the\b",
    r"\bclick\b", r"\binstall\b", r"\bdownload\b", r"\bverify\b", r"\bconfirm\b",
    r"\bprovide\b", r"\benter\b", r"\btransfer\b", r"\bpay\b", r"\bcall (this|the) number\b",
]

# --- Negation: demand verb preceded by a negator in the same clause -----

NEGATION_PATTERNS = [
    r"\bnever\b", r"\bdon'?t\b", r"\bdo not\b", r"\bwon'?t\b",
    r"\bshouldn'?t\b", r"\bshould not\b", r"\bno one (should|will)\b",
]

# --- Narration / benign-mention markers: speaker describing own past action -

NARRATION_PATTERNS = [
    r"\bi (went|called|visited|updated|asked|checked)\b",
    r"\byesterday\b", r"\blast (week|month|year)\b",
    r"\bthe bank (asked|told|said)\b",  # reporting what bank said, not bank speaking now
    r"\bmy bank (asked|told|said)\b",   # same, first-person possessive phrasing
    r"\bmy (bank|account)\b.*\b(is fine|was fine|works? fine)\b",
]


def _match_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _detect_entities(text: str) -> List[str]:
    return [cat for cat, pats in ENTITY_PATTERNS.items() if _match_any(pats, text)]


def _has_demand_structure(text: str) -> bool:
    """
    True if a demand verb is present AND not immediately negated
    (e.g. "never share", "don't send") within a short window before it.
    """
    for pat in DEMAND_VERBS:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        preceding = text[max(0, m.start() - 25):m.start()]
        if _match_any(NEGATION_PATTERNS, preceding):
            continue  # negated demand verb, e.g. "never share" -> not a live demand
        return True
    return False


def _is_narration(text: str) -> bool:
    return _match_any(NARRATION_PATTERNS, text)


def classify_utterance(
    text: str,
    prior_signals: Optional[List[str]] = None,
    llm_classify_fn: Optional[Callable[[str, List[str]], "UtteranceResult"]] = None,
) -> UtteranceResult:
    """
    Classify a single utterance for conversational risk.

    Parameters
    ----------
    text : the transcribed utterance
    prior_signals : accumulated signals from earlier turns (for context-aware scoring)
    llm_classify_fn : optional pluggable agentic/LLM classifier. If provided,
        it is called with (text, prior_signals) and must return an
        UtteranceResult. This lets Member 1 later swap in a model without
        breaking the contract for downstream members.
    """
    if llm_classify_fn is not None:
        return llm_classify_fn(text, prior_signals or [])

    text = (text or "").strip()
    if not text:
        return UtteranceResult(utterance=text, risk_score=0.0, risk_level="LOW",
                                intent="benign", signals=[], explanation=[], confidence=1.0)

    entities = _detect_entities(text)
    demand = _has_demand_structure(text)
    narration = _is_narration(text)

    modifier_signals = [c for c in entities if c in
                         ("urgency_time_pressure", "secrecy_isolation_request", "threat_intimidation")]
    core_signals = [c for c in entities if c not in modifier_signals]

    score = 0.0
    explanation = []

    if core_signals:
        if demand and not narration:
            score += 0.7
            explanation.append("Directs the listener to take a sensitive action")
            # additional distinct sensitive categories in one demand -> compounding risk
            score += 0.15 * (len(core_signals) - 1)
        elif narration:
            score += 0.05
            explanation.append("Sensitive term appears in past-tense/narrated context, not a live demand")
        else:
            # entity present, no clear demand, no clear narration -> ambiguous, mild weight
            score += 0.25
            explanation.append("Sensitive topic mentioned without a clear directive")

    for m in modifier_signals:
        score += 0.2
        if m == "urgency_time_pressure":
            explanation.append("Creates urgency / time pressure")
        elif m == "secrecy_isolation_request":
            explanation.append("Requests secrecy or isolation from other people")
        elif m == "threat_intimidation":
            explanation.append("Uses threats or intimidation")

    # combine prior conversational signals: repeated same-category pressure escalates risk
    if prior_signals:
        overlap = set(prior_signals) & set(core_signals + modifier_signals)
        if overlap:
            score += 0.1
            explanation.append("Continues a risk pattern already seen earlier in the call")

    score = max(0.0, min(1.0, score))

    if score >= 0.6:
        level = "HIGH"
    elif score >= 0.3:
        level = "CAUTION"
    else:
        level = "LOW"

    intent = "benign"
    if core_signals:
        intent = core_signals[0]
    elif modifier_signals and score >= 0.3:
        intent = modifier_signals[0]

    confidence = 0.6 + 0.4 * (1 if (demand or narration) else 0)

    return UtteranceResult(
        utterance=text,
        risk_score=score,
        risk_level=level,
        intent=intent,
        signals=core_signals + modifier_signals,
        explanation=explanation or ["No risk indicators detected"],
        confidence=confidence,
    )
