"""
conversation_state.py — Section 5: conversation-level reasoning.

Maintains a SHORT-LIVED window of recent utterance results (not the full
transcript, per Section 6 privacy requirements) and produces a cumulative
ConversationRiskResult that can escalate as the call develops.
"""

from collections import deque
from typing import Deque, List, Optional

from interface import ConversationRiskResult, UtteranceResult
from risk_classifier import classify_utterance


class ConversationState:
    def __init__(self, window_size: int = 8, decay: float = 0.85):
        """
        window_size: max number of recent utterance results retained.
        decay: multiplier applied to older signals so risk reflects
               recent behavior more than stale context.
        """
        self.window_size = window_size
        self.decay = decay
        self._history: Deque[UtteranceResult] = deque(maxlen=window_size)

    def process(self, text: str, llm_classify_fn=None) -> ConversationRiskResult:
        prior_signals = self._accumulated_signal_names()
        result = classify_utterance(text, prior_signals=prior_signals, llm_classify_fn=llm_classify_fn)
        self._history.append(result)
        return self._aggregate()

    def _accumulated_signal_names(self) -> List[str]:
        names = []
        for r in self._history:
            names.extend(r.signals)
        return names

    def _aggregate(self) -> ConversationRiskResult:
        if not self._history:
            return ConversationRiskResult(turn_count=0, risk_score=0.0, risk_level="LOW",
                                           dominant_intent="benign", accumulated_signals=[], history=[])

        weighted_scores = []
        n = len(self._history)
        for i, r in enumerate(self._history):
            age = n - 1 - i  # 0 = most recent
            weighted_scores.append(r.risk_score * (self.decay ** age))

        # cumulative risk = weighted average pulled up by the max recent spike
        avg = sum(weighted_scores) / n
        peak = max(r.risk_score for r in self._history)
        cumulative = min(1.0, 0.2 * avg + 0.8 * peak)

        if cumulative >= 0.6:
            level = "HIGH"
        elif cumulative >= 0.3:
            level = "CAUTION"
        else:
            level = "LOW"

        signal_counts = {}
        for r in self._history:
            for s in r.signals:
                signal_counts[s] = signal_counts.get(s, 0) + 1
        accumulated_signals = sorted(signal_counts, key=signal_counts.get, reverse=True)

        non_benign = [r.intent for r in self._history if r.intent != "benign"]
        dominant_intent = max(set(non_benign), key=non_benign.count) if non_benign else "benign"

        return ConversationRiskResult(
            turn_count=n,
            risk_score=cumulative,
            risk_level=level,
            dominant_intent=dominant_intent,
            accumulated_signals=accumulated_signals,
            history=list(self._history),
        )

    def reset(self):
        """Discard buffered content — required by Section 6 (short-lived buffers)."""
        self._history.clear()
