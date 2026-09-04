"""
demo_cli.py — interactive demo for the panel. Type utterances, watch risk
escalate; switch modes live with a command instead of restarting.

Commands (type instead of an utterance):
  :mode rule_only | :mode hybrid | :mode full_llm   - flip mode live
  :mode                                             - show current mode
  :reset                                             - clear conversation buffer
  :quit

Setup for FULL_LLM / HYBRID modes:
  export XAI_API_KEY="your-key-here"
  # optional: export GROK_MODEL="grok-4-fast"
  python3 demo_cli.py

RULE_ONLY mode needs no setup at all (no key, no network).
"""
import sys

from conversation_state import ConversationState
from llm_upgrade import MODES, classify, get_mode, set_mode

BANNER = """\
=== SIH26104 NLP Risk Demo ===
Mode: {mode}   (":mode hybrid" / ":mode full_llm" / ":mode rule_only" to switch)
Type an utterance and press Enter. ":reset" clears the call. ":quit" to exit.
"""


def main():
    cs = ConversationState()
    print(BANNER.format(mode=get_mode().value))

    while True:
        try:
            line = input(f"[{get_mode().value}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line == ":quit":
            break
        if line == ":reset":
            cs.reset()
            print("(conversation buffer cleared)")
            continue
        if line == ":mode":
            print(f"current mode: {get_mode().value}")
            continue
        if line.startswith(":mode "):
            requested = line.split(" ", 1)[1].strip()
            try:
                set_mode(requested)
                print(f"-> mode switched to {requested}")
            except ValueError:
                print(f"unknown mode {requested!r}. valid: {[m.value for m in MODES]}")
            continue

        result = cs.process(line, llm_classify_fn=lambda t, sig: classify(t, sig))
        print(f"  risk_level={result.risk_level}  risk_score={result.risk_score:.3f}  "
              f"intent={result.dominant_intent}  turn={result.turn_count}")
        if result.accumulated_signals:
            print(f"  signals so far: {result.accumulated_signals}")


if __name__ == "__main__":
    sys.exit(main())
