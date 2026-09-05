"""
app.py — Unified SIH prototype entry point.

Prototype flow:
    text/transcript -> Member 1 NLP/Groq -> \
    Member 2 audio (real model when audio is supplied) -> Member 3 fusion

The NLP path can be tested without audio:
    python app.py --text "Tell me the OTP you just received." --no-audio --mode rule_only

For a full run once a WAV is available:
    python app.py --text "Tell me the OTP you just received." --audio call.wav --mode full_llm

Modes:
    rule_only : no network, deterministic NLP baseline
    hybrid    : rule engine first; Groq only for CAUTION
    full_llm  : Groq for every utterance; falls back to rules on failure
"""
import argparse
import json
import os
import sys

from nlp_to_risk_adapter import get_final_risk
from audio_adapter import get_audio_result


def analyze_call(text, audio_path=None, mode=None, mock_audio=False):
    """Run the integrated single-turn pipeline."""
    if audio_path:
        audio_result = get_audio_result(audio_path=audio_path, mock=mock_audio)
    else:
        # Neutral audio input for NLP-only prototype testing.
        audio_result = {"deepfake_score": 0.0, "classification": "REAL"}

    return get_final_risk_from_audio_safe(
        text=text,
        audio_result=audio_result,
        mode=mode,
    )


def get_final_risk_from_audio_safe(text, audio_result, mode=None):
    # Import lazily to keep this file simple and avoid duplicating fusion logic.
    from nlp_to_risk_adapter import get_final_risk_from_audio
    return get_final_risk_from_audio(text, audio_result, mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Transcript/utterance to analyze")
    parser.add_argument("--audio", default=None, help="Optional WAV audio path")
    parser.add_argument("--mode", choices=["rule_only", "hybrid", "full_llm"], default=None)
    parser.add_argument("--mock-audio", action="store_true",
                        help="Use Member 2 mock audio output instead of the trained model")
    args = parser.parse_args()

    result = analyze_call(
        text=args.text,
        audio_path=args.audio,
        mode=args.mode,
        mock_audio=args.mock_audio,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
