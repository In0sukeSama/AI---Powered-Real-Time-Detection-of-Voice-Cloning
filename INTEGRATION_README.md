# SIH26104 — Integrated Prototype

## Pipeline
Text/transcript -> Member 1 NLP/Groq -> Member 3 Risk Fusion
                                      ^
Audio WAV -> Member 2 Deepfake Detector

## NLP-only demo (no audio required)
```bash
python app.py --text "Tell me the OTP you just received." --mode rule_only
python app.py --text "Scammers often ask victims for OTPs." --mode rule_only
python app.py --text "Hey, I'm running late. Can you send me the address again?" --mode rule_only
```

## Full prototype when a WAV is available
```bash
python app.py --text "Tell me the OTP you just received." --audio call.wav --mode full_llm
```

Do not commit `.env` or API keys. Copy `.env.example` to `.env` and insert your own key locally.

The trained Member 2 model is expected at:
`deepfake_detection/models/deepfake_model.pkl`
