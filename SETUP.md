# Quick Setup Guide

## ⚠️ Groq vs Grok — read this first

This project uses **Groq** (console.groq.com) — a fast-inference company
running open-source models (Llama, etc.) — NOT **Grok** (xAI,
console.x.ai). They are two different, unrelated companies with
similar-sounding names. Easy to mix up, and we did initially. The file is
still called `grok_client.py` for historical reasons, but it talks to
Groq's API.

## 1. Get your Groq API key (free, no credit card)

1. Go to https://console.groq.com
2. Sign up with email, Google, or GitHub (free, no credit card needed)
3. Go to https://console.groq.com/keys
4. Click "Create API Key"
5. Copy the key immediately — Groq only shows it once. If you lose it,
   you'll need to generate a new one.

## 2. Create your local `.env` file

**DO NOT commit this file to GitHub** — it's in `.gitignore` automatically.

```bash
# In the project root directory:
cp .env.example .env
```

Then edit `.env` and paste your key:
```
GROQ_API_KEY=your-actual-key-here
GROK_MODEL=openai/gpt-oss-120b
```

(Yes, the variable is `GROQ_API_KEY` but the setting name is `GROK_MODEL`
— left as-is to avoid touching every file that imports from
`llm_upgrade.py`.)

## 3. Run the demo

```bash
python demo_cli.py
```
(Use `python`, not `python3`, on Windows if you hit a "Python was not
found" error from the Microsoft Store alias.)

Type utterances, watch risk escalate. Try `:mode hybrid` to switch modes
live.

## 4. Run the formal evaluation (after sanity check)

```bash
cd dataset
python evaluate_grok.py hybrid
```

This compares Groq's accuracy against the rule-engine baseline (60.8%,
macro-F1 0.439).

---

## How it works

- `load_env.py` — automatically reads `.env` when any module imports it
- `grok_client.py` — imports `load_env` at the top, so your key is loaded
  before any API calls. Talks to `https://api.groq.com/openai/v1` by
  default (Groq's OpenAI-compatible endpoint).
- `.env.example` — template showing what you need (committed to repo, safe)
- `.env` — your actual secrets (NOT committed, stays local only)
- `.gitignore` — already configured to block `.env` and other secret files

**Verify it's working:**
```bash
python -c "import load_env; import os; print(os.environ.get('GROQ_API_KEY', 'NOT SET')[:8] + '...')"
```

Should print the start of your key (or `NOT SET` if `.env` doesn't exist
or the variable name is wrong).

---

## Troubleshooting

**"GROQ_API_KEY not set"**
- Make sure `.env` exists in the project root
- Make sure the file contains `GROQ_API_KEY=your-key` (no quotes, no
  spaces around `=`)
- Try: `type .env` (Windows) or `cat .env` (Mac/Linux) and check the line

**"Incorrect API key provided" / HTTP 401**
- Your key doesn't match what's in your Groq console — double check for
  typos or a truncated copy-paste
- Make sure you're using a key from **console.groq.com**, not
  console.x.ai (they will NOT work interchangeably)

**"Connection refused / Timeout"**
- Check your internet connection
- Try: `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer YOUR_KEY"`
  (should return a list of models, not a connection error)

**"model not found" / HTTP 400 or 404 on a specific model**
- Groq deprecated `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on
  2026-06-17. If you're on an old `.env`, update `GROK_MODEL` to
  `openai/gpt-oss-120b` (or `openai/gpt-oss-20b` for speed).
- Check https://console.groq.com/docs/models for the current catalog if
  the problem persists — Groq's lineup changes frequently.

---

## For your GitHub repo

Everyone cloning your repo will see `.env.example` and know they need to
create their own `.env`. The actual secrets stay local.

```
.env.example  ← commit this (template)
.env          ← DO NOT commit (your local secrets, in .gitignore)
.gitignore    ← already updated to block .env
```
