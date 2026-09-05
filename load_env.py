"""
load_env.py — read .env file if it exists, load vars into os.environ.

Pure Python, no external dependencies. Searches for .env in the current
working directory and one level up (common pattern for projects with
subdirectories like dataset/).

Usage:
    import load_env  # automatically loads .env if it exists
    os.environ.get("XAI_API_KEY")  # will have the value from .env
"""

import os
from pathlib import Path


def load_env():
    """Load .env file if it exists. Try current dir and parent dir."""
    env_paths = [
        Path(".env"),
        Path("../.env"),
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            _parse_env_file(env_path)
            print(f"[load_env] loaded from {env_path}")
            return
    
    # .env not found — that's OK, os.environ.get() will just return None
    # and grok_client.GrokClient will raise an error saying the key is missing,
    # which is the intended behavior.


def _parse_env_file(path):
    """Parse a simple .env file (line-by-line, KEY=value format)."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"\'')  # allow quoted values
            os.environ[key] = value


# Load on import
load_env()
