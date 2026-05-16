"""Public configuration for Ideas Have Genomes.

All secrets are read from environment variables. Imports work in a fresh
checkout; real API calls require the corresponding environment variables.
"""

from __future__ import annotations

import os

import httpx


def get_config(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


BASE_URL = get_config("BASE_URL", "https://api.openai.com/v1")
API_KEY = get_config("API_KEY", "")
MODEL_NAME = get_config("MODEL_NAME", "gpt-4o")

HTTP_CLIENT = httpx.Client(trust_env=False, timeout=httpx.Timeout(300.0, connect=15.0))
