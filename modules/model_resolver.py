"""Resolve a usable Gemini model name at runtime.

The Gemini API retires model aliases (``gemini-1.5-flash-latest`` now 404s on
v1beta), so pinning a single name in code breaks the app whenever Google rotates
the catalogue. This module asks the API which models the configured key can
actually call and picks the best available one.
"""
import os
from functools import lru_cache
from typing import List, Optional

import google.generativeai as genai

# Preference order, best first. The resolver picks the first one the API key
# can actually call.
PREFERRED_GEMINI_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-001",
]

# Used when the catalogue cannot be fetched (offline, quota, transient error).
FALLBACK_GEMINI_MODEL: str = PREFERRED_GEMINI_MODELS[0]


def _normalize(name: str) -> str:
    """Strip the ``models/`` prefix the API returns."""
    return name.split("/", 1)[1] if name.startswith("models/") else name


@lru_cache(maxsize=4)
def _available_models(api_key: str) -> tuple:
    """List model names that support generateContent for this key."""
    genai.configure(api_key=api_key)
    return tuple(
        _normalize(m.name)
        for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    )


def resolve_gemini_model(api_key: Optional[str] = None) -> str:
    """Return a Gemini model name that is callable with the configured key.

    An explicit ``GEMINI_MODEL`` env var always wins, so the deployment can pin a
    model without a code change. Otherwise the catalogue is queried once per
    process and matched against ``PREFERRED_GEMINI_MODELS``.
    """
    override = os.getenv("GEMINI_MODEL")
    if override:
        return _normalize(override.strip())

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return FALLBACK_GEMINI_MODEL

    try:
        available = _available_models(api_key)
    except Exception:
        # Never let model discovery take down answer generation.
        return FALLBACK_GEMINI_MODEL

    for candidate in PREFERRED_GEMINI_MODELS:
        if candidate in available:
            return candidate

    # Nothing from the preference list: take any non-preview flash/pro model.
    for name in available:
        if ("flash" in name or "pro" in name) and "preview" not in name:
            return name

    return available[0] if available else FALLBACK_GEMINI_MODEL
