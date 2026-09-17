"""Resolve a usable Gemini model name at runtime.

Google retires Gemini models faster than a pinned name in source can keep up
with: ``gemini-1.5-flash-latest`` 404'd, and the ``gemini-2.5-flash`` that
replaced it was itself closed to new users soon after. Any hard-coded
preference list is a future outage, so this module ranks whatever the API
actually offers and demotes models that turn out not to work.
"""
import os
import re
from functools import lru_cache
from typing import List, Optional, Set, Tuple

import google.generativeai as genai

# Last-resort chain, used only when the catalogue cannot be read at all.
# Verified against ai.google.dev/gemini-api/docs/models on 2026-09-17; the
# catalogue is the real source of truth, these are just the offline backstop.
FALLBACK_GEMINI_MODELS: List[str] = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

# Models whose names signal a non-general-purpose or unstable variant.
_EXCLUDED_MARKERS = ("preview", "-exp", "experimental", "tuning", "embedding", "aqa")

# Names the API advertises but that fail at call time for this key (for
# instance "no longer available to new users"). Populated by mark_unavailable.
_unavailable: Set[str] = set()

_VERSION_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")


def _normalize(name: str) -> str:
    """Strip the ``models/`` prefix the API returns."""
    return name.split("/", 1)[1] if name.startswith("models/") else name


def _rank(name: str) -> Tuple:
    """Sort key: newest generation first, then flash over pro, full over lite.

    Ranking the live catalogue rather than matching a fixed list is what keeps
    this working across model generations we have never heard of.
    """
    match = _VERSION_RE.match(name)
    major = int(match.group(1)) if match else 0
    minor = int(match.group(2)) if match and match.group(2) else 0
    is_flash = "flash" in name
    is_lite = "lite" in name
    # Higher tuple sorts first under reverse=True.
    return (major, minor, is_flash, not is_lite)


@lru_cache(maxsize=4)
def _available_models(api_key: str) -> tuple:
    """List model names that support generateContent for this key."""
    genai.configure(api_key=api_key)
    return tuple(
        _normalize(m.name)
        for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    )


def mark_unavailable(model: str) -> None:
    """Record that a model 404s at call time so the next resolve skips it."""
    _unavailable.add(_normalize(model))


def reset_unavailable() -> None:
    """Clear the runtime blocklist (used by tests)."""
    _unavailable.clear()


def candidate_models(api_key: Optional[str] = None) -> List[str]:
    """Every callable Gemini model for this key, best first."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        available = _available_models(api_key)
    except Exception:
        # Never let model discovery take down answer generation.
        return []

    usable = [
        name for name in available
        if name not in _unavailable
        and "gemini" in name
        and not any(marker in name for marker in _EXCLUDED_MARKERS)
    ]
    return sorted(usable, key=_rank, reverse=True)


def resolve_gemini_model(api_key: Optional[str] = None) -> str:
    """Return the best Gemini model name that is callable with this key.

    ``GEMINI_MODEL`` pins a model explicitly when a deployment needs one; note
    that a pinned model is still subject to retirement, so leaving it unset is
    the resilient choice.
    """
    override = os.getenv("GEMINI_MODEL")
    if override:
        override = _normalize(override.strip())
        if override not in _unavailable:
            return override

    candidates = candidate_models(api_key)
    if candidates:
        return candidates[0]

    # Catalogue unreadable: step through the offline chain instead of pinning
    # a single name that may already be retired.
    for name in FALLBACK_GEMINI_MODELS:
        if name not in _unavailable:
            return name
    return ""


def is_model_not_found(error: BaseException) -> bool:
    """Heuristically detect a 'this model does not exist' API error."""
    text = str(error).lower()
    if "404" in text or "not_found" in text or "not found" in text:
        return True
    return "model" in text and ("does not exist" in text or "no longer available" in text)


# Phrases meaning "this model exists but your key may not spend on it".
_BILLING_MARKERS = (
    "permission_denied", "billing", "paid tier", "free tier",
    "requires a paid", "not available on the free",
)


def is_billing_gated(error: BaseException) -> bool:
    """Detect a model that needs billing this account has not enabled.

    A plain 429 is deliberately NOT treated as billing-gated: that is transient
    throttling, and switching models would mask it rather than fix it.
    """
    text = str(error).lower()
    if "403" in text or "permission_denied" in text:
        return True
    return any(marker in text for marker in _BILLING_MARKERS)


def is_model_unusable(error: BaseException) -> bool:
    """True when the fix is to try a different model rather than give up."""
    return is_model_not_found(error) or is_billing_gated(error)
