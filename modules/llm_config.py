"""LLM configuration module for different providers."""
import os
from typing import FrozenSet, Optional, Dict, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel

from . import model_resolver
from .model_resolver import resolve_gemini_model


def message_text(response) -> str:
    """Flatten an LLM response to plain text.

    Gemini 3.x returns ``content`` as a list of typed blocks (text plus
    reasoning signatures) rather than a string. Passing that straight through
    put a raw block dump in front of the reader and broke citation parsing,
    which expects text.

    A thinking model that spends its whole output budget reasoning returns
    blocks with no text in them at all -- an empty list, or reasoning blocks
    only. Stringifying that shipped a literal ``[]`` to the reader as the
    answer, so an absent answer now reads as empty and the caller decides how
    to report it.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class ValidatorUnavailable(RuntimeError):
    """Raised when no validator provider has a usable API key configured."""


class PrimaryUnavailable(RuntimeError):
    """Raised when no provider has a usable API key for answer generation."""


class EmptyAnswer(RuntimeError):
    """Raised when a model returns no text at all (all budget spent thinking)."""


# (provider, env var holding its key, default model). The order is also the
# auto-detection preference when the configured provider has no key.
# A default model of None means "resolve it at runtime".
#
# Ordered free tiers first, so auto-detection never reaches for a provider that
# bills by default. Groq and Gemini both have no-cost tiers; OpenAI and
# Anthropic require credit and are only used when named explicitly via
# VALIDATOR_PROVIDER. Each default is the provider's cheap/fast tier, since
# validation is a deterministic high-volume check. Checked 2026-09-17.
VALIDATOR_PROVIDERS: List[Tuple[str, str, Optional[str]]] = [
    ("groq", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    ("gemini", "GEMINI_API_KEY", None),
    ("openai", "OPENAI_API_KEY", "gpt-5.6-luna"),
    ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
]

# Answer generation, same shape and same free-tier-first rule. Groq leads
# because it returns a full cited answer in ~2s where a Gemini thinking model
# takes ~80s on the identical prompt; a Gemini-only deployment still gets
# Gemini, because auto-detection only picks a provider whose key is present.
# PRIMARY_PROVIDER / PRIMARY_MODEL override this the way the validator vars do.
PRIMARY_PROVIDERS: List[Tuple[str, str, Optional[str]]] = [
    ("groq", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    ("gemini", "GEMINI_API_KEY", None),
    ("openai", "OPENAI_API_KEY", "gpt-5.6-luna"),
    ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
]

# Providers that bill from the first request. Auto-detection skips these; they
# are used only when VALIDATOR_PROVIDER names them.
PAID_PROVIDERS = frozenset({"openai", "anthropic"})

# Gemini 3.x reasons before answering. On the answer-generation prompt that
# reasoning consumed 1436 of the 1500 output tokens, so the response hit
# MAX_TOKENS mid-thought and the reader got a fragment of the scratchpad
# instead of an answer. Spending the budget on the answer is both correct and
# ~2.7x faster. Raise GEMINI_THINKING_BUDGET to trade latency back for
# reasoning depth; -1 lets the model choose.
DEFAULT_THINKING_BUDGET = 0

# The SDK default is 6 retries with exponential backoff, which turned a
# throttled model into an 80s stall before we ever saw the error. Fail fast and
# let invoke_primary step to another model instead.
MAX_CLIENT_RETRIES = 1

_PROVIDER_DEFAULTS: Dict[str, Tuple[str, Optional[str]]] = {
    name: (env, default) for name, env, default in VALIDATOR_PROVIDERS
}

_PRIMARY_DEFAULTS: Dict[str, Tuple[str, Optional[str]]] = {
    name: (env, default) for name, env, default in PRIMARY_PROVIDERS
}


def _usable_model(skip: FrozenSet[str], *candidates: Optional[str]) -> Optional[str]:
    """First candidate model name that is not blocklisted or skipped.

    A model pinned by PRIMARY_MODEL is still subject to retirement, so a pin
    that 404s must not trap the retry loop on the same dead name -- the same
    protection resolve_gemini_model already gives GEMINI_MODEL.
    """
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in skip or model_resolver.is_unavailable(candidate):
            continue
        return candidate
    return None


def _thinking_budget() -> int:
    """Gemini reasoning budget, from GEMINI_THINKING_BUDGET."""
    raw = os.getenv("GEMINI_THINKING_BUDGET")
    if raw is None or not raw.strip():
        return DEFAULT_THINKING_BUDGET
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_THINKING_BUDGET


class LLMConfig:
    """Configuration class for different LLM providers."""

    @staticmethod
    def detect_primary_provider(include_paid: bool = False) -> Optional[str]:
        """Return the first free-tier provider that can generate answers."""
        for provider, env_var, _ in PRIMARY_PROVIDERS:
            if provider in PAID_PROVIDERS and not include_paid:
                continue
            if os.getenv(env_var):
                return provider
        return None

    @staticmethod
    def resolve_primary_provider(provider: Optional[str] = None) -> Optional[str]:
        """Pick the answer-generation provider, mirroring the validator rules.

        A provider named by argument or by PRIMARY_PROVIDER is honoured only if
        its key is present; otherwise we fall through to whichever free-tier
        provider does have one.
        """
        requested = (provider or os.getenv("PRIMARY_PROVIDER") or "").strip().lower()
        if requested and requested not in _PRIMARY_DEFAULTS:
            raise ValueError(f"Unsupported provider: {requested}")

        if requested and os.getenv(_PRIMARY_DEFAULTS[requested][0]):
            return requested
        return LLMConfig.detect_primary_provider()

    @staticmethod
    def primary_provider_chain() -> List[str]:
        """Providers to try for one answer: the resolved one, then the rest.

        Falling through to a second provider is what keeps a question
        answerable when the first one rejects the prompt outright -- Groq's
        free tier refuses anything over 8000 tokens/minute, and five long
        sources can exceed that. Paid providers are never added automatically.
        """
        chain: List[str] = []
        first = LLMConfig.resolve_primary_provider()
        if first:
            chain.append(first)
        for provider, env_var, _ in PRIMARY_PROVIDERS:
            if provider in chain or provider in PAID_PROVIDERS:
                continue
            if os.getenv(env_var):
                chain.append(provider)
        return chain

    @staticmethod
    def get_primary_llm(
        skip: FrozenSet[str] = frozenset(),
        provider: Optional[str] = None,
    ) -> BaseLanguageModel:
        """Build the LLM used for answer generation.

        ``skip`` names models to pass over for this request only; see
        :func:`model_resolver.is_transient_overload`.
        """
        requested = (os.getenv("PRIMARY_PROVIDER") or "").strip().lower()
        resolved = provider or LLMConfig.resolve_primary_provider()

        if not resolved:
            # Preserved from the Gemini-only version: a deployment with no keys
            # at all still fails on the key it most likely meant to set.
            raise ValueError("GEMINI_API_KEY not found in environment")

        env_var, default_model = _PRIMARY_DEFAULTS[resolved]
        api_key = os.getenv(env_var)

        # PRIMARY_MODEL names a model for the *requested* provider, so ignore it
        # when we fell back to a different one, exactly as VALIDATOR_MODEL does.
        pinned = os.getenv("PRIMARY_MODEL") if resolved == requested else None
        model_name = _usable_model(skip, pinned, default_model)

        if resolved == "gemini":
            # Gemini has a live catalogue to fall back on, so an exhausted pin
            # is not fatal here.
            model_name = model_name or resolve_gemini_model(api_key, skip=skip)
            if not model_name:
                raise PrimaryUnavailable("No callable Gemini model for this key")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                max_tokens=1500,
                thinking_budget=_thinking_budget(),
                max_retries=MAX_CLIENT_RETRIES,
                google_api_key=api_key,
            )

        if not model_name:
            # Every name we know for this provider has already failed.
            raise PrimaryUnavailable(
                f"No callable {resolved} model left (tried: "
                f"{', '.join(n for n in (pinned, default_model) if n)})"
            )

        if resolved == "groq":
            return ChatOpenAI(
                model=model_name,
                temperature=0.2,
                max_tokens=1500,
                max_retries=MAX_CLIENT_RETRIES,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

        if resolved == "openai":
            return ChatOpenAI(
                model=model_name, temperature=0.2, max_tokens=1500,
                max_retries=MAX_CLIENT_RETRIES, api_key=api_key,
            )

        if resolved == "anthropic":
            return ChatAnthropic(
                model=model_name, temperature=0.2, max_tokens=1500,
                max_retries=MAX_CLIENT_RETRIES, anthropic_api_key=api_key,
            )

        raise ValueError(f"Unsupported provider: {resolved}")

    @staticmethod
    def invoke_primary(prompt: str, max_attempts: int = 3) -> str:
        """Call the primary LLM, stepping to another model when this one cannot answer.

        Two different failures both look like "try a different model", and they
        are handled differently:

        * unusable (404 / billing-gated) -- the name never works for this key,
          so blocklist it for the life of the process.
        * overloaded (429 / 503) -- the name is fine but is throttled or busy
          right now. Skip it for this request only, so the next question can
          use it again once quota resets.

        A prompt the provider rejects for size is a third case: no sibling
        model can take it, so the walk moves straight to the next provider.

        Either way the walk is bounded by ``max_attempts`` per provider, so
        throttling cannot burn through the whole catalogue before surfacing.
        """
        last_error: Optional[BaseException] = None

        for provider in LLMConfig.primary_provider_chain() or [None]:
            skip: set = set()

            for _ in range(max_attempts):
                try:
                    llm = LLMConfig.get_primary_llm(
                        skip=frozenset(skip), provider=provider)
                except PrimaryUnavailable:
                    break  # out of models here; try the next provider
                model_name = (getattr(llm, "model", "")
                              or getattr(llm, "model_name", ""))
                try:
                    answer = message_text(llm.invoke(prompt))
                except Exception as exc:
                    if model_resolver.is_context_too_large(exc):
                        # No model on this provider will take the prompt.
                        last_error = exc
                        break
                    unusable = model_resolver.is_model_unusable(exc)
                    overloaded = model_resolver.is_transient_overload(exc)
                    if not (unusable or overloaded):
                        raise
                    last_error = exc
                    if not model_name:
                        break
                    if unusable:
                        model_resolver.mark_unavailable(model_name)
                    else:
                        skip.add(model_resolver._normalize(model_name))
                    continue

                if answer.strip():
                    return answer

                # No text came back at all. Retrying the same model would only
                # reproduce it, so treat it like an overloaded one and step on.
                last_error = EmptyAnswer(
                    f"{model_name or 'model'} returned no answer text "
                    "(output budget likely spent on reasoning; "
                    "lower GEMINI_THINKING_BUDGET or raise max_tokens)"
                )
                if not model_name:
                    break
                skip.add(model_resolver._normalize(model_name))

        raise last_error if last_error else RuntimeError("No usable primary model")

    @staticmethod
    def detect_validator_provider(include_paid: bool = False) -> Optional[str]:
        """Return the first free-tier provider that has an API key configured.

        Paid providers are skipped unless explicitly requested, so a stray
        OPENAI_API_KEY in the environment never silently starts spending.
        """
        for provider, env_var, _ in VALIDATOR_PROVIDERS:
            if provider in PAID_PROVIDERS and not include_paid:
                continue
            if os.getenv(env_var):
                return provider
        return None

    @staticmethod
    def resolve_validator_provider(provider: Optional[str] = None) -> Optional[str]:
        """Pick the validator provider to actually use.

        A provider named by argument or by VALIDATOR_PROVIDER is only honoured if
        its key is present; otherwise we fall through to whichever provider does
        have a key. This keeps validation alive on deployments that set
        VALIDATOR_PROVIDER=openai but never added OPENAI_API_KEY.
        """
        requested = (provider or os.getenv("VALIDATOR_PROVIDER") or "").strip().lower()
        if requested and requested not in _PROVIDER_DEFAULTS:
            raise ValueError(f"Unsupported provider: {requested}")

        if requested and os.getenv(_PROVIDER_DEFAULTS[requested][0]):
            return requested
        return LLMConfig.detect_validator_provider()

    @staticmethod
    def get_validator_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> BaseLanguageModel:
        """Get the validator LLM based on configuration."""
        requested = (provider or os.getenv("VALIDATOR_PROVIDER") or "").strip().lower()
        resolved = LLMConfig.resolve_validator_provider(provider)

        if not resolved:
            raise ValidatorUnavailable(
                "No validator API key configured. Set one of: "
                + ", ".join(env for _, env, _ in VALIDATOR_PROVIDERS)
            )

        env_var, default_model = _PROVIDER_DEFAULTS[resolved]
        api_key = os.getenv(env_var)

        # VALIDATOR_MODEL names a model for the *requested* provider, so ignore it
        # when we fell back to a different one (a gpt-4o-mini name would 404 on Groq).
        if model is None and resolved == requested:
            model = os.getenv("VALIDATOR_MODEL")
        model_name = model or default_model

        if resolved == "openai":
            return ChatOpenAI(model=model_name, temperature=0.0, api_key=api_key,
                              max_retries=MAX_CLIENT_RETRIES)

        if resolved == "anthropic":
            return ChatAnthropic(model=model_name, temperature=0.0, anthropic_api_key=api_key,
                                 max_retries=MAX_CLIENT_RETRIES)

        if resolved == "gemini":
            return ChatGoogleGenerativeAI(
                model=model_name or resolve_gemini_model(api_key),
                temperature=0.0,
                thinking_budget=_thinking_budget(),
                max_retries=MAX_CLIENT_RETRIES,
                google_api_key=api_key
            )

        if resolved == "groq":
            return ChatOpenAI(
                model=model_name,
                temperature=0.0,
                api_key=api_key,
                max_retries=MAX_CLIENT_RETRIES,
                base_url="https://api.groq.com/openai/v1"
            )

        raise ValueError(f"Unsupported provider: {resolved}")

    @staticmethod
    def get_available_providers() -> Dict[str, list]:
        """Get list of available providers and their models."""
        return {
            "openai": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-6-astra"],
            "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-5"],
            "gemini": ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
            "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
        }

    @staticmethod
    def validate_config() -> Dict[str, bool]:
        """Validate API key configuration."""
        return {
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "youtube": bool(os.getenv("YOUTUBE_API_KEY")),
            "serpapi": bool(os.getenv("SERPAPI_KEY"))
        }
