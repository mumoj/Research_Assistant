"""LLM configuration module for different providers."""
import os
from typing import Optional, Dict, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel

from . import model_resolver
from .model_resolver import resolve_gemini_model


class ValidatorUnavailable(RuntimeError):
    """Raised when no validator provider has a usable API key configured."""


# (provider, env var holding its key, default model). The order is also the
# auto-detection preference when the configured provider has no key.
# A default model of None means "resolve it at runtime".
#
# Validation is a deterministic, high-volume fact-check, so each default is that
# provider's cheap/fast tier rather than its flagship. Checked against provider
# docs on 2026-09-17 - see get_available_providers for the current alternatives.
VALIDATOR_PROVIDERS: List[Tuple[str, str, Optional[str]]] = [
    ("openai", "OPENAI_API_KEY", "gpt-5.6-luna"),
    ("groq", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
    ("gemini", "GEMINI_API_KEY", None),
]

_PROVIDER_DEFAULTS: Dict[str, Tuple[str, Optional[str]]] = {
    name: (env, default) for name, env, default in VALIDATOR_PROVIDERS
}


class LLMConfig:
    """Configuration class for different LLM providers."""

    @staticmethod
    def get_primary_llm() -> BaseLanguageModel:
        """Get the primary LLM (Gemini) for answer generation."""
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        return ChatGoogleGenerativeAI(
            model=resolve_gemini_model(gemini_api_key),
            temperature=0.2,
            max_tokens=1500,
            google_api_key=gemini_api_key
        )

    @staticmethod
    def invoke_primary(prompt: str, max_attempts: int = 3) -> str:
        """Call the primary LLM, stepping to the next model on a 404.

        The catalogue advertises models that a given key cannot actually call
        ("no longer available to new users"), so a successful call is the only
        real proof a model works. On a not-found error we blocklist that name
        and retry with the next-best candidate rather than failing the search.
        """
        last_error: Optional[BaseException] = None

        for _ in range(max_attempts):
            llm = LLMConfig.get_primary_llm()
            model_name = getattr(llm, "model", "")
            try:
                response = llm.invoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                if not model_resolver.is_model_not_found(exc):
                    raise
                last_error = exc
                if not model_name:
                    break
                model_resolver.mark_unavailable(model_name)

        raise last_error if last_error else RuntimeError("No usable Gemini model")

    @staticmethod
    def detect_validator_provider() -> Optional[str]:
        """Return the first validator provider that has an API key configured."""
        for provider, env_var, _ in VALIDATOR_PROVIDERS:
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
            return ChatOpenAI(model=model_name, temperature=0.0, api_key=api_key)

        if resolved == "anthropic":
            return ChatAnthropic(model=model_name, temperature=0.0, anthropic_api_key=api_key)

        if resolved == "gemini":
            return ChatGoogleGenerativeAI(
                model=model_name or resolve_gemini_model(api_key),
                temperature=0.0,
                google_api_key=api_key
            )

        if resolved == "groq":
            return ChatOpenAI(
                model=model_name,
                temperature=0.0,
                api_key=api_key,
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
