"""Regression tests for model resolution and validator provider fallback."""
import pytest
from unittest.mock import patch, MagicMock

import modules.model_resolver as mr
from modules.llm_config import LLMConfig, ValidatorUnavailable
from modules.workflow_nodes import validate_answer_node, revise_answer_node


KEY_VARS = [
    "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
    "VALIDATOR_PROVIDER", "VALIDATOR_MODEL", "GEMINI_MODEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a known-empty configuration."""
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    mr._available_models.cache_clear()
    yield
    mr._available_models.cache_clear()


def fake_models(*names):
    models = []
    for name in names:
        model = MagicMock()
        model.name = f"models/{name}"
        model.supported_generation_methods = ["generateContent"]
        models.append(model)
    return models


def catalogue(*names, side_effect=None):
    kwargs = {"side_effect": side_effect} if side_effect else {"return_value": fake_models(*names)}
    return patch.object(mr.genai, "list_models", **kwargs), patch.object(mr.genai, "configure")


class TestModelResolver:

    def test_picks_best_available_preferred_model(self):
        listing, configure = catalogue("gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.0-pro")
        with listing, configure:
            assert mr.resolve_gemini_model("k") == "gemini-2.5-flash"

    def test_falls_back_to_any_usable_model(self):
        listing, configure = catalogue("gemini-9-pro")
        with listing, configure:
            assert mr.resolve_gemini_model("k") == "gemini-9-pro"

    def test_listing_failure_degrades_to_static_default(self):
        listing, configure = catalogue(side_effect=RuntimeError("quota"))
        with listing, configure:
            assert mr.resolve_gemini_model("k") == mr.FALLBACK_GEMINI_MODEL

    def test_env_override_wins_without_network_call(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "models/gemini-2.0-flash")
        with patch.object(mr.genai, "list_models", side_effect=AssertionError("must not call")):
            assert mr.resolve_gemini_model("k") == "gemini-2.0-flash"

    def test_catalogue_is_cached_per_key(self):
        listed = MagicMock(return_value=fake_models("gemini-2.5-flash"))
        with patch.object(mr.genai, "list_models", listed), patch.object(mr.genai, "configure"):
            for _ in range(3):
                mr.resolve_gemini_model("k")
        assert listed.call_count == 1

    def test_never_returns_the_retired_alias(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        listing, configure = catalogue("gemini-2.5-flash")
        with listing, configure:
            assert "1.5" not in LLMConfig.get_primary_llm().model


class TestValidatorFallback:

    def test_pinned_provider_without_key_falls_through(self, monkeypatch):
        """The deployed failure mode: VALIDATOR_PROVIDER=openai with no OpenAI key."""
        monkeypatch.setenv("VALIDATOR_PROVIDER", "openai")
        monkeypatch.setenv("VALIDATOR_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert LLMConfig.resolve_validator_provider() == "groq"
        llm = LLMConfig.get_validator_llm()
        # The gpt model name must not leak onto the Groq endpoint.
        assert llm.model_name == "llama-3.3-70b-versatile"
        assert "groq.com" in str(llm.openai_api_base)

    def test_pinned_provider_with_key_is_honoured(self, monkeypatch):
        monkeypatch.setenv("VALIDATOR_PROVIDER", "openai")
        monkeypatch.setenv("VALIDATOR_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert LLMConfig.get_validator_llm().model_name == "gpt-4o-mini"

    def test_gemini_only_deployment_uses_gemini(self, monkeypatch):
        monkeypatch.setenv("VALIDATOR_PROVIDER", "openai")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        assert LLMConfig.resolve_validator_provider() == "gemini"

    def test_no_keys_raises_validator_unavailable(self):
        with pytest.raises(ValidatorUnavailable):
            LLMConfig.get_validator_llm()

    def test_unsupported_provider_still_rejected(self, monkeypatch):
        monkeypatch.setenv("VALIDATOR_PROVIDER", "bogus")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        with pytest.raises(ValueError):
            LLMConfig.get_validator_llm()


class TestValidationDegradesGracefully:

    def base_state(self):
        return {
            "search_config": {"enable_validation": True},
            "primary_answer": "An answer [1].",
            "web_sources": [], "youtube_sources": [], "error_messages": [],
            "revision_count": 0, "max_revisions": 2,
        }

    def test_answer_survives_missing_validator_key(self):
        result = validate_answer_node(self.base_state())
        assert result["final_answer"] == "An answer [1]."
        assert result["validation_result"] is None
        assert result["needs_revision"] is False
        # No error surfaced to the user for an optional feature.
        assert "error_messages" not in result

    def test_revision_node_degrades_too(self):
        result = revise_answer_node(self.base_state())
        assert result["primary_answer"] == "An answer [1]."
        assert result["needs_revision"] is False
