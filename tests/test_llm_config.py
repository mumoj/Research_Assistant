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
    mr.reset_unavailable()
    yield
    mr._available_models.cache_clear()
    mr.reset_unavailable()


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

    def test_ranks_live_catalogue_newest_first(self):
        listing, configure = catalogue(
            "gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash")
        with listing, configure:
            assert mr.resolve_gemini_model("k") == "gemini-3.8-flash"

    def test_unknown_future_generation_wins_without_code_change(self):
        """A model line we have never heard of must still rank correctly."""
        listing, configure = catalogue("gemini-3.8-flash", "gemini-4.2-flash")
        with listing, configure:
            assert mr.resolve_gemini_model("k") == "gemini-4.2-flash"

    def test_prefers_flash_and_skips_preview_and_lite(self):
        listing, configure = catalogue(
            "gemini-3.6-pro", "gemini-3.6-flash", "gemini-3.6-flash-lite",
            "gemini-4.0-flash-preview")
        with listing, configure:
            assert mr.resolve_gemini_model("k") == "gemini-3.6-flash"

    def test_falls_back_to_any_usable_model(self):
        listing, configure = catalogue("gemini-9-pro")
        with listing, configure:
            assert mr.resolve_gemini_model("k") == "gemini-9-pro"

    def test_listed_but_uncallable_model_is_skipped_after_404(self):
        """The production failure: the catalogue advertised gemini-2.5-flash,
        but calling it returned 'no longer available to new users'."""
        listing, configure = catalogue("gemini-2.5-flash", "gemini-3.6-flash")
        with listing, configure:
            first = mr.resolve_gemini_model("k")
            assert first == "gemini-3.6-flash"
            mr.mark_unavailable(first)
            assert mr.resolve_gemini_model("k") == "gemini-2.5-flash"

    def test_listing_failure_degrades_to_fallback_chain(self):
        listing, configure = catalogue(side_effect=RuntimeError("quota"))
        with listing, configure:
            assert mr.resolve_gemini_model("k") == mr.FALLBACK_GEMINI_MODELS[0]

    def test_fallback_chain_steps_down_when_head_is_retired(self):
        listing, configure = catalogue(side_effect=RuntimeError("quota"))
        with listing, configure:
            mr.mark_unavailable(mr.FALLBACK_GEMINI_MODELS[0])
            assert mr.resolve_gemini_model("k") == mr.FALLBACK_GEMINI_MODELS[1]

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

    def test_never_returns_a_retired_generation(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        listing, configure = catalogue("gemini-3.8-flash")
        with listing, configure:
            model = LLMConfig.get_primary_llm().model
        assert "1.5" not in model and "2.5" not in model


class TestModelNotFoundDetection:
    """Both strings are copied verbatim from production errors."""

    GEMINI_404 = ("404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model "
                  "models/gemini-2.5-flash is no longer available to new users.'}}")
    GROQ_404 = ("Error code: 404 - {'error': {'message': 'The model "
                "llama-3.3-70b-versatile does not exist or you do not have access to it.'}}")

    @pytest.mark.parametrize("message", [GEMINI_404, GROQ_404])
    def test_detects_real_404s(self, message):
        assert mr.is_model_not_found(Exception(message))

    @pytest.mark.parametrize("message", ["429 rate limit exceeded", "invalid api key"])
    def test_ignores_unrelated_errors(self, message):
        assert not mr.is_model_not_found(Exception(message))


class TestSelfHealingInvoke:

    def test_404_on_first_model_steps_to_next(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        calls = []

        class FakeLLM:
            def __init__(self, model):
                self.model = model

            def invoke(self, prompt):
                calls.append(self.model)
                if self.model == "gemini-3.8-flash":
                    raise Exception(TestModelNotFoundDetection.GEMINI_404)
                result = MagicMock()
                result.content = f"answer from {self.model}"
                return result

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm",
            side_effect=lambda: FakeLLM(mr.resolve_gemini_model("fake"))
        ):
            answer = LLMConfig.invoke_primary("q")

        assert answer == "answer from gemini-3.6-flash"
        assert calls == ["gemini-3.8-flash", "gemini-3.6-flash"]

    def test_non_404_error_propagates_without_retrying(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        attempts = []

        class BadKeyLLM:
            model = "gemini-3.8-flash"

            def invoke(self, prompt):
                attempts.append(1)
                raise Exception("401 invalid api key")

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm", return_value=BadKeyLLM()
        ):
            with pytest.raises(Exception, match="invalid api key"):
                LLMConfig.invoke_primary("q")

        # Burning every candidate on an auth failure would mask the real cause.
        assert len(attempts) == 1


class TestValidatorFallback:

    def test_pinned_provider_without_key_falls_through(self, monkeypatch):
        """The deployed failure mode: VALIDATOR_PROVIDER=openai with no OpenAI key."""
        monkeypatch.setenv("VALIDATOR_PROVIDER", "openai")
        monkeypatch.setenv("VALIDATOR_MODEL", "gpt-5.6-luna")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert LLMConfig.resolve_validator_provider() == "groq"
        llm = LLMConfig.get_validator_llm()
        # The OpenAI model name must not leak onto the Groq endpoint.
        assert llm.model_name == "openai/gpt-oss-120b"
        assert "groq.com" in str(llm.openai_api_base)

    def test_pinned_provider_with_key_is_honoured(self, monkeypatch):
        monkeypatch.setenv("VALIDATOR_PROVIDER", "openai")
        monkeypatch.setenv("VALIDATOR_MODEL", "gpt-5.6-luna")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert LLMConfig.get_validator_llm().model_name == "gpt-5.6-luna"

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


class TestValidatorModelRetirement:
    """A retired validator model must not warn the reader of the answer."""

    def state(self):
        return {
            "question": "a question",
            "search_config": {"enable_validation": True},
            "primary_answer": "An answer [1].",
            "web_sources": [], "youtube_sources": [], "error_messages": [],
            "revision_count": 0, "max_revisions": 2,
        }

    def test_validator_404_is_silent(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        failing = MagicMock()
        failing.invoke.side_effect = Exception(TestModelNotFoundDetection.GROQ_404)
        with patch.object(LLMConfig, "get_validator_llm", return_value=failing):
            result = validate_answer_node(self.state())
        assert result["final_answer"] == "An answer [1]."
        assert "error_messages" not in result

    def test_genuine_validation_error_still_warns(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        failing = MagicMock()
        failing.invoke.side_effect = Exception("500 internal server error")
        with patch.object(LLMConfig, "get_validator_llm", return_value=failing):
            result = validate_answer_node(self.state())
        assert result["error_messages"], "real failures should still surface"
