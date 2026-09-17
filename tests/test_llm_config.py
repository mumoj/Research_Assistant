"""Regression tests for model resolution and validator provider fallback."""
import pytest
from unittest.mock import patch, MagicMock

import modules.model_resolver as mr
from modules.llm_config import (
    LLMConfig, ValidatorUnavailable, EmptyAnswer, message_text)
from modules.workflow_nodes import validate_answer_node, revise_answer_node
from modules.workflow_state import ValidationResult


KEY_VARS = [
    "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
    "VALIDATOR_PROVIDER", "VALIDATOR_MODEL", "GEMINI_MODEL",
    "PRIMARY_PROVIDER", "PRIMARY_MODEL", "GEMINI_THINKING_BUDGET",
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
            side_effect=lambda skip=frozenset(), **_: FakeLLM(
                mr.resolve_gemini_model("fake", skip=skip))
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


class TestNoUnintendedSpend:
    """Auto-detection must never start billing on its own."""

    def test_stray_openai_key_is_not_auto_selected(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert LLMConfig.detect_validator_provider() is None
        with pytest.raises(ValidatorUnavailable):
            LLMConfig.get_validator_llm()

    def test_stray_anthropic_key_is_not_auto_selected(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        assert LLMConfig.detect_validator_provider() is None

    def test_free_provider_wins_over_paid_when_both_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert LLMConfig.detect_validator_provider() == "groq"

    def test_paid_provider_used_only_when_named_explicitly(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("VALIDATOR_PROVIDER", "openai")
        assert LLMConfig.resolve_validator_provider() == "openai"

    def test_groq_preferred_over_gemini_to_spare_gemini_quota(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert LLMConfig.detect_validator_provider() == "groq"


class TestBillingGateDetection:

    BILLING_403 = ("403 PERMISSION_DENIED: Gemini API free tier is not available "
                   "for this model. Please enable billing.")

    def test_billing_gated_model_is_detected(self):
        assert mr.is_billing_gated(Exception(self.BILLING_403))
        assert mr.is_model_unusable(Exception(self.BILLING_403))

    def test_transient_rate_limit_is_not_billing_gated(self):
        """Switching models on a 429 would mask throttling instead of fixing it."""
        err = Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded, retry later")
        assert not mr.is_billing_gated(err)
        assert not mr.is_model_unusable(err)

    def test_billing_gated_model_steps_down_to_free_one(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        calls = []

        class FakeLLM:
            def __init__(self, model):
                self.model = model

            def invoke(self, prompt):
                calls.append(self.model)
                if self.model == "gemini-3.8-flash":
                    raise Exception(TestBillingGateDetection.BILLING_403)
                result = MagicMock()
                result.content = f"answer from {self.model}"
                return result

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm",
            side_effect=lambda skip=frozenset(), **_: FakeLLM(
                mr.resolve_gemini_model("fake", skip=skip))
        ):
            answer = LLMConfig.invoke_primary("q")

        assert answer == "answer from gemini-3.6-flash"
        assert calls == ["gemini-3.8-flash", "gemini-3.6-flash"]

    def test_rate_limit_steps_over_the_throttled_model(self, monkeypatch):
        """Free-tier quota is metered per model, so a sibling usually answers.

        Replaces an earlier rule that stopped dead on the first 429. Measured
        against the live API, gemini-3.8-flash was quota-exhausted and
        gemini-3.7-flash was returning 503 while gemini-3.5-flash answered
        normally, so refusing to step meant failing a request that could
        have been served.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        calls = []

        class FakeLLM:
            def __init__(self, model):
                self.model = model

            def invoke(self, prompt):
                calls.append(self.model)
                if self.model == "gemini-3.8-flash":
                    raise Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded")
                result = MagicMock()
                result.content = f"answer from {self.model}"
                return result

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm",
            side_effect=lambda skip=frozenset(), **_: FakeLLM(
                mr.resolve_gemini_model("fake", skip=skip))
        ):
            answer = LLMConfig.invoke_primary("q")

        assert answer == "answer from gemini-3.6-flash"
        assert calls == ["gemini-3.8-flash", "gemini-3.6-flash"]

    def test_throttling_is_bounded_and_not_permanent(self, monkeypatch):
        """Throttling must not burn the catalogue, nor blocklist a good model."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        attempts = []

        class ThrottledLLM:
            def __init__(self, model):
                self.model = model

            def invoke(self, prompt):
                attempts.append(self.model)
                raise Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded")

        listing, configure = catalogue(
            "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm",
            side_effect=lambda skip=frozenset(), **_: ThrottledLLM(
                mr.resolve_gemini_model("fake", skip=skip))
        ):
            with pytest.raises(Exception, match="429"):
                LLMConfig.invoke_primary("q")

        # Bounded by max_attempts, and each attempt is a *different* model.
        assert attempts == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
        # Throttling is temporary, so nothing is blocklisted for the process.
        assert mr.resolve_gemini_model("fake") == "gemini-3.8-flash"


class TestAnswerTextExtraction:
    """A thinking model can return blocks that contain no answer at all."""

    class Response:
        def __init__(self, content):
            self.content = content

    def test_text_blocks_are_joined(self):
        assert message_text(self.Response(
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])) == "ab"

    def test_plain_string_is_passed_through(self):
        assert message_text(self.Response("hello")) == "hello"

    def test_empty_block_list_is_empty_not_literal_brackets(self):
        """The shipped bug: the reader saw '[]' where the answer should be."""
        assert message_text(self.Response([])) == ""

    def test_reasoning_only_response_yields_no_text(self):
        reasoning = [{"type": "thinking", "thinking": "weighing the sources"}]
        assert message_text(self.Response(reasoning)) == ""


class TestEmptyAnswerIsNotShipped:

    def test_answerless_model_steps_aside_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        calls = []

        class FakeLLM:
            def __init__(self, model):
                self.model = model

            def invoke(self, prompt):
                calls.append(self.model)
                result = MagicMock()
                # The newest model burns its budget thinking and returns nothing.
                result.content = [] if self.model == "gemini-3.8-flash" else "a real answer"
                return result

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm",
            side_effect=lambda skip=frozenset(), **_: FakeLLM(
                mr.resolve_gemini_model("fake", skip=skip))
        ):
            assert LLMConfig.invoke_primary("q") == "a real answer"
        assert calls == ["gemini-3.8-flash", "gemini-3.6-flash"]

    def test_all_models_answerless_raises_rather_than_returning_junk(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")

        class SilentLLM:
            def __init__(self, model):
                self.model = model

            def invoke(self, prompt):
                result = MagicMock()
                result.content = []
                return result

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm",
            side_effect=lambda skip=frozenset(), **_: SilentLLM(
                mr.resolve_gemini_model("fake", skip=skip))
        ):
            with pytest.raises(EmptyAnswer):
                LLMConfig.invoke_primary("q")


class TestPrimaryProviderSelection:
    """Answer generation follows the same free-tier-first rules as validation."""

    def test_groq_is_preferred_when_available(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert LLMConfig.resolve_primary_provider() == "groq"
        llm = LLMConfig.get_primary_llm()
        assert llm.model_name == "openai/gpt-oss-120b"
        assert "groq.com" in str(llm.openai_api_base)

    def test_gemini_only_deployment_still_uses_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        listing, configure = catalogue("gemini-3.8-flash")
        with listing, configure:
            assert LLMConfig.resolve_primary_provider() == "gemini"
            assert LLMConfig.get_primary_llm().model == "gemini-3.8-flash"

    def test_pinned_provider_without_key_falls_through(self, monkeypatch):
        monkeypatch.setenv("PRIMARY_PROVIDER", "openai")
        monkeypatch.setenv("PRIMARY_MODEL", "gpt-5.6-luna")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        llm = LLMConfig.get_primary_llm()
        # The OpenAI model name must not leak onto the Groq endpoint.
        assert llm.model_name == "openai/gpt-oss-120b"

    def test_stray_paid_key_is_never_auto_selected(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert LLMConfig.detect_primary_provider() is None

    def test_paid_provider_used_only_when_named(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("PRIMARY_PROVIDER", "openai")
        assert LLMConfig.resolve_primary_provider() == "openai"

    def test_thinking_budget_defaults_to_zero_for_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        listing, configure = catalogue("gemini-3.8-flash")
        with listing, configure:
            assert LLMConfig.get_primary_llm().thinking_budget == 0

    def test_thinking_budget_is_configurable(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("GEMINI_THINKING_BUDGET", "512")
        listing, configure = catalogue("gemini-3.8-flash")
        with listing, configure:
            assert LLMConfig.get_primary_llm().thinking_budget == 512

    def test_garbage_thinking_budget_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("GEMINI_THINKING_BUDGET", "lots")
        listing, configure = catalogue("gemini-3.8-flash")
        with listing, configure:
            assert LLMConfig.get_primary_llm().thinking_budget == 0

    def test_no_keys_at_all_still_reports_the_missing_gemini_key(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            LLMConfig.get_primary_llm()


class TestPinnedPrimaryModelRecovers:
    """A pin that 404s must not trap the retry loop on the same dead name."""

    def test_pinned_gemini_model_is_dropped_once_it_proves_uncallable(self, monkeypatch):
        """PRIMARY_MODEL naming a model this key cannot call must not be sticky."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        monkeypatch.setenv("PRIMARY_PROVIDER", "gemini")
        monkeypatch.setenv("PRIMARY_MODEL", "gemini-1.5-flash-latest")

        listing, configure = catalogue("gemini-3.8-flash")
        with listing, configure:
            assert LLMConfig.get_primary_llm().model == "gemini-1.5-flash-latest"
            # The 404 handler in invoke_primary blocklists it...
            mr.mark_unavailable("gemini-1.5-flash-latest")
            # ...and the next build resolves from the live catalogue instead.
            assert LLMConfig.get_primary_llm().model == "gemini-3.8-flash"

    def test_throttled_pin_is_skipped_for_this_request_only(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        monkeypatch.setenv("PRIMARY_PROVIDER", "gemini")
        monkeypatch.setenv("PRIMARY_MODEL", "gemini-3.8-flash")

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure:
            skipped = frozenset({"gemini-3.8-flash"})
            assert LLMConfig.get_primary_llm(skip=skipped).model == "gemini-3.6-flash"
            # Nothing was blocklisted, so the pin is back for the next question.
            assert LLMConfig.get_primary_llm().model == "gemini-3.8-flash"

    def test_provider_with_no_models_left_surfaces_the_real_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        attempts = []

        class DeadLLM:
            model_name = "openai/gpt-oss-120b"

            def invoke(self, prompt):
                attempts.append(1)
                raise Exception(TestModelNotFoundDetection.GROQ_404)

        with patch.object(LLMConfig, "get_primary_llm", return_value=DeadLLM()):
            with pytest.raises(Exception, match="does not exist"):
                LLMConfig.invoke_primary("q")
        assert len(attempts) == 3


class TestRealQuotaMessageIsNotMistakenForBilling:
    """Gemini phrases free-tier exhaustion in the language of a billing gate."""

    # Verbatim from the live API on 2026-09-17.
    QUOTA_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
                 "'You exceeded your current quota, please check your plan and "
                 "billing details.', 'status': 'RESOURCE_EXHAUSTED'}}")
    OVERLOAD_503 = ("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model "
                    "is currently experiencing high demand. Spikes in demand are "
                    "usually temporary. Please try again later.'}}")

    def test_quota_message_is_not_billing_gated(self):
        err = Exception(self.QUOTA_429)
        assert not mr.is_billing_gated(err), "mentioning billing is not being billing-gated"
        assert not mr.is_model_unusable(err), "a throttled model must not be retired"
        assert mr.is_transient_overload(err)

    def test_overload_is_transient_not_terminal(self):
        err = Exception(self.OVERLOAD_503)
        assert not mr.is_model_unusable(err)
        assert mr.is_transient_overload(err)

    def test_real_billing_gate_still_detected(self):
        err = Exception(TestBillingGateDetection.BILLING_403)
        assert mr.is_billing_gated(err) and mr.is_model_unusable(err)

    def test_throttled_model_is_not_blocklisted_for_the_process(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")

        class ThrottledFirst:
            def __init__(self, model):
                self.model = model

            def invoke(self, prompt):
                if self.model == "gemini-3.8-flash":
                    raise Exception(TestRealQuotaMessageIsNotMistakenForBilling.QUOTA_429)
                result = MagicMock()
                result.content = "ok"
                return result

        listing, configure = catalogue("gemini-3.8-flash", "gemini-3.6-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm",
            side_effect=lambda skip=frozenset(), **_: ThrottledFirst(
                mr.resolve_gemini_model("fake", skip=skip))
        ):
            assert LLMConfig.invoke_primary("q") == "ok"

        # The next question must still reach for the best model.
        assert not mr._unavailable


class TestRevisionNeverBlanksTheAnswer:

    def state(self):
        return {
            "question": "a question",
            "search_config": {"enable_validation": True},
            "primary_answer": "An answer [1].",
            "validation_result": ValidationResult(
                status="NEEDS_REVISION", issues="x", confidence=50, recommendations="y"),
            "web_sources": [], "youtube_sources": [], "error_messages": [],
            "revision_count": 0, "max_revisions": 2,
        }

    def test_empty_revision_keeps_the_original(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        silent = MagicMock()
        silent.invoke.return_value = MagicMock(content=[])
        with patch.object(LLMConfig, "get_validator_llm", return_value=silent):
            result = revise_answer_node(self.state())
        assert result["primary_answer"] == "An answer [1]."
        assert result["needs_revision"] is False

    def test_real_revision_replaces_the_answer(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        better = MagicMock()
        better.invoke.return_value = MagicMock(content="A better answer [1].")
        with patch.object(LLMConfig, "get_validator_llm", return_value=better):
            result = revise_answer_node(self.state())
        assert result["primary_answer"] == "A better answer [1]."


class TestCrossProviderFallback:
    """Groq's free tier caps at 8000 tokens/minute; long source sets exceed it."""

    TOO_LARGE_413 = (
        "Error code: 413 - {'error': {'message': 'Request too large for model "
        "`openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` "
        "on tokens per minute (TPM): Limit 8000, Requested 8117, please reduce "
        "your message size and try again', 'code': 'rate_limit_exceeded'}}")

    def test_oversized_prompt_falls_through_to_the_other_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        tried = []

        def build(skip=frozenset(), provider=None):
            llm = MagicMock()
            llm.model = "gemini-3.8-flash" if provider == "gemini" else ""
            llm.model_name = "openai/gpt-oss-120b" if provider == "groq" else ""

            def invoke(prompt):
                tried.append(provider)
                if provider == "groq":
                    raise Exception(TestCrossProviderFallback.TOO_LARGE_413)
                return MagicMock(content="the gemini answer")

            llm.invoke = invoke
            return llm

        listing, configure = catalogue("gemini-3.8-flash")
        with listing, configure, patch.object(
            LLMConfig, "get_primary_llm", side_effect=build
        ):
            answer = LLMConfig.invoke_primary("a very long prompt")

        assert answer == "the gemini answer"
        # Groq is asked once -- no sibling model of its would fit either.
        assert tried == ["groq", "gemini"]

    def test_chain_puts_the_resolved_provider_first(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        assert LLMConfig.primary_provider_chain() == ["groq", "gemini"]

        monkeypatch.setenv("PRIMARY_PROVIDER", "gemini")
        assert LLMConfig.primary_provider_chain() == ["gemini", "groq"]

    def test_chain_never_adds_a_paid_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        assert LLMConfig.primary_provider_chain() == ["groq"]

    def test_paid_provider_is_used_alone_when_named(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        monkeypatch.setenv("PRIMARY_PROVIDER", "openai")
        assert LLMConfig.primary_provider_chain() == ["openai"]

    def test_sole_provider_still_surfaces_its_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        dead = MagicMock()
        dead.model_name = "openai/gpt-oss-120b"
        dead.model = ""
        dead.invoke.side_effect = Exception(TestCrossProviderFallback.TOO_LARGE_413)
        with patch.object(LLMConfig, "get_primary_llm", return_value=dead):
            with pytest.raises(Exception, match="Request too large"):
                LLMConfig.invoke_primary("q")
