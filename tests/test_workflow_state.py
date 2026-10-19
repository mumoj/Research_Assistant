"""Unit tests for workflow state models and initialization."""
import pytest
from pydantic import ValidationError

import modules.workflow_state as ws


class TestSearchResult:

    def test_snippet_defaults_to_empty_string(self):
        """A search result omits its snippet by default."""
        result = ws.SearchResult(title="Example", url="https://example.com")

        assert result.snippet == ""

    def test_title_and_url_are_required(self):
        """A search result requires title and URL."""
        with pytest.raises(ValidationError):
            ws.SearchResult()

    def test_identical_instances_are_equal(self):
        """Identical search results compare equal."""
        first = ws.SearchResult(title="Example", url="https://example.com")
        second = ws.SearchResult(title="Example", url="https://example.com")

        assert first == second


class TestWebSource:

    def test_all_fields_are_required(self):
        """A web source requires title, URL, and content."""
        with pytest.raises(ValidationError):
            ws.WebSource()

    def test_round_trips_through_model_dump(self):
        """A web source round-trips through its dumped fields."""
        source = ws.WebSource(
            title="Example", url="https://example.com", content="Content"
        )

        assert ws.WebSource(**source.model_dump()) == source


class TestYouTubeSource:

    def test_accepts_transcript_dicts(self):
        """A YouTube source accepts a list of transcript dictionaries."""
        source = ws.YouTubeSource(
            id="video-id",
            title="Video",
            url="https://youtube.com/watch?v=video-id",
            transcript=[{"text": "Hello", "start": 0.0}],
            transcript_text="Hello",
        )

        assert source.transcript == [{"text": "Hello", "start": 0.0}]

    def test_transcript_text_is_required(self):
        """A YouTube source requires transcript text."""
        with pytest.raises(ValidationError):
            ws.YouTubeSource(
                id="video-id",
                title="Video",
                url="https://youtube.com/watch?v=video-id",
                transcript=[],
            )


class TestValidationResult:

    def test_coerces_numeric_confidence_string(self):
        """Pydantic coerces a numeric confidence string to an integer."""
        result = ws.ValidationResult(
            status="approved",
            issues="",
            confidence="85",
            recommendations="",
        )

        assert result.confidence == 85
        assert isinstance(result.confidence, int)

    def test_rejects_non_numeric_confidence(self):
        """A non-numeric confidence value fails validation."""
        with pytest.raises(ValidationError):
            ws.ValidationResult(
                status="approved",
                issues="",
                confidence="high",
                recommendations="",
            )


class TestCreateInitialState:

    def test_default_state_has_documented_defaults(self):
        """The default state contains the documented initial values."""
        state = ws.create_initial_state("What is the answer?")

        assert state == {
            "question": "What is the answer?",
            "search_config": {
                "sources": "Both",
                "max_web_results": 5,
                "max_youtube_results": 3,
                "enable_validation": True,
                "confidence_threshold": 80,
            },
            "web_results": [],
            "youtube_results": [],
            "web_sources": [],
            "youtube_sources": [],
            "primary_answer": "",
            "validation_result": None,
            "final_answer": "",
            "sources_html": "",
            "needs_revision": False,
            "revision_count": 0,
            "max_revisions": 2,
            "processing_time": 0.0,
            "error_messages": [],
            "debug_info": {},
        }

    def test_state_keys_match_research_state_annotations(self):
        """The factory returns exactly the declared state keys."""
        state = ws.create_initial_state("Question")

        assert set(state) == set(ws.ResearchState.__annotations__)

    def test_custom_search_config_is_used_as_is(self):
        """A non-empty custom config is retained by identity."""
        search_config = {"sources": "Web Only", "max_web_results": 1}

        state = ws.create_initial_state("Question", search_config)

        assert state["search_config"] is search_config

    def test_empty_search_config_uses_default_config(self):
        """An empty config falls back to the default configuration."""
        state = ws.create_initial_state("Question", {})

        assert state["search_config"] == {
            "sources": "Both",
            "max_web_results": 5,
            "max_youtube_results": 3,
            "enable_validation": True,
            "confidence_threshold": 80,
        }

    def test_states_have_independent_mutable_containers(self):
        """Separate factory calls do not share mutable defaults."""
        first = ws.create_initial_state("First")
        second = ws.create_initial_state("Second")

        first["error_messages"].append("error")
        first["web_results"].append("result")
        first["debug_info"]["attempt"] = 1
        first["search_config"]["sources"] = "Web Only"

        assert second["error_messages"] == []
        assert second["web_results"] == []
        assert second["debug_info"] == {}
        assert second["search_config"]["sources"] == "Both"
