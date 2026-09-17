"""The validator's reply must parse whether or not it echoes the placeholders.

Requiring literal square brackets made every field come back "Unknown", which
left the revise loop permanently switched off.
"""
import pytest

from modules.prompts import format_validation_response


# Verbatim from Groq openai/gpt-oss-20b on 2026-09-17.
REAL_REPLY = """VALIDATION_STATUS: NEEDS_REVISION  
ISSUES_FOUND:  
- The answer is overly brief and does not fully address the question.
- The claim "QUIC is a UDP transport" is incomplete.
- Citation [1] is not verified against the provided source (https://a.test).

CONFIDENCE_SCORE: 35  

RECOMMENDED_CHANGES:  
- Expand the definition to mention multiplexing and encryption.
"""

APPROVED_REPLY = """VALIDATION_STATUS: APPROVED
ISSUES_FOUND: None
CONFIDENCE_SCORE: 92
RECOMMENDED_CHANGES: None
"""

# The format the old parser was written for.
BRACKETED_REPLY = """VALIDATION_STATUS: [APPROVED]
ISSUES_FOUND: [None]
CONFIDENCE_SCORE: [88]
RECOMMENDED_CHANGES: [None]
"""

MARKDOWN_REPLY = """**VALIDATION_STATUS:** REJECTED
**ISSUES_FOUND:** The answer cites a source that does not exist.
**CONFIDENCE_SCORE:** 10
**RECOMMENDED_CHANGES:** Start over from the sources.
"""


class TestStatusParsing:

    def test_bare_status_is_read(self):
        assert format_validation_response(REAL_REPLY)["status"] == "NEEDS_REVISION"

    def test_bracketed_status_still_read(self):
        """The old format must keep working."""
        assert format_validation_response(BRACKETED_REPLY)["status"] == "APPROVED"

    def test_markdown_emphasis_is_tolerated(self):
        assert format_validation_response(MARKDOWN_REPLY)["status"] == "REJECTED"

    def test_status_keyword_survives_trailing_commentary(self):
        reply = "VALIDATION_STATUS: NEEDS_REVISION (see the issues below)"
        assert format_validation_response(reply)["status"] == "NEEDS_REVISION"

    def test_unparseable_status_is_unknown_not_a_crash(self):
        assert format_validation_response("total gibberish")["status"] == "Unknown"

    def test_empty_input_is_safe(self):
        parsed = format_validation_response("")
        assert parsed["status"] == "Unknown" and parsed["confidence"] == 0


class TestConfidenceParsing:

    @pytest.mark.parametrize("line,expected", [
        ("CONFIDENCE_SCORE: 35", 35),
        ("CONFIDENCE_SCORE: [88]", 88),
        ("CONFIDENCE_SCORE: 72/100", 72),
        ("CONFIDENCE_SCORE: 60 - fairly confident", 60),
        ("CONFIDENCE_SCORE: high", 0),
    ])
    def test_confidence_forms(self, line, expected):
        assert format_validation_response(line)["confidence"] == expected

    def test_real_reply_confidence(self):
        assert format_validation_response(REAL_REPLY)["confidence"] == 35


class TestMultilineFields:

    def test_issues_capture_every_bullet(self):
        issues = format_validation_response(REAL_REPLY)["issues"]
        assert "overly brief" in issues
        assert "multiplexing" not in issues, "must stop before RECOMMENDED_CHANGES"
        assert issues.count("\n- ") == 2

    def test_bracketed_citation_inside_issues_does_not_truncate(self):
        """The body quotes [1]; stopping at the first ']' would cut it short."""
        issues = format_validation_response(REAL_REPLY)["issues"]
        assert "Citation [1] is not verified" in issues
        assert "https://a.test" in issues

    def test_recommendations_are_separate_from_issues(self):
        parsed = format_validation_response(REAL_REPLY)
        assert "multiplexing" in parsed["recommendations"]
        assert "overly brief" not in parsed["recommendations"]

    def test_approved_reply_has_no_issues(self):
        parsed = format_validation_response(APPROVED_REPLY)
        assert parsed["status"] == "APPROVED"
        assert parsed["issues"] == "None"
        assert parsed["confidence"] == 92
