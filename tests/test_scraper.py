from unittest.mock import MagicMock, patch

import requests

import modules.scraper as scraper


class TestExtractionErrors:
    def test_is_extraction_error_identifies_error_prefix(self):
        """Only scraper error notices are classified as extraction errors."""
        assert scraper.is_extraction_error(
            f"{scraper.WEB_EXTRACTION_ERROR_PREFIX} https://example.com: failed"
        )
        assert not scraper.is_extraction_error("normal text")
        assert not scraper.is_extraction_error("")
        assert not scraper.is_extraction_error(None)
        assert not scraper.is_extraction_error(["text"])


class TestWebExtraction:
    def test_newspaper_text_is_returned_without_fallback(self):
        """Long newspaper text avoids the requests fallback."""
        article = MagicMock(text="x" * 100)
        with patch.object(scraper.newspaper, "Article", return_value=article) as make_article, \
                patch.object(scraper.requests, "get") as get:
            result = scraper.extract_web_content("https://example.com/article")

        assert result == "x" * 100
        make_article.assert_called_once_with(
            "https://example.com/article", config=scraper._SCRAPE_CONFIG
        )
        get.assert_not_called()

    def test_short_newspaper_text_uses_cleaned_requests_fallback(self):
        """Short article text falls back to visible HTML content."""
        article = MagicMock(text="short")
        response = MagicMock(
            text="<html><style>.hidden { display: none; }</style>"
            "<script>drop this</script><body>Visible text</body></html>"
        )
        with patch.object(scraper.newspaper, "Article", return_value=article), \
                patch.object(scraper.requests, "get", return_value=response) as get:
            result = scraper.extract_web_content("https://example.com/fallback")

        assert result == "Visible text"
        get.assert_called_once()
        assert get.call_args.kwargs["timeout"] == 10
        assert "User-Agent" in get.call_args.kwargs["headers"]
        response.raise_for_status.assert_called_once_with()

    def test_long_text_is_truncated(self):
        """Extracted text is capped at 32000 characters plus an ellipsis."""
        article = MagicMock(text="a" * 32001)
        with patch.object(scraper.newspaper, "Article", return_value=article):
            result = scraper.extract_web_content("https://example.com/long")

        assert result == "a" * 32000 + "..."
        assert len(result) == 32003

    def test_article_download_failure_returns_error_notice(self):
        """Newspaper failures become URL-bearing extraction error notices."""
        article = MagicMock()
        article.download.side_effect = RuntimeError("download failed")
        url = "https://example.com/broken"
        with patch.object(scraper.newspaper, "Article", return_value=article):
            result = scraper.extract_web_content(url)

        assert scraper.is_extraction_error(result)
        assert url in result
        assert "download failed" in result

    def test_http_failure_returns_error_notice_after_status_check(self):
        """HTTP fallback failures are reported after raise_for_status is called."""
        article = MagicMock(text="short")
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("HTTP 503")
        url = "https://example.com/unavailable"
        with patch.object(scraper.newspaper, "Article", return_value=article), \
                patch.object(scraper.requests, "get", return_value=response):
            result = scraper.extract_web_content(url)

        assert scraper.is_extraction_error(result)
        assert url in result
        assert "HTTP 503" in result
        response.raise_for_status.assert_called_once_with()


class TestRawTranscript:
    def test_legacy_api_returns_get_transcript_result(self):
        """The legacy static API result is returned unchanged."""
        expected = [{"text": "hello", "start": 0, "duration": 1}]

        class LegacyApi:
            @staticmethod
            def get_transcript(video_id):
                assert video_id == "legacy-id"
                return expected

        with patch.object(scraper, "YouTubeTranscriptApi", LegacyApi):
            assert scraper._fetch_raw_transcript("legacy-id") == expected

    def test_new_api_uses_to_raw_data(self):
        """The new API's raw conversion method is preferred when available."""
        expected = [{"text": "hello", "start": 1.2, "duration": 2}]

        class Fetched:
            def to_raw_data(self):
                return expected

        class NewApi:
            def fetch(self, video_id):
                assert video_id == "new-id"
                return Fetched()

        with patch.object(scraper, "YouTubeTranscriptApi", NewApi):
            assert scraper._fetch_raw_transcript("new-id") == expected

    def test_new_api_converts_iterable_snippets(self):
        """Iterable snippet objects are converted into raw dictionaries."""
        class Snippet:
            def __init__(self, text, start, duration):
                self.text = text
                self.start = start
                self.duration = duration

        class NewApi:
            def fetch(self, video_id):
                assert video_id == "iterable-id"
                return [Snippet("one", 0.5, 1.0), Snippet("two", 2, 1.5)]

        with patch.object(scraper, "YouTubeTranscriptApi", NewApi):
            assert scraper._fetch_raw_transcript("iterable-id") == [
                {"text": "one", "start": 0.5, "duration": 1.0},
                {"text": "two", "start": 2, "duration": 1.5},
            ]


class TestVideoTranscript:
    def test_transcript_adds_expected_timestamps_and_keys(self):
        """Transcript entries use minute and second timestamps."""
        raw = [
            {"text": "zero", "start": 0, "duration": 1},
            {"text": "sixty-five", "start": 65.7, "duration": 1},
            {"text": "ten-minutes", "start": 600, "duration": 1},
        ]
        with patch.object(scraper, "_fetch_raw_transcript", return_value=raw):
            result = scraper.get_video_transcript("video-id")

        assert [entry["timestamp"] for entry in result] == [
            "00:00", "01:05", "10:00"
        ]
        assert all(
            set(entry) == {"text", "start", "timestamp", "timestamp_seconds"}
            for entry in result
        )
        assert result[1]["timestamp_seconds"] == 65.7

    def test_transcript_fetch_exception_returns_error_string(self):
        """Transcript fetch errors are returned with the expected prefix."""
        with patch.object(
            scraper, "_fetch_raw_transcript", side_effect=RuntimeError("blocked")
        ):
            result = scraper.get_video_transcript("video-id")

        assert result.startswith("Error getting transcript:")


class TestTranscriptFormatting:
    def test_list_is_formatted_one_entry_per_line(self):
        """Transcript entries are rendered with their timestamps."""
        transcript = [
            {"timestamp": "00:00", "text": "first"},
            {"timestamp": "01:05", "text": "second"},
        ]
        assert scraper.format_transcript_text(transcript) == (
            "[00:00] first\n[01:05] second\n"
        )

    def test_empty_list_is_empty_and_strings_pass_through(self):
        """Empty transcripts and existing error strings remain unchanged."""
        assert scraper.format_transcript_text([]) == ""
        error = "Error getting transcript: blocked"
        assert scraper.format_transcript_text(error) == error
