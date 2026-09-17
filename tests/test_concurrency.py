"""Concurrency must speed things up without reordering or dropping sources."""
import threading
import time
from unittest.mock import MagicMock, patch

import modules.search as search
from modules.workflow_nodes import (
    extract_web_content_node, extract_youtube_content_node)
from modules.workflow_state import SearchResult


def state(urls):
    return {
        "web_results": [
            SearchResult(title=f"T{i}", url=u, snippet="") for i, u in enumerate(urls)
        ],
        "error_messages": [],
    }


class TestThreadedWebExtraction:

    def test_sources_keep_search_result_order(self):
        """Citation numbering depends on this: [1] must be the first result."""
        urls = [f"https://e{i}.test" for i in range(5)]
        # Reverse the completion order: the last URL returns first.
        def slow(url):
            time.sleep(0.05 * (len(urls) - int(url[9])))
            return f"content of {url}"

        with patch.object(search, "search_web"), \
             patch("modules.scraper.extract_web_content", side_effect=slow):
            result = extract_web_content_node(state(urls))

        assert [s.url for s in result["web_sources"]] == urls
        assert [s.content for s in result["web_sources"]] == [f"content of {u}" for u in urls]

    def test_pages_are_fetched_concurrently(self):
        urls = [f"https://e{i}.test" for i in range(5)]
        with patch("modules.scraper.extract_web_content",
                   side_effect=lambda url: (time.sleep(0.2), "x")[1]):
            started = time.perf_counter()
            result = extract_web_content_node(state(urls))
            elapsed = time.perf_counter() - started

        assert len(result["web_sources"]) == 5
        # Serially this is 1.0s; overlapped it is barely more than one fetch.
        assert elapsed < 0.6, f"fetches did not overlap ({elapsed:.2f}s)"

    def test_one_failing_page_does_not_lose_the_others(self):
        urls = ["https://ok1.test", "https://bad.test", "https://ok2.test"]

        def flaky(url):
            if "bad" in url:
                raise RuntimeError("connection reset")
            return "fine"

        s = state(urls)
        with patch("modules.scraper.extract_web_content", side_effect=flaky):
            result = extract_web_content_node(s)

        assert [x.url for x in result["web_sources"]] == ["https://ok1.test", "https://ok2.test"]
        assert any("bad.test" in m and "connection reset" in m for m in s["error_messages"])

    def test_no_results_needs_no_pool(self):
        assert extract_web_content_node(state([]))["web_sources"] == []


def youtube_api(video_ids):
    """A fake YouTube service whose search returns the given ids in order."""
    service = MagicMock()
    service.search.return_value.list.return_value.execute.return_value = {
        "items": [
            {"id": {"kind": "youtube#video", "videoId": vid},
             "snippet": {"title": f"title {vid}"}}
            for vid in video_ids
        ]
    }
    return service


class TestYouTubeCaptionChecks:

    def test_results_keep_relevance_order_and_stop_at_max(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        with patch.object(search, "build", return_value=youtube_api(list("abcdef"))), \
             patch.object(search, "_has_captions", return_value=True):
            results = search.search_youtube("q", max_results=3)

        assert [r.snippet for r in results] == ["a", "b", "c"]
        assert results[0].url == "https://www.youtube.com/watch?v=a"

    def test_videos_without_captions_are_skipped(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        with patch.object(search, "build", return_value=youtube_api(list("abcdef"))), \
             patch.object(search, "_has_captions", side_effect=lambda v, k: v in "cdf"):
            results = search.search_youtube("q", max_results=3)

        assert [r.snippet for r in results] == ["c", "d", "f"]

    def test_first_batch_costs_no_more_quota_than_the_serial_version(self, monkeypatch):
        """When the top results all have captions, we make exactly max_results calls."""
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        checked = []
        lock = threading.Lock()

        def record(video_id, api_key):
            with lock:
                checked.append(video_id)
            return True

        with patch.object(search, "build", return_value=youtube_api(list("abcdef"))), \
             patch.object(search, "_has_captions", side_effect=record):
            search.search_youtube("q", max_results=3)

        # Concurrent, so completion order is not fixed; the cost is what matters.
        assert sorted(checked) == ["a", "b", "c"]

    def test_caption_checks_run_concurrently(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")

        def slow(video_id, api_key):
            time.sleep(0.2)
            return True

        with patch.object(search, "build", return_value=youtube_api(list("abcdef"))), \
             patch.object(search, "_has_captions", side_effect=slow):
            started = time.perf_counter()
            search.search_youtube("q", max_results=3)
            elapsed = time.perf_counter() - started

        # Serially the first batch is 0.6s.
        assert elapsed < 0.4, f"caption checks did not overlap ({elapsed:.2f}s)"

    def test_missing_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        with patch.object(search, "st") as ui:
            assert search.search_youtube("q") == []
        assert ui.error.called


class TestFailedSourcesAreDropped:
    """A failed fetch must not reach the model dressed up as source material.

    Both extractors report failure in their return value instead of raising,
    so the notice used to be stored as the source's content and handed to the
    LLM as SOURCE n -- spending prompt budget on an error and letting the
    model cite it.
    """

    def test_unfetchable_page_is_skipped_not_passed_through(self):
        urls = ["https://ok.test", "https://dead.test"]
        notice = "Error extracting content from https://dead.test: timed out"

        def extract(url):
            return notice if "dead" in url else "real article text"

        s = state(urls)
        with patch("modules.scraper.extract_web_content", side_effect=extract):
            result = extract_web_content_node(s)

        assert [x.url for x in result["web_sources"]] == ["https://ok.test"]
        assert not any(notice in x.content for x in result["web_sources"])
        assert any("dead.test" in m for m in s["error_messages"])

    def test_blank_page_is_skipped(self):
        s = state(["https://empty.test"])
        with patch("modules.scraper.extract_web_content", return_value="   "):
            result = extract_web_content_node(s)
        assert result["web_sources"] == []
        assert s["error_messages"]

    def test_good_pages_survive_a_failing_neighbour(self):
        urls = ["https://a.test", "https://bad.test", "https://b.test"]

        def extract(url):
            if "bad" in url:
                return "Error extracting content from https://bad.test: 403"
            return "content"

        s = state(urls)
        with patch("modules.scraper.extract_web_content", side_effect=extract):
            result = extract_web_content_node(s)
        assert [x.url for x in result["web_sources"]] == ["https://a.test", "https://b.test"]


class TestTranscriptFailuresAreDropped:

    # Verbatim from the live API when YouTube IP-blocks transcript fetches.
    BLOCKED = ("Error getting transcript: \nCould not retrieve a transcript for "
               "the video https://www.youtube.com/watch?v=x! This is most likely "
               "caused by:\n\nYouTube is blocking requests from your IP.")

    def yt_state(self, ids):
        return {
            "youtube_results": [
                SearchResult(title=f"V{v}", url=f"https://youtu.be/{v}", snippet=v)
                for v in ids
            ],
            "error_messages": [],
        }

    def test_ip_blocked_transcript_does_not_become_the_source(self):
        s = self.yt_state(["vid1"])
        with patch("modules.scraper.get_video_transcript", return_value=self.BLOCKED):
            result = extract_youtube_content_node(s)

        assert result["youtube_sources"] == []
        assert any("No transcript" in m and "vid1" in m for m in s["error_messages"])
        # The warning is one readable line, not the whole multi-line dump.
        assert "\n" not in s["error_messages"][0]

    def test_empty_transcript_is_skipped(self):
        s = self.yt_state(["vid1"])
        with patch("modules.scraper.get_video_transcript", return_value=[]):
            result = extract_youtube_content_node(s)
        assert result["youtube_sources"] == []

    def test_usable_transcript_still_kept(self):
        segs = [{"text": "hello", "start": 1.0, "timestamp": "00:01",
                 "timestamp_seconds": 1.0}]
        s = self.yt_state(["vid1"])
        with patch("modules.scraper.get_video_transcript", return_value=segs):
            result = extract_youtube_content_node(s)
        assert len(result["youtube_sources"]) == 1
        assert "[00:01] hello" in result["youtube_sources"][0].transcript_text
        assert s["error_messages"] == []

    def test_one_bad_video_does_not_drop_a_good_one(self):
        segs = [{"text": "ok", "start": 0.0, "timestamp": "00:00",
                 "timestamp_seconds": 0.0}]
        s = self.yt_state(["bad", "good"])
        with patch("modules.scraper.get_video_transcript",
                   side_effect=lambda v: self.BLOCKED if v == "bad" else segs):
            result = extract_youtube_content_node(s)
        assert [x.id for x in result["youtube_sources"]] == ["good"]
