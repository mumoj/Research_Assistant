import pytest
import sys
import os
import re
from unittest.mock import patch, MagicMock
from modules.citations import process_citations
from modules.scraper import get_video_transcript, format_transcript_text

# Add the parent directory to sys.path to import the app
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTranscriptExtraction:
    
    def test_format_transcript_text(self):
        # Test with a valid transcript list
        transcript = [
            {"text": "Hello world", "start": 10.5, "timestamp": "00:10", "timestamp_seconds": 10.5},
            {"text": "This is a test", "start": 15.2, "timestamp": "00:15", "timestamp_seconds": 15.2}
        ]
        result = format_transcript_text(transcript)
        assert "[00:10] Hello world" in result
        assert "[00:15] This is a test" in result
        
        # Test with an error string
        error_msg = "Error getting transcript: Video unavailable"
        assert format_transcript_text(error_msg) == error_msg
    
    @patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript')
    def test_get_video_transcript(self, mock_get_transcript):
        # Mock the transcript API response
        mock_get_transcript.return_value = [
            {'text': 'Hello world', 'start': 10.5, 'duration': 2.0},
            {'text': 'This is a test', 'start': 15.2, 'duration': 1.8}
        ]
        
        result = get_video_transcript('test_video_id')
        
        # Check result format
        assert len(result) == 2
        assert result[0]['text'] == 'Hello world'
        assert result[0]['timestamp'] == '00:10'
        assert result[1]['text'] == 'This is a test'
        assert result[1]['timestamp'] == '00:15'
        
    def test_process_citations(self):
        test_answer = "This is a fact from a web source [1]. This is from YouTube [2][01:45]. Another fact [1].\n\nSOURCES:\n1. Web Source Title\n2. YouTube Video Title"
        
        web_sources = [{"title": "Web Article", "url": "https://example.com"}]
        youtube_sources = [{"id": "abcd1234", "title": "YouTube Video", "url": "https://youtube.com/watch?v=abcd1234"}]
        
        processed_answer, sources_section, timestamps = process_citations(test_answer, web_sources, youtube_sources)
        
        # Check that web citation is properly linked
        assert 'href="https://example.com"' in processed_answer
        
        # Check that YouTube citation includes timestamp
        assert 'href="https://youtube.com/watch?v=abcd1234&t=105s"' in processed_answer
        
        # Check timestamps dictionary
        assert 2 in timestamps
        assert timestamps[2]["timestamp"] == "01:45"
        assert timestamps[2]["seconds"] == 105

class TestCitationParsing:
    
    def test_youtube_timestamp_parsing(self):
        # Test case for extracting timestamps from citations
        test_text = "This has a timestamp [3][01:30] in it."
        pattern = r'\[(\d+)(?:\]\[([0-9:]+))?\]'
        
        matches = re.findall(pattern, test_text)
        assert len(matches) == 1
        
        source_num, timestamp = matches[0]
        assert source_num == '3'
        assert timestamp == '01:30'
        
        # Convert timestamp to seconds
        minutes, seconds = map(int, timestamp.split(':'))
        total_seconds = minutes * 60 + seconds
        assert total_seconds == 90
        
