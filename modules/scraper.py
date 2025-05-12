import newspaper
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Union

    
def extract_web_content(url: str) -> str:
    """
    Extracts the main content from a web page.
    Args:
        url: The URL of the web page.
    Returns:
        The extracted text content, or an error message if extraction fails.
    """
    try:
        # Use newspaper3k to extract article content
        article = newspaper.Article(url)
        article.download()
        article.parse()
    
        # Get the main text
        text: str = article.text
    
        # If text is empty or very short, try BeautifulSoup as fallback
        if not text or len(text) < 100:
            headers: Dict[str, str] = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response: requests.Response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        
            soup: BeautifulSoup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator='\n', strip=True)
    
            # Truncate if too long (roughly 8000 tokens)
        if len(text) > 32000:
            text = text[:32000] + "..."
        return text
    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}"

def get_video_transcript(video_id: str) -> Union[List[Dict[str, str]], str]:
    """
    Retrieves the transcript of a YouTube video.
    Args:
        video_id: The ID of the YouTube video.
    Returns:
        A list of dictionaries representing the transcript, where each dictionary
        contains 'text' and 'start' and 'duration', or an error message if
        the transcript cannot be fetched.
    """
    try:
        transcript_list: YouTubeTranscriptApi = YouTubeTranscriptApi.get_transcript(video_id)

        # Process transcript to include timestamps
        formatted_transcript: List[Dict[str, str]] = []
        for entry in transcript_list:
            start_time = entry['start']
            text = entry['text']
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_entry = {
                'text': text,
                'start': start_time,
                'timestamp': timestamp,
                'timestamp_seconds': start_time
            }
            formatted_transcript.append(formatted_entry)
    
        return formatted_transcript
    except Exception as e:
        return f"Error getting transcript: {str(e)}"


def format_transcript_text(transcript: List[Dict[str, str]]) -> str:
    """
    Formats a YouTube transcript into a single string.]
    Args:
        transcript: A list of transcript segments, where each segment is a
                    dictionary containing atleast 'text', and 'timestamp'.
    Returns:
        A string containing the concatenated text of the transcript segments.
    """
    if isinstance(transcript, str):  # Error message
        return transcript

    formatted_text = ""
    for entry in transcript:
        formatted_text += f"[{entry['timestamp']}] {entry['text']}\n"
    return formatted_text
    