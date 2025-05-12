import newspaper
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

    
# Function to extract content from web pages
def extract_web_content(url):
    try:
        # Use newspaper3k to extract article content
        article = newspaper.Article(url)
        article.download()
        article.parse()
    
        # Get the main text
        text = article.text
    
        # If text is empty or very short, try BeautifulSoup as fallback
        if not text or len(text) < 100:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator='\n', strip=True)
    
            # Truncate if too long (roughly 8000 tokens)
        if len(text) > 32000:
            text = text[:32000] + "..."
        return text
    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}"

# Function to get YouTube video transcript
def get_video_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Process transcript to include timestamps
        formatted_transcript = []
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


# Function to format transcript into readable text
def format_transcript_text(transcript):
    if isinstance(transcript, str):  # Error message
        return transcript

    formatted_text = ""
    for entry in transcript:
        formatted_text += f"[{entry['timestamp']}] {entry['text']}\n"
    return formatted_text
    