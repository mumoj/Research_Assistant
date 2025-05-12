
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import os
import json
import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from dotenv import load_dotenv
import newspaper
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Configure Gemini Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set page config
st.set_page_config(
    page_title="Ask the Web & YouTube",
    page_icon="🔍",
    layout="wide"
)

# App title
st.title("Ask the Web & YouTube")

# Sidebar configuration
with st.sidebar:
    st.header("Search Settings")
    search_sources = st.radio(
        "Include sources from:",
        options=["Both", "Web Only", "YouTube Only"],
        index=0
    )

    st.markdown("---")
    
    # Debug expandable section
    with st.expander("Debug Settings"):
        show_debug = st.checkbox("Show Debug Info", value=False)
        st.write("API Keys status:")
        st.write(f"Gemini API Key: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
        st.write(f"YouTube API Key: {'✅ Set' if os.getenv('YOUTUBE_API_KEY') else '❌ Missing'}")


# Function to search web using DuckDuckGo
def search_web(query, max_results=5):
    try:
        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
            
            for result in search_results:
                results.append({
                    "title": result.get("title", "No title"),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "No snippet")
                })
        
        return results
    except Exception as e:
        st.error(f"Error searching the web: {str(e)}")
        return []


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
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
        
        # Truncate if too long (roughly 8000 tokens)
        if len(text) > 32000:
            text = text[:32000] + "..."
            
        return text
    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}"


# Function to search YouTube
def search_youtube(query, max_results=3):
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        st.error("YouTube API Key not found in environment variables")
        return []
        
    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        
        # Call the search.list method to retrieve matching videos
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results * 3,
            type='video'
        ).execute()
        
        videos = []
        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                video_id = search_result['id']['videoId']
                
                # Check if this video has captions/transcripts
                try:
                    caption_response = youtube.captions().list(
                        part='snippet',
                        videoId=video_id
                    ).execute()
                    
                    # Only include videos that have captions
                    if caption_response.get('items', []):
                        title = search_result['snippet']['title']
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        
                        videos.append({
                            'id': video_id,
                            'title': title,
                            'url': video_url
                        })
                        
                        # Check if we've reached our desired number of results
                        if len(videos) >= max_results:
                            break
                except HttpError:
                    # Skip videos where caption check fails
                    continue
        return videos
    except HttpError as e:
        st.error(f"Error searching YouTube: {str(e)}")
        return []


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


# Function to generate an answer using Gemini
def generate_answer(question, web_sources, youtube_sources):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API Key not found in environment variables")
        return "Error: Gemini API Key not configured"
    
    try:
        # Prepare sources for the prompt
        all_sources = []
        
        # Add web sources
        for i, source in enumerate(web_sources, 1):
            content = source.get("content", "").strip()
            # Truncate long content
            if len(content) > 8000:
                content = content[:8000] + "..."
            
            all_sources.append(f"SOURCE {i} (WEB): {source['url']}\n{content}\n")
        
        # Add YouTube sources starting from where web sources left off
        start_idx = len(web_sources) + 1
        for i, source in enumerate(youtube_sources, start_idx):
            transcript_text = source.get("transcript_text", "").strip()
            if transcript_text and len(transcript_text) > 8000:
                transcript_text = transcript_text[:8000] + "..."
            
            all_sources.append(f"SOURCE {i} (YOUTUBE): {source['url']}\n{transcript_text}\n")
        
        # Create the prompt
        prompt = f"""
        Answer the following question based ONLY on the provided sources:
        
        QUESTION: {question}
        
        SOURCES:
        {'\n'.join(all_sources)}
        
        INSTRUCTIONS:
        1. Answer the question directly and concisely based only on the information in the sources.
        2. Use numbered citations in square brackets [1], [2], etc. after every statement that uses information from the sources.
        3. For YouTube sources, include the timestamp in the citation like [3][02:15] where 02:15 is the timestamp of the relevant information.
        4. If the sources don't contain enough information to answer the question, state this clearly.
        5. End your answer with a "SOURCES:" section that lists all the sources you cited.
        6. For YouTube sources in the SOURCES section, include the title and URL with timestamp of the earliest reference.
        7. For web sources, include the title and URL.
        8. If you use multiple timestamps from the same video, list the earliest one in the SOURCES section.
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Generate response from Gemini
        response = model.generate_content(
            [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1500,
                top_p=0.95,
                top_k=40
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Function to process the answer to make citations clickable
def process_citations(answer_text, web_sources, youtube_sources):
    # Extract the main answer and sources sections
    if "SOURCES:" in answer_text:
        main_answer = answer_text.split("SOURCES:")[0].strip()
        sources_section = "SOURCES:" + answer_text.split("SOURCES:")[1]
    else:
        main_answer = answer_text
        sources_section = ""
    
    # Find all citations in the main answer [n] or [n][timestamp]
    citation_pattern = r'\[(\d+)(?:\]\[([0-9:]+))?\]'
    
    # Create a map of the earliest timestamp for each YouTube source
    earliest_timestamps = {}
    
    def replace_citation(match):
        source_num = int(match.group(1))
        timestamp = match.group(2)
        
        # Determine if it's a web or YouTube source
        is_youtube = False
        url = ""
        
        if source_num <= len(web_sources):
            # Web source
            url = web_sources[source_num - 1]["url"]
            link_text = match.group(0)
        else:
            # YouTube source
            youtube_index = source_num - len(web_sources) - 1
            if youtube_index < len(youtube_sources):
                is_youtube = True
                video_id = youtube_sources[youtube_index]["id"]
                url = youtube_sources[youtube_index]["url"]
                
                # Add timestamp if provided
                if timestamp:
                    # Convert timestamp to seconds
                    time_parts = timestamp.split(':')
                    if len(time_parts) == 2:
                        minutes, seconds = int(time_parts[0]), int(time_parts[1])
                        total_seconds = minutes * 60 + seconds
                        url += f"&t={total_seconds}s"
                        
                        # Store the earliest timestamp for this source
                        if source_num not in earliest_timestamps or total_seconds < earliest_timestamps[source_num]["seconds"]:
                            earliest_timestamps[source_num] = {
                                "timestamp": timestamp,
                                "seconds": total_seconds
                            }
                    
                link_text = match.group(0)
        
        # Return the citation as a clickable link
        return f'<a href="{url}" target="_blank">{link_text}</a>'
    
    # Replace citations with clickable links in the main answer
    processed_answer = re.sub(citation_pattern, replace_citation, main_answer)
    
    # Return the processed answer and the sources section
    return processed_answer, sources_section, earliest_timestamps


# Function to create the sources list
def create_sources_list(web_sources, youtube_sources, earliest_timestamps):
    sources_html = "<h3>Sources</h3><ol>"
    
    # Add web sources
    for i, source in enumerate(web_sources, 1):
        sources_html += f'<li><a href="{source["url"]}" target="_blank">{source["title"]}</a></li>'
    
    # Add YouTube sources
    start_idx = len(web_sources) + 1
    for i, source in enumerate(youtube_sources, start_idx):
        url = source["url"]
        
        # Add timestamp if we have one
        if i in earliest_timestamps:
            timestamp = earliest_timestamps[i]["timestamp"]
            seconds = earliest_timestamps[i]["seconds"]
            url = f"{source['url']}&t={seconds}s"
            timestamp_text = f" (starts at {timestamp})"
        else:
            timestamp_text = ""
            
        sources_html += f'<li><a href="{url}" target="_blank">{source["title"]}{timestamp_text}</a></li>'
    
    sources_html += "</ol>"
    return sources_html


# Main input form
question = st.text_input("Ask a question:")
ask_button = st.button("Ask")

# Process the question when the button is clicked
if ask_button and question:
    # Start the search process
    with st.spinner("Searching for information..."):
        # Initialize results containers
        web_results = []
        web_sources = []
        youtube_results = []
        youtube_sources = []
        
        # Search based on selected sources
        if search_sources in ["Both", "Web Only"]:
            web_results = search_web(question, max_results=5)
            
            # Extract content from web pages
            for result in web_results:
                content = extract_web_content(result["url"])
                web_sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "content": content
                })
        
        if search_sources in ["Both", "YouTube Only"]:
            youtube_results = search_youtube(question, max_results=3)
            
            # Get transcripts for YouTube videos
            for video in youtube_results:
                transcript = get_video_transcript(video["id"])
                transcript_text = ""
                
                if isinstance(transcript, list):
                    transcript_text = format_transcript_text(transcript)
                else:
                    transcript_text = transcript  # Error message
                
                youtube_sources.append({
                    "id": video["id"],
                    "title": video["title"],
                    "url": video["url"],
                    "transcript": transcript,
                    "transcript_text": transcript_text
                })
        
        # Collect all sources for debugging
        all_results = {
            "web_results": web_results,
            "youtube_results": youtube_results
        }
        
        # Generate the answer
        answer = generate_answer(question, web_sources, youtube_sources)
        
        # Process citations
        processed_answer, sources_section, earliest_timestamps = process_citations(answer, web_sources, youtube_sources)
        
        # Create the sources list
        sources_html = create_sources_list(web_sources, youtube_sources, earliest_timestamps)
    
    # Display the answer
    st.markdown("<h3>Answer</h3>", unsafe_allow_html=True)
    st.markdown(processed_answer, unsafe_allow_html=True)
    
    # Display sources
    st.markdown(sources_html, unsafe_allow_html=True)
    
    # Debug panel
    if show_debug:
        with st.expander("Debug Information"):
            st.json(all_results)
            
            # Show number of sources
            st.write(f"Web sources: {len(web_sources)}")
            st.write(f"YouTube sources: {len(youtube_sources)}")
            
            # Show prompt used
            st.markdown("### Prompt Template")
            st.code("""
            Answer the following question based ONLY on the provided sources:
            
            QUESTION: {question}
            
            SOURCES:
            {sources}
            
            INSTRUCTIONS:
            1. Answer the question directly and concisely based only on the information in the sources.
            2. Use numbered citations in square brackets [1], [2], etc. after every statement that uses information from the sources.
            3. For YouTube sources, include the timestamp in the citation like [3][02:15] where 02:15 is the timestamp of the relevant information.
            4. If the sources don't contain enough information to answer the question, state this clearly.
            5. End your answer with a "SOURCES:" section that lists all the sources you cited.
            6. For YouTube sources in the SOURCES section, include the title and URL with timestamp of the earliest reference.
            7. For web sources, include the title and URL.
            8. If you use multiple timestamps from the same video, list the earliest one in the SOURCES section.
            """)