import streamlit as st
from duckduckgo_search import DDGS
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


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
                    
                    if len(videos) >= max_results:
                        break
            except HttpError:
                # Skip videos where caption check fails
                continue
        return videos
    except HttpError as e:
        st.error(f"Error searching YouTube: {str(e)}")
    return []

