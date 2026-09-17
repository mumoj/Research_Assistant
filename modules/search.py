import streamlit as st
import requests
import os
try:
    # duckduckgo_search was renamed to ddgs; the old package still imports but
    # silently returns zero results, which left the app with no web sources.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for older installs
    from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List
from .workflow_state import SearchResult


def search_web(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search the web using DuckDuckGo with SerpAPI as a backup
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing search results
    """
    results: List[SearchResult] = []
    try:
        for result in DDGS().text(query, max_results=max_results):
            results.append(SearchResult(
                title=result.get("title", "No title"),
                url=result.get("href", ""),
                snippet=result.get("body", "No snippet")
            ))
    except Exception as e:
        st.error(f"DuckDuckGo search failed: {str(e)}")

    if results:
        return results

    # An empty result set is a failure too, not just an exception - a silently
    # empty search is what left answers with no sources at all.
    try:
        return search_with_serpapi(query, max_results)
    except Exception as e:
        st.error(f"SerpAPI search failed: {str(e)}")
        return []
    
def search_with_serpapi(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search the web using SerpAPI
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
    Returns:
        List of dictionaries containing search results
    """
    results = []
    serpapi_key = os.environ.get("SerpAPI_KEY")
    if not serpapi_key:
        st.error("SerpAPI key not found. Set the SerpAPI environment variable.")
        return results
    
    params = {
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "num": max_results
    }
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    if "organic_results" in data:
        for result in data["organic_results"][:max_results]:
            results.append({
                "title": result.get("title", "No title"),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", "No snippet")
            })
    
    return results


def search_youtube(query: str, max_results: int = 3) -> List[SearchResult]:
    """
    Searches YouTube using the YouTube Data API v3.
    Args:
        query: The search query.
        max_results: The maximum number of search results to return.
        
    Returns:
        A list of dictionaries, where each dictionary represents a video
        and contains the 'id', 'title', and 'url'. Returns an empty list if
        the YouTube API key is missing or if there are no results.
    """
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        st.error("YouTube API Key not found in environment variables")
        return []
    
    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
        # Call the search.list method to retrieve matching videos
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results * 2,
            type='video'
        ).execute()
    
        videos: List[SearchResult] = []
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
                    
                    videos.append(SearchResult(
                        title=title,
                        url=video_url,
                        snippet=video_id
                    ))
                    
                    if len(videos) >= max_results:
                        break
            except HttpError:
                # Skip videos where caption check fails
                continue
        return videos
    except HttpError as e:
        st.error(f"Error searching YouTube: {str(e)}")
    return []

