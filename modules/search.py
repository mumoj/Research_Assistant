import streamlit as st
import requests
import os
try:
    # duckduckgo_search was renamed to ddgs; the old package still imports but
    # silently returns zero results, which left the app with no web sources.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for older installs
    from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor
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


def _has_captions(video_id: str, api_key: str) -> bool:
    """True when the video exposes a caption track.

    Builds its own client: googleapiclient's service objects wrap a single
    httplib2 instance that is not safe to share across threads.
    """
    try:
        youtube = build('youtube', 'v3', developerKey=api_key, cache_discovery=False)
        response = youtube.captions().list(part='snippet', videoId=video_id).execute()
        return bool(response.get('items', []))
    except HttpError:
        # Caption check failed for this video; treat it as unusable rather than
        # failing the whole search.
        return False


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
        youtube = build('youtube', 'v3', developerKey=youtube_api_key,
                        cache_discovery=False)

        # Call the search.list method to retrieve matching videos
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results * 2,
            type='video'
        ).execute()

        candidates = [
            item for item in search_response.get('items', [])
            if item['id']['kind'] == 'youtube#video'
        ]

        # One captions.list round trip per video used to run strictly one after
        # another. They are checked a batch at a time instead: concurrent within
        # a batch for speed, but still stopping at the first batch that fills
        # the quota, so this costs no more API units than the serial version did
        # in the common case.
        videos: List[SearchResult] = []
        for start in range(0, len(candidates), max_results):
            batch = candidates[start:start + max_results]
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                captioned = list(pool.map(
                    lambda item: _has_captions(item['id']['videoId'], youtube_api_key),
                    batch
                ))

            # Order is preserved, so the most relevant videos still come first.
            for item, has_captions in zip(batch, captioned):
                if not has_captions:
                    continue
                video_id = item['id']['videoId']
                videos.append(SearchResult(
                    title=item['snippet']['title'],
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    snippet=video_id
                ))
                if len(videos) >= max_results:
                    return videos

        return videos
    except HttpError as e:
        st.error(f"Error searching YouTube: {str(e)}")
    return []
