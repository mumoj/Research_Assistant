"""State management for LangGraph research workflow."""
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Model for search results."""
    title: str
    url: str
    snippet: str = ""


class WebSource(BaseModel):
    """Model for web sources."""
    title: str
    url: str
    content: str


class YouTubeSource(BaseModel):
    """Model for YouTube sources."""
    id: str
    title: str
    url: str
    transcript: List[Dict[str, Any]]
    transcript_text: str


class ValidationResult(BaseModel):
    """Model for validation results."""
    status: str
    issues: str
    confidence: intpa
    recommendations: str


class ResearchState(TypedDict):
    """State definition for the research workflow."""
    # Input
    question: str
    search_config: Dict[str, Any]
    
    # Search results
    web_results: List[SearchResult]
    youtube_results: List[SearchResult]
    
    # Processed sources
    web_sources: List[WebSource]
    youtube_sources: List[YouTubeSource]
    
    # Generated content
    primary_answer: str
    validation_result: Optional[ValidationResult]
    final_answer: str
    sources_html: str
    
    # Control flow
    needs_revision: bool
    revision_count: int
    max_revisions: int
    
    # Metadata
    processing_time: float
    error_messages: List[str]
    debug_info: Dict[str, Any]


def create_initial_state(
    question: str,
    search_config: Optional[Dict[str, Any]] = None
) -> ResearchState:
    """Create initial state for the research workflow."""
    return ResearchState(
        question=question,
        search_config=search_config or {
            "sources": "Both",  # "Both", "Web Only", "YouTube Only"
            "max_web_results": 5,
            "max_youtube_results": 3,
            "enable_validation": True,
            "confidence_threshold": 80
        },
        web_results=[],
        youtube_results=[],
        web_sources=[],
        youtube_sources=[],
        primary_answer="",
        validation_result=None,
        final_answer="",
        sources_html="",
        needs_revision=False,
        revision_count=0,
        max_revisions=2,
        processing_time=0.0,
        error_messages=[],
        debug_info={}
    )