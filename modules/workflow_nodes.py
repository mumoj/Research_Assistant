"""LangGraph workflow nodes for research assistant."""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from .workflow_state import ResearchState, SearchResult, WebSource, YouTubeSource, ValidationResult
from .llm_config import LLMConfig, ValidatorUnavailable, message_text
from .model_resolver import is_model_unusable
from .prompts import ANSWER_GENERATION_PROMPT, FACT_CHECK_PROMPT, ANSWER_REVISION_PROMPT, format_validation_response
from . import search, scraper, citations


def _short_reason(message: str, limit: int = 140) -> str:
    """Condense a multi-line extraction failure into one readable line."""
    collapsed = " ".join(str(message).split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "..."


def _safe(fn):
    """Return the exception instead of raising, so one bad source is skipped.

    The sequential version caught per-source errors inline; a raise inside
    ThreadPoolExecutor.map would instead abort the remaining results.
    """
    def wrapper(item):
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001 - recorded per source, not swallowed
            return exc
    return wrapper


def search_web_node(state: ResearchState) -> Dict[str, Any]:
    """Search web sources."""
    if state["search_config"]["sources"] not in ["Both", "Web Only"]:
        return {"web_results": []}
    
    try:
        web_results = search.search_web(
            state["question"], 
            max_results=state["search_config"]["max_web_results"]
        )
        return {"web_results": web_results}
    except Exception as e:
        return {
            "web_results": [],
            "error_messages": state["error_messages"] + [f"Web search error: {str(e)}"]
        }


def search_youtube_node(state: ResearchState) -> Dict[str, Any]:
    """Search YouTube sources."""
    if state["search_config"]["sources"] not in ["Both", "YouTube Only"]:
        return {"youtube_results": []}
    
    try:
        youtube_results = search.search_youtube(
            state["question"],
            max_results=state["search_config"]["max_youtube_results"]
        )
        return {"youtube_results": youtube_results}
    except Exception as e:
        return {
            "youtube_results": [],
            "error_messages": state["error_messages"] + [f"YouTube search error: {str(e)}"]
        }


# Page fetches are pure network wait, so they overlap almost perfectly. Five
# sources took 7.7s one at a time and 1.5s together, with byte-identical text.
MAX_SCRAPE_WORKERS = 8


def extract_web_content_node(state: ResearchState) -> Dict[str, Any]:
    """Extract content from web pages, fetching them concurrently."""
    results = state["web_results"]
    if not results:
        return {"web_sources": []}

    def fetch(result):
        return result, scraper.extract_web_content(result.url)

    web_sources = []
    with ThreadPoolExecutor(max_workers=min(MAX_SCRAPE_WORKERS, len(results))) as pool:
        # executor.map preserves input order, so citation numbering still
        # follows search-result order rather than whichever page returned first.
        for result, outcome in zip(results, pool.map(_safe(fetch), results)):
            if isinstance(outcome, BaseException):
                state["error_messages"].append(
                    f"Error extracting {result.url}: {_short_reason(outcome)}"
                )
                continue
            _, content = outcome
            # A page that could not be fetched reports the failure in its
            # return value. Keeping it would hand the model an error notice
            # as though it were the article, and spend prompt budget on it.
            if scraper.is_extraction_error(content) or not content.strip():
                state["error_messages"].append(
                    f"Skipped {result.url}: {_short_reason(content)}"
                )
                continue
            web_sources.append(WebSource(
                title=result.title,
                url=result.url,
                content=content
            ))

    return {"web_sources": web_sources}


def extract_youtube_content_node(state: ResearchState) -> Dict[str, Any]:
    """Extract transcripts from YouTube videos."""
    youtube_sources = []
    
    for result in state["youtube_results"]:
        try:
            video_id = result.snippet  # stored in snippet field
            transcript = scraper.get_video_transcript(video_id)

            # get_video_transcript returns the failure as a string rather than
            # raising. Carrying that through made the error message itself the
            # video's "transcript", so the model was asked to answer from
            # "YouTube is blocking requests from your IP" and could cite it.
            if not isinstance(transcript, list) or not transcript:
                state["error_messages"].append(
                    f"No transcript for {result.url}: {_short_reason(transcript)}"
                )
                continue

            youtube_sources.append(YouTubeSource(
                id=video_id,
                title=result.title,
                url=result.url,
                transcript=transcript,
                transcript_text=scraper.format_transcript_text(transcript)
            ))
        except Exception as e:
            state["error_messages"].append(f"Error extracting {result.url}: {str(e)}")
    
    return {"youtube_sources": youtube_sources}


def generate_answer_node(state: ResearchState) -> Dict[str, Any]:
    """Generate primary answer using sources."""
    try:
        # Prepare sources for prompt
        all_sources = []
        for i, source in enumerate(state["web_sources"], 1):
            content = source.content[:8000] + "..." if len(source.content) > 8000 else source.content
            all_sources.append(f"SOURCE {i} (WEB): {source.url}\n{content}\n")
        
        start_idx = len(state["web_sources"]) + 1
        for i, source in enumerate(state["youtube_sources"], start_idx):
            transcript_text = source.transcript_text[:8000] + "..." if len(source.transcript_text) > 8000 else source.transcript_text
            all_sources.append(f"SOURCE {i} (YOUTUBE): {source.url}\n{transcript_text}\n")
        
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=state["question"],
            sources=chr(10).join(all_sources)
        )
        
        answer = LLMConfig.invoke_primary(prompt)

        return {"primary_answer": answer}
    
    except Exception as e:
        return {
            "primary_answer": f"Error generating answer: {str(e)}",
            "error_messages": state["error_messages"] + [f"Answer generation error: {str(e)}"]
        }


def validate_answer_node(state: ResearchState) -> Dict[str, Any]:
    """Validate the generated answer."""
    if not state["search_config"]["enable_validation"]:
        return {
            "validation_result": None,
            "final_answer": state["primary_answer"],
            "needs_revision": False
        }
    
    try:
        validator_llm = LLMConfig.get_validator_llm()
    except ValidatorUnavailable:
        # No validator key is configured. Ship the primary answer unvalidated
        # instead of failing the run with a missing-key error.
        return {
            "validation_result": None,
            "final_answer": state["primary_answer"],
            "needs_revision": False
        }

    try:
        # Prepare sources for validation
        sources_text = []
        for source in state["web_sources"]:
            sources_text.append(f"WEB: {source.title} - {source.url}")
        for source in state["youtube_sources"]:
            sources_text.append(f"YOUTUBE: {source.title} - {source.url}")
        
        validation_prompt = FACT_CHECK_PROMPT.format(
            question=state["question"],
            sources="\n".join(sources_text),
            answer=state["primary_answer"]
        )
        
        validation_text = message_text(validator_llm.invoke(validation_prompt))
        
        validation_data = format_validation_response(validation_text)
        validation_result = ValidationResult(
            status=validation_data["status"],
            issues=validation_data["issues"],
            confidence=validation_data["confidence"],
            recommendations=validation_data["recommendations"]
        )
        
        needs_revision = (
            validation_result.status == "NEEDS_REVISION" and 
            state["revision_count"] < state["max_revisions"]
        )
        
        return {
            "validation_result": validation_result,
            "needs_revision": needs_revision,
            "final_answer": state["primary_answer"] if not needs_revision else ""
        }
    
    except Exception as e:
        result = {
            "validation_result": None,
            "final_answer": state["primary_answer"],
            "needs_revision": False
        }
        # A retired validator model is a config detail, not something the reader
        # of an answer needs warned about; the answer itself is unaffected.
        if not is_model_unusable(e):
            result["error_messages"] = state["error_messages"] + [f"Validation error: {str(e)}"]
        return result


def revise_answer_node(state: ResearchState) -> Dict[str, Any]:
    """Revise answer based on validation feedback."""
    try:
        validator_llm = LLMConfig.get_validator_llm()
    except ValidatorUnavailable:
        return {
            "primary_answer": state["primary_answer"],
            "needs_revision": False
        }

    try:
        # Prepare sources for revision
        sources_text = []
        for source in state["web_sources"]:
            sources_text.append(f"WEB: {source.title} - {source.url}")
        for source in state["youtube_sources"]:
            sources_text.append(f"YOUTUBE: {source.title} - {source.url}")
        
        revision_prompt = ANSWER_REVISION_PROMPT.format(
            original_answer=state["primary_answer"],
            validation_feedback=f"Status: {state['validation_result'].status}\nIssues: {state['validation_result'].issues}\nRecommendations: {state['validation_result'].recommendations}",
            sources="\n".join(sources_text)
        )
        
        revised_answer = message_text(validator_llm.invoke(revision_prompt))

        # A model that returned no text (budget spent reasoning) would otherwise
        # replace a good answer with a blank one. Keep what we already had.
        if not revised_answer.strip():
            return {
                "primary_answer": state["primary_answer"],
                "revision_count": state["revision_count"] + 1,
                "needs_revision": False
            }

        return {
            "primary_answer": revised_answer,
            "revision_count": state["revision_count"] + 1,
            "needs_revision": False
        }
    
    except Exception as e:
        return {
            "primary_answer": state["primary_answer"],
            "needs_revision": False,
            "error_messages": state["error_messages"] + [f"Revision error: {str(e)}"]
        }


def finalize_answer_node(state: ResearchState) -> Dict[str, Any]:
    """Process final answer with citations and generate sources HTML."""
    try:
        # Convert back to dict format for citations module
        web_sources_dict = [
            {"title": source.title, "url": source.url, "content": source.content}
            for source in state["web_sources"]
        ]
        youtube_sources_dict = [
            {
                "id": source.id,
                "title": source.title, 
                "url": source.url,
                "transcript": source.transcript,
                "transcript_text": source.transcript_text
            }
            for source in state["youtube_sources"]
        ]
        
        processed_answer, _, earliest_timestamps = citations.process_citations(
            state["primary_answer"], 
            web_sources_dict, 
            youtube_sources_dict
        )
        
        sources_html = citations.create_sources_list(
            web_sources_dict, 
            youtube_sources_dict, 
            earliest_timestamps
        )
        
        return {
            "final_answer": processed_answer,
            "sources_html": sources_html
        }
    
    except Exception as e:
        return {
            "final_answer": state["primary_answer"],
            "sources_html": "",
            "error_messages": state["error_messages"] + [f"Citation processing error: {str(e)}"]
        }


def coordinate_searches_node(state: ResearchState) -> Dict[str, Any]:
    """Coordination node - waits for both searches to complete."""
    return {}  # No-op, just coordinates


def should_revise(state: ResearchState) -> str:
    """Determine if answer needs revision."""
    return "revise" if state["needs_revision"] else "finalize"


def create_workflow() -> StateGraph:
    """Create the research workflow graph."""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("search_youtube", search_youtube_node)
    workflow.add_node("coordinate", coordinate_searches_node)
    workflow.add_node("extract_web", extract_web_content_node)
    workflow.add_node("extract_youtube", extract_youtube_content_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("validate", validate_answer_node)
    workflow.add_node("revise", revise_answer_node)
    workflow.add_node("finalize", finalize_answer_node)
    
    # Add edges - parallel search then coordinate
    workflow.set_entry_point("coordinate")
    workflow.add_edge("coordinate", "search_web")
    workflow.add_edge("coordinate", "search_youtube")
    workflow.add_edge("search_web", "extract_web")
    workflow.add_edge("search_youtube", "extract_youtube")
    workflow.add_edge("extract_web", "generate_answer")
    workflow.add_edge("extract_youtube", "generate_answer")
    workflow.add_edge("generate_answer", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_revise,
        {
            "revise": "revise",
            "finalize": "finalize"
        }
    )
    workflow.add_edge("revise", "validate")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()