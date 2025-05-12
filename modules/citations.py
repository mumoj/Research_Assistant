import re
from typing import List, Dict, Tuple, Optional, Union

def process_citations(answer_text: str, web_sources: List[Dict[str, str]], youtube_sources: List[Dict[str, str]]) -> Tuple[str, str, Dict[int, Dict[str, Union[str, int]]]]:
    """
    Processes the answer text to make citations clickable links and separates the answer from the sources section.

    Args:
        answer_text: The answer text containing citations.
        web_sources: A list of dictionaries containing information about the web sources.
        youtube_sources: A list of dictionaries containing information about the YouTube sources.

    Returns:
        A tuple containing:
        - The processed answer text with clickable links.
        - The sources section of the answer text.
        - A dictionary mapping source numbers to their earliest timestamps (if available).
    """
    if "SOURCES:" in answer_text:
        main_answer: str = answer_text.split("SOURCES:")[0].strip()
        sources_section: str = "SOURCES:" + answer_text.split("SOURCES:")[1]
    else:
        main_answer: str = answer_text
        sources_section:str = ""

    # Find all citations in the main answer [n] or [n][timestamp]
    citation_pattern:re.Pattern[str] = r'\[(\d+)(?:\]\[([0-9:]+))?\]'

    # Create a map of the earliest timestamp for each YouTube source
    earliest_timestamps: Dict[int, Dict[str, Union[str, int]]] = {}

    def replace_citation(match: re.Match[str]) -> str:
        """
        Replaces a citation in the answer text with a clickable HTML link.
        Args:
            match: A regular expression match object containing the citation.
        Returns:
            An HTML anchor tag representing the clickable link.
        """
        source_num: int = int(match.group(1))
        timestamp: Optional[str] = match.group(2)
    
        # Determine if it's a web or YouTube source
        is_youtube: bool = False
        url: str = ""
    
        if source_num <= len(web_sources):
            # Web source
            url = web_sources[source_num - 1]["url"]
            link_text: str = match.group(0)
        else:
            # YouTube source
            youtube_index: int = source_num - len(web_sources) - 1
            if youtube_index < len(youtube_sources):
                is_youtube = True
                video_id: str = youtube_sources[youtube_index]["id"]
                url = youtube_sources[youtube_index]["url"]
            
                # Add timestamp if provided
                if timestamp:
                    # Convert timestamp to seconds
                    time_parts: List[str] = timestamp.split(':')
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
    
        return f'<a href="{url}" target="_blank">{link_text}</a>'
    
    processed_answer: str = re.sub(citation_pattern, replace_citation, main_answer)
    return processed_answer, sources_section, earliest_timestamps


def create_sources_list(web_sources: List[Dict[str, str]], youtube_sources: List[Dict[str, str]], earliest_timestamps: Dict[int, Dict[str, Union[str, int]]]) -> str:
    """
    Creates an HTML list of the sources used to generate the answer.
    Args:
        web_sources: A list of dictionaries containing information about the web sources.
        youtube_sources: A list of dictionaries containing information about the YouTube sources.
        earliest_timestamps: A dictionary mapping source numbers to their earliest timestamps.

    Returns:
        An HTML string representing an ordered list of the sources.
    """
    sources_html: str = "<h3>Sources</h3><ol>"

    for i, source in enumerate(web_sources, 1):
        sources_html += f'<li><a href="{source["url"]}" target="_blank">{source["title"]}</a></li>'

    
    start_idx = len(web_sources) + 1
    for i, source in enumerate(youtube_sources, start_idx):
        url: str = source["url"]
    
        # Add timestamp if we have one
        if i in earliest_timestamps:
            timestamp: str = earliest_timestamps[i]["timestamp"]
            seconds: int = earliest_timestamps[i]["seconds"]
            url = f"{source['url']}&t={seconds}s"
            timestamp_text:str = f" (starts at {timestamp})"
        else:
            timestamp_text: str = ""
        
        sources_html += f'<li><a href="{url}" target="_blank">{source["title"]}{timestamp_text}</a></li>'

    sources_html += "</ol>"
    return sources_html