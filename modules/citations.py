import re

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

    for i, source in enumerate(web_sources, 1):
        sources_html += f'<li><a href="{source["url"]}" target="_blank">{source["title"]}</a></li>'

    
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