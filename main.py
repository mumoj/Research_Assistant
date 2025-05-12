
import streamlit as st
import os
from dotenv import load_dotenv
from modules import citations, llm, scraper, search


# Load environment variables
load_dotenv()

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
            web_results = search.search_web(question, max_results=5)
            
            # Extract content from web pages
            for result in web_results:
                content = scraper.extract_web_content(result["url"])
                web_sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "content": content
                })
        
        if search_sources in ["Both", "YouTube Only"]:
            youtube_results = search.search_youtube(question, max_results=3)
            
            # Get transcripts for YouTube videos
            for video in youtube_results:
                transcript = scraper.get_video_transcript(video["id"])
                transcript_text = ""
                
                if isinstance(transcript, list):
                    transcript_text = scraper.format_transcript_text(transcript)
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
        answer = llm.generate_answer(question, web_sources, youtube_sources)
        
        # Process citations
        processed_answer, sources_section, earliest_timestamps = citations.process_citations(answer, web_sources, youtube_sources)
        
        # Create the sources list
        sources_html = citations.create_sources_list(web_sources, youtube_sources, earliest_timestamps)
    
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