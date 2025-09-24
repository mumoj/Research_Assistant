
import streamlit as st
import os
from dotenv import load_dotenv
from modules.workflow_nodes import create_workflow
from modules.workflow_state import create_initial_state
from modules.llm_config import LLMConfig
from modules import citations
from typing import Dict, Any
import time


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
    search_sources: str = st.radio(
        "Include sources from:",
        options=["Both", "Web Only", "YouTube Only"],
        index=0
    )

    st.markdown("---")
    
    # Debug expandable section
    with st.expander("Debug Settings"):
        show_debug: bool = st.checkbox("Show Debug Info", value=False)
        st.write("API Keys statuses:")
        config_status = LLMConfig.validate_config()
        for service, status in config_status.items():
            st.write(f"{service.title()}: {'✅ Set' if status else '❌ Missing'}")
        

# Main input form
question: str = st.text_input("Ask a question:")
ask_button: bool = st.button("Ask")

# Process the question when the button is clicked
if ask_button and question:
    start_time = time.time()
    
    with st.spinner("Researching your question..."):
        # Create search configuration
        search_config = {
            "sources": search_sources,
            "max_web_results": 5,
            "max_youtube_results": 3,
            "enable_validation": True,
            "confidence_threshold": 80
        }
        
        # Create initial state and workflow
        initial_state = create_initial_state(question, search_config)
        workflow = create_workflow()
        
        # Run the workflow
        try:
            final_state = workflow.invoke(initial_state)
            processing_time = time.time() - start_time
            

            processed_answer = final_state.get("final_answer", "No answer generated")
            sources_html = final_state.get("sources_html", "")
            validation_result = final_state.get("validation_result")
            error_messages = final_state.get("error_messages", [])
            
            if error_messages:
                for error in error_messages:
                    st.warning(f"⚠️ {error}")
            
        except Exception as e:
            st.error(f"Workflow error: {str(e)}")
            processed_answer = "An error occurred while processing your question."
            sources_html = ""
            validation_result = None
            processing_time = time.time() - start_time
    
    # Display the answer
    st.markdown("<h3>Answer</h3>", unsafe_allow_html=True)
    st.markdown(processed_answer, unsafe_allow_html=True)
    
    # Display validation info if available
    if validation_result:
        if validation_result.status == "APPROVED":
            st.success(f"✅ Validated (Confidence: {validation_result.confidence}%)")
        elif validation_result.status == "NEEDS_REVISION":
            st.warning(f"⚠️ Answer was revised based on validation feedback")
            if show_debug:
                st.info(f"Issues found: {validation_result.issues}")
    
    # Display sources
    st.markdown(sources_html, unsafe_allow_html=True)
    
    # Show processing time
    st.caption(f"⏱️ Processed in {processing_time:.2f} seconds")
    
    # Debug panel
    if show_debug:
        with st.expander("Debug Information"):
            # Show workflow state
            debug_info = {
                "web_results_count": len(final_state.get("web_sources", [])),
                "youtube_results_count": len(final_state.get("youtube_sources", [])),
                "validation_enabled": search_config.get("enable_validation", False),
                "processing_time": f"{processing_time:.2f}s"
            }
            if validation_result:
                debug_info["validation_status"] = validation_result.status
                debug_info["validation_confidence"] = validation_result.confidence
            
            st.json(debug_info)
            
            # Show API key status
            config_status = LLMConfig.validate_config()
            st.write("**API Configuration:**")
            for service, status in config_status.items():
                st.write(f"- {service.title()}: {'✅' if status else '❌'}")
            
            
            st.markdown("### Prompt Template")
            st.code("""
        Answer the following question based ONLY on the provided sources:
    
        QUESTION: {question}
    
        SOURCES:
        {'\n'.join(all_sources)}
    
        INSTRUCTIONS:
        1. Answer the question directly and concisely based only on the information in the sources.
        2. Use numbered citations in square brackets [1], [2], etc. after every statement that uses information from the sources.
        3. In the case of multiple citations for one statement, list them one after another like [1],[2] not [1, 2].
        4. For YouTube sources, include the timestamp in the citation like [3][02:15] where 02:15 is the timestamp of the relevant information.
        5. If the sources don't contain enough information to answer the question, state this clearly.
        6. End your answer with a "SOURCES:" section that lists all the sources you cited.
        7. For YouTube sources in the SOURCES section, include the title and URL with timestamp of the earliest reference.
        8. For web sources, include the title and URL.
        9. If you use multiple timestamps from the same video, list the earliest one in the SOURCES section.
            """)