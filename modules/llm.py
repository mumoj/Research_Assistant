import streamlit as st
import google.generativeai as genai
import os

#Function to generate an answer using Gemini
def generate_answer(question, web_sources, youtube_sources):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API Key not found in environment variables")
        return "Error: Gemini API Key not configured"
    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    try:
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
    
        
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
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

