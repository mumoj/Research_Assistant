# Ask the Web & YouTube

An app that answers questions using content from both the web and YouTube videos, with proper citations.

## Setup

```
git clone https://github.com/mumoj/Research_Assistant.git
cd Research_Assistant
cp .env.example .env  # Add your API keys
docker built -t ask-qa .
docker run -p 8501:8501 ask-qa
```

## Architecture

```
User Question
    ↓
┌─────────────────┐   ┌────────────────┐
│ Web Search      │   │ YouTube Search │
│ (DuckDuckGo)    │   │ (YouTube API)  │
└────────┬────────┘   └────────┬───────┘
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌────────────────┐
│ Content         │   │ Transcript     │
│ Extraction      │   │ Extraction     │
└────────┬────────┘   └────────┬───────┘
         │                     │
         └─────────┬───────────┘
                   ↓
         ┌───────────────────┐
         │ LLM Processing    │
         │ (Answer + Cite)   │
         └────────┬──────────┘
                  ↓
         ┌───────────────────┐
         │ Streamlit UI      │
         └───────────────────┘
```

## LLM Prompt Approach
    Answer the following question based ONLY on the provided sources:
            
    QUESTION: {question}
            
    SOURCES:
    {sources}
            
    INSTRUCTIONS:
      1. Answer the question directly and concisely based only on the information in the sources.
      2. Use numbered citations in square brackets [1], [2], etc. after every statement that uses information from the sources.
      3. In the case of multiple citations for one statement, list them as [1],[2] not [1, 2].
      4. For YouTube sources, include the timestamp in the citation like [3][02:15] where 02:15 is the timestamp of the relevant information.
      5. If the sources don't contain enough information to answer the question, state this clearly.
      6. End your answer with a "SOURCES:" section that lists all the sources you cited.
      7. For YouTube sources in the SOURCES section, include the title and URL with timestamp of the earliest reference.
      8. For web sources, include the title and URL.
      9. If you use multiple timestamps from the same video, list the earliest one in the SOURCES section.

A structured prompt that explicitly instructs the LLM to cite sources with numbered references and include timestamps for YouTube content, ensuring precise attribution.

## Known Limitations

- DuckDuckGo search may be less comprehensive than commercial search APIs
- YouTube transcript availability varies across videos
- Answers are depended on the Gemini LLM and by the quality and relevance of top search results
- Large videos/pages may be truncated to fit token limits
- API rate limits may apply when handling many requests

## Third-Party Libraries and Tools Acknowledgments

This project uses several open-source libraries and tools:

#### Web Scraping and Content Extraction
- [Newspaper3k](https://newspaper.readthedocs.io/en/latest/): Article scraping and parsing
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/): HTML parsing
- [Requests](https://docs.python-requests.org/): HTTP requests
#### Search
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/): Web search functionality
- [Google API Python Client](https://github.com/googleapis/google-api-python-client): YouTube Data API integration
#### Transcription
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/): YouTube video transcript extraction
#### Machine Learning and AI
- [Google Generative AI](https://ai.google.dev/): Gemini language model for answer generation
#### Web Framework
- [Streamlit](https://streamlit.io/): Interactive web application framework
#### Development and Testing
- [Pytest](https://docs.pytest.org/): Testing framework
#### Environment and Configuration
- [Python-dotenv](https://pypi.org/project/python-dotenv/): Environment variable management
#### XML Processing
- [lxml](https://lxml.de/): XML and HTML processing

