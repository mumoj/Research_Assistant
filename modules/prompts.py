"""All prompt templates for the research workflow."""
from langchain.prompts import ChatPromptTemplate


ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
Answer the following question based ONLY on the provided sources:

QUESTION: {question}

SOURCES:
{sources}

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

FACT_CHECK_PROMPT = ChatPromptTemplate.from_template("""
You are a fact-checking assistant. Your job is to verify the accuracy of an \
AI-generated answer.

ORIGINAL QUESTION: {question}

SOURCES PROVIDED: {sources}

AI GENERATED ANSWER: {answer}

VALIDATION TASKS:
1. Check if all claims in the answer are supported by the provided sources
2. Verify that citations [1], [2], etc. actually correspond to relevant \
information in the sources
3. Look for any potential hallucinations or unsupported statements
4. Check if YouTube timestamps (if any) are reasonable given the context
5. Ensure the answer directly addresses the original question

RESPONSE FORMAT:
VALIDATION_STATUS: [APPROVED/NEEDS_REVISION/REJECTED]
ISSUES_FOUND: [List specific problems, or "None" if approved]
CONFIDENCE_SCORE: [0-100]
RECOMMENDED_CHANGES: [Specific suggestions, or "None" if approved]

Be thorough but concise in your analysis.
""")

ANSWER_REVISION_PROMPT = ChatPromptTemplate.from_template("""
Based on the validation feedback, revise the original answer to fix any issues.

ORIGINAL ANSWER: {original_answer}

VALIDATION FEEDBACK: {validation_feedback}

SOURCES: {sources}

INSTRUCTIONS:
1. Address all issues mentioned in the validation feedback
2. Maintain the same citation format [1], [2], etc.
3. Include YouTube timestamps where appropriate [3][02:15]
4. Keep the answer concise and directly relevant to the question
5. End with a "SOURCES:" section listing all cited sources

Provide only the revised answer, maintaining the original format and style.
""")

QUALITY_CHECK_PROMPT = ChatPromptTemplate.from_template("""
Perform a final quality check on this research answer.

QUESTION: {question}
ANSWER: {answer}
SOURCES COUNT: Web: {web_count}, YouTube: {youtube_count}

Check for:
1. Answer completeness and relevance
2. Proper citation format
3. Source utilization
4. Overall coherence

Rate the answer quality (1-10) and provide brief feedback.

FORMAT:
QUALITY_SCORE: [1-10]
FEEDBACK: [Brief assessment]
""")

def format_validation_response(response_text: str) -> dict:
    """Parse structured validation response into dictionary."""
    import re
    
    patterns = {
        "status": r"VALIDATION_STATUS:\s*\[([^\]]+)\]",
        "issues": r"ISSUES_FOUND:\s*\[([^\]]+)\]",
        "confidence": r"CONFIDENCE_SCORE:\s*\[(\d+)\]",
        "recommendations": r"RECOMMENDED_CHANGES:\s*\[([^\]]+)\]"
    }
    
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = "Unknown"
    
    # Convert confidence to integer
    try:
        result["confidence"] = int(result.get("confidence", "0"))
    except ValueError:
        result["confidence"] = 0
    
    return result