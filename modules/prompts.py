"""All prompt templates for the research workflow."""
from langchain_core.prompts import ChatPromptTemplate


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

# The RESPONSE FORMAT above writes each field as ``LABEL: [placeholder]``, where
# the brackets mark a slot to fill rather than punctuation to reproduce. Models
# read it that way and answer ``VALIDATION_STATUS: NEEDS_REVISION``, so a parser
# that required literal brackets matched nothing and every field came back
# "Unknown" -- which silently made validation a no-op. Brackets are optional now.
_FIELD_LABELS = ("VALIDATION_STATUS", "ISSUES_FOUND",
                 "CONFIDENCE_SCORE", "RECOMMENDED_CHANGES")

_STATUSES = ("NEEDS_REVISION", "APPROVED", "REJECTED")


def _field_pattern(label: str) -> str:
    """Match one labelled field up to the next label or the end of the text.

    Stopping at the next label rather than at the first ``]`` matters: the
    issues a validator reports routinely quote citations like [1], which would
    otherwise cut the value short.
    """
    others = "|".join(other for other in _FIELD_LABELS if other != label)
    return (
        rf"\*{{0,2}}{label}\*{{0,2}}\s*:\s*\**"
        rf"(.*?)"
        rf"(?=\n\s*\*{{0,2}}(?:{others})\*{{0,2}}\s*:|\Z)"
    )


def _unwrap(value: str) -> str:
    """Strip markdown emphasis and one enclosing pair of brackets."""
    value = value.strip().strip("*").strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    return value


def format_validation_response(response_text: str) -> dict:
    """Parse structured validation response into dictionary."""
    import re

    result = {}
    for key, label in zip(
        ("status", "issues", "confidence", "recommendations"), _FIELD_LABELS
    ):
        match = re.search(_field_pattern(label), response_text or "",
                          re.IGNORECASE | re.DOTALL)
        value = _unwrap(match.group(1)) if match else ""
        result[key] = value or "Unknown"

    # The status drives the revision loop, so reduce it to one of the three
    # keywords wherever it appears -- "NEEDS_REVISION (see below)" still counts.
    status_text = result["status"].upper()
    result["status"] = next(
        (name for name in _STATUSES if name in status_text), "Unknown"
    )

    # Confidence may arrive as "35", "[35]", "35/100" or "35 - fairly sure".
    confidence = re.search(r"\d+", result.get("confidence", ""))
    result["confidence"] = int(confidence.group()) if confidence else 0

    return result
