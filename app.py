"""
StudySense: AI Study Material Analyzer and Quiz Generator

A Streamlit dashboard that lets students:
- Upload PDF, DOCX, or TXT study material
- Paste study material
- View document statistics
- Analyze keyword frequency
- Generate an AI summary
- Generate an AI multiple-choice quiz
- Ask an AI study assistant questions

Run:
    streamlit run app.py
"""

import io
import os
import re
import json
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# PDF extraction
import fitz  # PyMuPDF

# DOCX extraction
import docx

# Google Gemini SDK
from google import genai
from google.genai import types


# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

APP_TITLE = "StudySense"
APP_ICON = "📚"

# Gemini model fallback list.
#
# The application tries these models in order.
# Model availability may depend on your API key, account, region, and quota.
MODEL_NAMES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# Maximum text sent to the AI.
MAX_PROMPT_CHARS = 12000

# Basic English stopword list.
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "also",
    "however",
    "thus",
    "therefore",
    "e.g",
    "i.e",
    "etc",
}


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE
# =============================================================================

DEFAULT_STATE = {
    "raw_text": "",
    "clean_text": "",
    "source_name": "",
    "stats": None,
    "keywords_df": None,
    "summary": "",
    "summary_model": "",
    "quiz": None,
    "quiz_model": "",
    "quiz_raw_error": "",
    "assistant_model": "",
    "qa_history": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# GEMINI API KEY HANDLING
# =============================================================================

def get_api_key() -> str:
    """
    Retrieve the Gemini API key from:
    1. Streamlit secrets
    2. Environment variable

    Never hardcode your API key in app.py.
    """

    key = ""

    # Streamlit Cloud secrets
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    # Local .env fallback
    if not key:
        key = os.getenv("GEMINI_API_KEY", "")

    return key.strip() if key else ""


def get_ai_client():
    """
    Return an initialized Gemini client,
    or None if no API key is configured.
    """

    api_key = get_api_key()

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        st.error(f"Could not initialize Gemini client: {exc}")
        return None


# =============================================================================
# GEMINI MODEL FALLBACK
# =============================================================================

def generate_with_model_fallback(
    client,
    prompt: str,
    json_mode: bool = False
):
    """
    Try multiple Gemini models in order.

    Returns:
        tuple[str, str]: generated content and model used

    Raises:
        RuntimeError: if all models fail
    """

    errors = []

    for model_name in MODEL_NAMES:
        try:
            config = None

            if json_mode:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json"
                )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            content = response.text if response else ""

            if content and content.strip():
                return content.strip(), model_name

            errors.append(
                f"{model_name}: Empty response returned."
            )

        except Exception as exc:
            error_text = str(exc)

            errors.append(
                f"{model_name}: {error_text}"
            )

            # A 403 permission error is project-level.
            # Trying other models will not solve it.
            if (
                "403" in error_text
                or "PERMISSION_DENIED" in error_text
                or "project has been denied access" in error_text.lower()
            ):
                raise RuntimeError(
                    "Gemini access was denied for this project.\n\n"
                    f"Google API error:\n{error_text}\n\n"
                    "This is an API project permission problem, not a "
                    "quiz or summary formatting problem. Create a new "
                    "Gemini API key from a permitted Google AI Studio "
                    "project and replace your current GEMINI_API_KEY."
                )

            # Authentication errors also affect all models.
            if (
                "401" in error_text
                or "UNAUTHENTICATED" in error_text
                or "API key not valid" in error_text.lower()
            ):
                raise RuntimeError(
                    "The Gemini API key is invalid or not authorized.\n\n"
                    f"Google API error:\n{error_text}\n\n"
                    "Create a new API key in Google AI Studio and update "
                    "GEMINI_API_KEY in your Streamlit secrets."
                )

            # Continue to the next model for model-not-found,
            # quota, rate-limit, or temporary errors.
            continue

    raise RuntimeError(
        "All Gemini models failed.\n\n"
        + "\n".join(errors)
    )


# =============================================================================
# TEXT EXTRACTION HELPERS
# =============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""

    text_parts = []

    try:
        with fitz.open(
            stream=file_bytes,
            filetype="pdf"
        ) as pdf_doc:
            for page in pdf_doc:
                text_parts.append(page.get_text())

    except Exception as exc:
        raise ValueError(
            f"Could not read PDF file: {exc}"
        )

    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""

    try:
        document = docx.Document(
            io.BytesIO(file_bytes)
        )

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        # Extract text from tables.
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        paragraphs.append(cell.text)

        return "\n".join(paragraphs)

    except Exception as exc:
        raise ValueError(
            f"Could not read DOCX file: {exc}"
        )


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Safely decode a TXT file using common encodings."""

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "latin-1"
    ):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Could not decode text file with common encodings."
    )


def extract_text(uploaded_file) -> str:
    """Dispatch extraction based on file extension."""

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)

    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)

    elif filename.endswith(".txt"):
        text = extract_text_from_txt(file_bytes)

    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF, DOCX, or TXT file."
        )

    if not text or not text.strip():
        raise ValueError(
            "No extractable text was found in this file."
        )

    return text


# =============================================================================
# TEXT CLEANING AND STATISTICS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Normalize line endings
    - Collapse repeated spaces and tabs
    - Collapse multiple blank lines
    - Preserve paragraph breaks
    """

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse repeated spaces and tabs within each line.
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    # Collapse consecutive blank lines.
    cleaned_lines = []
    blank_run = 0

    for line in lines:
        if line == "":
            blank_run += 1

            if blank_run <= 1:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def compute_stats(text: str) -> dict:
    """Compute basic document statistics."""

    words = re.findall(
        r"\b[\w'-]+\b",
        text
    )

    word_count = len(words)
    char_count = len(text)

    char_count_no_spaces = len(
        text.replace(" ", "").replace("\n", "")
    )

    paragraphs = [
        paragraph
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    paragraph_count = (
        max(len(paragraphs), 1)
        if text.strip()
        else 0
    )

    # Rough sentence count.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    sentence_count = len([
        sentence
        for sentence in sentences
        if sentence.strip()
    ])

    # Estimated reading speed: 200 words per minute.
    reading_time_minutes = (
        max(1, round(word_count / 200))
        if word_count > 0
        else 0
    )

    return {
        "word_count": word_count,
        "char_count": char_count,
        "char_count_no_spaces": char_count_no_spaces,
        "paragraph_count": paragraph_count,
        "sentence_count": sentence_count,
        "reading_time_minutes": reading_time_minutes,
    }


# =============================================================================
# KEYWORD ANALYSIS
# =============================================================================

def analyze_keywords(
    text: str,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Basic keyword-frequency analysis.

    This is not semantic topic modeling.
    It counts repeated English words.
    """

    words = re.findall(
        r"[a-zA-Z']+",
        text.lower()
    )

    filtered = [
        word
        for word in words
        if word not in STOPWORDS
        and len(word) > 3
    ]

    if not filtered:
        return pd.DataFrame(
            columns=["Keyword", "Frequency"]
        )

    counts = Counter(filtered)

    most_common = counts.most_common(top_n)

    return pd.DataFrame(
        most_common,
        columns=["Keyword", "Frequency"]
    )


# =============================================================================
# AI PROMPT HELPERS
# =============================================================================

def truncate_for_prompt(
    text: str,
    max_chars: int = MAX_PROMPT_CHARS
) -> str:
    """
    Keep prompt sizes reasonable.

    Long study material is truncated and marked clearly.
    """

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[Note: Material truncated for length.]"
    )


# =============================================================================
# AI SUMMARY
# =============================================================================

def generate_summary(
    client,
    material: str
) -> str:
    """
    Generate a structured, student-friendly summary using Gemini.
    """

    system_prompt = (
        "You are a careful study assistant. "
        "You summarize study material for students. "
        "Only use information present in the material provided. "
        "Never invent facts, names, dates, or figures that are not in the text. "
        "If a requested section cannot be filled from the material, "
        "say so explicitly instead of guessing. "
        "Use simple, clear language."
    )

    user_prompt = (
        "Summarize the following study material for a student. "
        "Structure your response with these exact headings:\n\n"
        "## Main Idea\n"
        "## Important Concepts\n"
        "## Key Definitions\n"
        "## Important Processes or Examples\n"
        "## Quick Review\n\n"
        "If a section has no relevant content in the material, write "
        "'Not covered in the provided material.' under that heading.\n\n"
        f"STUDY MATERIAL:\n{truncate_for_prompt(material)}"
    )

    full_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

USER REQUEST:

{user_prompt}
"""

    content, model_used = generate_with_model_fallback(
        client=client,
        prompt=full_prompt,
        json_mode=False
    )

    st.session_state["summary_model"] = model_used

    return content


# =============================================================================
# AI QUIZ
# =============================================================================

def generate_quiz(
    client,
    material: str,
    num_questions: int
) -> dict:
    """
    Generate a multiple-choice quiz using Gemini.

    Returns:
        Dictionary containing a questions list.
    """

    system_prompt = (
        "You are a quiz-generating assistant for students. "
        "Create multiple-choice questions strictly based on the study material. "
        "Do not include facts that are not supported by the material. "
        "Return only valid JSON. Do not include markdown code fences "
        "or extra commentary."
    )

    schema_example = {
        "questions": [
            {
                "question": "Question text",
                "choices": [
                    "A. Choice one",
                    "B. Choice two",
                    "C. Choice three",
                    "D. Choice four"
                ],
                "correct_answer": "B",
                "explanation": "Explanation text"
            }
        ]
    }

    user_prompt = (
        f"Create exactly {num_questions} multiple-choice questions "
        "based on the study material below. Each question must have "
        "exactly four choices labeled A, B, C, and D. "
        "Each question must have one correct answer represented "
        "as a single letter: A, B, C, or D. "
        "Include a short explanation of why the answer is correct.\n\n"
        "Return JSON matching this structure:\n"
        f"{json.dumps(schema_example, indent=2)}\n\n"
        f"STUDY MATERIAL:\n{truncate_for_prompt(material)}"
    )

    full_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

USER REQUEST:

{user_prompt}
"""

    content, model_used = generate_with_model_fallback(
        client=client,
        prompt=full_prompt,
        json_mode=True
    )

    st.session_state["quiz_model"] = model_used

    return parse_quiz_json(content)


def parse_quiz_json(raw_text: str) -> dict:
    """
    Parse and validate AI-generated quiz JSON.
    """

    cleaned = raw_text.strip()

    # Remove opening Markdown code fence.
    cleaned = re.sub(
        r"^```(?:json)?",
        "",
        cleaned,
        flags=re.IGNORECASE
    ).strip()

    # Remove closing Markdown code fence.
    cleaned = re.sub(
        r"```$",
        "",
        cleaned
    ).strip()

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON returned by the AI: {exc}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            "AI response did not return a JSON object."
        )

    if "questions" not in data:
        raise ValueError(
            "AI response did not contain a questions field."
        )

    if not isinstance(data["questions"], list):
        raise ValueError(
            "The questions field must be a list."
        )

    if len(data["questions"]) == 0:
        raise ValueError(
            "AI response contained no questions."
        )

    for question in data["questions"]:
        if not isinstance(question, dict):
            raise ValueError(
                "Each quiz question must be an object."
            )

        required_fields = (
            "question",
            "choices",
            "correct_answer",
            "explanation",
        )

        if not all(
            field in question
            for field in required_fields
        ):
            raise ValueError(
                "A quiz question is missing required fields."
            )

        if not isinstance(question["choices"], list):
            raise ValueError(
                "Quiz choices must be a list."
            )

        if len(question["choices"]) != 4:
            raise ValueError(
                "A quiz question must have exactly four choices."
            )

        correct_answer = str(
            question["correct_answer"]
        ).strip().upper()

        if correct_answer not in {
            "A",
            "B",
            "C",
            "D"
        }:
            raise ValueError(
                "The correct answer must be A, B, C, or D."
            )

        question["correct_answer"] = correct_answer

    return data


# =============================================================================
# AI STUDY ASSISTANT
# =============================================================================

def ask_study_assistant(
    client,
    material: str,
    question: str
) -> str:
    """
    Answer a student's question using only the provided material.
    """

    system_prompt = (
        "You are a study assistant. "
        "Answer the student's question using only the study material "
        "provided as your source of truth. "
        "If the answer is not available in the material, clearly say so "
        "instead of guessing or inventing information. "
        "Keep answers clear and educational."
    )

    user_prompt = (
        f"STUDY MATERIAL:\n{truncate_for_prompt(material)}\n\n"
        f"STUDENT QUESTION:\n{question}"
    )

    full_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

USER REQUEST:

{user_prompt}
"""

    content, model_used = generate_with_model_fallback(
        client=client,
        prompt=full_prompt,
        json_mode=False
    )

    st.session_state["assistant_model"] = model_used

    return content


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    st.sidebar.title(
        f"{APP_ICON} {APP_TITLE}"
    )

    st.sidebar.caption(
        "AI Study Material Analyzer and Quiz Generator"
    )

    st.sidebar.divider()

    # API key status.
    api_key = get_api_key()

    if api_key:
        st.sidebar.success(
            "Gemini API key detected."
        )
    else:
        st.sidebar.warning(
            "No Gemini API key found.\n\n"
            "Set GEMINI_API_KEY in .streamlit/secrets.toml "
            "or configure it as an environment variable."
        )

    st.sidebar.divider()

    # Model list.
    st.sidebar.subheader("AI Model Fallback")

    st.sidebar.caption(
        "The app tries models in this order until one works."
    )

    for index, model_name in enumerate(
        MODEL_NAMES,
        start=1
    ):
        st.sidebar.write(
            f"{index}. `{model_name}`"
        )

    st.sidebar.divider()

    st.sidebar.subheader(
        "1. Add Your Study Material"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload a file (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
    )

    pasted_text = st.sidebar.text_area(
        "...or paste study material here",
        height=180,
        placeholder=(
            "Paste your notes, textbook excerpt, "
            "or lecture transcript here..."
        ),
    )

    load_clicked = st.sidebar.button(
        "Load Material",
        type="primary",
        use_container_width=True,
    )

    if load_clicked:
        try:
            raw_text = None
            source_name = None

            if uploaded_file is not None:
                raw_text = extract_text(uploaded_file)
                source_name = uploaded_file.name

            elif pasted_text and pasted_text.strip():
                raw_text = pasted_text
                source_name = "Pasted text"

            else:
                st.sidebar.error(
                    "Please upload a file or paste some text first."
                )

            if raw_text:
                cleaned = clean_text(raw_text)

                st.session_state["raw_text"] = raw_text
                st.session_state["clean_text"] = cleaned
                st.session_state["source_name"] = source_name
                st.session_state["stats"] = compute_stats(cleaned)
                st.session_state["keywords_df"] = analyze_keywords(cleaned)

                # Reset AI results when material changes.
                st.session_state["summary"] = ""
                st.session_state["summary_model"] = ""
                st.session_state["quiz"] = None
                st.session_state["quiz_model"] = ""
                st.session_state["quiz_raw_error"] = ""
                st.session_state["assistant_model"] = ""
                st.session_state["qa_history"] = []

                st.sidebar.success(
                    f"Loaded: {source_name}"
                )

        except ValueError as exc:
            st.sidebar.error(str(exc))

        except Exception as exc:
            st.sidebar.error(
                f"Unexpected error while loading material: {exc}"
            )

    st.sidebar.divider()

    if st.session_state["clean_text"]:
        st.sidebar.caption(
            f"Current material: **{st.session_state['source_name']}**"
        )

        clear_clicked = st.sidebar.button(
            "Clear Material",
            use_container_width=True
        )

        if clear_clicked:
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value

            st.rerun()

    st.sidebar.divider()

    st.sidebar.caption(
        "⚠️ AI-generated content may contain errors. "
        "Always verify against your original study material."
    )


# =============================================================================
# OVERVIEW TAB
# =============================================================================

def render_overview():
    st.subheader("📊 Document Overview")

    stats = st.session_state["stats"]

    if not stats:
        st.info(
            "Upload a file or paste text in the sidebar, "
            "then click **Load Material** to begin."
        )
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Words",
        f"{stats['word_count']:,}"
    )

    col2.metric(
        "Characters",
        f"{stats['char_count']:,}"
    )

    col3.metric(
        "Paragraphs",
        f"{stats['paragraph_count']:,}"
    )

    col4.metric(
        "Sentences",
        f"{stats['sentence_count']:,}"
    )

    col5.metric(
        "Est. Reading Time",
        f"{stats['reading_time_minutes']} min"
    )

    with st.expander("View extracted / cleaned text"):
        st.text_area(
            "Cleaned study material",
            value=st.session_state["clean_text"],
            height=250,
            disabled=True,
        )

    st.divider()

    st.subheader(
        "🔑 Keyword Analysis (Basic Frequency Only)"
    )

    st.caption(
        "This is a simple word-frequency count, not true topic understanding. "
        "It highlights frequently repeated terms only."
    )

    keywords_df = st.session_state["keywords_df"]

    if keywords_df is None or keywords_df.empty:
        st.warning(
            "No significant keywords could be extracted from this material."
        )
        return

    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        chart_data = keywords_df.sort_values(
            "Frequency",
            ascending=True
        )

        fig = px.bar(
            chart_data,
            x="Frequency",
            y="Keyword",
            orientation="h",
            title="Top 15 Keywords by Frequency",
        )

        fig.update_layout(
            height=450,
            margin=dict(
                l=10,
                r=10,
                t=40,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with table_col:
        st.dataframe(
            keywords_df,
            use_container_width=True,
            height=450
        )

    csv_bytes = keywords_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Keyword Analysis (CSV)",
        data=csv_bytes,
        file_name="studysense_keywords.csv",
        mime="text/csv",
    )


# =============================================================================
# SUMMARY TAB
# =============================================================================

def render_summary_tab():
    st.subheader("📝 AI Summary")

    if not st.session_state["clean_text"]:
        st.info(
            "Load study material from the sidebar first."
        )
        return

    client = get_ai_client()

    generate_clicked = st.button(
        "Generate AI Summary",
        type="primary",
        key="generate_summary_button"
    )

    if generate_clicked:
        if client is None:
            st.error(
                "No Gemini API key configured. "
                "Add GEMINI_API_KEY in Streamlit secrets "
                "or environment variables."
            )

        else:
            with st.spinner(
                "Generating summary using available Gemini models..."
            ):
                try:
                    summary = generate_summary(
                        client,
                        st.session_state["clean_text"]
                    )

                    st.session_state["summary"] = summary

                except Exception as exc:
                    st.error(
                        f"Could not generate summary: {exc}"
                    )

    if st.session_state["summary"]:
        used_model = st.session_state.get(
            "summary_model",
            ""
        )

        if used_model:
            st.caption(
                f"Generated using model: `{used_model}`"
            )

        st.markdown(
            st.session_state["summary"]
        )

        st.download_button(
            "⬇️ Download Summary (TXT)",
            data=st.session_state["summary"].encode("utf-8"),
            file_name="studysense_summary.txt",
            mime="text/plain",
        )


# =============================================================================
# QUIZ TAB
# =============================================================================

def render_quiz_tab():
    st.subheader("🧠 AI Quiz Generator")

    if not st.session_state["clean_text"]:
        st.info(
            "Load study material from the sidebar first."
        )
        return

    client = get_ai_client()

    num_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=15,
        value=5,
        key="number_of_questions"
    )

    generate_clicked = st.button(
        "Generate Quiz",
        type="primary",
        key="generate_quiz_button"
    )

    if generate_clicked:
        if client is None:
            st.error(
                "No Gemini API key configured. "
                "Add GEMINI_API_KEY in Streamlit secrets "
                "or environment variables."
            )

        else:
            with st.spinner(
                "Generating quiz using available Gemini models..."
            ):
                try:
                    quiz_data = generate_quiz(
                        client,
                        st.session_state["clean_text"],
                        num_questions
                    )

                    st.session_state["quiz"] = quiz_data
                    st.session_state["quiz_raw_error"] = ""

                except ValueError as exc:
                    st.session_state["quiz"] = None
                    st.session_state["quiz_raw_error"] = str(exc)

                except Exception as exc:
                    st.session_state["quiz"] = None
                    st.session_state["quiz_raw_error"] = (
                        f"Unexpected error: {exc}"
                    )

    if st.session_state["quiz_raw_error"]:
        st.error(
            "The quiz could not be generated.\n\n"
            f"Details: {st.session_state['quiz_raw_error']}"
        )

    quiz = st.session_state["quiz"]

    if quiz:
        used_model = st.session_state.get(
            "quiz_model",
            ""
        )

        if used_model:
            st.caption(
                f"Quiz generated using model: `{used_model}`"
            )

        quiz_text_lines = []

        for i, question in enumerate(
            quiz["questions"],
            start=1
        ):
            st.markdown(
                f"**Q{i}. {question['question']}**"
            )

            st.radio(
                f"Choose an answer for Q{i}",
                options=question["choices"],
                key=f"quiz_q_{i}",
                label_visibility="collapsed",
            )

            with st.expander(
                "Show correct answer and explanation"
            ):
                st.markdown(
                    f"**Correct answer:** "
                    f"{question['correct_answer']}"
                )

                st.markdown(
                    f"**Explanation:** "
                    f"{question['explanation']}"
                )

            st.divider()

            quiz_text_lines.append(
                f"Q{i}. {question['question']}"
            )

            quiz_text_lines.extend(
                question["choices"]
            )

            quiz_text_lines.append(
                f"Correct answer: {question['correct_answer']}"
            )

            quiz_text_lines.append(
                f"Explanation: {question['explanation']}"
            )

            quiz_text_lines.append("")

        quiz_text = "\n".join(
            quiz_text_lines
        )

        st.download_button(
            "⬇️ Download Quiz (TXT)",
            data=quiz_text.encode("utf-8"),
            file_name="studysense_quiz.txt",
            mime="text/plain",
        )


# =============================================================================
# STUDY ASSISTANT TAB
# =============================================================================

def render_assistant_tab():
    st.subheader("💬 Study Assistant")

    if not st.session_state["clean_text"]:
        st.info(
            "Load study material from the sidebar first."
        )
        return

    client = get_ai_client()

    st.caption(
        "Ask questions like: "
        "**What is the main topic?**, "
        "**Explain this concept in simple terms**, "
        "**What are the important definitions?**, "
        "**Compare two concepts in the material**, or "
        "**Create a short reviewer**."
    )

    question = st.text_area(
        "Your question",
        height=100,
        key="assistant_question"
    )

    ask_clicked = st.button(
        "Ask Question",
        type="primary",
        key="ask_assistant_button"
    )

    if ask_clicked:
        if client is None:
            st.error(
                "No Gemini API key configured. "
                "Add GEMINI_API_KEY in Streamlit secrets "
                "or environment variables."
            )

        elif not question or not question.strip():
            st.warning(
                "Please enter a question first."
            )

        else:
            with st.spinner(
                "Thinking using available Gemini models..."
            ):
                try:
                    answer = ask_study_assistant(
                        client,
                        st.session_state["clean_text"],
                        question
                    )

                    st.session_state["qa_history"].append(
                        (question, answer)
                    )

                except Exception as exc:
                    st.error(
                        f"Could not get an answer: {exc}"
                    )

    if st.session_state["qa_history"]:
        st.divider()

        st.markdown(
            "### Conversation History"
        )

        used_model = st.session_state.get(
            "assistant_model",
            ""
        )

        if used_model:
            st.caption(
                f"Latest answer generated using model: `{used_model}`"
            )

        for question_text, answer in reversed(
            st.session_state["qa_history"]
        ):
            st.markdown(
                f"**You:** {question_text}"
            )

            st.markdown(
                f"**StudySense:** {answer}"
            )

            st.divider()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    render_sidebar()

    st.title(
        f"{APP_ICON} {APP_TITLE}"
    )

    st.caption(
        "AI Study Material Analyzer and Quiz Generator"
    )

    st.warning(
        "⚠️ **Educational Disclaimer:** AI-generated summaries, quizzes, "
        "and answers may contain errors. Always verify important information "
        "against your original study material. StudySense is an educational "
        "assistant and does not replace teachers, textbooks, or professional "
        "instruction.",
        icon="⚠️",
    )

    render_overview()

    st.divider()

    tab_summary, tab_quiz, tab_assistant = st.tabs(
        [
            "📝 Summary",
            "🧠 Quiz Generator",
            "💬 Study Assistant"
        ]
    )

    with tab_summary:
        render_summary_tab()

    with tab_quiz:
        render_quiz_tab()

    with tab_assistant:
        render_assistant_tab()


if __name__ == "__main__":
    main()