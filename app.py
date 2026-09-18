"""
StudySense: AI Study Material Analyzer and Quiz Generator
-----------------------------------------------------------
A Streamlit dashboard that lets students upload or paste study material,
view document statistics and keyword analysis, generate an AI summary,
generate an AI multiple-choice quiz, and ask an AI study assistant
questions about the material.

Run with:
    streamlit run app.py
"""

import io
import os
import re
import json
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# PDF extraction
import fitz  # PyMuPDF

# DOCX extraction
import docx

# OpenAI client
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

APP_TITLE = "StudySense"
APP_ICON = "📚"

# Central place to change the model used for every AI feature.
MODEL_NAME = "gpt-4.1-mini"

# Basic English stopword list used for the keyword-frequency analysis.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "however", "thus", "therefore", "e.g",
    "i.e", "etc",
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
# SESSION STATE INITIALIZATION
# =============================================================================

DEFAULT_STATE = {
    "raw_text": "",           # extracted / pasted study material
    "clean_text": "",         # cleaned version of the above
    "source_name": "",        # filename or "Pasted text"
    "stats": None,            # dict of document statistics
    "keywords_df": None,      # pandas DataFrame of keyword analysis
    "summary": "",            # AI generated summary
    "quiz": None,             # parsed quiz dict
    "quiz_raw_error": "",     # raw text if quiz JSON parsing failed
    "qa_history": [],         # list of (question, answer) tuples
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# API KEY HANDLING
# =============================================================================

def get_api_key() -> str:
    """
    Retrieve the OpenAI API key from Streamlit secrets first, then from
    environment variables. Never hardcode a key in this file.
    """
    key = ""
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.getenv("OPENAI_API_KEY", "")
    return key or ""


def get_openai_client():
    """Return an initialized OpenAI client, or None if no key is configured."""
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# =============================================================================
# TEXT EXTRACTION HELPERS
# =============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    text_parts = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
            for page in pdf_doc:
                text_parts.append(page.get_text())
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {exc}")
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs]
        # Also pull text out of any tables in the document.
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        paragraphs.append(cell.text)
        return "\n".join(paragraphs)
    except Exception as exc:
        raise ValueError(f"Could not read DOCX file: {exc}")


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Safely decode a TXT file, trying a few common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue
    raise ValueError("Could not decode text file with common encodings.")


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
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

    if not text or not text.strip():
        raise ValueError("No extractable text was found in this file.")

    return text


# =============================================================================
# TEXT CLEANING & STATISTICS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - collapse repeated whitespace within lines
    - collapse 3+ blank lines into a single blank line
    - strip leading/trailing whitespace
    - keep paragraph breaks readable
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces/tabs within a line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Rejoin, collapsing 2+ consecutive blank lines into exactly one blank line
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

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned


def compute_stats(text: str) -> dict:
    """Compute basic document statistics used for the metric cards."""
    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)

    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", "").replace("\n", ""))

    paragraphs = [p for p in text.split("\n") if p.strip()]
    paragraph_count = max(len(paragraphs), 1) if text.strip() else 0

    # Rough sentence count based on sentence-ending punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentence_count = len([s for s in sentences if s.strip()])

    # Average adult reading speed ~200 words per minute.
    reading_time_minutes = max(1, round(word_count / 200)) if word_count > 0 else 0

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

def analyze_keywords(text: str, top_n: int = 15) -> pd.DataFrame:
    """
    Basic keyword-frequency analysis (NOT topic modeling).
    Lowercases text, removes stopwords and short tokens, and counts frequency.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 3]

    if not filtered:
        return pd.DataFrame(columns=["Keyword", "Frequency"])

    counts = Counter(filtered)
    most_common = counts.most_common(top_n)
    return pd.DataFrame(most_common, columns=["Keyword", "Frequency"])


# =============================================================================
# OPENAI HELPERS
# =============================================================================

def truncate_for_prompt(text: str, max_chars: int = 12000) -> str:
    """
    Keep prompt sizes reasonable. Truncates very long study material and
    notes the truncation so the AI (and the user) is aware.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Note: material truncated for length.]"


def generate_summary(client: OpenAI, material: str) -> str:
    """Call the OpenAI API to generate a structured, student-friendly summary."""
    system_prompt = (
        "You are a careful study assistant. You summarize study material for "
        "students. Only use information present in the material provided. "
        "Never invent facts, names, dates, or figures that are not in the text. "
        "If a requested section cannot be filled from the material, say so "
        "explicitly instead of guessing. Use simple, clear language."
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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("The AI returned an empty summary.")
    return content.strip()


def generate_quiz(client: OpenAI, material: str, num_questions: int) -> dict:
    """
    Call the OpenAI API to generate a multiple-choice quiz.
    Returns a parsed dict matching the required schema, or raises ValueError
    if the response cannot be parsed as valid JSON.
    """
    system_prompt = (
        "You are a quiz-generating assistant for students. You create multiple "
        "choice questions strictly based on the study material provided. "
        "Do not include facts that are not supported by the material. "
        "Respond with ONLY valid JSON and no extra commentary, markdown, or "
        "code fences."
    )

    schema_example = {
        "questions": [
            {
                "question": "Question text",
                "choices": [
                    "A. Choice one",
                    "B. Choice two",
                    "C. Choice three",
                    "D. Choice four",
                ],
                "correct_answer": "B",
                "explanation": "Explanation text",
            }
        ]
    }

    user_prompt = (
        f"Create exactly {num_questions} multiple-choice questions based on the "
        "study material below. Each question must have exactly four choices "
        "labeled A, B, C, and D, one correct answer (as a single letter), and "
        "a short explanation of why that answer is correct.\n\n"
        f"Return JSON matching this exact structure:\n{json.dumps(schema_example, indent=2)}\n\n"
        f"STUDY MATERIAL:\n{truncate_for_prompt(material)}"
    )

    kwargs = dict(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    # Ask for JSON mode if the model/SDK supports it; fall back gracefully.
    try:
        response = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except Exception:
        response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("The AI returned an empty quiz response.")

    return parse_quiz_json(content)


def parse_quiz_json(raw_text: str) -> dict:
    """
    Safely parse the AI's quiz JSON response. Strips markdown code fences if
    present and validates the basic structure before returning.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON returned by the AI: {exc}")

    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("AI response did not match the expected quiz format.")

    if not isinstance(data["questions"], list) or len(data["questions"]) == 0:
        raise ValueError("AI response contained no questions.")

    for q in data["questions"]:
        if not all(k in q for k in ("question", "choices", "correct_answer", "explanation")):
            raise ValueError("A quiz question is missing required fields.")
        if not isinstance(q["choices"], list) or len(q["choices"]) != 4:
            raise ValueError("A quiz question does not have exactly four choices.")

    return data


def ask_study_assistant(client: OpenAI, material: str, question: str) -> str:
    """Call the OpenAI API to answer a student's question about the material."""
    system_prompt = (
        "You are a study assistant. Answer the student's question using only "
        "the study material provided as your source of truth. If the answer "
        "is not available in the material, clearly say so instead of guessing "
        "or inventing information. Keep answers clear and educational."
    )

    user_prompt = (
        f"STUDY MATERIAL:\n{truncate_for_prompt(material)}\n\n"
        f"STUDENT QUESTION:\n{question}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("The AI returned an empty answer.")
    return content.strip()


# =============================================================================
# SIDEBAR: SETTINGS & UPLOAD
# =============================================================================

def render_sidebar():
    st.sidebar.title(f"{APP_ICON} {APP_TITLE}")
    st.sidebar.caption("AI Study Material Analyzer and Quiz Generator")

    st.sidebar.divider()

    # API key status
    api_key = get_api_key()
    if api_key:
        st.sidebar.success("OpenAI API key detected.")
    else:
        st.sidebar.warning(
            "No OpenAI API key found.\n\n"
            "Set it in `.streamlit/secrets.toml` as `OPENAI_API_KEY`, or as an "
            "environment variable `OPENAI_API_KEY`. AI features will not work "
            "until a key is configured."
        )

    st.sidebar.divider()
    st.sidebar.subheader("1. Add Your Study Material")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a file (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
    )

    pasted_text = st.sidebar.text_area(
        "...or paste study material here",
        height=180,
        placeholder="Paste your notes, textbook excerpt, or lecture transcript here...",
    )

    load_clicked = st.sidebar.button("Load Material", type="primary", use_container_width=True)

    if load_clicked:
        try:
            if uploaded_file is not None:
                raw_text = extract_text(uploaded_file)
                source_name = uploaded_file.name
            elif pasted_text and pasted_text.strip():
                raw_text = pasted_text
                source_name = "Pasted text"
            else:
                st.sidebar.error("Please upload a file or paste some text first.")
                raw_text = None
                source_name = None

            if raw_text:
                cleaned = clean_text(raw_text)
                st.session_state["raw_text"] = raw_text
                st.session_state["clean_text"] = cleaned
                st.session_state["source_name"] = source_name
                st.session_state["stats"] = compute_stats(cleaned)
                st.session_state["keywords_df"] = analyze_keywords(cleaned)
                # Reset downstream AI results since the material changed.
                st.session_state["summary"] = ""
                st.session_state["quiz"] = None
                st.session_state["qa_history"] = []
                st.sidebar.success(f"Loaded: {source_name}")

        except ValueError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:
            st.sidebar.error(f"Unexpected error while loading material: {exc}")

    st.sidebar.divider()
    if st.session_state["clean_text"]:
        st.sidebar.caption(f"Current material: **{st.session_state['source_name']}**")
        if st.sidebar.button("Clear Material", use_container_width=True):
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value
            st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ AI-generated content may contain errors. Always verify against "
        "your original study material."
    )


# =============================================================================
# MAIN DASHBOARD SECTIONS
# =============================================================================

def render_overview():
    st.subheader("📊 Document Overview")

    stats = st.session_state["stats"]
    if not stats:
        st.info("Upload a file or paste text in the sidebar, then click **Load Material** to begin.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Words", f"{stats['word_count']:,}")
    col2.metric("Characters", f"{stats['char_count']:,}")
    col3.metric("Paragraphs", f"{stats['paragraph_count']:,}")
    col4.metric("Sentences", f"{stats['sentence_count']:,}")
    col5.metric("Est. Reading Time", f"{stats['reading_time_minutes']} min")

    with st.expander("View extracted / cleaned text"):
        st.text_area(
            "Cleaned study material",
            value=st.session_state["clean_text"],
            height=250,
            disabled=True,
        )

    st.divider()
    st.subheader("🔑 Keyword Analysis (Basic Frequency Only)")
    st.caption(
        "This is a simple word-frequency count, not true topic understanding. "
        "It highlights frequently repeated terms only."
    )

    keywords_df = st.session_state["keywords_df"]
    if keywords_df is None or keywords_df.empty:
        st.warning("No significant keywords could be extracted from this material.")
        return

    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        fig = px.bar(
            keywords_df.sort_values("Frequency", ascending=True),
            x="Frequency",
            y="Keyword",
            orientation="h",
            title="Top 15 Keywords by Frequency",
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.dataframe(keywords_df, use_container_width=True, height=450)

    csv_bytes = keywords_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Keyword Analysis (CSV)",
        data=csv_bytes,
        file_name="studysense_keywords.csv",
        mime="text/csv",
        use_container_width=False,
    )


def render_summary_tab():
    st.subheader("📝 AI Summary")

    if not st.session_state["clean_text"]:
        st.info("Load study material from the sidebar first.")
        return

    client = get_openai_client()

    generate_clicked = st.button("Generate AI Summary", type="primary")

    if generate_clicked:
        if client is None:
            st.error("No OpenAI API key configured. Add one in Streamlit secrets or environment variables.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summary = generate_summary(client, st.session_state["clean_text"])
                    st.session_state["summary"] = summary
                except Exception as exc:
                    st.error(f"Could not generate summary: {exc}")

    if st.session_state["summary"]:
        st.markdown(st.session_state["summary"])
        st.download_button(
            "⬇️ Download Summary (TXT)",
            data=st.session_state["summary"].encode("utf-8"),
            file_name="studysense_summary.txt",
            mime="text/plain",
        )


def render_quiz_tab():
    st.subheader("🧠 AI Quiz Generator")

    if not st.session_state["clean_text"]:
        st.info("Load study material from the sidebar first.")
        return

    client = get_openai_client()

    num_questions = st.slider("Number of questions", min_value=3, max_value=15, value=5)
    generate_clicked = st.button("Generate Quiz", type="primary")

    if generate_clicked:
        if client is None:
            st.error("No OpenAI API key configured. Add one in Streamlit secrets or environment variables.")
        else:
            with st.spinner("Generating quiz..."):
                try:
                    quiz_data = generate_quiz(client, st.session_state["clean_text"], num_questions)
                    st.session_state["quiz"] = quiz_data
                    st.session_state["quiz_raw_error"] = ""
                except ValueError as exc:
                    st.session_state["quiz"] = None
                    st.session_state["quiz_raw_error"] = str(exc)
                except Exception as exc:
                    st.session_state["quiz"] = None
                    st.session_state["quiz_raw_error"] = f"Unexpected error: {exc}"

    if st.session_state["quiz_raw_error"]:
        st.error(
            "The quiz could not be generated in the expected format. "
            f"Details: {st.session_state['quiz_raw_error']}"
        )

    quiz = st.session_state["quiz"]
    if quiz:
        quiz_text_lines = []
        for i, q in enumerate(quiz["questions"], start=1):
            st.markdown(f"**Q{i}. {q['question']}**")
            st.radio(
                f"Choose an answer for Q{i}",
                options=q["choices"],
                key=f"quiz_q_{i}",
                label_visibility="collapsed",
            )
            with st.expander("Show correct answer and explanation"):
                st.markdown(f"**Correct answer:** {q['correct_answer']}")
                st.markdown(f"**Explanation:** {q['explanation']}")
            st.divider()

            quiz_text_lines.append(f"Q{i}. {q['question']}")
            quiz_text_lines.extend(q["choices"])
            quiz_text_lines.append(f"Correct answer: {q['correct_answer']}")
            quiz_text_lines.append(f"Explanation: {q['explanation']}")
            quiz_text_lines.append("")

        quiz_text = "\n".join(quiz_text_lines)
        st.download_button(
            "⬇️ Download Quiz (TXT)",
            data=quiz_text.encode("utf-8"),
            file_name="studysense_quiz.txt",
            mime="text/plain",
        )


def render_assistant_tab():
    st.subheader("💬 Study Assistant")

    if not st.session_state["clean_text"]:
        st.info("Load study material from the sidebar first.")
        return

    client = get_openai_client()

    st.caption(
        "Ask questions like: *What is the main topic?*, *Explain this concept "
        "in simple terms*, *What are the important definitions?*, *Compare two "
        "concepts in the material*, *Create a short reviewer*."
    )

    question = st.text_area("Your question", height=100, key="assistant_question")
    ask_clicked = st.button("Ask Question", type="primary")

    if ask_clicked:
        if client is None:
            st.error("No OpenAI API key configured. Add one in Streamlit secrets or environment variables.")
        elif not question or not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = ask_study_assistant(client, st.session_state["clean_text"], question)
                    st.session_state["qa_history"].append((question, answer))
                except Exception as exc:
                    st.error(f"Could not get an answer: {exc}")

    if st.session_state["qa_history"]:
        st.divider()
        st.markdown("### Conversation History")
        for q, a in reversed(st.session_state["qa_history"]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**StudySense:** {a}")
            st.divider()


# =============================================================================
# MAIN APP LAYOUT
# =============================================================================

def main():
    render_sidebar()

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("AI Study Material Analyzer and Quiz Generator")

    st.warning(
        "⚠️ **Educational Disclaimer:** AI-generated summaries, quizzes, and "
        "answers may contain errors. Always verify important information "
        "against your original study material. StudySense is an educational "
        "assistant and does not replace teachers, textbooks, or professional "
        "instruction.",
        icon="⚠️",
    )

    render_overview()

    st.divider()

    tab_summary, tab_quiz, tab_assistant = st.tabs(
        ["📝 Summary", "🧠 Quiz Generator", "💬 Study Assistant"]
    )

    with tab_summary:
        render_summary_tab()

    with tab_quiz:
        render_quiz_tab()

    with tab_assistant:
        render_assistant_tab()


if __name__ == "__main__":
    main()
