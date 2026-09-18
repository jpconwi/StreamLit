"""
Streamlit user interface components.
"""

import streamlit as st
import plotly.express as px

from utils.config import APP_TITLE, APP_ICON, GEMINI_MODEL
from utils.file_utils import extract_text
from utils.text_utils import (
    clean_text,
    compute_stats,
    analyze_keywords,
)
from utils.gemini_client import (
    get_api_key,
    get_ai_client,
)
from utils.ai_features import (
    generate_summary,
    generate_quiz,
    ask_study_assistant,
)


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


def initialize_session_state():
    """
    Initialize Streamlit session state.
    """

    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_material_state():
    """
    Clear all loaded material and generated results.
    """

    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def render_sidebar():
    """
    Render the application sidebar.
    """

    st.sidebar.title(
        f"{APP_ICON} {APP_TITLE}"
    )

    st.sidebar.caption(
        "AI Study Material Analyzer and Quiz Generator"
    )

    st.sidebar.divider()

    # API key status
    api_key = get_api_key()

    if api_key:
        st.sidebar.success(
            "Explore Your Study Materials"
        )
    else:
        st.sidebar.warning(
            "No Gemini API key found.\n\n"
            "Set GEMINI_API_KEY in .streamlit/secrets.toml "
            "or in your .env file."
        )

    # st.sidebar.divider()

    # # st.sidebar.subheader("AI Model")

    # # st.sidebar.code(
    # #     GEMINI_MODEL,
    # #     language="text",
    # # )

    # # st.sidebar.caption(
    # #     "This is the confirmed Gemini model used by the app."
    # # )

    # st.sidebar.divider()

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
                    "Please upload a file or paste study material first."
                )

            if raw_text:
                cleaned = clean_text(raw_text)

                st.session_state["raw_text"] = raw_text
                st.session_state["clean_text"] = cleaned
                st.session_state["source_name"] = source_name
                st.session_state["stats"] = compute_stats(cleaned)
                st.session_state["keywords_df"] = analyze_keywords(cleaned)

                # Reset AI results
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
            f"Current material: "
            f"**{st.session_state['source_name']}**"
        )

        clear_clicked = st.sidebar.button(
            "Clear Material",
            use_container_width=True,
        )

        if clear_clicked:
            reset_material_state()
            st.rerun()

    st.sidebar.divider()

    st.sidebar.caption(
        "⚠️ AI-generated content may contain errors. "
        "Always verify against your original study material."
    )


def render_overview():
    """
    Render document statistics and keyword analysis.
    """

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
        f"{stats['word_count']:,}",
    )

    col2.metric(
        "Characters",
        f"{stats['char_count']:,}",
    )

    col3.metric(
        "Paragraphs",
        f"{stats['paragraph_count']:,}",
    )

    col4.metric(
        "Sentences",
        f"{stats['sentence_count']:,}",
    )

    col5.metric(
        "Est. Reading Time",
        f"{stats['reading_time_minutes']} min",
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
            ascending=True,
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
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with table_col:
        st.dataframe(
            keywords_df,
            use_container_width=True,
            height=450,
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


def render_summary_tab():
    """
    Render the AI summary tab.
    """

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
        key="generate_summary_button",
    )

    if generate_clicked:
        if client is None:
            st.error(
                "No Gemini API key configured. "
                "Add GEMINI_API_KEY in Streamlit secrets or .env."
            )

        else:
            with st.spinner(
                f"Generating summary using {GEMINI_MODEL}..."
            ):
                try:
                    summary = generate_summary(
                        client,
                        st.session_state["clean_text"],
                    )

                    st.session_state["summary"] = summary

                except Exception as exc:
                    st.error(
                        f"Could not generate summary: {exc}"
                    )

    if st.session_state["summary"]:
        used_model = st.session_state.get(
            "summary_model",
            "",
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


def render_quiz_tab():
    """
    Render the AI quiz generator tab.
    """

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
        key="number_of_questions",
    )

    generate_clicked = st.button(
        "Generate Quiz",
        type="primary",
        key="generate_quiz_button",
    )

    if generate_clicked:
        if client is None:
            st.error(
                "No Gemini API key configured. "
                "Add GEMINI_API_KEY in Streamlit secrets or .env."
            )

        else:
            with st.spinner(
                f"Generating quiz using {GEMINI_MODEL}..."
            ):
                try:
                    quiz_data = generate_quiz(
                        client,
                        st.session_state["clean_text"],
                        num_questions,
                    )

                    st.session_state["quiz"] = quiz_data
                    st.session_state["quiz_raw_error"] = ""

                except ValueError as exc:
                    st.session_state["quiz"] = None
                    st.session_state["quiz_raw_error"] = str(exc)

                except Exception as exc:
                    st.session_state["quiz"] = None
                    st.session_state["quiz_raw_error"] = str(exc)

    if st.session_state["quiz_raw_error"]:
        st.error(
            "The quiz could not be generated.\n\n"
            f"Details: {st.session_state['quiz_raw_error']}"
        )

    quiz = st.session_state["quiz"]

    if quiz:
        used_model = st.session_state.get(
            "quiz_model",
            "",
        )

        if used_model:
            st.caption(
                f"Quiz generated using model: `{used_model}`"
            )

        quiz_text_lines = []

        for i, question in enumerate(
            quiz["questions"],
            start=1,
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

        quiz_text = "\n".join(quiz_text_lines)

        st.download_button(
            "⬇️ Download Quiz (TXT)",
            data=quiz_text.encode("utf-8"),
            file_name="studysense_quiz.txt",
            mime="text/plain",
        )


def render_assistant_tab():
    """
    Render the AI study assistant tab.
    """

    st.subheader("💬 Study Assistant")

    if not st.session_state["clean_text"]:
        st.info(
            "Load study material from the sidebar first."
        )
        return

    client = get_ai_client()

    st.caption(
        "Ask questions such as: "
        "**What is the main topic?**, "
        "**Explain this concept in simple terms**, "
        "**What are the important definitions?**, or "
        "**Create a short reviewer**."
    )

    question = st.text_area(
        "Your question",
        height=100,
        key="assistant_question",
    )

    ask_clicked = st.button(
        "Ask Question",
        type="primary",
        key="ask_assistant_button",
    )

    if ask_clicked:
        if client is None:
            st.error(
                "No Gemini API key configured. "
                "Add GEMINI_API_KEY in Streamlit secrets or .env."
            )

        elif not question or not question.strip():
            st.warning(
                "Please enter a question first."
            )

        else:
            with st.spinner(
                f"Thinking using {GEMINI_MODEL}..."
            ):
                try:
                    answer = ask_study_assistant(
                        client,
                        st.session_state["clean_text"],
                        question,
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

        st.markdown("### Conversation History")

        used_model = st.session_state.get(
            "assistant_model",
            "",
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