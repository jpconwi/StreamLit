"""
StudySense: AI Study Material Analyzer and Quiz Generator

Run:
    streamlit run app.py
"""

import streamlit as st


from utils.config import APP_TITLE, APP_ICON
from utils.ui import (
    initialize_session_state,
    render_sidebar,
    render_overview,
    render_summary_tab,
    render_quiz_tab,
    render_assistant_tab,
)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    render_sidebar()

    st.title(f"{APP_ICON} {APP_TITLE}")

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
            "💬 Study Assistant",
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