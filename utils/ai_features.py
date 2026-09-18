"""
AI summary, quiz, and study assistant features.
"""

import json
import re

import streamlit as st

from utils.gemini_client import generate_with_gemini
from utils.text_utils import truncate_for_prompt


def generate_summary(
    client,
    material: str,
) -> str:
    """
    Generate a structured student-friendly summary.
    """

    system_prompt = (
        "You are a careful study assistant. "
        "Summarize study material for students. "
        "Only use information present in the material provided. "
        "Never invent facts, names, dates, or figures. "
        "Use simple and clear language."
    )

    user_prompt = (
        "Summarize the following study material. "
        "Use these exact headings:\n\n"
        "## Main Idea\n"
        "## Important Concepts\n"
        "## Key Definitions\n"
        "## Important Processes or Examples\n"
        "## Quick Review\n\n"
        "If a section is not covered, write: "
        "'Not covered in the provided material.'\n\n"
        f"STUDY MATERIAL:\n{truncate_for_prompt(material)}"
    )

    full_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

USER REQUEST:

{user_prompt}
"""

    content, model_used = generate_with_gemini(
        client=client,
        prompt=full_prompt,
        json_mode=False,
    )

    st.session_state["summary_model"] = model_used

    return content


def generate_quiz(
    client,
    material: str,
    num_questions: int,
) -> dict:
    """
    Generate a multiple-choice quiz.
    """

    system_prompt = (
        "You are a quiz-generating assistant for students. "
        "Create questions strictly based on the supplied study material. "
        "Do not include unsupported facts. "
        "Return only valid JSON."
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
        f"Create exactly {num_questions} multiple-choice questions "
        "based on the study material below. "
        "Each question must have exactly four choices labeled A, B, C, and D. "
        "The correct_answer must be only one letter: A, B, C, or D. "
        "Include a short explanation.\n\n"
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

    content, model_used = generate_with_gemini(
        client=client,
        prompt=full_prompt,
        json_mode=True,
    )

    st.session_state["quiz_model"] = model_used

    return parse_quiz_json(content)


def parse_quiz_json(raw_text: str) -> dict:
    """
    Parse and validate quiz JSON returned by Gemini.
    """

    cleaned = raw_text.strip()

    # Remove Markdown code fences if Gemini adds them
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON returned by the AI: {exc}"
        ) from exc

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
                "Each quiz question must have exactly four choices."
            )

        correct_answer = str(
            question["correct_answer"]
        ).strip().upper()

        if correct_answer not in {
            "A",
            "B",
            "C",
            "D",
        }:
            raise ValueError(
                "The correct answer must be A, B, C, or D."
            )

        question["correct_answer"] = correct_answer

    return data


def ask_study_assistant(
    client,
    material: str,
    question: str,
) -> str:
    """
    Answer a student's question using the supplied study material.
    """

    system_prompt = (
        "You are a study assistant. "
        "Answer using only the study material provided. "
        "If the answer is not available, clearly say so. "
        "Do not guess or invent information. "
        "Keep the answer clear and educational."
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

    content, model_used = generate_with_gemini(
        client=client,
        prompt=full_prompt,
        json_mode=False,
    )

    st.session_state["assistant_model"] = model_used

    return content