# utils/gemini_client.py

import json
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


# Your confirmed working model
PRIMARY_MODEL = "gemini-3.6-flash"

# Backup models (2.5-flash / 2.5-flash-lite are deprecated and 404 for new
# users as of mid-2026 — Google's own API error names these as the successors)
BACKUP_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

MAX_PROMPT_CHARS = 12000


def get_api_key():
    """
    Get the Gemini API key from Streamlit secrets or .env.
    """

    api_key = ""

    # Try Streamlit Cloud secrets first
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    # If no Streamlit secret exists, use .env
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    return api_key.strip()


@st.cache_resource
def create_client(api_key):
    """
    Create and cache the Gemini client.
    """

    return genai.Client(api_key=api_key)


def get_ai_client():
    """
    Return the Gemini client.
    """

    api_key = get_api_key()

    if not api_key:
        return None

    return create_client(api_key)


def truncate_text(text, max_chars=MAX_PROMPT_CHARS):
    """
    Limit the amount of text sent to Gemini.
    """

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n[Text truncated.]"


def is_temporary_error(error):
    """
    Check if the Gemini error may be temporary.
    """

    error_text = str(error).upper()

    temporary_errors = [
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "500",
        "INTERNAL",
        "TIMEOUT",
        "DEADLINE_EXCEEDED",
    ]

    return any(
        error_code in error_text
        for error_code in temporary_errors
    )


def generate_with_retry(
    client,
    prompt,
    json_mode=False,
    max_attempts_per_model=2,
):
    """
    Generate a Gemini response.

    This function:
    - Tries the primary model.
    - Retries temporary errors.
    - Tries backup models if needed.
    """

    if client is None:
        raise RuntimeError(
            "Gemini API key is missing. "
            "Please configure GEMINI_API_KEY."
        )

    prompt = truncate_text(prompt)

    model_list = [PRIMARY_MODEL] + BACKUP_MODELS

    # Remove duplicate model names
    model_list = list(dict.fromkeys(model_list))

    errors = []

    for model_name in model_list:

        for attempt in range(max_attempts_per_model):

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

                response_text = getattr(response, "text", None)

                if not response_text:
                    raise RuntimeError(
                        f"Gemini returned an empty response "
                        f"using {model_name}."
                    )

                return response_text.strip(), model_name

            except Exception as error:

                errors.append(
                    f"{model_name}, attempt {attempt + 1}: {error}"
                )

                # Stop retrying permanent errors for this model
                if not is_temporary_error(error):
                    break

                # Wait before retrying temporary errors
                if attempt < max_attempts_per_model - 1:
                    time.sleep(2)

        # Short delay before trying another model
        time.sleep(1)

    error_details = "\n".join(errors)

    raise RuntimeError(
        "Gemini request failed after trying the available models.\n\n"
        + error_details
    )



# utils.ai_features imports this exact name — generate_with_retry is the
# real implementation, this is just the public alias it expects.
generate_with_gemini = generate_with_retry


def generate_summary(client, material):
    """
    Generate a summary from study material.
    """

    material = truncate_text(material)

    prompt = f"""
You are an academic study assistant.

Summarize the following study material clearly and accurately.

Requirements:
- Identify the main topic.
- Explain important concepts.
- Define important terms.
- Include the key points.
- Use headings and bullet points.
- Do not invent information.
- Base the response only on the provided material.

STUDY MATERIAL:
{material}
"""

    return generate_with_retry(
        client=client,
        prompt=prompt,
        json_mode=False,
    )


def generate_quiz(client, material, number_of_questions=5):
    """
    Generate multiple-choice questions.
    """

    material = truncate_text(material)

    prompt = f"""
You are an academic quiz generator.

Create exactly {number_of_questions} multiple-choice questions
based only on the study material below.

Return valid JSON using this exact structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "choices": [
        "Choice A",
        "Choice B",
        "Choice C",
        "Choice D"
      ],
      "answer": "Correct choice exactly as written",
      "explanation": "Short explanation"
    }}
  ]
}}

Rules:
- Create exactly {number_of_questions} questions.
- Each question must have four choices.
- The answer must exactly match one of the choices.
- Do not include Markdown code fences.
- Do not include extra text outside the JSON.
- Use only information from the study material.

STUDY MATERIAL:
{material}
"""

    raw_response, model_name = generate_with_retry(
        client=client,
        prompt=prompt,
        json_mode=True,
    )

    quiz_data = parse_quiz_json(raw_response)

    return quiz_data, model_name


def parse_quiz_json(raw_response):
    """
    Parse and validate Gemini's quiz JSON.
    """

    cleaned = raw_response.strip()

    # Remove Markdown code fences if Gemini includes them
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
        flags=re.IGNORECASE,
    )

    try:
        quiz_data = json.loads(cleaned)

    except json.JSONDecodeError:

        # Extract JSON from surrounding text
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "Gemini did not return valid quiz JSON."
            )

        try:
            quiz_data = json.loads(
                cleaned[start:end + 1]
            )

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Could not parse quiz JSON: {error}"
            )

    if not isinstance(quiz_data, dict):
        raise ValueError(
            "Quiz response must be a JSON object."
        )

    questions = quiz_data.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "Quiz JSON does not contain a valid questions list."
        )

    valid_questions = []

    for item in questions:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get("question", "")
        ).strip()

        choices = item.get("choices", [])

        answer = str(
            item.get("answer", "")
        ).strip()

        explanation = str(
            item.get("explanation", "")
        ).strip()

        if not question:
            continue

        if not isinstance(choices, list):
            continue

        choices = [
            str(choice).strip()
            for choice in choices
            if str(choice).strip()
        ]

        if len(choices) < 2:
            continue

        # Match the answer without case sensitivity
        if answer not in choices:

            matching_choice = next(
                (
                    choice
                    for choice in choices
                    if choice.lower() == answer.lower()
                ),
                None,
            )

            if matching_choice:
                answer = matching_choice
            else:
                continue

        valid_questions.append(
            {
                "question": question,
                "choices": choices[:4],
                "answer": answer,
                "explanation": explanation,
            }
        )

    if not valid_questions:
        raise ValueError(
            "No valid questions were found."
        )

    return {
        "questions": valid_questions
    }


def ask_study_assistant(
    client,
    material,
    user_question,
):
    """
    Answer a question based on the study material.
    """

    material = truncate_text(material)

    prompt = f"""
You are a helpful study assistant.

Answer the student's question using the provided study material.

Rules:
- Base the answer on the material.
- If the answer is not found in the material, say that clearly.
- Explain in simple language.
- Do not invent information.

STUDY MATERIAL:
{material}

STUDENT QUESTION:
{user_question}
"""

    return generate_with_retry(
        client=client,
        prompt=prompt,
        json_mode=False,
    )