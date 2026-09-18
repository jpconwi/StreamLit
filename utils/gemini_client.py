"""
Gemini API client and generation helpers.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.config import GEMINI_MODEL


load_dotenv()


def get_api_key() -> str:
    """
    Retrieve the Gemini API key from:

    1. Streamlit secrets
    2. Environment variables
    """

    api_key = ""

    # Streamlit Cloud or local Streamlit secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    # Local .env fallback
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    return api_key.strip() if api_key else ""


def get_ai_client():
    """
    Create and return a Gemini client.

    Returns:
        genai.Client or None
    """

    api_key = get_api_key()

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        st.error(f"Could not initialize Gemini client: {exc}")
        return None


def generate_with_gemini(
    client,
    prompt: str,
    json_mode: bool = False,
):
    """
    Generate content using the confirmed Gemini model.

    Returns:
        tuple[str, str]: generated content and model used

    Raises:
        RuntimeError: when the API request fails
    """

    config_kwargs = {
        "temperature": 0.4,
        "max_output_tokens": 4096,
    }

    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    config = types.GenerateContentConfig(**config_kwargs)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

        content = response.text if response else ""

        if not content or not content.strip():
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return content.strip(), GEMINI_MODEL

    except Exception as exc:
        error_text = str(exc)

        if (
            "403" in error_text
            or "PERMISSION_DENIED" in error_text
            or "project has been denied access" in error_text.lower()
        ):
            raise RuntimeError(
                "Gemini access was denied for this project.\n\n"
                f"Google API error:\n{error_text}\n\n"
                "Check that your GEMINI_API_KEY belongs to an active "
                "Google AI Studio project."
            ) from exc

        if (
            "401" in error_text
            or "UNAUTHENTICATED" in error_text
            or "API key not valid" in error_text.lower()
        ):
            raise RuntimeError(
                "The Gemini API key is invalid or not authorized.\n\n"
                f"Google API error:\n{error_text}\n\n"
                "Create a new API key in Google AI Studio."
            ) from exc

        if (
            "404" in error_text
            or "NOT_FOUND" in error_text
        ):
            raise RuntimeError(
                f"The Gemini model '{GEMINI_MODEL}' is unavailable "
                "for this API key.\n\n"
                f"Google API error:\n{error_text}"
            ) from exc

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
        ):
            raise RuntimeError(
                "Gemini free-tier quota or rate limit was reached. "
                "Please wait and try again later.\n\n"
                f"Google API error:\n{error_text}"
            ) from exc

        raise RuntimeError(
            f"Gemini request failed:\n{error_text}"
        ) from exc