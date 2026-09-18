"""
Text cleaning, statistics, and keyword analysis utilities.
"""

import re
from collections import Counter

import pandas as pd

from utils.config import STOPWORDS, MAX_PROMPT_CHARS


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

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Correct regex: spaces and tabs
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

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
    """
    Compute basic document statistics.
    """

    words = re.findall(
        r"\b[\w'-]+\b",
        text,
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

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    sentence_count = len(
        [
            sentence
            for sentence in sentences
            if sentence.strip()
        ]
    )

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


def analyze_keywords(
    text: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Count frequently used English words.
    """

    words = re.findall(
        r"[a-zA-Z']+",
        text.lower(),
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
        columns=["Keyword", "Frequency"],
    )


def truncate_for_prompt(
    text: str,
    max_chars: int = MAX_PROMPT_CHARS,
) -> str:
    """
    Limit the amount of text sent to Gemini.
    """

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[Note: Material truncated for length.]"
    )