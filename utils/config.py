# utils/config.py

APP_TITLE = "StudySense"
APP_ICON = "📚"

# Primary Gemini model
GEMINI_MODEL = "gemini-3.6-flash"

# Keep this alias if other files use PRIMARY_MODEL
PRIMARY_MODEL = GEMINI_MODEL

# Backup models (2.5-flash / 2.5-flash-lite are deprecated and 404 for new
# users as of mid-2026 — Google's own API error names these as the successors)
BACKUP_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

MAX_PROMPT_CHARS = 12000

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from",
    "are", "was", "were", "been", "have", "has", "had",
    "will", "would", "could", "should", "about", "into",
    "their", "there", "these", "those", "then", "than",
    "they", "them", "you", "your", "our", "not", "but",
    "can", "may", "also", "such", "its", "his", "her",
    "who", "what", "when", "where", "which", "how",
    "a", "an", "in", "on", "at", "to", "of", "is", "it",
    "as", "by", "or", "be", "we", "he", "she", "i"
}