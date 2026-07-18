import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def normalize_text(text):
    """
    Convert text to lowercase and remove punctuation/special characters.
    """

    text = str(text).lower()

    # Keep letters, digits and whitespace.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Replace repeated whitespace with one space.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Normalize text, split it into tokens and remove stopwords.
    """

    cleaned_text = normalize_text(text)

    tokens = cleaned_text.split()

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def clean_text(text):
    """
    Return cleaned tokens joined back into a searchable string.
    """

    return " ".join(tokenize(text))