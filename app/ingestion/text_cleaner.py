import re


def clean_text(text: str) -> str:
    # Replace multiple spaces with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple newlines with two newlines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()