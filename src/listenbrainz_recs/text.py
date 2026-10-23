"""Artist-name normalization shared by index building and the web app."""

import unicodedata


def normalize_name(value: str | None) -> str:
    """Normalize punctuation, accents, and case for searchable tokens."""
    value = unicodedata.normalize("NFKD", value or "")
    chars = []
    for ch in value.casefold():
        category = unicodedata.category(ch)
        if category.startswith("M"):
            continue
        chars.append(ch if category[0] in ("L", "N") else " ")
    return " ".join("".join(chars).split())
