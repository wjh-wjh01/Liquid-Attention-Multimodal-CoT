from __future__ import annotations

import re
import string


ARTICLES = {"a", "an", "the"}


def normalize_answer(value: str | int | float) -> str:
    text = str(value).lower().strip()
    text = _strip_punctuation(text)
    tokens = [tok for tok in text.split() if tok not in ARTICLES]
    return " ".join(tokens)


def choice_to_index(answer: str | int, choices: list[str]) -> int:
    if isinstance(answer, int):
        return answer
    raw = str(answer).strip()
    if len(raw) == 1 and raw.upper() in string.ascii_uppercase:
        idx = string.ascii_uppercase.index(raw.upper())
        if idx < len(choices):
            return idx
    normalized = normalize_answer(raw)
    for idx, choice in enumerate(choices):
        if normalize_answer(choice) == normalized:
            return idx
    raise ValueError(f"Could not map answer {answer!r} to choices {choices!r}")


def exact_match(prediction: str, target: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(target)


def _strip_punctuation(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text.translate(str.maketrans("", "", string.punctuation))
