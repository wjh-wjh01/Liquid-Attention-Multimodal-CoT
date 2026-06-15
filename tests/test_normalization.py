from __future__ import annotations

from ct_mcot.datasets.normalization import choice_to_index, exact_match, normalize_answer


def test_normalize_answer() -> None:
    assert normalize_answer("The, Cat!") == "cat"
    assert normalize_answer("1,234") == "1234"


def test_choice_to_index() -> None:
    assert choice_to_index("B", ["red", "blue"]) == 1
    assert choice_to_index("blue", ["red", "Blue!"]) == 1


def test_exact_match() -> None:
    assert exact_match("A triangle.", "triangle")
