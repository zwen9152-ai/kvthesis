import pytest

from eval.grader import extract_answer, is_equiv, last_boxed


def test_last_boxed_nested():
    assert last_boxed(r"so \boxed{1} then \boxed{\frac{a}{b}}") == r"\frac{a}{b}"


def test_extract_prefers_answer_after_think():
    text = r"<think>maybe \boxed{3}</think> Final: \boxed{4}"
    assert extract_answer(text) == "4"


def test_extract_falls_back_to_last_number():
    assert extract_answer("The total is 1,234 dollars.") == "1,234"


@pytest.mark.parametrize("pred,gold", [
    ("18", "18"), ("1,000", "1000"), (r"\dfrac{1}{2}", r"\frac{1}{2}"), ("0.5", r"\frac{1}{2}"),
    ("x = 5", "5"), (r"10\%", "10"), (r"90^\circ", "90"), (r"\text{(C)}", "(C)"), ("3.0", "3"),
    (r"\frac12", r"\frac{1}{2}"), ("033", "33"),
])
def test_equivalent(pred, gold):
    assert is_equiv(pred, gold)


@pytest.mark.parametrize("pred,gold", [("17", "18"), (None, "1"), (r"\frac{1}{3}", "0.33")])
def test_not_equivalent(pred, gold):
    assert not is_equiv(pred, gold)


def test_split_methods():
    from eval.run_eval import split_methods
    assert split_methods("full,snapkv@rollback=1,recall=0,h2o") == ["full", "snapkv@rollback=1,recall=0", "h2o"]
