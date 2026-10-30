"""Answer extraction and equivalence checking for GSM8K / MATH / AIME."""

import re
from fractions import Fraction


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}), with nested braces handled."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def extract_answer(text: str) -> str | None:
    """Prefer the final \\boxed answer after </think>; fall back to the last number."""
    tail = text.split("</think>")[-1]
    boxed = last_boxed(tail) or last_boxed(text)
    if boxed is not None:
        return boxed
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return nums[-1] if nums else None


_REPLACEMENTS = [
    ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
    ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""),
    ("\\$", ""), ("$", ""), ("\\displaystyle", ""),
]


def _strip_text(s: str) -> str:
    return re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}", r"\1", s)


def normalize(ans: str) -> str:
    s = ans.strip()
    for a, b in _REPLACEMENTS:
        s = s.replace(a, b)
    s = _strip_text(s)
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"^[a-zA-Z]\s*=", "", s)  # "x=5" -> "5"
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)  # \frac12 -> \frac{1}{2}
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)  # 1,000 -> 1000
    s = re.sub(r"^0+(\d)", r"\1", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def to_number(s: str) -> Fraction | None:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        frac = Fraction(int(m.group(2)), int(m.group(3)))
        return -frac if m.group(1) else frac
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def is_equiv(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None:
        return abs(float(pn) - float(gn)) <= 1e-6 * max(1.0, abs(float(gn)))
    return False
