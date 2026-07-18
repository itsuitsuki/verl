"""Extract numbers admitted by translation-faithfulness checks."""

import re
from fractions import Fraction


FREE_NUMS = {Fraction(0), Fraction(1), Fraction(2)}


WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
    "half": "1/2", "third": "1/3", "quarter": "1/4",
    "first": 1, "second": 2,
}


_SUPERSCRIPT = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺",
    "0123456789-+")


def num_values(text: str, words: bool = False) -> set:
    vals = set()
    text = text.translate(_SUPERSCRIPT)
    merged = re.sub(r"(?<=\d),(?:\\!)?(?=\d{3})", "", text)
    split = re.sub(r"(?<=\d),(?=\d)", " ", text)
    for t in ((merged, split) if merged != text else (text,)):
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\.?\d)", t):
            try:
                vals.add(Fraction(m.group(1)))
            except ValueError:
                pass
        for m in re.finditer(r"(?<![\w.])(\.\d+)", t):
            vals.add(Fraction("0" + m.group(1)))
    if words:
        for w, v in WORD_NUMS.items():
            if re.search(rf"\b{w}\b", text, re.IGNORECASE):
                vals.add(Fraction(v))
        for m in re.finditer(r"10\s*\^\s*\{?\s*(-?\d+)\s*\}?", text):
            k = int(m.group(1))
            if -30 <= k <= 30:
                vals.add(Fraction(10) ** k)
        for m in re.finditer(
                r"(\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|\*)\s*10\s*"
                r"\^\s*\{?\s*(-?\d+)\s*\}?", text):
            k = int(m.group(2))
            if -30 <= k <= 30:
                vals.add(Fraction(m.group(1)) * Fraction(10) ** k)
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*"
                             r"(thousand|million|billion|trillion)", text,
                             re.IGNORECASE):
            mult = {"thousand": 3, "million": 6, "billion": 9,
                    "trillion": 12}[m.group(2).lower()]
            vals.add(Fraction(m.group(1)) * Fraction(10) ** mult)
    return vals


def range_ints(text: str, cap: int = 200) -> set:
    """Return integers covered by inclusive range phrases.

    ``num_values`` sees only the endpoints of phrases such as ``1 through 9``. Including the interior integers prevents a faithful LCM translation from treating 4, 6, and 8 as invented. Ranges wider than ``cap`` are ignored so a large incidental range cannot defeat the provenance check.
    """
    out = set()
    pats = (
        r"(\d+)\s+(?:through|thru)\s+(\d+)",
        r"from\s+(\d+)\s+to\s+(\d+)",
        r"(?:integers?|numbers?|digits?|values?|terms?)\s+(\d+)\s+to\s+(\d+)",
        r"between\s+(\d+)\s+and\s+(\d+)",
        # ellipsis: "1, 2, ..., 9" / "1, 2, \ldots, 9" / "1, 2, \cdots, 9"
        r"(\d+)\s*,\s*\d+\s*,\s*(?:\.\.\.|…|\\l?dots|\\cdots)\s*,?\s*(\d+)",
    )
    for pat in pats:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                lo, hi = int(m.group(1)), int(m.group(2))
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            if 0 <= hi - lo <= cap:
                out |= {Fraction(i) for i in range(lo, hi + 1)}
    return out
