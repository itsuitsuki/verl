"""CPU-only tests for the balanced-brace boxed_answer (xml_utils.py).

2026-07-11 fix: the old flat regex ([^{}]+) rejected ANY nested braces, so
answers like \\boxed{\\frac{1}{2}} silently failed the boxed gate and
fail-closed the whole response's Isabelle process reward."""
from verl.utils.isabelle_utils.xml_utils import boxed_answer


def test_flat():
    assert boxed_answer(r"foo \boxed{42} bar") == "42"


def test_nested_frac():
    assert boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_deep_nesting():
    assert boxed_answer(r"\boxed{\sqrt{\frac{a}{b}}}") == r"\sqrt{\frac{a}{b}}"


def test_last_of_multiple():
    assert boxed_answer(r"\boxed{1} then \boxed{2}") == "2"


def test_unterminated_ignored():
    assert boxed_answer(r"\boxed{\frac{1}{2}") is None
    # a valid box BEFORE an unterminated one is kept
    assert boxed_answer(r"\boxed{ok} junk \boxed{\frac{1}") == "ok"


def test_no_box():
    assert boxed_answer("no box here") is None


def test_empty_box():
    assert boxed_answer(r"\boxed{}") is None
