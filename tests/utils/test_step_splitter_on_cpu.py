"""CPU-only tests for the shared XML step grammar."""

from verl.utils.fol_utils.common import (
    check_step_format_fol as compatibility_check_step_format_fol,
)
from verl.utils.fol_utils.common import get_step_list as compatibility_get_step_list
from verl.utils.fol_utils.common import parse_step_tags as compatibility_parse_step_tags
from verl.utils.step_splitter import (
    analyze_xml_steps,
    boxed_spans,
    check_step_format,
    find_stray_xml_tags,
    check_step_format_fol,
    first_boxed_end,
    get_step_list,
    get_step_token_positions,
    parse_step_tag_contents,
    parse_step_tags,
    parse_xml_steps,
    split_by_xml_step_tags,
)


def test_boxed_spans_cover_every_boxed_with_nesting_and_unbalanced_tail():
    text = "a \\boxed{\\frac{1}{2}} b \\boxed{8"
    spans = boxed_spans(text)
    assert len(spans) == 2
    assert text[spans[0][0]: spans[0][1]] == "\\boxed{\\frac{1}{2}}"
    assert spans[1][1] == len(text)          # braces never close: span runs to the end
    assert first_boxed_end(text) == spans[0][1]
    assert boxed_spans("no answer here") == []
    assert first_boxed_end("no answer here") is None


class CharacterTokenizer:
    """Tokenizer with one token per character for boundary tests."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_parse_step_tags_uses_last_conclusion():
    step = (
        "<step>"
        "<premise> first premise </premise>"
        "<premise>second premise</premise>"
        "<conclusion>draft</conclusion>"
        "<conclusion> final conclusion </conclusion>"
        "</step>"
    )

    expected = {
        "premises": ["first premise", "second premise"],
        "conclusion": "final conclusion",
    }
    assert parse_step_tags(step) == expected
    assert compatibility_parse_step_tags(step) == expected
    assert parse_step_tag_contents(step) == (
        ["first premise", "second premise"],
        ["draft", "final conclusion"],
    )


def test_parse_step_tags_allows_missing_conclusion_for_fol_callers():
    expected = {"premises": ["p"], "conclusion": None}
    assert parse_step_tags("<step><premise>p</premise></step>") == expected


def test_parse_xml_steps_preserves_isabelle_contract():
    response = (
        "prefix"
        "<step>\n"
        "  <premise> p1 </premise>\n"
        "  <premise>p2</premise>\n"
        "  <conclusion> first </conclusion>\n"
        "  <conclusion>second</conclusion>\n"
        "</step>"
        "suffix"
    )

    expected = [
        {
            "premises": ["p1", "p2"],
            "conclusion": "first",
            "block_text": (
                "\n"
                "  <premise> p1 </premise>\n"
                "  <premise>p2</premise>\n"
                "  <conclusion> first </conclusion>\n"
                "  <conclusion>second</conclusion>\n"
            ),
        }
    ]
    assert parse_xml_steps(response) == expected


def test_parse_xml_steps_rejects_missing_or_unclosed_steps():
    assert parse_xml_steps("no XML steps") is None
    assert parse_xml_steps("<step><premise>p</premise></step>") is None
    assert parse_xml_steps("<step><conclusion>c</conclusion>") is None


def test_parse_xml_steps_requires_the_fol_interior_per_step():
    """Interior requirement aligned with check_step_format (2026-07-17): every closed step needs at least one nonempty premise and a nonempty conclusion; a premise-free step was the cheapest trivially-true-statement reward surface."""
    assert parse_xml_steps(
        "<step><conclusion>c</conclusion></step>\n\\boxed{7}") is None
    assert parse_xml_steps(
        "<step><premise></premise><conclusion>c</conclusion></step>") is None
    assert parse_xml_steps(
        "<step><premise>p</premise><conclusion></conclusion></step>") is None


def test_parse_xml_steps_ignores_unclosed_tail_after_closed_step():
    response = (
        "<step><premise>p</premise><conclusion>closed</conclusion></step>"
        "<step><premise>q</premise><conclusion>truncated</conclusion>"
    )
    assert parse_xml_steps(response) == [
        {
            "premises": ["p"],
            "conclusion": "closed",
            "block_text": "<premise>p</premise><conclusion>closed</conclusion>",
        }
    ]


def test_split_by_xml_step_tags_includes_unclosed_tail_and_offsets():
    response = (
        "before "
        "<step><conclusion>one</conclusion></step>"
        " between "
        "<step><conclusion>two</conclusion>"
    )
    expected_texts = [
        "<step><conclusion>one</conclusion></step>",
        "<step><conclusion>two</conclusion>",
    ]

    steps = split_by_xml_step_tags(response)
    assert [text for text, _, _ in steps] == expected_texts
    for text, start, end in steps:
        assert response[start:end] == text


def test_xml_step_tags_remain_case_sensitive_and_attribute_free():
    assert split_by_xml_step_tags("<STEP><conclusion>x</conclusion></STEP>") == []
    assert split_by_xml_step_tags(
        '<step id="1"><conclusion>x</conclusion></step>'
    ) == []
    assert parse_xml_steps("<STEP><conclusion>x</conclusion></STEP>") is None
    assert parse_xml_steps(
        '<step id="1"><conclusion>x</conclusion></step>'
    ) is None


def test_analyze_xml_steps_reports_each_defect():
    """The splitter reports format FACTS only (which steps, valid or not, why); the penalty policy lives in reward_manager.format_penalty."""
    cases = {
        "<step><premise>p</premise><conclusion>c</conclusion></step>": None,
        "<step><premise>p</premise><conclusion>c</conclusion>": "unclosed",
        "<step><conclusion>c</conclusion></step>": "no_premise",
        "<step><premise></premise><conclusion>c</conclusion></step>": "empty_premise",
        "<step><premise>p</premise></step>": "no_conclusion",
        "<step><premise>p</premise><conclusion></conclusion></step>": "empty_conclusion",
        "<step><conclusion>c</conclusion><premise>p</premise></step>": "bad_tag_order",
        "<step><premise>p</conclusion><conclusion>c</premise></step>": "bad_tag_structure",
    }
    for text, expected_error in cases.items():
        (report,) = analyze_xml_steps(text)
        assert report.error == expected_error, (text, report.error)
        assert report.valid is (expected_error is None)


def test_analyze_xml_steps_aligns_with_split_by_xml_step_tags():
    response = (
        "intro <step><premise>p</premise><conclusion>c</conclusion></step>"
        " mid <step><premise>q</premise></step>"
        " tail <step><premise>r</premise><conclusion>d</conclusion>"
    )
    reports = analyze_xml_steps(response)
    assert [(r.text, r.start, r.end) for r in reports] == split_by_xml_step_tags(response)
    assert [r.error for r in reports] == [None, "no_conclusion", "unclosed"]


def test_find_stray_xml_tags_reports_tags_outside_blocks():
    """Reasoning tags outside every step block are format facts too: a dangling conclusion, a stray closing step. Tags inside a block, an unclosed tail block included, are the per-step analyzer's business."""
    response = (
        "<step><premise>p</premise><conclusion>c</conclusion></step>"
        "<conclusion>extra</conclusion>"
    )
    assert [tag for _, _, tag in find_stray_xml_tags(response)] == ["conclusion", "/conclusion"]
    assert find_stray_xml_tags("</step> alone") == [(0, len("</step>"), "/step")]
    assert find_stray_xml_tags(
        "<step><premise>p</premise><conclusion>c</conclusion></step>") == []
    # tags after an unclosed <step> sit inside its block (it runs to the end of the text)
    assert find_stray_xml_tags("<step><premise>p</premise><conclusion>c</conclusion>") == []


def test_get_step_list_returns_closed_inner_text_only():
    response = (
        "<step> <premise>p</premise><conclusion>c</conclusion> </step>"
        "<step><conclusion>truncated</conclusion>"
    )
    expected = ["<premise>p</premise><conclusion>c</conclusion>"]
    assert get_step_list(response) == expected
    assert compatibility_get_step_list(response) == expected


def test_strict_format_accepts_multiple_nonempty_tags():
    step = (
        "<step>"
        "<premise>p1</premise>"
        "<premise>p2</premise>"
        "<conclusion>c1</conclusion>"
        "<conclusion>c2</conclusion>"
        "</step>"
    )
    assert check_step_format(step)
    assert check_step_format_fol(step)
    assert compatibility_check_step_format_fol(step)


def test_strict_format_rejects_invalid_structures():
    invalid_steps = [
        "<step><conclusion>c</conclusion></step>",
        "<step><premise>p</premise></step>",
        "<step><premise></premise><conclusion>c</conclusion></step>",
        "<step><premise>p</premise><conclusion></conclusion></step>",
        "<step><conclusion>c</conclusion><premise>p</premise></step>",
        "<step><premise>p</conclusion><conclusion>c</premise></step>",
        "<step><premise>p</premise><conclusion>c</conclusion>",
        "prefix<step><premise>p</premise><conclusion>c</conclusion></step>",
        "<step><premise>p</premise><conclusion>c</conclusion></step>suffix",
        (
            "<step><premise>p</premise><conclusion>c</conclusion></step>"
            "<step><premise>q</premise><conclusion>d</conclusion></step>"
        ),
    ]
    assert all(not check_step_format_fol(step) for step in invalid_steps)


def test_xml_token_positions_use_actual_response_ids():
    response = (
        "prefix"
        "<step><premise>p</premise><conclusion>one</conclusion></step>"
        "middle"
        "<step><premise>q</premise><conclusion>two</conclusion></step>"
    )
    tokenizer = CharacterTokenizer()
    response_ids = tokenizer.encode(response)
    xml_steps = split_by_xml_step_tags(response)

    positions = get_step_token_positions(
        response,
        len(response_ids),
        tokenizer,
        use_xml=True,
        response_ids=response_ids,
    )

    assert [text for text, _ in positions] == [step[0] for step in xml_steps]
    assert [position for _, position in positions] == [
        char_end - 1 for _, _, char_end in xml_steps
    ]
