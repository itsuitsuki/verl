"""CPU-only pins for the unified bad-format penalty policy (reward_manager.format_penalty).

Pins the rules from the Bad Format Penalty design note: truncation is a whole-response process
penalty; zero \\boxed replaces the OUTCOME score only (the steps still verify); with several
\\boxed the FIRST one is the committed answer (outcome regraded on the prefix ending there,
each LATER \\boxed penalized once at its own position); a badly formatted step is penalized
alone while the valid steps still reach the verifier; the no-step case puts one penalty at the
last token; and the boxed rules stay inert for tasks without a boxed contract (the logic FOL
tasks). Pure python: the manager plumbing is not exercised here, only the shared decision
function both managers call.
"""
from verl.experimental.reward_loop.reward_manager.format_penalty import (
    assess_response, place_extra_penalty, strip_char_ranges, verifier_response)
from verl.utils.step_splitter import split_by_xml_step_tags

VALID = "<step><premise>p</premise><conclusion>c</conclusion></step>"
NO_CONCLUSION = "<step><premise>q</premise></step>"
UNCLOSED = "<step><premise>r</premise><conclusion>d</conclusion>"


def _positions(response):
    """Fake (step_text, token_end_pos) list index-aligned with the XML scanner, token position = index."""
    return [(text, i) for i, (text, _s, _e) in enumerate(split_by_xml_step_tags(response))]


def _assess(response, positions=None, **overrides):
    kwargs = dict(
        use_xml=True, valid_response_length=50, response_length=100,
        penalty_score=-1.0, penalty_max_steps=0,
        penalty_on_truncated=True, penalty_on_multi_boxed=True,
        penalty_on_bad_format=True)
    kwargs.update(overrides)
    return assess_response(
        response, _positions(response) if positions is None else positions, **kwargs)


def test_truncation_penalizes_every_step_and_skips_the_verifier():
    response = VALID + VALID + " \\boxed{7}"
    decision = _assess(response, valid_response_length=100)
    assert decision.skip_verifier and decision.penalized
    assert decision.reasons == ["truncated"]
    assert decision.process_rewards == [(0, -1.0), (1, -1.0)]
    assert decision.outcome_override is None


def test_multi_boxed_commits_to_the_first_answer():
    """Several \\boxed: the first one is the committed answer (the caller regrades on the prefix ending there), each LATER \\boxed costs one penalty at its own position, and the steps keep their own results wherever the \\boxed sit."""
    response = VALID + " \\boxed{7} " + VALID + " \\boxed{8}"
    decision = _assess(response)
    assert decision.outcome_override is None
    assert decision.first_boxed_end == response.index("\\boxed{7}") + len("\\boxed{7}")
    assert decision.extra_boxed_char_ends == [len(response)]
    assert decision.skip_verifier is False
    assert decision.penalty_step_indices == set()
    assert decision.penalized is True
    assert decision.reasons == ["multi_boxed=2"]


def test_first_boxed_end_matches_nested_braces():
    response = VALID + " \\boxed{\\frac{1}{2}} tail \\boxed{8}"
    decision = _assess(response)
    assert response[: decision.first_boxed_end].endswith("\\boxed{\\frac{1}{2}}")


def test_no_boxed_penalizes_the_outcome_only():
    response = VALID + VALID
    decision = _assess(response)
    assert decision.outcome_override == -1.0
    assert decision.skip_verifier is False
    assert decision.process_rewards is None
    assert decision.penalized is False
    assert decision.reasons == ["no_boxed"]


def test_multi_boxed_and_an_invalid_step_compose_independently():
    """The boxed contract sets the committed-answer cut plus the extra-boxed position, the XML rules mark the invalid step; both apply."""
    response = VALID + NO_CONCLUSION + " \\boxed{7} \\boxed{8}"
    decision = _assess(response)
    assert decision.outcome_override is None
    assert decision.first_boxed_end is not None
    assert decision.extra_boxed_char_ends == [len(response)]
    assert decision.penalty_step_indices == {1}
    assert decision.reasons == ["multi_boxed=2", "bad_step_format@1:no_conclusion"]


def test_boxed_rules_stay_inert_without_a_boxed_contract():
    """Tasks without a \\boxed answer (logic FOL) must stay untouched."""
    response = VALID + VALID
    decision = _assess(response, penalty_on_multi_boxed=False)
    assert decision.outcome_override is None
    assert decision.skip_verifier is False
    assert decision.penalty_step_indices == set()


def test_no_xml_step_gets_one_penalty_at_the_last_token():
    decision = _assess("plain text \\boxed{7}", positions=[("plain text", 3)])
    assert decision.skip_verifier and decision.penalized
    assert decision.process_rewards == [(49, -1.0)]
    assert decision.reasons == ["bad_format(no_xml_step)"]
    assert decision.num_steps_override == 0


def test_invalid_step_is_penalized_alone_and_stripped_for_the_verifier():
    response = VALID + NO_CONCLUSION + VALID + " \\boxed{7}"
    decision = _assess(response)
    assert decision.skip_verifier is False
    assert decision.penalty_step_indices == {1}
    assert decision.reasons == ["bad_step_format@1:no_conclusion"]
    assert verifier_response(response, decision) == VALID + VALID + " \\boxed{7}"


def test_verifier_response_keeps_the_answer_swallowed_by_an_unclosed_block():
    """The unclosed tail block runs to the end of the response, so stripping it would also delete the \\boxed line; the answer belongs to the response, not to the bad step."""
    response = VALID + "\n<step>\n<premise>r</premise>\n<conclusion>d</conclusion>\n\\boxed{7}"
    decision = _assess(response)
    assert decision.penalty_step_indices == {1}
    assert decision.reasons == ["bad_step_format@1:unclosed"]
    cleaned = verifier_response(response, decision)
    assert VALID in cleaned
    assert "<premise>r</premise>" not in cleaned
    assert cleaned.endswith("\\boxed{7}")


def test_unclosed_tail_without_truncation_is_penalized_alone():
    response = VALID + " \\boxed{7} " + UNCLOSED
    decision = _assess(response)
    assert decision.skip_verifier is False
    assert decision.penalty_step_indices == {1}
    assert decision.reasons == ["bad_step_format@1:unclosed"]


def test_max_steps_cap_penalizes_only_the_suffix():
    response = VALID * 3 + " \\boxed{7}"
    decision = _assess(response, penalty_max_steps=2)
    assert decision.skip_verifier is False
    assert decision.penalty_step_indices == {2}
    assert decision.reasons == ["num_steps=3>2"]


def test_stray_tags_cost_one_localized_penalty():
    """A reasoning tag outside every step block is penalized once at its own position; the steps keep their results (replaces the old whole-response has_conclusion_outside_step rule)."""
    response = VALID + "<conclusion>extra</conclusion> \\boxed{7}"
    decision = _assess(response)
    assert decision.penalty_step_indices == set()
    assert decision.stray_tag_char_end == response.rindex("</conclusion>") + len("</conclusion>")
    assert decision.penalized is True
    assert decision.reasons == ["stray_xml_tags=/conclusion,conclusion"]
    assert decision.skip_verifier is False


def test_place_extra_penalty_never_overwrites_an_existing_entry():
    entries = [(5, 1.0), (6, 0.0)]
    place_extra_penalty(entries, 5, -1.0, 9)
    assert (5, 1.0) in entries and (6, 0.0) in entries and (7, -1.0) in entries
    fully_taken = [(0, 1.0)]
    place_extra_penalty(fully_taken, 0, -1.0, 0)
    assert fully_taken == [(0, 1.0)]


def test_strip_char_ranges_removes_exactly_the_given_ranges():
    assert strip_char_ranges("aXXbYYc", [(1, 3), (4, 6)]) == "abc"
    assert strip_char_ranges("abc", []) == "abc"
