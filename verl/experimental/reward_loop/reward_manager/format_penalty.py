"""Unified bad-format penalty policy for the step-level reward managers.

Division of labor per the Bad Format Penalty design note (Notion, under 260724): the XML
splitter (verl.utils.step_splitter.analyze_xml_steps) reports which steps exist, whether each
is valid, and the defect reason; THIS module turns those facts plus the response-level
conditions (truncation, boxed count, step-count cap) into reward assignments. The splitter
never knows the penalty score, and both StepRewardManager and TreeRewardManager call this one
implementation instead of keeping private copies.

Rule order and shapes (penalty_score is the configured value, typically -1):

1. Truncation (response hit the token limit, penalty_on_truncated): every step position gets
   penalty_score (one entry at the last token when no steps exist); the verifier does not run.
2. Boxed contract, active only when penalty_on_multi_boxed is set. Despite its name the flag
   switches the WHOLE contract, the zero-\\boxed rule included; tasks without a \\boxed answer
   (e.g. the logic FOL tasks) keep it off and stay untouched. Zero \\boxed: the OUTCOME score
   is replaced with penalty_score and the steps still verify normally. Several \\boxed: the
   FIRST one is the committed answer, so the caller regrades the outcome on the response
   prefix that ends with the first \\boxed (first_boxed_end) and cuts the verifier response
   there, and each LATER \\boxed costs one penalty_score at its own token position
   (extra_boxed_char_ends, placed by the caller via place_extra_penalty). A later correct
   \\boxed can therefore neither rescue the outcome nor farm process reward. The steps
   themselves keep their own verdicts wherever the \\boxed sit: measured rollouts contain no
   steps after a \\boxed, so there is deliberately no per-step rule tied to boxed positions.
3. Per-step XML validity (use_xml and penalty_on_bad_format): no <step> block at all means one
   penalty_score entry at the last token and no verifier run; otherwise each INVALID step
   (unclosed without truncation, bad tag structure or order, missing or empty premise or
   conclusion) is penalized alone through penalty_step_indices while the valid steps proceed
   to normal FOL / Isabelle verification, and reasoning tags OUTSIDE every block (a stray
   </step>, a dangling <conclusion>) cost ONE penalty_score placed at the last stray tag's
   own token position (stray_tag_char_end + place_extra_penalty). This replaces the old
   whole-response penalty on a tag-count mismatch or an outside conclusion. The old gate on a
   FOL-family reward type is gone: the rule now applies to any reward type whenever use_xml
   and penalty_on_bad_format are set.
4. Step-count cap (penalty_max_steps): indices beyond the cap join penalty_step_indices, the
   prefix is unaffected (unchanged behavior).

The Isabelle backend translates whole responses, so the caller strips the invalid steps'
character ranges (invalid_char_ranges + strip_char_ranges) from the response it forwards; the
per-step FOL backend simply skips the penalized indices.

All of the above shapes TRAINING rewards only: a validation call (validate_with_step_reward
false) returns the raw outcome score before assess_response runs, so validation metrics stay
plain benchmark accuracy, comparable across baselines.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from verl.utils.step_splitter import (analyze_xml_steps, boxed_spans,
                                      find_stray_xml_tags, first_boxed_end)

_BOXED_RE = re.compile(r"\\boxed\{")


@dataclass
class FormatDecision:
    """What the reward manager should do with one response's rewards. `penalized` refers to the PROCESS rewards; an outcome_override alone (boxed contract) does not set it."""

    outcome_override: Optional[float] = None    # replace the outcome score (zero \boxed)
    first_boxed_end: Optional[int] = None       # several \boxed: character index just past the FIRST one; the caller regrades the outcome on response[:first_boxed_end] and cuts the verifier response there
    extra_boxed_char_ends: list = field(default_factory=list)  # several \boxed: character end of each one AFTER the first; the caller maps each to a token position and places one penalty_score there (collision-nudged, see place_extra_penalty)
    process_rewards: Optional[list] = None      # final (position, score) list when skip_verifier
    skip_verifier: bool = False                 # process_rewards is complete; no backend call
    penalty_step_indices: set = field(default_factory=set)   # per-step penalties for the caller's existing plumbing
    invalid_char_ranges: list = field(default_factory=list)  # (start, end) of invalid step blocks, for the whole-response Isabelle call
    stray_tag_char_end: Optional[int] = None    # reasoning tags outside every step block: character end of the last one; the caller maps it to a token position and places ONE penalty_score there (collision-nudged, see place_extra_penalty)
    num_steps_override: Optional[int] = None    # e.g. 0 when no XML step exists
    penalized: bool = False
    reasons: list = field(default_factory=list)


def strip_char_ranges(text: str, ranges: list) -> str:
    """`text` minus the given (start, end) character ranges."""
    out, cursor = [], 0
    for start, end in sorted(ranges):
        out.append(text[cursor:start])
        cursor = max(cursor, end)
    out.append(text[cursor:])
    return "".join(out)


def place_extra_penalty(entries: list, token_pos: int, score: float, last_token_pos: int) -> None:
    """Append (token_pos, score) to `entries` without overwriting an existing entry: the advantage builder assigns scores by position (last write wins), so a collision would erase a step's own reward. Nudges forward, then backward, to the nearest free position; drops the entry in the degenerate case where every position is taken."""
    used = {int(pos) for pos, _ in entries}
    position = min(max(int(token_pos), 0), int(last_token_pos))
    while position in used and position < last_token_pos:
        position += 1
    if position in used:
        position = min(max(int(token_pos), 0), int(last_token_pos))
        while position in used and position > 0:
            position -= 1
    if position not in used:
        entries.append((int(position), float(score)))


def verifier_response(response_text: str, decision: FormatDecision) -> str:
    """The response the whole-response (Isabelle) verifier should see: cut at the committed first \\boxed when several exist, then the invalid step blocks stripped, and the answer text preserved when an invalid block swallowed it. The unclosed tail block runs to the end of the response by construction, so its range may contain the \\boxed line; the answer belongs to the response, not to the bad step, so the first \\boxed is re-appended when stripping would have deleted every \\boxed."""
    text = response_text
    if decision.first_boxed_end is not None:
        text = text[: decision.first_boxed_end]
    if not decision.invalid_char_ranges:
        return text
    stripped = strip_char_ranges(text, decision.invalid_char_ranges)
    first = _BOXED_RE.search(text)
    if first is not None and _BOXED_RE.search(stripped) is None:
        stripped = stripped.rstrip() + "\n\n" + text[first.start(): first_boxed_end(text)]
    return stripped


def _all_positions_penalty(step_positions, penalty_score, last_token_pos):
    if step_positions:
        return [(int(pos), float(penalty_score)) for _, pos in step_positions]
    return [(last_token_pos, float(penalty_score))]


def assess_response(
    response_text: str,
    step_positions: list,
    *,
    use_xml: bool,
    valid_response_length: int,
    response_length: int,
    penalty_score: float,
    penalty_max_steps: int,
    penalty_on_truncated: bool,
    penalty_on_multi_boxed: bool,
    penalty_on_bad_format: bool,
) -> FormatDecision:
    """Apply the unified rules to one response. `step_positions` is the manager's (step_text, token_end_pos) list; when use_xml found steps it is index-aligned with analyze_xml_steps (same scanner, unclosed tail included)."""
    decision = FormatDecision()
    last_token_pos = max(0, int(valid_response_length) - 1)

    # 1. Truncation keeps the existing whole-response rule.
    if penalty_on_truncated and valid_response_length >= response_length:
        decision.process_rewards = _all_positions_penalty(step_positions, penalty_score, last_token_pos)
        decision.skip_verifier = True
        decision.penalized = True
        decision.reasons.append("truncated")
        return decision

    # 2. Boxed contract, separated from the XML step rules.
    if penalty_on_multi_boxed:
        spans = boxed_spans(response_text)
        if len(spans) > 1:
            # The FIRST \boxed is the committed answer: the caller regrades the outcome on
            # the prefix ending there and cuts the verifier response at the same point, and
            # each LATER \boxed costs one penalty_score at its own token position. The steps
            # keep their own verdicts: the boxed contract never touches step verification.
            decision.first_boxed_end = spans[0][1]
            decision.extra_boxed_char_ends = [end for _, end in spans[1:]]
            decision.penalized = True
            decision.reasons.append(f"multi_boxed={len(spans)}")
        elif not spans:
            decision.outcome_override = float(penalty_score)
            decision.reasons.append("no_boxed")

    # 3. Per-step XML validity.
    if use_xml and penalty_on_bad_format:
        reports = analyze_xml_steps(response_text)
        if not reports:
            decision.process_rewards = [(last_token_pos, float(penalty_score))]
            decision.skip_verifier = True
            decision.penalized = True
            decision.reasons.append("bad_format(no_xml_step)")
            decision.num_steps_override = 0
            return decision
        for i, report in enumerate(reports):
            if not report.valid:
                decision.penalty_step_indices.add(i)
                decision.invalid_char_ranges.append((report.start, report.end))
                decision.reasons.append(f"bad_step_format@{i}:{report.error}")
        stray_tags = find_stray_xml_tags(response_text)
        if stray_tags:
            # A reasoning tag outside every step block (a stray </step>, a <conclusion>
            # dangling after the blocks) breaks the format without belonging to any step
            # index: ONE penalty_score placed at the last stray tag's own position, while
            # the steps keep their own verdicts. The old whole-response rule
            # (has_conclusion_outside_step) is replaced by this localized one.
            decision.stray_tag_char_end = max(end for _, end, _ in stray_tags)
            decision.penalized = True
            decision.reasons.append(
                "stray_xml_tags=" + ",".join(sorted({tag for _, _, tag in stray_tags})))

    # 4. Step-count cap.
    num_steps = len(step_positions)
    if penalty_max_steps > 0 and num_steps > penalty_max_steps:
        decision.penalty_step_indices.update(range(penalty_max_steps, num_steps))
        decision.reasons.append(f"num_steps={num_steps}>{penalty_max_steps}")

    if decision.penalty_step_indices:
        decision.penalized = True
    return decision
