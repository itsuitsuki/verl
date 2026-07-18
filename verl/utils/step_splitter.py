"""Split reasoning responses into text steps and token boundaries.

XML responses share one scanner and one premise/conclusion parser. Callers may
then choose either permissive splitting for truncated generation or strict
validation for formal-verification rewards.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class _XmlStepBlock:
    full_text: str
    inner_text: str
    start: int
    end: int
    closed: bool


_XML_STEP_PATTERN = re.compile(r"<step>.*?(?:</step>|$)", re.DOTALL)
_PREMISE_PATTERN = re.compile(r"<premise>(.*?)</premise>", re.DOTALL)
_CONCLUSION_PATTERN = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)
_TAG_PATTERN = re.compile(r"<(/?\w+)>")


def _scan_xml_step_blocks(
    response_text: str,
    *,
    include_unclosed: bool,
) -> list[_XmlStepBlock]:
    """Return XML step blocks and their character ranges."""
    blocks = []
    for match in _XML_STEP_PATTERN.finditer(response_text):
        full_text = match.group(0)
        closed = full_text.endswith("</step>")
        if not closed and not include_unclosed:
            continue
        inner_end = -len("</step>") if closed else None
        blocks.append(
            _XmlStepBlock(
                full_text=full_text,
                inner_text=full_text[len("<step>") : inner_end],
                start=match.start(),
                end=match.end(),
                closed=closed,
            )
        )
    return blocks


def parse_step_tag_contents(step_text: str) -> tuple[list[str], list[str]]:
    """Return all premise and conclusion tag contents from one step."""
    premises = [item.strip() for item in _PREMISE_PATTERN.findall(step_text)]
    conclusions = [item.strip() for item in _CONCLUSION_PATTERN.findall(step_text)]
    return premises, conclusions


def parse_step_tags(step_text: str) -> dict:
    """Parse premise tags and the last conclusion tag from one step."""
    premises, conclusions = parse_step_tag_contents(step_text)
    conclusion = conclusions[-1] if conclusions else None
    return {"premises": premises, "conclusion": conclusion}


def parse_xml_steps(response_text: str):
    """Parse all closed XML steps in a response for formal verification.

    Each result contains the stripped premises, the first conclusion, and the
    text inside the outer ``<step>`` tags. A response with no closed steps is
    invalid, and so is one with a closed step that has no premise, no
    conclusion, or an empty premise or conclusion: the same interior
    requirement check_step_format enforces per step on the FOL path (aligned
    2026-07-17; a premise-free step was also the cheapest way to farm process
    reward with trivially true statements).
    """
    steps = []
    for block in _scan_xml_step_blocks(response_text, include_unclosed=False):
        premises, conclusions = parse_step_tag_contents(block.inner_text)
        if not premises or not all(premises) or not conclusions or not all(conclusions):
            return None
        steps.append(
            {
                "premises": premises,
                "conclusion": conclusions[0],
                "block_text": block.inner_text,
            }
        )
    return steps or None


def get_step_list(response_text: str) -> list[str]:
    """Return the stripped contents of all closed XML steps."""
    return [
        block.inner_text.strip()
        for block in _scan_xml_step_blocks(response_text, include_unclosed=False)
    ]


@dataclass(frozen=True)
class XmlStepReport:
    """Format facts about one identified ``<step>`` block: where it is, whether it closed, and its first defect (None = valid). Reward policy (what score a defective step receives) deliberately lives with the reward manager (reward_manager.format_penalty), never here."""

    text: str          # full block text, tags included
    start: int         # character start in the response
    end: int           # character end in the response
    closed: bool
    error: Optional[str]   # None, or one of: unclosed, bad_tag_structure, bad_tag_order, no_premise, empty_premise, no_conclusion, empty_conclusion

    @property
    def valid(self) -> bool:
        return self.error is None


def _classify_step_block(block: _XmlStepBlock) -> Optional[str]:
    """First applicable defect of one step block, mirroring the per-step rules check_step_format enforces on the FOL path."""
    if not block.closed:
        return "unclosed"
    stack = []
    for match in _TAG_PATTERN.finditer(block.full_text):
        tag = match.group(1)
        if tag.startswith("/"):
            if not stack or stack[-1] != tag[1:]:
                return "bad_tag_structure"
            stack.pop()
        else:
            stack.append(tag)
    if stack:
        return "bad_tag_structure"
    premises, conclusions = parse_step_tag_contents(block.inner_text)
    if not premises:
        return "no_premise"
    if not all(premises):
        return "empty_premise"
    if not conclusions:
        return "no_conclusion"
    if not all(conclusions):
        return "empty_conclusion"
    if block.full_text.find("<premise>") > block.full_text.find("<conclusion>"):
        return "bad_tag_order"
    return None


_BOXED_START_RE = re.compile(r"\\boxed\{")


def boxed_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of every ``\\boxed{...}``: (start of the ``\\boxed``, index just past its matching brace). The brace scan tolerates nesting (``\\boxed{\\frac{1}{2}}``); a span whose braces never close runs to len(text)."""
    text = text or ""
    spans = []
    for match in _BOXED_START_RE.finditer(text):
        depth = 1
        i = match.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        spans.append((match.start(), i))
    return spans


def first_boxed_end(text: str) -> Optional[int]:
    """Character index just past the matching brace of the FIRST ``\\boxed{...}``, or None when the text has no ``\\boxed`` (see :func:`boxed_spans`)."""
    spans = boxed_spans(text)
    return spans[0][1] if spans else None


_REASONING_TAG_NAMES = {"step", "/step", "premise", "/premise", "conclusion", "/conclusion"}


def find_stray_xml_tags(response_text: str) -> list[tuple[int, int, str]]:
    """Reasoning tags that sit OUTSIDE every identified ``<step>`` block: a stray ``</step>`` with no opener, a ``<conclusion>`` dangling after the blocks, premise or conclusion tags that belong to no step. Facts only ((start, end, tag) per stray tag); the reward managers' format policy decides what they cost. Text without reasoning tags outside blocks is legal and reported as nothing."""
    spans = [(block.start, block.end)
             for block in _scan_xml_step_blocks(response_text, include_unclosed=True)]
    stray = []
    for match in _TAG_PATTERN.finditer(response_text):
        tag = match.group(1)
        if tag not in _REASONING_TAG_NAMES:
            continue
        if any(start <= match.start() < end for start, end in spans):
            continue
        stray.append((match.start(), match.end(), tag))
    return stray


def analyze_xml_steps(response_text: str) -> list[XmlStepReport]:
    """Per-step format report for the reward managers: every identified ``<step>`` block in order, a final unclosed block included, each with its first defect. The list is index-aligned with split_by_xml_step_tags (same scanner, unclosed included)."""
    return [
        XmlStepReport(
            text=block.full_text,
            start=block.start,
            end=block.end,
            closed=block.closed,
            error=_classify_step_block(block),
        )
        for block in _scan_xml_step_blocks(response_text, include_unclosed=True)
    ]


def check_step_format(step_text: str) -> bool:
    """Return whether one step has complete, nonempty XML reasoning tags."""
    step_text = step_text.strip()
    blocks = _scan_xml_step_blocks(step_text, include_unclosed=False)
    if len(blocks) != 1 or blocks[0].full_text != step_text:
        return False

    if step_text.count("<step>") != 1 or step_text.count("</step>") != 1:
        return False

    premise_open = step_text.count("<premise>")
    premise_close = step_text.count("</premise>")
    conclusion_open = step_text.count("<conclusion>")
    conclusion_close = step_text.count("</conclusion>")
    if premise_open <= 0 or premise_open != premise_close:
        return False
    if conclusion_open <= 0 or conclusion_open != conclusion_close:
        return False
    if step_text.find("<premise>") > step_text.find("<conclusion>"):
        return False

    stack = []
    for match in _TAG_PATTERN.finditer(step_text):
        tag = match.group(1)
        if tag.startswith("/"):
            closing_tag = tag[1:]
            if not stack or stack[-1] != closing_tag:
                return False
            stack.pop()
        else:
            stack.append(tag)
    if stack:
        return False

    premises, conclusions = parse_step_tag_contents(step_text)
    return all(premises) and all(conclusions)


# Compatibility name retained for existing reward configuration and imports.
check_step_format_fol = check_step_format


def default_split_fn(response_text: str) -> list[str]:
    """Default step splitter: split by double newline."""
    if not response_text:
        return [""]
    return response_text.split("\n\n")


def split_response_into_steps(
    response_text: str,
    split_fn: Optional[Callable[[str], list[str]]] = None,
) -> list[tuple[str, int, int]]:
    """Split response text into steps using a given splitter function.

    Args:
        response_text: The full decoded response text.
        split_fn: A callable that splits the text into segments.
            Defaults to ``default_split_fn`` (split by ``\\n\\n``).

    Returns:
        List of (step_text, char_start, char_end) tuples.
    """
    if split_fn is None:
        split_fn = default_split_fn
    segments = split_fn(response_text)
    steps: list[tuple[str, int, int]] = []
    cursor = 0
    for seg in segments:
        start = response_text.find(seg, cursor)
        if start == -1:
            start = cursor
        end = start + len(seg)
        steps.append((seg, start, end))
        cursor = end
    return steps


def split_by_xml_step_tags(response_text: str) -> list[tuple[str, int, int]]:
    """Return XML steps with character ranges.

    A final unclosed ``<step>`` is included because generation may stop at the
    response token limit. Callers that validate complete responses should use
    :func:`parse_xml_steps` instead.
    """
    return [
        (block.full_text, block.start, block.end)
        for block in _scan_xml_step_blocks(
            response_text,
            include_unclosed=True,
        )
    ]


def char_end_to_token_pos(response_ids, tokenizer, char_end: int, valid_response_length: int) -> int:
    """Public name for the character-to-token boundary mapping (the reward managers place stray-tag penalties with it)."""
    return _char_end_to_token_pos(response_ids, tokenizer, char_end, valid_response_length)


def _char_end_to_token_pos(response_ids, tokenizer, char_end: int, valid_response_length: int) -> int:
    """Binary search: return the 0-indexed token position whose decoded prefix covers ``char_end`` chars.

    Works directly on the actual token IDs so there is no BPE round-trip drift
    (``encode(decode(ids[:k]))`` can differ from ``k`` at BPE merge boundaries).
    """
    ids = list(response_ids[:valid_response_length])
    n = len(ids)
    if n == 0:
        return 0
    # Find smallest prefix length k (1-indexed) s.t. len(decode(ids[:k])) >= char_end.
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) >= char_end:
            hi = mid
        else:
            lo = mid + 1
    return max(0, min(lo - 1, n - 1))  # convert to 0-indexed token position


def get_step_token_positions(
    response_text: str,
    valid_response_length: int,
    tokenizer,
    use_xml: bool = False,
    split_fn: Optional[Callable[[str], list[str]]] = None,
    response_ids=None,
) -> list[tuple[str, int]]:
    """Map character-level step boundaries to token positions.

    Tries XML ``<step>`` tag splitting first when *use_xml* is ``True``.
    Falls back to the delimiter-based splitter.

    Args:
        response_text: The full decoded response text.
        valid_response_length: Number of valid (non-padding) response tokens.
        tokenizer: HuggingFace tokenizer for encoding text.
        use_xml: If ``True``, attempt XML ``<step>`` tag splitting first.
        split_fn: Custom text splitter; defaults to ``default_split_fn``.
        response_ids: Optional actual token ID sequence (tensor or list).
            When provided, token positions are derived directly from the token
            IDs instead of re-encoding text prefixes, eliminating BPE drift.

    Returns:
        List of (step_text, token_end_pos) tuples where *token_end_pos* is the
        0-indexed position of the last token in this step within the response.
    """
    if response_ids is not None:
        if not use_xml:
            # Token-space splitting: no decode→encode round-trip at all.
            ids = list(response_ids[:valid_response_length])
            token_steps = split_tokens_by_delimiter(ids, tokenizer)
            result: list[tuple[str, int]] = []
            for _tok_start, tok_end, step_text in token_steps:
                token_end_pos = max(0, min(tok_end - 1, valid_response_length - 1))
                result.append((step_text, token_end_pos))
            return result
        else:
            # XML splitting: text-level boundaries, then binary-search token positions.
            steps = split_by_xml_step_tags(response_text)
            if not steps:
                steps = split_response_into_steps(response_text, split_fn)
            result = []
            for step_text, _char_start, char_end in steps:
                token_end_pos = _char_end_to_token_pos(response_ids, tokenizer, char_end, valid_response_length)
                result.append((step_text, token_end_pos))
            return result

    # Legacy path (no response_ids): original encode-prefix behaviour.
    steps: list[tuple[str, int, int]] = []
    if use_xml:
        steps = split_by_xml_step_tags(response_text)
    if not steps:
        steps = split_response_into_steps(response_text, split_fn)

    result = []
    for step_text, _char_start, char_end in steps:
        text_up_to_end = response_text[:char_end]
        tokens_up_to_end = tokenizer.encode(text_up_to_end, add_special_tokens=False)
        token_end_pos = min(len(tokens_up_to_end) - 1, valid_response_length - 1)
        token_end_pos = max(0, token_end_pos)
        result.append((step_text, token_end_pos))
    return result


def split_tokens_by_delimiter(
    token_ids,
    tokenizer,
    delimiter: str = "\n\n",
) -> list[tuple[int, int, str]]:
    """Split a token sequence at delimiter boundaries directly in token space.

    This avoids the decode |→| split |→| re-encode round-trip that causes BPE drift (``len(encode(decode(tokens[:k]))) != k``).

    The delimiter tokens are included at the **start** of the following step,
    matching the behaviour of the character-level ``split_response_into_steps()`` helper.

    Args:
        token_ids: Flat list/sequence of token IDs (e.g. from ``.tolist()``).
        tokenizer: HuggingFace tokenizer used to encode the delimiter and
            decode step text.
        delimiter: The text delimiter whose token encoding is searched for in
            *token_ids*.  Defaults to ``"\\n\\n"``.

    Returns:
        List of ``(token_start, token_end, step_text)`` tuples.
    """
    token_ids = list(token_ids)
    if not token_ids:
        return [(0, 0, "")]

    delim_tokens = tokenizer.encode(delimiter, add_special_tokens=False)
    delim_len = len(delim_tokens)

    if delim_len == 0:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return [(0, len(token_ids), text)]

    # Scan for delimiter token sequence
    split_points: list[int] = []
    i = 0
    while i <= len(token_ids) - delim_len:
        if token_ids[i : i + delim_len] == delim_tokens:
            split_points.append(i)
            i += delim_len
        else:
            i += 1

    # Build step ranges – delimiter tokens go with the *following* step
    ranges: list[tuple[int, int]] = []
    start = 0
    for pos in split_points:
        if pos > start:
            ranges.append((start, pos))
        start = pos
    if start < len(token_ids):
        ranges.append((start, len(token_ids)))

    if not ranges:
        ranges = [(0, len(token_ids))]

    # Decode each range to obtain step_text
    result: list[tuple[int, int, str]] = []
    for tok_start, tok_end in ranges:
        step_text = tokenizer.decode(
            token_ids[tok_start:tok_end], skip_special_tokens=True
        )
        result.append((tok_start, tok_end, step_text))

    return result


def get_split_fn(
    use_xml: bool = False,
) -> Callable[[str], list[str]]:
    """Return a text-level split function controlled by an explicit flag.

    When *use_xml* is ``True``, returns a splitter that tries ``<step>`` XML
    tags first and falls back to ``\\n\\n``.  When ``False`` (default), returns
    ``default_split_fn`` (``\\n\\n`` only).

    This is useful when callers need only the *text segments* (not token
    positions) — e.g. ``TreeManager`` which manages its own token bookkeeping.
    """
    if use_xml:

        def _xml_or_default(response_text: str) -> list[str]:
            steps = split_by_xml_step_tags(response_text)
            if steps:
                return [s[0] for s in steps]
            return default_split_fn(response_text)

        return _xml_or_default
    return default_split_fn


def get_split_fn_for_reward_type(
    step_reward_types: Optional[list[str]] = None,
) -> Callable[[str], list[str]]:
    """Deprecated. Use :func:`get_split_fn` with an explicit ``use_xml`` flag.

    Infers ``use_xml`` from reward type names for backward compatibility.
    """
    use_xml = step_reward_types is not None and any(
        rt in ("fol", "format") for rt in step_reward_types
    )
    return get_split_fn(use_xml=use_xml)
