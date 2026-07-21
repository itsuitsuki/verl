"""Deterministic quality checks for mathematical training snapshots.

The checks are intentionally conservative. They remove records that cannot be
used reliably as a self-contained math training example and quarantine records
that need semantic review. Solve rate is not a quality label.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import defaultdict

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_REPLACEMENT_RE = re.compile("�")
_EMBEDDED_RESPONSE_INSTRUCTION_RE = re.compile(
    r"\s*让我们一步一步地思考。请以[“\"]Answer:\s*\\boxed\{<final_answer>\}[”\"]"
    r"的格式输出最终答案。如果是选择题，请按顺序输出正确的选项，不带任何标点或空格。"
    r"对于其他类型的问题，请只输出最终答案的数值。\s*$"
)
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_REPAIRABLE_FORM_FEED_FRAC_RE = re.compile("\x0crac(?=[{A-Za-z0-9+-])")
_SPACED_LATEX_COMMAND_RE = re.compile(r"\\\s+(frac|sqrt)\b")
_LITERAL_NEWLINE_RE = re.compile(
    r"\\n(?=(?:\\n|\\[\[$]|\\(?:left|right|begin|end|text)|[A-Za-z]\s*[_(=]|answer\b|\$|\s|\||[^\x00-\x7f]))",
    re.I,
)
_SOURCE_LATEX_REPAIRS = (
    (re.compile(r"\\df(?=\s*\{)"), lambda _: r"\frac"),
    (re.compile(r"\\dd(?=\s*[A-Za-z])"), lambda _: r"\,\mathrm{d}"),
    (re.compile(r"\\b([RNCZQ])\b"), lambda match: rf"\mathbb{{{match.group(1)}}}"),
    (re.compile(r"\\sqrtb\b"), lambda _: r"\sqrt{b}"),
)
_REPAIRABLE_CONTEXT_REPAIRS = (
    (re.compile("\x08eta"), r"\\beta"),
    # PDF ligature damage sometimes leaves the visible ``fi`` before the
    # control character. Delete that character before reconstructing cases
    # where the entire ligature disappeared.
    (re.compile("fi\x0c"), "fi"),
    (re.compile("satis\x0ces"), "satisfies"),
    (re.compile("satis\x0cfies"), "satisfies"),
    (re.compile("de\x0cne"), "define"),
    (re.compile("in\x0cfinite"), "infinite"),
    (re.compile("\x0crst"), "first"),
    (re.compile("\x0ctting"), "fitting"),
    (re.compile("\x0cnd"), "find"),
    (re.compile("\x0cfirst"), "first"),
    (re.compile("\x0cfive"), "five"),
    (re.compile("\x0cfigure"), "figure"),
    (re.compile("\x0cfinds"), "finds"),
    (re.compile("\x0c+(?=and\\b)"), ""),
    (re.compile("\x0cfind(?=\\s)"), "find"),
    (re.compile("\x0c+(?=equal\\b)"), ""),
    (re.compile("(?<=\\s)\x0c(?=x\\^2)"), ""),
    (re.compile("(?<=\\s)\x14(?=\\s+\\\\le)"), ""),
    (re.compile("(?<=\\s)\x01(?=\\s+\\\\cdot)"), ""),
    (re.compile("\x0c+(?=[\\s$|\\\\])"), ""),
    (re.compile("di\x0bfference"), "difference"),
    (re.compile("\x14(?=\\\\le)"), ""),
    (re.compile("(?<=\\\\le)\x14"), ""),
    (re.compile("\x01(?=\\s*c\\b)"), ""),
    (re.compile("\x12(?=\\\\[({])"), ""),
    (re.compile("\x12(?=\\()"), ""),
    (re.compile("\x0e(?=\\s+with\\b)"), ""),
    (re.compile("(?<=\\\\star\\$)\\s*\x88(?=\\s+\\$)"), ""),
)
_HTML_MATH_RE = re.compile(r"</?(?:br|sup|sub|span)(?:\s+[^>]*)?>", re.I)
_IMAGE_RE = re.compile(
    r"\b(?:as shown in (?:the )?(?:figure|diagram)|"
    r"(?:in|from|according to) (?:the )?(?:figure|diagram)(?: below| above)?|"
    r"see (?:the )?(?:figure|diagram)|(?:figure|diagram) below)\b",
    re.I,
)
_MATH_SIGNAL_RE = re.compile(
    r"(?:\\frac|\\sqrt|\\sum|\\int|\\infty|[=+*/^<>]|"
    r"\b(?:calculate|compute|determine|equation|expression|function|"
    r"find|how many|how much|integer|number|probability|ratio|solve|"
    r"triangle|circle|angle|area|volume|sequence|polynomial|prime|"
    r"permutation|combination|proof|theorem)\b|\d)",
    re.I,
)
_ANSWER_MATH_RE = re.compile(
    r"(?:\\frac|\\sqrt|\\sum|\\int|\\infty|\d|"
    r"\b(?:hours?|minutes?|percent|degrees?|cups?|miles?|km|kg)\b)",
    re.I,
)
_META_ANSWER_RE = re.compile(
    r"^(?:solution is awarded\b|reached solution within\b|"
    r"answer is awarded\b)",
    re.I,
)
_TRUNCATED_END_RE = re.compile(
    r"\b(?:what|which|where|when|who|how|find|determine|calculate|"
    r"probability|number|value|angular|equation|the|that|of|for|with|"
    r"and|then|is|are|can|does)\s*$",
    re.I,
)


def repair_unambiguous_encoding(value: object) -> str:
    """Repair only control-character corruption identified by exact context."""
    text = _REPAIRABLE_FORM_FEED_FRAC_RE.sub(r"\\frac", str(value or ""))
    for pattern, replacement in _REPAIRABLE_CONTEXT_REPAIRS:
        text = pattern.sub(replacement, text)
    return text


def repair_source_latex(value: object) -> str:
    """Normalize exact source-specific LaTeX aliases and escaped newlines."""
    text = _SPACED_LATEX_COMMAND_RE.sub(
        lambda match: "\\" + match.group(1), str(value or "")
    )
    previous = None
    while text != previous:
        previous = text
        text = _LITERAL_NEWLINE_RE.sub("\n", text)
    for pattern, replacement in _SOURCE_LATEX_REPAIRS:
        text = pattern.sub(replacement, text)
    return text


def strip_embedded_response_instruction(value: object) -> str:
    """Strip the exact source-level response instruction from a prompt suffix."""
    return _EMBEDDED_RESPONSE_INSTRUCTION_RE.sub("", str(value or "")).rstrip()


def repaired_record(record: dict) -> dict:
    """Return a copy with only deterministic prompt and answer repairs."""
    repaired_prompt = repair_source_latex(
        repair_unambiguous_encoding(record.get("prompt"))
    )
    return {
        **record,
        "prompt": strip_embedded_response_instruction(repaired_prompt),
        "answer": repair_source_latex(
            repair_unambiguous_encoding(record.get("answer"))
        ),
    }


def normalize_text(value: object) -> str:
    """Normalize text for grouping without changing mathematical content."""
    text = html.unescape(unicodedata.normalize("NFKC", str(value or "")))
    text = _HTML_MATH_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_answer(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"^\s*=\s*", "", text)
    return text.replace(r"\!", "").strip()


def has_bad_encoding(prompt: object, answer: object) -> bool:
    values = (str(prompt or ""), str(answer or ""))
    return any(_CONTROL_RE.search(value) or _REPLACEMENT_RE.search(value)
               for value in values)


def contains_cjk(prompt: object) -> bool:
    return bool(_CJK_RE.search(str(prompt or "")))


def has_malformed_math_delimiters(prompt: object) -> bool:
    """Detect unbalanced explicit math delimiters in non-currency prompts."""
    text = str(prompt or "")
    if contains_cjk(text) and text.count("$") % 2:
        return True
    return text.count(r"\(") != text.count(r"\)")


def looks_truncated(prompt: object) -> bool:
    text = normalize_text(prompt)
    if not re.search(r"(?:\.\.\.|…)$", text):
        return False
    body = re.sub(r"(?:\.\.\.|…)$", "", text).rstrip()
    # `a_n =...` and sequences ending in ellipsis are valid notation.
    if re.search(r"(?:=|\\dots|\\ldots|\bdots)\s*$", body):
        return False
    return bool(_TRUNCATED_END_RE.search(body))


def is_forum_admin(prompt: object, source: object) -> bool:
    if str(source or "").casefold() not in {"aops_forum", "oliforum"}:
        return False
    text = normalize_text(prompt)
    categories = (
        r"\b(?:email address|e-mail)\b",
        r"\b(?:rename the pdf|private messages?|upload)\b",
        r"\b(?:register|registration|enrollment|contest rules?)\b",
    )
    hits = sum(bool(re.search(pattern, text, re.I)) for pattern in categories)
    math_request = re.search(
        r"\b(?:find|calculate|determine|what is|how many|probability|"
        r"equation|expression|function|solve)\b",
        text,
        re.I,
    )
    return hits >= 2 and not math_request


def is_non_math_prompt_with_math_answer(prompt: object, answer: object) -> bool:
    text = normalize_text(prompt)
    # This signal vocabulary is English-only. Chinese or mixed-language prompts
    # must not be rejected merely because English keywords are absent.
    if contains_cjk(prompt):
        return False
    return (not _MATH_SIGNAL_RE.search(text)
            and "?" not in text
            and bool(_ANSWER_MATH_RE.search(normalize_text(answer))))


def references_missing_figure(prompt: object) -> bool:
    return bool(_IMAGE_RE.search(normalize_text(prompt)))


def is_meta_answer(answer: object) -> bool:
    return bool(_META_ANSWER_RE.match(normalize_text(answer)))


def row_flags(prompt: object, answer: object, source: object) -> list[str]:
    flags = []
    if not normalize_text(prompt) or not normalize_text(answer):
        flags.append("empty_prompt_or_answer")
    if has_bad_encoding(prompt, answer):
        flags.append("bad_encoding")
    if has_malformed_math_delimiters(prompt):
        flags.append("malformed_math_delimiters")
    if looks_truncated(prompt):
        flags.append("truncated_prompt")
    if is_forum_admin(prompt, source):
        flags.append("forum_admin")
    if is_non_math_prompt_with_math_answer(prompt, answer):
        flags.append("non_math_prompt_with_math_answer")
    if references_missing_figure(prompt):
        flags.append("missing_figure_reference")
    if is_meta_answer(answer):
        flags.append("meta_answer")
    if "this problem was not fully correct" in normalize_text(prompt):
        flags.append("explicit_invalid_problem")
    return flags


def answers_equivalent(left: object, right: object) -> bool:
    """Return whether two answer strings represent the same math object."""
    if normalize_answer(left) == normalize_answer(right):
        return True
    try:
        from math_verify import parse, verify

        gold = parse(str(left), extraction_mode="first_match")
        pred = parse(str(right), extraction_mode="first_match")
        return bool(gold and pred and any(
            verify(g, p, timeout_seconds=2) for g in gold for p in pred
        ))
    except Exception:
        return False


def build_group_flags(records: list[dict]) -> dict[int, list[str]]:
    """Flag semantic duplicates and all rows in true answer conflicts."""
    prompt_groups = defaultdict(list)
    for record in records:
        prompt_groups[normalize_text(record["prompt"])].append(record)

    flags = defaultdict(list)
    for group in prompt_groups.values():
        clusters = []
        for record in group:
            for cluster in clusters:
                if answers_equivalent(record["answer"], cluster[0]["answer"]):
                    cluster.append(record)
                    break
            else:
                clusters.append([record])

        if len(clusters) > 1:
            for record in group:
                flags[record["source_index"]].append("conflicting_answers")
            continue

        for duplicate in clusters[0][1:]:
            flags[duplicate["source_index"]].append(
                "duplicate_prompt_equivalent_answer"
            )

    return dict(flags)


def clean_record(record: dict, group_flags: dict[int, list[str]]) -> list[str]:
    flags = row_flags(record["prompt"], record["answer"], record["source"])
    flags.extend(group_flags.get(record["source_index"], []))
    return sorted(set(flags))


# Missing figures are excluded from train because the current parquet and
# rollout pipeline do not provide images. They remain intact in quarantine.
_TRAIN_EXCLUSION_FLAGS = {
    "empty_prompt_or_answer",
    "bad_encoding",
    "malformed_math_delimiters",
    "truncated_prompt",
    "forum_admin",
    "non_math_prompt_with_math_answer",
    "meta_answer",
    "explicit_invalid_problem",
    "missing_figure_reference",
    "conflicting_answers",
    "duplicate_prompt_equivalent_answer",
}


def exclusion_flags(flags: list[str]) -> list[str]:
    return [flag for flag in flags if flag in _TRAIN_EXCLUSION_FLAGS]
