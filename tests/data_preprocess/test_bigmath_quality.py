import importlib.util
import json
import sys
import types
from pathlib import Path

_PREPROCESS_DIR = (Path(__file__).resolve().parents[2] / "examples" /
                   "data_preprocess")
sys.path.insert(0, str(_PREPROCESS_DIR))


def _load(name):
    path = _PREPROCESS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


quality = _load("bigmath_quality")


def _load_shared_without_optional_io_dependencies():
    datasets = types.ModuleType("datasets")
    datasets.Dataset = object
    pandas = types.ModuleType("pandas")
    old_datasets = sys.modules.get("datasets")
    old_pandas = sys.modules.get("pandas")
    sys.modules["datasets"] = datasets
    sys.modules["pandas"] = pandas
    try:
        return _load("math_rl_data")
    finally:
        if old_datasets is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = old_datasets
        if old_pandas is None:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = old_pandas


shared = _load_shared_without_optional_io_dependencies()


def test_flags_non_math_prompt_answer_mismatch():
    flags = quality.row_flags(
        "It is not allowed to ride a bicycle together.",
        r"2 \frac{2}{3} hours",
        "olympiads",
    )
    assert "non_math_prompt_with_math_answer" in flags


def test_chinese_math_prompt_skips_english_non_math_heuristic():
    assert "non_math_prompt_with_math_answer" not in quality.row_flags(
        "将两个相同的白球放入三个袋子，求没有空袋的放法数。",
        "723",
        "math_dapo",
    )


def test_repairs_only_contextually_identified_control_characters():
    cases = {
        "value \x0crac{1}{2}": r"value \frac{1}{2}",
        "angle = \x08eta": r"angle = \beta",
        "de\x0cne": "define",
        "satis\x0ces": "satisfies",
        "satis\x0cfies": "satisfies",
        "in\x0cfinite": "infinite",
        "\x0crst": "first",
        "\x0ctting together": "fitting together",
        "\x0cnd the distance": "find the distance",
        "\x0cnd the area": "find the area",
        "\x0cfirst": "first",
        "\x0cfive": "five",
        "\x0cfigure": "figure",
        "\x0cfinds": "finds",
        "fi\x0cve": "five",
        "\x0cfind $p(10)$": "find $p(10)$",
        "\x0cequal 2020": "equal 2020",
        "2 | \x0cx^2 - 9 |": "2 | x^2 - 9 |",
        "a \\le b \x14 \\le c": r"a \le b  \le c",
        "a \x01 \\cdot b": r"a  \cdot b",
        "\x0c\x0cand": "and",
        "\x0c\x0cequal": "equal",
        "\x0cand": "and",
        "di\x0bfference": "difference",
        "x \x14\\le y": r"x \le y",
        "x \\le\x14 y": r"x \le y",
        "\\cdot \x01 c": r"\cdot  c",
        "\x12\\{1,2\\}": r"\{1,2\}",
        "\x12(x+1)": "(x+1)",
        "180^\\circ\x0e with": r"180^\circ with",
        "$\\star$ \x88 $x \\le 2y$": r"$\star$ $x \le 2y$",
    }
    for source, expected in cases.items():
        assert quality.repair_unambiguous_encoding(source) == expected


def test_repairs_exact_source_latex_aliases_and_literal_newlines():
    cases = {
        r"\df{a}{b}": r"\frac{a}{b}",
        r"\bR \bN \bC \bZ \bQ": r"\mathbb{R} \mathbb{N} \mathbb{C} \mathbb{Z} \mathbb{Q}",
        r"\dd x": r"\,\mathrm{d} x",
        r"a-\sqrtb": r"a-\sqrt{b}",
        r"\ frac{a}{b} + \ sqrt{c}": r"\frac{a}{b} + \sqrt{c}",
        r"text\n\n$f(x)$": "text\n\n$f(x)$",
        r"table\n| a | b |\n| 1 | 2 |": "table\n| a | b |\n| 1 | 2 |",
        r"given\n$$\nx=1\n$$\nanswer": "given\n$$\nx=1\n$$\nanswer",
    }
    for source, expected in cases.items():
        assert quality.repair_source_latex(source) == expected


def test_flags_malformed_math_delimiters_without_flagging_currency():
    assert "malformed_math_delimiters" in quality.row_flags(
        "则 $x=1。", "1", "math_dapo"
    )
    assert "malformed_math_delimiters" in quality.row_flags(
        r"满足\(1) $x=1$。", "1", "math_dapo"
    )
    assert "malformed_math_delimiters" not in quality.row_flags(
        "A ticket costs $5.", "5", "math_dapo"
    )
    assert "malformed_math_delimiters" not in quality.row_flags(
        r"$a=1$ and \(b=2\).", "3", "math_dapo"
    )

    assert quality.repair_unambiguous_encoding("word \x0czebra") == "word \x0czebra"
    assert "bad_encoding" in quality.row_flags("word \x0czebra", "4", "test")
    assert "bad_encoding" in quality.row_flags("word \x88zebra", "4", "test")


def test_repaired_record_keeps_other_control_characters():
    record = {"prompt": "\x0crac{1}{2}", "answer": "4", "source_index": 1}
    fixed = quality.repaired_record(record)
    assert fixed["prompt"] == r"\frac{1}{2}"
    assert fixed["answer"] == "4"




def test_strips_only_exact_embedded_chinese_response_instruction_suffix():
    instruction = (
        "让我们一步一步地思考。请以“Answer: \\boxed{<final_answer>}”的格式输出最终答案。"
        "如果是选择题，请按顺序输出正确的选项，不带任何标点或空格。"
        "对于其他类型的问题，请只输出最终答案的数值。"
    )
    record = {
        "source_index": 1,
        "prompt": "求 $1+1$。\n" + instruction,
        "answer": "2",
    }
    assert quality.repaired_record(record)["prompt"] == "求 $1+1$。"
    assert quality.strip_embedded_response_instruction(
        "题目中提到让我们一步一步地思考，但后缀并不完整。"
    ) == "题目中提到让我们一步一步地思考，但后缀并不完整。"


def test_flags_forum_administration_without_banning_forum_math():
    admin = ("Register for the contest using your email address. Rename the "
             "PDF before upload and ask questions in private messages.")
    assert "forum_admin" in quality.row_flags(admin, "p = 3", "aops_forum")
    math = "Find the largest integer n such that n^2 < 100."
    assert "forum_admin" not in quality.row_flags(math, "9", "aops_forum")


def test_distinguishes_truncation_from_sequence_ellipsis():
    assert "truncated_prompt" in quality.row_flags(
        "What is the probability that...", r"\frac{3}{181}", "olympiads")
    assert "truncated_prompt" not in quality.row_flags(
        "Given 1, 3, 7, 15, ... find the general formula.",
        r"2^n-1", "big_math")


def test_flags_missing_figure_reference():
    flags = quality.row_flags(
        "As shown in Figure 7, find the number of paths.", "28", "olympiads")
    assert "missing_figure_reference" in flags


def test_flags_control_character_but_allows_whitespace():
    assert "bad_encoding" in quality.row_flags(
        "Find \ffrac{1}{2} of 8.", "4", "cn_k12")
    assert "bad_encoding" not in quality.row_flags(
        "Find\n\t1/2 of 8.", "4", "cn_k12")


def test_flags_all_rows_in_true_answer_conflict():
    records = [
        {"source_index": 1, "prompt": "How many socks?", "answer": "180"},
        {"source_index": 2, "prompt": "How many socks?", "answer": "90"},
    ]
    flags = quality.build_group_flags(records)
    assert flags[1] == ["conflicting_answers"]
    assert flags[2] == ["conflicting_answers"]


def test_deduplicates_equivalent_answer_only():
    records = [
        {"source_index": 1, "prompt": "What percent?", "answer": "25%"},
        {"source_index": 2, "prompt": " What percent? ", "answer": "= 25%"},
    ]
    flags = quality.build_group_flags(records)
    assert 1 not in flags
    assert flags[2] == ["duplicate_prompt_equivalent_answer"]


def test_low_solve_rate_is_not_a_quality_input():
    assert quality.row_flags(
        "Find all primes p such that p + 2 is prime.", "3", "harp") == []


def test_shared_verl_schema_matches_existing_math_files():
    record = {"prompt": "What is 1 + 1?", "answer": "2"}
    row = shared.make_verl_row(record, 0, "test", None, "Answer in boxed form.")
    assert set(row) == {"data_source", "prompt", "ability", "reward_model",
                        "extra_info"}
    assert set(row["extra_info"]) == {
        "split", "index", "answer", "question", "math_question",
        "math_solution", "math_final_answer", "fol_context", "fol_question",
        "fol_options",
    }


def test_processing_uses_repaired_records_before_grouping(monkeypatch, tmp_path):
    captured = {}

    def fake_write(records, path):
        captured[path.name] = records

    monkeypatch.setattr(shared, "write_dataset", fake_write)
    monkeypatch.setattr(shared, "sha256", lambda path: "test-hash")
    shared.process_records(
        records=[
            {"source_index": 1, "prompt": "Find \x0crac{1}{2}.", "answer": "2", "source": "test"},
            {"source_index": 2, "prompt": r"Find \frac{1}{2}.", "answer": "2", "source": "test"},
        ],
        save_dir=tmp_path,
        data_source="test",
        source_description="test",
        system_prompt=None,
        instruction="test",
    )
    assert len(captured["train.parquet"]) == 1
    assert captured["train.parquet"][0]["extra_info"]["math_question"] == r"Find \frac{1}{2}."
    assert captured["quarantine.parquet"][0]["reason"] == "duplicate_prompt_equivalent_answer"
    report = json.loads((tmp_path / "quality_report.json").read_text())
    assert report["counts"]["clean_output"] == 1
    assert report["counts"]["quarantine"] == 1


def test_stripping_embedded_instruction_precedes_duplicate_grouping(monkeypatch, tmp_path):
    captured = {}
    instruction = (
        "让我们一步一步地思考。请以“Answer: \\boxed{<final_answer>}”的格式输出最终答案。"
        "如果是选择题，请按顺序输出正确的选项，不带任何标点或空格。"
        "对于其他类型的问题，请只输出最终答案的数值。"
    )

    monkeypatch.setattr(
        shared, "write_dataset",
        lambda records, path: captured.__setitem__(path.name, records),
    )
    monkeypatch.setattr(shared, "sha256", lambda path: "test-hash")
    shared.process_records(
        records=[
            {"source_index": 1, "prompt": "求 $1+1$。\n" + instruction, "answer": "2", "source": "test"},
            {"source_index": 2, "prompt": "求 $1+1$。", "answer": "2", "source": "test"},
        ],
        save_dir=tmp_path,
        data_source="test",
        source_description="test",
        system_prompt=None,
        instruction="test",
    )
    assert len(captured["train.parquet"]) == 1
    assert captured["quarantine.parquet"][0]["reason"] == "duplicate_prompt_equivalent_answer"
