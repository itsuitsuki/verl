import importlib.util
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
