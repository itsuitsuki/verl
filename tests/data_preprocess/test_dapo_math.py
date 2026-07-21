import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PREPROCESS_DIR = (Path(__file__).resolve().parents[2] / "examples" /
                   "data_preprocess")
sys.path.insert(0, str(_PREPROCESS_DIR))


def _load_with_fake_dependencies():
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = None
    shared = types.ModuleType("math_rl_data")
    shared.load_prompt_file = None
    shared.process_records = None
    old_datasets = sys.modules.get("datasets")
    old_shared = sys.modules.get("math_rl_data")
    sys.modules["datasets"] = datasets
    sys.modules["math_rl_data"] = shared
    try:
        path = _PREPROCESS_DIR / "dapo_math.py"
        spec = importlib.util.spec_from_file_location("dapo_math", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_datasets is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = old_datasets
        if old_shared is None:
            sys.modules.pop("math_rl_data", None)
        else:
            sys.modules["math_rl_data"] = old_shared


dapo_math = _load_with_fake_dependencies()


@pytest.mark.parametrize("config", dapo_math.DATASET_CONFIGS)
def test_selected_hub_config_controls_source_and_report(monkeypatch, tmp_path, config):
    dataset = [{
        "prompt": "What is 1 + 1?",
        "solution": "2",
        "data_source": "math_dapo",
    }]
    calls = {}

    def fake_load_dataset(dataset_id, selected_config, *, split):
        calls["load"] = (dataset_id, selected_config, split)
        return dataset

    def fake_process_records(**kwargs):
        calls["process"] = kwargs

    monkeypatch.setattr(dapo_math, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(dapo_math, "load_prompt_file", lambda *args: "prompt")
    monkeypatch.setattr(dapo_math, "process_records", fake_process_records)
    monkeypatch.setattr(sys, "argv", [
        "dapo_math.py",
        "--config", config,
        "--local_save_dir", str(tmp_path),
    ])

    dapo_math.main()

    assert calls["load"] == (dapo_math.DATASET_ID, config, "train")
    process = calls["process"]
    assert process["data_source"] == dapo_math.DATA_SOURCE
    assert process["source_description"] == f"{dapo_math.DATASET_ID}:{config}"
    assert process["stage_counts"] == {"raw": 1, "dataset_config": config}


def test_training_data_source_matches_reward_router():
    assert dapo_math.DATA_SOURCE == "open-r1/DAPO-Math-17k-Processed"
    assert "/en" not in dapo_math.DATA_SOURCE
    assert "/cn" not in dapo_math.DATA_SOURCE


def test_default_config_remains_english(monkeypatch, tmp_path):
    calls = {}

    def fake_load_dataset(dataset_id, selected_config, *, split):
        calls["load"] = (dataset_id, selected_config, split)
        return []

    monkeypatch.setattr(dapo_math, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(dapo_math, "load_prompt_file", lambda *args: None)
    monkeypatch.setattr(dapo_math, "process_records", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "dapo_math.py",
        "--local_save_dir", str(tmp_path),
    ])

    dapo_math.main()

    assert calls["load"] == (dapo_math.DATASET_ID, "en", "train")
