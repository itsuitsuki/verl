"""Manager-level integration tests for the unified bad-format rules.

Runs StepRewardManager.run_single end to end with a character tokenizer and stubbed
backends: the per-step FOL path (histories and per-step penalties), the whole-response
Isabelle path (reward remapping onto the original token positions, the cleaned verifier
response, the preserved answer), the response-level stray-tag cases, and the boxed outcome
rules. TreeManager is exercised through its REAL node-building path: initialize_trees and
commit_branches run on hand-built DataProto batches (no generation needed), so the XML
splitting, the terminal answer tail node, the branch format marking, evaluate_leaves, and
build_flat_batch behave exactly as in training.
"""
import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace

import verl.utils.reward_score.formal_verify as formal_verify
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.step import StepRewardManager
from verl.utils.step_splitter import split_by_xml_step_tags
from verl.utils.tree_structure import TreeManager

VALID1 = "<step><premise>a</premise><conclusion>b</conclusion></step>"
INVALID = "<step><premise>q</premise></step>"
VALID2 = "<step><premise>c</premise><conclusion>d</conclusion></step>"

PROMPT_TOKENS = 3
PAD = 6


class CharacterTokenizer:
    """One token per character, so token positions equal character positions."""

    eos_token_id = None
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _data_proto(response_text):
    ids = [ord(char) for char in response_text]
    responses = torch.tensor([ids + [0] * PAD], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1] * PROMPT_TOKENS + [1] * len(ids) + [0] * PAD], dtype=torch.long)
    return DataProto.from_dict(
        tensors={"responses": responses, "attention_mask": attention_mask},
        non_tensors={
            "data_source": ["mock"],
            "reward_model": [{"ground_truth": "7"}],
            "extra_info": [{"answer": "7"}],
            "raw_prompt": [[{"role": "user", "content": "problem text"}]],
        },
    )


def _manager(compute_score, step_reward_type="random", step_reward_fns=None, api_config=None):
    reward_cfg = {
        "step_reward_type": step_reward_type,
        "use_xml_steps": True,
        "penalty_on_truncated": True,
        "penalty_on_multi_boxed": True,
        "penalty_on_bad_format": True,
        "penalty_score": -1.0,
        "random_reward_seed": 11,
    }
    if api_config is not None:
        reward_cfg["api_config"] = api_config
    config = {"reward": reward_cfg, "algorithm": {}}
    return StepRewardManager(
        config, CharacterTokenizer(), compute_score,
        step_reward_type=step_reward_type, step_reward_fns=step_reward_fns)


def _math_api_config():
    return {
        "fol_task_type": "math",
        "model": "Qwen3.6-35B-A3B",
        "base_url": "http://127.0.0.1:4873/v1",
    }


def test_math_isabelle_requires_explicit_translator_config():
    with pytest.raises(ValueError, match="model, base_url"):
        _manager(lambda **kw: 1.0, step_reward_type="fol",
                 api_config={"fol_task_type": "math"})


def _run(manager, data):
    return manager.loop.run_until_complete(manager.run_single(data))


def _step_end_positions(response_text):
    """Token position of each step's last character under the character tokenizer."""
    return [end - 1 for _, _, end in split_by_xml_step_tags(response_text)]


def test_fol_integration_penalizes_only_the_invalid_step(monkeypatch):
    calls = []

    def fol_stub(step_text, prompt_text, step_history, **kwargs):
        calls.append((step_text, list(step_history)))
        return {"score": 1.0, "debug": {}}

    monkeypatch.setattr(formal_verify, "prepare_fol_shared_state", lambda *a, **k: None)
    manager = _manager(lambda **kw: 0.0, step_reward_type="fol",
                       step_reward_fns={"fol": fol_stub},
                       api_config={"fol_task_type": "logic"})
    response = VALID1 + INVALID + VALID2 + " \\boxed{7}"
    out = _run(manager, _data_proto(response))

    rewards = dict(out["reward_extra_info"]["fol_step_reward"])
    first, bad, second = _step_end_positions(response)
    assert rewards[bad] == -1.0
    assert rewards[first] == 1.0 and rewards[second] == 1.0
    # the later valid step's history contains the valid prefix only, never the invalid step
    assert [history for _, history in calls] == [[VALID1], [VALID1, VALID2]]
    assert "bad_step_format@1:no_conclusion" in out["reward_extra_info"]["penalty_reason"]


def test_isabelle_rewards_map_to_original_positions_and_response_is_cleaned(monkeypatch):
    seen = {}

    def isabelle_stub(problem, response, ground_truth, **kwargs):
        seen["response"] = response
        return [1.0, 0.0], {}

    monkeypatch.setattr(formal_verify, "compute_solution_reward_isabelle", isabelle_stub)
    manager = _manager(lambda **kw: 1.0, step_reward_type="fol",
                       api_config=_math_api_config())
    response = VALID1 + INVALID + VALID2 + " \\boxed{7}"
    out = _run(manager, _data_proto(response))

    rewards = dict(out["reward_extra_info"]["fol_step_reward"])
    first, bad, second = _step_end_positions(response)
    assert rewards[first] == 1.0      # first engine value lands on the first valid step
    assert rewards[bad] == -1.0       # the invalid step keeps its own penalty
    assert rewards[second] == 0.0     # second engine value lands on the second valid step
    assert INVALID not in seen["response"]
    assert VALID1 in seen["response"] and "\\boxed{7}" in seen["response"]


def test_isabelle_profile_metrics_use_training_batch_keys(monkeypatch):
    debug = {
        "judge_http_wall_time": 1.25,
        "translate_validate_wall_time": 2.5,
        "prove_queue_time": 3.75,
        "prove_run_time": 4.5,
        "reward_wall_time": 5.25,
        "external_solver_reaps": 6,
    }

    def isabelle_stub(problem, response, ground_truth, **kwargs):
        return [1.0], debug

    monkeypatch.setattr(formal_verify, "compute_solution_reward_isabelle", isabelle_stub)
    manager = _manager(lambda **kw: 1.0, step_reward_type="fol",
                       api_config=_math_api_config())
    out = _run(manager, _data_proto(VALID1 + " \\boxed{7}"))
    extra = out["reward_extra_info"]
    assert extra["isabelle_judge_http_wall_time"] == 1.25
    assert extra["isabelle_translate_validate_wall_time"] == 2.5
    assert extra["isabelle_prove_queue_time"] == 3.75
    assert extra["isabelle_prove_run_time"] == 4.5
    assert extra["isabelle_reward_wall_time"] == 5.25
    assert extra["isabelle_external_solver_reaps"] == 6


def test_isabelle_valid_steps_survive_a_boxed_swallowing_unclosed_step(monkeypatch):
    seen = {}

    def isabelle_stub(problem, response, ground_truth, **kwargs):
        seen["response"] = response
        return [1.0], {}

    monkeypatch.setattr(formal_verify, "compute_solution_reward_isabelle", isabelle_stub)
    manager = _manager(lambda **kw: 1.0, step_reward_type="fol",
                       api_config=_math_api_config())
    response = VALID1 + "\n<step>\n<premise>r</premise>\n<conclusion>d</conclusion>\n\\boxed{7}"
    out = _run(manager, _data_proto(response))

    rewards = dict(out["reward_extra_info"]["fol_step_reward"])
    first, unclosed = _step_end_positions(response)
    assert rewards[first] == 1.0      # the earlier valid step is NOT zeroed
    assert rewards[unclosed] == -1.0
    assert "\\boxed{7}" in seen["response"]
    assert "<premise>r</premise>" not in seen["response"]


def test_stray_tags_and_no_step_responses():
    manager = _manager(lambda **kw: 0.5, step_reward_type="random")
    for stray in ("<conclusion>x</conclusion>", "<premise>p</premise>", "</step>"):
        response = VALID1 + stray + " \\boxed{7}"
        out = _run(manager, _data_proto(response))
        entries = out["reward_extra_info"]["random_step_reward"]
        assert sum(1 for _, score in entries if score == -1.0) == 1, stray
        (step_pos,) = _step_end_positions(response)
        assert dict(entries)[step_pos] != -1.0
        assert "stray_xml_tags=" in out["reward_extra_info"]["penalty_reason"]

    out = _run(manager, _data_proto("just text \\boxed{3}"))
    response_len = len("just text \\boxed{3}")
    assert out["reward_extra_info"]["random_step_reward"] == [(response_len - 1, -1.0)]
    assert out["reward_extra_info"]["num_steps"] == 0
    assert "bad_format(no_xml_step)" in out["reward_extra_info"]["penalty_reason"]


def test_boxed_outcome_rules():
    graded = []

    def scorer(**kwargs):
        graded.append(kwargs["solution_str"])
        return {"score": 1.0}

    manager = _manager(scorer, step_reward_type="random")

    out = _run(manager, _data_proto(VALID1 + " \\boxed{7}"))
    assert out["reward_score"] == 1.0 and len(graded) == 1

    graded.clear()
    out = _run(manager, _data_proto(VALID1 + VALID2))
    # zero \boxed forfeits the outcome with penalty_score (the decided rule; the original
    # design note said 0.0 here, flag any change of mind before editing this pin)
    assert out["reward_score"] == -1.0
    assert len(graded) == 1

    graded.clear()
    response = VALID1 + " \\boxed{7} tail \\boxed{8}"
    out = _run(manager, _data_proto(response))
    assert len(graded) == 2
    assert graded[1].endswith("\\boxed{7}")
    assert out["reward_score"] == 1.0
    # each \boxed after the committed first one costs one penalty_score at its own position
    entries = dict(out["reward_extra_info"]["random_step_reward"])
    assert entries[len(response) - 1] == -1.0


ANSWER7 = "\n\\boxed{7}"


def _tree_manager(**overrides):
    kwargs = dict(use_xml=True, penalty_on_multi_boxed=True,
                  penalty_on_bad_format=True, penalty_score=-1.0)
    kwargs.update(overrides)
    return TreeManager({}, CharacterTokenizer(), **kwargs)


def _rollout_proto(response_text):
    prompt_ids = [ord(char) for char in "P>"]
    response_ids = [ord(char) for char in response_text]
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([prompt_ids], dtype=torch.long),
            "responses": torch.tensor([response_ids], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1] * (len(prompt_ids) + len(response_ids))], dtype=torch.long),
        },
        non_tensors={
            "data_source": ["mock"],
            "reward_model": [{"ground_truth": "7"}],
            "extra_info": [{}],
        },
    )


def _branch_proto(branch_text, prefix_len=4):
    ids = [ord(char) for char in branch_text]
    return DataProto.from_dict(tensors={
        "responses": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * (prefix_len + len(ids))], dtype=torch.long),
    })


def _fork_info(node):
    return [{"tree_idx": node.tree_idx, "fork_node_id": node.node_id, "fork_token_idx": 0}]


def test_tree_initial_chain_keeps_the_answer_tail_in_a_node():
    """The XML split leaves the text after the last </step> outside every step block; that tail (the boxed answer, EOS) must still land in a node, because everything that rebuilds a path from node token_ids (build_flat_batch training rows, full_token_ids outcome grading) only sees tokens that live in some node."""
    manager = _tree_manager()
    response = VALID1 + VALID2 + ANSWER7
    manager.initialize_trees(_rollout_proto(response))
    (leaf,) = manager.trees[0].all_leaves
    assert manager.tokenizer.decode(leaf.full_token_ids()) == response
    path = leaf.path_from_root()
    assert [node.step_text for node in path] == [VALID1, VALID2, ANSWER7]
    assert [node.process_rewardable for node in path] == [True, True, False]


def test_tree_branch_real_split_grades_outcome_on_the_tail_answer():
    manager = _tree_manager()
    manager.initialize_trees(_rollout_proto(VALID1 + "\n\\boxed{3}"))
    fork = manager.trees[0].all_nodes[0]          # the VALID1 step node
    new_nodes = manager.commit_branches(_branch_proto(VALID2 + ANSWER7), _fork_info(fork))
    assert [node.step_text for node in new_nodes] == [VALID2, ANSWER7]
    assert [node.process_rewardable for node in new_nodes] == [True, False]
    assert not any(node.ext_prm_penalty for node in new_nodes)   # canonical branch: no marks

    manager.evaluate_leaves(
        lambda **kwargs: 1.0 if "\\boxed{7}" in kwargs["solution_str"] else 0.0)
    assert new_nodes[-1].correctness == 1.0       # the branch outcome sees its own answer
    (initial_leaf,) = [n for n in manager.trees[0].all_leaves if not n.is_forked]
    assert initial_leaf.correctness == 0.0        # \boxed{3} present, graded normally


def test_tree_zero_boxed_leaf_outcome_is_penalty_score():
    """A path without any \\boxed forfeits its outcome as penalty_score whatever the dataset scorer would extract from the plain text (same rule as the step manager, so all three paths agree)."""
    manager = _tree_manager()
    manager.initialize_trees(_rollout_proto(VALID1 + VALID2))
    called = []
    manager.evaluate_leaves(lambda **kwargs: called.append(1) or 0.5)
    (leaf,) = manager.trees[0].all_leaves
    assert leaf.correctness == -1.0
    assert called == []                           # the scorer never runs for a forfeited path
    # no phantom tail node when the response ends exactly at </step>
    assert [node.step_text for node in leaf.path_from_root()] == [VALID1, VALID2]


def test_tree_branch_extra_boxed_real_split_marks_the_later_step():
    """Branch boxed rule on real splitting: the FIRST \\boxed on the branch is the committed answer, a LATER \\boxed inside a step costs that node its score, and evaluate_leaves grades the path on the prefix ending at the first \\boxed."""
    manager = _tree_manager()
    manager.initialize_trees(_rollout_proto(VALID1 + "\n\\boxed{3}"))
    fork = manager.trees[0].all_nodes[0]
    step_boxed9 = "<step><premise>a</premise><conclusion>\\boxed{9}</conclusion></step>"
    step_boxed8 = "<step><premise>c</premise><conclusion>\\boxed{8}</conclusion></step>"
    new_nodes = manager.commit_branches(
        _branch_proto(step_boxed9 + step_boxed8), _fork_info(fork))
    assert [node.ext_prm_penalty for node in new_nodes] == [False, True]
    assert new_nodes[1].ext_prm_penalty_reason == "extra_boxed"

    seen = []
    manager.evaluate_leaves(lambda **kwargs: seen.append(kwargs["solution_str"]) or 0.0)
    branch_texts = [text for text in seen if "\\boxed{9}" in text]
    assert branch_texts and all("\\boxed{8}" not in text for text in branch_texts)


def test_tree_branch_stray_tags_mark_the_last_rewardable_node():
    """Stray reasoning tags in branch text cost the last process-rewardable node; the pure-answer tail carries no process reward, so the penalty must not silently land there and vanish."""
    manager = _tree_manager()
    manager.initialize_trees(_rollout_proto(VALID1 + "\n\\boxed{3}"))
    fork = manager.trees[0].all_nodes[0]
    new_nodes = manager.commit_branches(
        _branch_proto("<premise>x</premise>" + VALID2 + ANSWER7), _fork_info(fork))
    assert [node.process_rewardable for node in new_nodes] == [True, False]
    assert new_nodes[0].ext_prm_penalty is True
    assert new_nodes[0].ext_prm_penalty_reason == "stray_xml_tags"
    assert new_nodes[1].ext_prm_penalty is False


def test_tree_flat_batch_rows_keep_the_answer_tokens():
    """build_flat_batch rebuilds every training row from node token_ids: each row must reproduce the exact sampled token sequence, boxed included, and rm_scores must land on the true last token."""
    manager = _tree_manager()
    rollout = _rollout_proto(VALID1 + "\n\\boxed{3}")
    manager.initialize_trees(rollout)
    fork = manager.trees[0].all_nodes[0]
    manager.commit_branches(_branch_proto(VALID2 + ANSWER7), _fork_info(fork))
    manager.evaluate_leaves(
        lambda **kwargs: 1.0 if "\\boxed{7}" in kwargs["solution_str"] else 0.0)
    flat = manager.build_flat_batch(rollout)

    rows = flat.batch["responses"]
    masks = flat.batch["response_mask"]
    texts = [manager.tokenizer.decode(rows[i][masks[i] > 0].tolist())
             for i in range(rows.shape[0])]
    assert sorted(texts) == sorted([VALID1 + "\n\\boxed{3}", VALID1 + VALID2 + ANSWER7])
    branch_row = texts.index(VALID1 + VALID2 + ANSWER7)
    valid_len = int(masks[branch_row].sum().item())
    assert flat.batch["rm_scores"][branch_row, valid_len - 1].item() == 1.0


def test_tree_per_step_check_and_whole_path_rule_removal():
    stub = SimpleNamespace(use_xml=True, penalty_on_bad_format=True)
    assert TreeManager._should_penalize_step_format(stub, INVALID) is True
    assert TreeManager._should_penalize_step_format(stub, VALID1) is False
    assert not hasattr(TreeManager, "_should_penalize_path_format")
