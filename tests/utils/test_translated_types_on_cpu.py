"""CPU-only contract tests for PyExprGiven / PyExprStep (phase 2).

Pins the parser output types, the normalization and junk filtering they carry over unchanged, the derived direct-domain property (prop-only, distinct from final routing which also sees premises), and the schema-2 disk payload round trip; byte-identical to the format stored before these types existed. Pure python, no Isabelle.
"""
import dataclasses
import os

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("engine imports server_pool (Linux-only os.sysconf)",
                allow_module_level=True)

from verl.utils.isabelle_utils import translator
from verl.utils.isabelle_utils.stages import direct_verify, formalization
from verl.utils.isabelle_utils.state_classes import PyExprGiven, PyExprStep

GIVENS_REPLY = """VARS: a int
GIVEN: a = 1
Some commentary the model loves to add.
VARS: x int, y real, n nat
GIVEN: x = 3
GIVEN: y ^ 2 = x + 1
GIVEN: answer = y
"""


def test_parse_givens_returns_typed_from_last_vars_block():
    parsed = formalization.parse_givens_vars_to_pyexpr(GIVENS_REPLY)
    assert isinstance(parsed, PyExprGiven)
    # nat folds to int; only the block after the LAST VARS line counts
    assert parsed.pyexpr_variable_types == [("x", "int"), ("y", "real"), ("n", "int")]
    assert parsed.pyexpr_givens == ["x == 3", "y ** 2 == x + 1", "answer == y"]


def test_parse_givens_none_cases():
    assert formalization.parse_givens_vars_to_pyexpr("GIVEN: x = 3") is None          # no VARS
    assert formalization.parse_givens_vars_to_pyexpr("VARS: x int") is None           # no GIVEN


def test_parse_props_returns_typed_steps_with_normalization():
    reply = ("STEP 1 | premises: x = 3; y > 0 | prop: x + y = 4\n"
             "STEP 2 | prop: y ^ 2 = 1\n")
    parsed = formalization.parse_step_translation_to_pyexpr(reply)
    assert parsed == {
        1: PyExprStep(pyexpr_conclusion="x + y == 4", pyexpr_premises=["x == 3", "y > 0"]),
        2: PyExprStep(pyexpr_conclusion="y ** 2 == 1", pyexpr_premises=[]),
    }
    assert not parsed[1].direct and not parsed[2].direct


def test_parse_props_keeps_direct_domain_statements_verbatim():
    reply = ("STEP 1 | premises: x \\<in> carrier G | "
             "prop: inv x \\<otimes> x = \\<one>\n")
    parsed = formalization.parse_step_translation_to_pyexpr(reply)
    step = parsed[1]
    # verbatim (form B (constrained pyexpr) skipped): the single `=` is NOT doubled, `\<in>` survives untouched
    assert step.pyexpr_conclusion == "inv x \\<otimes> x = \\<one>"
    assert step.pyexpr_premises == ["x \\<in> carrier G"]
    assert step.direct is True


def test_direct_is_prop_only_while_routing_sees_premises_too():
    step = PyExprStep(pyexpr_conclusion="x == 3", pyexpr_premises=["y \\<in> carrier G"])
    assert step.direct is False
    # process_one's routing authority matches on claim AND premises together
    assert direct_verify.match_domain(step.pyexpr_conclusion, *step.pyexpr_premises) is not None


def test_translated_step_is_immutable():
    step = PyExprStep(pyexpr_conclusion="x == 3", pyexpr_premises=[])
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.pyexpr_conclusion = "x == 4"


def test_disk_payload_round_trip_and_format_unchanged():
    given = PyExprGiven(pyexpr_variable_types=[("x", "int"), ("y", "real")],
                            pyexpr_givens=["x == 3", "answer == y"])
    payload = translator._tr_encode(given)
    # exact payload an old process would have written -- format is pinned
    assert payload == {"schema": 2, "kind": "givens",
                       "fixes": [["x", "int"], ["y", "real"]],
                       "givens": ["x == 3", "answer == y"]}
    assert translator._tr_decode(payload) == given

    steps = {1: PyExprStep(pyexpr_conclusion="x == 3", pyexpr_premises=["y > 0"]),
             2: PyExprStep(pyexpr_conclusion="inv x \\<otimes> x = \\<one>",
                               pyexpr_premises=[])}
    payload = translator._tr_encode(steps)
    assert payload == {"schema": 2, "kind": "steps",
                       "items": [[1, "x == 3", ["y > 0"]],
                                 [2, "inv x \\<otimes> x = \\<one>", []]]}
    decoded = translator._tr_decode(payload)
    assert decoded == steps
    assert decoded[2].direct and not decoded[1].direct


def test_foreign_parse_shapes_json_fallback_or_not_disk_cached():
    # a tuple does not survive a JSON round trip -> simply not disk-cached
    assert translator._tr_encode(([("x", "int")], ["x == 3"])) is None
    # plain JSON values (e.g. a raw reply string) still cache via the fallback
    assert translator._tr_encode("RAW") == {"schema": 2, "kind": "json",
                                            "value": "RAW"}


def test_translate_disk_hit_returns_typed_givens(tmp_path, monkeypatch):
    monkeypatch.setenv("ISABELLE_TRANSLATE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ISABELLE_TRANSLATE_DISK_CACHE", "1")
    calls = {"n": 0}

    def fake_judge(prompt, **_kw):
        calls["n"] += 1
        return "VARS: x int\nGIVEN: x = 3\nGIVEN: answer = x"

    monkeypatch.setattr(translator, "call_judge", fake_judge)
    parsed1, _, _ = translator.translate(
        "PB", formalization.parse_givens_vars_to_pyexpr, lambda p: [],
        translator_url="u", translator_model="m")
    assert isinstance(parsed1, PyExprGiven) and calls["n"] == 1
    with translator._TR_LOCK:
        translator._TR_CACHE.clear()          # cold memory, disk kept
    parsed2, attempts, _ = translator.translate(
        "PB", formalization.parse_givens_vars_to_pyexpr, lambda p: [],
        translator_url="u", translator_model="m")
    assert isinstance(parsed2, PyExprGiven) and parsed2 == parsed1
    assert calls["n"] == 1                    # served from disk, no new call
    assert attempts and attempts[0].get("cache") == "disk"
