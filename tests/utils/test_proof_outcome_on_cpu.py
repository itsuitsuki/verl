"""CPU-only contract tests for ProofOutcome / VerificationOutcome (phase 2 step M1).

Pins the classification of every result shape the producers emit today, the explicit resolution of illegal boolean combinations, the equivalences that make the later migration steps provably behavior-preserving, and the persisted 3-field cache-entry round trip. Pure python, no Isabelle.
"""
import itertools

from verl.utils.isabelle_utils.state_classes import (PremiseConsistency,
                                                     ProofOutcome,
                                                     VerificationOutcome)


def classify(raw):
    return VerificationOutcome.from_raw(raw).outcome


def test_producer_shapes_classify_faithfully():
    # the four combinations that actually occur in production, plus the three distinguishable infrastructure flavors
    assert classify({"success": True, "elapsed": 0.4, "errors": []}) is ProofOutcome.PROVED
    assert classify({"success": False, "elapsed": 0.4, "errors": ["mock fail"]}) is ProofOutcome.UNPROVED
    assert classify({"success": False, "incomplete": True,
                     "errors": ["node not consolidated: {...}"]}) is ProofOutcome.INCOMPLETE
    assert classify({"success": False, "worker_error": True, "elapsed": 75.0,
                     "errors": ["hard timeout: worker force-restarted"]}) is ProofOutcome.TIMEOUT
    assert classify({"success": False, "worker_error": True, "elapsed": 0.0,
                     "errors": ["worker error: TimeoutError('no terminal message within deadline')"]}) is ProofOutcome.TIMEOUT
    assert classify({"success": False, "worker_error": True, "premise_consistency_unknown": True,
                     "errors": ["premise consistency unknown: two prior timeouts"]}) is ProofOutcome.TIMEOUT
    assert classify({"success": False, "worker_error": True,
                     "errors": ["FAILED: something"]}) is ProofOutcome.WORKER_ERROR
    assert classify({"success": False, "worker_error": True,
                     "errors": ["node canceled: {...}"]}) is ProofOutcome.WORKER_ERROR


def test_disk_cache_entry_and_minimal_shapes():
    # stored entries carry exactly success/elapsed/errors; mock pools may omit elapsed entirely
    assert classify({"success": True, "elapsed": 0.1, "errors": []}) is ProofOutcome.PROVED
    assert classify({"success": False, "elapsed": 0.5, "errors": ["by fails"]}) is ProofOutcome.UNPROVED
    assert classify({}) is ProofOutcome.UNPROVED
    assert classify({"success": True}) is ProofOutcome.PROVED
    result = VerificationOutcome.from_raw({"success": True})
    assert result.elapsed == 0.0 and result.errors == []


def test_illegal_combinations_resolve_by_documented_precedence():
    # these never come from production producers, but mocks or foreign payloads can build them; success wins, then worker_error, then incomplete -- the same order every legacy consumer read the booleans in
    assert classify({"success": True, "worker_error": True}) is ProofOutcome.PROVED
    assert classify({"success": True, "incomplete": True}) is ProofOutcome.PROVED
    assert classify({"success": False, "worker_error": True, "incomplete": True}) is ProofOutcome.WORKER_ERROR
    assert classify({"success": False, "premise_consistency_unknown": True}) is ProofOutcome.UNPROVED


def _raw_dict_space():
    # cartesian sweep over the field combinations from_raw can see, legal and illegal alike
    heads = [[], ["hard timeout: worker force-restarted"],
             ["worker error: TimeoutError('x')"], ["FAILED: y"],
             ["node not consolidated: z"]]
    for success, worker_error, incomplete, unknown, errors, elapsed in itertools.product(
            [True, False, None], [True, None], [True, None], [True, None],
            heads, [0.0, 5.0, 15.0]):
        raw = {"errors": list(errors), "elapsed": elapsed}
        if success is not None:
            raw["success"] = success
        if worker_error is not None:
            raw["worker_error"] = worker_error
        if incomplete is not None:
            raw["incomplete"] = incomplete
        if unknown is not None:
            raw["premise_consistency_unknown"] = unknown
        yield raw


def test_cacheable_equivalent_to_legacy_expression():
    for raw in _raw_dict_space():
        legacy = bool(raw.get("success")) or (
            not raw.get("worker_error") and not raw.get("incomplete")
            and raw.get("elapsed", 0.0) < 10.0
            and not any("not consolidated" in str(e) for e in raw.get("errors", [])))
        assert VerificationOutcome.from_raw(raw).cacheable(10.0) == legacy, raw


def test_proved_and_infrastructure_failure_properties():
    for raw in _raw_dict_space():
        result = VerificationOutcome.from_raw(raw)
        assert result.proved == (result.outcome is ProofOutcome.PROVED)
        assert result.infrastructure_failure == (
            result.outcome in (ProofOutcome.TIMEOUT, ProofOutcome.WORKER_ERROR))
        # the retry set and the repeated-failure set are deliberately the same set today
        assert result.infrastructure_failure == result.counts_toward_repeated_timeout


def test_strike_and_fail_closed_equivalent_on_non_success_dicts():
    # the legacy strike condition was a bare worker_error read and the legacy fail-closed read was incomplete-or-worker_error; both applied only to results that were not successes, which is the only domain production ever evaluated them on
    for raw in _raw_dict_space():
        if raw.get("success"):
            continue
        result = VerificationOutcome.from_raw(raw)
        assert result.counts_toward_repeated_timeout == bool(raw.get("worker_error")), raw
        assert result.fail_closed == bool(raw.get("worker_error") or raw.get("incomplete")), raw


def test_premise_consistency_equivalent_to_engine_classifier():
    for raw in _raw_dict_space():
        expected = (PremiseConsistency.INCONSISTENT if raw.get("success")
                    else PremiseConsistency.UNKNOWN
                    if (raw.get("worker_error") or raw.get("incomplete")
                        or raw.get("premise_consistency_unknown"))
                    else PremiseConsistency.CONSISTENT)
        assert VerificationOutcome.from_raw(raw).premise_consistency is expected, raw


def test_from_raw_is_idempotent():
    for raw in _raw_dict_space():
        result = VerificationOutcome.from_raw(raw)
        assert VerificationOutcome.from_raw(result) is result


def test_cache_entry_shape_and_reconstruction():
    proved = VerificationOutcome.from_raw({"success": True, "elapsed": 0.3, "errors": []})
    entry = proved.to_cache_entry()
    assert set(entry) == {"success", "elapsed", "errors"}
    assert VerificationOutcome.from_raw(entry).outcome is ProofOutcome.PROVED
    refused = VerificationOutcome.from_raw({"success": False, "elapsed": 0.5, "errors": ["by fails"]})
    assert VerificationOutcome.from_raw(refused.to_cache_entry()).outcome is ProofOutcome.UNPROVED
