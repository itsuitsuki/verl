"""CPU-only tests for worker-scoped external solver cleanup."""
import os

import pytest

if not hasattr(os, "sysconf"):
    pytest.skip("server_pool is Linux-only (os.sysconf at module level)",
                allow_module_level=True)

from verl.utils.isabelle_utils import state_classes  # noqa: E402
from verl.utils.isabelle_utils._server_pool import processes  # noqa: E402
from verl.utils.isabelle_utils._server_pool.worker import IsabelleWorker  # noqa: E402


@pytest.fixture
def proc_tree(monkeypatch):
    target = {
        110: (110, 110, 1000),       # bash leader for csdp
        111: (110, 110, 1010),       # csdp
        120: (120, 120, 2000),       # bash leader for veriT
        121: (120, 120, 2010),       # veriT
        130: (130, 130, 3000),       # bash leader for poly
        131: (130, 130, 3010),       # poly
        140: (140, 140, 4000),       # leader for a nested non-solver
        141: (140, 140, 4010),       # csdp-like executable below own group
        150: (999, 999, 5000),       # solver not attached to a self-led descendant group
    }
    names = {111: "csdp", 121: "veriT", 131: "poly", 141: "bash", 150: "csdp"}
    monkeypatch.setattr(processes, "_descendants", lambda root: dict(target))
    monkeypatch.setattr(processes, "_process_name", lambda pid: names.get(pid, "bash"))
    return target, names


def test_selects_only_external_solver_groups_below_target_jvm(monkeypatch, proc_tree):
    monkeypatch.setattr(os, "getpgrp", lambda: 999)
    assert processes._external_solver_groups(10, os.getpgrp()) == {
        110: 1000,
        120: 2000,
    }


def test_kill_rechecks_leader_identity_and_counts_success(monkeypatch, proc_tree):
    calls = []

    def kill(pgid, start):
        calls.append((pgid, start))
        return pgid == 110

    monkeypatch.setattr(processes, "_kill_pgid", kill)
    assert processes._kill_external_solver_groups(10, 999) == 1
    assert calls == [(110, 1000), (120, 2000)]


def test_caller_process_group_is_excluded(monkeypatch, proc_tree):
    assert processes._external_solver_groups(10, 110) == {120: 2000}


def test_same_named_solver_outside_target_jvm_is_untouched(monkeypatch):
    monkeypatch.setattr(processes, "_descendants", lambda root: {
        210: (210, 210, 6000),
        211: (210, 210, 6010),
    } if root == 20 else {})
    monkeypatch.setattr(processes, "_process_name", lambda pid: "csdp" if pid == 211 else "bash")
    assert processes._external_solver_groups(10, 999) == {}
    assert processes._external_solver_groups(20, 999) == {210: 6000}


def _worker_for_check(tmp_path, payload):
    worker = IsabelleWorker(0, tmp_path)
    worker.master_dir.mkdir(parents=True, exist_ok=True)
    worker.jvm_pid = 777
    worker.session_id = "session"

    class Conn:
        def __init__(self):
            self.name = ""

        def request_task(self, _command, request):
            self.name = request["theories"][0]
            return "task"

        def wait_task(self, *_args, **_kwargs):
            if isinstance(payload, BaseException):
                raise payload
            kind, response = payload
            response = {
                **response,
                "nodes": [
                    {**node, "theory_name": self.name}
                    for node in response["nodes"]
                ],
            }
            return kind, response

    worker.conn = Conn()
    return worker


def _finished_payload(ok=True, consolidated=True):
    return "FINISHED", {
        "ok": ok,
        "errors": [] if ok else [{"message": "not proved"}],
        "nodes": [{
            "theory_name": "V",
            "status": {
                "ok": ok,
                "consolidated": consolidated,
                "percentage": 100 if consolidated else 50,
                "failed": 0 if ok else 1,
            },
        }],
    }


def test_worker_cleans_after_proved_result(tmp_path, monkeypatch):
    worker = _worker_for_check(tmp_path, _finished_payload())
    monkeypatch.setattr(processes, "_kill_external_solver_groups", lambda *_args: 2)
    result = worker.check('theorem chk: shows "True" by simp')
    assert result.outcome is state_classes.ProofOutcome.PROVED
    assert worker.external_solver_reaps == 2


def test_worker_cleans_after_unproved_and_incomplete_results(tmp_path, monkeypatch):
    monkeypatch.setattr(processes, "_kill_external_solver_groups", lambda *_args: 1)

    unproved = _worker_for_check(tmp_path / "unproved", _finished_payload(ok=False))
    unproved_result = unproved.check('theorem chk: shows "False" by simp')
    assert unproved_result.outcome is state_classes.ProofOutcome.UNPROVED
    assert unproved.external_solver_reaps == 1

    incomplete = _worker_for_check(
        tmp_path / "incomplete", _finished_payload(ok=False, consolidated=False))
    incomplete_result = incomplete.check('theorem chk: shows "False" by simp')
    assert incomplete_result.outcome is state_classes.ProofOutcome.INCOMPLETE
    assert incomplete.external_solver_reaps == 1


def test_worker_cleans_when_check_raises(tmp_path, monkeypatch):
    worker = _worker_for_check(tmp_path, TimeoutError("deadline"))
    calls = []
    monkeypatch.setattr(
        processes, "_kill_external_solver_groups",
        lambda *args: calls.append(args) or 1,
    )
    with pytest.raises(TimeoutError, match="deadline"):
        worker.check('theorem chk: shows "True" by simp')
    assert calls == [(777, os.getpgrp())]
    assert worker.external_solver_reaps == 1


def test_cleanup_failure_does_not_change_completed_result(tmp_path, monkeypatch, capsys):
    worker = _worker_for_check(tmp_path, _finished_payload())

    def fail(*_args):
        raise OSError("proc scan failed")

    monkeypatch.setattr(processes, "_kill_external_solver_groups", fail)
    result = worker.check('theorem chk: shows "True" by simp')
    assert result.outcome is state_classes.ProofOutcome.PROVED
    assert worker.external_solver_reaps == 0
    assert "external solver cleanup failed" in capsys.readouterr().out
