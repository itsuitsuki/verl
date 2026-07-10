"""CPU-only unit tests for the runaway-poly reaper (server_pool.py). Mocks all
/proc access (_descendants / _poly_cpu_state / _read_stat) and os.kill so NO
Isabelle is needed. Verifies the 2026-07-09 step-135 wedge fix: a `poly` that
sustains ~100% CPU for >= _runaway_streak monitor ticks (a native tactic that
ignored Isabelle's cooperative verify_timeout) is SIGKILLed; idle / intermittent
polys, and a pid-reused slot, are NOT. Linux-only (server_pool uses os.sysconf)."""
import os

import pytest

from verl.utils.isabelle_utils import server_pool as sp

POLY = 12345


@pytest.fixture
def pool(tmp_path, monkeypatch):
    p = sp.IsabelleServerPool(num_workers=1, base_dir=str(tmp_path))
    p.workers[0].jvm_pid = 9999
    monkeypatch.setattr(sp, "_read_stat",
                        lambda pid: (1, 1, 1, 100) if pid == 9999 else None)
    monkeypatch.setattr(sp, "_descendants", lambda root: {POLY: (0, 0, 0)})
    return p


def _drive(pool, monkeypatch, cpu_per_tick, ticks, start=500):
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    state = {"cpu": 0, "start": start}
    monkeypatch.setattr(sp, "_poly_cpu_state",
                        lambda pid: ("R", state["cpu"], state["start"]))
    pool._poly_cpu = {}
    for _ in range(ticks):
        pool._reap_runaway_polys_once()
        state["cpu"] += cpu_per_tick
    return killed


def _full(pool):
    return int(pool.MONITOR_INTERVAL_S * pool._clk_tck)   # 100% of one interval


def test_sustained_spinner_killed(pool, monkeypatch):
    killed = _drive(pool, monkeypatch, _full(pool), pool._runaway_streak + 1)
    assert POLY in killed
    assert pool.restart_reasons.get("runaway_poly", 0) >= 1


def test_idle_not_killed(pool, monkeypatch):
    killed = _drive(pool, monkeypatch, 0, pool._runaway_streak + 3)
    assert POLY not in killed


def test_half_busy_not_killed(pool, monkeypatch):
    killed = _drive(pool, monkeypatch, _full(pool) // 2, pool._runaway_streak + 3)
    assert POLY not in killed


def test_pid_reuse_resets_streak(pool, monkeypatch):
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    full = _full(pool)
    state = {"cpu": 0, "start": 500}
    monkeypatch.setattr(sp, "_poly_cpu_state",
                        lambda pid: ("R", state["cpu"], state["start"]))
    pool._poly_cpu = {}
    pool._reap_runaway_polys_once()                 # baseline @ start 500
    state["cpu"] += full
    state["start"] = 600                            # SAME pid, NEW process
    pool._reap_runaway_polys_once()                 # streak must reset to 0
    state["cpu"] += full
    pool._reap_runaway_polys_once()                 # streak now 1 < threshold
    assert POLY not in killed
