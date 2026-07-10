"""CPU-only unit tests for the runaway-poly reaper (server_pool.py), v2
discriminator (2026-07-10). Mocks all /proc access and os.kill so NO Isabelle
is needed.

v1 reaped ANY poly with sustained ~100% CPU and mass-killed HEALTHY provers
grinding queued checks (8 kills/12min incl. 3-at-once -> pool wedge). v2 reaps
only polys in an OVERSIZED (>2) poly tree -- the zombie-with-replacement
signature -- and always protects the 2 newest-started polys (the live prover
pair). Linux-only (server_pool uses os.sysconf)."""
import os

import pytest

from verl.utils.isabelle_utils import server_pool as sp


@pytest.fixture
def pool(tmp_path, monkeypatch):
    p = sp.IsabelleServerPool(num_workers=1, base_dir=str(tmp_path))
    p.workers[0].jvm_pid = 9999
    monkeypatch.setattr(sp, "_read_stat",
                        lambda pid: (1, 1, 1, 100) if pid == 9999 else None)
    return p


def _wire(monkeypatch, polys):
    """polys: {pid: {"cpu": int, "start": int}} mutable state."""
    monkeypatch.setattr(sp, "_descendants",
                        lambda root: {pid: (0, 0, 0) for pid in polys})
    monkeypatch.setattr(
        sp, "_poly_cpu_state",
        lambda pid: ("R", polys[pid]["cpu"], polys[pid]["start"])
        if pid in polys else None)
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    return killed


def _full(pool):
    return int(pool.MONITOR_INTERVAL_S * pool._clk_tck)   # 100% of one tick


def _tick(pool, polys, spin_pids):
    for pid in spin_pids:
        polys[pid]["cpu"] += _full(pool)
    pool._reap_runaway_polys_once()


def test_live_pair_spinner_protected(pool, monkeypatch):
    # healthy tree of TWO polys: the ML poly grinding queued checks sustains
    # 100% CPU legitimately -- must NEVER be reaped (the v1 false positive)
    polys = {11: {"cpu": 0, "start": 500}, 12: {"cpu": 0, "start": 510}}
    killed = _wire(monkeypatch, polys)
    pool._poly_cpu = {}
    pool._reap_runaway_polys_once()                   # baseline
    for _ in range(pool._runaway_streak + 3):
        _tick(pool, polys, [11])
    assert killed == []


def test_zombie_in_oversized_tree_killed(pool, monkeypatch):
    # 3 polys: oldest (100) is the abandoned zombie spinning; newest two are
    # the live pair -> only the zombie dies
    polys = {21: {"cpu": 0, "start": 100},
             22: {"cpu": 0, "start": 200},
             23: {"cpu": 0, "start": 300}}
    killed = _wire(monkeypatch, polys)
    pool._poly_cpu = {}
    pool._reap_runaway_polys_once()
    for _ in range(pool._runaway_streak):
        _tick(pool, polys, [21])
    assert killed == [21]


def test_newest_two_protected_even_spinning(pool, monkeypatch):
    # all 3 spin: only the oldest is reaped; the protected pair survives
    polys = {31: {"cpu": 0, "start": 100},
             32: {"cpu": 0, "start": 200},
             33: {"cpu": 0, "start": 300}}
    killed = _wire(monkeypatch, polys)
    pool._poly_cpu = {}
    pool._reap_runaway_polys_once()
    for _ in range(pool._runaway_streak + 2):
        _tick(pool, polys, [31, 32, 33])
    assert 31 in killed and 32 not in killed and 33 not in killed


def test_idle_zombie_not_killed(pool, monkeypatch):
    # oversized tree but the old poly is IDLE (cpu static) -> left alone
    polys = {41: {"cpu": 0, "start": 100},
             42: {"cpu": 0, "start": 200},
             43: {"cpu": 0, "start": 300}}
    killed = _wire(monkeypatch, polys)
    pool._poly_cpu = {}
    for _ in range(pool._runaway_streak + 3):
        pool._reap_runaway_polys_once()
    assert killed == []


def test_pid_reuse_resets_streak(pool, monkeypatch):
    assert pool._runaway_streak == 2                  # test hinges on this
    polys = {51: {"cpu": 0, "start": 100},
             52: {"cpu": 0, "start": 200},
             53: {"cpu": 0, "start": 300}}
    killed = _wire(monkeypatch, polys)
    pool._poly_cpu = {}
    pool._reap_runaway_polys_once()                   # baseline @ start 100
    _tick(pool, polys, [51])                          # streak 1
    polys[51]["start"] = 50                           # SAME pid, NEW process
    #                                                   (still the oldest)
    _tick(pool, polys, [51])   # without the reset this tick would kill (2)
    assert killed == []
