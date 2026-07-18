"""Contract tests for the public Isabelle server-pool interface."""
from pathlib import Path

import verl.utils.isabelle_utils.server_pool as server_pool
from verl.utils.isabelle_utils._server_pool import pool as pool_impl


def test_public_interface_exports_only_pool_class():
    assert server_pool.__all__ == ["IsabelleServerPool"]
    assert server_pool.IsabelleServerPool is pool_impl.IsabelleServerPool
    assert not hasattr(server_pool, "IsabelleWorker")


def test_watchdog_reaper_is_packaged_with_pool_implementation():
    assert Path(pool_impl.__file__).with_name("reaper.py").is_file()
