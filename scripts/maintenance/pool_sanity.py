"""Minimal Isabelle pool sanity check: start a 2-worker pool (exercises the
session_start path that was failing), run one real check, shut down clean.
PASS = POOL_STARTED + CHECK success=True. Validates the symlink fix before a
full training restart."""
import sys
import time

sys.path.insert(0, "/2022533109/zhouchuyan/verl")
from verl.utils.isabelle_utils import server_pool as sp

sp.CHECK_DEADLINE = 90.0
pool = sp.IsabelleServerPool(num_workers=2, base_dir="/tmp/pool_sanity")
t0 = time.time()
try:
    pool.start()
    print(f"POOL_STARTED in {time.time()-t0:.1f}s", flush=True)
    r = pool.check('theorem t: shows "(2::nat) + 2 = 4" by simp')
    print(f"CHECK_RESULT success={r.get('success')} "
          f"errors={str(r.get('errors'))[:120]}", flush=True)
finally:
    try:
        pool.shutdown()
        print("SHUTDOWN_OK", flush=True)
    except Exception as e:
        print(f"SHUTDOWN_ERR {e!r}", flush=True)
