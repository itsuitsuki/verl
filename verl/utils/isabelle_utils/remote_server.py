"""Standalone HTTP server exposing an IsabelleServerPool to remote training nodes.

Runs on a CPU-only machine and owns one full local pool: workers, memory and disk caches, retry, the timeout
short-circuit, and the reapers, all unchanged. Training processes reach it through RemoteIsabellePool
(verl.utils.isabelle_utils._server_pool.remote_pool). Endpoints:

  POST /check   body {"theorem": str} -> the wire dict of one VerificationOutcome (remote_pool.outcome_to_wire)
  GET  /health  {"status": "ok", "idle_workers": N, "num_workers": M, ...gauge counters}

Launch:
  python -m verl.utils.isabelle_utils.remote_server --workers 16 --port 8477

Importing this module never starts a pool or touches Isabelle; the pool is created by the FastAPI lifespan
when the server actually starts, and shut down when it stops.
"""
import argparse
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from verl.utils.isabelle_utils._server_pool.pool import IsabelleServerPool
from verl.utils.isabelle_utils._server_pool.remote_pool import outcome_to_wire

_DEFAULT_WORKERS = 16
_DEFAULT_PORT = 8477
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_BASE_DIR = "/tmp/isabelle_pool_server"

# Filled by main() from CLI arguments before uvicorn starts; lifespan falls back to the same defaults, so
# `uvicorn verl.utils.isabelle_utils.remote_server:app` also works (configured through the env vars below).
_settings: dict = {}
_pool: IsabelleServerPool | None = None


class CheckRequest(BaseModel):
    theorem: str


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pool
    pool = IsabelleServerPool(
        num_workers=int(_settings.get("workers") or os.environ.get("ISABELLE_SERVER_WORKERS", _DEFAULT_WORKERS)),
        # PID-qualified like the engine's local pool dir: worker startup rmtrees its own master dir, so a
        # shared path would let two server processes wipe each other's live workers.
        base_dir=f"{_settings.get('base_dir') or _DEFAULT_BASE_DIR}_{os.getpid()}",
        verify_timeout=_settings.get("verify_timeout"),
        each_worker_proc_tree_mem_max_gb=_settings.get("worker_rss_cap_gb"),
        runaway_cpu_seconds=_settings.get("runaway_cpu_s"),
    )
    pool.start()
    _pool = pool
    try:
        yield
    finally:
        _pool = None
        pool.shutdown()


app = FastAPI(title="Isabelle remote verification pool", lifespan=_lifespan)


@app.get("/health")
async def health():
    if _pool is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    # idle_workers is Queue.qsize(), an instantaneous approximation; the gauge counters back the client's
    # restart_count / external_solver_reaps / cache_hits / cache_misses properties.
    return {
        "status": "ok",
        "idle_workers": _pool.idle.qsize(),
        "num_workers": _pool.num_workers,
        "restart_count": _pool.restart_count,
        "external_solver_reaps": _pool.external_solver_reaps,
        "cache_hits": _pool.cache_hits,
        "cache_misses": _pool.cache_misses,
    }


@app.post("/check")
async def check(request: CheckRequest):
    if _pool is None:
        return JSONResponse(status_code=503, content={"detail": "pool not started"})
    # pool.submit routes through the pool's own dispatcher FIFO (with its memory-cache fast path); awaiting
    # the wrapped future keeps the uvicorn event loop free while provers run, so concurrent requests queue in
    # the pool exactly as concurrent local callers would.
    result = await asyncio.wrap_future(_pool.submit(request.theorem))
    return outcome_to_wire(result)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Serve an IsabelleServerPool over HTTP for remote training nodes (CPU only).")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("ISABELLE_SERVER_WORKERS", _DEFAULT_WORKERS)),
                        help="Isabelle worker count in the pool (prover processes, not HTTP workers).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ISABELLE_SERVER_PORT", _DEFAULT_PORT)))
    parser.add_argument("--host", default=os.environ.get("ISABELLE_SERVER_HOST", _DEFAULT_HOST))
    parser.add_argument("--verify-timeout", type=float, default=None,
                        help="Per-theorem prover deadline in seconds (pool default when omitted).")
    parser.add_argument("--worker-rss-cap-gb", type=float, default=None,
                        help="Poly/ML process-tree memory cap per worker in GB (pool default when omitted).")
    parser.add_argument("--runaway-cpu-s", type=float, default=None,
                        help="Sustained-CPU seconds before a stray prover process is reaped (pool default when omitted).")
    parser.add_argument("--base-dir", default=_DEFAULT_BASE_DIR,
                        help="Pool directory prefix; the server PID is appended.")
    args = parser.parse_args()
    _settings.update(
        workers=args.workers,
        verify_timeout=args.verify_timeout,
        worker_rss_cap_gb=args.worker_rss_cap_gb,
        runaway_cpu_s=args.runaway_cpu_s,
        base_dir=args.base_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
