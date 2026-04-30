#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP load balancer for local judge servers."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
from dataclasses import dataclass
from typing import Iterable

import aiohttp
from aiohttp import web


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass
class Backend:
    url: str
    inflight: int = 0
    requests: int = 0
    failures: int = 0
    last_error: str = ""


class OpenAILoadBalancer:
    def __init__(self, backends: Iterable[str], timeout_s: float):
        self.backends = [Backend(url.rstrip("/")) for url in backends]
        if not self.backends:
            raise ValueError("at least one backend is required")
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._counter = itertools.count()
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self.started_at = time.time()

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def choose_backend(self) -> Backend:
        async with self._lock:
            offset = next(self._counter)
            indexed = list(enumerate(self.backends))
            indexed.sort(key=lambda x: (x[1].inflight, (x[0] - offset) % len(self.backends)))
            backend = indexed[0][1]
            backend.inflight += 1
            backend.requests += 1
            return backend

    async def release_backend(self, backend: Backend, error: Exception | None = None) -> None:
        async with self._lock:
            backend.inflight = max(0, backend.inflight - 1)
            if error is not None:
                backend.failures += 1
                backend.last_error = repr(error)

    def stats(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started_at, 3),
            "backends": [
                {
                    "url": b.url,
                    "inflight": b.inflight,
                    "requests": b.requests,
                    "failures": b.failures,
                    "last_error": b.last_error,
                }
                for b in self.backends
            ],
        }


def filtered_headers(headers: aiohttp.typedefs.LooseHeaders) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}


async def stats_handler(request: web.Request) -> web.Response:
    lb: OpenAILoadBalancer = request.app["lb"]
    return web.json_response(lb.stats())


async def proxy_handler(request: web.Request) -> web.Response:
    lb: OpenAILoadBalancer = request.app["lb"]
    backend = await lb.choose_backend()
    error: Exception | None = None
    try:
        body = await request.read()
        target = f"{backend.url}{request.rel_url}"
        session = await lb.session()
        async with session.request(
            request.method,
            target,
            data=body,
            headers=filtered_headers(request.headers),
            allow_redirects=False,
        ) as resp:
            payload = await resp.read()
            headers = filtered_headers(resp.headers)
            headers["x-openai-lb-backend"] = backend.url
            return web.Response(status=resp.status, body=payload, headers=headers)
    except Exception as exc:  # noqa: BLE001 - proxy should report backend failures as 502.
        error = exc
        return web.json_response(
            {
                "error": {
                    "message": f"backend request failed: {exc!r}",
                    "type": "backend_error",
                    "backend": backend.url,
                }
            },
            status=502,
        )
    finally:
        await lb.release_backend(backend, error)


async def on_cleanup(app: web.Application) -> None:
    await app["lb"].close()


def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application(client_max_size=args.client_max_size_mb * 1024 * 1024)
    app["lb"] = OpenAILoadBalancer(args.backend, args.timeout)
    app.router.add_get("/__lb_stats", stats_handler)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4874)
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Backend base URL, e.g. http://127.0.0.1:4872",
    )
    parser.add_argument("--timeout", type=float, default=700.0)
    parser.add_argument("--client-max-size-mb", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.backend:
        args.backend = ["http://127.0.0.1:4872", "http://127.0.0.1:4873"]
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "backends": args.backend,
                "timeout": args.timeout,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    web.run_app(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
