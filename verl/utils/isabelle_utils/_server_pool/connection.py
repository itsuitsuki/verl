"""Blocking client for the Isabelle PIDE server protocol."""
import json
import re
import socket
import time


class _Conn:
    """A blocking authenticated connection to one Isabelle server."""

    def __init__(self, host: str, port: int, password: str):
        self.sock = socket.create_connection((host, port))
        self.buf = b""
        self.send_line(password)
        reply = self.read_msg()
        if not reply.startswith("OK"):
            raise RuntimeError(f"server auth failed: {reply!r}")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def send_line(self, s: str):
        self.sock.sendall(s.encode() + b"\n")

    def _read_line(self) -> bytes:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("server closed connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def _read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("server closed connection")
            self.buf += chunk
        data, self.buf = self.buf[:n], self.buf[n:]
        return data

    def read_msg(self) -> str:
        line = self._read_line()
        if re.fullmatch(rb"\d+", line.strip()):
            return self._read_exact(int(line)).decode()
        return line.decode()

    def command(self, name: str, args=None):
        self.send_line(name if args is None else f"{name} {json.dumps(args)}")

    def request(self, name: str, args=None) -> str:
        """Send a command and return its immediate reply."""
        self.command(name, args)
        return self.read_msg()

    def request_task(self, name: str, args=None) -> str:
        """Start an asynchronous command and return its task ID.

        A malformed acknowledgement leaves the stream state uncertain, so callers must restart the worker instead of reusing this connection."""
        reply = self.request(name, args)
        if not reply.startswith("OK"):
            raise RuntimeError(f"{name} rejected: {reply[:200]}")
        try:
            task = json.loads(reply[2:].strip()).get("task")
        except (json.JSONDecodeError, AttributeError):
            task = None
        if not task:
            raise RuntimeError(f"{name}: no task id in reply: {reply[:200]}")
        return task

    def wait_task(self, deadline: float, task: str | None = None):
        """Wait for the terminal message belonging to `task`.

        A terminal message for another task indicates a desynchronized stream. Raising lets the pool discard the worker rather than associate one theorem with another theorem's result."""
        self.sock.settimeout(deadline)
        try:
            t0 = time.time()
            while time.time() - t0 < deadline:
                msg = self.read_msg()
                kind = msg.split(" ", 1)[0]
                if kind not in ("FINISHED", "FAILED", "ERROR"):
                    continue
                body = msg[len(kind) + 1:] if " " in msg else "{}"
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {"raw": body}
                got = payload.get("task") if isinstance(payload, dict) else None
                if task is not None and got is not None and got != task:
                    raise RuntimeError(
                        f"desync: terminal message for task {got!r} while "
                        f"waiting for {task!r}")
                return kind, payload
            raise TimeoutError("no terminal message within deadline")
        finally:
            self.sock.settimeout(None)
