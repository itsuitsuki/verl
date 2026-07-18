"""Coordinate access to on-disk cache entries across processes.

Each RewardLoopWorker process has its own in-memory cache and pending-request map, so the first occurrence of a prompt or theorem could otherwise be translated or proved once per process. 
These helpers add a lock file beside the disk-cache entry. The process that acquires the lock computes and stores the result; other processes wait for that stored result and reuse it.

SAME-NODE ONLY: both cache directories live under /tmp (node-local), where O_CREAT|O_EXCL is atomic. 
This is not correct on NFS; do not point the cache directories at a shared filesystem across nodes.

Failure behavior: a process that produced no cacheable result (failed translation, worker error, or slow/incomplete proof result) leaves a short-lived ``.fail`` marker so waiting processes stop immediately and recompute. 
A process killed before releasing its lock leaves a stale lock; waiting processes time out and recompute, and the next acquire() removes a lock older than ``stale_s``.
"""
import os
import time


def acquire(lock_path: str, stale_s: float) -> bool:
    """Try to acquire exclusive responsibility for one cache entry."""
    for _ in range(2):
        try:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                # os.write can fail with ENOSPC on a full tmpfs even after the directory entry was created. Always close the descriptor, or each disk-cache miss leaks one descriptor until the process reaches EMFILE.
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            # A new owner supersedes any previous failure marker.
            try:
                os.unlink(lock_path + ".fail")
            except OSError:
                pass
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_s:
                    os.unlink(lock_path)   # owner exited without releasing it
                    continue
            except OSError:
                continue   # raced with release/removal; retry the open once
            return False
        except OSError:
            # File-system trouble (permissions, ENOSPC): compute locally rather
            # than blocking the reward path on a broken optimization.
            return True
    return False


def release(lock_path: str) -> None:
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def mark_failed(lock_path: str) -> None:
    """Record that the lock owner produced no cacheable result."""
    try:
        with open(lock_path + ".fail", "w") as fh:
            fh.write(str(os.getpid()))
    except OSError:
        pass


def failed_recently(lock_path: str, ttl_s: float = 120.0) -> bool:
    try:
        return time.time() - os.path.getmtime(lock_path + ".fail") < ttl_s
    except OSError:
        return False


def wait(lock_path: str, load_fn, deadline_s: float, poll_s: float):
    """Wait for another process to store the cache entry.

    Returns the loaded payload, or None when the owner failed, exited, or did not finish before the deadline. 
    In every None case the caller recomputes, so the worst case is duplicate work rather than a missing result.
    """
    t_end = time.time() + deadline_s
    while time.time() < t_end:
        value = load_fn()
        if value is not None:
            return value
        if failed_recently(lock_path):
            return None
        if not os.path.exists(lock_path):
            # The owner released its lock without a readable result, either because it exited between storing and releasing or because storage failed. Inspect the cache once more, then recompute.
            return load_fn()
        time.sleep(poll_s)
    return None
