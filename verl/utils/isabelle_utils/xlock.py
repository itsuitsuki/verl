"""Cross-process single-flight for the on-disk caches (2026-07-11 review #4).

The 4 RewardLoopWorker processes of one training run each have their own
in-memory caches and in-process single-flight, so the FIRST occurrence of a
prompt/theorem was translated/proved up to 4x (once per process). These
helpers add a lock file next to the disk-cache entry: one process becomes the
cross-process leader, the others poll for its stored result and reuse it.

SAME-NODE ONLY: both cache dirs live under /tmp (node-local), where
O_CREAT|O_EXCL is atomic. This is NOT correct on NFS -- do not point the
cache dirs at a shared filesystem across nodes.

Failure protocol: a leader that produced an uncacheable result (failed
translation, worker_error, slow/incomplete verdict) leaves a short-lived
``.fail`` marker so waiters stop polling immediately and recompute
themselves -- mirroring the in-process single-flight, where a follower of a
failed flight becomes the new leader. A leader that died hard (SIGKILL)
leaves a stale lock; waiters time out and recompute, and the next acquire()
steals any lock older than ``stale_s``.
"""
import os
import time


def acquire(lock_path: str, stale_s: float) -> bool:
    """Try to become the cross-process leader for one cache entry."""
    for _ in range(2):
        try:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                # write can raise (ENOSPC on a full tmpfs -- data pages fail
                # even when the dirent fit); the fd MUST still be closed or
                # every disk miss leaks one fd until EMFILE (review round 2)
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            # a fresh leader supersedes any previous failure marker
            try:
                os.unlink(lock_path + ".fail")
            except OSError:
                pass
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_s:
                    os.unlink(lock_path)   # abandoned (leader SIGKILLed)
                    continue
            except OSError:
                continue   # raced with a release/steal; retry the open once
            return False
        except OSError:
            # fs trouble (permissions, ENOSPC): act as leader rather than
            # blocking the reward path on a broken optimization
            return True
    return False


def release(lock_path: str) -> None:
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def mark_failed(lock_path: str) -> None:
    """Leader produced no cacheable result: tell waiters to take over."""
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
    """Follower: poll for the leader's stored result.

    Returns the loaded payload, or None when the leader failed / vanished /
    timed out -- in every None case the caller simply recomputes, so the
    worst case degrades to the pre-lock behavior (duplicate work), never to
    a missing result.
    """
    t_end = time.time() + deadline_s
    while time.time() < t_end:
        v = load_fn()
        if v is not None:
            return v
        if failed_recently(lock_path):
            return None
        if not os.path.exists(lock_path):
            # released without a result (crashed between store and release,
            # or the store itself failed): one last look, then recompute
            return load_fn()
        time.sleep(poll_s)
    return None
