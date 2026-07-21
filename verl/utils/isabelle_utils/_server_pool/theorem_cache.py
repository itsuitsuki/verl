"""Persistent theorem-result cache for Isabelle server pools.

The cache stores the same stable proof outcomes as the in-memory memo. Change the cache version when a result can change without changing either the theorem text or the environment fingerprint."""
import hashlib
import json
import os
import re

from verl.utils.isabelle_utils._server_pool import config

# The memo and disk key are computed from normalize_theorem_text() (2026-07-21),
# which alpha-renames pv_ free variables and collapses whitespace so alpha-equivalent
# steps share one proof outcome. No version bump needed: normalized keys contain \x00
# placeholders that raw text never has, so old v0 entries cannot collide with new ones;
# entries whose theorems have no pv_ variables normalize to the original text and
# are reused from the existing cache.
_THM_CACHE_VERSION = os.environ.get("ISABELLE_THEOREM_CACHE_VERSION", "v0")
_THM_ENV_FPRINT: dict = {}

_PV_FIX_RE = re.compile(
    r"\bfixes\s+"
    r"(?P<names>pv_[A-Za-z0-9_']*(?:\s+(?:and\s+)?pv_[A-Za-z0-9_']*)*)"
    r"\s*::"
)
_WS_RE = re.compile(r"\s+")


def normalize_theorem_text(theorem_code: str) -> str:
    """Map alpha-equivalent generated theorems onto one cache-key string.

    Only names declared by theorem-header ``fixes ... ::`` clauses are renamed.
    This avoids touching a coincidentally ``pv_``-prefixed Isabelle constant or a
    direct-domain theorem that merely mentions such a token. Each distinct fixed
    variable receives a positional placeholder in order of declaration, and each
    run of whitespace is collapsed; numerals, operators, term structure, types,
    assumption labels, tactics, and every other identifier remain untouched. The
    value is a hashing key only; Isabelle always receives the original theorem.
    """
    names = []
    seen = set()
    for match in _PV_FIX_RE.finditer(theorem_code):
        for name in re.findall(r"\bpv_[A-Za-z0-9_']*\b", match.group("names")):
            if name not in seen:
                seen.add(name)
                names.append(name)
    if names:
        name_re = re.compile(
            r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b"
        )
        positions = {name: idx for idx, name in enumerate(names)}
        theorem_code = name_re.sub(
            lambda match: f"\x00{positions[match.group(0)]}\x00", theorem_code
        )
    return _WS_RE.sub(" ", theorem_code).strip()


def _thm_env_fprint(session=None, imports=None, options=None) -> str:
    """Return the cache identity of one pool's theorem environment.

    The identity includes the session, imports, options, Isabelle executable, heap image, and result schema. Deriving it from each pool's actual specification keeps general and direct-domain results separate."""
    session = session if session is not None else config.SESSION
    imports = imports if imports is not None else config.THEORY_IMPORTS
    options = list(options) if options is not None else list(config.SESSION_OPTIONS)
    key = (session, imports, tuple(options))
    cached = _THM_ENV_FPRINT.get(key)
    if cached is not None:
        return cached
    parts = [session, imports] + list(options)
    parts.append("result-schema=v2")
    try:
        st = os.stat(config.ISABELLE_BIN)
        parts.append(f"bin={config.ISABELLE_BIN}:{st.st_size}:{int(st.st_mtime)}")
    except OSError:
        parts.append(f"bin={config.ISABELLE_BIN}:unknown")
    try:
        import glob as _glob
        ihu = os.environ.get(
            "ISABELLE_HOME_USER",
            os.path.expanduser("~/.isabelle/Isabelle2025"))
        heaps = sorted(
            _glob.glob(os.path.join(ihu, "heaps", "*", session))
            + _glob.glob(os.path.join(ihu, "Isabelle2025", "heaps",
                                      "*", session)))
        hp = []
        for h in heaps[:4]:
            try:
                hst = os.stat(h)
                hp.append(f"{h}:{hst.st_size}:{int(hst.st_mtime)}")
            except OSError:
                pass
        parts.append("heaps=" + (";".join(hp) if hp else "none"))
    except Exception:  # noqa: BLE001
        parts.append("heaps=err")
    fp = hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    _THM_ENV_FPRINT[key] = fp
    return fp


def _thm_disk_enabled():
    return os.environ.get("ISABELLE_THEOREM_DISK_CACHE", "1") not in (
        "0", "false", "False")


def _thm_disk_path(theorem_code, fprint):
    key = hashlib.sha1(
        (_THM_CACHE_VERSION + "\0" + fprint + "\0" + theorem_code)
        .encode("utf-8")).hexdigest()
    base = os.environ.get("ISABELLE_THEOREM_CACHE_DIR",
                          "/tmp/verl_isabelle_theorem_cache")
    return os.path.join(base, _THM_CACHE_VERSION, key[:2], f"{key}.json")


def _thm_disk_load(theorem_code, fprint):
    if not _thm_disk_enabled():
        return None
    try:
        with open(_thm_disk_path(theorem_code, fprint)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _thm_disk_store(theorem_code, value, fprint):
    if not _thm_disk_enabled():
        return
    path = _thm_disk_path(theorem_code, fprint)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(value, fh)
        os.replace(tmp, path)
    except OSError:
        pass
