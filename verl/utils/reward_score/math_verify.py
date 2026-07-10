# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes as _ctypes
import logging as _logging
import multiprocessing as _mp
import os as _os
import re as _re
import signal as _signal
import threading as _threading

_MV_LOG = _logging.getLogger("math_verify_guard")

try:
    from math_verify.errors import TimeoutException
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

# Hard wall-clock cap (seconds) on ONE outcome-reward grade. math-verify's own
# timeout is signal.alarm-based (POSIX main-thread only) and is disabled here
# because reward runs in Ray worker threads; a giant-number answer like
# \boxed{2^{100000000}} then makes sympy/gmpy2 materialize a ~30M-digit integer
# in C (GIL held, no Python frame) and spin for hours (2026-07-09 ~24min wedge,
# reproduced). A subprocess SIGKILL is the ONLY way to bound it (thread timeouts
# cannot preempt a GIL-holding C call). See project_mathverify_bignum_spin.
_MV_WALL_TIMEOUT = float(_os.environ.get("MATH_VERIFY_WALL_TIMEOUT_S", "15"))
# Bound concurrent grading forks: during validation thousands of answers are
# graded at once across 128-thread pools; unbounded fork-per-answer would spike
# node memory (each pathological child can allocate GBs before the wall kill).
_MV_FORK_SEM = _threading.BoundedSemaphore(
    int(_os.environ.get("MATH_VERIFY_MAX_FORKS", "4")))
# (2026-07-11) RLIMIT_AS on the child was removed: a forked child inherits the
# parent's ~50GB virtual address space, so any useful cap is exceeded at birth
# and every forked grade silently zeroes. Orphans are prevented by PDEATHSIG;
# runtime is bounded by the wall timeout.

_GOLD_TARGETS = (LatexExtractionConfig(),)
_PRED_TARGETS = (ExprExtractionConfig(), LatexExtractionConfig())


# ---------------------------------------------------------------------------
# Quantified-proposition tier (2026-07-05).
#
# math-verify parses quantified statements (\exists x \in D, P(x)) as plain
# strings -- sympy has no quantifier object -- so grading degrades to string
# equality after light normalization. That made grading a formatting lottery:
# gt "\left(0,+\infty \right)" vs pred "(0,+\infty)" scored 0 despite being
# the same proposition (Big-Math cn_k12 "negate the proposition" items, ~192
# rows). This tier fires ONLY when both sides parse to strings AND look like
# quantified propositions; it decomposes quantifier / bound variable /
# domain / body and compares each part semantically (domain via sympy
# Interval/Set, body via math-verify's own inequality grading). Everything
# else keeps the exact old behavior.
# ---------------------------------------------------------------------------

def _split_domain_body(rest: str):
    """Split '<domain> , <body>' at the first comma that sits at bracket
    depth 0 -- interval domains like (0,+\\infty) contain commas themselves,
    so a naive comma split would cut the interval in half."""
    depth = 0
    for i, ch in enumerate(rest):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return rest[:i].strip(), rest[i + 1:].strip()
    return None


def _parse_quantified(s: str):
    """Split a LaTeX quantified proposition into (quant, var, domain, body).

    Returns None if `s` does not match the shape
        \\exists|\\forall <var> (\\in <domain>)? , <body>
    Domain may be an interval (a,b) / [a,b], \\mathbb{R}/Z/N, R, or a bare
    inequality bound like `\\exists x > 1, ...`.
    """
    import re

    s = s.strip()
    m = re.match(
        r"^\\(exists|forall)\s*(?P<var>[a-zA-Z](?:_\{?\w+\}?)?)\s*(?P<rest>.*)$",
        s,
    )
    if not m:
        return None
    quant = m.group(1)
    var = m.group("var").strip()
    rest = m.group("rest").strip()
    if rest.startswith("\\in"):
        split = _split_domain_body(rest[len("\\in"):].strip())
        if split is None:
            return None
        dom, body = split
    elif rest.startswith((">", "<")) or rest.startswith("\\geq") or rest.startswith("\\leq"):
        split = _split_domain_body(rest)
        if split is None:
            return None
        bound, body = split
        dom = var + " " + bound  # "\exists x > 1, ..." -> inequality domain
    elif rest.startswith(","):
        dom, body = "", rest[1:].strip()
    else:
        return None
    if not body:
        return None
    return quant, var, dom, body


def _norm_domain(dom: str):
    """Normalize a domain string to a comparable canonical form.

    Intervals -> sympy Interval; named sets -> canonical token; inequality
    domains -> math-verify-parsed relational. Returns a pair (kind, value)
    or None when unparseable (caller then falls back to string compare).
    """
    import re

    import sympy

    d = dom.strip()
    # strip \left \right and stray spacing artifacts
    d = d.replace("\\left", "").replace("\\right", "").replace("\\,", "")
    d = re.sub(r"\s+", "", d)
    if not d:
        return ("none", "")
    # named sets
    named = {
        "\\mathbb{R}": "R", "R": "R", "\\mathbb{R}^*": "R*",
        "\\mathbb{Z}": "Z", "Z": "Z",
        "\\mathbb{N}": "N", "N": "N",
        "\\mathbb{Q}": "Q", "Q": "Q",
        "\\mathbb{C}": "C", "C": "C",
    }
    if d in named:
        return ("set", named[d])
    # interval (a,b) [a,b] (a,b] [a,b)
    m = re.match(r"^([\[(])([^,]+),([^\])]+)([\])])$", d)
    if m:
        lo_open = m.group(1) == "("
        hi_open = m.group(4) == ")"

        def _ep(t):
            t = t.strip()
            if t in ("+\\infty", "\\infty", "+infty", "infty"):
                return sympy.oo
            if t in ("-\\infty", "-infty"):
                return -sympy.oo
            try:
                return sympy.sympify(sympy.parsing.latex.parse_latex(t)) if "\\" in t else sympy.sympify(t)
            except Exception:
                return None

        lo, hi = _ep(m.group(2)), _ep(m.group(3))
        if lo is not None and hi is not None:
            try:
                return ("interval", sympy.Interval(lo, hi, left_open=lo_open, right_open=hi_open))
            except Exception:
                return None
        return None
    # inequality-style domain (x>1): compare via the body grader
    if re.search(r"[<>]|\\geq|\\leq", d):
        return ("ineq", d)
    return None


def _verify_quantified(gold_str: str, pred_str: str) -> bool:
    """Semantic equality of two quantified propositions; False if unsure."""
    g = _parse_quantified(gold_str)
    p = _parse_quantified(pred_str)
    if g is None or p is None:
        return False
    gq, gv, gd, gb = g
    pq, pv, pd, pb = p
    if gq != pq:
        return False
    # bound variable must match textually: alpha-renaming would also have to
    # rename inside body/domain, and mismatched names in a student answer
    # usually mean a transcription slip -- stay strict (fail-closed).
    if gv != pv:
        return False
    # domain
    gn, pn = _norm_domain(gd), _norm_domain(pd)
    if gn is None or pn is None:
        if gd.replace(" ", "") != pd.replace(" ", ""):
            return False
    elif gn[0] != pn[0]:
        return False
    elif gn[0] == "interval":
        if gn[1] != pn[1]:
            return False
    elif gn[0] == "ineq":
        if not _grade_bodies(gn[1], pn[1]):
            return False
    else:
        if gn[1] != pn[1]:
            return False
    # body: relational formula -> math-verify's sympy inequality grading
    return _grade_bodies(gb, pb)


def _grade_bodies(gold_body: str, pred_body: str) -> bool:
    """Compare two relational LaTeX bodies with math-verify itself."""
    try:
        eg = parse("\\boxed{" + gold_body + "}", _GOLD_TARGETS, parsing_timeout=None)
        ep = parse("\\boxed{" + pred_body + "}", _GOLD_TARGETS, parsing_timeout=None)
        if not eg or not ep:
            return False
        # require a non-string (sympy) representation on each side: two
        # strings would just re-run the equality lottery this tier replaces
        import re

        def _norm_s(x):
            return re.sub(r"\s+|\\left|\\right", "", str(x))

        for g in eg:
            for p in ep:
                if not isinstance(g, str) or not isinstance(p, str):
                    if verify(g, p, timeout_seconds=None):
                        return True
                elif _norm_s(g) == _norm_s(p):
                    return True
        return False
    except Exception:
        return False


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        # Use parsing_timeout=None and timeout_seconds=None to disable
        # signal.alarm() which crashes in non-main threads (Ray workers).
        extracted_gold = parse(ground_truth_boxed, _GOLD_TARGETS, parsing_timeout=None)
        extracted_pred = parse(model_output, _PRED_TARGETS, parsing_timeout=None)
        if extracted_gold and extracted_pred:
            ret_score = max(
                1.0 if any(verify(g, p, timeout_seconds=None) for g in extracted_gold) else 0.0 for p in extracted_pred
            )
        # Quantified-proposition tier: only when standard grading said 0 AND
        # both sides degraded to strings that look like \exists/\forall
        # statements (sympy has no quantifier), compare them structurally.
        if ret_score == 0.0 and extracted_gold and extracted_pred:
            for g in extracted_gold:
                if not (isinstance(g, str) and ("\\exists" in g or "\\forall" in g)):
                    continue
                for p in extracted_pred:
                    if isinstance(p, str) and ("\\exists" in p or "\\forall" in p):
                        if _verify_quantified(g, p):
                            ret_score = 1.0
                            break
                if ret_score == 1.0:
                    break
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    return ret_score


def _mv_safe_inthread(answer: str) -> bool:
    """True if this boxed answer CANNOT drive sympy/gmpy2 to materialize a huge
    integer, so it is safe to grade in-thread with no subprocess. Only GIANT
    number shapes are routed to the killable subprocess -- ordinary powers like
    x^2 or 2^{10} (present in most MATH answers) stay in-thread, otherwise
    validation (thousands of concurrent grades) becomes a fork storm."""
    if answer is None or len(answer) > 500:
        return False
    if _re.search(r"\d{200,}", answer):          # 200+ digit literal
        return False
    if _re.search(r"\d{5,}\s*!", answer):        # factorial of >=10000
        return False
    if _re.search(r"\^\s*\{?\s*-?\d{6,}", answer):   # exponent >= 6 digits
        return False
    if _re.search(r"\^\s*\{[^}]*\^", answer):    # power tower: ^ inside exponent
        return False
    if "!" in answer and _re.search(r"[)}]\s*!", answer) and "^" in answer:
        return False                             # factorial of a power, e.g. (10^8)!
    return True


def _mv_child(reboxed, ground_truth, timeout_score, q):
    # Die with the parent (PDEATHSIG): if the reward worker is killed while we
    # grade, we must NOT survive as an orphan -- an orphaned giant-number grade
    # grew to 189GB and OOM-killed the whole training node (2026-07-09).
    try:
        _ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, _signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass
    # NOTE (2026-07-11): do NOT set RLIMIT_AS here. A forked child INHERITS
    # the parent's address space (a reward worker maps ~50GB virtual), so any
    # RLIMIT_AS below that is exceeded from birth -- the first malloc raises
    # MemoryError and even the result queue cannot be written, silently
    # zeroing EVERY forked grade (found by the value-asserting test). The
    # wall timeout + PDEATHSIG bound runtime and orphans; memory growth
    # within the 15s wall is at most ~1GB (bignum spins grow ~1GB/min).
    #
    # Silence ONLY math-verify's per-process "Timeout is disabled" warnings:
    # they fire once per process, and with fork-per-grade EVERY grade is a
    # fresh process -- 2 lines x 256 grades/step flooded the logs
    # (2026-07-11). The external 15s wall satisfies what the warning asks
    # for, so the message is moot here. Targeted filters only (review
    # feedback): other child warnings (sympy changes, parser issues,
    # resource warnings) must stay visible. Parent-side guard logs (wall
    # timeout culprits) are unaffected -- they are emitted by the parent.
    try:
        import warnings as _warnings
        _warnings.filterwarnings(
            "ignore", message=r".*Timeout is disabled.*")

        def _drop_timeout_disabled(record):
            return "Timeout is disabled" not in record.getMessage()

        root = _logging.getLogger()
        root.addFilter(_drop_timeout_disabled)
        for h in root.handlers:
            h.addFilter(_drop_timeout_disabled)
        for _n in list(_logging.root.manager.loggerDict):
            if _n.startswith("math_verify"):
                _logging.getLogger(_n).addFilter(_drop_timeout_disabled)
    except Exception:
        pass
    try:
        q.put(compute_score(reboxed, ground_truth, timeout_score=timeout_score))
    except BaseException:
        try:
            q.put(timeout_score)
        except Exception:
            pass


def _compute_score_subproc(reboxed, ground_truth, timeout_score, wall):
    """Grade in a forked child SIGKILLed after `wall`s -- the only way to bound
    a GIL-holding C spin (giant-number materialization): math-verify's own
    signal.alarm timeout is main-thread-only and thread watchdogs cannot
    preempt C code. Concurrency is bounded by _MV_FORK_SEM. NEVER falls back
    to grading in-thread: an answer routed here is a suspected giant-number
    spin, and grading it in-thread would wedge the whole step (2026-07-09)."""
    if not _MV_FORK_SEM.acquire(timeout=4 * wall):
        _MV_LOG.warning(
            "math-verify grader saturated (no fork slot in %.0fs); "
            "scored 0: answer=%r gt=%r", 4 * wall, reboxed[:200],
            str(ground_truth)[:100])
        return timeout_score                # grader saturated by degenerates
    p = None
    q = None
    try:
        ctx = _mp.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=_mv_child,
                        args=(reboxed, ground_truth, timeout_score, q),
                        daemon=True)
        p.start()
        try:
            result = q.get(timeout=wall)
        except Exception:
            # THE culprit-sample log (2026-07-11): every grade that outruns
            # the wall is exactly the pathological answer we could never
            # identify during the freezes -- make it visible.
            _MV_LOG.warning(
                "math-verify wall timeout (%.0fs), scored 0: answer=%r gt=%r",
                wall, reboxed[:200], str(ground_truth)[:100])
            result = timeout_score          # spun past the deadline
        return result
    except Exception:
        return timeout_score
    finally:
        try:
            if p is not None:
                if p.is_alive():
                    p.kill()
                p.join(5)
        except Exception:
            pass
        # Explicit queue teardown (2026-07-11 review): every grade creates a
        # Queue = a feeder thread + a pipe (2 FDs) in the PARENT; without
        # close()+join_thread() they linger until GC and can accumulate over
        # a long run's ~hundreds of thousands of grades.
        try:
            if q is not None:
                q.close()
                q.join_thread()
        except Exception:
            pass
        _MV_FORK_SEM.release()


def compute_score_boxed(solution_str, ground_truth, timeout_score: float = 0) -> float:
    """Boxed-gated Math-Verify scoring (verl-fol default since 2026-07-05).

    The plain compute_score lets math-verify's ExprExtractionConfig pick up
    ANY expression in the response, so a model that never writes \\boxed{}
    can still score 1.0. Our training format (math_reasoning.txt) requires
    the final answer in \\boxed{}; the XML-step penalty path and the Isabelle
    outcome check both assume it. This wrapper keeps that contract: no boxed
    answer -> 0 (fail-closed), otherwise sympy equivalence via math-verify.

    Answers with a power/factorial/giant-literal are graded in a killable
    SUBPROCESS: math-verify's own timeout is disabled in Ray worker threads, so
    an answer like \\boxed{2^{100000000}} would otherwise spin gmpy2 for hours
    (2026-07-09 wedge). Safe answers grade in-thread (no fork overhead).
    """
    from .math_reward import last_boxed_only_string, remove_boxed

    try:
        boxed = last_boxed_only_string(solution_str)
        if boxed is None:
            return 0.0
        answer = remove_boxed(boxed)
    except Exception:
        return 0.0
    # Re-wrap so math-verify's LatexExtractionConfig parses exactly the boxed
    # payload, not stray expressions elsewhere in the response.
    reboxed = "\\boxed{" + answer + "}"
    # EVERY grade goes through the killable subprocess (2026-07-11). The
    # in-thread fast path for "safe-looking" answers was a hole: the step-134
    # batch produced answers that pass the giant-number predicate yet still
    # spin sympy for 40+ minutes in C (GIL held, uninterruptible) -- the exact
    # class the fork exists to bound. A fork costs ~0.1s per grade
    # (~seconds/step through the 4-slot semaphore), which is cheap insurance
    # for a hard 15s bound on EVERY answer shape.
    return _compute_score_subproc(reboxed, ground_truth, timeout_score, _MV_WALL_TIMEOUT)
