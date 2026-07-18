"""Configuration defaults for the Isabelle server pool."""
import os
import re

ISABELLE_HOME = os.environ.get(
    "ISABELLE_HOME", "/2022533109/zhouchuyan/isabelle/Isabelle2025")
ISABELLE_BIN = f"{ISABELLE_HOME}/bin/isabelle"

SESSION = "Isa_Step"
# quick_and_dirty=false (2026-07-17 merge): the session serves BOTH the general path and the
# direct-domain (group/ring/field) checks, and nothing emits sorry, so the kernel enforces that
# no unchecked proof path exists. The watchdog options match what the former HOL-Algebra pools ran.
SESSION_OPTIONS = [
    "quick_and_dirty=false",
    "headless_consolidate_delay=0.02",
    "headless_check_delay=0.02",
    "headless_watchdog_timeout=15",
]
THEORY_IMPORTS = """  imports
    "Isa_Step.Isa_Step_Base"
"""
PURGE_EVERY = 10          # checks per worker between purge_theories all=true
VERIFY_TIMEOUT = 60.0     # seconds per use_theories call before worker restart
BANNER_RE = re.compile(r'server "(.+?)" = ([\d.]+):(\d+) \(password "(.+?)"\)')
