"""Pipeline stages of the Isabelle step verifier.

engine.process_one_response runs formalization.formalize -> preparation.prepare -> verification.verify_response under engine.py's orchestration; each stage split out of engine.py lives here as one module. direct_verify is the direct-domain (group/ring/field) checking path, dispatched per step by verify_response, with its own session and theorem builder; its assumptions carry the same-domain chain and the injected general conclusions. Form naming (nl_* / pyexpr_* / isabelle_*) is defined at the top of ../state_classes.py.
"""
