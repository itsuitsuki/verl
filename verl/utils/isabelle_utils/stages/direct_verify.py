"""Direct Isabelle verification for domains that pyexpr cannot express.

pyexpr covers arithmetic only. For domains with separate vocabularies, such as group, ring, and field theory, the translator emits assumptions and one claim but no proof. The engine tries a fixed ordered list of tactics, so translated text cannot introduce `sorry` or another unchecked proof.

Each domain is described by a `DomainSpec`. Verification applies the same structure and provenance requirements to every domain:

* reject proof, definition, and axiom commands in translated statements;
* require every non-vocabulary claim identifier to occur in the source step;
* require every numeral in the claim and its own premises to occur in the source step (0, 1, 2 free);
* require the claim to be provable with its premises but not premise-free (the negative check runs the same tactic battery as the positive proof);
* reject inconsistent premises and incomplete consistency checks.

Injected chain conclusions (earlier same-family claims and general-chain conclusions, admitted at translation time like the general s*) join the consistency probe and the with-premises proof as explicit theorem assumptions; never as kernel facts, never as sorry-admitted lemmas. They are exempt from this step's entity anchor (anchored at their origin step), but a claim that merely RESTATES an injected conclusion is rejected as claim_repeats_chain: the chain admits translations before verification, so the twin may be unverified, and repetition is never new work.

The Isabelle kernel runs with `quick_and_dirty=false`. The non-triviality requirement rejects premise-free tautologies such as `a = a`, and the consistency check rejects ex-falso proofs from assumptions that already prove `False`. A tactic cannot prove a false goal, so accepted claims have no unchecked proof path.
"""
import re
from dataclasses import dataclass

import verl.utils.isabelle_utils.state_classes as state_classes


class DirectVerifyError(ValueError):
    """The statement has a banned construct or is malformed -> caller scores the step `x`."""


@dataclass
class DomainSpec:
    name: str          # short label, e.g. "group"
    locale: str        # locale target for the theorem, e.g. "group" ("" = none)
    claim_tactics: list  # tactics tried in order until one proves the claim
    freecheck: str     # one fast-decisive tactic (simp) for the premise-free non-triviality check
    vocab: set         # fixed operation/predicate names, exempt from the entity anchor
    chain_family: str  # conclusion-sharing family: ring and field share "algebra" (same signature, locale-compatible statements); group is its own structure


# Proof, definition, and axiom keywords are invalid because the translator may emit only a proposition.
_BANNED = ("sorry oops axiomatization consts definition fun primrec function nitpick quickcheck "
           "undefined termination declare unfolding apply done proof qed by using theory begin "
           "end lemma theorem instance interpretation").split()
_BANNED_RE = re.compile(r"\b(?:%s)\b" % "|".join(_BANNED))
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
_SYMBOL_RE = re.compile(r"\\<[A-Za-z]+>")   # \<otimes>, \<in>, ... -- strip before reading idents
_NUM_RE = re.compile(r"\b\d+\b")


def _ids(text):
    """Identifiers in `text`, with Isabelle symbol tokens stripped first."""
    return set(_IDENT_RE.findall(_SYMBOL_RE.sub(" ", text or "")))


def _stmt_forms(text):
    """Canonical comparison forms of one direct-path statement, for the restatement guards: whitespace collapsed, top-level ``\\<and>`` conjuncts split by a paren-depth scan, bare ``True`` conjuncts dropped, and a conjunct with exactly one top-level ``=`` normalized so ``a = b`` and ``b = a`` compare equal. Logical equivalence in general needs a parser these verbatim statements do not have; these are the cheap textual disguises (swapped equality, a conjoined True) worth closing."""
    def top_level_conjuncts(t):
        parts, cur, depth, i = [], [], 0, 0
        while i < len(t):
            ch = t[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and t.startswith("\\<and>", i):
                parts.append("".join(cur))
                cur = []
                i += len("\\<and>")
                continue
            cur.append(ch)
            i += 1
        parts.append("".join(cur))
        return parts

    forms = set()
    for part in top_level_conjuncts(text or ""):
        ws = " ".join(part.split())
        if not ws or ws == "True":
            continue
        eq_positions, depth = [], 0
        for i, ch in enumerate(ws):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "=" and depth == 0:
                eq_positions.append(i)
        if len(eq_positions) == 1:
            left = ws[: eq_positions[0]].strip()
            right = ws[eq_positions[0] + 1:].strip()
            forms.add(" = ".join(sorted((left, right))))
        else:
            forms.add(ws)
    return frozenset(forms)


def sanitize(assumptions, claim):
    """Reject any banned construct; return the identifier set of the claim."""
    if not isinstance(claim, str) or not claim.strip():
        raise DirectVerifyError("empty claim")
    banned = _BANNED_RE.search(_SYMBOL_RE.sub(" ", " ".join(list(assumptions) + [claim])))
    if banned:
        raise DirectVerifyError("banned construct: %r" % banned.group(0))
    return _ids(claim)


def make_theorem(spec, assumptions, claim, tactic, with_premises):
    """Return theorem text for the domain-specific pool. `with_premises=False` removes assumptions for the non-triviality check."""
    target = "(in %s) " % spec.locale if spec.locale else ""
    if with_premises and assumptions:
        head = "  assumes " + " and ".join('"%s"' % a for a in assumptions) + "\n"
        using = "using assms "
    else:
        head = using = ""
    return 'theorem %sclaim:\n%s  shows "%s"\n  %sby %s' % (target, head, claim, using, tactic)


def verify(spec, assumptions, claim, source_text, check, previous_conclusions=(), general_conclusions=()):
    """Return `(rewarded, reason)` after checking the theorem in the domain-specific pool. Incomplete checks are rejected.

    `previous_conclusions` are earlier same-family steps' claims (ring and field share one chain family; group is separate); `general_conclusions` are earlier general-chain conclusions (transpiled pv_ arithmetic, collision-free with the domain vocabulary). Both are admitted at translation time exactly like the general chain's s* premises: anchored at their origin, they join only the consistency probe and the with-premises proof, never this step's entity anchor, never the premise-free non-triviality check. A claim equal to an injected conclusion is rejected (claim_repeats_chain), the direct-path analog of claim_repeats_earlier.
    """
    try:
        claim_ids = sanitize(
            list(general_conclusions) + list(previous_conclusions) + list(assumptions), claim)
    except DirectVerifyError as e:
        return False, "sanitize:%s" % e

    # No OWN premises may restate the claim. The translator is untrusted input, so a step written as `assumes "C" shows "C"` would otherwise pass every check (consistent premises, C provable from assms by simp) while proving nothing. This is the direct-path analog of the general path's premise==conclusion guard (preparation compares `ast.dump`); direct-path terms are Isabelle strings with no AST, so compare on _stmt_forms (collapsed whitespace, top-level conjunct split, orientation-normalized equality) and by COVERAGE, not exact equality: a claim whose every conjunct form appears among the own premises' conjunct forms (assume `C \<and> D`, show `C`; or assume C and D separately, show `C \<and> D`) is fully assumed and proves nothing either.
    claim_forms = _stmt_forms(claim)
    own_premise_forms = frozenset().union(*(_stmt_forms(a) for a in assumptions)) \
        if assumptions else frozenset()
    if claim_forms and claim_forms <= own_premise_forms:
        return False, "claim_repeated_as_premise"

    # A claim covered by the injected chain conclusions' conjunct forms is the direct-path s*-chain echo (2026-07-17 decision, replacing the earlier chain-reuse exemption): the chain admits statements at translation time, so the twin may itself be unverified, and repetition is no new work. Coverage over the UNION of all injected conclusions (and the own premises) catches a claim reassembled from several of them. Same fate as the general path's claim_repeats_earlier.
    injected = list(previous_conclusions) + list(general_conclusions)
    chain_forms = frozenset().union(*(_stmt_forms(x) for x in injected)) \
        if injected else frozenset()
    if claim_forms and claim_forms <= (chain_forms | own_premise_forms) \
            and claim_forms & chain_forms:
        return False, "claim_repeats_chain"

    # Entity anchor: every non-vocabulary, non-numeric name must appear in the source step text. The claim and the premises are anchored separately so a fabricated premise entity (an element the step never introduced) is distinguishable from a fabricated claim entity. This gives premises the same provenance the general path requires of admitted premises (their numbers must trace to the problem/givens/earlier conclusions).
    src_ids = _ids(source_text)

    def _invented(ids):
        return sorted(i for i in ids
                      if i not in spec.vocab and not i.isdigit() and i not in src_ids)

    claim_invented = _invented(claim_ids)
    if claim_invented:
        return False, "invented:" + ",".join(claim_invented)
    premise_ids = set()
    for a in assumptions:
        premise_ids |= _ids(a)
    premise_invented = _invented(premise_ids)
    if premise_invented:
        return False, "invented_premise:" + ",".join(premise_invented)

    # Numeral provenance, the direct-path analog of the general path's number windows: every numeral in the claim or an OWN premise must appear in the step's own text (0, 1, 2 stay free, like the general allow-list). Injected chain conclusions are exempt (anchored at their origin step). Without this the entity anchor lets the translator pin any quantity the model never stated (`order G = 15` from a step that says no number).
    src_nums = set(_NUM_RE.findall(source_text or "")) | {"0", "1", "2"}
    claim_num_invented = sorted(
        {m for m in _NUM_RE.findall(claim) if m not in src_nums})
    if claim_num_invented:
        return False, "invented_numeral:" + ",".join(claim_num_invented)
    premise_num_invented = sorted(
        {m for m in _NUM_RE.findall(" ".join(assumptions)) if m not in src_nums})
    if premise_num_invented:
        return False, "invented_premise_numeral:" + ",".join(premise_num_invented)

    def _outcome(theorem):
        """(proved, fail_closed) for one prover call. `check` may be the engine's PoolVerifier (typed VerificationOutcome) or a contract-test fake (legacy two-key dict); from_raw folds both here, so no adapter sits between the stages and the pool."""
        classified = state_classes.VerificationOutcome.from_raw(check(theorem))
        return (classified.outcome is state_classes.ProofOutcome.PROVED,
                classified.fail_closed)

    # Reject contradictory or unchecked premises before proving the claim. The probe covers everything injected: a poisoned earlier conclusion (same-family or general) fails every later consuming step closed.
    # The probe runs the freecheck AND every claim tactic against False: a claim tactic that could exploit an inconsistency for an ex-falso proof must find that same inconsistency here first, so consistency established by this loop rules out ex-falso routing by construction. The freecheck alone would miss a contradiction only a stronger tactic (metis, algebra) can see.
    chained_assumptions = list(general_conclusions) + list(previous_conclusions) + list(assumptions)
    if chained_assumptions:
        for probe_tactic in [spec.freecheck] + [t for t in spec.claim_tactics
                                                if t != spec.freecheck]:
            probe_proved, probe_fail_closed = _outcome(
                make_theorem(spec, chained_assumptions, "False", probe_tactic, True))
            if probe_proved:
                return False, "inconsistent"
            if probe_fail_closed:
                return False, "premise_consistency_unknown"

    # C1: prove the claim with its premises using the ordered tactic list.
    if not any(_outcome(make_theorem(spec, chained_assumptions, claim, t, True))[0]
               for t in spec.claim_tactics):
        return False, "not_proved"

    # C2: it must NOT prove premise-free (else it is trivial / a blank-fill vacuity). The negative check runs the SAME tactic battery as the positive proof: a claim that simp cannot close premise-free but metis or algebra can is exactly as vacuous, so a weaker check would let blank-fill claims through. Any ambiguous outcome fails closed.
    any_unknown = False
    for free_tactic in [spec.freecheck] + [t for t in spec.claim_tactics
                                           if t != spec.freecheck]:
        free_proved, free_fail_closed = _outcome(
            make_theorem(spec, [], claim, free_tactic, False))
        if free_proved:
            return False, "trivial"
        any_unknown = any_unknown or free_fail_closed
    if any_unknown:
        return False, "freecheck_unknown"
    return True, "rewarded"


# Domain registry (add a domain = add a DomainSpec without new logic)

GROUP_SPEC = DomainSpec(
    name="group",
    locale="group",
    claim_tactics=["simp", "auto",
             "(simp add: m_assoc l_inv r_inv inv_inv inv_mult_group l_one r_one)",
             "(metis inv_mult_group inv_inv m_assoc l_inv r_inv l_one r_one m_closed)",
             "(rule lagrange)", "force"],
    freecheck="simp",
    # G is the locale carrier and stays vocabulary; H, K, N are PROBLEM entities (a subgroup, a kernel) and must appear in the step's own text to pass the entity anchor, so they are deliberately not exempt.
    vocab={"G", "carrier", "inv", "one", "mult", "order", "card", "rcosets",
           "lcosets", "subgroup", "normal", "ord", "monoid", "group", "generate", "kernel",
           "hom", "iso", "cong", "class", "the_elem", "True", "False", "Suc", "nat", "int",
           "real"},
    chain_family="group",
)

# Ring and field claims select the `ring` or `field` locale and use additive vocabulary such as \<zero>, \<oplus>, and \<ominus>.
# Measurements on dt3 selected these tactic orders: ring facts close with simp, distributivity lemmas, or algebra; field inverses additionally need field_Units to move a nonzero premise into the unit group.
_ALGEBRA_VOCAB = {"R", "G", "carrier", "inv", "one", "zero", "mult", "add",
                  "Units", "order", "card", "subgroup", "ideal", "monoid", "group", "ring",
                  "field", "domain", "hom", "iso", "the_elem", "True", "False", "Suc",
                  "nat", "int", "real"}

RING_SPEC = DomainSpec(
    name="ring",
    locale="ring",
    claim_tactics=["simp", "auto", "(simp add: l_distr r_distr)",
             "(simp add: ring.ring_simprules)", "(metis ring.ring_simprules)", "algebra"],
    freecheck="simp",
    vocab=_ALGEBRA_VOCAB,
    chain_family="algebra",
)

FIELD_SPEC = DomainSpec(
    name="field",
    locale="field",
    claim_tactics=["(simp add: field_Units)", "simp", "auto",
             "(metis field_Units)", "(simp add: ring.ring_simprules)", "algebra"],
    freecheck="simp",
    vocab=_ALGEBRA_VOCAB,
    chain_family="algebra",
)

# Every domain checks in the single Isa_Step session (its base theory imports HOL-Algebra.Coset / Multiplicative_Group / Ring since the 2026-07-17 merge); a DomainSpec now only selects the locale, the tactic battery, and the vocabulary. match_domain() picks one from the vocabulary used by the translated statement. Adding another domain requires one DomainSpec, a match_domain() condition, and the theories in Isa_Step_Base.


def match_domain(*texts):
    r"""Return the DomainSpec whose vocabulary appears in any of `texts`, else None.

    Routing among the algebra locales is by which structure the vocabulary implies, checked on
    the WHOLE statement (claim + premises together), most specific first:
      field: has a multiplicative inverse `inv` AND a zero (\<zero> or a `\<noteq> \<zero>`
                premise): only a field has both a mult. inverse and additive structure;
      ring: additive ring vocabulary (\<oplus>, \<ominus>, \<zero>, ideal, "ring"), or the
                carrier named R (the translator prompt reserves R for rings/fields, G for
                groups, so `carrier R` identifies the algebra family even in a purely
                multiplicative statement);
      group: multiplicative carrier vocabulary (\<otimes>, carrier, subgroup, cosets), or a
                standalone arithmetic-valued group function applied to an entity (`order G`,
                `ord a`; the APPLICATION form with a space, so the general path's pyexpr
                `order_G == 15` and a variable named ord never match).
    General-path pyexpr statements contain none of these, so they return None."""
    blob = " ".join(t for t in texts if t)
    has_inv = re.search(r"\binv\b", blob)
    has_zero = re.search(r"\\<zero>|field", blob)
    if re.search(r"\bfield\b", blob) or (has_inv and has_zero):
        return FIELD_SPEC
    if re.search(r"\\<oplus>|\\<ominus>|\\<zero>|\bideal\b|\bring\b|\bcarrier R\b", blob):
        return RING_SPEC
    if re.search(r"\\<otimes>|\bcarrier\b|\bsubgroup\b|\brcosets\b"
                 r"|\border [A-Za-z_]|\bord [A-Za-z_]", blob):
        return GROUP_SPEC
    return None
