"""State classes for Isabelle step verification & process rewarding.

Naming contract for the three forms a step passes through: (see engine.py)
A. natural language (nl_*): the model's own step text, parsed from the response XML;
B. constrained pyexpr (pyexpr_*): the translator's output; direct-domain (group/ring/field) statements skip this form and stay verbatim Isabelle;
C. Isabelle (isabelle_*): transpiled text ready for the prover. VARS declarations become isabelle_fixes (a theorem header, never premises);
    isabelle_premises collects the problem GIVENs, earlier general-chain conclusions (bridged direct claims included), the step's own admitted premises, its definitions, and (joined at check time) proven nonzero conditions.

The per-step classes follow the same axis: NaturalLanguageStep (form A) -> PyExprGiven/PyExprStep (form B; DirectDomainStep for the skip branch) -> IsabelleStep (form C, just before the prover) -> StepVerificationResult (after the prover).
"""

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction


@dataclass
class NaturalLanguageStep:
    """One step of the model's answer as written (form A, natural language): the <premise> lines, the first <conclusion>, and the full text inside the <step> tags. Mutable because the corrupt-mode negative test edits a conclusion in place."""

    nl_premises: list
    nl_conclusion: str
    nl_step_text: str


@dataclass
class PyExprGiven:
    """The translated problem conditions: the VARS variable declarations and the GIVEN facts as constrained pyexpr sources.

    pyexpr_variable_types holds (name, "int"|"real") pairs; the parser already folds nat to int. pyexpr_givens holds the pyexpr source strings. validate_givens auto-declares undeclared identifiers by appending to pyexpr_variable_types in place, and the translation caches store the object after validation, so cached entries carry those declarations too.
    """

    pyexpr_variable_types: list
    pyexpr_givens: list


@dataclass(frozen=True)
class PyExprStep:
    """One reasoning step's form-B (constrained pyexpr) translation: its conclusion and its stated premises as constrained pyexpr sources.

    A step recognized as a direct-domain Isabelle statement stays verbatim in these fields: the parser routes on the WHOLE step (conclusion and premises together), and formalize turns such a step into a DirectDomainStep with the same whole-step routing.
    """

    pyexpr_conclusion: str
    pyexpr_premises: list

    @property
    def direct(self):
        """True when the conclusion ALONE is a direct-domain (group/ring/field) Isabelle statement kept verbatim rather than a pyexpr source. Routing is whole-step (conclusion and premises together), so a step can route direct while this stays False, e.g. an arithmetic-valued claim like `order G = 15` riding with carrier premises: use this as a conclusion-shape hint only, never as the routing decision."""
        from verl.utils.isabelle_utils.stages import direct_verify
        return direct_verify.match_domain(self.pyexpr_conclusion) is not None


@dataclass(frozen=True)
class DirectDomainStep:
    """A step written in a direct domain's own Isabelle vocabulary (group/ring/field),
    parsed out by the formalization stage:
    it skips form B (constrained pyexpr) entirely (premises and claim stay the translator's verbatim Isabelle)
    and is checked in the domain's own session.

    Its check-time assumptions are its own stated premises plus the injected chains (previous_conclusions, general_conclusions),
    all admitted at translation time as UNVERIFIED theorem hypotheses, exactly like the general chain's s* premises:
    the prover verifies the implication premises ==> claim, never the premises themselves
    (no kernel fact, no sorry; the pool runs quick_and_dirty=false),
    and the domain pool's consistency probe rejects a contradictory assumption set.
    A whitelisted arithmetic-valued claim (`order G = 15`) is in turn bridged into later general steps' s* premises."""

    spec: object          # the matched direct_verify.DomainSpec
    premises: list        # verbatim Isabelle premise statements
    claim: str            # verbatim Isabelle claim: the statement to prove
    nl_step_text: str     # the model's own step text (form A, natural language); entity anchoring reads it
    # Earlier same-family steps' claims (ring and field share one chain family; group is separate), filled by preparation (empty as built by formalization): translation-time admission, symmetric with the general chain's s*; direct_verify feeds them to the consistency probe and the with-premises proof only.
    previous_conclusions: tuple = ()
    # Earlier conclusions expressible in the general (Isa_Step) session (transpiled pv_ terms: general s* terms and earlier bridged direct claims), filled by preparation: the general->direct half of the cross-session bridge, admitted at translation time and handled by direct_verify exactly like previous_conclusions.
    general_conclusions: tuple = ()


@dataclass
class FormalizationOutput:
    """The formalization stage's output for one response: the parsed steps (form A, natural language),
    the validated form-B (constrained pyexpr) translation of the givens,
    and the ordered `steps` list holding each response step as its typed alternative.
    Built by formalization.formalize; the preparation stage consumes it. Engine-internal; never serialized or cached.

    givens_ok / steps_ok mirror the solution record's flags, and the fields of a translation that did not complete stay None. translation_record_from_problem / translation_record_from_steps carry the translator attempt logs in the exact shape the record stores them (the steps record unwraps a single chunk).
    """

    nl_steps: list        # NaturalLanguageStep items, after truncation and corrupt-mode edits
    problem_nums: set     # numbers admissible from the problem text: literals, subscript indices, range interiors, 0/1/2
    translation_record_from_problem: list | None = None
    translation_record_from_steps: list | None = None
    givens_ok: bool = False
    steps_ok: bool = False
    pyexpr_givens: list | None = None
    pyexpr_variable_types_declared: dict | None = None  # VARS declarations plus validate-time auto-declares
    # One entry per response step, in step order: a general step is the parser's PyExprStep; a step written in a direct domain's vocabulary is a DirectDomainStep. The single source of truth; no placeholder entries, no side table.
    steps: list | None = None
    # Per-step lists from the authoritative transcription pass, aligned with `steps`. The identical computation inside validate_props only feeds the translator retry loop; soft-accepted misses and disk-cache hits reach the record through THIS field.
    transcription_missing: list | None = None

class PremiseConsistency(Enum):
    """Outcome of trying to derive False from accumulated premises."""

    CONSISTENT = "consistent"
    UNKNOWN = "unknown"
    INCONSISTENT = "inconsistent"


class PremiseSource(Enum):
    """Logical origin of an admitted constrained-pyexpr assumption."""

    PROBLEM = "problem"
    PREVIOUS_CONCLUSION = "previous_conclusion"
    STEP = "step"
    DEFINITION = "definition"


@dataclass(frozen=True)
class AdmittedPyExprSource:
    """One admission-passed constrained-pyexpr source and its named Isabelle assumption."""

    source_kind: PremiseSource
    name: str
    pyexpr_source: str
    isabelle_term: str


@dataclass(frozen=True)
class TrigValueEvidence:
    """An admitted equality assigning a value to one trigonometric application."""

    source_names: tuple[str, ...]
    function: str
    angle_key: str
    angle_source: str
    angle_term: str
    value_source: str
    value_term: str


@dataclass(frozen=True)
class TrigSignEvidence:
    """An admitted strict sign assertion for one trigonometric application."""

    source_names: tuple[str, ...]
    function: str
    angle_key: str
    angle_source: str
    angle_term: str
    positive: bool


@dataclass(frozen=True)
class PiBoundEvidence:
    """An admitted lower or upper angle bound that is an exact rational multiple of pi."""

    source_names: tuple[str, ...]
    angle_key: str
    angle_source: str
    angle_term: str
    side: str
    coefficient: Fraction
    strict: bool
    proposition: str


@dataclass(frozen=True)
class TrigContext:
    """Structured trigonometric evidence extracted from admitted assumptions."""

    values: tuple[TrigValueEvidence, ...] = ()
    signs: tuple[TrigSignEvidence, ...] = ()
    pi_bounds: tuple[PiBoundEvidence, ...] = ()
    ambiguous_value_keys: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PremiseConsistencyBoundaries:
    """First step indices where consistency is unknown or inconsistent."""

    inconsistent_from_step: int | None = None
    unknown_from_step: int | None = None

    def status_for(self, step: int) -> PremiseConsistency:
        if self.inconsistent_from_step is not None and step >= self.inconsistent_from_step:
            return PremiseConsistency.INCONSISTENT
        if self.unknown_from_step is not None and step >= self.unknown_from_step:
            return PremiseConsistency.UNKNOWN
        return PremiseConsistency.CONSISTENT

@dataclass
class IsabelleStep:
    """One general-path step's complete form-B (constrained pyexpr) -> form-C (Isabelle) conversion, ready for Isabelle checking.

    Verification input only: the guards, counts, and transcription record computed during preparation ride here, and the verification stage copies them onto the StepVerificationResult it creates. Preparation creates no result objects.
    """

    k: int                # step index (0-based)
    # -- form B: the step as constrained pyexpr --
    pyexpr_premises: list    # the step's own stated premises (source strings)
    pyexpr_conclusion: str   # the step's whole translated conclusion (source string)
    pyexpr_definitions: list  # definition conjuncts split from the conclusion (AST nodes); conservative extensions that become assumptions, not obligations
    pyexpr_claims: list      # claim conjuncts of the conclusion (AST nodes); tolerance / integral recipes need them
    claim_restates_given: list  # per claim: True when it textually equals a GIVEN (auto-satisfied)
    pyexpr_variable_types: dict  # identifier -> "int" | "real" across the whole response (declared + inferred)
    # -- form C: the step as Isabelle text. The four premise sources are held separately so code never infers origin from the g/s/p/d label prefixes; isabelle_premises assembles them. --
    isabelle_fixes: list     # (name, type) declarations for the theorem header (the VARS; never premises)
    isabelle_problem_premises: list      # (g{i}, term): the transpiled GIVEN conditions
    isabelle_previous_conclusions: list  # (s{j}, term): earlier steps' general-session transpiled conclusions (general steps' conclusions and whitelisted bridged direct claims; non-bridgeable direct slots skipped)
    isabelle_step_premises: list         # (p{k}_{i}, term): this step's own premises, admitted with provenance
    isabelle_definitions: list           # (d{k}_{i}, term): this step's transpiled definition conjuncts
    isabelle_claims: list    # transpiled claim terms -- the proof obligations
    isabelle_nonzero_divisors: list  # divisor terms to prove nonzero; each proven one joins the premises for the claim tactics
    # -- tactic routing --
    numeral_carrier: str     # numeral sort of the conclusion: "int" | "real" | "mixed"
    nonlinear: bool          # nonlinear given/premise/claim present -> sos rescue eligible
    linear_premises: bool    # entire premise chain affine -> LINEAR_FALSE decides consistency
    linear_step: bool        # premises AND claims affine -> LINEAR_CLAIM applies
    admitted_pyexpr_sources: tuple[AdmittedPyExprSource, ...]  # admission-passed form-B sources bound explicitly to their form-C assumption names and source kinds
    trig_context: TrigContext  # structured trig values, signs, and exact pi bounds extracted from admitted_pyexpr_sources; theorem text is built only during verification
    # -- verification input computed during preparation; the verification stage copies these onto this step's StepVerificationResult --
    transcription_missing: list = field(default_factory=list)  # numbers the NL conclusion mentions but the translation omits (from the formalization stage's authoritative pass)
    guard_invented: list = field(default_factory=list)  # translated numbers with no provenance (not problem/givens/earlier conclusions/step text)
    guard_ok: bool = True
    n_definitions: int = 0
    n_admitted_premises: int = 0
    # Every claim conjunct restates an earlier admitted conclusion (an s*-chain echo). The chain admits translations BEFORE verification, so the echo would prove trivially from its injected twin even when the twin's own step earned nothing; repetition is never new work, so the step is never rewarded and surfaces as x. An echo whose conjuncts mix chain and given restatements lands here too.
    claim_repeats_earlier: bool = False
    # Every claim conjunct restates a problem GIVEN (the g*-side echo): true by assumption but contentless, so the step is never rewarded and surfaces as x. A restated conjunct inside an otherwise novel claim is not this; it stays auto-satisfied per claim_restates_given while the novel conjuncts verify.
    claim_repeats_given: bool = False
    # Mandatory tangent definedness conditions: one (angle term, decomposition term or None) pair per value-sensitive tangent application in the claim conjuncts AND in the model-authored assumptions the theorem admits (definitions, admitted step premises, earlier chain conclusions; only the structural identity rewrites and assumption-side nonzero tangent values carry no condition, see trigonometry.tangent_definedness_conditions). A None angle term is untranspilable and fails closed; the decomposition, present for non-base literal pi multiples, lets the cosine-nonzero check prove the condition through literal_condition_theorem. HOL totalizes tan (tan(pi/2) evaluates to 0 in the kernel), so verification builds `cos <angle> \<noteq> 0` from each pair and must PROVE it before any claim attempt runs, whichever tactic family would prove the claim; an unprovable condition fails the step closed.
    isabelle_tan_conditions: list = field(default_factory=list)
    # The ready-to-submit consistency probe (goal False over isabelle_premises). None when the premises contain dangerous terms: the orchestrator scores them UNKNOWN without a prover call.
    premise_consistency_theorem: str | None = None
    # Set once the consistency probes resolve; None until then.
    premise_consistency: PremiseConsistency | None = None

    @property
    def isabelle_premises(self):
        """The full assumption list in theorem order: problem GIVENs, earlier general-chain conclusions (including bridged direct claims), this step's admitted premises, its definitions. Proven nonzero conditions (nz*) join only at check time."""
        return (self.isabelle_problem_premises
                + self.isabelle_previous_conclusions
                + self.isabelle_step_premises
                + self.isabelle_definitions)


@dataclass
class PreparationOutput:
    """The preparation stage's output for one response: the ordered `steps` list (an IsabelleStep per general step; a DirectDomainStep passes through unchanged) plus the response-wide products the verification stage reads. This list is the complete set of steps the verification stage will check; no placeholder entries."""

    steps: list              # per response step, in order: IsabelleStep (general) | DirectDomainStep
    isabelle_fixes: list     # response-wide (name, type) declarations; every IsabelleStep shares this object
    function_arities: dict   # uninterpreted function name -> arity; the verification stage's SMT dispatch condition keys on non-emptiness
    isabelle_problem_conditions: list  # transpiled GIVEN terms (debug dump)
    isabelle_step_conclusions: list    # (term, carrier) per step, None at direct-domain slots (debug dump)


@dataclass
class StepVerificationResult:
    """One step's final verification and reward decision,
    with the reasons (guards, transcription, premise consistency, direct-domain outcome).
    Created by the verification stage (verify_response) and serialized to the solution record's step-list dict at the very end.

    Conditional keys in to_dict (tolerance, premise_consistency_unknown, claim_repeats_earlier, claim_repeats_given, domain_reason) appear only when their condition holds; the same presence rules the free dict had.
    """

    step: int
    neutral: bool = False
    transcription_missing: list = field(default_factory=list)
    guard_invented: list = field(default_factory=list)
    guard_ok: bool = True
    n_definitions: int = 0
    n_admitted_premises: int = 0
    premise_consistency_inconsistent: bool = False
    premise_consistency_unknown: bool = False
    verified: bool = False
    rewarded: bool = False
    tolerance: bool = False
    claim_repeats_earlier: bool = False
    claim_repeats_given: bool = False
    domain_reason: str | None = None

    def to_dict(self):
        out = {"step": self.step, "neutral": self.neutral,
               "transcription_missing": list(self.transcription_missing),
               "guard_invented": list(self.guard_invented),
               "guard_ok": self.guard_ok,
               "n_definitions": self.n_definitions,
               "n_admitted_premises": self.n_admitted_premises,
               "premise_consistency_inconsistent":
                   self.premise_consistency_inconsistent,
               "verified": self.verified,
               "rewarded": self.rewarded}
        if self.premise_consistency_unknown:
            out["premise_consistency_unknown"] = True
        if self.tolerance:
            out["tolerance"] = True
        if self.claim_repeats_earlier:
            out["claim_repeats_earlier"] = True
        if self.claim_repeats_given:
            out["claim_repeats_given"] = True
        if self.domain_reason is not None:
            out["domain_reason"] = self.domain_reason
        return out


@dataclass
class VerificationOutput:
    """The verification stage's output for one response: one StepVerificationResult per response step (general and direct-domain, in step order), the premise-consistency boundaries, and the per-step pattern string. process_one_response copies these onto the ResponseVerificationResult."""

    steps: list                               # StepVerificationResult items in step order
    boundaries: PremiseConsistencyBoundaries  # first inconsistent / unknown general step indices
    pattern: str                              # per-step symbols o/c/u/m/g/x (legend in verification.verify_response)


@dataclass
class ResponseVerificationResult:
    """The verification result of one whole response: identity, format and translation status, every step's result, the o/x/c/u/m/g pattern, and profiling. 
    Built during process_one; to_dict at the engine boundary produces the exact record shape downstream consumers and the characterization tests read.

    Key-presence rules preserved from the free dict: translation_record_from_problem appears once the givens translation ran, translation_record_from_steps once the step translation ran, premise_consistency_unknown_at only when some step's premises were undecidable, pattern only when verification ran to completion. Everything else is always present.
    """

    response_id: int
    dataset: str
    idx: int
    sample: int
    format_ok: bool
    boxed: str | None
    outcome_correct: bool
    givens_ok: bool = False
    steps_ok: bool = False
    n_steps: int = 0
    steps: list = field(default_factory=list)  # StepVerificationResult items
    premise_consistency_inconsistent_at: int | None = None
    premise_consistency_unknown_at: int | None = None
    corrupt_info: dict | None = None
    prof: dict = field(default_factory=dict)
    translation_record_from_problem: list | None = None
    translation_record_from_steps: list | None = None
    pattern: str | None = None

    def to_dict(self):
        out = {"rid": self.response_id, "dataset": self.dataset, "idx": self.idx,
               "sample": self.sample, "format_ok": self.format_ok,
               "boxed": self.boxed, "outcome_correct": self.outcome_correct,
               "givens_ok": self.givens_ok, "steps_ok": self.steps_ok,
               "n_steps": self.n_steps,
               "steps": [s.to_dict() for s in self.steps],
               "premise_consistency_inconsistent_at":
                   self.premise_consistency_inconsistent_at,
               "corrupt_info": self.corrupt_info,
               "prof": self.prof}
        if self.translation_record_from_problem is not None:
            out["translation_record_from_problem"] = self.translation_record_from_problem
        if self.translation_record_from_steps is not None:
            out["translation_record_from_steps"] = self.translation_record_from_steps
        if self.premise_consistency_unknown_at is not None:
            out["premise_consistency_unknown_at"] = (
                self.premise_consistency_unknown_at)
        if self.pattern is not None:
            out["pattern"] = self.pattern
        return out


class ProofOutcome(Enum):
    """Classified enum outcome of one theorem verification. 
    Exactly one outcome per result. """

    PROVED = "proved"                # theorem verified
    UNPROVED = "unproved"            # genuine refusal on a fully consolidated node
    INCOMPLETE = "incomplete"        # node found but never consolidated (headless watchdog abort)
    TIMEOUT = "timeout"              # our wall-clock caps fired (hard cap or cooperative PIDE deadline)
    WORKER_ERROR = "worker_error"    # any other infrastructure failure (PIDE FAILED, node missing, canceled, crash)


@dataclass
class VerificationOutcome:
    """One theorem check, classified. All consumers read attributes; raw dicts (disk cache entries, test mocks) enter only through from_raw at the pool boundaries."""

    outcome: ProofOutcome
    elapsed: float = 0.0
    errors: list = field(default_factory=list)
    premise_consistency_unknown: bool = False
    cache_hit: bool = False
    queue_wait: float = 0.0
    check_time: float = 0.0
    worker: int | None = None       # diagnostic only, never persisted
    attempts: int | None = None     # diagnostic only, never persisted

    @classmethod
    def from_raw(cls, raw):
        """Classify a raw result dict (idempotent on an already-typed result). Raw dicts come from the disk cache (3-field entries) and from test mocks at the pool seams.

        Precedence success > worker_error > incomplete > else, so the handful of illegal combinations that can arrive from mocks or foreign disk payloads resolve deterministically. A timeout carries no flag of its own in raw dicts; it is recognized by the two load-bearing error-text prefixes (do not reword them at the producers).
        """
        if isinstance(raw, cls):
            return raw
        errors = list(raw.get("errors") or [])
        head = str(errors[0]) if errors else ""
        if raw.get("success"):
            outcome = ProofOutcome.PROVED
        elif raw.get("worker_error"):
            if (raw.get("premise_consistency_unknown")
                    or head.startswith("hard timeout:")
                    or head.startswith("worker error: TimeoutError")):
                outcome = ProofOutcome.TIMEOUT
            else:
                outcome = ProofOutcome.WORKER_ERROR
        elif raw.get("incomplete"):
            outcome = ProofOutcome.INCOMPLETE
        else:
            outcome = ProofOutcome.UNPROVED
        return cls(
            outcome=outcome,
            elapsed=float(raw.get("elapsed") or 0.0),
            errors=errors,
            premise_consistency_unknown=bool(raw.get("premise_consistency_unknown")),
            cache_hit=bool(raw.get("cache_hit")),
            queue_wait=float(raw.get("queue_wait") or 0.0),
            check_time=float(raw.get("check_time") or 0.0),
            worker=raw.get("worker"),
            attempts=raw.get("attempts"),
        )

    def to_cache_entry(self):
        """The 3-field subset that is the ONLY shape ever stored in the memory memo or on disk. The cacheability check admits only PROVED and fast UNPROVED, so the outcome is fully reconstructible from a stored entry; no store-format change, no cache-version bump."""
        return {"success": self.outcome is ProofOutcome.PROVED,
                "elapsed": self.elapsed, "errors": list(self.errors)}

    def cacheable(self, fail_fast_s):
        """PROVED is always replayable. UNPROVED is cacheable only when it failed fast and is not a consolidation artifact; the substring guard is redundant with INCOMPLETE for fresh results but protects legacy or foreign dicts whose incomplete flag was lost."""
        if self.outcome is ProofOutcome.PROVED:
            return True
        if self.outcome is not ProofOutcome.UNPROVED:
            return False
        return (self.elapsed < fail_fast_s
                and not any("not consolidated" in str(e) for e in self.errors))

    @property
    def proved(self):
        """The theorem was verified."""
        return self.outcome is ProofOutcome.PROVED

    @property
    def infrastructure_failure(self):
        """The prover infrastructure failed (our timeout or a worker error), so the theorem itself was never decided. These are the results the retry loop re-runs on a fresh worker, and the ones the repeated-failure short circuit counts."""
        return self.outcome in (ProofOutcome.TIMEOUT, ProofOutcome.WORKER_ERROR)

    @property
    def counts_toward_repeated_timeout(self):
        """Faithful to today's behavior: the repeated-failure table strikes on EVERY infrastructure failure, not just timeouts. Narrowing this to TIMEOUT only would be a deliberate one-line behavior change here, not part of the representation migration."""
        return self.infrastructure_failure

    @property
    def fail_closed(self):
        """The check could not decide the theorem: reward logic must treat it as unverified without recording it as a genuine refusal."""
        return self.outcome in (ProofOutcome.INCOMPLETE, ProofOutcome.TIMEOUT,
                                ProofOutcome.WORKER_ERROR)

    @property
    def premise_consistency(self):
        """Ternary classification when this result is a consistency probe (goal False): a PROVED probe means the premises derive False, i.e. they are contradictory."""
        if self.outcome is ProofOutcome.PROVED:
            return PremiseConsistency.INCONSISTENT
        if self.fail_closed or self.premise_consistency_unknown:
            return PremiseConsistency.UNKNOWN
        return PremiseConsistency.CONSISTENT
