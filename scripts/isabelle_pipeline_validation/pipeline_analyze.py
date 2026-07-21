"""Task #37 / #34-(6) analysis: per-dataset funnel + step-pattern aggregates from the
verification records, and the rewarded-step audit file for the zero-false-positive review.

Definitions used here:
  reached  = steps that got a pattern symbol (o/x/c/u/g/m), i.e. survived translation and
             reached the prover; this is the measured "checkable fraction" numerator.
  t        = steps that never reached the prover: in-response translation drops plus every
             step of a givens/steps-translation-failed response (pattern is empty there).
  format-failed responses have no parseable steps at all (n_steps=0) and are reported as a
  response-level rate, not in the step table.

  python -u pipeline_analyze.py --records records.jsonl --debug-dir debug --audit-out audit.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--debug-dir", required=True)
    ap.add_argument("--audit-out", required=True)
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.records, encoding="utf-8")]
    debug_dir = Path(args.debug_dir)
    by_ds = defaultdict(list)
    for r in recs:
        by_ds[r.get("dataset", "?")].append(r)

    audit = open(args.audit_out, "w", encoding="utf-8")
    n_audit = 0
    measured = {}   # per-dataset measured values, printed as labeled tables after the loop

    grand = Counter()
    grand_steps = grand_reached = grand_resp = 0
    direct_counts = Counter()
    echo_counts = Counter()

    for ds in sorted(by_ds):
        rows = by_ds[ds]
        n = len(rows)
        errors = [r for r in rows if r.get("error")]
        fmt = [r for r in rows if r.get("format_ok")]
        giv = [r for r in fmt if r.get("givens_ok")]
        stp = [r for r in giv if r.get("steps_ok")]
        outc = sum(1 for r in rows if r.get("outcome_correct"))

        sym = Counter()
        total_steps = 0
        for r in fmt:
            ns = int(r.get("n_steps") or 0)
            pat = str(r.get("pattern") or "")
            total_steps += ns
            sym.update(pat)
            sym["t"] += max(0, ns - len(pat))
            rewarded_at = {int(s["step"]) for s in (r.get("steps") or []) if s.get("rewarded")}
            for s in (r.get("steps") or []):
                if s.get("domain_reason") is not None:
                    direct_counts[(ds, s.get("domain_reason"))] += 1
                if s.get("rewarded"):
                    dbg = {}
                    dbg_path = debug_dir / ("%03d_debug.json" % int(r["response_id"]))
                    if dbg_path.exists():
                        dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
                    k = int(s["step"])
                    nl = (dbg.get("nl_steps") or [])
                    concl = (dbg.get("isabelle_step_conclusions") or [])
                    # Echo classification: a rewarded conclusion identical (modulo whitespace)
                    # to an EARLIER step's pyexpr conclusion proves trivially from the admitted
                    # s* chain. echo_of_unverified (origin step earned no reward) is the
                    # false-positive-suspect class measured on the algebra slice.
                    pyx = dbg.get("pyexpr_conclusions") or []
                    norm = lambda t: " ".join(str(t).split())
                    echo = None
                    if k < len(pyx) and pyx[k] is not None:
                        for j in range(k):
                            if j < len(pyx) and pyx[j] is not None and norm(pyx[j]) == norm(pyx[k]):
                                echo = "echo_of_%s" % ("verified" if j in rewarded_at else "unverified")
                                break
                    echo_counts[(ds, echo or "fresh")] += 1
                    audit.write(json.dumps({
                        "echo": echo,
                        "pyexpr_conclusion": pyx[k] if k < len(pyx) else None,
                        "dataset": ds, "response_id": r["response_id"],
                        "idx": r.get("idx"), "sample": r.get("sample"), "step": k,
                        "nl_conclusion": nl[k]["nl_conclusion"] if k < len(nl) else None,
                        "nl_premises": nl[k]["nl_premises"] if k < len(nl) else None,
                        "isabelle_term": concl[k] if k < len(concl) else None,
                        "domain_reason": s.get("domain_reason"),
                        "tolerance": bool(s.get("tolerance")),
                        "n_definitions": s.get("n_definitions"),
                    }, ensure_ascii=False) + "\n")
                    n_audit += 1

        reached = sum(sym[c] for c in "oxcugm")
        measured[ds] = {
            "n": n, "errors": errors, "parsed": total_steps, "reached": reached, "sym": sym,
            "format_ok": 100.0 * len(fmt) / max(1, n),
            "givens_ok": 100.0 * len(giv) / max(1, len(fmt)),
            "steps_ok": 100.0 * len(stp) / max(1, len(giv)),
            "checkable_steps": 100.0 * reached / max(1, total_steps),
            "reward_rate": 100.0 * sym["o"] / max(1, total_steps),
            "o_among_checkable": 100.0 * sym["o"] / max(1, reached),
            "outcome_acc": 100.0 * outc / max(1, n),
        }
        grand.update(sym)
        grand_steps += total_steps
        grand_reached += reached
        grand_resp += n

    # Every column below is computed above from this records file; none is hand-entered.
    # Names and definitions match the 260717 Notion "Overall (process reward engine)" table.
    print("Definitions (all values are percentages measured on %s):" % args.records)
    print("  format_ok = response parses to <step>/<premise>/<conclusion> + one \\boxed{}.")
    print("  givens_ok = among format_ok responses, the problem givens formalized.")
    print("  steps_ok  = among givens_ok responses, the reasoning steps formalized.")
    print("  checkable_steps   = steps that reached the prover / all parsed steps of format_ok responses.")
    print("  reward_rate       = o steps / all parsed steps of format_ok responses.")
    print("  o_among_checkable = o steps / checkable steps.")
    print("  outcome_acc       = responses whose \\boxed{} equals the reference answer.\n")

    fmt_a = "%-32s %9s %9s %8s %15s %11s %17s %11s"
    head_a = fmt_a % ("dataset", "format_ok", "givens_ok", "steps_ok", "checkable_steps",
                      "reward_rate", "o_among_checkable", "outcome_acc")
    print(head_a)
    print("-" * len(head_a))
    for ds in sorted(measured):
        d = measured[ds]
        print(fmt_a % (ds, "%.1f" % d["format_ok"], "%.1f" % d["givens_ok"],
                       "%.1f" % d["steps_ok"], "%.1f" % d["checkable_steps"],
                       "%.1f" % d["reward_rate"], "%.1f" % d["o_among_checkable"],
                       "%.1f" % d["outcome_acc"]))
        if d["errors"]:
            print("  !! %d responses errored: %s"
                  % (len(d["errors"]), [e.get("error") for e in d["errors"][:3]]))
    print("-" * len(head_a))
    print(fmt_a % ("TOTAL", "", "", "", "%.1f" % (100.0 * grand_reached / max(1, grand_steps)),
                   "%.1f" % (100.0 * grand["o"] / max(1, grand_steps)),
                   "%.1f" % (100.0 * grand["o"] / max(1, grand_reached)), ""))

    # The checkable-step result-symbol counts behind checkable_steps and reward_rate.
    print("\nResult-symbol detail (one symbol per checkable step; t = parsed but never reached):")
    print("  o proved+rewarded, x unproved, c premise-contradiction, u consistency-undecided,")
    print("  g guard-withheld, m number-omitted.")
    fmt_b = "%-32s %8s %6s %6s %6s %6s %5s %5s %8s"
    head_b = fmt_b % ("dataset", "parsed", "o", "x", "c", "u", "g", "m", "t")
    print(head_b)
    print("-" * len(head_b))
    for ds in sorted(measured):
        s = measured[ds]["sym"]
        print(fmt_b % (ds, measured[ds]["parsed"], s["o"], s["x"], s["c"],
                       s["u"], s["g"], s["m"], s["t"]))
    print("-" * len(head_b))
    print(fmt_b % ("TOTAL", grand_steps, grand["o"], grand["x"], grand["c"],
                   grand["u"], grand["g"], grand["m"], grand["t"]))

    if direct_counts:
        print("\ndirect-domain steps (dataset, reason -> count):")
        for (ds, reason), c in sorted(direct_counts.items()):
            print("  %-10s %-28s %d" % (ds, reason, c))
    if echo_counts:
        print("\nrewarded-step echo classes (dataset, class -> count; echo_of_unverified = FP suspect):")
        for (ds, cls), c in sorted(echo_counts.items()):
            print("  %-10s %-20s %d" % (ds, cls, c))
    audit.close()
    print("\naudit rows (all rewarded steps): %d -> %s" % (n_audit, args.audit_out), flush=True)


if __name__ == "__main__":
    main()
