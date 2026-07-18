"""Task #37 / #34-(6) analysis: per-dataset funnel + step-pattern aggregates from the
verification records, and the rewarded-step audit file for the zero-false-positive review.

Definitions used here:
  reached  = steps that got a pattern symbol (o/x/c/u/g/m), i.e. survived translation and
             reached the prover; this is the measured "checkable fraction" numerator.
  t        = steps that never reached the prover: in-response translation drops plus every
             step of a givens/steps-translation-failed response (pattern is empty there).
  format-failed responses have no parseable steps at all (n_steps=0) and are reported as a
  response-level rate, not in the step table.

  python -u e2e_analyze.py --records records.jsonl --debug-dir debug --audit-out audit.jsonl
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
    header = ("%-10s %5s %6s %6s %6s %6s | %5s %5s %5s %5s %5s %5s %6s | %6s %7s %6s %6s"
              % ("dataset", "resp", "fmt%", "giv%", "stp%", "outc%",
                 "o", "x", "c", "u", "g", "m", "t",
                 "steps", "reach%", "o/all%", "o/rch%"))
    print(header)
    print("-" * len(header))

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
        print("%-10s %5d %6.1f %6.1f %6.1f %6.1f | %5d %5d %5d %5d %5d %5d %6d | %6d %7.1f %6.1f %6.1f"
              % (ds, n, 100.0 * len(fmt) / max(1, n), 100.0 * len(giv) / max(1, len(fmt)),
                 100.0 * len(stp) / max(1, len(giv)), 100.0 * outc / max(1, n),
                 sym["o"], sym["x"], sym["c"], sym["u"], sym["g"], sym["m"], sym["t"],
                 total_steps, 100.0 * reached / max(1, total_steps),
                 100.0 * sym["o"] / max(1, total_steps),
                 100.0 * sym["o"] / max(1, reached)))
        if errors:
            print("           !! %d responses errored: %s"
                  % (len(errors), [e.get("error") for e in errors[:3]]))
        grand.update(sym)
        grand_steps += total_steps
        grand_reached += reached
        grand_resp += n

    print("-" * len(header))
    print("%-10s %5d %s | %5d %5d %5d %5d %5d %5d %6d | %6d %7.1f %6.1f %6.1f"
          % ("TOTAL", grand_resp, " " * 27,
             grand["o"], grand["x"], grand["c"], grand["u"], grand["g"], grand["m"], grand["t"],
             grand_steps, 100.0 * grand_reached / max(1, grand_steps),
             100.0 * grand["o"] / max(1, grand_steps),
             100.0 * grand["o"] / max(1, grand_reached)))

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
