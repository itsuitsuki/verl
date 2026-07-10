"""XML step parsing and answer extraction utilities.

Lifted from scripts/isabelle_poc_math500/pipeline_v4.py.
"""
import re


def parse_xml_steps(response: str):
    """Parse <step><premise>...</premise><conclusion>...</conclusion></step>."""
    blocks = re.findall(r"<step>(.*?)</step>", response, re.DOTALL)
    steps = []
    for b in blocks:
        prem = [p.strip() for p in
                re.findall(r"<premise>(.*?)</premise>", b, re.DOTALL)]
        con = re.search(r"<conclusion>(.*?)</conclusion>", b, re.DOTALL)
        if con is None:
            return None
        steps.append({"premises": prem, "conclusion": con.group(1).strip(),
                      "block_text": b})
    return steps or None


def boxed_answer(response: str):
    """Extract the last \\boxed{...} value from the response."""
    m = re.findall(r"\\boxed\{([^{}]+)\}", response)
    return m[-1].strip() if m else None


def corrupt_steps(steps_xml, ground_truth):
    """Step-level negative test: increment ONE number in ONE conclusion."""
    order = sorted(range(len(steps_xml)),
                   key=lambda i: abs(i - (len(steps_xml) - 1) / 2))
    for k in order:
        con = steps_xml[k]["conclusion"]
        cands = [m for m in re.finditer(r"(?<![\w.])(\d+)(?![\w.])", con)
                 if m.group(1) not in {"0", "1", "2"}
                 and m.group(1) != ground_truth.strip()]
        if not cands:
            continue
        m = cands[len(cands) // 2]
        old = m.group(1)
        new = str(int(old) + 1)
        new_con = con[: m.start()] + new + con[m.end():]
        steps_xml[k]["block_text"] = steps_xml[k]["block_text"].replace(
            con, new_con)
        steps_xml[k]["conclusion"] = new_con
        return {"step": k, "old": old, "new": new}
    return None


def block_stats(rs, label):
    """Print verification statistics for a batch of results."""
    n = len(rs)
    if n == 0:
        print(f"{label}: no data")
        return
    fmt = [r for r in rs if r["format_ok"]]
    tr = [r for r in fmt if r["steps_ok"]]
    steps = [e for r in tr for e in r["steps"]]
    nv = sum(1 for e in steps if e["verified"])
    nr = sum(1 for e in steps if e["rewarded"])
    nn = sum(1 for e in steps if e.get("neutral"))
    t_rate = len(tr) / len(fmt) if fmt else 0
    v_rate = nv / len(steps) if steps else 0
    r_rate = nr / len(steps) if steps else 0
    print(f"{label}: n={n} format={len(fmt)}/{n} "
          f"translation={len(tr)}/{len(fmt)} ({100 * t_rate:.1f}%) "
          f"verified={nv}/{len(steps)} ({100 * v_rate:.1f}%) "
          f"rewarded={nr}/{len(steps)} ({100 * r_rate:.1f}%) "
          f"neutral={nn}")
    print(f"  GOAL METRIC translation x verified = "
          f"{100 * t_rate * v_rate:.1f}%   (x rewarded = "
          f"{100 * t_rate * r_rate:.1f}%)")
