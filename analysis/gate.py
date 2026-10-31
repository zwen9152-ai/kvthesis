"""Gate G2': is degeneration a large share of the accuracy lost to eviction, and does
rollback recover it?

For every base run (e.g. ``snapkv|256``) and its rollback variant
(``snapkv@rollback=1|256``) this prints:
  lost       problems full-KV solves but the base run misses
  degen      share of ``lost`` whose trace contains a repetition loop
  recovered  (acc_rollback - acc_base) / (acc_full - acc_base)
  overhead   computed tokens of the rollback run / computed tokens of the base run
  loops      problems whose trace contains a loop: base run -> rollback run

    python -m analysis.gate --runs results/math500_n40.jsonl
"""

import argparse
import json
from collections import defaultdict


def load(path):
    runs = defaultdict(dict)
    with open(path) as f:
        for r in map(json.loads, f):
            runs[r["run"]][r["id"]] = r
    return runs


def acc(rs, ids):
    return sum(rs[i]["correct"] for i in ids) / len(ids)


def n_correct(rs, ids):
    return sum(bool(rs[i]["correct"]) for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--min_degen", type=float, default=0.30)
    ap.add_argument("--min_recovered", type=float, default=0.50)
    args = ap.parse_args()

    runs = load(args.runs)
    full = runs.get("full")
    if not full:
        raise SystemExit("no full-KV run in file")

    print(f"{'base run':24s} {'acc':>6s} {'lost':>5s} {'degen':>6s} | {'+rollback acc':>13s} "
          f"{'recovered':>9s} {'overhead':>8s} {'rb/prob':>7s} loops    extra")
    verdicts = []
    for run, rs in sorted(runs.items()):
        if run == "full" or "rollback=1" in run:
            continue
        ids = [i for i in rs if i in full]
        if not ids:
            continue
        lost = [i for i in ids if full[i]["correct"] and not rs[i]["correct"]]
        degen = sum(rs[i]["degenerate"] for i in lost) / len(lost) if lost else 0.0
        a_base = acc(rs, ids)
        line = f"{run:24s} {a_base:6.1%} {len(lost):5d} {degen:6.1%} | "
        name, _, budget = run.partition("|")
        variants = sorted(r for r in runs if r.startswith(f"{name}@rollback=1") and r.endswith(f"|{budget}"))
        if not variants:
            line += f"{'-':>13s}"
        for k, variant in enumerate(variants):
            rb = runs[variant]
            common = [i for i in ids if i in rb]
            a_rb = acc(rb, common)
            # integer counts: a float ratio misses the 50% threshold by rounding (3/6 -> 0.4999)
            c_rb, c_b, c_f = n_correct(rb, common), n_correct(rs, common), n_correct(full, common)
            rec = (c_rb - c_b) / (c_f - c_b) if c_f > c_b else float("nan")
            ovh = sum(rb[i]["computed_tokens"] for i in common) / max(1, sum(rs[i]["computed_tokens"] for i in common))
            n_rb = sum(len(rb[i]["rollbacks"]) for i in common) / len(common)
            loops = (sum(rs[i]["degenerate"] for i in common), sum(rb[i]["degenerate"] for i in common))
            if k:
                print(line)
                line = f"{'':24s} {'':>6s} {'':>5s} {'':>6s} | "
            tag = variant.partition("@rollback=1")[2].partition("|")[0].lstrip(",") or "-"
            line += (f"{a_rb:13.1%} {rec:9.1%} {ovh:8.2f} {n_rb:7.2f} {loops[0]:>3d}->{loops[1]:<3d} {tag}")
            verdicts.append((variant, degen, rec))
        print(line)
    print(f"\nfull KV accuracy: {acc(full, list(full)):.1%} on {len(full)} problems")

    print("\n=== gate G2' ===")
    for run, degen, rec in verdicts:
        ok = degen >= args.min_degen and rec >= args.min_recovered
        print(f"{run:36s} degen share {degen:.0%} (>= {args.min_degen:.0%}), "
              f"recovered {rec:.0%} (>= {args.min_recovered:.0%}) -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
