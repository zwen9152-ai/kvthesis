"""Gate G2': is degeneration a large share of the accuracy lost to eviction, and does
rollback recover it?

For every base run (e.g. ``snapkv|256``) and its rollback variant
(``snapkv@rollback=1|256``) this prints:
  lost       problems full-KV solves but the base run misses
  degen      share of ``lost`` whose trace contains a repetition loop
  recovered  (acc_rollback - acc_base) / (acc_full - acc_base)
  overhead   computed tokens of the rollback run / computed tokens of the base run

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
          f"{'recovered':>9s} {'overhead':>8s} {'rb/prob':>7s}")
    verdicts = []
    for run, rs in sorted(runs.items()):
        if run == "full" or "rollback=1" in run:
            continue
        ids = [i for i in rs if i in full]
        if not ids:
            continue
        lost = [i for i in ids if full[i]["correct"] and not rs[i]["correct"]]
        degen = sum(rs[i]["degenerate"] for i in lost) / len(lost) if lost else 0.0
        a_full, a_base = acc(full, ids), acc(rs, ids)
        line = f"{run:24s} {a_base:6.1%} {len(lost):5d} {degen:6.1%} | "
        name, _, budget = run.partition("|")
        rb = runs.get(f"{name}@rollback=1|{budget}")
        if rb:
            common = [i for i in ids if i in rb]
            a_rb, a_b, a_f = acc(rb, common), acc(rs, common), acc(full, common)
            rec = (a_rb - a_b) / (a_f - a_b) if a_f > a_b else float("nan")
            ovh = sum(rb[i]["computed_tokens"] for i in common) / max(1, sum(rs[i]["computed_tokens"] for i in common))
            n_rb = sum(len(rb[i]["rollbacks"]) for i in common) / len(common)
            line += f"{a_rb:13.1%} {rec:9.1%} {ovh:8.2f} {n_rb:7.2f}"
            verdicts.append((run, degen, rec))
        else:
            line += f"{'-':>13s}"
        print(line)
    print(f"\nfull KV accuracy: {acc(full, list(full)):.1%} on {len(full)} problems")

    print("\n=== gate G2' ===")
    for run, degen, rec in verdicts:
        ok = degen >= args.min_degen and rec >= args.min_recovered
        print(f"{run:24s} degen share {degen:.0%} (>= {args.min_degen:.0%}), "
              f"recovered {rec:.0%} (>= {args.min_recovered:.0%}) -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
