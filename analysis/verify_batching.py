"""Compare batched decoding with batch-1 decoding on the real model (greedy).

In bf16 a different batch shape changes reduction order, so long greedy traces may
diverge after a while; what must agree is the behaviour: accuracy, lengths and the
long common prefixes. Prints, per run: accuracy and mean length for both, the share of
problems with identical output and the mean common-prefix length.

    python -m analysis.verify_batching results/verify/math500_batching_bs1.jsonl \
        results/verify/math500_batching_bs16.jsonl
"""

import json
import sys
from collections import defaultdict


def load(path):
    runs = defaultdict(dict)
    for r in map(json.loads, open(path)):
        runs[r["run"]][r["id"]] = r
    return runs


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print(f"{'run':48s} {'acc bs1':>8s} {'acc bsN':>8s} {'len bs1':>8s} {'len bsN':>8s} {'same':>6s} {'prefix':>7s}")
    for run in sorted(set(a) & set(b)):
        ids = sorted(set(a[run]) & set(b[run]))
        if not ids:
            continue
        acc = [sum(x[i]["correct"] for i in ids) / len(ids) for x in (a[run], b[run])]
        ln = [sum(x[i]["num_tokens"] for i in ids) / len(ids) for x in (a[run], b[run])]
        same = sum(a[run][i]["gen_ids"] == b[run][i]["gen_ids"] for i in ids) / len(ids)
        pre = sum(prefix(a[run][i]["gen_ids"], b[run][i]["gen_ids"]) for i in ids) / len(ids)
        print(f"{run:48s} {acc[0]:8.1%} {acc[1]:8.1%} {ln[0]:8.0f} {ln[1]:8.0f} {same:6.0%} {pre:7.0f}")


if __name__ == "__main__":
    main()
