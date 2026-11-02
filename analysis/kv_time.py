"""Time-averaged KV occupancy, replayed from saved result rows.

Peak KV (``max_cache_len``) penalises RewindKV for a short spike: it decodes most of the
time at the small base budget and only escalates after a rollback. What a batched server
pays for is the KV that is resident at each decoding step. This script replays, for every
problem, the cache length after every decoded token (including tokens later discarded by
a rollback) using the compressor's schedule (``Compressor.due``):

* the generated part grows by one entry per step; once it reaches ``budget + buffer`` it
  is compressed back to ``budget`` (the prompt is never evicted);
* a rollback at position ``at`` truncates generation to ``cut``; with recall the whole
  history before ``cut`` is resident for one step (``transient_cache_len``), then it is
  re-selected down to the escalated budget;
* full KV never compresses.

Reported per run: accuracy, mean peak KV (as in ``degeneration_stats``), time-averaged KV
(token-weighted over all decoded steps of all problems), and the mean KV area per problem
(sum over steps of resident entries, in millions of entry-steps), which also charges for
longer outputs. Replay totals are checked against ``computed_tokens``.

    python -m analysis.kv_time --runs results/main/1.5b_math500_s0.jsonl --out results/analysis/math500/kv_time.md
"""
import argparse
import json
from collections import defaultdict


def replay(r: dict, buffer: int = 128) -> tuple[int, int, int]:
    """Return (decoded steps, sum of resident KV entries over steps, max resident)."""
    P, budget = r["prompt_len"], r.get("budget")
    end = P + r["num_tokens"]
    events = sorted(r.get("rollbacks") or [], key=lambda e: e["at"]) if budget else []
    steps = area = peak = 0
    pos, gen = P, 0  # next absolute position; generated entries resident in the cache

    def run_to(stop):
        nonlocal pos, gen, steps, area, peak
        while pos < stop:
            pos += 1
            gen += 1
            steps += 1
            area += P + gen  # resident during this step's attention
            peak = max(peak, P + gen)
            if budget and gen >= budget + buffer:
                gen = budget

    for e in events:
        run_to(e["at"])
        pos = e["cut"]
        gen = min(gen, pos - P)
        t = e.get("transient_cache_len") or 0
        peak = max(peak, t)
        budget = e.get("budget", budget)  # KV-Rescue takeovers keep the budget
        if t:  # recalled history is resident for the re-encode step
            gen = t - P
        if gen >= budget + buffer:
            gen = budget
    run_to(end)
    return steps, area, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--buffer", type=int, default=128)
    args = ap.parse_args()
    agg = defaultdict(lambda: dict(n=0, ok=0, steps=0, area=0, peak=0, maxlen=0, comp=0, mismatch=0))
    for path in args.runs:
        for line in open(path):
            r = json.loads(line)
            s, a, p = replay(r, args.buffer)
            g = agg[r["run"]]
            g["n"] += 1
            g["ok"] += bool(r["correct"])
            g["steps"] += s
            g["area"] += a
            g["peak"] += p
            g["maxlen"] += r.get("max_cache_len") or 0
            comp = r.get("computed_tokens") or r["num_tokens"]
            g["comp"] += comp
            g["mismatch"] += abs(s - comp) > 2 * (1 + len(r.get("rollbacks") or []))
    rows = ["| run | n | acc | peak KV (logged) | peak KV (replay) | time-avg KV | KV area (M) | replay≠computed |",
            "|---|---|---|---|---|---|---|---|"]
    for run, g in sorted(agg.items(), key=lambda kv: kv[0]):
        n = g["n"]
        rows.append(f"| `{run}` | {n} | {100 * g['ok'] / n:.1f}% | {g['maxlen'] / n:.0f} | {g['peak'] / n:.0f} | "
                    f"{g['area'] / g['steps']:.0f} | {g['area'] / n / 1e6:.2f} | {g['mismatch']} |")
    text = "\n".join(rows)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
