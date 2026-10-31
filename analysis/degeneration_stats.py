"""Chapters 3-4: degeneration under eviction, and what RewindKV recovers.

Reads result files of ``eval.run_eval --save_ids`` (one or more; runs are matched by
problem id within a file) and writes to ``--out``:

  summary.md / summary.tex   per run: accuracy, mean output length, peak KV entries,
                             truncation and loop rates, output-length inflation vs full KV
  loss.md                    problems full KV solves but the run misses, split into
                             loop / truncated without loop / other wrong answer
  paired.md                  each rollback / rescue variant vs its base run on the same
                             problems: accuracy delta with a paired-bootstrap 95% CI and
                             an exact McNemar p-value (wins / losses)
  loops.md                   loop onset (tokens after the first eviction) and period
  fig_pareto.pdf             accuracy vs mean peak KV entries
  fig_loss.pdf               loss attribution per run
  fig_length.pdf             output-length distributions

    python -m analysis.degeneration_stats --runs results/phase1b_math500_n50_bfloat16.jsonl \
        --out results/analysis/phase1b
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

from kvcomp.degeneration import DegenerationDetector

FAMILY_ORDER = ["full", "streaming", "h2o", "snapkv", "rkv", "lazy", "vase", "stepkv"]


def load(paths):
    runs = defaultdict(dict)
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                runs[r["run"]][r["id"]] = r
    return runs


def variant(run: str) -> str:
    """'' for a plain run, 'rewind' / 'rescue' / '+<opts>' for degeneration-handling variants."""
    spec = run.split("|")[0]
    if "rescue=1" in spec:
        return "rescue"
    if "rollback=1" in spec:
        return "rewind" if spec.endswith("rollback=1,escalate=2,max_budget=1024") else "rollback:" + spec.split("@", 1)[1]
    return ""


def base_of(run: str) -> str:
    spec, _, budget = run.partition("|")
    return f"{spec.split('@')[0]}|{budget}"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def loop_onset(gen_ids, delim, budget, buffer=128):
    """(tokens from the first eviction to the first detected loop, loop period) or None."""
    det = DegenerationDetector()
    seen: list[int] = []
    for i, t in enumerate(gen_ids):
        seen.append(t)
        start = det.update(seen, t in delim)
        if start is not None:
            tail = gen_ids[start: i + 1]
            period = next((p for p in range(1, 65) if len(tail) >= 2 * p and tail[-p:] == tail[-2 * p: -p]), None)
            first_evict = 0 if budget is None else budget + buffer  # full KV: count from the start
            return start - first_evict, period
    return None


def mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_ci(a, b, iters=10000, seed=0):
    """95% bootstrap CI of mean(b - a) over paired problems."""
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return boots[int(0.025 * iters)], boots[int(0.975 * iters)]


def sort_key(run):
    fam = run.split("@")[0].split("|")[0]
    budget = run.split("|")[1] if "|" in run else "0"
    return (int(budget or 0), FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99, variant(run), run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="for the step delimiters used by the loop detector")
    ap.add_argument("--no_figs", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    from kvcomp.segment import delimiter_token_ids
    delim = set(delimiter_token_ids(AutoTokenizer.from_pretrained(args.tokenizer)))

    runs = load(args.runs)
    full = runs.get("full", {})
    order = sorted(runs, key=sort_key)

    # --- summary ---
    rows = []
    for run in order:
        rs = runs[run]
        ids = list(rs)
        common = [i for i in ids if i in full]
        infl = mean(rs[i]["num_tokens"] / max(1, full[i]["num_tokens"]) for i in common) if run != "full" else 1.0
        rows.append({
            "run": run, "n": len(ids), "acc": mean(r["correct"] for r in rs.values()),
            "len": mean(r["num_tokens"] for r in rs.values()),
            "computed": mean(r["computed_tokens"] for r in rs.values()),
            "peak_kv": mean(r["max_cache_len"] for r in rs.values()),
            "trunc": mean(not r["finished"] for r in rs.values()),
            "loop": mean(r["degenerate"] for r in rs.values()),
            "inflation": infl,
            "rb": mean(len(r.get("rollbacks", [])) for r in rs.values()),
        })
    hdr = "| run | n | acc | len | computed | peak KV | trunc | loop | len/full | rb/prob |\n|---|---|---|---|---|---|---|---|---|---|\n"
    md = hdr + "".join(
        f"| `{r['run']}` | {r['n']} | {r['acc']:.1%} | {r['len']:.0f} | {r['computed']:.0f} | {r['peak_kv']:.0f} | "
        f"{r['trunc']:.1%} | {r['loop']:.1%} | {r['inflation']:.2f} | {r['rb']:.2f} |\n" for r in rows)
    open(os.path.join(args.out, "summary.md"), "w").write(md)
    tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "Method & Budget & Acc. & Len. & Peak KV & Trunc. & Loop \\\\", "\\midrule"]
    for r in rows:
        spec, _, budget = r["run"].partition("|")
        name = spec.replace("_", "\\_").replace("@rollback=1,escalate=2,max_budget=1024", " + RewindKV")
        tex.append(f"{name} & {budget or '--'} & {100 * r['acc']:.1f} & {r['len']:.0f} & {r['peak_kv']:.0f} & "
                   f"{100 * r['trunc']:.1f} & {100 * r['loop']:.1f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(args.out, "summary.tex"), "w").write("\n".join(tex) + "\n")

    # --- loss attribution ---
    loss_rows = []
    for run in order:
        if run == "full" or not full:
            continue
        rs = runs[run]
        lost = [i for i in rs if i in full and full[i]["correct"] and not rs[i]["correct"]]
        gained = [i for i in rs if i in full and not full[i]["correct"] and rs[i]["correct"]]
        loop = sum(rs[i]["degenerate"] for i in lost)
        trunc = sum((not rs[i]["finished"]) and not rs[i]["degenerate"] for i in lost)
        loss_rows.append((run, len(lost), loop, trunc, len(lost) - loop - trunc, len(gained)))
    md = "| run | lost | loop | truncated (no loop) | other wrong | gained |\n|---|---|---|---|---|---|\n"
    md += "".join(f"| `{r}` | {n} | {lp} ({lp / n:.0%}) | {t} | {o} | {g} |\n" if n else f"| `{r}` | 0 | - | - | - | {g} |\n"
                  for r, n, lp, t, o, g in loss_rows)
    open(os.path.join(args.out, "loss.md"), "w").write(md)

    # --- paired comparisons ---
    md = "| variant | vs base | n | base acc | variant acc | delta [95% CI] | wins/losses | McNemar p |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    for run in order:
        if not variant(run) or base_of(run) not in runs:
            continue
        b, v = runs[base_of(run)], runs[run]
        ids = [i for i in v if i in b]
        if not ids:
            continue
        xa = [int(b[i]["correct"]) for i in ids]
        xb = [int(v[i]["correct"]) for i in ids]
        wins = sum(y > x for x, y in zip(xa, xb))
        losses = sum(y < x for x, y in zip(xa, xb))
        lo, hi = paired_ci(xa, xb)
        md += (f"| `{run}` | `{base_of(run)}` | {len(ids)} | {mean(xa):.1%} | {mean(xb):.1%} | "
               f"{mean(xb) - mean(xa):+.1%} [{lo:+.1%}, {hi:+.1%}] | {wins}/{losses} | {mcnemar_p(wins, losses):.3f} |\n")
    open(os.path.join(args.out, "paired.md"), "w").write(md)

    # --- loop onset / period ---
    md = "| run | loops | onset after 1st eviction (median; full KV: from start) | onset < 0 (before eviction) | period (median) |\n"
    md += "|---|---|---|---|---|\n"
    for run in order:
        if variant(run):
            continue
        rs = runs[run]
        budget = rs[next(iter(rs))].get("budget")
        info = [loop_onset(r["gen_ids"], delim, budget) for r in rs.values() if r.get("gen_ids") and r["degenerate"]]
        info = [x for x in info if x]
        if not info:
            continue
        onsets = sorted(o for o, _ in info)
        periods = sorted(p for _, p in info if p)
        md += (f"| `{run}` | {len(info)} | {onsets[len(onsets) // 2]} | {sum(o < 0 for o in onsets)} | "
               f"{periods[len(periods) // 2] if periods else '-'} |\n")
    open(os.path.join(args.out, "loops.md"), "w").write(md)

    if not args.no_figs:
        figures(args.out, rows, loss_rows, runs, full)
    print(open(os.path.join(args.out, "summary.md")).read())
    print(open(os.path.join(args.out, "paired.md")).read())


def figures(out, rows, loss_rows, runs, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {"": "o", "rewind": "*", "rescue": "s"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rows:
        fam = r["run"].split("@")[0].split("|")[0]
        v = variant(r["run"])
        if v.startswith("rollback:"):
            continue
        ax.scatter(r["peak_kv"], 100 * r["acc"], marker=markers.get(v, "o"), s=90 if v else 40,
                   label=f"{fam}{' + ' + v if v else ''}")
        ax.annotate(r["run"].split("|")[-1] if "|" in r["run"] else "full", (r["peak_kv"], 100 * r["acc"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("mean peak KV entries (log)")
    ax.set_ylabel("accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_pareto.pdf"))

    if loss_rows:
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(loss_rows) + 1))
        names = [r[0] for r in loss_rows]
        loop = [r[2] for r in loss_rows]
        trunc = [r[3] for r in loss_rows]
        other = [r[4] for r in loss_rows]
        ax.barh(names, loop, label="loop")
        ax.barh(names, trunc, left=loop, label="truncated, no loop")
        ax.barh(names, other, left=[a + b for a, b in zip(loop, trunc)], label="other wrong")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("problems solved by full KV but missed")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_loss.pdf"))

    fig, ax = plt.subplots(figsize=(6, 4))
    for run in ["full"] + [r["run"] for r in rows if not variant(r["run"]) and r["run"] != "full"][:4]:
        if run in runs:
            ax.hist([r["num_tokens"] for r in runs[run].values()], bins=20, histtype="step", label=run)
    ax.set_xlabel("output tokens")
    ax.set_ylabel("problems")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_length.pdf"))


if __name__ == "__main__":
    main()
