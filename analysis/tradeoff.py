"""Cost table and cache-accuracy trade-off (chapters 4/5): what RewindKV-style methods buy and pay.

    python -m analysis.tradeoff results/main/1.5b_math500_s0.jsonl --dataset math500 \
        --out results/analysis/math500/tradeoff

Outputs (markdown + one figure):

* ``cost.md``     one row per run: accuracy, time-averaged / peak KV, KV saved vs Full,
                  extra KV vs the static run at the same start budget, wasted computation,
                  per-sequence speed vs its base, trigger / truncation / miss rate.
* ``curve.md``    every adaptive run against the *static* curve of its base method
                  (budgets 128..1024 interpolated at the same time-averaged KV): the
                  accuracy a plain larger budget would reach with the same memory.
* ``levels.md``   by MATH difficulty level: accuracy and time-averaged KV of Full, the static
                  base and each adaptive run (does the extra memory go to the hard problems?).
* ``curve.png/pdf`` accuracy vs time-averaged KV, one panel per base method.

Time-averaged KV = resident KV entries averaged over every decoded step, including steps a
rollback later discards. Taken from the logged ``kv_area`` when present, otherwise replayed
with ``analysis.kv_time.replay`` (older result files).
"""

import argparse
import json
import os
from collections import defaultdict

from analysis.degeneration_stats import mcnemar_p, mean, paired_ci
from analysis.kv_time import replay

BASES = ["snapkv", "rkv", "lazy", "vase", "h2o", "streaming"]


def load(paths):
    runs = defaultdict(dict)
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            runs[r["run"]][r["id"]] = r
    return runs


def kv_stats(r, buffer):
    """(decoded steps, KV area) of one problem."""
    if r.get("kv_area") is not None and r.get("computed_tokens"):
        return r["computed_tokens"], r["kv_area"]
    s, a, _ = replay(r, buffer)
    return s, a


def parse(run):
    spec, _, budget = run.partition("|")
    name, _, opts = spec.partition("@")
    return name, opts, (int(budget) if budget.isdigit() else None)


def is_static(run):
    name, opts, budget = parse(run)
    return name in BASES and not opts and budget is not None


def is_full(run):
    return run == "full"  # instrumented Full runs (full@probe=...) are not reference points


def label(run):
    """Short human name of a run's variant."""
    name, opts, budget = parse(run)
    o = dict(kv.split("=") for kv in opts.split(",")) if opts else {}
    parts = []
    if o.get("merge") == "1":
        parts.append("prevent")
    if o.get("rollback") == "1" and o.get("act", "1") != "0":
        if any(k in o for k in ("fuzzy", "hesitate", "decay")) or o.get("recall") == "aligned":
            parts.append("RewindKV-v2")
        elif o.get("recall", "1") == "0":
            parts.append("RewindKV-light")
        else:
            parts.append("RewindKV")
    if o.get("rescue") == "1":
        parts.append("KV-Rescue-lite")
    std = {"rollback": None, "escalate": "2", "max_budget": "1024", "recall": None, "offload": None, "merge": None,
           "rescue": None, "act": None, "probe": None, "fuzzy": None, "hesitate": None, "decay": None}
    extra = [f"{k}={v}" for k, v in o.items() if k not in std or (std[k] is not None and v != std[k])]
    if o.get("rollback") == "1" and "escalate" not in o:
        extra.append("no escalation")
    return "+".join(parts) + (f"({','.join(extra)})" if extra else "") if parts else (opts or "static")


def summarize(rs, buffer):
    steps = area = 0
    for r in rs.values():
        s, a = kv_stats(r, buffer)
        steps += s
        area += a
    num = sum(r["num_tokens"] for r in rs.values())
    comp = sum(r.get("computed_tokens") or r["num_tokens"] for r in rs.values())
    return {"n": len(rs), "acc": mean(r["correct"] for r in rs.values()),
            "tavg": area / max(steps, 1), "peak": mean(r["max_cache_len"] for r in rs.values()),
            "waste": comp / max(num, 1) - 1, "speed": mean(r["tok_per_s"] for r in rs.values()),
            "trigger": mean(bool(r.get("rollbacks")) for r in rs.values()),
            "trunc": mean(not r["finished"] for r in rs.values()),
            "len": mean(r["num_tokens"] for r in rs.values())}


def miss_rate(rs, full):
    """Of the problems Full solves and this run fails: share where the detector never fired."""
    lost = [i for i in rs if i in full and full[i]["correct"] and not rs[i]["correct"]]
    if not lost:
        return None, 0
    miss = sum(not rs[i].get("rollbacks") and not rs[i].get("detections") for i in lost)
    return miss / len(lost), len(lost)


def static_curve(runs, base, stats):
    pts = sorted((stats[r]["tavg"], stats[r]["acc"], parse(r)[2]) for r in runs
                 if is_static(r) and parse(r)[0] == base)
    return pts


def interp(pts, x):
    """Accuracy of the static curve at time-averaged KV ``x`` (None outside its range)."""
    if len(pts) < 2 or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1e-9)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dataset", default="math500", help="for difficulty levels (MATH only)")
    ap.add_argument("--buffer", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_fig", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.files)
    full = runs.get("full", {})
    stats = {run: summarize(rs, args.buffer) for run, rs in runs.items()}
    F = stats.get("full")
    order = sorted(runs, key=lambda r: (parse(r)[0] != "full", parse(r)[0], parse(r)[2] or 0, label(r)))

    # ---- cost table ----
    md = ["| method | start budget | variant | n | acc % | Δacc vs Full | time-avg KV | KV saved vs Full | "
          "extra KV vs static | peak KV | wasted compute | speed vs base | trigger | trunc | miss |",
          "|" + "---|" * 15]
    for run in order:
        if run.startswith("full@"):
            continue
        name, opts, budget = parse(run)
        s = stats[run]
        static = f"{name}|{budget}"
        st = stats.get(static) if budget else None
        extra = f"{s['tavg'] - st['tavg']:+.0f} ({s['tavg'] / st['tavg']:.2f}×)" if st and run != static else "-"
        speed = f"{s['speed'] / st['speed']:.0%}" if st and run != static else "-"
        miss, n_lost = miss_rate(runs[run], full) if full and "rollback=1" in run else (None, 0)
        md.append(
            f"| {name} | {budget or '-'} | {label(run) if run != 'full' else 'Full KV'} | {s['n']} | {100 * s['acc']:.1f} | "
            f"{100 * (s['acc'] - F['acc']):+.1f} | {s['tavg']:.0f} | "
            f"{1 - s['tavg'] / F['tavg']:.0%} | {extra} | {s['peak']:.0f} | {s['waste']:.1%} | {speed} | "
            f"{s['trigger']:.0%} | {s['trunc']:.0%} | "
            f"{f'{miss:.0%} ({n_lost})' if miss is not None else '-'} |" if F else "")
    open(os.path.join(args.out, "cost.md"), "w").write("\n".join(md) + "\n")

    # ---- equal-memory comparison with the static curve ----
    md = ["| run | time-avg KV | acc % | static curve at same KV % | Δ vs curve | Δ vs same-budget static [95% CI] | McNemar p |",
          "|---|---|---|---|---|---|---|"]
    curve_rows = []
    for run in order:
        name, opts, budget = parse(run)
        if not opts or name not in BASES or budget is None:
            continue
        s = stats[run]
        pts = static_curve(runs, name, stats)
        y = interp(pts, s["tavg"])
        static = f"{name}|{budget}"
        ci_txt, p_txt = "-", "-"
        if static in runs:
            ids = [i for i in runs[run] if i in runs[static]]
            a = [float(runs[static][i]["correct"]) for i in ids]
            b = [float(runs[run][i]["correct"]) for i in ids]
            lo, hi = paired_ci(a, b)
            w, ls = sum(y_ > x_ for x_, y_ in zip(a, b)), sum(y_ < x_ for x_, y_ in zip(a, b))
            ci_txt = f"{100 * (mean(b) - mean(a)):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            p_txt = f"{mcnemar_p(w, ls):.3g}"
        curve_rows.append((run, s, y))
        at_curve = f"{100 * y:.1f}" if y is not None else "out of range"
        vs_curve = f"{100 * (s['acc'] - y):+.1f}" if y is not None else "-"
        md.append(f"| `{run}` | {s['tavg']:.0f} | {100 * s['acc']:.1f} | {at_curve} | {vs_curve} | {ci_txt} | {p_txt} |")
    md.append("")
    md.append("Static curve points (time-avg KV → acc): " + "; ".join(
        f"{b}: " + ", ".join(f"{bud}→{x:.0f}/{100 * acc:.1f}%" for x, acc, bud in static_curve(runs, b, stats))
        for b in BASES if static_curve(runs, b, stats)))
    open(os.path.join(args.out, "curve.md"), "w").write("\n".join(md) + "\n")

    # ---- by difficulty ----
    level = {}
    if args.dataset.startswith("math500"):
        try:
            from eval import data
            level = {r["id"]: r.get("level") for r in data.LOADERS["math500"]()}
        except Exception as e:  # dataset not available offline
            print(f"(levels skipped: {e})")
    if level:
        lv = sorted({level[i] for rs in runs.values() for i in rs if i in level})
        md = ["| run | " + " | ".join(f"L{x} acc / KV / trig" for x in lv) + " |", "|" + "---|" * (len(lv) + 1)]
        for run in order:
            if run.startswith("full@"):
                continue
            cells = []
            for x in lv:
                sub = {i: r for i, r in runs[run].items() if level.get(i) == x}
                if not sub:
                    cells.append("-")
                    continue
                s = summarize(sub, args.buffer)
                cells.append(f"{100 * s['acc']:.0f}% / {s['tavg']:.0f} / {s['trigger']:.0%} (n={s['n']})")
            md.append(f"| `{run}` | " + " | ".join(cells) + " |")
        open(os.path.join(args.out, "levels.md"), "w").write("\n".join(md) + "\n")

    if not args.no_fig:
        figure(args.out, runs, stats, F)
    for f in ("cost.md", "curve.md"):
        print(open(os.path.join(args.out, f)).read())


# categorical slots in fixed order (dataviz reference palette); identity follows the variant
VARIANT_COLORS = {"RewindKV": "#2a78d6", "RewindKV-light": "#eb6834", "prevent": "#1baf7a",
                  "RewindKV-v2": "#eda100", "prevent+RewindKV-light": "#e87ba4", "prevent+RewindKV-v2": "#6250d6",
                  "KV-Rescue-lite": "#008300"}
MARKERS = {"RewindKV": "o", "RewindKV-light": "s", "prevent": "^", "RewindKV-v2": "D",
           "prevent+RewindKV-light": "v", "prevent+RewindKV-v2": "P", "KV-Rescue-lite": "X"}


def figure(out, runs, stats, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    bases = [b for b in BASES[:4] if any(parse(r)[0] == b for r in runs)]
    if not bases:
        return
    fig, axes = plt.subplots(1, len(bases), figsize=(3.6 * len(bases), 3.4), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], bases):
        pts = static_curve(runs, b, stats)
        if pts:
            ax.plot([x for x, _, _ in pts], [100 * y for _, y, _ in pts], color="#8a8a85", lw=2, marker="o", ms=5,
                    label="static budget", zorder=2)
            for x, y, bud in pts:
                ax.annotate(str(bud), (x, 100 * y), textcoords="offset points", xytext=(4, -10), fontsize=7,
                            color="#5f5e5a")
        seen = set()
        for run in runs:
            name, opts, budget = parse(run)
            if name != b or not opts:
                continue
            v = label(run)
            if v not in VARIANT_COLORS:  # ablations (labels with options) stay out of the figure
                continue
            s = stats[run]
            ax.scatter(s["tavg"], 100 * s["acc"], color=VARIANT_COLORS[v], marker=MARKERS[v], s=46,
                       edgecolor="white", linewidth=1.5, zorder=3, label=v if v not in seen else None)
            seen.add(v)
        if F:
            ax.scatter(F["tavg"], 100 * F["acc"], color="#1f1f1d", marker="*", s=110, zorder=3, label="Full KV")
        ax.set_xscale("log")
        ax.set_xticks([300, 500, 1000, 2000, 3000])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("time-averaged KV entries")
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("accuracy (%)")
    handles, labels = {}, []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(5, len(handles)), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"curve.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
