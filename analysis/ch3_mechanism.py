"""Chapter 3 mechanism analysis from the ``ch3`` queue phase (results/ch3/*.jsonl).

    python -m analysis.ch3_mechanism results/ch3/1.5b_math500_s0.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --out results/analysis/ch3

Three tables (markdown):

A. Forgetting, correlational: at every step-repeat detection of an observe-only run
   (``rollback=1,act=0,probe=1``) compare the KV retention of the step being repeated
   (its first copy) with control steps of similar age and length.
B. Forgetting, causal: intervention runs (``intervene=aligned|random``) against the
   observe run of the same base (= "none"; tokens are identical up to the first
   intervention). Relapse = another detection within ``--relapse`` tokens after it.
C. What gets evicted and what the model looks back at: token classes (numbers, operators,
   conclusion words, reflection cues, step delimiters, other); Full-KV lookback per class
   (``full@probe=1,lookback=1``) and the retention of each class under compression.
"""

import argparse
import json
import os
from collections import defaultdict

from analysis.degeneration_stats import load, mcnemar_p, mean, paired_ci

CUES = {"Wait", "Hmm", "Alternatively", "Hold", "But"}
CONCLUDE = {"so", "therefore", "thus", "hence", "answer", "boxed", "final", "result", "get", "gives"}
CLASSES = ["number", "operator", "conclusion", "cue", "delimiter", "other"]


def spec_of(run):
    return run.split("|")[0]


def opts(run):
    spec = spec_of(run)
    return dict(kv.split("=") for kv in spec.split("@", 1)[1].split(",")) if "@" in spec else {}


def base_key(run):
    """base method|budget of a ch3 run."""
    spec, _, budget = run.partition("|")
    return f"{spec.split('@')[0]}|{budget}"


# ---------------- A: retention of the repeated step ----------------

def retention_table(runs, age_win=512, len_ratio=2.0):
    rows = []
    for run, probs in sorted(runs.items()):
        o = opts(run)
        if o.get("act") != "0" or "intervene" in o:
            continue
        if spec_of(run).startswith("full"):
            continue
        orig, ctrl, wins, losses = [], [], 0, 0
        for r in probs.values():
            for ev in r.get("probe_events", []):
                if not ev.get("origin") or ev.get("kind") != "step":
                    continue
                o0, o1 = ev["origin"]
                steps = ev["steps"]
                me = [s for s in steps if s[0] <= o0 < s[1]]
                if not me:
                    continue
                L = o1 - o0
                cs = [s[2] for s in steps
                      if s[1] <= ev["loop_start"] and not (s[0] < o1 and o0 < s[1])
                      and abs(s[0] - o0) <= age_win and L / len_ratio <= s[1] - s[0] <= L * len_ratio]
                if not cs:
                    continue
                a, c = me[0][2], sum(cs) / len(cs)
                orig.append(a)
                ctrl.append(c)
                wins += a < c
                losses += a > c
        if orig:
            rows.append({"run": run, "events": len(orig), "orig": mean(orig) / 10, "ctrl": mean(ctrl) / 10,
                         "lower": wins / len(orig), "p": mcnemar_p(wins, losses)})
    lines = ["| run | events | retention of repeated step % | control steps % | share lower | sign p |",
             "|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['run']} | {x['events']} | {x['orig']:.1f} | {x['ctrl']:.1f} | {x['lower']:.0%} | {x['p']:.3g} |")
    return "\n".join(lines), rows


# ---------------- B: causal intervention ----------------

def first_point(r, key="intervene"):
    """Generated-token index of the first intervention (or, for the observe run, of the first
    step-repeat detection with an origin, where the intervention would have happened)."""
    for ev in r.get("detections", []):
        if key == "intervene" and "intervene" in ev:
            return ev["at"]
        if key == "observe" and ev.get("origin") and ev.get("kind") == "step":
            return ev["at"]
    return None


def relapsed(r, t0, horizon):
    return any(t0 < ev["at"] <= t0 + horizon and ev.get("kind") == "step" for ev in r.get("detections", []))


def intervention_table(runs, horizon):
    groups = defaultdict(dict)
    for run in runs:
        o = opts(run)
        if o.get("act") != "0" or spec_of(run).startswith("full"):
            continue
        groups[base_key(run)][o.get("intervene", "none")] = run
    lines = ["| base | arm | n (intervened) | relapse % | acc % | trunc % | mean len | restored/head | "
             "Δacc vs none [95% CI] | McNemar p |", "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for base, arms in sorted(groups.items()):
        if "none" not in arms:
            continue
        none = runs[arms["none"]]
        for arm in ("none", "aligned", "random"):
            if arm not in arms:
                continue
            rs = runs[arms[arm]]
            ids = [i for i in rs if i in none and first_point(none[i], "observe") is not None]
            if arm != "none":
                ids = [i for i in ids if first_point(rs[i]) is not None]
            if not ids:
                continue
            key = "observe" if arm == "none" else "intervene"
            rel = [relapsed(rs[i], first_point(rs[i], key), horizon) for i in ids]
            acc = [float(rs[i]["correct"]) for i in ids]
            acc0 = [float(none[i]["correct"]) for i in ids]
            restored = [ev.get("restored", 0) for i in ids for ev in rs[i].get("detections", []) if "intervene" in ev]
            w = sum(a > b for a, b in zip(acc, acc0))
            lo = sum(a < b for a, b in zip(acc, acc0))
            ci = paired_ci(acc0, acc) if arm != "none" else (0, 0)
            row = {"base": base, "arm": arm, "n": len(ids), "relapse": mean(rel), "acc": mean(acc),
                   "trunc": mean(not rs[i]["finished"] for i in ids), "len": mean(rs[i]["num_tokens"] for i in ids),
                   "restored": mean(restored) if restored else 0, "dacc": mean(acc) - mean(acc0), "ci": ci,
                   "p": mcnemar_p(w, lo) if arm != "none" else 1.0}
            out.append(row)
            lines.append(f"| {base} | {arm} | {row['n']} | {100 * row['relapse']:.1f} | {100 * row['acc']:.1f} | "
                         f"{100 * row['trunc']:.1f} | {row['len']:.0f} | {row['restored']:.1f} | "
                         f"{100 * row['dacc']:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] | {row['p']:.3g} |")
    return "\n".join(lines), out


# ---------------- C: token classes ----------------

def classifier(tok):
    cache = {}

    def cls(t):
        c = cache.get(t)
        if c is not None:
            return c
        s = tok.decode([t])
        w = s.strip()
        if "\n\n" in s:
            c = "delimiter"
        elif w in CUES:
            c = "cue"
        elif any(ch.isdigit() for ch in w):
            c = "number"
        elif w and (all(ch in "=+-*/^()[]{}<>\\|_.,:;!" for ch in w) or w.startswith("\\")):
            c = "operator"
        elif w.lower().strip(",.:") in CONCLUDE:
            c = "conclusion"
        else:
            c = "other"
        cache[t] = c
        return c
    return cls


def class_table(runs, cls, top=0.05):
    """Per class: share of tokens, Full-KV lookback (relative to the mean), share of the top-5%
    looked-back tokens, and retention at age ~1024 under every compressed observe run."""
    share, look, top_cnt, n_top, n_all = defaultdict(int), defaultdict(list), defaultdict(int), 0, 0
    full_runs = [r for r in runs if spec_of(r).startswith("full") and "lookback=1" in r]
    for run in full_runs:
        for r in runs[run].values():
            g, lk = r.get("gen_ids"), r.get("probe_look")
            if not g or not lk:
                continue
            vals = [(x, cls(t)) for t, x in zip(g, lk) if x >= 0]
            if not vals:
                continue
            thr = sorted(x for x, _ in vals)[int((1 - top) * len(vals))]
            for x, c in vals:
                share[c] += 1
                look[c].append(x)
                n_all += 1
                if x >= thr:
                    top_cnt[c] += 1
                    n_top += 1
    overall = mean(x for xs in look.values() for x in xs) if n_all else 0
    ret = defaultdict(lambda: defaultdict(list))
    comp_runs = [r for r in runs if not spec_of(r).startswith("full") and "probe=1" in r]
    for run in comp_runs:
        for r in runs[run].values():
            g, pr = r.get("gen_ids"), r.get("probe_ret")
            if not g or not pr:
                continue
            for t, x in zip(g, pr):
                if x >= 0:
                    ret[run][cls(t)].append(x / 10)
    head = "| class | share of tokens % | Full lookback (× mean) | share of top-5% looked-back % | " + \
        " | ".join(f"retained % {base_key(r)}" for r in comp_runs) + " |"
    lines = [head, "|" + "---|" * (4 + len(comp_runs))]
    for c in CLASSES:
        cells = [f"{100 * share[c] / max(n_all, 1):.1f}", f"{mean(look[c]) / overall:.2f}" if look[c] and overall else "-",
                 f"{100 * top_cnt[c] / max(n_top, 1):.1f}"]
        cells += [f"{mean(ret[r][c]):.1f}" if ret[r][c] else "-" for r in comp_runs]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default=None, help="tokenizer for the token-class table (C)")
    ap.add_argument("--relapse", type=int, default=1024, help="tokens after an intervention that count as relapse")
    ap.add_argument("--out", default=None, help="directory for ch3_mechanism.md / .json")
    args = ap.parse_args()
    runs = load(args.files)
    a, a_rows = retention_table(runs)
    b, b_rows = intervention_table(runs, args.relapse)
    parts = ["## A. Retention of the repeated step at detection (age-matched controls)", a,
             f"## B. Causal intervention (relapse within {args.relapse} tokens)", b]
    if args.model:
        from transformers import AutoTokenizer
        parts += ["## C. Token classes: Full-KV lookback and retention under compression",
                  class_table(runs, classifier(AutoTokenizer.from_pretrained(args.model)))]
    text = "\n\n".join(parts)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "ch3_mechanism.md"), "w") as f:
            f.write(text + "\n")
        with open(os.path.join(args.out, "ch3_mechanism.json"), "w") as f:
            json.dump({"retention": a_rows, "intervention": b_rows}, f, indent=1)


if __name__ == "__main__":
    main()
