"""Offline comparison of alarm signals and rules (chapter 4): misses vs. unnecessary alarms.

    python -m analysis.detectors results/ch3/1.5b_math500_s0.jsonl results/main/1.5b_math500_s0.jsonl \
        --runs "snapkv|128;rkv|128;lazy|128" --out results/analysis/detectors.md

Token-based signals (exact / fuzzy step repeats, hesitation) are replayed with the real
``DegenerationDetector`` on the saved token ids; the forgetting signal is read from the
``forget_trace`` logged by runs with ``forget_every`` (e.g. the ch3 observe runs). The replay
is exact for any trajectory that was never rolled back, so use plain or observe-only runs
(``rollback=1,act=0``) — the latter are matched to their plain base for Full/base labels.

* "missed-type" problems: Full KV solves them, the compressed run fails them;
* "healthy" problems: the compressed run solves them; an alarm there is unnecessary.
Per rule: share of missed-type problems caught, share of healthy problems alarmed, and how
far into the output the alarm comes (median). Graded rows: the yellow tier fires on any
signal (cheap, no rollback), the red tier on the ``red_after``-th signal.
"""

import argparse
import json
from collections import defaultdict

START = 256  # alarms before the first compression cannot be acted on


def load(paths):
    runs = defaultdict(dict)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            runs[r["run"]][r["id"]] = r
    return runs


def fire_times(g, delim, kw):
    from kvcomp.degeneration import DegenerationDetector
    d = DegenerationDetector(**kw)
    buf, out = [], []
    for j, t in enumerate(g):
        buf.append(t)
        if d.update(buf, t in delim) is not None and j >= START:
            out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--runs", required=True, help="';'-separated runs to evaluate (plain or observe-only)")
    ap.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--fuzzy", type=float, default=0.6)
    ap.add_argument("--hesitate", type=int, default=5)
    ap.add_argument("--forget_thrs", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from eval.run_eval import cue_token_ids
    from kvcomp.segment import delimiter_token_ids
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    delim, cues = delimiter_token_ids(tok), cue_token_ids(tok)
    off = dict(min_span=10 ** 9)  # periodic loops: part of every rule through the exact detector
    sig_kw = {"exact": {}, "fuzzy": dict(fuzzy=args.fuzzy, **off),
              "hes": dict(hesitate=args.hesitate, cue_ids=cues, step_repeats=10 ** 9, **off)}
    runs = load(args.files)
    full = runs.get("full") or next((v for k, v in runs.items() if k.startswith("full")), {})
    thrs = [float(x) for x in args.forget_thrs.split(",") if x]

    sigs = {}  # (run, id) -> {signal: [times]}
    groups = {}
    for run in args.runs.split(";"):
        rs = runs.get(run)
        if not rs:
            print(f"(skip {run}: not found)")
            continue
        missed = [i for i in rs if i in full and full[i]["correct"] and not rs[i]["correct"]]
        healthy = [i for i in rs if rs[i]["correct"]]
        groups[run] = (missed, healthy)
        for i in missed + healthy:
            g = rs[i].get("gen_ids") or []
            s = {k: fire_times(g, delim, kw) for k, kw in sig_kw.items()}
            tr = rs[i].get("forget_trace")
            if tr is not None:
                for t in thrs:
                    s[f"forget>{t:g}"] = [j for j, v in tr if v >= 1000 * t and j >= START]
            s["_len"] = max(1, len(g))
            sigs[(run, i)] = s

    def first(s, keys, k=1):
        ts = sorted(t for key in keys for t in s.get(key, []))
        return ts[k - 1] if len(ts) >= k else None

    has_forget = any(k.startswith("forget") for s in sigs.values() for k in s)
    rules = {"v1: exact repeats": (["exact"], 1), "fuzzy": (["fuzzy"], 1), "hesitation": (["hes"], 1),
             "any signal (yellow tier)": (["exact", "fuzzy", "hes"], 1),
             "2nd signal (red tier, red_after=2)": (["exact", "fuzzy", "hes"], 2),
             "3rd signal (red_after=3)": (["exact", "fuzzy", "hes"], 3)}
    if has_forget:
        for t in thrs:
            rules[f"forgetting > {t:g}"] = ([f"forget>{t:g}"], 1)
        mid = f"forget>{thrs[len(thrs) // 2]:g}"
        rules[f"any incl. {mid} (yellow)"] = (["exact", "fuzzy", "hes", mid], 1)
        rules[f"2nd incl. {mid} (red)"] = (["exact", "fuzzy", "hes", mid], 2)

    lines = ["| rule | " + " | ".join(f"{r}: caught / unnecessary" for r in groups) + " | all: caught | all: unnecessary | alarm at (median share of output) |",
             "|" + "---|" * (len(groups) + 4)]
    for name, (keys, k) in rules.items():
        cells, tc, tm, tu, th, when = [], 0, 0, 0, 0, []
        for run, (missed, healthy) in groups.items():
            c = [first(sigs[(run, i)], keys, k) for i in missed]
            u = [first(sigs[(run, i)], keys, k) for i in healthy]
            nc, nu = sum(x is not None for x in c), sum(x is not None for x in u)
            cells.append(f"{nc}/{len(missed)} / {nu}/{len(healthy)}")
            tc, tm, tu, th = tc + nc, tm + len(missed), tu + nu, th + len(healthy)
            when += [x / sigs[(run, i)]["_len"] for x, i in zip(c, missed) if x is not None]
        when.sort()
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {tc / max(tm, 1):.0%} | {tu / max(th, 1):.0%} | "
                     + (f"{when[len(when) // 2]:.0%}" if when else "-") + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
