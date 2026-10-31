"""Case studies for the thesis: what the detector's trigger looks like in a failing
compressed trace, and what RewindKV did about it.

For problems where ``--base`` is flagged and fails while ``--variant`` (e.g. its RewindKV
run) is correct, writes a Markdown file with: the step (or periodic span) that triggered
the detector and where it recurs, the text leading into it, how the base trace ends, and
for the variant the text after every rollback cut. Expect both genuine degeneration and
false positives (e.g. legitimate enumerations) -- report both honestly.

    python -m analysis.loop_examples --runs results/phase1b_math500_n50_bfloat16.jsonl \
        --base "snapkv|128" --variant "snapkv@rollback=1,escalate=2,max_budget=1024|128" \
        --out results/analysis/phase1b/loop_examples.md
"""

import argparse
import json

from transformers import AutoTokenizer

from kvcomp.degeneration import DegenerationDetector
from kvcomp.segment import delimiter_token_ids


def first_loop(gen_ids, delim):
    det = DegenerationDetector()
    seen = []
    for i, t in enumerate(gen_ids):
        seen.append(t)
        start = det.update(seen, t in delim)
        if start is not None:
            return start, i + 1
    return None


def period_of(tokens, max_p=64):
    for p in range(1, max_p + 1):
        if len(tokens) >= 2 * p and tokens[-p:] == tokens[-2 * p: -p]:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--max_examples", type=int, default=5)
    ap.add_argument("--context", type=int, default=150, help="tokens of text shown around each point")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    delim = set(delimiter_token_ids(tok))
    runs = {}
    for path in args.runs:
        for r in map(json.loads, open(path)):
            runs.setdefault(r["run"], {})[r["id"]] = r
    base, var = runs[args.base], runs[args.variant]
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True).replace("```", "'''")  # noqa: E731

    out = [f"# Loop case studies: `{args.base}` vs `{args.variant}`\n"]
    shown = 0
    for pid, b in base.items():
        v = var.get(pid)
        if v is None or b["correct"] or not v["correct"] or not b.get("gen_ids") or not b["degenerate"]:
            continue
        g = b["gen_ids"]
        loop = first_loop(g, delim)
        if loop is None:
            continue
        start, detected = loop
        p = period_of(g[start:detected]) or 0
        trigger = g[start:detected] if not p else g[detected - p: detected]
        # where the first step of the trigger occurs in the whole base trace (verbatim re-derivations)
        first = []
        for t in g[start:detected]:
            first.append(t)
            if t in delim:
                break
        steps = [i for i in range(len(g) - len(first) + 1) if g[i: i + len(first)] == first] if not p else []
        out.append(f"## {pid}\n\ngold: `{b['gold']}` | base pred: `{b['pred']}` ({b['num_tokens']} tokens, "
                   f"finished={b['finished']}) | variant pred: `{v['pred']}` ({v['num_tokens']} tokens, "
                   f"{len(v.get('rollbacks', []))} rollbacks, final budget {v.get('final_budget')})\n")
        out.append(f"**Base: text before the trigger** (trigger at token {start}):\n\n```\n"
                   f"{dec(g[max(0, start - args.context): start])}\n```\n")
        kind = f"periodic span, period {p}" if p else f"repeated step, occurrences at tokens {steps[:10]}"
        out.append(f"**Trigger** ({kind}):\n\n```\n{dec(trigger)}\n```\n")
        out.append(f"**Base: how the trace ends**:\n\n```\n{dec(g[-args.context:])}\n```\n")
        vg = v.get("gen_ids") or []
        for k, rb in enumerate(v.get("rollbacks", [])):
            cut = rb["cut"]
            out.append(f"**Variant rollback {k + 1}**: loop detected at token {rb['at']}, cut back to {cut}, "
                       f"budget -> {rb.get('budget')}. Text after the cut in the final trace:\n\n```\n"
                       f"{dec(vg[cut: cut + args.context])}\n```\n")
        shown += 1
        if shown >= args.max_examples:
            break
    if shown == 0:
        out.append("(no problem where the base run loops and fails while the variant is correct)\n")
    with open(args.out, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {shown} examples to {args.out}")


if __name__ == "__main__":
    main()
