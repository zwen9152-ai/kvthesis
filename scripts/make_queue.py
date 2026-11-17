"""Write the thesis experiment matrix as a prioritized run_eval queue (one job per line).

    python scripts/make_queue.py verify   > logs/queue.txt   # batching check (scripts/run_verify.sh)
    python scripts/make_queue.py main ablation 7b > logs/queue.txt
    bash scripts/run_queue.sh logs/queue.txt                  # workers pop lines in order

Each line is the argument string of one ``python -m eval.run_eval`` call; results of one
(model, dataset, seed) go to one file under results/main/. Everything resumes: finished
(run, problem) pairs are skipped, so re-generating and re-running the queue is safe.
``--estimate`` prints the expected decoded tokens and GPU hours instead.
"""

import argparse
import sys

M15 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
M7 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
REWIND = "rollback=1,escalate=2,max_budget=1024"  # RewindKV as evaluated in phase 1b

# dataset -> (problems, max_new_tokens, seeds, mean decoded tokens per problem for the estimate)
DATA = {
    "math500": (200, 8192, [0], 4500),
    "gsm8k": (100, 4096, [0], 1500),
    "aime24": (30, 16384, [0, 1], 11000),
    "math500dev": (100, 8192, [0], 4500),  # tuning split, disjoint from the 200 main problems
}
BASES = ["streaming", "h2o", "snapkv", "rkv", "lazy", "vase"]
REWIND_ON = ["snapkv", "rkv", "lazy", "vase"]  # H2O / StreamingLLM stay plain baselines
BUDGETS = [128, 256]


def job(model, dataset, n, maxtok, seed, methods, budgets, extra=""):
    tag = "7b" if model == M7 else "1.5b"
    out = f"results/main/{tag}_{dataset}_s{seed}.jsonl"
    b = ",".join(map(str, budgets))
    return (f"--model {model} --dataset {dataset} --n {n} --seed {seed} --max_new_tokens {maxtok} "
            f"--methods {methods} --budgets {b} --save_ids --out {out} {extra}").strip()


def verify():
    """Batched decoding on the real model: same accuracy / lengths as batch 1 (greedy)."""
    out = "results/verify/math500_batching.jsonl"
    common = f"--model {M15} --dataset math500 --n 16 --max_new_tokens 2048 --temperature 0 --save_ids"
    jobs = []
    for bs in (1, 16):
        for m in ("full", f"snapkv@{REWIND}", "lazy", "vase"):
            jobs.append(f"{common} --methods {m} --budgets 128 --batch_size {bs} "
                        f"--out {out.replace('.jsonl', f'_bs{bs}.jsonl')}")
    return jobs


def main_15b():
    jobs = []
    # priority: MATH first (the core table), then AIME (hard, long), then GSM8K (short reasoning)
    for ds in ("math500", "aime24", "gsm8k"):
        n, maxtok, seeds, _ = DATA[ds]
        for seed in seeds:
            if ds == "math500" or seed == seeds[0]:
                jobs.append(job(M15, ds, n, maxtok, seed, "full", [0]))
            for base in BASES:
                jobs.append(job(M15, ds, n, maxtok, seed, base, BUDGETS))
                if base in REWIND_ON:
                    jobs.append(job(M15, ds, n, maxtok, seed, f"{base}@{REWIND}", BUDGETS))
            if seed != seeds[0]:
                jobs.append(job(M15, ds, n, maxtok, seed, "full", [0]))
    return jobs


def ablation():
    n, maxtok, _, _ = DATA["math500"]
    specs = [
        f"snapkv@{REWIND},recall=0",            # recall matters
        "snapkv@rollback=1",                      # escalation matters
        "snapkv@rollback=1,escalate=1.5,max_budget=1024",
        "snapkv@rollback=1,escalate=4,max_budget=1024",
        f"snapkv@{REWIND},max_rewind=0",          # cut at the loop itself, not the step start
        f"snapkv@{REWIND},min_span=64",           # detector sensitivity
        f"snapkv@{REWIND},min_span=192",
        f"snapkv@{REWIND},step_repeats=2",        # repeated-step trigger (the dominant signal)
        f"snapkv@{REWIND},step_repeats=4",
        f"snapkv@{REWIND},min_step_len=24",       # ignore short repeated lines (enumerations)
        "snapkv",                                 # static memory-matched control (budget 512)
    ]
    jobs = [job(M15, "math500", n, maxtok, 0, s, [128]) for s in specs[:-1]]
    jobs.append(job(M15, "math500", n, maxtok, 0, specs[-1], [512]))
    # static large budgets: Pareto curve for chapter 4, and a sanity check that the
    # rkv / lazy ports approach full KV in the regime their papers evaluate (30-50% of length)
    jobs.append(job(M15, "math500", n, maxtok, 0, "snapkv,rkv,lazy,vase", [512, 1024]))
    return jobs


def seven_b():
    jobs = []
    for ds, n, maxtok in (("math500", 100, 8192), ("aime24", 30, 16384)):
        jobs.append(job(M7, ds, n, maxtok, 0, "full", [0]))
        for spec in ("snapkv", f"snapkv@{REWIND}", "snapkv@rescue=1", "rkv", f"rkv@{REWIND}"):
            extra = f"--helper_model {M15}" if "rescue" in spec else ""
            jobs.append(job(M7, ds, n, maxtok, 0, spec, BUDGETS, extra))
    return jobs


LIGHT = f"{REWIND},recall=0,offload=0"  # lightweight RewindKV: no CPU store, no recall transient


def light():
    """Follow-up after the time-averaged KV analysis: RewindKV without recall/offload on all
    four bases at both start budgets, and a stronger escalation for the no-recall variant."""
    n, maxtok, _, _ = DATA["math500"]
    jobs = [job(M15, "math500", n, maxtok, 0, f"{b}@{LIGHT}", BUDGETS) for b in REWIND_ON]
    jobs.append(job(M15, "math500", n, maxtok, 0, "snapkv@rollback=1,escalate=4,max_budget=1024,recall=0,offload=0",
                    BUDGETS))
    return jobs


# ---- chapters 3 / 4 (2026-11 plan) ----
OBSERVE = "rollback=1,act=0,probe=1"  # detector logs events, nothing is rolled back; chapter 3 probe on
MERGE = "merge=1"  # step-summary merging (chapter 3 prevention), 25% of the budget reserved
V2 = "fuzzy=0.6,hesitate=6,decay=8"  # RewindKV v2 detector + budget decay (defaults until v2dev picks)


def ch3():
    """Mechanism analysis (chapter 3): Full-KV lookback reference, retention at detection events,
    and the causal intervention (restore the repeated step vs. as many random evicted tokens).
    The observe-only runs double as the "intervene=none" arm (identical tokens by construction)."""
    n, maxtok = 100, 8192
    jobs = [job(M15, "math500", n, maxtok, 0, "full@probe=1,lookback=1,rollback=1,act=0", [0])]
    jobs.append(job(M15, "math500", n, maxtok, 0, ",".join(f"{b}@{OBSERVE}" for b in ("snapkv", "rkv", "lazy")),
                    [128]))
    for b in ("snapkv", "rkv"):
        jobs.append(job(M15, "math500", n, maxtok, 0,
                        ",".join(f"{b}@rollback=1,act=0,intervene={m}" for m in ("aligned", "random")), [128]))
    return [j.replace("results/main/", "results/ch3/") for j in jobs]


def prevent_dev():
    """Tune step-summary merging on the dev split (SnapKV@128; one R-KV check)."""
    n, maxtok, _, _ = DATA["math500dev"]
    specs = ["snapkv", f"snapkv@{MERGE}", f"snapkv@{MERGE},merge_frac=0.125", f"snapkv@{MERGE},merge_frac=0.5",
             f"snapkv@{MERGE},merge_w=uniform", f"snapkv@{MERGE},merge_key=recent"]
    jobs = [job(M15, "math500dev", n, maxtok, 0, ",".join(specs), [128])]
    jobs.append(job(M15, "math500dev", n, maxtok, 0, f"rkv,rkv@{MERGE}", [128]))
    return [j.replace("results/main/", "results/dev/") for j in jobs]


def prevent():
    """Main table of the prevention method (+ combination with lightweight RewindKV at 128)."""
    n, maxtok, _, _ = DATA["math500"]
    jobs = [job(M15, "math500", n, maxtok, 0, f"{b}@{MERGE}", BUDGETS) for b in REWIND_ON]
    jobs += [job(M15, "math500", n, maxtok, 0, f"{b}@{MERGE},{LIGHT}", [128]) for b in REWIND_ON]
    return jobs


def v2_dev():
    """Tune the v2 components one at a time on the dev split (SnapKV@128, lightweight base)."""
    n, maxtok, _, _ = DATA["math500dev"]
    base = f"snapkv@{LIGHT}"
    specs = [base, f"{base},fuzzy=0.5", f"{base},fuzzy=0.7", f"{base},hesitate=5", f"{base},hesitate=6",
             f"{base},decay=4", f"{base},decay=8",
             f"snapkv@{REWIND},recall=aligned"]  # aligned recall + pinning needs the CPU store
    jobs = [job(M15, "math500dev", n, maxtok, 0, s, [128]) for s in specs]
    return [j.replace("results/main/", "results/dev/") for j in jobs]


def v2():
    """Main table of RewindKV v2 and of v2 + prevention."""
    n, maxtok, _, _ = DATA["math500"]
    jobs = [job(M15, "math500", n, maxtok, 0, f"{b}@{LIGHT},{V2}", BUDGETS) for b in REWIND_ON]
    jobs += [job(M15, "math500", n, maxtok, 0, f"{b}@{MERGE},{LIGHT},{V2}", BUDGETS) for b in REWIND_ON]
    return jobs


def curve():
    """Densify the static cache-accuracy curve where the adaptive runs land (time-avg KV 350-650,
    between static 256 and 512): budget 384, SnapKV/R-KV first, then LazyEviction/VaSE."""
    n, maxtok, _, _ = DATA["math500"]
    return [job(M15, "math500", n, maxtok, 0, "snapkv,rkv", [384]),
            job(M15, "math500", n, maxtok, 0, "lazy,vase", [384])]


def batch_size(j: str, default: int) -> int:
    """Fit a 24 GB GPU: KV per entry is ~28 KB (1.5B) / ~57 KB (7B, plus 15 GB of weights)."""
    a = j.split()
    maxtok = int(a[a.index("--max_new_tokens") + 1])
    full = a[a.index("--methods") + 1].split("@")[0] == "full"
    if M7 in j:
        if full:
            return 4 if maxtok > 8192 else 8
        return 16
    if full:
        return 16 if maxtok > 8192 else 32  # full KV grows to max_new_tokens per slot
    return default


PHASES = {"curve": curve, "ch3": ch3, "prevent_dev": prevent_dev, "prevent": prevent, "v2dev": v2_dev, "v2": v2,
          "light": light, "verify": verify, "main": main_15b, "ablation": ablation, "7b": seven_b}


def estimate(jobs, tok_s_15b: float, tok_s_7b: float):
    total_h = 0.0
    for j in jobs:
        a = j.split()
        get = lambda k: a[a.index(k) + 1]  # noqa: E731
        ds, n = get("--dataset"), int(get("--n"))
        n_budgets = 1 if get("--methods").startswith("full") else len(get("--budgets").split(","))
        n_budgets *= get("--methods").count("@") or 1  # several specs in one job
        per = DATA.get(ds, (0, 0, 0, 2000))[3] if "verify" not in j else 1500
        toks = n * n_budgets * per * (1.1 if "rollback" in j or "rescue" in j else 1.0)
        rate = tok_s_7b if M7 in j else tok_s_15b
        total_h += toks / rate / 3600
    return total_h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="+", choices=sorted(PHASES))
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--tok_s_15b", type=float, default=400, help="aggregate decode throughput, 1.5B")
    ap.add_argument("--tok_s_7b", type=float, default=150, help="aggregate decode throughput, 7B")
    ap.add_argument("--batch_size", type=int, default=64)  # 1.5B compressed: measured best on a 3090
    args = ap.parse_args()
    jobs = [j for p in args.phases for j in PHASES[p]()]
    if args.estimate:
        h = estimate(jobs, args.tok_s_15b, args.tok_s_7b)
        print(f"{len(jobs)} jobs, ~{h:.1f} GPU hours at {args.tok_s_15b:g} / {args.tok_s_7b:g} tok/s", file=sys.stderr)
    else:
        for j in jobs:
            if "--batch_size" not in j:
                j += f" --batch_size {batch_size(j, args.batch_size)}"
            print(j)
