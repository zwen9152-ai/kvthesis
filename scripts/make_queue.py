"""Write the thesis experiment matrix as a prioritized run_eval queue (one job per line).

    python scripts/make_queue.py verify   > logs/queue.txt   # ~2 GPU-h: batching check + throughput
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
    """Batched decoding on the real model: equality with batch 1 (greedy) and throughput."""
    out = "results/verify/math500_batching.jsonl"
    common = f"--model {M15} --dataset math500 --n 16 --max_new_tokens 2048 --temperature 0 --save_ids"
    jobs = []
    for bs in (1, 16):
        for m in ("full", f"snapkv@{REWIND}", "lazy", "vase"):
            jobs.append(f"{common} --methods {m} --budgets 128 --batch_size {bs} "
                        f"--out {out.replace('.jsonl', f'_bs{bs}.jsonl')}")
    for bs in (8, 32, 64):  # throughput sweep
        jobs.append(f"--model {M15} --dataset math500 --n 64 --max_new_tokens 2048 --methods snapkv "
                    f"--budgets 128 --batch_size {bs} --out results/verify/throughput_bs{bs}.jsonl")
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
        "snapkv",                                 # static memory-matched control (budget 512)
    ]
    jobs = [job(M15, "math500", n, maxtok, 0, s, [128]) for s in specs[:-1]]
    jobs.append(job(M15, "math500", n, maxtok, 0, specs[-1], [512]))
    return jobs


def seven_b():
    jobs = []
    for ds, n, maxtok in (("math500", 100, 8192), ("aime24", 30, 16384)):
        jobs.append(job(M7, ds, n, maxtok, 0, "full", [0]))
        for spec in ("snapkv", f"snapkv@{REWIND}", "snapkv@rescue=1", "rkv", f"rkv@{REWIND}"):
            extra = f"--helper_model {M15}" if "rescue" in spec else ""
            jobs.append(job(M7, ds, n, maxtok, 0, spec, BUDGETS, extra))
    return jobs


def batch_size(j: str, default: int) -> int:
    """Fit a 24 GB GPU: KV per entry is ~28 KB (1.5B) / ~57 KB (7B, plus 15 GB of weights)."""
    a = j.split()
    maxtok = int(a[a.index("--max_new_tokens") + 1])
    full = a[a.index("--methods") + 1] == "full"
    if M7 in j:
        if full:
            return 4 if maxtok > 8192 else 8
        return 16
    if full and maxtok > 8192:
        return 16
    return default


PHASES = {"verify": verify, "main": main_15b, "ablation": ablation, "7b": seven_b}


def estimate(jobs, tok_s_15b: float, tok_s_7b: float):
    total_h = 0.0
    for j in jobs:
        a = j.split()
        get = lambda k: a[a.index(k) + 1]  # noqa: E731
        ds, n = get("--dataset"), int(get("--n"))
        n_budgets = 1 if "full" == get("--methods") else len(get("--budgets").split(","))
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
    ap.add_argument("--batch_size", type=int, default=32)
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
