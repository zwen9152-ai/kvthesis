"""Run a (method x budget) grid on a benchmark and append one JSON line per problem.

Method specs accept overrides, e.g. ``stepkv@anchors=4,beta=2,step_trigger=1``.
``rollback=1`` turns on degeneration detection + recall + rollback (implies offload),
e.g. ``snapkv@rollback=1``; knobs: ``max_rollbacks``, ``refeed``, ``min_span``, and
``escalate`` (budget multiplier per rollback, e.g. ``snapkv@rollback=1,escalate=2``).
Re-running with the same --out resumes: finished (run, problem) pairs are skipped.

Example:
    python -m eval.run_eval --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --dataset math500 --n 50 --methods full,h2o,snapkv,stepkv --budgets 256,512 \
        --out results/math500_small.jsonl
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import data
from eval.grader import extract_answer, is_equiv
from kvcomp import KVCompressor, generate
from kvcomp.batch import generate_batch
from kvcomp.rescue import HelperState
from kvcomp.degeneration import DegenerationDetector
from kvcomp.segment import delimiter_token_ids


def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_method(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition("@")
    cfg = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        cfg[k] = parse_value(v)
    return name, cfg


def split_methods(s: str) -> list[str]:
    """Split "a,b@x=1,y=2,c" into ["a", "b@x=1,y=2", "c"]: a bare key=value continues the previous spec."""
    specs: list[str] = []
    for part in s.split(","):
        if "=" in part and "@" not in part and specs:
            specs[-1] += "," + part
        else:
            specs.append(part)
    return specs


def pop_shard(spec: str) -> tuple[str, int, int]:
    """Strip a ``shard=k/m`` option: this process then runs only problems with index % m == k.

    The shard is not part of the run name, so shards of one run share their results.
    """
    m = re.search(r"[@,]shard=(\d+)/(\d+)", spec)
    if not m:
        return spec, 0, 1
    rest = spec[: m.start()] + ("@" if m.group(0)[0] == "@" else "") + spec[m.end():]
    rest = rest.replace("@,", "@").rstrip("@")
    return rest, int(m.group(1)), int(m.group(2))


def run_key(spec: str, budget: int | None) -> str:
    return spec if budget is None else f"{spec}|{budget}"


def has_loop(gen_ids, delim) -> bool:
    """Post-hoc degeneration flag: does the finished trace contain a repetition loop?"""
    det = DegenerationDetector()
    seen: list[int] = []  # grown in place: slicing a prefix per token would be O(n^2)
    for t in gen_ids:
        seen.append(t)
        if det.update(seen, t in delim) is not None:
            return True
    return False


def summarize(path: str):
    rows = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["run"]].append(r)
    cols = ["n", "acc", "len", "computed", "maxKV", "tok/s", "trunc", "loop", "rollbk"]
    print(f"\n{'run':36s} " + " ".join(f"{c:>8s}" for c in cols))
    for run, rs in sorted(rows.items()):
        n = len(rs)
        mean = lambda key: sum(r.get(key, 0) for r in rs) / n  # noqa: E731
        vals = [n, f"{mean('correct'):.1%}", f"{mean('num_tokens'):.0f}", f"{mean('computed_tokens'):.0f}",
                f"{mean('max_cache_len'):.0f}", f"{mean('tok_per_s'):.1f}",
                f"{sum(not r['finished'] for r in rs) / n:.1%}", f"{mean('degenerate'):.1%}",
                f"{sum(len(r.get('rollbacks', [])) for r in rs) / n:.2f}"]
        print(f"{run:36s} " + " ".join(f"{v:>8}" for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=sorted(data.LOADERS), required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--methods", default="full,streaming,h2o,snapkv,rkv,stepkv")
    ap.add_argument("--budgets", default="256,512")
    ap.add_argument("--buffer", type=int, default=128)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_ids", action="store_true", help="store generated token ids (needed for analysis)")
    ap.add_argument("--batch_size", type=int, default=1, help="problems decoded together (continuous batching)")
    ap.add_argument("--helper_model", default=None, help="full-KV helper for the rescue=1 (KV-Rescue-lite) baseline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 attn_implementation=args.attn).to(device).eval()
    helper = None
    if args.helper_model:
        helper = AutoModelForCausalLM.from_pretrained(args.helper_model, dtype=getattr(torch, args.dtype),
                                                      attn_implementation=args.attn).to(device).eval()
    delim = delimiter_token_ids(tok)
    gen_eos = model.generation_config.eos_token_id
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    problems = data.load(args.dataset, args.n, args.seed)
    budgets = [int(b) for b in args.budgets.split(",")]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["run"], r["id"]) for r in map(json.loads, f)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def write(run, name, cfg, budget, i, prob, ids, res):
        text = tok.decode(res["gen_ids"], skip_special_tokens=True)
        pred = extract_answer(text)
        rec = {"run": run, "method": name, "cfg": cfg, "budget": budget, "id": prob["id"],
               "gold": prob["answer"], "pred": pred, "correct": is_equiv(pred, prob["answer"]),
               "prompt_len": ids.shape[1], "degenerate": has_loop(res["gen_ids"], delim),
               "dataset": args.dataset, "seed": args.seed, "dtype": args.dtype,
               **{k: v for k, v in res.items() if k != "gen_ids"}}
        if args.save_ids:
            rec["prompt_ids"] = ids[0].tolist()
            rec["gen_ids"] = res["gen_ids"]
        # one unbuffered O_APPEND write per line, so parallel workers never interleave
        with open(args.out, "ab", buffering=0) as f:
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode())
        done.add((run, prob["id"]))
        print(f"[{run}] {i + 1}/{len(problems)} {prob['id']} correct={rec['correct']} "
              f"len={res['num_tokens']} maxKV={res['max_cache_len']} {res['tok_per_s']:.1f} tok/s",
              flush=True)

    for spec in split_methods(args.methods):
        spec, shard, n_shards = pop_shard(spec)
        name, cfg = parse_method(spec)
        for budget in ([None] if name == "full" else budgets):
            run = run_key(spec, budget)
            comp_opts, det_opts, rb_opts = None, None, {}
            observe_alpha = None
            if name != "full":
                comp_opts = dict(cfg)
                if name == "lazy":  # official schedule: evict every W steps, W ~ 0.117 x capacity (MATH)
                    w = int(comp_opts.pop("obs", max(8, round(0.117 * budget))))
                    comp_opts.setdefault("buffer", w)
                    comp_opts.setdefault("window", w)
                    observe_alpha = float(comp_opts.pop("alpha", 1e-4))
                comp_opts.setdefault("buffer", args.buffer)
                comp_opts.setdefault("window", args.window)
                rescue = comp_opts.pop("rescue", 0)
                if rescue:  # KV-Rescue-lite: same trigger as rollback, the helper rewrites the step
                    if helper is None:
                        raise SystemExit("rescue=1 needs --helper_model")
                    rb_opts = {k: comp_opts.pop(k) for k in ("max_rollbacks", "max_rewind", "helper_max_step")
                               if k in comp_opts}
                    det_opts = {k: comp_opts.pop(k) for k in ("min_span", "max_period") if k in comp_opts}
                elif comp_opts.pop("rollback", 0):
                    comp_opts["offload"] = True
                    rb_opts = {k: comp_opts.pop(k) for k in
                               ("max_rollbacks", "refeed", "max_rewind", "recall", "escalate", "max_budget")
                               if k in comp_opts}
                    det_opts = {k: comp_opts.pop(k) for k in ("min_span", "max_period") if k in comp_opts}
            todo = [(i, prob) for i, prob in enumerate(problems)
                    if i % n_shards == shard and (run, prob["id"]) not in done]
            if not todo:
                continue
            prompts = []
            for _, prob in todo:
                ids = data.build_prompt(tok, prob["question"])
                prompts.append(ids if torch.is_tensor(ids) else ids["input_ids"])
            prompts = [p.to(device) for p in prompts]
            common = dict(delimiter_ids=delim, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                          top_p=args.top_p, eos_ids=eos, **rb_opts)

            if args.batch_size > 1 or observe_alpha is not None:
                make_comp = None if comp_opts is None else (
                    lambda capture: KVCompressor(model, name, budget, capture=capture, **comp_opts))
                make_det = None if det_opts is None else (lambda: DegenerationDetector(**det_opts))
                make_help = (lambda: HelperState(helper)) if name != "full" and cfg.get("rescue") else None
                results = generate_batch(model, prompts, make_compressor=make_comp, make_detector=make_det,
                                         make_helper=make_help,
                                         batch_size=args.batch_size,
                                         window=comp_opts["window"] if comp_opts else args.window,
                                         observe_alpha=observe_alpha,
                                         seeds=[args.seed + i for i, _ in todo], **common)
                for j, res in results:
                    i, prob = todo[j]
                    write(run, name, cfg, budget, i, prob, prompts[j], res)
            else:
                comp = None if comp_opts is None else KVCompressor(model, name, budget, **comp_opts)
                det = None if det_opts is None else DegenerationDetector(**det_opts)
                for j, (i, prob) in enumerate(todo):
                    hs = HelperState(helper) if name != "full" and cfg.get("rescue") else None
                    res = generate(model, prompts[j], compressor=comp, detector=det, seed=args.seed + i,
                                   helper_state=hs, **common)
                    write(run, name, cfg, budget, i, prob, prompts[j], res)
                if comp is not None:
                    comp.close()
    summarize(args.out)


if __name__ == "__main__":
    main()
